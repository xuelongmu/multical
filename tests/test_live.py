import copy
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import cv2
import numpy as np
from structs.numpy import Table
from structs.struct import struct

from multical.board import load_config
from multical.camera import Camera
from multical.graph import select_pairs
from multical.live.calibration import AnchoredPoseSet, solve_snapshot, validate
from multical.live.metrics import Coverage, observe
from multical.live.session import Session
from multical.live.sources import Frame, FrameSet, PTPNotReady, PySpinSource, SimulatedSource
from multical.optimization.calibration import error_stats
from multical.optimization.parameters import IndexMapper
from multical.transform.matrix import align_transforms_robust


BOARD = Path(__file__).resolve().parents[1] / 'example_boards/charuco_36x54.yaml'


def make_record(batch, board, index, role):
    observations = {s: observe(board, f) for s, f in batch.frames.items()}
    return dict(id=f'{index:06d}', role=role,
                frames={s: f.metadata() for s, f in batch.frames.items()},
                detections={s: dict(ids=o['ids'].tolist(), corners=o['corners'].tolist(), usable=bool(o['usable']))
                            for s, o in observations.items()})


class SourceTests(unittest.TestCase):
    def test_ptp_excursion_preserves_preview_but_blocks_capture_until_recovered(self):
        driver = PySpinSource(expected_serials=('A',), expected_count=1)
        driver.serials = ('A',)
        driver.held = [dict(serial='A', cam=struct(GetNodeMap=lambda: None))]
        driver.deadline = driver.last_ptp = 0
        driver._ptp = Mock(side_effect=[PTPNotReady('Clock not ready'), None])
        driver._get = lambda nm, name, kind: {'ExposureTime': 1000, 'Gain': 0, 'TimestampLatchValue': 1_000_000_000}[name]
        driver._command = Mock()
        frames = iter([Frame('A', np.ones((10, 10), np.uint8), i, 1_151_000_000, 1000) for i in (1, 2)])
        driver.pool = struct(submit=lambda *args: struct(result=lambda: next(frames)))
        with patch('multical.live.sources.time.monotonic', return_value=100.):
            bad = driver.read()
        self.assertEqual(len(bad.frames), 1)
        self.assertTrue(any('Clock not ready' in error for error in bad.problems(capture_mode='motion')))
        self.assertEqual(bad.problems(capture_mode='stationary'), [])
        with patch('multical.live.sources.time.monotonic', return_value=104.):
            good = driver.read()
        self.assertEqual(good.problems(), [])

    def test_write_only_action_device_key_can_be_set_without_reading(self):
        driver = PySpinSource()
        node = Mock()
        driver.sdk = struct(CIntegerPtr=lambda value: value, IsReadable=lambda value: False,
                            IsWritable=lambda value: True)
        nodemap = struct(GetNode=lambda name: node)
        driver._set(nodemap, 'ActionDeviceKey', 'Integer', 42)
        node.SetValue.assert_called_once_with(42)
        with self.assertRaisesRegex(RuntimeError, 'not readable'):
            driver._get(nodemap, 'ActionDeviceKey', 'Integer')

    def batch(self):
        image = np.zeros((100, 100), np.uint8)
        return FrameSet(1, 1_000_000_000, ('A', 'B'),
                        {'A': Frame('A', image, 10, 1_001_200_000, 1200),
                         'B': Frame('B', image, 91, 1_001_206_000, 1200)})

    def test_different_per_camera_ids_are_valid_for_one_scheduled_action(self):
        self.assertEqual(self.batch().problems(), [])
        self.assertEqual(self.batch().spread_us, 6)

    def test_missing_camera_never_becomes_a_smaller_complete_rig(self):
        batch = self.batch()
        del batch.frames['B']
        self.assertTrue(any('Missing' in s for s in batch.problems()))

    def test_previous_trigger_frame_is_rejected(self):
        batch = self.batch()
        batch.frames['B'].timestamp_ns -= 200_000_000
        self.assertTrue(any('scheduled action' in s for s in batch.problems()))

    def test_exposure_midpoints_are_checked_even_when_starts_match(self):
        batch = self.batch()
        batch.frames['B'].exposure_us += 1000
        batch.frames['B'].timestamp_ns += 1_000_000
        self.assertEqual(batch.spread_us, 6)
        self.assertTrue(any('midpoints' in s for s in batch.problems(capture_mode='motion')))

    def test_stationary_mode_allows_different_exposures_and_small_clock_offsets(self):
        batch = self.batch()
        batch.frames['B'].exposure_us += 1000
        batch.frames['B'].timestamp_ns += 1_100_000  # Also 100 us of start skew, still a fresh frame.
        batch.timing_warnings['ptp:B'] = 'B: clock offset 5000 ns'
        self.assertEqual(batch.problems(capture_mode='stationary'), [])
        self.assertEqual(len(batch.problems(capture_mode='motion')), 3)
        self.assertEqual(len(batch.timing_issues()), 3)

    def test_stationary_mode_keeps_missing_stale_and_transport_failures_blocking(self):
        for fault in ('missing', 'stale', 'transport'):
            batch = self.batch()
            batch.timing_warnings['ptp:B'] = 'Clock quality warning'
            if fault == 'missing':
                del batch.frames['B']
            elif fault == 'stale':
                batch.frames['B'].timestamp_ns -= 200_000_000
            else:
                batch.errors['B'] = 'Transport failed'
            self.assertTrue(batch.problems(capture_mode='stationary'), fault)

    def test_unknown_capture_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Capture mode'):
            self.batch().problems(capture_mode='typo')

    def test_collect_copies_image_and_always_releases_buffer(self):
        driver = PySpinSource()
        driver.sdk = struct(PixelFormat_Mono8=1)
        pixels = np.ones((10, 10), np.uint8)
        class Image:
            released = False
            def IsIncomplete(self): return False
            def GetFrameID(self): return 4
            def GetTimeStamp(self): return 1000000
            def Release(self): self.released = True; pixels[:] = 0
        image = Image()
        state = dict(serial='A', cam=struct(GetNextImage=lambda timeout: image),
                     processor=struct(Convert=lambda *args: struct(GetNDArray=lambda: pixels)),
                     exposure=100, gain=0, settings={})
        frame = driver._collect(state)
        self.assertTrue(image.released)
        self.assertTrue(np.all(frame.image == 1))

    def test_collect_releases_incomplete_image(self):
        driver = PySpinSource()
        class Image:
            released = False
            def IsIncomplete(self): return True
            def GetImageStatus(self): return 9
            def Release(self): self.released = True
        image = Image()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            driver._collect(dict(cam=struct(GetNextImage=lambda timeout: image)))
        self.assertTrue(image.released)

    def test_cancelled_connection_never_opens_sdk(self):
        driver = PySpinSource()
        driver.stop_event = Event()
        driver.stop_event.set()
        with patch('importlib.import_module') as load, self.assertRaises(InterruptedError):
            driver.open()
        load.assert_not_called()

    def test_cleanup_continues_after_a_setting_restore_fails(self):
        calls = []
        driver = PySpinSource()
        cam = struct(EndAcquisition=lambda: calls.append('end'), GetNodeMap=lambda: 'map',
                     DeInit=lambda: calls.append('deinit'))
        driver.held = [dict(serial='A', cam=cam, acquiring=True, trigger_mode='Off', trigger_selector='FrameStart',
                            restore=[('ExposureAuto', 'Enumeration', 'Continuous'), ('Gain', 'Float', 2.)])]
        driver.camera_list = struct(Clear=lambda: calls.append('clear'))
        driver.system = struct(ReleaseInstance=lambda: calls.append('release'))
        def restore(nm, name, kind, value):
            calls.append(name)
            if name == 'Gain':
                raise RuntimeError('simulated write failure')
        driver._set = restore
        with self.assertRaisesRegex(RuntimeError, 'simulated write failure'):
            driver.close()
        for step in ('end', 'ExposureAuto', 'deinit', 'clear', 'release'):
            self.assertIn(step, calls)
        self.assertIsNone(driver.system)


class CoreRegressionTests(unittest.TestCase):
    def test_solver_is_reaped_when_pipe_closes_before_process_exit(self):
        from multical.live.ui import LiveWindow
        process = Mock()
        process.is_alive.return_value = False
        process.exitcode = 0
        window = SimpleNamespace(connection=None, process=process, cancel_button=Mock(), fail=Mock())
        LiveWindow.poll_solve(window)
        self.assertIsNone(window.process)
        process.join.assert_called_once()
        process.close.assert_called_once()

    def test_charuco_rows_use_corner_grid(self):
        board = load_config(BOARD)['big36_0']
        ids = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19])
        self.assertFalse(board.has_min_detections(struct(ids=ids)))
        self.assertTrue(board.has_min_detections(struct(ids=np.array([0,1,2,4,5,6,8,9,10]))))

    def test_empty_errors_are_not_a_perfect_fit(self):
        result = error_stats(np.array([]))
        self.assertEqual(result.n, 0)
        self.assertTrue(np.isnan(result.rms))

    def test_tree_does_not_destroy_graph(self):
        overlaps = np.array([[0., 24, 0], [24, 0, 10], [0, 10, 0]])
        before = overlaps.copy()
        _, pairs = select_pairs(overlaps)
        np.testing.assert_array_equal(overlaps, before)
        self.assertEqual(len(pairs), 2)

    def test_perfect_single_relative_pose_remains_an_inlier(self):
        identity = np.eye(4)[None]
        pose, mask = align_transforms_robust(identity, identity)
        np.testing.assert_allclose(pose, np.eye(4))
        self.assertTrue(mask[0])

    def test_anchor_has_no_parameters_but_keeps_observations(self):
        table = Table.create(poses=np.array([np.eye(4), np.eye(4)]), valid=np.ones(2, bool))
        poses = AnchoredPoseSet(table, ['A', 'B'])
        self.assertEqual(poses.param_vec.size, 6)
        moved = poses.with_param_vec(np.array([0., 0, 0, 1, 2, 3]))
        np.testing.assert_array_equal(moved.poses[0], np.eye(4))
        np.testing.assert_allclose(moved.poses[1, :3, 3], [1, 2, 3])
        np.testing.assert_array_equal(poses.poses[1], np.eye(4))
        self.assertTrue(moved.valid.all())
        self.assertEqual(len(poses.sparsity(IndexMapper(np.ones((2, 1, 1, 24), bool)), 0)), 1)

    def test_intrinsic_fit_does_not_prune_or_relax_target(self):
        points = struct(object_points=[np.zeros((10,3),np.float32)]*20,
                        corners=[np.zeros((10,2),np.float32)]*20, ids=[np.arange(10)]*20,
                        board_offset=[0]*20, image_ids=list(range(20)))
        result = (1.2, np.eye(3), np.zeros(5), None, None, None, None, np.ones((20, 1)))
        with patch('multical.camera.calibration_points', return_value=points), patch('cv2.calibrateCameraExtended', return_value=result) as call:
            camera, error = Camera.calibrate([], .5, [], (100, 100))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(error, 1.2)
        self.assertFalse(camera.intrinsic_dataset['target_met'])
        self.assertEqual(len(camera.intrinsic_dataset['image_ids']), 20)
        self.assertEqual(len(camera.copy().error_perview), 20)


class CoverageAndSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.board = load_config(BOARD)['big36_0']
        cls.source = SimulatedSource(cls.board, 2)

    def packet(self):
        batch = self.source.render(2)
        return dict(batch=batch, observations={s: observe(self.board, f) for s, f in batch.frames.items()})

    def test_same_pose_does_not_inflate_coverage_or_overlap(self):
        packet = self.packet()
        coverage = Coverage(packet['batch'].serials)
        coverage.add(packet['observations'])
        before = coverage.snapshot()
        coverage.add(packet['observations'])
        np.testing.assert_array_equal(before['overlaps'], coverage.overlaps)
        self.assertEqual(before['views'], coverage.views)

    def test_session_preserves_role_original_images_and_detection_identity(self):
        packet = self.packet()
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory, BOARD, packet['batch'].serials, simulated=True)
            session.add(packet, 'validation')
            saved = json.loads((session.directory / 'manifest.json').read_text())
            self.assertTrue(saved['simulated'])
            self.assertEqual(saved['captures'][0]['role'], 'validation')
            image = cv2.imread(str(session.directory / '000000/SIM-01.png'), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(image, packet['batch'].frames['SIM-01'].image)
            self.assertNotIn('image', session.samples[0]['frames']['SIM-01'])
            with self.assertRaisesRegex(ValueError, 'already been retained'):
                session.add(packet, 'training')

    def test_auto_validation_requires_shared_steady_novel_nontraining_pose(self):
        from multical.live.engine import LiveEngine
        packet = self.packet()
        engine = LiveEngine(Mock(), self.board, BOARD, '/tmp/unused')
        engine.coverage = Coverage(packet['batch'].serials)
        engine.validation_coverage = Coverage(packet['batch'].serials)
        engine.auto_capture = True
        engine.auto_capture_role = 'validation'
        observations = packet['observations']
        self.assertIsNone(engine.automatic_role(observations, False))
        single = copy.deepcopy(observations)
        single[next(iter(single))]['usable'] = False
        self.assertIsNone(engine.automatic_role(single, True))
        self.assertEqual(engine.automatic_role(observations, True), 'validation')
        engine.validation_coverage.add(observations)
        self.assertIsNone(engine.automatic_role(observations, True))
        engine.validation_coverage = Coverage(packet['batch'].serials)
        engine.coverage.add(observations)
        self.assertIsNone(engine.automatic_role(observations, True))
        engine.coverage = Coverage(packet['batch'].serials)
        engine.set_paused(True)
        self.assertIsNone(engine.automatic_role(observations, True))

    def test_pause_clears_queued_capture_and_blocks_manual_requests(self):
        from multical.live.engine import LiveEngine
        engine = LiveEngine(Mock(), self.board, BOARD, '/tmp/unused')
        engine.capture('training')
        engine.set_paused(True)
        self.assertIsNone(engine.pending_capture)
        self.assertTrue(engine.snapshot()['paused'])
        with self.assertRaisesRegex(ValueError, 'paused'):
            engine.capture('validation')
        engine.set_paused(False)
        engine.capture('validation')
        self.assertEqual(engine.pending_capture, 'validation')

    def test_resume_restores_captures_coverage_and_sequence_without_overwrite(self):
        from multical.live.engine import LiveEngine
        packet = self.packet()
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory, BOARD, packet['batch'].serials)
            session.add(packet, 'training')
            original = (session.directory / '000000/SIM-01.png').read_bytes()
            source = Mock()
            source.open.return_value = packet['batch'].serials
            engine = LiveEngine(source, self.board, BOARD, directory, resume=session.directory)
            fresh = self.packet()
            fresh['batch'].sequence = 0
            def read():
                engine.stop_event.set()
                return fresh['batch']
            source.read.side_effect = read
            engine._acquire()
            self.assertIsNone(engine.error)
            self.assertEqual(len(engine.session.samples), 1)
            self.assertTrue(all(v == 1 for v in engine.coverage.views.values()))
            self.assertGreater(engine.latest_batch.sequence, session.samples[0]['sequence'])
            engine.session.add(fresh, 'training')
            self.assertEqual(len(engine.session.samples), 2)
            self.assertEqual((session.directory / '000000/SIM-01.png').read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'must match'):
                Session(directory, BOARD, packet['batch'].serials, simulated=True, resume=session.directory)
            (session.directory / '000000/SIM-01.png').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing saved image'):
                Session(directory, BOARD, packet['batch'].serials, resume=session.directory)

    def test_stationary_session_records_timing_warnings_without_rejecting_capture(self):
        packet = self.packet()
        batch = packet['batch']
        frame = batch.frames[batch.serials[0]]
        frame.exposure_us += 1000
        frame.timestamp_ns += 1_000_000
        batch.timing_warnings['ptp:test'] = 'Clock offset beyond motion limit'
        with tempfile.TemporaryDirectory() as directory:
            stationary = Session(directory, BOARD, batch.serials, simulated=True, capture_mode='stationary')
            stationary.add(packet, 'training')
            saved = json.loads((stationary.directory / 'manifest.json').read_text())
            self.assertEqual(saved['capture_mode'], 'stationary')
            self.assertEqual(saved['captures'][0]['capture_mode'], 'stationary')
            self.assertEqual(len(saved['captures'][0]['timing_warnings']), 2)
            moving = Session(directory, BOARD, batch.serials, simulated=True, capture_mode='motion')
            with self.assertRaisesRegex(ValueError, 'Clock offset'):
                moving.add(packet, 'training')
            self.assertEqual(moving.samples, [])

    def test_reserved_pose_matching_detects_repeated_validation_geometry(self):
        packet = self.packet()
        validation = Coverage(packet['batch'].serials)
        self.assertFalse(validation.matches_pose(packet['observations']))
        validation.add(packet['observations'])
        self.assertTrue(validation.matches_pose(packet['observations']))
        different = self.source.render(15)
        observations = {s: observe(self.board, f) for s, f in different.frames.items()}
        self.assertFalse(validation.matches_pose(observations))

    def test_failed_image_write_never_commits_manifest(self):
        packet = self.packet()
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory, BOARD, packet['batch'].serials, simulated=True)
            with patch('cv2.imwrite', return_value=False), self.assertRaises(OSError):
                session.add(packet, 'training')
            self.assertEqual(session.samples, [])
            self.assertEqual(json.loads((session.directory / 'manifest.json').read_text())['captures'], [])


class CalibrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.board = load_config(BOARD)['big36_0']
        cls.source = SimulatedSource(cls.board, 4)
        cls.records = [make_record(cls.source.render(i), cls.board, i, 'training' if i < 24 else 'validation')
                       for i in range(30)]
        cls.result = solve_snapshot(BOARD, cls.source.serials, cls.records, max_iterations=70)

    def test_real_rendered_images_complete_a_connected_solve(self):
        self.assertTrue(self.result['solver']['success'])
        self.assertEqual(set(self.result['cameras']), set(self.source.serials))
        np.testing.assert_allclose(self.result['camera_poses']['SIM-01'], np.eye(4), atol=1e-12)
        self.assertEqual(self.result['accuracy_status'], 'unverified')
        for s in self.source.serials:
            self.assertLess(self.result['training'][s]['rms'], .6)
            self.assertLess(self.result['validation'][s]['rms'], 1.0)
            self.assertEqual(self.result['validation'][s]['failed_points'], 0)
        self.assertFalse(set(self.result['training_ids']) & set(self.result['validation_ids']))

    def test_held_out_validation_detects_wrong_camera_translation(self):
        cameras = [Camera(v['image_size'], np.array(v['K']), np.array(v['dist'])) for v in self.result['cameras'].values()]
        poses = np.array(list(self.result['camera_poses'].values()))
        poses[-1, 0, 3] += .05
        measured = validate(self.board, self.records[24:], self.source.serials, cameras, poses)
        self.assertGreater(measured['SIM-04']['rms'], self.result['validation']['SIM-04']['rms'] * 10)

    def test_unevaluable_validation_points_are_counted_as_failed(self):
        records = copy.deepcopy(self.records[24:25])
        for serial in self.source.serials[1:]:
            records[0]['detections'][serial] = dict(ids=[], corners=[], usable=False)
        cameras = [Camera(v['image_size'], np.array(v['K']), np.array(v['dist'])) for v in self.result['cameras'].values()]
        measured = validate(self.board, records, self.source.serials, cameras, np.array(list(self.result['camera_poses'].values())))
        self.assertEqual(measured['SIM-01']['n'], 0)
        self.assertIsNone(measured['SIM-01']['rms'])
        self.assertGreater(measured['SIM-01']['failed_points'], 0)

    def test_seed_can_be_validated_without_refitting_or_training(self):
        seed = copy.deepcopy(self.result)
        measured = solve_snapshot(BOARD, self.source.serials, self.records[24:], seed=seed)
        self.assertEqual(measured['solver']['mode'], 'validation_only')
        self.assertEqual(measured['camera_poses'], seed['camera_poses'])
        self.assertEqual(measured['cameras'], seed['cameras'])
        self.assertEqual(measured['validation'], self.result['validation'])
        self.assertEqual(measured['training_ids'], [])

    def test_seed_refinement_fixes_intrinsics_and_preserves_nonidentity_world_anchor(self):
        seed = copy.deepcopy(self.result)
        world = np.eye(4)
        world[:3, :3] = cv2.Rodrigues(np.array([.2, .1, -.3]))[0]
        world[:3, 3] = [2, -1, .5]
        for serial in self.source.serials:
            seed['camera_poses'][serial] = (np.asarray(seed['camera_poses'][serial]) @ world).tolist()
        seed['camera_poses']['SIM-04'][0][3] += .03
        result = solve_snapshot(BOARD, self.source.serials, self.records, seed=seed, max_iterations=50)
        self.assertEqual(result['cameras'], seed['cameras'])
        np.testing.assert_allclose(result['camera_poses']['SIM-01'], seed['camera_poses']['SIM-01'], atol=1e-12)
        self.assertLess(result['validation']['SIM-04']['rms'], result['seed_validation']['SIM-04']['rms']/5)


if __name__ == '__main__':
    unittest.main()
