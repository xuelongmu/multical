"""Recorder controller and integrity tests; GPU integration is a separate smoke script."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from multical.live.recorder import NativeSource
from multical.live.recording_verify import verify_take


class RecorderTests(unittest.TestCase):
    def test_stop_during_preparation_is_not_lost_and_status_never_waits_for_io(self):
        source = NativeSource(expected_count=1, simulated=True)
        source.serials = ('SIM-01',)
        messages = []
        def send(message):
            # This would deadlock if preparation held the status mutex over IPC.
            self.assertEqual(source.recording_status()['state'], 'preparing')
            messages.append(message)
            if message['command'] == 'record':
                source.stop_recording()
        source._send = send
        with tempfile.TemporaryDirectory() as root:
            output = source.start_recording(root, seconds=1, calibration={'cameras': {}})
            source.prepare_thread.join(timeout=5)
            self.assertFalse(source.prepare_thread.is_alive())
            self.assertEqual([m['command'] for m in messages], ['record', 'stop'])
            context = json.loads((output / 'context.json').read_text())
            self.assertEqual(context['calibration_file'], 'calibration.json')
            self.assertEqual(len(context['calibration_sha256']), 64)

    def test_invalid_recording_settings_create_no_take(self):
        source = NativeSource()
        with tempfile.TemporaryDirectory() as root:
            for kwargs in ({'fps': 61}, {'seconds': 121}, {'quality': 40}):
                with self.assertRaises(ValueError):
                    source.start_recording(root, **kwargs)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_close_reports_restore_failure(self):
        source = NativeSource()
        source.restore_error = 'Camera settings restore failed'
        with self.assertRaisesRegex(RuntimeError, 'restore failed'):
            source.close()

    def test_failed_start_keeps_new_take_identity_when_old_take_status_arrives(self):
        source = NativeSource()
        source.closing = True
        source.state = dict(state='arming', directory='new-take')
        source.process = Mock(stdout=iter([
            json.dumps(dict(event='command_error', error='PTP not ready')),
            json.dumps(dict(event='recording', status=dict(state='saved', directory='old-take')))]))
        source._read_events()
        status = source.recording_status()
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['directory'], 'new-take')

    def test_cancelled_preparation_is_not_erased_by_idle_updates(self):
        source = NativeSource()
        source.closing = True
        source.state = dict(state='cancelled', directory='new-take')
        source.process = Mock(stdout=iter([json.dumps(dict(event='recording', status=dict(state='idle')))]))
        source._read_events()
        self.assertEqual(source.recording_status()['state'], 'cancelled')

    def fixture(self, root):
        rows = []
        for i in range(3):
            t = 1_000_000_000 + i * 1_000_000_000 // 60
            rows.append(dict(frame_index=i, scheduled_ns=t, timestamp_ns=t+100_000,
                             exposure_start_ns=t, exposure_midpoint_ns=t+50_000,
                             frame_id=i+30, segment=0, packet=i, bytes=10))
        manifest = dict(state='saved', complete=True, error='', fps=60, scheduled_frames=3,
                        configuration=dict(cameras=[], simulated=True), cameras={}, segments={})
        for serial in ('A', 'B'):
            path = root / 'cameras' / serial
            path.mkdir(parents=True)
            self.write_rows(path / 'frames.csv', rows)
            (path / 'segment_00000.mkv').write_bytes(b'x' * 50)
            manifest['configuration']['cameras'].append(dict(serial=serial, width=1440, height=1080, exposure_us=100))
            manifest['cameras'][serial] = dict(received=3, encoded=3, written=3)
            manifest['segments'][serial] = [dict(segment=0, frames=3, finalized=True, first_scheduled_index=0,
                                               jpeg_bytes=30, file=f'cameras/{serial}/segment_00000.mkv')]
        (root / 'context.json').write_text(json.dumps(dict(camera_serials=['A', 'B'])))
        (root / 'take.json').write_text(json.dumps(manifest))
        return rows, manifest

    @staticmethod
    def write_rows(path, rows):
        with path.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_verifier_decodes_every_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch('multical.live.recording_verify.decode_segment', return_value=3) as decode:
                result = verify_take(root)
            self.assertEqual(result['decoded_frames'], 6)
            self.assertEqual(decode.call_count, 2)

    def test_verifier_rejects_missing_frame_even_if_summary_claims_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _ = self.fixture(root)
            self.write_rows(root / 'cameras/A/frames.csv', rows[:1] + rows[2:])
            with self.assertRaisesRegex(ValueError, 'missing, duplicate'):
                verify_take(root, decode=False)

    def test_verifier_rejects_camera_schedule_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _ = self.fixture(root)
            for row in rows:
                row['scheduled_ns'] += 100
            self.write_rows(root / 'cameras/B/frames.csv', rows)
            with self.assertRaisesRegex(ValueError, 'differ between cameras'):
                verify_take(root, decode=False)

    def test_verifier_never_accepts_incomplete_take(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self.fixture(root)
            manifest['complete'] = False
            (root / 'take.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'not marked complete'):
                verify_take(root)
