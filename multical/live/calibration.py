"""Calibrate a frozen live-session snapshot in a separate process."""
import cv2
import numpy as np
from cached_property import cached_property
from structs.numpy import Table
from structs.struct import struct

from multical import tables
from multical.board import load_config
from multical.camera import Camera
from multical.motion.static_frames import StaticFrames
from multical.optimization.calibration import Calibration
from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical.transform import rtvec


class AnchoredPoseSet(PoseSet):
    """A fixed pose remains valid and observable but has no optimizer variables."""
    def __init__(self, pose_table, names=None, fixed=0):
        super().__init__(pose_table, names)
        self.fixed = fixed

    @property
    def free(self):
        return [i for i in range(self.size) if self.valid[i] and i != self.fixed]

    @cached_property
    def params(self):
        return rtvec.from_matrix(self.poses[self.free]).ravel() if self.free else np.empty(0)

    def with_params(self, params):
        poses = self.poses.copy()
        if self.free:
            poses[self.free] = rtvec.to_matrix(params.reshape(-1, 6))
        return self.copy(pose_table=self.pose_table._extend(poses=poses))

    def sparsity(self, index_mapper, axis):
        return [(6, index_mapper.point_indexes(i, axis)) for i in self.free]

    def __getstate__(self):
        return dict(pose_table=self.pose_table, names=self.names, fixed=self.fixed)


def stats(errors):
    errors = np.asarray(errors)
    if not errors.size:
        return dict(n=0, rms=None, p95=None, maximum=None)
    return dict(n=int(errors.size), rms=float(np.sqrt(np.mean(errors**2))),
                p95=float(np.quantile(errors, .95)), maximum=float(errors.max()))


def _detections(records, serials):
    return [[[struct(corners=np.asarray(r['detections'][s]['corners'], dtype=np.float32).reshape(-1, 2),
                     ids=np.asarray(r['detections'][s]['ids'], dtype=int))]
             for r in records] for s in serials]


def validate(board, records, serials, cameras, camera_poses):
    """Each target view is predicted from another camera's board-pose estimate.

    Evaluation has no residual rejection. Failures count explicitly instead of
    silently shrinking a validation denominator.
    """
    errors = {s: [] for s in serials}
    expected = {s: 0 for s in serials}
    failures = {s: 0 for s in serials}
    for record in records:
        candidates = {}
        for index, serial in enumerate(serials):
            det = record['detections'][serial]
            corners = np.asarray(det['corners'], np.float32).reshape(-1, 2)
            ids = np.asarray(det['ids'], int)
            expected[serial] += len(ids)
            if not board.has_min_detections(struct(ids=ids, corners=corners)):
                continue
            ok, r, t = cv2.solvePnP(board.points[ids], corners, cameras[index].intrinsic, cameras[index].dist)
            if ok:
                pose = np.eye(4)
                pose[:3, :3] = cv2.Rodrigues(r)[0]
                pose[:3, 3] = t.ravel()
                candidates[index] = (len(ids), np.linalg.inv(camera_poses[index]) @ pose)
        for index, serial in enumerate(serials):
            det = record['detections'][serial]
            ids = np.asarray(det['ids'], int)
            alternatives = [i for i in candidates if i != index]
            if not len(ids):
                continue
            if not alternatives:
                failures[serial] += len(ids)
                continue
            source = max(alternatives, key=lambda i: candidates[i][0])
            pose = camera_poses[index] @ candidates[source][1]
            points = board.points[ids] @ pose[:3, :3].T + pose[:3, 3]
            if np.any(points[:, 2] <= 0):
                failures[serial] += len(ids)
                continue
            predicted = cameras[index].project(points)
            errors[serial].extend(np.linalg.norm(predicted - np.asarray(det['corners']), axis=1))
    return {s: dict(**stats(errors[s]), expected_points=expected[s], failed_points=failures[s]) for s in serials}


def solve_snapshot(board_file, serials, records, progress=lambda message: None, max_iterations=100):
    cv2.setNumThreads(1)
    board_map = load_config(board_file)
    if len(board_map) != 1:
        raise ValueError('The live workflow currently supports one rigid ChArUco board')
    board_name, board = next(iter(board_map.items()))
    training = [r for r in records if r['role'] == 'training']
    validation = [r for r in records if r['role'] == 'validation']
    if not training:
        raise ValueError('Capture training poses before solving')
    detections = _detections(training, serials)
    cameras, intrinsic_errors = [], {}
    for index, serial in enumerate(serials):
        count = sum(board.has_min_detections(d[0]) for d in detections[index])
        if count < 12:
            raise ValueError(f'{serial}: {count}/12 usable training views. Add varied board poses before solving.')
        sizes = {tuple(r['frames'][serial]['image_size']) for r in records}
        if len(sizes) != 1:
            raise ValueError(f'{serial}: image geometry changed')
        progress(f'Intrinsics {index+1}/{len(serials)} · {serial} · {count} views')
        camera, error = Camera.calibrate([board], .5, detections[index], sizes.pop(), max_iter=80, eps=1e-8)
        if not np.isfinite(camera.param_vec).all() or min(camera.focal_length) <= 0:
            raise ValueError(f'{serial}: invalid intrinsic fit')
        cameras.append(camera)
        intrinsic_errors[serial] = float(error)
    point_table = tables.make_point_table(detections, [board])
    pose_table = tables.make_pose_table(point_table, [board], cameras, True, 2.0)
    overlaps = tables.pattern_overlaps(pose_table)
    reached, pending = {0}, [0]
    while pending:
        current = pending.pop()
        for other in np.flatnonzero(overlaps[current] > 0):
            if int(other) not in reached:
                reached.add(int(other))
                pending.append(int(other))
    if len(reached) != len(serials):
        missing = [s for i, s in enumerate(serials) if i not in reached]
        raise ValueError('Disconnected camera observations: ' + ', '.join(missing) + '. Capture shared board poses.')
    progress('Initializing connected camera and board poses')
    initial = tables.initialise_poses(pose_table)
    if not initial.camera.valid.all() or not initial.times.valid.all():
        raise ValueError('Initialization could not explain every training camera/frame')
    frame_names = [r['id'] for r in training]
    calibration = Calibration(ParamList(cameras, list(serials)), ParamList([board], [board_name]), point_table,
                              AnchoredPoseSet(initial.camera, list(serials), fixed=0),
                              PoseSet(Table.create(poses=np.eye(4)[None], valid=np.ones(1, bool)), [board_name]),
                              StaticFrames(initial.times, frame_names))
    calibration = calibration.enable(cameras=False, boards=False, board_poses=False, camera_poses=True, motion=True)
    progress('Optimizing camera poses · fixed intrinsics and board geometry · no corner deletion')
    calibration = calibration.bundle_adjust(loss='soft_l1', f_scale=.5, max_iterations=max_iterations)
    residuals, valid = tables.reprojection_error(calibration.reprojected, point_table)
    if int(valid.sum()) != int(point_table.valid.sum()):
        raise ValueError('Calibration lost observations from the fixed training dataset')
    if not np.isfinite(residuals[valid]).all():
        raise ValueError('Calibration produced non-finite residuals')
    for ci, camera_pose in enumerate(calibration.camera_poses.poses):
        for ti, target_pose in enumerate(calibration.motion.poses):
            ids = np.flatnonzero(point_table.valid[ci, ti, 0])
            if ids.size:
                pose = camera_pose @ target_pose
                depths = (board.points[ids] @ pose[:3, :3].T + pose[:3, 3])[:, 2]
                if np.any(depths <= 0):
                    raise ValueError('Fitted observations lie behind a camera')
    training_stats = {s: stats(residuals[i][valid[i]]) for i, s in enumerate(serials)}
    progress('Evaluating independent validation captures across cameras')
    validation_stats = validate(board, validation, serials, cameras, calibration.camera_poses.poses)
    return dict(schema_version=1, accuracy_status='unverified', master=serials[0],
                transform_convention='world_to_camera', units='metres',
                training_ids=frame_names, validation_ids=[r['id'] for r in validation],
                cameras={s: dict(K=c.intrinsic.tolist(), dist=c.dist.tolist(), image_size=list(c.image_size), model=c.model)
                         for s, c in zip(serials, cameras)},
                camera_poses={s: pose.tolist() for s, pose in zip(serials, calibration.camera_poses.poses)},
                frame_poses={r['id']: pose.tolist() for r, pose in zip(training, calibration.motion.poses)},
                intrinsic_rms=intrinsic_errors, training=training_stats, validation=validation_stats,
                overlaps=overlaps.tolist(), solver=calibration.solver_status)


def solve_process(connection, board_file, serials, records):
    try:
        result = solve_snapshot(board_file, serials, records,
                                progress=lambda message: connection.send(('progress', message)))
        connection.send(('result', result))
    except Exception as exc:
        connection.send(('error', str(exc)))
    finally:
        connection.close()
