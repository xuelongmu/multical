"""Detector measurements and coverage; no RMS-to-millimetres conversion."""
import cv2
import numpy as np


GRID = (12, 9)


def observe(board, frame):
    detection = board.detect(frame.image)
    corners, ids = detection.corners.copy(), detection.ids.copy()
    usable = board.has_min_detections(detection)
    height, width = frame.image.shape
    grid = np.zeros(GRID[::-1], dtype=bool)
    descriptor = None
    area = 0.0
    square_px = None
    if len(corners):
        xy = np.clip((corners / [width, height] * GRID).astype(int), 0, np.array(GRID)-1)
        grid[xy[:, 1], xy[:, 0]] = True
    if usable:
        homography, _ = cv2.findHomography(board.points[ids, :2], corners, 0)
        if homography is not None:
            width_m, height_m = np.array(board.size) * board.square_length
            outline = np.float32([[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]])
            projected = cv2.perspectiveTransform(outline[None], homography)[0]
            if np.isfinite(projected).all():
                descriptor = (projected / [width, height]).ravel()
                area = cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / (width * height)
                square_px = min(np.linalg.norm(projected[1]-projected[0]) / board.size[0],
                                np.linalg.norm(projected[3]-projected[0]) / board.size[1])
    sharpness = float(cv2.Laplacian(cv2.resize(frame.image, (480, 360)), cv2.CV_64F).var())
    return dict(ids=ids, corners=corners, usable=usable and descriptor is not None,
                grid=grid, descriptor=descriptor, area=area,
                marker_px=None if square_px is None else square_px * board.marker_length / board.square_length,
                sharpness=sharpness)


class Coverage:
    def __init__(self, serials):
        self.serials = tuple(serials)
        self.cells = {s: np.zeros(GRID[::-1], dtype=np.int32) for s in serials}
        self.descriptors = {s: [] for s in serials}
        self.views = {s: 0 for s in serials}
        self.overlaps = np.zeros((len(serials), len(serials)), dtype=int)
        self.pose_groups = []

    def matches_pose(self, observations):
        for group in self.pose_groups:
            common = [s for s in group if s in observations and observations[s]['usable']]
            if common and all(np.linalg.norm(group[s] - observations[s]['descriptor']) < .075 for s in common):
                return True
        return False

    def novel(self, serial, observation):
        value = observation['descriptor']
        if value is None:
            return False
        previous = self.descriptors[serial]
        return not previous or min(np.linalg.norm(value - d) for d in previous) >= .075

    def add(self, observations):
        if not self.matches_pose(observations):
            self.pose_groups.append({s: o['descriptor'].copy() for s, o in observations.items() if o['usable']})
        useful = []
        novel = {s: self.novel(s, observations[s]) for s in self.serials}
        for index, serial in enumerate(self.serials):
            observation = observations[serial]
            if observation['usable']:
                useful.append(index)
                if novel[serial]:
                    self.cells[serial] += observation['grid']
                    self.descriptors[serial].append(observation['descriptor'].copy())
                    self.views[serial] += 1
        for i in useful:
            for j in useful:
                if i != j and (novel[self.serials[i]] or novel[self.serials[j]]):
                    self.overlaps[i, j] += 1

    def snapshot(self):
        return dict(cells={s: c.copy() for s, c in self.cells.items()},
                    views=self.views.copy(), overlaps=self.overlaps.copy())

    def guidance(self, observations, problems):
        if problems:
            return problems[0]
        usable = [s for s, o in observations.items() if o['usable']]
        if not usable:
            return 'Show the board to a camera. Spread corners over at least three rows and columns.'
        tiny = [s for s in usable if observations[s]['marker_px'] is not None and observations[s]['marker_px'] < 24]
        if tiny:
            return f'Move closer to {tiny[0]} or turn the board toward it; markers are small in this view.'
        if len(usable) == 1 and len(self.serials) > 1:
            return 'For camera alignment, show the same board to two or more cameras at once.'
        weakest = min(usable, key=lambda s: np.count_nonzero(self.cells[s]))
        if not self.novel(weakest, observations[weakest]):
            cell = np.unravel_index(np.argmin(self.cells[weakest]), GRID[::-1])
            vertical = ['upper', 'middle', 'lower'][min(2, cell[0] // 3)]
            horizontal = ['left', 'centre', 'right'][min(2, cell[1] // 4)]
            return f'{weakest}: move toward the {vertical} {horizontal} of the image and change the board tilt.'
        return 'Useful new coverage. Hold the board still, then capture a training pose.'


def live_projection(board, packet, result):
    """Estimate board pose in one view and predict the other cameras."""
    choices = [s for s, o in packet['observations'].items()
               if o['usable'] and s in result['cameras']]
    if not choices:
        return None
    reference = max(choices, key=lambda s: packet['observations'][s]['area'])
    obs = packet['observations'][reference]
    camera = result['cameras'][reference]
    ok, rv, tv = cv2.solvePnP(board.points[obs['ids']], obs['corners'],
                            np.array(camera['K']), np.array(camera['dist']))
    if not ok:
        return None
    target = np.eye(4)
    target[:3, :3] = cv2.Rodrigues(rv)[0]
    target[:3, 3] = tv.ravel()
    world_target = np.linalg.inv(np.array(result['camera_poses'][reference])) @ target
    predictions = {}
    for serial, observation in packet['observations'].items():
        if serial not in result['cameras'] or not len(observation['ids']):
            continue
        model = result['cameras'][serial]
        pose = np.array(result['camera_poses'][serial]) @ world_target
        points = board.points[observation['ids']]
        if np.any((points @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
                                     np.array(model['K']), np.array(model['dist']))[0].reshape(-1, 2)
        error = np.linalg.norm(projected - observation['corners'], axis=1)
        predictions[serial] = dict(points=projected, rms=float(np.sqrt(np.mean(error**2))))
    return dict(reference=reference, world_target=world_target, predictions=predictions)
