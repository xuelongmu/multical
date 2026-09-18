"""Captury -> multical -> centered, undistorted COLMAP camera bridge.

Run `python -m multical.io.interop --help`. Exported camera files alone are not
a trainable 4C4D dataset: RGB sequences and scene points are separate inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .captury import read_captury
from .import_calib import load_calibration


def load_seed(path, units=None):
    """Normalize stock multical R/T or live 4x4 poses for the live interface."""
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('units', units) != 'metres':
        raise ValueError('Calibration units must be metres; legacy JSON needs an explicit units declaration')
    if data.get('transform_convention', 'world_to_camera') != 'world_to_camera':
        raise ValueError('Expected world_to_camera transforms')
    if not data.get('cameras') or not data.get('camera_poses'):
        raise ValueError('Seed needs intrinsics and extrinsics for every camera')
    if all(isinstance(v, dict) for v in data['camera_poses'].values()):
        imported = load_calibration(path)
        data['camera_poses'] = {s: v.tolist() for s, v in imported.camera_poses.items()}
    if set(data['camera_poses']) != set(data['cameras']):
        raise ValueError('Intrinsics and poses must have identical camera names')
    for serial, model in data['cameras'].items():
        k, d = np.asarray(model['K'], float), np.asarray(model['dist'], float).ravel()
        size = np.asarray(model['image_size'])
        pose = np.asarray(data['camera_poses'][serial], float)
        if model.get('model', 'standard') != 'standard':
            raise ValueError(f'{serial}: bridge currently supports the standard OpenCV camera model')
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError(f'{serial}: invalid intrinsics')
        if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
            raise ValueError(f'{serial}: unsupported skew/projective intrinsics')
        if size.shape != (2,) or np.any(size <= 0) or not np.equal(size, np.floor(size)).all():
            raise ValueError(f'{serial}: invalid resolution')
        if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
            raise ValueError(f'{serial}: invalid distortion')
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f'{serial}: invalid pose')
        r = pose[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
            raise ValueError(f'{serial}: pose must contain a proper rotation')
    # Historical scores/board poses belong to the source session, not this one.
    # The original file remains available via its path and content hash.
    data.update(frame_poses={}, training={}, validation={}, training_ids=[], validation_ids=[],
                accuracy_status='unverified')
    data.update(units='metres', transform_convention='world_to_camera')
    data['bootstrap_source'] = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    return data


def validate_seed_geometry(seed, serials, sizes):
    if set(serials) != set(seed['cameras']):
        missing = sorted(set(serials) - set(seed['cameras']))
        extra = sorted(set(seed['cameras']) - set(serials))
        raise ValueError(f'Seed camera roster differs: missing={missing}, extra={extra}')
    for serial in serials:
        if tuple(sizes[serial]) != tuple(seed['cameras'][serial]['image_size']):
            raise ValueError(f'{serial}: seed resolution differs from native image; do not silently scale or rotate')


def pinhole_model(camera):
    """Center the output principal point to match 4C4D's COLMAP reader.

    Keep focal length and resolution. Remapping can create black borders; do not
    crop/resize later without updating K. Both radial and tangential terms apply.
    """
    result = copy.deepcopy(camera)
    width, height = result['image_size']
    k = np.asarray(result['K'], float)
    # Its CUDA ndc2Pix(v, S) = ((v+1)*S-1)/2 puts NDC zero at
    # (S-1)/2 in OpenCV's integer pixel-centre coordinates, not S/2.
    k[0, 2], k[1, 2] = (width-1)/2, (height-1)/2
    result.update(K=k.tolist(), dist=[0., 0., 0., 0., 0.], model='standard')
    return result


def rectification_maps(camera, pinhole):
    return cv2.initUndistortRectifyMap(np.asarray(camera['K']), np.asarray(camera['dist']),
                                     None, np.asarray(pinhole['K']), tuple(pinhole['image_size']), cv2.CV_32FC1)


def export_colmap(seed, output, image_root=None):
    """Write one fixed-pose entry per camera, optionally rectify PNG sequences.

    image_root/<serial>/<frame>.png must use the same 0000..9999 frame set for
    every camera. Matching names do not establish hardware synchronization.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; stale binary COLMAP files can override text')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.multical-export-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        manifest = _write_colmap(seed, staging, image_root)
        staging.rename(output)
    return manifest


def _write_colmap(seed, output, image_root):
    serials = list(seed['cameras'])
    images = {}
    if image_root is not None:
        for serial in serials:
            paths = sorted((Path(image_root) / serial).glob('*.png'))
            if not paths or [p.stem for p in paths] != [f'{i:04d}' for i in range(len(paths))] or len(paths) > 10000:
                raise ValueError(f'{serial}: need contiguous PNG frames named 0000.png through at most 9999.png')
            images[serial] = paths
        if len({len(paths) for paths in images.values()}) != 1:
            raise ValueError('Unequal camera frame counts; synchronize streams before export')
    target = {s: pinhole_model(c) for s, c in seed['cameras'].items()}
    sparse = output / 'sparse' / '0'
    sparse.mkdir(parents=True)
    camera_lines = ['# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]']
    pose_lines = ['# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME; next line contains no tracks']
    mapping = {}
    for index, serial in enumerate(serials):
        camera, pose = target[serial], np.asarray(seed['camera_poses'][serial])
        w, h = camera['image_size']
        k = np.asarray(camera['K'])
        camera_lines.append(f'{index+1} PINHOLE {w} {h} ' + ' '.join(f'{v:.16g}' for v in (k[0, 0], k[1, 1], k[0, 2], k[1, 2])))
        xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        values = [xyzw[3], *xyzw[:3], *pose[:3, 3]]
        pose_lines.extend([f'{index+1} ' + ' '.join(f'{v:.16g}' for v in values) + f' {index+1} cam{index:02d}_0000.png', ''])
        mapping[serial] = f'cam{index:02d}'
        if images:
            maps = rectification_maps(seed['cameras'][serial], camera)
            image_dir = output / 'images'
            image_dir.mkdir(exist_ok=True)
            for frame, source in enumerate(images[serial]):
                image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape != (h, w, 3) or image.dtype != np.uint8:
                    raise ValueError(f'{source}: expected native-size, unrotated 8-bit RGB/BGR image')
                rectified = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
                if not cv2.imwrite(str(image_dir / f'cam{index:02d}_{frame:04d}.png'), rectified):
                    raise OSError('Could not write rectified image')
    (sparse / 'cameras.txt').write_text('\n'.join(camera_lines) + '\n')
    (sparse / 'images.txt').write_text('\n'.join(pose_lines) + '\n')
    manifest = dict(units='metres', transform_convention='world_to_camera', camera_map=mapping,
                    pixel_convention='OpenCV integer centres; target principal point (W-1)/2,(H-1)/2 for 4C4D ndc2Pix',
                    source=seed.get('bootstrap_source'), provenance=seed.get('provenance'),
                    source_cameras=seed['cameras'], pinhole_cameras=target,
                    images_rectified=bool(images), frames_per_camera=len(next(iter(images.values()))) if images else 0,
                    point_cloud_present=False, training_ready=False,
                    requirements=['Scene point cloud in this same world frame, built from training views',
                                  'Synchronized RGB sequences using these exact rectified camera models',
                                  'Explicit training/test view selection and 4C4D time-duration configuration'])
    (output / 'conversion.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    captury = commands.add_parser('captury', help='Convert a Captury v0.3 calibration into multical JSON')
    captury.add_argument('input')
    captury.add_argument('output')
    captury.add_argument('--image-size', nargs=2, type=int, required=True, metavar=('WIDTH', 'HEIGHT'))
    captury.add_argument('--frame', type=int)
    colmap = commands.add_parser('colmap', help='Export centered pinhole cameras; optional RGB PNG undistortion')
    colmap.add_argument('input')
    colmap.add_argument('output')
    colmap.add_argument('--image-root', help='Native RGB PNG folders named by camera serial')
    colmap.add_argument('--units', choices=['metres'], help='Declare units for legacy multical JSON only')
    args = parser.parse_args()
    try:
        if args.command == 'captury':
            data = read_captury(args.input, args.image_size, args.frame)
            with Path(args.output).open('x') as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
            print(f'Converted {len(data["cameras"])} cameras; native unrotated images, metres, accuracy unverified')
        else:
            manifest = export_colmap(load_seed(args.input, args.units), args.output, args.image_root)
            print(f'Exported {len(manifest["camera_map"])} cameras. Scene point cloud still required; see conversion.json.')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Conversion failed: {exc}\n')


if __name__ == '__main__':
    main()
