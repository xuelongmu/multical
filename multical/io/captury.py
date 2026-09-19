"""Read Captury's documented `tc camera calibration v0.3` format.

Geometry is for native, unrotated sensor images. EXIF orientation is retained as
metadata, not applied a second time to the camera basis. See CALIBRATION_INTEROP.md.
"""
from pathlib import Path
import hashlib
import numpy as np


def read_captury(path, image_size, frame=None):
    """Return stock-multical-compatible JSON data, keyed by hardware serial.

    Resolution is required: Captury stores physical sensor dimensions, not pixels.
    Animated files require an explicit frame; unsupported lens models fail closed.
    """
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    if not lines or lines[0].strip() != 'tc camera calibration v0.3':
        raise ValueError('Expected Captury calibration v0.3')
    width, height = image_size
    if any(int(v) != v or v <= 0 for v in image_size):
        raise ValueError('image_size must contain positive integer width and height')
    entries, camera, current = [], None, None
    for lineno, line in enumerate(lines[1:], 2):
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        key, values = fields[0], fields[1:]
        if key == 'camera':
            if len(values) < 2:
                raise ValueError(f'Line {lineno}: camera requires index and name')
            camera = dict(index=int(values[0]), name=' '.join(values[1:]), frames={})
            entries.append(camera)
            current = None
        elif camera is None:
            raise ValueError(f'Line {lineno}: field before camera')
        elif key == 'frame':
            number = int(values[0])
            if number in camera['frames']:
                raise ValueError(f'Line {lineno}: duplicate frame {number}')
            current = {}
            camera['frames'][number] = current
        elif key in ('serialNumber', 'cameraModel'):
            camera[key] = ' '.join(values)
        elif key not in ('colorCorrection', 'red', 'green', 'blue'):
            if current is None:
                raise ValueError(f'Line {lineno}: {key} outside a frame')
            if key in current:
                raise ValueError(f'Line {lineno}: duplicate {key}')
            current[key] = values
    if not entries:
        raise ValueError('No cameras in calibration')
    cameras, poses, metadata = {}, {}, {}
    for entry in entries:
        serial = entry.get('serialNumber')
        if not serial or serial in cameras:
            raise ValueError('Every camera needs a distinct serialNumber; camera indexes are not hardware identities')
        frames = entry['frames']
        if frame is None and len(frames) != 1:
            raise ValueError(f'{serial}: multiple/no frames; select --frame explicitly')
        selected = next(iter(frames)) if frame is None else frame
        if selected not in frames:
            raise ValueError(f'{serial}: frame {selected} absent')
        data = frames[selected]

        def numbers(key, count):
            value = np.asarray(data[key], dtype=float)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f'{serial}: invalid {key}')
            return value

        sensor = numbers('sensorSize', 2)
        focal = numbers('focalLength', 1)[0]
        aspect = numbers('pixelAspect', 1)[0]
        offset = numbers('centerOffset', 2)
        if min(*sensor, focal, aspect) <= 0:
            raise ValueError(f'{serial}: invalid sensor size, focal length or aspect')
        if data.get('distortionModel') != ['OpenCV']:
            raise ValueError(f'{serial}: only Captury OpenCV distortion is supported')
        dist = np.asarray(data['distortion'], dtype=float)
        if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
            raise ValueError(f'{serial}: unsupported OpenCV coefficient count')
        # Captury normalizes BOTH image axes by sensor width. Its reference
        # matrices multiply fy by pixelAspect (including the non-unit doc example).
        k = np.array([[focal, 0, sensor[0]/2 + offset[0]],
                      [0, focal*aspect, sensor[1]/2 + offset[1]],
                      [0, 0, sensor[0]/width]]) * (width / sensor[0])
        right, up = numbers('right', 3), numbers('up', 3)
        rotation = np.stack([right, -up, np.cross(right, -up)])
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0):
            raise ValueError(f'{serial}: right/up do not define an orthonormal camera')
        # Remove text/float serialization roundoff before Rodrigues/optimization.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        centre = numbers('origin', 3) / 1000.0
        translation = -rotation @ centre
        cameras[serial] = dict(model='standard', image_size=[int(width), int(height)],
                               K=k.tolist(), dist=dist.tolist())
        poses[serial] = dict(R=rotation.tolist(), T=translation.tolist())
        orientation = int(data.get('orientation', ['1'])[0])
        if orientation not in range(1, 9):
            raise ValueError(f'{serial}: invalid EXIF orientation')
        metadata[serial] = dict(index=entry['index'], name=entry['name'],
                                model=entry.get('cameraModel'), frame=selected,
                                orientation=orientation, focal_mm=float(focal),
                                time=int(data.get('time', ['0'])[0]))
    return dict(cameras=cameras, camera_poses=poses, units='metres',
                transform_convention='world_to_camera', accuracy_status='unverified',
                provenance=dict(format='captury-v0.3', source=str(path.resolve()),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                image_convention='native_unrotated_sensor', world_axes='Captury Y-up',
                                cameras=metadata))
