"""Frame sources. Importing this module never opens a camera."""
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import site
from pathlib import Path
import sys
import time

import cv2
import numpy as np


@dataclass
class Frame:
    serial: str
    image: np.ndarray
    frame_id: int
    timestamp_ns: int
    exposure_us: float
    gain_db: float = 0.0
    settings: dict = field(default_factory=dict)

    @property
    def start_ns(self):
        # CaptureNet documents this model's image timestamp as end of exposure.
        return self.timestamp_ns - int(self.exposure_us * 1000)

    def metadata(self):
        return dict(serial=self.serial, frame_id=self.frame_id,
                    timestamp_ns=self.timestamp_ns, exposure_us=self.exposure_us,
                    gain_db=self.gain_db, image_size=list(self.image.shape[1::-1]), settings=self.settings)


@dataclass
class FrameSet:
    sequence: int
    scheduled_ns: int
    serials: tuple
    frames: dict
    errors: dict = field(default_factory=dict)
    simulated: bool = False

    def problems(self, max_spread_us=20.0):
        issues = list(self.errors.values())
        missing = set(self.serials) - set(self.frames)
        if missing:
            issues.append("Missing cameras: " + ", ".join(sorted(missing)))
        if set(self.frames) - set(self.serials):
            issues.append("Unexpected camera identity")
        starts = []
        for serial, frame in self.frames.items():
            if serial != frame.serial or not np.isfinite(frame.exposure_us) or frame.exposure_us <= 0:
                issues.append(f"{serial}: invalid identity or exposure")
                continue
            if frame.image.ndim != 2 or frame.image.size == 0:
                issues.append(f"{serial}: invalid image")
            if abs(frame.start_ns - self.scheduled_ns) > 1_000_000:
                issues.append(f"{serial}: frame does not match the scheduled action")
            starts.append(frame.start_ns)
        if len(starts) > 1 and (max(starts) - min(starts)) / 1000 > max_spread_us:
            issues.append(f"Exposure-start spread exceeds {max_spread_us:g} us")
        midpoints = [f.timestamp_ns - f.exposure_us*500 for f in self.frames.values()]
        if len(midpoints) > 1 and (max(midpoints) - min(midpoints)) / 1000 > max_spread_us:
            issues.append('Exposure midpoints differ; use a shared exposure for all cameras')
        return issues

    @property
    def spread_us(self):
        starts = [f.start_ns for f in self.frames.values()]
        return (max(starts) - min(starts)) / 1000 if len(starts) > 1 else None


class SimulatedSource:
    """Rendered ChArUco images exercise the real detector, never injected corners."""
    def __init__(self, board, count=4, fps=8.0):
        self.board, self.count, self.fps = board, count, fps
        self.serials = tuple(f"SIM-{i+1:02d}" for i in range(count))
        self.sequence = 0
        self.start_time = None
        self.texture = board.draw(pixels_mm=1, margin=0)
        self.size = (960, 720)
        self.K = np.array([[1000., 0, 480], [0, 1000., 360], [0, 0, 1.]])
        self.camera_poses = []
        for i in range(count):
            pose = np.eye(4)
            pose[:3, 3] = [-0.9 + 1.8 * i / max(1, count-1), 0.10 * np.sin(i), 0]
            self.camera_poses.append(pose)

    def open(self):
        self.start_time = time.monotonic()
        self.deadline = self.start_time
        return self.serials

    def render(self, pose_index):
        rng = np.random.default_rng(pose_index + 83)
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rng.uniform([-.45, -.45, -.2], [.45, .45, .2]))[0]
        centre = np.array([self.board.size[0], self.board.size[1], 0.]) * self.board.square_length / 2
        transform[:3, 3] = rng.uniform([-.6, -.4, 2.5], [.6, .4, 3.9]) - transform[:3, :3] @ centre
        width, height = np.array(self.board.size) * self.board.square_length
        outline = np.float32([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]])
        h, w = self.texture.shape
        source = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        self.sequence += 1
        scheduled = self.sequence * 1_000_000_000
        frames = {}
        for index, (serial, camera) in enumerate(zip(self.serials, self.camera_poses)):
            pose = camera @ transform
            projected = cv2.projectPoints(outline, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], self.K, None)[0]
            homography = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
            image = cv2.warpPerspective(self.texture, homography, self.size, borderValue=205)
            image = cv2.GaussianBlur(image, (3, 3), .5)
            frames[serial] = Frame(serial, image, self.sequence, scheduled + 1_000_000 + index*200, 1000.)
        return FrameSet(self.sequence, scheduled, self.serials, frames, simulated=True)

    def read(self):
        delay = self.deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.deadline = time.monotonic() + 1 / self.fps
        return self.render(int((time.monotonic() - self.start_time) / 1.8))

    def close(self):
        pass


class PySpinSource:
    """Direct PTP + scheduled Action0 capture, based on CaptureNet fire3.py.

    ref: /home/zerospace/capturenet/scripts/fire3.py (keys, end timestamps,
    multi-interface action commands). No Captury process or files are used.
    """
    def __init__(self, expected_serials=(), expected_count=25, fps=5.0, sdk_path=None,
                 exposure_us=None, gain_db=None):
        self.expected_serials = tuple(expected_serials)
        self.expected_count = expected_count
        self.fps, self.sdk_path = fps, sdk_path
        self.exposure_us, self.gain_db = exposure_us, gain_db
        self.system = self.camera_list = self.interfaces = self.pool = None
        self.held = []
        self.sequence = 0
        self.last_ids = {}
        self.sizes = {}

    def _check_cancelled(self):
        event = getattr(self, 'stop_event', None)
        if event is not None and event.is_set():
            raise InterruptedError('Camera connection cancelled')

    def _node(self, nm, name, kind):
        ptr = getattr(self.sdk, f"C{kind}Ptr")(nm.GetNode(name))
        if not self.sdk.IsReadable(ptr):
            raise RuntimeError(f"Camera node {name} is not readable")
        return ptr

    def _get(self, nm, name, kind):
        node = self._node(nm, name, kind)
        return node.ToString() if kind == "Enumeration" else node.GetValue()

    def _set(self, nm, name, kind, value):
        node = self._node(nm, name, kind)
        if not self.sdk.IsWritable(node):
            raise RuntimeError(f"Camera node {name} is not writable")
        if kind == "Enumeration":
            node.SetIntValue(node.GetEntryByName(value).GetValue())
        else:
            node.SetValue(value)

    def _command(self, nm, name):
        self.sdk.CCommandPtr(nm.GetNode(name)).Execute()

    def _ptp(self, cam):
        nm = cam.GetNodeMap()
        self._command(nm, 'GevIEEE1588DataSetLatch')
        status = self._get(nm, 'GevIEEE1588StatusLatched', 'Enumeration')
        offset = self._get(nm, 'GevIEEE1588OffsetFromMasterLatched', 'Integer')
        if status != 'Slave' or abs(offset) > 1000:
            raise RuntimeError(f"PTP is {status}, offset {offset} ns; require Slave within 1000 ns")

    def open(self):
        self._check_cancelled()
        if self.sdk_path:
            sys.path.append(self.sdk_path)
        elif importlib.util.find_spec('PySpin') is None:
            # Spinnaker's Python wheel is often installed for the user rather
            # than in this virtualenv. Append so the venv's NumPy stays first.
            user_site = site.getusersitepackages()
            if (Path(user_site) / 'PySpin.py').is_file():
                sys.path.append(user_site)
        try:
            self.sdk = importlib.import_module('PySpin')
        except ImportError as exc:
            raise RuntimeError("Spinnaker PySpin is unavailable in this Python. Use --sdk-path for its installation directory.") from exc
        self.system = self.sdk.System.GetInstance()
        self.camera_list = self.system.GetCameras()
        self.interfaces = self.system.GetInterfaces()
        if self.camera_list.GetSize() == 0:
            raise RuntimeError('No cameras discovered. Check GigE interfaces and camera power.')
        discovered = {}
        for i in range(self.camera_list.GetSize()):
            cam = self.camera_list[i]
            serial = self._get(cam.GetTLDeviceNodeMap(), 'DeviceSerialNumber', 'String')
            discovered[serial] = i
        del cam  # SDK references must not survive cleanup.
        serials = self.expected_serials or tuple(sorted(discovered))
        missing = set(serials) - set(discovered)
        if missing or len(serials) != self.expected_count or len(set(serials)) != len(serials):
            raise RuntimeError(f"Expected {self.expected_count} distinct cameras; discovered {len(discovered)}, selected {len(set(serials) & set(discovered))}. Missing: {sorted(missing)}")
        self.serials = tuple(serials)
        settings = [('AcquisitionMode', 'Enumeration', 'Continuous'),
                    ('ExposureAuto', 'Enumeration', 'Off'), ('GainAuto', 'Enumeration', 'Off'),
                    ('TriggerSource', 'Enumeration', 'Action0'),
                    ('TriggerOverlap', 'Enumeration', 'ReadOut'),
                    ('ActionUnconditionalMode', 'Enumeration', 'Off'),
                    ('ActionDeviceKey', 'Integer', 42), ('ActionGroupKey', 'Integer', 1),
                    ('ActionGroupMask', 'Integer', 0xFFFFFFFF)]
        if self.exposure_us is not None:
            settings.append(('ExposureTime', 'Float', self.exposure_us))
        if self.gain_db is not None:
            settings.append(('Gain', 'Float', self.gain_db))
        for serial in serials:
            self._check_cancelled()
            cam = self.camera_list[discovered[serial]]
            cam.Init()  # fails visibly when another application owns the camera
            state = dict(serial=serial, cam=cam, restore=[], acquiring=False)
            self.held.append(state)
            nm = cam.GetNodeMap()
            self._ptp(cam)
            state['trigger_selector'] = self._get(nm, 'TriggerSelector', 'Enumeration')
            self._set(nm, 'TriggerSelector', 'Enumeration', 'FrameStart')
            state['trigger_mode'] = self._get(nm, 'TriggerMode', 'Enumeration')
            self._set(nm, 'TriggerMode', 'Enumeration', 'Off')
            for name, kind, value in settings:
                self._check_cancelled()
                old = self._get(nm, name, kind)
                state['restore'].append((name, kind, old))
                self._set(nm, name, kind, value)
            self._set(nm, 'TriggerMode', 'Enumeration', 'On')
            # Reading exposure must succeed; zero is never a substitute.
            state['exposure'] = self._get(nm, 'ExposureTime', 'Float')
            state['gain'] = self._get(nm, 'Gain', 'Float')
            state['processor'] = self.sdk.ImageProcessor()
            state['processor'].SetColorProcessing(self.sdk.HQ_LINEAR)
            state['settings'] = dict(pixel_format=self._get(nm, 'PixelFormat', 'Enumeration'),
                                     offset_x=self._get(nm, 'OffsetX', 'Integer'),
                                     offset_y=self._get(nm, 'OffsetY', 'Integer'),
                                     exposure_auto='Off', gain_auto='Off')
            cam.BeginAcquisition()
            state['acquiring'] = True
        self.pool = ThreadPoolExecutor(max_workers=len(self.held))
        self.last_ptp = time.monotonic()
        self.deadline = time.monotonic()
        return self.serials

    def _collect(self, state):
        image = None
        try:
            image = state['cam'].GetNextImage(1200)
            if image.IsIncomplete():
                raise RuntimeError(f"Incomplete image ({image.GetImageStatus()})")
            converted = state['processor'].Convert(image, self.sdk.PixelFormat_Mono8)
            pixels = converted.GetNDArray().copy()
            return Frame(state['serial'], pixels, int(image.GetFrameID()),
                         int(image.GetTimeStamp()), state['exposure'], state['gain'], state['settings'])
        finally:
            if image is not None:
                image.Release()

    def read(self):
        delay = self.deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._check_cancelled()
        self.deadline = time.monotonic() + 1 / self.fps
        if time.monotonic() - self.last_ptp > 3:
            for state in self.held:
                self._ptp(state['cam'])
                state['exposure'] = self._get(state['cam'].GetNodeMap(), 'ExposureTime', 'Float')
                state['gain'] = self._get(state['cam'].GetNodeMap(), 'Gain', 'Float')
            self.last_ptp = time.monotonic()
        nm = self.held[0]['cam'].GetNodeMap()
        self._command(nm, 'TimestampLatch')
        scheduled = self._get(nm, 'TimestampLatchValue', 'Integer') + 150_000_000
        for i in range(self.interfaces.GetSize()):
            iface = self.interfaces[i]
            cams = iface.GetCameras()
            count = cams.GetSize()
            cams.Clear()
            if not count:
                continue
            inm = iface.GetTLNodeMap()
            for name, value in [('GevActionDeviceKey', 42), ('GevActionGroupKey', 1),
                                ('GevActionGroupMask', 0xFFFFFFFF), ('GevActionTime', scheduled)]:
                self._set(inm, name, 'Integer', value)
            self._command(inm, 'ActionCommand')
        futures = [(s['serial'], self.pool.submit(self._collect, s)) for s in self.held]
        frames, errors = {}, {}
        for serial, future in futures:
            try:
                frame = future.result()
                if frame.frame_id <= self.last_ids.get(serial, -1):
                    raise RuntimeError("Repeated or out-of-order camera frame ID")
                if serial in self.sizes and frame.image.shape != self.sizes[serial]:
                    raise RuntimeError("Image geometry changed during this session")
                self.last_ids[serial] = frame.frame_id
                self.sizes[serial] = frame.image.shape
                frames[serial] = frame
            except Exception as exc:
                errors[serial] = f"{serial}: {exc}"
        self.sequence += 1
        return FrameSet(self.sequence, scheduled, self.serials, frames, errors)

    def close(self):
        failures = []
        if self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None
        while self.held:
            state = self.held.pop()
            cam = state.pop('cam')
            if state['acquiring']:
                try:
                    cam.EndAcquisition()
                except Exception as exc:
                    failures.append(f"End acquisition {state['serial']}: {exc}")
            try:
                nm = cam.GetNodeMap()
                if 'trigger_mode' in state:
                    self._set(nm, 'TriggerMode', 'Enumeration', 'Off')
                for name, kind, value in reversed(state['restore']):
                    try:
                        self._set(nm, name, kind, value)
                    except Exception as exc:
                        failures.append(f"Restore {state['serial']} {name}: {exc}")
                if 'trigger_mode' in state:
                    self._set(nm, 'TriggerMode', 'Enumeration', state['trigger_mode'])
                if 'trigger_selector' in state:
                    self._set(nm, 'TriggerSelector', 'Enumeration', state['trigger_selector'])
            except Exception as exc:
                failures.append(f"Close {state['serial']}: {exc}")
            finally:
                try:
                    cam.DeInit()
                except Exception as exc:
                    failures.append(f"Deinitialize {state['serial']}: {exc}")
                del cam
        if self.camera_list is not None:
            self.camera_list.Clear()
            self.camera_list = None
        if self.interfaces is not None:
            self.interfaces.Clear()
            self.interfaces = None
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None
        if failures:
            raise RuntimeError("; ".join(failures))
