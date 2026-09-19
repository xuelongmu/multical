"""Native recorder controller and latest-frame preview source.

The child process owns Spinnaker and every full-rate recording buffer. Python
only reads sampled shared-memory previews; a slow GUI cannot drop recorded frames.
"""
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
from threading import Event, Lock, Thread
import time
import uuid

import cv2
import numpy as np

from .sources import Frame, FrameSet


BUSY_STATES = frozenset({'preparing', 'arming', 'armed', 'recording', 'draining'})
NATIVE_BINARY = Path(__file__).resolve().parents[2] / '.native' / 'multical-recorder'
WIDTH, HEIGHT = 1440, 1080
SLOT_BYTES = 64 + WIDTH * HEIGHT


def write_json_durable(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class NativeSource:
    def __init__(self, expected_serials=(), expected_count=25, fps=5., sdk_path=None,
                 exposure_us=None, gain_db=None, capture_mode='stationary', *,
                 simulated=False, binary=None, test_config=None):
        self.expected_serials = tuple(expected_serials)
        self.expected_count = expected_count
        self.fps, self.capture_mode = fps, capture_mode
        self.exposure_us, self.gain_db = exposure_us, gain_db
        self.simulated = simulated
        self.binary = Path(binary or NATIVE_BINARY)
        self.test_config = test_config or {}
        self.stop_event = Event()
        self.ready = Event()
        self.lock, self.command_lock = Lock(), Lock()
        self.previews = queue.Queue(maxsize=1)
        self.process = self.reader = self.runtime = self.memory = self.log = None
        self.prepare_thread = None
        self.closing = False
        self.error = None
        self.restore_error = None
        self.connection_status = 'Connecting native capture service…'
        self.camera_info = []
        self.serials = ()
        self.mac_to_serial = {}
        self.state = {'state': 'idle'}
        self.command_error = None
        self.start_pending = self.record_request_sent = self.cancel_requested = False

    def open(self):
        if not self.binary.is_file():
            raise RuntimeError('Build the recording service with bash tools/build_recorder.sh, '
                               'or select PySpin for calibration only.')
        if not 1 <= self.expected_count <= 25:
            raise ValueError('The native recorder supports 1–25 cameras')
        if not .2 <= self.fps <= 10:
            raise ValueError('Native preview rate must be 0.2–10 Hz; recording has its own 1–60 fps setting')
        self.runtime = tempfile.TemporaryDirectory(prefix='multical-recorder-', dir='/dev/shm')
        runtime = Path(self.runtime.name)
        preview = runtime / 'preview.bin'
        with preview.open('w+b') as data:
            data.truncate(self.expected_count * 2 * SLOT_BYTES)
            self.memory = mmap.mmap(data.fileno(), 0, access=mmap.ACCESS_READ)
        configuration = dict(preview_path=str(preview), count=self.expected_count,
                             serials=list(self.expected_serials), preview_fps=self.fps,
                             exposure_us=self.exposure_us, gain_db=self.gain_db, simulate=self.simulated)
        if self.test_config and not self.simulated:
            raise ValueError('Fault injection is allowed only with simulated cameras')
        configuration.update(self.test_config)
        config = runtime / 'config.json'
        config.write_text(json.dumps(configuration, allow_nan=False))
        self.log = tempfile.NamedTemporaryFile(prefix='multical-recorder-', suffix='.log', delete=False)
        self.process = subprocess.Popen([str(self.binary), str(config)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                        start_new_session=True)
        self.reader = Thread(target=self._read_events, name='recorder-events', daemon=True)
        self.reader.start()
        deadline = time.monotonic() + 180
        while not self.ready.wait(.1):
            self._check_running()
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out connecting native cameras; see ' + self.log.name)
        self._check_running()
        return self.serials

    def _check_running(self):
        if self.stop_event.is_set():
            raise InterruptedError('Native acquisition cancelled')
        if self.error:
            raise RuntimeError(self.error)
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f'Native recorder exited ({self.process.returncode}); see {self.log.name}')

    def _read_events(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # SDK diagnostic text is not an IPC message.
                event = message.get('event')
                if event == 'ready':
                    if message.get('protocol') != 1:
                        raise RuntimeError('Unsupported native recorder protocol')
                    self.camera_info = message['cameras']
                    self.serials = tuple(c['serial'] for c in self.camera_info)
                    if len(self.serials) != self.expected_count:
                        raise RuntimeError('Native camera count changed')
                    self.mac_to_serial = {c['mac']: c['serial'] for c in self.camera_info if c['mac']}
                    self.connection_status = f'{len(self.serials)} cameras connected'
                    self.ready.set()
                elif event == 'connecting':
                    self.connection_status = f"Connecting cameras · {message['connected']}/{message['total']}"
                elif event == 'preview':
                    try:
                        self.previews.get_nowait()
                    except queue.Empty:
                        pass
                    self.previews.put_nowait(message)
                elif event == 'recording':
                    incoming = message['status']
                    with self.lock:
                        if self.command_error or self.state.get('state') == 'cancelled':
                            continue
                        if incoming['state'] in BUSY_STATES:
                            self.start_pending = False
                        if not self.start_pending:
                            self.state = incoming
                elif event == 'command_error':
                    with self.lock:
                        self.command_error = message['error']
                        self.start_pending = False
                elif event == 'fatal':
                    self.error = message['error']
                    self.ready.set()
                elif event == 'restore_warning':
                    self.restore_error = f"Camera settings restore: {message['error']}"
        except Exception as exc:
            self.error = f'Recorder communication failed: {exc}'
            self.ready.set()
        finally:
            if not self.closing and not self.error:
                self.error = 'Native recorder connection closed'
                self.ready.set()

    def read(self):
        while True:
            self._check_running()
            try:
                packet = self.previews.get(timeout=.2)
            except queue.Empty:
                continue
            frames, errors, warnings = {}, {}, {}
            for index, camera in enumerate(self.camera_info):
                serial = camera['serial']
                if not packet['present'][index]:
                    errors[serial] = f'{serial}: missing preview frame'
                    continue
                offset = (index * 2 + packet['preview'] % 2) * SLOT_BYTES
                header = struct.unpack_from('<8Q', self.memory, offset)
                if header[0] & 1 or header[1] != packet['preview'] or header[2] != packet['sequence']:
                    errors[serial] = f'{serial}: preview superseded'
                    continue
                pixels = np.frombuffer(self.memory, dtype=np.uint8, count=WIDTH * HEIGHT, offset=offset + 64).copy()
                if struct.unpack_from('<Q', self.memory, offset)[0] != header[0]:
                    errors[serial] = f'{serial}: preview changed while reading'
                    continue
                if header[6:8] != (WIDTH, HEIGHT):
                    raise ValueError('Native preview geometry changed')
                # Long Bayer names make the RGGB phase explicit; short OpenCV
                # Bayer aliases are easy to confuse with the output channel order.
                rgb = cv2.cvtColor(pixels.reshape(HEIGHT, WIDTH), cv2.COLOR_BayerRGGB2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                frames[serial] = Frame(serial, gray, header[5], header[4], camera['exposure_us'],
                                       camera['gain_db'], camera['settings'], display_image=rgb)
                ptp = packet.get('ptp', {}).get(serial, {})
                if not self.simulated and (ptp.get('status') != 'Slave' or abs(ptp.get('offset_ns', 10**9)) > 1000):
                    warnings['ptp:' + serial] = f"{serial}: PTP {ptp.get('status')}, offset {ptp.get('offset_ns')} ns"
            return FrameSet(packet['sequence'], packet['scheduled_ns'], self.serials, frames, errors,
                            simulated=self.simulated, timing_warnings=warnings)

    def _send(self, command):
        with self.command_lock:
            if not self.process or self.process.poll() is not None:
                raise RuntimeError('Recorder is not connected')
            self.process.stdin.write(json.dumps(command, allow_nan=False) + '\n')
            self.process.stdin.flush()

    def recording_status(self):
        with self.lock:
            status = dict(self.state)
            if self.command_error:
                status.update(state='failed', error=self.command_error)
            return status

    def recording_busy(self):
        return self.recording_status()['state'] in BUSY_STATES

    def start_recording(self, root, fps=60, seconds=120, quality=85, calibration=None, session=None):
        if not 1 <= fps <= 60 or not 1 <= seconds <= 120 or not 50 <= quality <= 95:
            raise ValueError('Use 1–60 fps, 1–120 seconds and JPEG quality 50–95')
        if self.recording_busy() or (self.prepare_thread and self.prepare_thread.is_alive()):
            raise ValueError('A take is already active')
        calibration_text = json.dumps(calibration, indent=2, allow_nan=False) if calibration else None
        directory = Path(root).expanduser().resolve() / ('take-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        with self.lock:
            self.command_error = None
            self.start_pending = True
            self.record_request_sent = self.cancel_requested = False
            self.state = dict(state='preparing', directory=str(directory), fps=fps, planned_frames=fps * seconds)

        def prepare():
            try:
                directory.mkdir(parents=True, exist_ok=False)
                context = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                               simulated=self.simulated, calibration_session=str(session) if session else None,
                               camera_serials=list(self.serials), calibration_file=None)
                if calibration_text:
                    write_json_durable(directory / 'calibration.json', json.loads(calibration_text))
                    context['calibration_file'] = 'calibration.json'
                    context['calibration_sha256'] = hashlib.sha256((directory / 'calibration.json').read_bytes()).hexdigest()
                write_json_durable(directory / 'context.json', context)
                with self.lock:
                    cancelled = self.cancel_requested or self.closing
                    if cancelled:
                        self.start_pending = False
                        self.state.update(state='cancelled')
                        cancelled_state = dict(self.state, complete=False)
                if cancelled:
                    write_json_durable(directory / 'take.json', cancelled_state)
                    return
                self._send(dict(command='record', directory=str(directory), fps=fps, frames=fps * seconds, quality=quality))
                with self.lock:
                    self.record_request_sent = True
                    cancelled = self.cancel_requested or self.closing
                if cancelled:
                    self._send({'command': 'stop'})
            except Exception as exc:
                with self.lock:
                    self.start_pending = False
                    self.command_error = str(exc)
                    self.state.update(state='failed', error=str(exc))
        self.prepare_thread = Thread(target=prepare, name='prepare-recording', daemon=True)
        self.prepare_thread.start()
        return directory

    def stop_recording(self):
        with self.lock:
            self.cancel_requested = True
            sent = self.record_request_sent
        if sent:
            self._send({'command': 'stop'})

    def close(self):
        self.closing = True
        if self.prepare_thread:
            self.prepare_thread.join(timeout=5)
        failure = None
        if self.process:
            if self.process.poll() is None:
                try:
                    self._send({'command': 'shutdown'})
                except (OSError, RuntimeError):
                    pass
                try:
                    self.process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                    failure = 'Recorder shutdown timed out; camera settings restoration is unconfirmed'
            if self.reader:
                self.reader.join(timeout=3)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.memory:
            self.memory.close()
            self.memory = None
        if self.runtime:
            self.runtime.cleanup()
            self.runtime = None
        if self.log:
            self.log.close()
        if failure or self.restore_error:
            raise RuntimeError(failure or self.restore_error)
