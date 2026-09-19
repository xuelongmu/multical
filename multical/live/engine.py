"""Bounded acquisition/analysis mailboxes keep the UI and capture independent."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
import time

from .metrics import Coverage, observe
from .guidance import capture_progress
from .session import Session


class LiveEngine:
    def __init__(self, source, board, board_file, output, workers=4, capture_mode='stationary', resume=None):
        if capture_mode not in ('stationary', 'motion'):
            raise ValueError('Capture mode must be stationary or motion')
        self.capture_mode = capture_mode
        self.resume = resume
        self.source, self.board = source, board
        self.board_file, self.output = board_file, output
        self.workers = workers
        self.lock = Lock()
        self.stop_event = Event()
        self.source.stop_event = self.stop_event
        self.new_frame = Event()
        self.latest_batch = self.packet = self.coverage = self.session = None
        self.validation_coverage = None
        self.status = 'Connecting cameras…'
        self.error = None
        self.pending_capture = None
        self.capture_event = None
        self.auto_capture = False
        self.auto_capture_role = 'training'
        self.paused = False
        self.previous = None
        self.captured_sequence = -1
        self.threads = []

    def start(self):
        self.threads = [Thread(target=self._acquire, daemon=True, name='live-acquisition'),
                        Thread(target=self._analyze, daemon=True, name='live-detection')]
        for thread in self.threads:
            thread.start()

    def _acquire(self):
        try:
            serials = self.source.open()
            coverage = Coverage(serials)
            session = Session(self.output, self.board_file, serials,
                              simulated=self.source.__class__.__name__ == 'SimulatedSource' or getattr(self.source, 'simulated', False) is True,
                              capture_mode=self.capture_mode, resume=self.resume)
            validation_coverage = Coverage(serials)
            if session.samples:
                # Rebuild diversity/partition state from saved images using the same detector.
                import cv2
                from types import SimpleNamespace
                self.status = 'Restoring saved capture coverage…'
                for record in session.samples:
                    if self.stop_event.is_set():
                        return
                    observations = {}
                    for serial in serials:
                        image = cv2.imread(str(session.directory / record['id'] / f'{serial}.png'), cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            raise ValueError(f"Cannot read saved capture {record['id']}/{serial}")
                        observations[serial] = observe(self.board, SimpleNamespace(image=image))
                    (coverage if record['role'] == 'training' else validation_coverage).add(observations)
            sequence_offset = max((r['sequence'] for r in session.samples), default=-1) + 1
            with self.lock:
                self.coverage, self.session = coverage, session
                self.validation_coverage = validation_coverage
                self.status = f'{len(serials)} cameras connected'
            while not self.stop_event.is_set():
                batch = self.source.read()
                batch.sequence += sequence_offset
                with self.lock:
                    self.latest_batch = batch
                self.new_frame.set()
        except Exception as exc:
            with self.lock:
                if not self.stop_event.is_set():
                    self.error = str(exc)
                self.status = 'Acquisition stopped'
            self.stop_event.set()
        finally:
            try:
                self.source.close()
            except Exception as exc:
                with self.lock:
                    self.error = (self.error + '; ' if self.error else '') + str(exc)
            self.stop_event.set()
            self.new_frame.set()

    def _analyze(self):
        sequence = -1
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while not self.stop_event.is_set():
                self.new_frame.wait(.2)
                self.new_frame.clear()
                with self.lock:
                    batch = self.latest_batch
                if batch is None or batch.sequence == sequence:
                    continue
                sequence = batch.sequence
                if self.recording_busy():
                    self.previous = None
                    continue
                try:
                    jobs = {s: pool.submit(observe, self.board, f) for s, f in batch.frames.items()}
                    observations = {s: future.result() for s, future in jobs.items()}
                    problems = batch.problems(capture_mode=self.capture_mode)
                    now = time.monotonic()
                    packet = dict(batch=batch, observations=observations, analyzed_at=now,
                                  problems=problems)
                    usable = any(o['usable'] for o in observations.values())
                    # Demand consecutive stationary observations for automatic retention.
                    stationary = False
                    if self.previous is not None:
                        movements = []
                        import numpy as np
                        for serial, observation in observations.items():
                            old = self.previous['observations'].get(serial)
                            if old is None or not observation['usable']:
                                continue
                            common, a, b = np.intersect1d(old['ids'], observation['ids'], return_indices=True)
                            if len(common) >= 6:
                                movements.append(float(np.median(np.linalg.norm(old['corners'][a] - observation['corners'][b], axis=1))))
                        stationary = bool(movements) and max(movements) < 1.0
                    with self.lock:
                        role = requested_role = self.pending_capture
                        if role is None:
                            role = self.automatic_role(observations, stationary)
                    if self.recording_busy():
                        self.previous = None
                        continue
                    partition_conflict = (role == 'validation' and self.coverage.matches_pose(observations)) or \
                                         (role == 'training' and self.validation_coverage.matches_pose(observations))
                    if role and not self.paused and usable and not problems and not partition_conflict and batch.sequence != self.captured_sequence:
                        record = self.session.add(packet, role)
                        milestone = False
                        if role == 'training':
                            before = len(capture_progress(self.coverage.snapshot(), {})['groups'])
                            self.coverage.add(observations)
                            milestone = len(capture_progress(self.coverage.snapshot(), {})['groups']) < before
                        else:
                            self.validation_coverage.add(observations)
                        self.captured_sequence = batch.sequence
                        with self.lock:
                            if requested_role == self.pending_capture:
                                self.pending_capture = None
                            self.status = f"Saved {role} pose {record['id']}"
                            self.capture_event = dict(id=record['id'], role=role, session=str(self.session.directory), milestone=milestone)
                    guiding = self.validation_coverage if self.auto_capture and self.auto_capture_role == 'validation' else self.coverage
                    packet['guidance'] = guiding.guidance(observations, problems)
                    visible_count = sum(bool(o['usable']) for o in observations.values())
                    packet['guidance_cue'] = None
                    if not problems:
                        if visible_count == 1 and len(self.coverage.serials) > 1:
                            packet['guidance_cue'] = 'single'
                        elif visible_count and not stationary:
                            packet['guidance_cue'] = 'hold'
                        elif visible_count and not any(guiding.novel(s, o) for s, o in observations.items() if o['usable']):
                            packet['guidance_cue'] = 'tilt'
                    if partition_conflict:
                        packet['guidance'] = 'Move to a different board pose. Training and validation poses are kept separate.'
                    packet['coverage'] = self.coverage.snapshot()
                    packet['stationary'] = stationary
                    packet['validation_views'] = self.validation_coverage.views.copy()
                    packet['novel'] = {s: self.coverage.novel(s, o) for s, o in observations.items()}
                    self.previous = packet
                    with self.lock:
                        self.packet = packet
                except Exception as exc:
                    with self.lock:
                        self.error = f'Detection/capture failed: {exc}'
                    self.stop_event.set()

    def automatic_role(self, observations, stationary):
        if not self.auto_capture or self.paused or not stationary or self.recording_busy():
            return None
        role = self.auto_capture_role
        if role not in ('training', 'validation'):
            return None
        if role == 'validation' and sum(bool(o['usable']) for o in observations.values()) < 2:
            return None
        coverage = self.coverage if role == 'training' else self.validation_coverage
        other = self.validation_coverage if role == 'training' else self.coverage
        if other.matches_pose(observations):
            return None
        return role if any(coverage.novel(s, o) for s, o in observations.items() if o['usable']) else None

    def set_paused(self, paused):
        with self.lock:
            self.paused = bool(paused)
            if paused:
                self.pending_capture = None

    def capture(self, role):
        with self.lock:
            if self.recording_busy():
                raise ValueError('Pose collection is suspended during video recording')
            if self.paused:
                raise ValueError('Capture is paused; resume before saving a pose')
            if self.pending_capture:
                raise ValueError('A capture is already waiting for a complete usable frame set')
            self.pending_capture = role

    def snapshot(self):
        with self.lock:
            return dict(batch=self.latest_batch, packet=self.packet, status=self.status,
                        error=self.error, pending=self.pending_capture, paused=self.paused,
                        session=self.session, capture_event=self.capture_event)

    def recording_busy(self):
        return bool(getattr(self.source, 'recording_busy', lambda: False)())

    def stop(self):
        self.stop_event.set()
        self.new_frame.set()

    def running(self):
        return any(t.is_alive() for t in self.threads)
