"""Immutable, independently labelled captures and atomic session persistence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import shutil
import uuid

import cv2
import numpy as np


class Session:
    def __init__(self, output, board_file, serials, simulated=False, capture_mode='stationary', resume=None):
        if capture_mode not in ('stationary', 'motion'):
            raise ValueError('Capture mode must be stationary or motion')
        self.capture_mode = capture_mode
        if resume:
            self.directory = Path(resume)
            manifest = json.loads((self.directory / 'manifest.json').read_text())
            if (manifest.get('schema_version') != 1 or
                manifest['camera_serials'] != list(serials) or
                manifest['board_sha256'] != hashlib.sha256(Path(board_file).read_bytes()).hexdigest() or
                manifest['simulated'] != simulated or
                manifest.get('capture_mode', 'stationary') != capture_mode):
                raise ValueError('Resume session must match board, camera roster, source type and capture mode')
            self.serials = tuple(serials)
            self.manifest = manifest
            self.samples = list(manifest['captures'])
            for index, record in enumerate(self.samples):
                if record['id'] != f'{index:06d}' or record['role'] not in ('training', 'validation'):
                    raise ValueError('Invalid capture ordering or role in resumed session')
                for serial in serials:
                    if not (self.directory / record['id'] / f'{serial}.png').is_file():
                        raise ValueError(f"Missing saved image: {record['id']}/{serial}")
            if (self.directory / f'{len(self.samples):06d}').exists():
                raise ValueError('Uncommitted capture directory exists; inspect it before resuming')
            return
        self.directory = Path(output) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.serials = tuple(serials)
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in serials):
            raise ValueError("Camera serials must be safe file names")
        board_bytes = Path(board_file).read_bytes()
        (self.directory / 'board.yaml').write_bytes(board_bytes)
        self.manifest = dict(schema_version=1, simulated=simulated,
                             capture_mode=capture_mode,
                             camera_serials=list(serials), captures=[],
                             board_sha256=hashlib.sha256(board_bytes).hexdigest(),
                             versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__),
                             accuracy_status='unverified')
        self.samples = []
        self._write_manifest()

    def _write_manifest(self):
        temporary = self.directory / 'manifest.json.tmp'
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False))
        os.replace(temporary, self.directory / 'manifest.json')

    def add(self, packet, role):
        if role not in ('training', 'validation'):
            raise ValueError('Unknown capture role')
        batch = packet['batch']
        problems = batch.problems(capture_mode=self.capture_mode)
        if problems:
            raise ValueError('; '.join(problems))
        if batch.serials != self.serials:
            raise ValueError('Camera roster changed')
        if self.samples:
            for serial in self.serials:
                if batch.frames[serial].metadata()['image_size'] != self.samples[0]['frames'][serial]['image_size']:
                    raise ValueError(f'{serial}: image geometry changed since the first capture')
        if any(r['sequence'] == batch.sequence for r in self.samples):
            raise ValueError('This frame set has already been retained')
        capture_id = f'{len(self.samples):06d}'
        staging = self.directory / ('.pending-' + capture_id)
        final = self.directory / capture_id
        staging.mkdir()
        record = dict(id=capture_id, role=role, sequence=batch.sequence,
                      scheduled_ns=batch.scheduled_ns, frames={}, detections={},
                      capture_mode=self.capture_mode, timing_warnings=batch.timing_issues())
        try:
            for serial in self.serials:
                frame = batch.frames[serial]
                detection = packet['observations'][serial]
                if not cv2.imwrite(str(staging / f'{serial}.png'), frame.image):
                    raise OSError(f'Could not save image for {serial}')
                record['frames'][serial] = frame.metadata()
                record['detections'][serial] = dict(ids=detection['ids'].tolist(),
                                                  corners=detection['corners'].tolist(),
                                                  usable=bool(detection['usable']))
            (staging / 'capture.json').write_text(json.dumps(record, indent=2, allow_nan=False))
            staging.rename(final)
            self.manifest['captures'].append(record)
            try:
                self._write_manifest()
            except Exception:
                self.manifest['captures'].pop()
                raise
            # Memory retains observations, not the full-resolution images.
            self.samples.append(record)
            return record
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def write_result(directory, result):
    """Versioned result; snapshot IDs make later captures distinguishable."""
    directory = Path(directory)
    name = 'calibration-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:4] + '.json'
    temporary = directory / (name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    os.replace(temporary, directory / name)
    return directory / name
