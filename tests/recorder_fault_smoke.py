"""Native GPU fault-injection tests with synthetic cameras only."""
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
cv2.setNumThreads(1)
from multical.live.recorder import NativeSource
from multical.live.recording_verify import verify_take


def wait(source, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = source.recording_status()
        if predicate(state):
            return state
        time.sleep(.05)
    raise AssertionError(source.recording_status())


def main():
    root = Path(tempfile.mkdtemp(prefix='multical-recorder-faults-'))
    for config, error in [({'test_drop_camera': 0, 'test_drop_frame': 20}, 'Missing or repeated'),
                          ({'test_fail_write_after': 1_000_000}, 'storage failure')]:
        source = NativeSource(expected_count=3, simulated=True, test_config=config)
        try:
            source.open()
            take = source.start_recording(root, seconds=3)
            state = wait(source, lambda s: s['state'] in ('saved', 'failed'))
            assert state['state'] == 'failed' and error in state['error'], state
            manifest = json.loads((take / 'take.json').read_text())
            assert manifest['complete'] is False, manifest
            try:
                verify_take(take)
            except ValueError:
                pass
            else:
                raise AssertionError('Verifier accepted failed take')
            assert len(source.read().frames) == 3, 'Preview did not recover after take fault'
            print('PASS:', error, take, flush=True)
        finally:
            source.close()
    source = NativeSource(expected_count=3, simulated=True)
    try:
        source.open()
        take = source.start_recording(root, seconds=10)
        wait(source, lambda s: s.get('scheduled_frames', 0) > 50)
    finally:
        source.close()
    manifest = json.loads((take / 'take.json').read_text())
    assert manifest['complete'] and manifest['stop_reason'] == 'shutdown', manifest
    print('PASS: active shutdown', verify_take(take), flush=True)


if __name__ == '__main__':
    main()
