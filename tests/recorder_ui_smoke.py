"""Actual Qt recording controls with native simulated cameras. Requires CUDA; no hardware cameras."""
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    cv2.setNumThreads(1)
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(key, ''):
            os.environ.pop(key)
    from qtpy import QtCore, QtWidgets
    from multical.live.ui import LiveWindow
    from multical.live.recording_verify import verify_take
    output = Path(tempfile.mkdtemp(prefix='multical-recorder-ui-'))
    args = SimpleNamespace(boards=str(Path(__file__).resolve().parents[1] / 'example_boards/charuco_36x54.yaml'),
                           demo=False, count=3, serials=[], auto_capture=False, autostart=False,
                           fps=5., workers=2, sdk_path=None, output=str(output / 'sessions'),
                           recordings=str(output / 'takes'), exposure_us=None, gain_db=None)
    app = QtWidgets.QApplication([])
    preferences = QtCore.QSettings(str(output / 'preferences.ini'), QtCore.QSettings.IniFormat)
    with patch.object(QtCore, 'QSettings', return_value=preferences):
        window = LiveWindow(args)
    window.sounds.setChecked(False)
    window.source_choice.setCurrentIndex(3)
    window.record_seconds.setValue(3)
    window.show()
    window.toggle_capture()
    state = dict(phase='connecting', errors=[], takes=[], last_tick=time.monotonic(), max_tick=0)
    start = time.monotonic()
    def tick():
        now = time.monotonic()
        state['max_tick'] = max(state['max_tick'], now-state['last_tick'])
        state['last_tick'] = now
        try:
            if now-start > 60:
                raise AssertionError('UI recorder workflow timed out')
            engine = window.engine
            if not engine or not engine.session:
                return
            assert not engine.error, engine.error
            status = engine.source.recording_status()
            phase = state['phase']
            if phase == 'connecting' and window.record_button.isEnabled():
                state['pid'] = engine.source.process.pid
                window.record_button.click()
                assert window.previews_paused
                state['phase'] = 'first'
            elif phase == 'first' and status['state'] == 'saved':
                window.refresh()
                assert not window.previews_paused
                assert window.training_button.isEnabled()
                state['takes'].append(status['directory'])
                assert len(engine.session.samples) == 0
                window.record_seconds.setValue(10)
                window.record_button.click()
                state['phase'] = 'second'
            elif phase == 'second' and status['state'] == 'recording' and status['scheduled_frames'] > 60:
                assert not window.solve_button.isEnabled()
                window.solve()  # Keyboard/direct method must also respect the guard.
                assert window.process is None
                window.record_button.click()
                state['phase'] = 'stopping'
            elif phase == 'stopping' and status['state'] == 'saved':
                window.refresh()
                assert status['stop_reason'] == 'operator'
                assert status['scheduled_frames'] < 600
                assert engine.source.process.pid == state['pid'], 'Recorder reconnected between takes'
                assert not window.previews_paused
                state['takes'].append(status['directory'])
                window.grab().save('/tmp/multical-recorder-ui.png')
                timer.stop()
                window.close()
            elif status['state'] == 'failed':
                raise AssertionError(status)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            state['errors'].append(str(exc))
            timer.stop()
            window.close()
    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    app.exec_()
    assert not state['errors'], state['errors']
    assert len(state['takes']) == 2, state
    assert state['max_tick'] < 2, state['max_tick']
    for take in state['takes']:
        print(verify_take(take), flush=True)
    image = cv2.VideoCapture(str(Path(state['takes'][0]) / 'cameras/SIM-01/segment_00000.mkv'))
    ok, pixels = image.read()
    image.release()
    assert ok
    # OpenCV gives BGR: check the native RGGB -> RGB -> JPEG path preserves colour.
    assert [int(pixels[500, x].argmax()) for x in (200, 700, 1200)] == [2, 1, 0]
    print(f'UI PASS: complete take, early stop, preview restore, same service, correct colour; max event-loop gap {state["max_tick"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()
