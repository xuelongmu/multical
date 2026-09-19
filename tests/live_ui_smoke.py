"""Drive the actual UI through live collection and a background calibration.

Run with QT_QPA_PLATFORM=offscreen. No hardware is accessed.
"""
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    cv2.setNumThreads(1)
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(key, ''):
            os.environ.pop(key)
    from qtpy import QtCore, QtWidgets
    from multical.live.ui import LiveWindow
    output = tempfile.mkdtemp(prefix='multical-live-ui-')
    args = SimpleNamespace(boards=str(Path(__file__).resolve().parents[1] / 'example_boards/charuco_36x54.yaml'),
                           demo=True, count=4, serials=[], auto_capture=True, autostart=False,
                           fps=8., workers=4, sdk_path=None, output=output, exposure_us=None, gain_db=None)
    app = QtWidgets.QApplication([])
    window = LiveWindow(args)
    window.show()
    start = time.monotonic()
    state = dict(phase='training', solve_frame=None, errors=[])
    def tick():
        try:
            if time.monotonic() - start > 130:
                raise AssertionError('UI workflow timed out')
            engine = window.engine
            if engine is None or engine.session is None:
                return
            if engine.error:
                raise AssertionError(engine.error)
            records = list(engine.session.samples)
            if state['phase'] == 'training' and sum(r['role'] == 'training' for r in records) >= 24:
                window.auto.setChecked(False)
                window.solve()
                window.cancel_solve()
                assert window.process is None and engine.running(), 'Cancelling a solve interrupted acquisition'
                state['phase'] = 'validation'
                print('UI: 24 training captures retained; cancellation preserved acquisition', flush=True)
            if state['phase'] == 'validation':
                if sum(r['role'] == 'validation' for r in records) < 4:
                    if engine.pending_capture is None:
                        window.capture('validation')
                else:
                    state['phase'] = 'solving'
                    state['solve_frame'] = engine.latest_batch.sequence
                    window.solve()
                    print('UI: validation retained; background solve started', flush=True)
            if state['phase'] == 'solving':
                if window.error_label.isVisible():
                    raise AssertionError(window.error_label.text())
                if window.result and window.process is None:
                    assert window.result['solver']['success'], window.result['solver']
                    assert engine.latest_batch.sequence > state['solve_frame'], 'Preview stopped during calibration'
                    assert len(window.result['validation_ids']) == 4
                    window.grab().save('/tmp/multical-live-calibrated.png')
                    print(f'UI PASS: saved calibrated screenshot and session in {output}', flush=True)
                    state['original_session'] = engine.session.directory
                    state['original_count'] = len(records)
                    state['saved_calibration'] = str(Path(output) / 'export.json')
                    with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(state['saved_calibration'], 'JSON')):
                        window.save_calibration()
                    assert Path(state['saved_calibration']).is_file()
                    window.auto.setChecked(True)
                    window.pause_button.setChecked(True)
                    state['pause_frame'] = engine.latest_batch.sequence
                    state['paused_at'] = time.monotonic()
                    state['phase'] = 'paused'
            if state['phase'] == 'paused' and time.monotonic() - state['paused_at'] > 1.5:
                assert len(engine.session.samples) == state['original_count'], 'Pause saved an unexpected capture'
                assert engine.latest_batch.sequence > state['pause_frame'], 'Pause stopped preview'
                window.auto.setChecked(False)
                window.new_session()
                state['phase'] = 'new_session'
            if state['phase'] == 'new_session' and engine.session and engine.session.directory != state['original_session']:
                assert len(engine.session.samples) == 0
                with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(state['original_session'])):
                    window.open_session()
                state['phase'] = 'resume_session'
            if state['phase'] == 'resume_session' and engine.session and engine.session.directory == state['original_session'] and engine.running():
                assert len(engine.session.samples) == state['original_count']
                assert any(engine.coverage.views.values())
                with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(state['saved_calibration'], 'JSON')):
                    window.choose_seed()
                state['phase'] = 'loaded_calibration'
            if state['phase'] == 'loaded_calibration' and window.seed_checked and engine.session and engine.session.directory != state['original_session']:
                assert window.seed is not None and window.result is not None
                print('UI PASS: pause preserved previews; saved calibration; new session; resumed coverage; loaded calibration', flush=True)
                timer.stop()
                window.close()
        except Exception as exc:
            state['errors'].append(str(exc))
            print('UI FAIL:', exc, flush=True)
            timer.stop()
            window.close()
    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(200)
    app.exec_()
    if state['errors']:
        raise AssertionError('; '.join(state['errors']))


if __name__ == '__main__':
    main()
