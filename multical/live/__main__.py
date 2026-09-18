import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Standalone live multi-camera calibration and visual guidance')
    parser.add_argument('--boards', default=str(Path(__file__).with_name('default_board.yaml')))
    parser.add_argument('--output', default='live-sessions')
    parser.add_argument('--seed', help='Converted Captury, multical, or live calibration JSON (metres, native images)')
    parser.add_argument('--demo', action='store_true', help='Start an explicitly labelled rendered test rig')
    parser.add_argument('--count', type=int, help='Required number of cameras (25 hardware, 4 simulated by default)')
    parser.add_argument('--serials', nargs='*', default=[], help='Explicit expected hardware serials')
    parser.add_argument('--fps', type=float, default=5., help='Scheduled capture frequency (default 5 Hz)')
    parser.add_argument('--workers', type=int, default=4, help='Detection worker count')
    parser.add_argument('--sdk-path', help='Directory containing the Spinnaker PySpin module')
    parser.add_argument('--exposure-us', type=float, help='Apply one exposure to all cameras for this session')
    parser.add_argument('--gain-db', type=float, help='Apply one gain to all cameras for this session')
    parser.add_argument('--autostart', action='store_true', help='Connect hardware immediately rather than with the UI button')
    parser.add_argument('--auto-capture', action='store_true', help='Automatically retain novel stationary training poses')
    parser.add_argument('--screenshot', help='Save an offscreen/onscreen UI screenshot after --run-seconds')
    parser.add_argument('--run-seconds', type=float, help='Close after this duration (for UI smoke tests)')
    args = parser.parse_args()
    args.count = args.count if args.count is not None else (4 if args.demo else 25)
    if not 1 <= args.count <= 64 or not .2 <= args.fps <= 20 or not 1 <= args.workers <= 32:
        parser.error('count must be 1–64, fps 0.2–20, workers 1–32')
    if args.serials and (len(args.serials) != args.count or len(set(args.serials)) != args.count):
        parser.error('--serials must contain exactly --count distinct serials')
    if args.run_seconds is not None and args.run_seconds <= 0:
        parser.error('--run-seconds must be positive')
    import cv2
    cv2.setNumThreads(1)
    # opencv-python's bundled Qt plugin is incompatible with the GUI's Qt build.
    # Let the Qt binding select its own plugins, only undo OpenCV's injected path.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(key, ''):
            os.environ.pop(key)
    from qtpy import QtCore, QtWidgets
    from .ui import LiveWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        window = LiveWindow(args)
    except Exception as exc:
        parser.exit(1, f'Cannot start live calibration: {exc}\n')
    window.show()
    if args.run_seconds:
        def finish():
            if args.screenshot:
                window.grab().save(args.screenshot)
            window.close()
        QtCore.QTimer.singleShot(int(args.run_seconds * 1000), finish)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
