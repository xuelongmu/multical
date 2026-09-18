"""Native live visualizer. All hardware and fitting work stays off the GUI thread."""
import math
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from multical.board import load_config
from .calibration import solve_process
from .engine import LiveEngine
from .metrics import GRID, live_projection
from .session import write_result
from .sources import PySpinSource, SimulatedSource


ACCENT = '#52dfbc'
STYLE = '''
QMainWindow, QWidget { background: #10171f; color: #e4edf3; font-family: sans-serif; font-size: 13px; }
QFrame#panel { background: #18222d; border: 1px solid #2a3947; border-radius: 9px; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#muted { color: #9cabb9; }
QLabel#guidance { background: #183630; color: #9cf0d9; border: 1px solid #285648; border-radius: 8px; padding: 14px; font-size: 15px; }
QLabel#error { color: #ffb2a5; background: #3b2528; border-radius: 6px; padding: 10px; }
QPushButton { background: #263746; border: 1px solid #3b5163; border-radius: 6px; padding: 9px 12px; }
QPushButton:hover { background: #344b5e; }
QPushButton:disabled { color: #65798a; background: #1b2731; border-color: #263542; }
QPushButton#primary { background: #52dfbc; color: #0c2521; font-weight: 600; border: none; }
QLineEdit, QSpinBox, QComboBox { background: #101922; border: 1px solid #384957; border-radius: 5px; padding: 7px; }
QCheckBox { spacing: 9px; padding: 5px 0; }
QTableWidget { background: #141f29; alternate-background-color: #1a2733; border: none; gridline-color: #283947; }
QHeaderView::section { background: #233240; color: #b7c6d2; padding: 7px; border: none; }
QScrollArea { border: none; }
QSplitter::handle { background: #25333f; }
QTabBar::tab { background: #1d2b37; padding: 8px 16px; }
QTabBar::tab:selected { color: #52dfbc; background: #293b49; }
QTabWidget::pane { border: 1px solid #293b49; }
QToolTip { background: #243848; color: white; border: 1px solid #4d6476; }
'''


def label(text, name=None):
    item = QtWidgets.QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(True)
    return item


def as_image(pixels, width=900):
    h, w = pixels.shape[:2]
    if w > width:
        pixels = cv2.resize(pixels, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    pixels = np.ascontiguousarray(pixels)
    return QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QtGui.QImage.Format_Grayscale8).copy()


class CameraTile(QtWidgets.QWidget):
    selected = QtCore.Signal(str)

    def __init__(self, serial):
        super().__init__()
        self.serial, self.image = serial, None
        self.detail = 'Waiting for frames'
        self.active = False
        self.missing = False
        self.setMinimumSize(175, 155)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.selected.emit(self.serial)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#18232d'))
        if self.image is not None and not self.missing:
            available = QtCore.QRect(4, 28, self.width()-8, self.height()-54)
            size = self.image.size().scaled(available.size(), QtCore.Qt.KeepAspectRatio)
            target = QtCore.QRect(QtCore.QPoint(0, 0), size)
            target.moveCenter(available.center())
            painter.drawImage(target, self.image)
        painter.setPen(QtGui.QColor('#e4edf3'))
        painter.drawText(10, 20, self.serial)
        painter.setPen(QtGui.QColor('#ffb2a5' if self.missing else '#9cabb9'))
        painter.drawText(10, self.height()-9, self.detail)
        if self.active:
            painter.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 2))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)


class InspectionView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.observation = None
        self.prediction = None
        self.cells = None
        self.show_coverage = True
        self.zoom = 1.
        self.pan = QtCore.QPointF(0, 0)
        self.drag = None
        self.setMinimumSize(400, 280)

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * math.exp(event.angleDelta().y() / 800), 1, 8))
        if self.zoom == 1:
            self.pan = QtCore.QPointF(0, 0)
        self.update()

    def mousePressEvent(self, event):
        self.drag = event.pos()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            self.pan += QtCore.QPointF(event.pos() - self.drag)
            self.drag = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseDoubleClickEvent(self, event):
        self.zoom = 1.
        self.pan = QtCore.QPointF(0, 0)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#0b1118'))
        if self.image is None:
            p.setPen(QtGui.QColor('#9cabb9'))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, 'Select a camera to inspect board detection')
            return
        size = self.image.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        size = QtCore.QSize(int(size.width()*self.zoom), int(size.height()*self.zoom))
        rect = QtCore.QRect(QtCore.QPoint(0, 0), size)
        rect.moveCenter(self.rect().center())
        rect.translate(self.pan.toPoint())
        p.drawImage(rect, self.image)
        if self.show_coverage and self.cells is not None:
            for y in range(GRID[1]):
                for x in range(GRID[0]):
                    cell = QtCore.QRectF(rect.x()+x*rect.width()/GRID[0], rect.y()+y*rect.height()/GRID[1],
                                         rect.width()/GRID[0], rect.height()/GRID[1])
                    if self.cells[y, x]:
                        p.fillRect(cell, QtGui.QColor(40, 185, 158, min(100, 25 + 15*int(self.cells[y, x]))))
                    p.setPen(QtGui.QColor(140, 180, 200, 60))
                    p.drawRect(cell)
        if self.observation is None:
            return
        observation = self.observation
        w, h = observation['image_size']
        def pixel(point):
            return QtCore.QPointF(rect.x() + point[0] / w * rect.width(), rect.y() + point[1] / h * rect.height())
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        for index, (point, identity) in enumerate(zip(observation['corners'], observation['ids'])):
            at = pixel(point)
            p.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 2))
            p.drawEllipse(at, 3, 3)
            p.drawText(at + QtCore.QPointF(4, -4), str(identity))
            if self.prediction is not None:
                predicted = pixel(self.prediction['points'][index])
                p.setPen(QtGui.QPen(QtGui.QColor('#ffb45f'), 1.5))
                p.drawLine(at, predicted)
                p.drawLine(predicted+QtCore.QPointF(-3, 0), predicted+QtCore.QPointF(3, 0))
                p.drawLine(predicted+QtCore.QPointF(0, -3), predicted+QtCore.QPointF(0, 3))


class RigView(QtWidgets.QWidget):
    """Dependency-free orbitable projection of calibrated camera centres and poses."""
    def __init__(self):
        super().__init__()
        self.result = self.target = None
        self.yaw, self.pitch, self.zoom = -.55, .35, 1.0
        self.drag = None
        self.setMinimumSize(350, 200)

    def mousePressEvent(self, event):
        self.drag = event.pos()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            delta = event.pos() - self.drag
            self.yaw += delta.x() * .008
            self.pitch = np.clip(self.pitch + delta.y() * .008, -1.4, 1.4)
            self.drag = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def wheelEvent(self, event):
        self.zoom = np.clip(self.zoom * math.exp(event.angleDelta().y() / 1200), .3, 4.)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#111c26'))
        p.setPen(QtGui.QColor('#9cabb9'))
        if self.result is None:
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, 'Camera geometry appears after calibration\nCapture shared poses across the rig')
            return
        poses = {s: np.array(v) for s, v in self.result['camera_poses'].items()}
        centres = {s: -v[:3, :3].T @ v[:3, 3] for s, v in poses.items()}
        targets = [np.array(v)[:3, 3] for v in self.result['frame_poses'].values()]
        points = np.array(list(centres.values()) + targets)
        centre = (points.min(0) + points.max(0)) / 2
        span = max(float(np.ptp(points, axis=0).max()), .5)
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        rotation = np.array([[cy, 0, sy], [sp*sy, cp, -sp*cy], [-cp*sy, sp, cp*cy]])
        scale = min(self.width(), self.height()) * .7 / span * self.zoom
        def project(point):
            transformed = rotation @ (point-centre)
            return QtCore.QPointF(self.width()/2 + transformed[0]*scale, self.height()/2 + transformed[1]*scale)
        p.setPen(QtGui.QPen(QtGui.QColor('#2a414f'), 1))
        for a in np.linspace(-span, span, 9):
            p.drawLine(project(np.array([a, 0, -span])+centre), project(np.array([a, 0, span])+centre))
            p.drawLine(project(np.array([-span, 0, a])+centre), project(np.array([span, 0, a])+centre))
        for target in targets:
            p.setPen(QtGui.QColor('#5d90ac'))
            p.drawEllipse(project(target), 2, 2)
        for serial, pos in centres.items():
            p.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 2))
            at = project(pos)
            direction = poses[serial][:3, :3].T @ np.array([0, 0, span*.12])
            p.drawLine(at, project(pos+direction))
            p.setBrush(QtGui.QColor(ACCENT))
            p.drawEllipse(at, 4, 4)
            p.drawText(at+QtCore.QPointF(6, -6), serial)
        if self.target is not None:
            world = self.target['world_target']
            p.setPen(QtGui.QPen(QtGui.QColor('#ffb45f'), 3))
            pos = world[:3, 3]
            p.drawEllipse(project(pos), 7, 7)
            p.drawLine(project(pos), project(pos + world[:3, 2] * span*.15))
        p.setPen(QtGui.QColor('#9cabb9'))
        p.drawText(12, 20, f'Relative camera frame · extent {span:.2f} m · drag to orbit, scroll to zoom')


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.board_file = str(Path(args.boards).resolve())
        self.board = self._load_board(self.board_file)
        self.engine = self.result = self.process = self.connection = None
        self.last_packet = self.last_batch = None
        self.selected = None
        self.tiles = {}
        self.closing = False
        self.setWindowTitle('Multical Live — capture, inspect, calibrate')
        self.resize(1550, 1000)
        self.setStyleSheet(STYLE)
        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 14)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QVBoxLayout()
        title.addWidget(label('Multical / Live', 'title'))
        subtitle = label('Direct camera acquisition · board guidance · camera calibration', 'muted')
        subtitle.setMinimumWidth(650)
        title.addWidget(subtitle)
        header.addLayout(title)
        header.addStretch()
        self.mode_badge = label('DISCONNECTED', 'muted')
        self.mode_badge.setMinimumWidth(220)
        header.addWidget(self.mode_badge)
        outer.addLayout(header)
        body = QtWidgets.QHBoxLayout()
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName('panel')
        sidebar.setFixedWidth(290)
        controls = QtWidgets.QVBoxLayout(sidebar)
        controls.setContentsMargins(16, 16, 16, 16)
        controls.addWidget(label('ACQUISITION', 'muted'))
        self.source_choice = QtWidgets.QComboBox()
        self.source_choice.addItems(['FLIR cameras · PySpin', 'Simulated rig · testing'])
        self.source_choice.setCurrentIndex(1 if args.demo else 0)
        controls.addWidget(self.source_choice)
        controls.addWidget(label('Expected cameras', 'muted'))
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 64)
        self.count.setValue(args.count)
        controls.addWidget(self.count)
        controls.addWidget(label('Camera serials (optional, comma separated)', 'muted'))
        self.serials = QtWidgets.QLineEdit(','.join(args.serials))
        controls.addWidget(self.serials)
        controls.addWidget(label('Shared exposure (µs) / gain (dB)', 'muted'))
        exposure_row = QtWidgets.QHBoxLayout()
        self.exposure = QtWidgets.QLineEdit('' if args.exposure_us is None else str(args.exposure_us))
        self.gain = QtWidgets.QLineEdit('' if args.gain_db is None else str(args.gain_db))
        self.exposure.setPlaceholderText('Camera value')
        self.gain.setPlaceholderText('Camera value')
        exposure_row.addWidget(self.exposure)
        exposure_row.addWidget(self.gain)
        controls.addLayout(exposure_row)
        self.board_label = label(f'{Path(self.board_file).name}\n{self.board.num_points} corners · {self.board.square_length*1000:g} mm squares', 'muted')
        controls.addWidget(self.board_label)
        self.board_button = QtWidgets.QPushButton('Choose board…')
        self.board_button.clicked.connect(self.choose_board)
        controls.addWidget(self.board_button)
        self.start_button = QtWidgets.QPushButton('Connect cameras')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.toggle_capture)
        controls.addWidget(self.start_button)
        controls.addWidget(label('Close other camera applications before connecting.', 'muted'))
        controls.addSpacing(16)
        controls.addWidget(label('COLLECT POSES', 'muted'))
        self.training_button = QtWidgets.QPushButton('Capture training  [Space]')
        self.validation_button = QtWidgets.QPushButton('Capture validation  [V]')
        self.training_button.clicked.connect(lambda: self.capture('training'))
        self.validation_button.clicked.connect(lambda: self.capture('validation'))
        controls.addWidget(self.training_button)
        controls.addWidget(self.validation_button)
        self.auto = QtWidgets.QCheckBox('Auto-capture new poses')
        self.auto.setToolTip('Retain new training coverage after consecutive stationary detections. Validation is always manual.')
        self.auto.setChecked(args.auto_capture)
        self.auto.toggled.connect(self.set_auto)
        controls.addWidget(self.auto)
        self.count_label = label('0 training · 0 validation', 'muted')
        controls.addWidget(self.count_label)
        self.solve_button = QtWidgets.QPushButton('Calibrate captures  [C]')
        self.solve_button.clicked.connect(self.solve)
        controls.addWidget(self.solve_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel calibration')
        self.cancel_button.clicked.connect(self.cancel_solve)
        self.cancel_button.setVisible(False)
        controls.addWidget(self.cancel_button)
        self.solve_status = label('Collect varied poses in each camera, including shared views.', 'muted')
        controls.addWidget(self.solve_status)
        controls.addStretch()
        self.session_label = label('A new session is saved when acquisition starts.', 'muted')
        self.session_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.session_label)
        body.addWidget(sidebar)
        main = QtWidgets.QVBoxLayout()
        self.guidance = label('Connect cameras to begin. Use the simulated rig to exercise the full workflow.', 'guidance')
        main.addWidget(self.guidance)
        self.error_label = label('', 'error')
        self.error_label.hide()
        main.addWidget(self.error_label)
        upper = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        wall_box = QtWidgets.QWidget()
        wall_layout = QtWidgets.QVBoxLayout(wall_box)
        wall_layout.setContentsMargins(0, 0, 0, 0)
        self.wall_title = label('CAMERA WALL · live previews', 'muted')
        wall_layout.addWidget(self.wall_title)
        self.wall = QtWidgets.QWidget()
        self.wall_grid = QtWidgets.QGridLayout(self.wall)
        self.wall_grid.setContentsMargins(0, 0, 4, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.wall)
        wall_layout.addWidget(scroll)
        upper.addWidget(wall_box)
        detail = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 0, 0, 0)
        self.inspection_title = label('BOARD INSPECTION', 'muted')
        detail_layout.addWidget(self.inspection_title)
        self.inspection = InspectionView()
        detail_layout.addWidget(self.inspection, 1)
        self.inspection_info = label('Green: detected corners · amber: predicted corners from another camera', 'muted')
        detail_layout.addWidget(self.inspection_info)
        coverage_toggle = QtWidgets.QCheckBox('Show retained training coverage')
        coverage_toggle.setChecked(True)
        coverage_toggle.toggled.connect(self.toggle_coverage)
        detail_layout.addWidget(coverage_toggle)
        upper.addWidget(detail)
        upper.setSizes([650, 550])
        lower = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.rig = RigView()
        lower.addWidget(self.rig)
        self.tabs = QtWidgets.QTabWidget()
        self.metrics = QtWidgets.QTableWidget(0, 6)
        self.metrics.setHorizontalHeaderLabels(['Camera', 'Poses', 'Coverage', 'Train px', 'Test px', 'Test n'])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.metrics.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metrics.setAlternatingRowColors(True)
        self.metrics.verticalHeader().hide()
        self.overlaps = QtWidgets.QTableWidget()
        self.overlaps.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.metrics, 'Coverage & residuals')
        self.tabs.addTab(self.overlaps, 'Shared pose counts')
        lower.addWidget(self.tabs)
        lower.setSizes([500, 700])
        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical.addWidget(upper)
        vertical.addWidget(lower)
        vertical.setSizes([570, 250])
        main.addWidget(vertical, 1)
        body.addLayout(main, 1)
        outer.addLayout(body, 1)
        self.footer = label('Metric accuracy unverified · pixel residuals describe image agreement, not millimetre accuracy.', 'muted')
        outer.addWidget(self.footer)
        self.setCentralWidget(root)
        for key, callback in [('Space', lambda: self.capture('training')), ('V', lambda: self.capture('validation')), ('C', self.solve)]:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.refresh()
        if args.demo or args.autostart:
            QtCore.QTimer.singleShot(0, self.toggle_capture)

    @staticmethod
    def _load_board(filename):
        boards = load_config(filename)
        if len(boards) != 1:
            raise ValueError('Select a configuration with one ChArUco board')
        board = next(iter(boards.values()))
        if not hasattr(board, 'marker_length'):
            raise ValueError('Live capture currently requires a ChArUco board')
        return board

    def fail(self, message):
        self.error_label.setText(message)
        self.error_label.show()

    def choose_board(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Board configuration', str(Path(self.board_file).parent), 'YAML (*.yaml *.yml)')
        if filename:
            try:
                board = self._load_board(filename)
                self.board, self.board_file = board, filename
                self.board_label.setText(f'{Path(filename).name}\n{board.num_points} corners · {board.square_length*1000:g} mm squares')
            except Exception as exc:
                self.fail(str(exc))

    def toggle_capture(self):
        if self.engine and self.engine.running():
            self.engine.stop()
            self.start_button.setText('Disconnecting…')
            return
        self.error_label.hide()
        self.cancel_solve()
        self.result = self.rig.result = self.rig.target = None
        self.last_packet = self.last_batch = None
        for tile in self.tiles.values():
            self.wall_grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles = {}
        self.selected = None
        self.metrics.setRowCount(0)
        self.overlaps.setRowCount(0)
        self.inspection.image = None
        demo = self.source_choice.currentIndex() == 1
        if demo:
            source = SimulatedSource(self.board, self.count.value(), fps=self.args.fps)
        else:
            serials = tuple(s.strip() for s in self.serials.text().split(',') if s.strip())
            if serials and (len(serials) != self.count.value() or len(set(serials)) != len(serials)):
                self.fail('Enter one distinct serial for each expected camera.')
                return
            try:
                exposure = float(self.exposure.text()) if self.exposure.text().strip() else None
                gain = float(self.gain.text()) if self.gain.text().strip() else None
                if any(v is not None and not np.isfinite(v) for v in (exposure, gain)) or (exposure is not None and exposure <= 0):
                    raise ValueError()
            except ValueError:
                self.fail('Enter a positive exposure in microseconds and a finite gain in dB, or leave them blank.')
                return
            source = PySpinSource(serials, self.count.value(), fps=self.args.fps, sdk_path=self.args.sdk_path,
                                  exposure_us=exposure, gain_db=gain)
        self.engine = LiveEngine(source, self.board, self.board_file, self.args.output, self.args.workers)
        self.engine.auto_capture = self.auto.isChecked()
        self.engine.start()
        self.mode_badge.setText('SIMULATION · generated images' if demo else 'LIVE · direct PySpin')

    def set_auto(self, enabled):
        if self.engine:
            with self.engine.lock:
                self.engine.auto_capture = enabled

    def toggle_coverage(self, enabled):
        self.inspection.show_coverage = enabled
        self.inspection.update()

    def capture(self, role):
        if not self.engine or self.engine.stop_event.is_set():
            return
        try:
            self.engine.capture(role)
        except ValueError as exc:
            self.fail(str(exc))

    def select(self, serial):
        self.selected = serial
        for s, tile in self.tiles.items():
            tile.active = s == serial
            tile.update()
        if self.last_packet:
            self.update_inspection(self.last_packet)

    def update_inspection(self, packet):
        serial = self.selected
        observation = packet['observations'].get(serial)
        frame = packet['batch'].frames.get(serial)
        if frame is None or observation is None:
            self.inspection.image = None
            self.inspection.update()
            return
        self.inspection.image = as_image(frame.image)
        self.inspection.observation = dict(observation, image_size=frame.image.shape[1::-1])
        self.inspection.cells = packet['coverage']['cells'][serial]
        self.inspection.prediction = None
        prediction = getattr(self, 'prediction', None)
        if prediction:
            self.inspection.prediction = prediction['predictions'].get(serial)
        marker = observation['marker_px']
        marker_text = f'{marker:.0f} px markers' if marker is not None else 'No board scale'
        detail = f"{len(observation['ids'])}/{self.board.num_points} corners · {marker_text} · exposure {frame.exposure_us:g} µs"
        if prediction and serial in prediction['predictions']:
            error = prediction['predictions'][serial]['rms']
            detail += f"\n{error:.2f} px from reference {prediction['reference']}"
            if serial == prediction['reference']:
                detail += ' (pose-fit view)'
        self.inspection_info.setText(detail)
        self.inspection.update()

    def solve(self):
        if self.process is not None or not self.engine or not self.engine.session:
            return
        session = self.engine.session
        records = list(session.samples)
        if not records:
            self.fail('Capture training poses before starting calibration.')
            return
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe(duplex=False)
        self.process = context.Process(target=solve_process, args=(child, str(session.directory / 'board.yaml'), session.serials, records))
        self.process.start()
        child.close()
        self.solve_directory = session.directory
        self.solve_simulated = session.manifest['simulated']
        self.cancel_button.show()
        self.solve_status.setText('Starting calibration worker… live acquisition continues.')
        self.error_label.hide()

    def cancel_solve(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            self.process.close()
            self.process = None
            self.solve_status.setText('Calibration cancelled')
        if self.connection:
            self.connection.close()
            self.connection = None
        if hasattr(self, 'cancel_button'):
            self.cancel_button.hide()

    def poll_solve(self):
        try:
            while self.connection is not None and self.connection.poll():
                kind, value = self.connection.recv()
                if kind == 'progress':
                    self.solve_status.setText(value)
                elif kind == 'error':
                    self.fail(value)
                    self.solve_status.setText('Calibration needs attention')
                elif kind == 'result':
                    value['simulated'] = self.solve_simulated
                    path = write_result(self.solve_directory, value)
                    self.result = self.rig.result = value
                    self.rig.update()
                    state = 'Converged' if value['solver']['success'] else 'Provisional: iteration limit / solver issue'
                    self.solve_status.setText(f"{state}\n{len(value['training_ids'])} training · {len(value['validation_ids'])} validation\nSaved {path.name}\nMetric accuracy remains unverified.")
                    self.last_packet = None
        except EOFError:
            self.connection.close()
            self.connection = None
        except Exception as exc:
            self.fail(f'Calibration result: {exc}')
        if self.process and not self.process.is_alive():
            exitcode = self.process.exitcode
            self.process.join()
            self.process.close()
            self.process = None
            self.cancel_button.hide()
            if exitcode:
                self.fail(f'Calibration process exited with code {exitcode}')

    def refresh(self):
        self.poll_solve()
        running = self.engine is not None and self.engine.running()
        ready = running and self.engine.session is not None and not self.engine.stop_event.is_set()
        self.start_button.setText('Disconnect cameras' if running else 'Connect cameras')
        for item in (self.source_choice, self.count, self.serials, self.board_button, self.exposure, self.gain):
            item.setEnabled(not running and self.process is None)
        self.training_button.setEnabled(ready)
        self.validation_button.setEnabled(ready)
        self.solve_button.setEnabled(self.process is None and self.engine is not None and self.engine.session is not None)
        if self.closing and not running:
            self.close()
            return
        if self.engine is None:
            return
        state = self.engine.snapshot()
        if state['error']:
            self.fail(state['error'])
        session = state['session']
        if session:
            samples = list(session.samples)
            training = sum(r['role'] == 'training' for r in samples)
            self.count_label.setText(f'{training} training · {len(samples)-training} validation')
            self.session_label.setText(str(session.directory))
        batch, packet = state['batch'], state['packet']
        if batch and batch is not self.last_batch:
            columns = min(5, max(1, math.ceil(math.sqrt(len(batch.serials)))))
            for index, serial in enumerate(batch.serials):
                if serial not in self.tiles:
                    tile = CameraTile(serial)
                    if len(batch.serials) > 9:
                        tile.setMinimumSize(100, 95)
                    tile.selected.connect(self.select)
                    self.tiles[serial] = tile
                    self.wall_grid.addWidget(tile, index // columns, index % columns)
                tile = self.tiles[serial]
                frame = batch.frames.get(serial)
                tile.missing = frame is None
                if frame:
                    tile.image = as_image(frame.image, 360)
                    tile.detail = f'#{frame.frame_id} · {frame.exposure_us:g} µs'
                else:
                    tile.detail = 'MISSING FRAME'
                tile.update()
            if self.selected is None:
                self.select(batch.serials[0])
            spread = batch.spread_us
            self.wall_title.setText(f'CAMERA WALL · {len(batch.frames)}/{len(batch.serials)} · ' + (f'{spread:.1f} µs start spread' if spread is not None else 'single camera'))
            self.last_batch = batch
        if packet and packet is not self.last_packet:
            self.prediction = None
            if self.result:
                try:
                    self.prediction = live_projection(self.board, packet, self.result)
                except cv2.error:
                    pass
            self.rig.target = self.prediction
            self.rig.update()
            self.update_inspection(packet)
            self.guidance.setText(('Capture queued: waiting for a usable synchronized pose. ' if state['pending'] else '') + packet['guidance'])
            serials = packet['batch'].serials
            coverage = packet['coverage']
            self.metrics.setRowCount(len(serials))
            for row, serial in enumerate(serials):
                fraction = np.count_nonzero(coverage['cells'][serial]) / (GRID[0]*GRID[1])
                train = self.result['training'][serial]['rms'] if self.result else None
                test = self.result['validation'][serial] if self.result else None
                values = [serial, str(coverage['views'][serial]), f'{fraction:.0%}',
                          '—' if train is None else f'{train:.3f}',
                          '—' if test is None or test['rms'] is None else f"{test['rms']:.3f}",
                          '—' if test is None else f"{test['n']}/{test['expected_points']}"]
                for col, text in enumerate(values):
                    self.metrics.setItem(row, col, QtWidgets.QTableWidgetItem(text))
            self.overlaps.setRowCount(len(serials))
            self.overlaps.setColumnCount(len(serials))
            self.overlaps.setHorizontalHeaderLabels(serials)
            self.overlaps.setVerticalHeaderLabels(serials)
            for row in range(len(serials)):
                for col in range(len(serials)):
                    count = coverage['overlaps'][row, col]
                    item = QtWidgets.QTableWidgetItem('—' if row == col else str(count))
                    if count:
                        item.setBackground(QtGui.QColor(25, min(120, 45+int(count)*4), 90))
                    self.overlaps.setItem(row, col, item)
            self.last_packet = packet
        if packet:
            age = time.monotonic() - packet['analyzed_at']
            self.inspection_title.setText(f'BOARD INSPECTION · {self.selected} · analyzed {age:.1f}s ago')
        if not running:
            self.mode_badge.setText('DISCONNECTED · retained session')

    def closeEvent(self, event):
        self.cancel_solve()
        if self.engine and self.engine.running():
            self.closing = True
            self.engine.stop()
            self.guidance.setText('Closing acquisition and restoring camera settings…')
            event.ignore()
        else:
            self.timer.stop()
            event.accept()
