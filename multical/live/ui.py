"""Native live visualizer. All hardware and fitting work stays off the GUI thread."""
import math
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from multical.board import load_config
from multical.io.interop import load_seed, load_captury_seed, validate_seed_geometry
from .calibration import solve_process
from .engine import LiveEngine
from .audio import CaptureSounds, AttentionCue, GuidanceCue
from .guidance import capture_progress, inspection_advice, inspection_group
from .metrics import GRID, live_projection
from .session import write_result
from .sources import PySpinSource, SimulatedSource


ACCENT = '#52dfbc'
STYLE = '''
QMainWindow, QWidget { background: #171717; color: #eeeeee; font-family: sans-serif; font-size: 12px; }
QFrame#panel { background: #202020; border: 1px solid #363636; border-radius: 9px; }
QLabel { background: transparent; }
QLabel#muted { color: #aaaaaa; }
QLabel#guidance { background: #232827; color: #d5e9e3; border: 1px solid #3b4a45; border-radius: 8px; padding: 9px; font-size: 14px; }
QLabel#error { color: #ffb2a5; background: #3b2528; border-radius: 6px; padding: 10px; }
QPushButton { background: #303030; border: 1px solid #454545; border-radius: 6px; padding: 6px 9px; }
QPushButton:checked { border: 1px solid #52dfbc; background: #303a36; }
QPushButton:hover { background: #3c3c3c; }
QPushButton:disabled { color: #777777; background: #252525; border-color: #353535; }
QPushButton#primary { background: #52dfbc; color: #0c2521; font-weight: 600; border: none; }
QLineEdit, QSpinBox, QComboBox { background: #1b1b1b; border: 1px solid #444444; border-radius: 5px; padding: 5px; }
QCheckBox { spacing: 7px; padding: 4px 0; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #999999; border-radius: 2px; background: #666666; }
QCheckBox::indicator:hover { background: #808080; border-color: #bbbbbb; }
QCheckBox::indicator:checked { background: #a0a0a0; border-color: #bdbdbd; image: url(CHECK_ICON_PATH); }
QCheckBox::indicator:disabled { background: #444444; border-color: #606060; }
QTableWidget { background: #1b1b1b; alternate-background-color: #242424; border: none; gridline-color: #333333; }
QHeaderView::section { background: #292929; color: #bbbbbb; padding: 5px; border: none; }
QScrollArea { border: none; }
QSplitter::handle { background: #353535; }
QTabBar::tab { background: #242424; padding: 8px 16px; }
QTabBar::tab:selected { color: #52dfbc; background: #363636; }
QTabWidget::pane { border: 1px solid #363636; }
QToolTip { background: #303030; color: white; border: 1px solid #555555; }
'''


def label(text, name=None):
    item = QtWidgets.QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(True)
    return item


def capture_icon(paused):
    pixels = QtGui.QPixmap(20, 20)
    pixels.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixels)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor('#eeeeee'))
    if paused:
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(6, 3), QtCore.QPoint(17, 10), QtCore.QPoint(6, 17)]))
    else:
        painter.drawRect(5, 4, 4, 12)
        painter.drawRect(12, 4, 4, 12)
    painter.end()
    return QtGui.QIcon(pixels)


def sound_icon(enabled):
    pixels = QtGui.QPixmap(24, 20)
    pixels.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixels)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor('#eeeeee'))
    painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in
        [(2, 7), (6, 7), (11, 3), (11, 17), (6, 13), (2, 13)]]))
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.setPen(QtGui.QPen(QtGui.QColor('#eeeeee'), 1.6))
    if enabled:
        painter.drawArc(10, 5, 9, 10, -60 * 16, 120 * 16)
        painter.drawArc(9, 1, 14, 18, -60 * 16, 120 * 16)
    else:
        painter.drawLine(15, 7, 21, 13)
        painter.drawLine(15, 13, 21, 7)
    painter.end()
    return QtGui.QIcon(pixels)


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
        painter.fillRect(self.rect(), QtGui.QColor('#202020'))
        if self.image is not None and not self.missing:
            available = QtCore.QRect(4, 28, self.width()-8, self.height()-54)
            size = self.image.size().scaled(available.size(), QtCore.Qt.KeepAspectRatio)
            target = QtCore.QRect(QtCore.QPoint(0, 0), size)
            target.moveCenter(available.center())
            painter.drawImage(target, self.image)
        painter.setPen(QtGui.QColor('#eeeeee'))
        painter.drawText(10, 20, self.serial)
        painter.setPen(QtGui.QColor('#ffb2a5' if self.missing else '#aaaaaa'))
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
        self.show_ids = False
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
        p.fillRect(self.rect(), QtGui.QColor('#111111'))
        if self.image is None:
            p.setPen(QtGui.QColor('#aaaaaa'))
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
            if self.show_ids:
                p.drawText(at + QtCore.QPointF(4, -4), str(identity))
            if self.prediction is not None:
                predicted = pixel(self.prediction['points'][index])
                p.setPen(QtGui.QPen(QtGui.QColor('#ffb45f'), 1.5))
                p.drawLine(at, predicted)
                p.drawLine(predicted+QtCore.QPointF(-3, 0), predicted+QtCore.QPointF(3, 0))
                p.drawLine(predicted+QtCore.QPointF(0, -3), predicted+QtCore.QPointF(0, 3))


class RigView(QtWidgets.QWidget):
    """Dependency-free orbitable projection of calibrated camera centres and poses."""
    camera_selected = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.result = self.target = None
        self.validation_views, self.observations, self.hit_points = {}, {}, {}
        self.drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(280, 170)
        self.reset_button = QtWidgets.QToolButton(self)
        self.reset_button.setText('↺')
        self.reset_button.setToolTip('Reset to isometric view (fit all cameras)')
        self.reset_button.setAccessibleName('Reset isometric view')
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_view()

    def reset_view(self):
        self.yaw, self.pitch, self.zoom = -math.pi / 4, math.asin(1 / math.sqrt(3)), 1.0
        self.drag = None
        self.update()

    def resizeEvent(self, event):
        self.reset_button.setGeometry(self.width()-36, 4, 28, 26)
        super().resizeEvent(event)

    def camera_status(self, serial):
        views = self.validation_views.get(serial, 0)
        metric = (self.result or {}).get('validation', {}).get(serial, {})
        n, failed = metric.get('n', 0), metric.get('failed_points', 0)
        if n:
            color, status = '#69bafa', 'Evaluated (not an accuracy pass)'
        elif failed:
            color, status = '#ee8d86', 'Could not independently evaluate'
        elif views:
            color, status = '#ffb45f', 'Captured; awaiting evaluation'
        else:
            color, status = '#888888', 'No validation poses'
        detail = f'{serial} · {status}\n{views} varied validation poses in this session'
        if n:
            detail += f"\nLast evaluation: {metric['rms']:.2f} px RMS · {n} corners · {failed} failed"
            detail += '\nNew captures require another evaluation.'
        elif failed:
            detail += f'\n{failed} corners lacked an independent prediction'
        detail += '\nLabels show last four serial digits; hover shows full serial.\nArrow: camera viewing direction · ring: board visible now\nClick to inspect camera; drag to orbit; scroll to zoom'
        return color, detail

    def mousePressEvent(self, event):
        self.drag = self.press_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.drag is None:
            serial = next((s for s, point in self.hit_points.items()
                           if (point - QtCore.QPointF(event.pos())).manhattanLength() < 14), None)
            if serial:
                QtWidgets.QToolTip.showText(event.globalPos(), self.camera_status(serial)[1], self)
            else:
                QtWidgets.QToolTip.hideText()
        if self.drag is not None:
            delta = event.pos() - self.drag
            self.yaw += delta.x() * .008
            self.pitch = np.clip(self.pitch + delta.y() * .008, -1.4, 1.4)
            self.drag = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag is not None and (event.pos() - self.press_pos).manhattanLength() < 4:
            serial = next((s for s, point in self.hit_points.items()
                           if (point - QtCore.QPointF(event.pos())).manhattanLength() < 14), None)
            if serial:
                self.camera_selected.emit(serial)
        self.drag = None

    def wheelEvent(self, event):
        self.zoom = np.clip(self.zoom * math.exp(event.angleDelta().y() / 1200), .3, 4.)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#191919'))
        p.setPen(QtGui.QColor('#aaaaaa'))
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
        projected_extent = np.maximum(np.ptp((points-centre) @ rotation.T, axis=0), .5)
        scale = min(max(self.width()-110, 60) / projected_extent[0],
                    max(self.height()-100, 40) / projected_extent[1]) * .85 * self.zoom
        def project(point):
            transformed = rotation @ (point-centre)
            return QtCore.QPointF(self.width()/2 + transformed[0]*scale, self.height()/2 - transformed[1]*scale)
        p.setPen(QtGui.QPen(QtGui.QColor('#3b3b3b'), 1))
        for a in np.linspace(-span, span, 9):
            p.drawLine(project(np.array([a, 0, -span])+centre), project(np.array([a, 0, span])+centre))
            p.drawLine(project(np.array([-span, 0, a])+centre), project(np.array([span, 0, a])+centre))
        for target in targets:
            p.setPen(QtGui.QColor('#aaaaaa'))
            p.drawEllipse(project(target), 2, 2)
        self.hit_points = {}
        for serial, pos in centres.items():
            color, _ = self.camera_status(serial)
            p.setPen(QtGui.QPen(QtGui.QColor(color), 2))
            at = project(pos)
            self.hit_points[serial] = at
            direction = poses[serial][:3, :3].T @ np.array([0, 0, span*.12])
            tip = project(pos+direction)
            p.drawLine(at, tip)
            delta = tip-at
            length = math.hypot(delta.x(), delta.y())
            if length > 7:
                unit = delta / length
                side = QtCore.QPointF(-unit.y(), unit.x())
                p.drawLine(tip, tip-unit*5+side*3)
                p.drawLine(tip, tip-unit*5-side*3)
            p.setBrush(QtGui.QColor(color))
            p.drawEllipse(at, 4, 4)
            p.drawText(at+QtCore.QPointF(6, -6), serial[-4:])
            if self.observations.get(serial, {}).get('usable'):
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 2))
                p.drawEllipse(at, 8, 8)
        if self.target is not None:
            world = self.target['world_target']
            p.setPen(QtGui.QPen(QtGui.QColor('#ffb45f'), 3))
            pos = world[:3, 3]
            p.drawEllipse(project(pos), 7, 7)
            p.drawLine(project(pos), project(pos + world[:3, 2] * span*.15))
        p.setPen(QtGui.QColor('#aaaaaa'))
        missing = sum(self.camera_status(s)[0] == '#888888' for s in poses)
        p.drawText(12, 20, f'World · {missing}/{len(poses)} no poses')
        x, y = 12, self.height() - (28 if self.width() < 380 else 10)
        for color, title in [('#888888', 'No poses'), ('#ffb45f', 'Pending'), ('#69bafa', 'Evaluated'), ('#ee8d86', 'Failed')]:
            p.setPen(QtGui.QColor(color))
            width = p.fontMetrics().horizontalAdvance('● ' + title) + 12
            if x + width > self.width():
                x, y = 12, y + 18
            p.drawText(x, y, '● ' + title)
            x += width


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.capture_sounds = CaptureSounds(self)
        self.last_sound_event = None
        self.attention_cue = AttentionCue()
        self.guidance_cue = GuidanceCue()
        self.last_capture_sound_at = float('-inf')
        preferences = Path(__file__).resolve().parents[2] / 'live-sessions' / 'ui-preferences.ini'
        preferences.parent.mkdir(parents=True, exist_ok=True)
        self.preferences = QtCore.QSettings(str(preferences), QtCore.QSettings.IniFormat)
        self.board_file = str(Path(args.boards).resolve())
        self.board = self._load_board(self.board_file)
        self.engine = self.result = self.process = self.connection = None
        self.seed = load_seed(args.seed) if getattr(args, 'seed', None) else None
        self.seed_checked = False
        self.last_packet = self.last_batch = None
        self.selected = None
        self.focus_changed_at = 0.
        self.quad_serials = []
        self.tiles = {}
        self.closing = False
        self.pending_session_action = None
        self.setWindowTitle('Multical Live')
        self.resize(1500, 920)
        self.setStyleSheet(STYLE.replace('CHECK_ICON_PATH', (Path(__file__).with_name('check.svg')).as_posix()))
        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        self.mode_badge = label('DISCONNECTED', 'muted')
        self.mode_badge.setWordWrap(False)
        body = QtWidgets.QHBoxLayout()
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName('panel')
        sidebar.setFixedWidth(245)
        controls = QtWidgets.QVBoxLayout(sidebar)
        controls.setContentsMargins(12, 12, 12, 12)
        controls.setSpacing(7)
        settings_toggle = QtWidgets.QToolButton()
        settings_toggle.setText('Camera settings')
        settings_toggle.setCheckable(True)
        settings_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        settings_toggle.setArrowType(QtCore.Qt.RightArrow)
        controls.addWidget(settings_toggle)
        settings = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 4, 0, 4)
        self.source_choice = QtWidgets.QComboBox()
        self.source_choice.addItems(['FLIR cameras · PySpin', 'Simulated rig · testing'])
        self.source_choice.setCurrentIndex(1 if args.demo else 0)
        settings_layout.addWidget(self.source_choice)
        self.capture_mode = QtWidgets.QComboBox()
        self.capture_mode.addItem('Stationary board', 'stationary')
        self.capture_mode.addItem('Moving board · strict timing', 'motion')
        self.capture_mode.setCurrentIndex(1 if getattr(args, 'capture_mode', 'stationary') == 'motion' else 0)
        self.capture_mode.setToolTip('Hold the board still during each capture. Different fixed exposures are allowed in stationary mode.')
        settings_layout.addWidget(self.capture_mode)
        settings_layout.addWidget(label('Expected cameras', 'muted'))
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 64)
        self.count.setValue(args.count)
        settings_layout.addWidget(self.count)
        settings_layout.addWidget(label('Camera serials (optional, comma separated)', 'muted'))
        self.serials = QtWidgets.QLineEdit(','.join(args.serials))
        settings_layout.addWidget(self.serials)
        settings_layout.addWidget(label('Exposure (µs) / gain (dB) override · optional', 'muted'))
        exposure_row = QtWidgets.QHBoxLayout()
        self.exposure = QtWidgets.QLineEdit('' if args.exposure_us is None else str(args.exposure_us))
        self.gain = QtWidgets.QLineEdit('' if args.gain_db is None else str(args.gain_db))
        self.exposure.setPlaceholderText('Camera value')
        self.gain.setPlaceholderText('Camera value')
        exposure_row.addWidget(self.exposure)
        exposure_row.addWidget(self.gain)
        settings_layout.addLayout(exposure_row)
        settings.hide()
        settings_toggle.toggled.connect(settings.setVisible)
        settings_toggle.toggled.connect(lambda expanded: settings_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow))
        controls.addWidget(settings)
        self.board_label = label(f'{Path(self.board_file).name}\n{self.board.num_points} corners · {self.board.square_length*1000:g} mm squares', 'muted')
        controls.addWidget(self.board_label)
        self.board_button = QtWidgets.QPushButton('Board…')
        self.board_button.clicked.connect(self.choose_board)
        controls.addWidget(self.board_button)
        self.seed_button = QtWidgets.QPushButton('Load calibration…')
        self.seed_button.clicked.connect(self.choose_seed)
        self.seed_button.setToolTip('Import Captury .calib or load Multical JSON. Captury import uses the connected cameras to verify identities and image sizes.')
        controls.addWidget(self.seed_button)
        self.start_button = QtWidgets.QPushButton('Connect cameras')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.toggle_capture)
        controls.addWidget(self.start_button)
        self.start_button.setToolTip('Close other camera applications before connecting.')
        controls.addSpacing(6)
        controls.addWidget(label('COLLECT POSES', 'muted'))
        self.training_button = QtWidgets.QPushButton('Capture  [Space]')
        self.validation_button = QtWidgets.QPushButton('Validation  [V]')
        self.training_button.clicked.connect(lambda: self.capture('training'))
        self.validation_button.clicked.connect(lambda: self.capture('validation'))
        controls.addWidget(self.training_button)
        controls.addWidget(self.validation_button)
        self.auto = QtWidgets.QCheckBox('Auto-capture')
        self.auto.setToolTip('Save new steady poses in the selected role. Validation needs two usable cameras and stays separate from training.')
        self.auto.setChecked(args.auto_capture)
        self.auto.toggled.connect(self.set_auto)
        auto_options = QtWidgets.QHBoxLayout()
        auto_options.addWidget(self.auto)
        auto_options.addStretch()
        self.sounds = QtWidgets.QToolButton()
        self.sounds.setCheckable(True)
        self.sounds.setIconSize(QtCore.QSize(24, 20))
        self.sounds.setChecked(self.preferences.value('sounds/enabled', True, type=bool))
        self.sounds.toggled.connect(self.set_sounds)
        self.set_sounds(self.sounds.isChecked())
        sound_menu = QtWidgets.QMenu(self.sounds)
        for cue, title in [('training', 'Training saved'), ('validation', 'Validation saved'),
                           ('attention', 'Attention: capture blocked'), ('milestone', 'Milestone: groups joined'),
                           ('hold', 'Hold still'), ('tilt', 'Change tilt or position'), ('single', 'Only one camera sees board')]:
            action = sound_menu.addAction(title)
            action.triggered.connect(lambda checked=False, role=cue: self.preview_sound(role))
        self.sounds.setMenu(sound_menu)
        self.sounds.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        auto_options.addWidget(self.sounds)
        if self.capture_sounds.error:
            self.sounds.setEnabled(False)
            self.sounds.setToolTip('Audio unavailable: ' + self.capture_sounds.error)
        controls.addLayout(auto_options)
        self.auto_role = QtWidgets.QComboBox()
        self.auto_role.addItem('Training', 'training')
        self.auto_role.addItem('Validation', 'validation')
        self.auto_role.setCurrentIndex(1 if getattr(args, 'auto_capture_role', 'training') == 'validation' else 0)
        self.auto_role.setToolTip('Role for automatic captures. Manual Space and V keep their usual roles.')
        self.auto_role.currentIndexChanged.connect(self.set_auto_role)
        controls.addWidget(self.auto_role)
        self.count_label = label('0 training · 0 validation', 'muted')
        controls.addWidget(self.count_label)
        self.saved_label = label('No poses saved yet.', 'muted')
        controls.addWidget(self.saved_label)
        help_button = QtWidgets.QPushButton('Help')
        help_button.clicked.connect(self.show_walkthrough)
        controls.addWidget(help_button)
        self.solve_button = QtWidgets.QPushButton('Calibrate  [C]')
        self.solve_button.clicked.connect(self.solve)
        controls.addWidget(self.solve_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel calibration')
        self.cancel_button.clicked.connect(self.cancel_solve)
        self.cancel_button.setVisible(False)
        controls.addWidget(self.cancel_button)
        self.solve_status = label('Collect poses to begin.', 'muted')
        controls.addWidget(self.solve_status)
        controls.addStretch()
        self.session_label = label('No session', 'muted')
        self.session_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.session_label)
        body.addWidget(sidebar)
        main = QtWidgets.QVBoxLayout()
        session_controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton('Pause')
        self.pause_button.setCheckable(True)
        self.pause_button.setToolTip('Pause training and validation capture; previews and detection continue. An in-progress disk save may finish.')
        self.pause_button.toggled.connect(self.pause_capture)
        self.new_button = QtWidgets.QPushButton('New session')
        self.new_button.clicked.connect(self.new_session)
        self.load_session_button = QtWidgets.QPushButton('Open session…')
        self.load_session_button.clicked.connect(self.open_session)
        self.save_button = QtWidgets.QPushButton('Save calibration…')
        self.save_button.clicked.connect(self.save_calibration)
        for button, icon in ((self.pause_button, QtWidgets.QStyle.SP_MediaPause),
                             (self.load_session_button, QtWidgets.QStyle.SP_DialogOpenButton),
                             (self.save_button, QtWidgets.QStyle.SP_DialogSaveButton)):
            button.setIcon(self.style().standardIcon(icon))
        self.new_button.setToolTip('New calibration session — preserve the current session on disk.')
        self.load_session_button.setToolTip('Open session — restore saved captures and coverage.')
        self.save_button.setToolTip('Save calibration — export camera parameters as JSON. Images remain in the session folder.')
        plus = QtGui.QPixmap(20, 20)
        plus.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(plus)
        painter.setPen(QtGui.QPen(QtGui.QColor('#eeeeee'), 2))
        painter.drawLine(4, 10, 16, 10)
        painter.drawLine(10, 4, 10, 16)
        painter.end()
        self.new_button.setIcon(QtGui.QIcon(plus))
        for button in (self.new_button, self.load_session_button, self.save_button, self.pause_button):
            button.setAccessibleName(button.text())
            button.setText('')
            button.setFixedSize(36, 32)
            button.setIconSize(QtCore.QSize(20, 20))
            if button is self.pause_button:
                session_controls.addStretch()
            session_controls.addWidget(button)
        self.pause_button.setIcon(capture_icon(False))
        controls.insertLayout(0, session_controls)
        self.guidance = label('Connect cameras to begin.', 'guidance')
        main.addWidget(self.guidance)
        progress_row = QtWidgets.QHBoxLayout()
        self.progress_label = label('Waiting for cameras', 'muted')
        progress_row.addWidget(self.progress_label, 1)
        self.target_button = QtWidgets.QPushButton('Next camera')
        self.target_button.clicked.connect(self.inspect_target)
        self.target_button.setEnabled(False)
        progress_row.addWidget(self.target_button)
        main.addLayout(progress_row)
        self.next_step = label('', 'muted')
        self.next_step.setWordWrap(False)
        progress_row.addWidget(self.next_step)
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
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.wall_scroll = scroll
        self.wall_layout_key = None
        scroll.viewport().installEventFilter(self)
        self.wall_grid.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        scroll.setWidget(self.wall)
        wall_layout.addWidget(scroll)
        upper.addWidget(wall_box)
        detail = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 0, 0, 0)
        self.inspection_title = label('BOARD INSPECTION', 'muted')
        detail_layout.addWidget(self.inspection_title)
        focus_controls = QtWidgets.QHBoxLayout()
        self.four_up = QtWidgets.QCheckBox('4-up')
        self.four_up.setChecked(True)
        self.auto_focus = QtWidgets.QCheckBox('Auto-focus')
        self.auto_focus.setChecked(True)
        self.auto_focus.setToolTip('Reconsider the target every 10 seconds. Prefer cameras seeing the board with fewer saved views. Clicking a camera pins it by turning auto-focus off.')
        focus_controls.addWidget(self.four_up)
        focus_controls.addWidget(self.auto_focus)
        detail_layout.addLayout(focus_controls)
        self.quad = QtWidgets.QWidget()
        quad_layout = QtWidgets.QGridLayout(self.quad)
        quad_layout.setContentsMargins(0, 0, 0, 0)
        self.quad_views, self.quad_labels = [], []
        for i in range(4):
            box = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(box)
            layout.setContentsMargins(2, 2, 2, 2)
            title = label('Waiting for cameras', 'muted')
            view = InspectionView()
            view.setMinimumSize(170, 120)
            layout.addWidget(title)
            layout.addWidget(view, 1)
            self.quad_labels.append(title)
            self.quad_views.append(view)
            quad_layout.addWidget(box, i // 2, i % 2)
        detail_layout.addWidget(self.quad, 1)
        self.inspection = InspectionView()
        detail_layout.addWidget(self.inspection, 1)
        self.inspection.hide()
        self.four_up.toggled.connect(self.toggle_four_up)
        self.inspection_info = label('Green: detected corners · amber: predicted corners from another camera', 'muted')
        detail_layout.addWidget(self.inspection_info)
        self.inspection_advice = label('Select a camera to see its capture advice.', 'muted')
        detail_layout.addWidget(self.inspection_advice)
        coverage_toggle = QtWidgets.QCheckBox('Coverage')
        coverage_toggle.setChecked(True)
        coverage_toggle.toggled.connect(self.toggle_coverage)
        overlay_controls = QtWidgets.QHBoxLayout()
        overlay_controls.addWidget(coverage_toggle)
        ids_toggle = QtWidgets.QCheckBox('Corner IDs')
        ids_toggle.toggled.connect(self.toggle_corner_ids)
        overlay_controls.addWidget(ids_toggle)
        overlay_controls.addStretch()
        detail_layout.addLayout(overlay_controls)
        upper.addWidget(detail)
        upper.setSizes([650, 550])
        lower = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.rig = RigView()
        self.rig.camera_selected.connect(self.select)
        if self.seed:
            self.result = self.rig.result = self.seed
            self.solve_status.setText('Seed loaded · native images required. Capture validation to check it, or training to refine poses with fixed lenses.')
        lower.addWidget(self.rig)
        self.tabs = QtWidgets.QTabWidget()
        self.metrics = QtWidgets.QTableWidget(0, 6)
        self.metrics.setHorizontalHeaderLabels(['Camera', 'Varied poses', 'Coverage', 'Train px', 'Test px', 'Test n'])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.metrics.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metrics.setAlternatingRowColors(True)
        self.metrics.verticalHeader().hide()
        self.metrics.cellClicked.connect(lambda row, col: self.select(self.metrics.item(row, 0).text()))
        self.metrics.setToolTip('Click a row to inspect that camera. Pose counts and coverage describe capture diversity, not accuracy.')
        self.overlaps = QtWidgets.QTableWidget()
        self.overlaps.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.metrics, 'Coverage & residuals')
        self.tabs.addTab(self.overlaps, 'Shared pose counts')
        lower.addWidget(self.tabs)
        lower.setSizes([500, 700])
        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical.addWidget(upper)
        vertical.addWidget(lower)
        vertical.setSizes([750, 150])
        main.addWidget(vertical, 1)
        body.addLayout(main, 1)
        outer.addLayout(body, 1)
        self.footer = label('Accuracy unverified', 'muted')
        self.footer.setToolTip('Pixel residuals describe image agreement, not millimetre accuracy.')
        footer_row = QtWidgets.QHBoxLayout()
        footer_row.addWidget(self.footer)
        footer_row.addStretch()
        footer_row.addWidget(self.mode_badge)
        outer.addLayout(footer_row)
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

    def pause_capture(self, paused):
        if self.engine:
            self.engine.set_paused(paused)
        self.pause_button.setAccessibleName('Resume capture' if paused else 'Pause capture')
        self.pause_button.setToolTip('Resume capture' if paused else 'Pause capture — previews continue; an in-progress save may finish.')
        self.pause_button.setIcon(capture_icon(paused))

    def switch_session(self, configure):
        self.cancel_solve()
        self.pending_session_action = configure
        if self.engine and self.engine.running():
            self.engine.set_paused(True)
            self.engine.stop()
            self.guidance.setText('Switching session…')
        # refresh invokes configure only after the old engine releases the cameras.

    def new_session(self):
        def configure():
            self.args.resume = None
            self.seed = None
        self.switch_session(configure)

    def open_session(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open saved capture session', self.args.output)
        if not directory:
            return
        try:
            path = Path(directory)
            manifest = json.loads((path / 'manifest.json').read_text())
            board = self._load_board(path / 'board.yaml')
            serials = manifest['camera_serials']
            if manifest.get('schema_version') != 1 or not serials or len(set(serials)) != len(serials):
                raise ValueError('Unsupported or invalid session manifest')
            seed = load_seed(path / 'bootstrap.json') if (path / 'bootstrap.json').exists() else None
            # Validate session completeness before disconnecting the current rig.
            from .session import Session
            Session(self.args.output, path / 'board.yaml', serials,
                    simulated=manifest['simulated'], capture_mode=manifest.get('capture_mode', 'stationary'), resume=path)
            def configure():
                self.args.resume = str(path)
                self.seed = seed
                self.board, self.board_file = board, str(path / 'board.yaml')
                self.board_label.setText(f'board.yaml\n{board.num_points} corners · {board.square_length*1000:g} mm squares')
                self.count.setValue(len(serials))
                self.serials.setText(','.join(serials))
                self.source_choice.setCurrentIndex(1 if manifest['simulated'] else 0)
                self.capture_mode.setCurrentIndex(1 if manifest.get('capture_mode') == 'motion' else 0)
            self.switch_session(configure)
        except Exception as exc:
            self.fail(f'Cannot open session: {exc}')

    def save_calibration(self):
        if self.result is None:
            self.fail('No fitted or loaded calibration yet. Captures are already saved automatically.')
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save calibration', 'calibration.json', 'JSON (*.json)')
        if not filename:
            return
        try:
            import os
            import tempfile
            destination = Path(filename)
            payload = json.dumps(self.result, indent=2, allow_nan=False)
            with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, prefix='.calibration-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            self.solve_status.setText(f'Saved calibration to {destination}. Captures remain in the session folder.')
        except Exception as exc:
            self.fail(f'Cannot save calibration: {exc}')

    def choose_seed(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load calibration into a new session', '', 'Calibrations (*.json *.calib);;Captury (*.calib);;Multical (*.json)')
        if filename:
            try:
                if Path(filename).suffix.lower() == '.calib':
                    batch = self.engine.snapshot()['batch'] if self.engine else None
                    if batch is None or len(batch.frames) != len(batch.serials):
                        raise ValueError('Connect all cameras before importing Captury so image size and MAC identities can be verified.')
                    sizes = {tuple(f.image.shape[1::-1]) for f in batch.frames.values()}
                    if len(sizes) != 1:
                        raise ValueError('Captury import currently requires matching native image sizes across cameras.')
                    identities = getattr(self.engine.source, 'mac_to_serial', {})
                    if len(identities) != len(batch.serials):
                        raise ValueError('Connected cameras do not expose a complete MAC identity map; cannot safely match Captury cameras.')
                    try:
                        seed = load_captury_seed(filename, next(iter(sizes)), identities)
                    except ValueError as exc:
                        if 'select --frame explicitly' not in str(exc):
                            raise
                        frame, accepted = QtWidgets.QInputDialog.getInt(self, 'Animated calibration', 'Frame number to import:', 0, 0, 2147483647)
                        if not accepted:
                            return
                        seed = load_captury_seed(filename, next(iter(sizes)), identities, frame)
                    validate_seed_geometry(seed, batch.serials, {s: f.metadata()['image_size'] for s, f in batch.frames.items()})
                else:
                    seed = load_seed(filename)
                serials = list(seed['cameras'])
                def configure():
                    self.args.resume = None
                    self.seed = seed
                    self.count.setValue(len(serials))
                    self.serials.setText(','.join(serials))
                self.switch_session(configure)
            except Exception as exc:
                self.fail(str(exc))

    def toggle_capture(self):
        if self.engine and self.engine.running():
            self.engine.stop()
            self.start_button.setText('Disconnecting…')
            return
        self.error_label.hide()
        self.cancel_solve()
        self.result = self.rig.result = self.seed
        self.rig.target = None
        self.rig.validation_views = {}
        self.rig.observations = {}
        self.rig.reset_view()
        self.seed_checked = False
        self.last_packet = self.last_batch = None
        for tile in self.tiles.values():
            self.wall_grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles = {}
        self.wall_layout_key = None
        self.selected = None
        self.focus_changed_at = 0.
        self.quad_serials = []
        self.metrics.setRowCount(0)
        self.overlaps.setRowCount(0)
        self.inspection.image = None
        demo = self.source_choice.currentIndex() == 1
        capture_mode = self.capture_mode.currentData()
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
                                  exposure_us=exposure, gain_db=gain, capture_mode=capture_mode)
        self.engine = LiveEngine(source, self.board, self.board_file, self.args.output, self.args.workers,
                                 capture_mode=capture_mode, resume=getattr(self.args, 'resume', None))
        self.engine.auto_capture = self.auto.isChecked()
        self.engine.auto_capture_role = self.auto_role.currentData()
        self.engine.set_paused(self.pause_button.isChecked())
        self.engine.start()
        self.mode_badge.setText('SIMULATION · generated images' if demo else 'LIVE · direct PySpin')

    def set_sounds(self, enabled):
        self.sounds.setIcon(sound_icon(enabled))
        action = 'Mute sounds' if enabled else 'Unmute sounds'
        self.sounds.setAccessibleName(action)
        self.sounds.setToolTip(action + ' · arrow previews each cue (also while muted)')
        self.preferences.setValue('sounds/enabled', enabled)
        self.preferences.sync()
        if not enabled:
            self.capture_sounds.stop()

    def preview_sound(self, role):
        self.capture_sounds.play(role)

    def notify_capture(self, event):
        if event is None:
            return
        key = (event['session'], event['id'])
        if key == self.last_sound_event:
            return
        self.last_sound_event = key
        self.last_capture_sound_at = time.monotonic()
        if self.sounds.isChecked():
            self.capture_sounds.play('milestone' if event.get('milestone') else event['role'])

    def set_auto_role(self, _index):
        if self.engine:
            with self.engine.lock:
                self.engine.auto_capture_role = self.auto_role.currentData()

    def set_auto(self, enabled):
        if self.engine:
            with self.engine.lock:
                self.engine.auto_capture = enabled

    def eventFilter(self, watched, event):
        if hasattr(self, 'wall_scroll') and watched is self.wall_scroll.viewport():
            if event.type() == QtCore.QEvent.Resize:
                QtCore.QTimer.singleShot(0, self.reflow_wall)
        return super().eventFilter(watched, event)

    def reflow_wall(self):
        if not self.tiles:
            return
        margins = self.wall_grid.contentsMargins()
        width = max(1, self.wall_scroll.viewport().width() - margins.left() - margins.right())
        spacing = max(0, self.wall_grid.horizontalSpacing())
        columns = min(len(self.tiles), max(1, (width + spacing) // (160 + spacing)))
        key = (columns, tuple(self.tiles))
        if key == self.wall_layout_key:
            return
        self.wall_layout_key = key
        for tile in self.tiles.values():
            self.wall_grid.removeWidget(tile)
        for col in range(self.wall_grid.columnCount()):
            self.wall_grid.setColumnStretch(col, 0)
            self.wall_grid.setColumnMinimumWidth(col, 0)
        for row in range(self.wall_grid.rowCount()):
            self.wall_grid.setRowStretch(row, 0)
            self.wall_grid.setRowMinimumHeight(row, 0)
        for index, tile in enumerate(self.tiles.values()):
            self.wall_grid.addWidget(tile, index // columns, index % columns)
        for col in range(columns):
            self.wall_grid.setColumnStretch(col, 1)
        self.wall_grid.invalidate()

    def toggle_four_up(self, enabled):
        self.quad.setVisible(enabled)
        self.inspection.setVisible(not enabled)
        if self.last_packet:
            self.update_inspection(self.last_packet)

    def update_focus(self, packet):
        now = time.monotonic()
        if self.auto_focus.isChecked() and now - self.focus_changed_at >= 10:
            group = inspection_group(packet['coverage'], packet['observations'])
            if group:
                self.select(group[0][0], automatic=True)
            self.focus_changed_at = now
        if not self.quad_serials or now - getattr(self, 'group_changed_at', 0.) >= 10:
            self.quad_serials = [s for s, reason in inspection_group(packet['coverage'], packet['observations'], self.selected)]
            self.group_changed_at = now

    def update_quad(self, packet):
        ranked = dict(inspection_group(packet['coverage'], packet['observations'], self.selected, limit=len(packet['coverage']['views'])))
        for i, view in enumerate(self.quad_views):
            if i >= len(self.quad_serials):
                view.image = view.observation = None
                self.quad_labels[i].setText('No additional camera')
                view.update()
                continue
            serial = self.quad_serials[i]
            if getattr(view, 'serial', None) != serial:
                view.serial = serial
                view.zoom = 1.
                view.pan = QtCore.QPointF(0, 0)
            obs = packet['observations'].get(serial)
            frame = packet['batch'].frames.get(serial)
            count = len(obs['ids']) if obs else 0
            self.quad_labels[i].setToolTip(inspection_advice(obs, packet.get('novel', {}).get(serial, True)))
            self.quad_labels[i].setText(f"{serial} · {count}/{self.board.num_points} · {packet['coverage']['views'][serial]} poses\n{ranked.get(serial, 'Scout: overlap not established')}")
            view.image = as_image(frame.image, 640) if frame else None
            view.observation = dict(obs, image_size=frame.image.shape[1::-1]) if obs and frame else None
            view.cells = packet['coverage']['cells'][serial]
            view.prediction = (getattr(self, 'prediction', None) or {}).get('predictions', {}).get(serial)
            view.update()

    def toggle_corner_ids(self, enabled):
        for view in [self.inspection, *self.quad_views]:
            view.show_ids = enabled
            view.update()

    def toggle_coverage(self, enabled):
        for view in self.quad_views:
            view.show_coverage = enabled
            view.update()
        self.inspection.show_coverage = enabled
        self.inspection.update()

    def capture(self, role):
        if not self.engine or self.engine.stop_event.is_set():
            return
        try:
            self.engine.capture(role)
        except ValueError as exc:
            self.fail(str(exc))

    def show_walkthrough(self):
        QtWidgets.QMessageBox.information(self, 'Help',
            f'1. Use a rigid board matching the loaded configuration ({self.board.square_length*1000:g} mm squares). Keep focus and zoom fixed.\n\n'
            '2. Select a camera. Face the pattern toward it and move closer until corners appear. '
            'Start at waist/chest height with an upward tilt; camera height is not required.\n\n'
            '3. Hold still, then Space saves training. Move between captures: vary image position, tilt and distance. '
            'Auto-capture saves new steady poses in the selected Training or Validation role.\n\n'
            '4. Share several poses between neighbouring cameras and across the volume. '
            'All cameras need to belong to one connected group; they need not all see the board at once.\n\n'
            '5. Collect varied views for every camera. The displayed minimum is only a starting point. '
            'Reserve different poses with V, visible in two or more cameras.\n\n'
            '6. Press C to calibrate. Inspect validation errors and missing predictions. '
            'Low pixel errors do not certify millimetre accuracy.')

    def inspect_target(self):
        if self.last_packet:
            progress = capture_progress(self.last_packet['coverage'],
                                        self.last_packet.get('validation_views', {}), self.seed is not None)
            if progress['target']:
                self.select(progress['target'])

    def select(self, serial, automatic=False):
        if not automatic:
            self.auto_focus.setChecked(False)
        self.selected = serial
        self.quad_serials = []
        self.focus_changed_at = time.monotonic()
        for s, tile in self.tiles.items():
            tile.active = s == serial
            tile.update()
        if self.last_packet:
            self.update_inspection(self.last_packet)

    def update_inspection(self, packet):
        if not self.quad_serials:
            self.quad_serials = [s for s, _ in inspection_group(packet['coverage'], packet['observations'], self.selected)]
            self.group_changed_at = time.monotonic()
        if self.four_up.isChecked():
            self.update_quad(packet)
        serial = self.selected
        observation = packet['observations'].get(serial)
        frame = packet['batch'].frames.get(serial)
        if frame is None or observation is None:
            self.inspection.image = None
            self.inspection_info.setText('No analyzed frame for this camera.')
            self.inspection_advice.setText(inspection_advice(None, False))
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
        self.inspection_info.setVisible(not self.four_up.isChecked())
        self.inspection_advice.setText(inspection_advice(observation, packet.get('novel', {}).get(serial, True)))
        self.inspection_advice.setVisible(not self.four_up.isChecked())
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
        if self.seed is not None and not self.seed_checked:
            self.fail('Seed has not passed camera identity and image-geometry checks.')
            self.connection.close()
            child.close()
            self.connection = None
            return
        self.process = context.Process(target=solve_process, args=(child, str(session.directory / 'board.yaml'), session.serials, records, self.seed))
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
                    state = ('Seed evaluated · unchanged' if value['solver'].get('mode') == 'validation_only' else
                             ('Converged' if value['solver']['success'] else 'Provisional: iteration limit / solver issue'))
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
        self.rig.setVisible(self.result is not None)
        running = self.engine is not None and self.engine.running()
        ready = running and self.engine.session is not None and not self.engine.stop_event.is_set()
        if self.pending_session_action is not None and not running and not self.closing:
            configure, self.pending_session_action = self.pending_session_action, None
            try:
                configure()
                self.pause_button.setChecked(False)
                self.count_label.setText('0 training · 0 validation')
                self.solve_status.setText('Collect varied poses, or validation poses to check a loaded calibration.')
                self.toggle_capture()
            except Exception as exc:
                self.fail(f'Cannot switch session: {exc}')
            return
        switching = self.pending_session_action is not None
        self.start_button.setEnabled(not switching)
        self.new_button.setEnabled(not switching and self.process is None)
        self.load_session_button.setEnabled(not switching and self.process is None)
        self.seed_button.setEnabled(not switching and self.process is None)
        self.pause_button.setEnabled(ready and not switching)
        self.save_button.setEnabled(self.result is not None and self.process is None and not switching)
        self.start_button.setText('Disconnect cameras' if running else 'Connect cameras')
        for item in (self.source_choice, self.capture_mode, self.count, self.serials, self.board_button, self.exposure, self.gain):
            item.setEnabled(not running and self.process is None)
        self.training_button.setEnabled(ready and not self.pause_button.isChecked() and not switching)
        self.validation_button.setEnabled(ready and not self.pause_button.isChecked() and not switching)
        self.solve_button.setEnabled(self.process is None and self.engine is not None and self.engine.session is not None)
        if self.closing and not running:
            self.close()
            return
        if self.engine is None:
            return
        state = self.engine.snapshot()
        self.notify_capture(state.get('capture_event'))
        batch_for_audio = state.get('batch')
        blocked = bool(state['error']) or bool(batch_for_audio and batch_for_audio.problems(capture_mode=self.engine.capture_mode))
        audible = self.attention_cue.update(blocked, time.monotonic(),
            active=not self.closing and not switching and not state['paused'] and (ready or bool(state['error'])),
            fatal=bool(state['error']))
        if audible and self.sounds.isChecked():
            self.capture_sounds.play('attention')
        hint = self.guidance_cue.update((state.get('packet') or {}).get('guidance_cue'), time.monotonic(),
            active=ready and not blocked and not switching and not state['paused'] and not self.closing
                   and time.monotonic() - self.last_capture_sound_at >= 3.)
        if hint and self.sounds.isChecked():
            self.capture_sounds.play(hint)
        if state['error']:
            self.fail(state['error'])
        session = state['session']
        if session:
            samples = list(session.samples)
            training = sum(r['role'] == 'training' for r in samples)
            self.count_label.setText(f'{training} training · {len(samples)-training} validation')
            self.session_label.setText(f'Session · {session.directory.name}')
            self.session_label.setToolTip(str(session.directory.resolve()))
            self.saved_label.setText(state['status'] if samples else 'No poses saved yet.')
        batch, packet = state['batch'], state['packet']
        if packet is None:
            self.guidance.setText(state['status'])
        if self.seed is not None and not self.seed_checked and batch and len(batch.frames) == len(batch.serials):
            try:
                validate_seed_geometry(self.seed, batch.serials, {s: f.metadata()['image_size'] for s, f in batch.frames.items()})
                for frame in batch.frames.values():
                    if any(frame.settings.get(k, 0) for k in ('offset_x', 'offset_y', 'reverse_x', 'reverse_y')):
                        raise ValueError('Seed needs native unrotated images with zero ROI offset and no sensor reversal')
                if session:
                    (session.directory / 'bootstrap.json').write_text(json.dumps(self.seed, indent=2, allow_nan=False))
                self.seed_checked = True
            except (ValueError, OSError) as exc:
                self.fail(str(exc))
                self.result = self.rig.result = None
                self.engine.stop()
                return
        if batch and batch is not self.last_batch:
            for index, serial in enumerate(batch.serials):
                if serial not in self.tiles:
                    tile = CameraTile(serial)
                    tile.setMinimumSize(100, 125)
                    tile.selected.connect(self.select)
                    self.tiles[serial] = tile
                tile = self.tiles[serial]
                frame = batch.frames.get(serial)
                tile.missing = frame is None
                if frame:
                    tile.image = as_image(frame.image, 360)
                    if packet is None:
                        tile.detail = 'Waiting for detection'
                else:
                    tile.detail = 'MISSING FRAME'
                tile.update()
            self.reflow_wall()
            if self.selected is None:
                self.select(batch.serials[0], automatic=True)
            spread = batch.spread_us
            self.wall_title.setText(f'CAMERA WALL · {len(batch.frames)}/{len(batch.serials)} · ' + (f'{spread:.1f} µs start spread' if spread is not None else 'single camera'))
            self.wall_title.setToolTip('\n'.join(batch.timing_issues()) or 'No motion-timing warnings')
            self.last_batch = batch
        if packet and packet is not self.last_packet:
            self.prediction = None
            if self.result:
                try:
                    self.prediction = live_projection(self.board, packet, self.result)
                except cv2.error:
                    pass
            self.rig.target = self.prediction
            self.rig.validation_views = packet.get('validation_views', {})
            self.rig.observations = packet['observations']
            self.rig.update()
            self.update_focus(packet)
            self.update_inspection(packet)
            pending_text = f"{state['pending'].capitalize()} queued: " if state['pending'] else ''
            self.guidance.setText(('PAUSED — previews continue; captures stay saved. ' if state['paused'] else pending_text) + packet['guidance'])
            progress = capture_progress(packet['coverage'], packet.get('validation_views', {}), self.seed is not None)
            self.progress_label.setText(
                f"Views ≥{progress['minimum']}: {progress['ready']}/{progress['total']} · "
                f"Groups: {len(progress['groups'])} · Validation: {progress['validation_cameras']}/{progress['total']}")
            self.progress_label.setToolTip('Capture planning only: the solver also checks pose estimates and connectivity. '
                'Validation needs another camera to predict held-out corners.\nGroups: ' +
                ' | '.join(', '.join(group) for group in progress['groups']))
            visible = sum(bool(o['usable']) for o in packet['observations'].values())
            still = '● Steady' if packet['stationary'] else '○ Hold still'
            self.next_step.setText(f'{visible} detecting · {still}')
            self.next_step.setToolTip(progress['action'])
            self.target_button.setToolTip(progress['action'])
            self.target_button.setEnabled(bool(progress['target']))
            for serial, tile in self.tiles.items():
                observation = packet['observations'].get(serial)
                if observation is not None:
                    tile.detail = f"{len(observation['ids'])}/{self.board.num_points} · {packet['coverage']['views'][serial]} poses"
                    tile.setToolTip(inspection_advice(observation, packet.get('novel', {}).get(serial, True)))
                    tile.update()
            serials = packet['batch'].serials
            coverage = packet['coverage']
            self.metrics.setRowCount(len(serials))
            for row, serial in enumerate(serials):
                fraction = np.count_nonzero(coverage['cells'][serial]) / (GRID[0]*GRID[1])
                train = self.result.get('training', {}).get(serial, {}).get('rms') if self.result else None
                test = self.result.get('validation', {}).get(serial) if self.result else None
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
            self.inspection_title.setText(f'INSPECT · {self.selected} · {age:.1f}s')
        if self.pause_button.isChecked():
            self.guidance.setText('PAUSED — previews and detection continue. Resume when ready; an in-progress save may finish.')
        if switching:
            self.guidance.setText('Switching session…')
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
