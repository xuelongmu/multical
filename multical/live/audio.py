"""Short, nonblocking confirmation sounds for committed captures."""
from pathlib import Path
from qtpy import QtCore


class CaptureSounds(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.effects = {}
        self.error = None
        try:
            from qtpy import QtMultimedia
            for role in ('training', 'validation'):
                effect = QtMultimedia.QSoundEffect(self)
                effect.setSource(QtCore.QUrl.fromLocalFile(str(Path(__file__).with_name('sounds') / f'{role}.wav')))
                effect.setVolume(.45)
                self.effects[role] = effect
        except (ImportError, RuntimeError) as exc:
            self.error = str(exc)

    def play(self, role):
        effect = self.effects.get(role)
        if effect is not None:
            effect.play()

    def stop(self):
        for effect in self.effects.values():
            effect.stop()
