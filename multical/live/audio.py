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
            for role in ('training', 'validation', 'attention', 'milestone', 'hold', 'tilt', 'single'):
                effect = QtMultimedia.QSoundEffect(self)
                effect.setSource(QtCore.QUrl.fromLocalFile(str(Path(__file__).with_name('sounds') / f'{role}.wav')))
                effect.setVolume(.45)
                self.effects[role] = effect
        except (ImportError, RuntimeError) as exc:
            self.error = str(exc)

    def play(self, role):
        effect = self.effects.get(role)
        if effect is not None:
            self.stop()
            effect.play()

    def stop(self):
        for effect in self.effects.values():
            effect.stop()


class AttentionCue:
    """One cue per sustained failure, with recovery debounce and a global cooldown."""
    def __init__(self):
        self.since = None
        self.healthy_since = None
        self.notified = False
        self.last_cue = float('-inf')

    def update(self, blocked, now, active=True, fatal=False):
        if not active:
            self.since = self.healthy_since = None
            self.notified = False
            return False
        if not blocked:
            if self.healthy_since is None:
                self.healthy_since = now
            if now - self.healthy_since >= 1.:
                self.since = None
                self.notified = False
            return False
        self.healthy_since = None
        if self.since is None:
            self.since = now
        if not self.notified and (fatal or now-self.since >= 1.) and now-self.last_cue >= 10.:
            self.notified = True
            self.last_cue = now
            return True
        return False


class GuidanceCue:
    """Debounce operator hints and keep them quieter than capture confirmations."""
    def __init__(self):
        self.candidate = None
        self.since = 0.
        self.announced = None
        self.last_cue = float('-inf')

    def update(self, kind, now, active=True):
        if not active or kind is None:
            self.candidate = self.announced = None
            return None
        if kind != self.candidate:
            self.candidate, self.since = kind, now
        if now-self.since >= 2. and kind != self.announced and now-self.last_cue >= 12.:
            self.announced, self.last_cue = kind, now
            return kind
        return None
