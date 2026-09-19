import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from multical.live.ui import LiveWindow


class CaptureAudioTests(unittest.TestCase):
    def test_only_new_committed_events_sound_and_muted_events_are_not_replayed(self):
        window = SimpleNamespace(last_sound_event=None, capture_sounds=Mock(), sounds=Mock())
        window.sounds.isChecked.return_value = True
        first = dict(session='A', id='000000', role='training')
        LiveWindow.notify_capture(window, None)
        LiveWindow.notify_capture(window, first)
        LiveWindow.notify_capture(window, first)
        window.capture_sounds.play.assert_called_once_with('training')
        window.sounds.isChecked.return_value = False
        second = dict(session='A', id='000001', role='validation')
        LiveWindow.notify_capture(window, second)
        window.sounds.isChecked.return_value = True
        LiveWindow.notify_capture(window, second)
        self.assertEqual(window.capture_sounds.play.call_count, 1)
        LiveWindow.notify_capture(window, dict(session='B', id='000000', role='validation'))
        window.capture_sounds.play.assert_called_with('validation')
        self.assertEqual(window.capture_sounds.play.call_count, 2)
