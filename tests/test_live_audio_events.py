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

    def test_milestone_replaces_training_confirmation(self):
        window = SimpleNamespace(last_sound_event=None, capture_sounds=Mock(), sounds=Mock())
        window.sounds.isChecked.return_value = True
        event = dict(session='A', id='1', role='training', milestone=True)
        LiveWindow.notify_capture(window, event)
        LiveWindow.notify_capture(window, event)
        window.capture_sounds.play.assert_called_once_with('milestone')

    def test_attention_debounce_recovery_cooldown_and_pause(self):
        from multical.live.audio import AttentionCue
        cue = AttentionCue()
        self.assertFalse(cue.update(True, 0))
        self.assertTrue(cue.update(True, 1))
        self.assertFalse(cue.update(True, 40))
        cue.update(False, 41)
        cue.update(False, 42)
        self.assertFalse(cue.update(True, 43))
        self.assertTrue(cue.update(True, 44))
        self.assertFalse(cue.update(True, 45, active=False))
        self.assertFalse(cue.update(True, 46, fatal=True))
        self.assertTrue(cue.update(True, 54, fatal=True))

    def test_operator_hints_require_stability_and_do_not_repeat(self):
        from multical.live.audio import GuidanceCue
        cue = GuidanceCue()
        self.assertIsNone(cue.update('single', 0))
        self.assertEqual(cue.update('single', 2), 'single')
        self.assertIsNone(cue.update('single', 30))
        self.assertIsNone(cue.update('hold', 31))
        self.assertEqual(cue.update('hold', 33), 'hold')
        self.assertIsNone(cue.update('tilt', 34))
        self.assertIsNone(cue.update('tilt', 36))
        self.assertEqual(cue.update('tilt', 45), 'tilt')
        self.assertIsNone(cue.update('single', 70, active=False))
