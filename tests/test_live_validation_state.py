import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from multical.live.validation_state import save_evaluation, restore_evaluation
from multical.live.calibration import evaluate_snapshot
from multical.live.ui import LiveWindow


class ValidationStateTests(unittest.TestCase):
    def test_cache_restores_with_new_captures_but_rejects_changed_inputs(self):
        base = dict(cameras={'A': 1}, camera_poses={'A': 2})
        result = dict(base, validation_ids=['0'], training_ids=[])
        records = [dict(id='0', role='validation', detections=[1])]
        with tempfile.TemporaryDirectory() as directory:
            board = Path(directory) / 'board.yaml'
            board.write_text('board')
            save_evaluation(directory, result, base, records)
            self.assertEqual(restore_evaluation(directory, base, records + [dict(id='1')]), result)
            with self.assertRaises(ValueError):
                restore_evaluation(directory, dict(base, cameras={'A': 3}), records)
            with self.assertRaises(ValueError):
                restore_evaluation(directory, base, [dict(records[0], detections=[2])])
            board.write_text('other board')
            with self.assertRaises(ValueError):
                restore_evaluation(directory, base, records)

    def test_evaluation_preserves_fit_and_excludes_training(self):
        base = dict(cameras={'A': 1}, camera_poses={'A': 2}, training_ids=['t'],
                    training={'A': 3}, solver={'success': True})
        original = copy.deepcopy(base)
        records = [dict(id='t', role='training'), dict(id='v', role='validation')]
        with patch('multical.live.calibration.solve_snapshot', return_value=dict(validation={'A': 4}, validation_ids=['v'])) as solve:
            result = evaluate_snapshot('board', ['A'], records, base)
        self.assertEqual(solve.call_args.args[2], [records[1]])
        self.assertEqual(base, original)
        for key in base:
            self.assertEqual(result[key], base[key])

    def test_scheduler_coalesces_and_does_not_retry_same_failed_snapshot(self):
        window = SimpleNamespace(closing=False, pending_session_action=None, process=None, engine=None,
            result=dict(cameras={}, camera_poses={}), seed=None, seed_checked=False,
            evaluation_restored=True, auto_validation_attempt=None, evaluated_key=None, solve=Mock())
        session = SimpleNamespace(directory='session', samples=[dict(id='1', role='validation')])
        LiveWindow.auto_evaluate(window, session)
        window.solve.assert_called_once_with(evaluate_only=True)
        LiveWindow.auto_evaluate(window, session)
        self.assertEqual(window.solve.call_count, 1)
        window.process = object()
        session.samples.append(dict(id='2', role='validation'))
        LiveWindow.auto_evaluate(window, session)
        self.assertEqual(window.solve.call_count, 1)
        window.process = None
        LiveWindow.auto_evaluate(window, session)
        self.assertEqual(window.solve.call_count, 2)
