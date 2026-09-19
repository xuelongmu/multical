import unittest
import numpy as np
from multical.live.guidance import capture_progress, inspection_advice


class GuidanceTests(unittest.TestCase):
    def coverage(self, views=(12, 12, 12), edges=()):
        serials = ['A', 'B', 'C']
        overlaps = np.zeros((3, 3), int)
        for a, b in edges:
            overlaps[a, b] = overlaps[b, a] = 1
        return dict(views=dict(zip(serials, views)), overlaps=overlaps,
                    cells={s: np.zeros((9, 12), int) for s in serials})

    def test_many_views_do_not_hide_disconnected_rig(self):
        progress = capture_progress(self.coverage(edges=[(0, 1)]), {})
        self.assertEqual(progress['ready'], 3)
        self.assertEqual(len(progress['groups']), 2)
        self.assertIn('Connect', progress['action'])

    def test_transitive_connections_then_validation_then_solve(self):
        coverage = self.coverage(edges=[(0, 1), (1, 2)])
        progress = capture_progress(coverage, {'A': 2})
        self.assertEqual(len(progress['groups']), 1)
        self.assertIn('V.', progress['action'])
        progress = capture_progress(coverage, dict(A=1, B=1, C=1))
        self.assertIn('Press C', progress['action'])
        self.assertIn('do not establish accuracy', progress['action'])

    def test_seed_changes_minimum_and_target_uses_least_coverage(self):
        coverage = self.coverage(views=(3, 2, 2))
        coverage['cells']['B'][:] = 1
        progress = capture_progress(coverage, {}, seeded=True)
        self.assertEqual(progress['minimum'], 3)
        self.assertEqual(progress['ready'], 1)
        self.assertEqual(progress['target'], 'C')
        self.assertEqual(capture_progress(coverage, {})['ready'], 0)

    def test_no_detection_is_not_a_capture_instruction(self):
        self.assertIn('Waiting', inspection_advice(None, False))
        self.assertIn('move closer', inspection_advice(dict(usable=False), True))
        self.assertIn('advisory', inspection_advice(dict(usable=True, marker_px=10), True))
        self.assertIn('repeats', inspection_advice(dict(usable=True, marker_px=30), False))
        self.assertIn('Space', inspection_advice(dict(usable=True, marker_px=30), True))
