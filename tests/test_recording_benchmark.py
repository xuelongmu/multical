"""Protect the benchmark from misleading quality scores due to timestamp/range conversion."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from tools.recording_benchmark import quality


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for codec checks')
class QualityMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='multical-quality-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / 'reference.yuv'
        raw.write_bytes(b''.join(bytes([y]) * 256 + bytes([128]) * 128 for y in (10, 120, 245)))
        (self.root / 'dataset.json').write_text(json.dumps({
            'width': 16, 'height': 16, 'sources': [{'raw': str(raw)}]}))
        self.encoded = self.root / 'lossless.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'yuv420p',
                        '-color_range', 'pc', '-video_size', '16x16', '-framerate', '25',
                        '-i', str(raw), '-c:v', 'ffv1', '-color_range', 'pc', str(self.encoded)],
                       check=True, capture_output=True, timeout=15)

    def arguments(self, frames):
        return SimpleNamespace(dataset=self.root, encoded=self.encoded, source=0, crop=None,
                               frames=frames, output=self.root / 'quality.json')

    def test_lossless_samples_match_despite_different_input_timebases(self):
        # The reference is read at 60 Hz; the encoded clip uses 25 Hz plus MKV's
        # millisecond timebase. Compare matching frame indices, not rounded PTS.
        with redirect_stdout(io.StringIO()):
            quality(self.arguments(3))
        result = json.loads((self.root / 'quality.json').read_text())
        self.assertEqual(result['evaluated_frames'], 3)
        self.assertEqual(result['psnr_all_db'], 'inf')
        self.assertEqual(result['ssim_all'], 1.0)

    def test_incomplete_comparison_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'evaluated 3, expected 4'):
            quality(self.arguments(4))


if __name__ == '__main__':
    unittest.main()
