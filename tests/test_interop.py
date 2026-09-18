import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from multical.io.captury import read_captury
from multical.io.import_calib import load_calibration
from multical.io.interop import export_colmap, load_seed, pinhole_model, rectification_maps, validate_seed_geometry


def captury_text():
    # Deliberately nontrivial rotation, off-centre K, non-unit pixelAspect, and EXIF 3.
    rotation = Rotation.from_euler('xyz', [.2, -.4, 1.1]).as_matrix()
    right, up = rotation[0], -rotation[1]
    vector = lambda a: ' '.join(f'{v:.12g}' for v in a)
    return f'''tc camera calibration v0.3
camera 7 mac-address
 serialNumber SERIAL7
 cameraModel Test camera
 frame 0
 sensorSize 4.8 3.6
 focalLength 6
 pixelAspect 1.01
 centerOffset -.12 .06
 distortionModel OpenCV
 distortion -.2 .04 .001 -.002 .003
 origin 1200 -450 3400
 right {vector(right)}
 up {vector(up)}
 orientation 3
 time 123
''', rotation


class InteropTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.calib'
        self.text, self.rotation = captury_text()
        self.source.write_text(self.text)
        self.data = read_captury(self.source, (640, 480))
        self.json = self.root / 'seed.json'
        self.json.write_text(json.dumps(self.data))
        self.seed = load_seed(self.json)

    def test_native_projection_units_axes_and_stock_multical_reader_agree(self):
        loaded = load_calibration(self.json)
        camera = loaded.cameras['SERIAL7']
        pose = loaded.camera_poses['SERIAL7']
        np.testing.assert_allclose(camera.intrinsic, [[800, 0, 304], [0, 808, 248], [0, 0, 1]])
        np.testing.assert_allclose(pose[:3, :3], self.rotation, atol=1e-11)
        np.testing.assert_allclose(-pose[:3, :3].T @ pose[:3, 3], [1.2, -.45, 3.4])
        camera_points = np.array([[.1, -.2, 2.], [-.3, .4, 3.]])
        world_points = (camera_points - pose[:3, 3]) @ self.rotation
        from_world = cv2.projectPoints(world_points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], camera.intrinsic, camera.dist)[0][:, 0]
        np.testing.assert_allclose(from_world, camera.project(camera_points), atol=1e-9)
        self.assertEqual(self.data['provenance']['cameras']['SERIAL7']['orientation'], 3)

    def test_animated_file_requires_selection(self):
        frame_block = self.text[self.text.index(' frame 0'):].replace('frame 0', 'frame 2', 1)
        self.source.write_text(self.text + frame_block)
        with self.assertRaisesRegex(ValueError, 'select --frame'):
            read_captury(self.source, (640, 480))
        selected = read_captury(self.source, (640, 480), frame=2)
        self.assertEqual(selected['provenance']['cameras']['SERIAL7']['frame'], 2)

    def test_unsupported_lens_and_invalid_basis_are_not_silently_accepted(self):
        for changed in (self.text.replace('OpenCV', 'SynthEyes'), self.text.replace(' up ', ' up 100 ')):
            self.source.write_text(changed)
            with self.assertRaises(ValueError):
                read_captury(self.source, (640, 480))

    def test_duplicate_identity_is_rejected(self):
        self.source.write_text(self.text + self.text.split('\n', 1)[1])
        with self.assertRaisesRegex(ValueError, 'distinct serialNumber'):
            read_captury(self.source, (640, 480))

    def test_unknown_units_reflections_and_partial_rosters_are_rejected(self):
        data = copy.deepcopy(self.data)
        del data['units']
        self.json.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'units'):
            load_seed(self.json)
        load_seed(self.json, units='metres')
        data['units'] = 'metres'
        data['camera_poses']['SERIAL7']['R'] = (-self.rotation).tolist()
        self.json.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'proper rotation'):
            load_seed(self.json)
        with self.assertRaisesRegex(ValueError, 'roster'):
            validate_seed_geometry(self.seed, ['WRONG'], {'WRONG': [640, 480]})
        with self.assertRaisesRegex(ValueError, 'resolution'):
            validate_seed_geometry(self.seed, ['SERIAL7'], {'SERIAL7': [480, 640]})

    def test_undistortion_recentring_and_colmap_pose_roundtrip(self):
        output = self.root / 'colmap'
        manifest = export_colmap(self.seed, output)
        self.assertFalse(manifest['training_ready'])
        self.assertFalse((output / 'sparse/0/points3D.txt').exists())
        camera = self.data['cameras']['SERIAL7']
        pinhole = pinhole_model(camera)
        k = np.asarray(pinhole['K'])
        np.testing.assert_equal(k[:2, 2], [319.5, 239.5])
        maps = rectification_maps(camera, pinhole)
        # A destination pixel's source sample must project the SAME camera ray
        # through the original off-centre distorted camera, including tangential d.
        pixels = np.array([[120, 100], [320, 240], [520, 350]], dtype=float)
        rays = np.c_[(pixels - k[:2, 2]) / [k[0, 0], k[1, 1]], np.ones(len(pixels))]
        # 4C4D's symmetric projection followed by the CUDA ndc2Pix transform.
        size = np.array(camera['image_size'])
        ndc = rays[:, :2] * (2*np.array([k[0, 0], k[1, 1]]) / size)
        np.testing.assert_allclose(((ndc+1)*size-1)/2, pixels, atol=1e-12)
        expected = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), np.array(camera['K']), np.array(camera['dist']))[0][:, 0]
        actual = np.array([[maps[0][int(y), int(x)], maps[1][int(y), int(x)]] for x, y in pixels])
        np.testing.assert_allclose(actual, expected, atol=4e-5)
        lines = (output / 'sparse/0/images.txt').read_text().splitlines()
        values = lines[1].split()
        q = np.array(values[1:5], float)
        r = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
        t = np.array(values[5:8], float)
        np.testing.assert_allclose(r, self.rotation, atol=1e-11)
        np.testing.assert_allclose(-r.T @ t, [1.2, -.45, 3.4])
        self.assertEqual(lines[2], '')
        with self.assertRaisesRegex(ValueError, 'new output'):
            export_colmap(self.seed, output)

    def test_export_rectifies_rgb_and_refuses_greyscale_or_missing_frames(self):
        image_root = self.root / 'raw'
        folder = image_root / 'SERIAL7'
        folder.mkdir(parents=True)
        image = np.zeros((480, 640, 3), np.uint8)
        image[210:260, 290:340] = [10, 150, 255]
        cv2.imwrite(str(folder / '0000.png'), image)
        output = self.root / 'images-export'
        result = export_colmap(self.seed, output, image_root)
        self.assertTrue(result['images_rectified'])
        camera = self.seed['cameras']['SERIAL7']
        expected = cv2.remap(image, *rectification_maps(camera, pinhole_model(camera)), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(cv2.imread(str(output / 'images/cam00_0000.png')), expected)
        cv2.imwrite(str(folder / '0000.png'), image[:, :, 0])
        with self.assertRaisesRegex(ValueError, 'RGB/BGR'):
            export_colmap(self.seed, self.root / 'grey', image_root)
        self.assertFalse((self.root / 'grey').exists())
        (folder / '0000.png').rename(folder / '0002.png')
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            export_colmap(self.seed, self.root / 'missing', image_root)


if __name__ == '__main__':
    unittest.main()
