# Captury → multical → 4C4D

Captury calibration can bootstrap both the live visualizer and multical's optimizer. It supplies the lens model and the fixed camera transforms; no Captury process needs to run after that one-time transfer. The implementation is in [captury.py](multical/io/captury.py:11), [interop.py](multical/io/interop.py:21), and the seeded [live solver](multical/live/calibration.py:105). Captury is an optional starting calibration, not an acquisition dependency.

This investigation used the installed Captury 269 format specification, two actual `.calib` files, multical's import/initialization/export code, and `/home/zerospace/4C4D` at `021000f`. This is **4C4D: 4 Camera 4D Gaussian Splatting**, whose documented interchange is COLMAP cameras, poses, image sequences, and scene points. See the [official repository](https://github.com/yangzf-1023/4C4D#2-data-preparation) and [local README](../../../4C4D/README.md:64). A complete new 25-camera calibration is unnecessary to inspect this format; the existing files are format samples, not an assumed current rig calibration.

**The actual Captury format.** The header is `tc camera calibration v0.3`. It is readable text: camera sections contain frame sections, and comments start with `#`. Multiple calibration frames are legal, including animated cameras. The reader requires explicit frame selection when a camera has multiple frames. These rules are documented in [fileformats.html:161](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:161), [fileformats.html:173](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:173), and implemented in [captury.py:24](multical/io/captury.py:24).

| Captury field | Meaning and multical mapping |
|---|---|
| `camera <index> <name>` | Captury identity/display name; in the samples the name is a MAC address. Do not use its positional index as a live camera identity. |
| `serialNumber` | Key for multical cameras and direct PySpin matching. Both sample files include this field. |
| `sensorSize`, `focalLength`, `pixelAspect`, `centerOffset` | Physical lens/sensor values used to construct pixel-space `K`. |
| `distortionModel OpenCV`, `distortion` | OpenCV coefficient vector; the samples have `[k1,k2,p1,p2,k3]`. Copy the vector without reordering. Other Captury lens models require a different adapter and are rejected. |
| `origin` | Camera centre in world coordinates, in **millimetres**, not world-to-camera translation. |
| `right`, `up` | Camera basis vectors expressed in world coordinates. Captury's world vertical is Y. |
| `orientation` | EXIF view orientation. Retained as metadata; this adapter targets native unrotated sensor images. |
| `colorCorrection` | RGB lookup tables, not geometric distortion. Not applied by the calibration adapter. |

Field definitions: [fileformats.html:165](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:165), [fileformats.html:177](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:177). Serial/model and actual distortion values: [seungki_xuelong_test_cal.calib:7](/home/zerospace/seungki_xuelong_test_cal.calib:7). Stock multical expects pixel intrinsics, distortion, image dimensions and `R,T`: [import_calib.py:14](multical/io/import_calib.py:14).

Let `W,H` be native pixel dimensions, `sw,sh` sensor dimensions in mm, `f` focal length in mm, `a` pixelAspect, and `ox,oy` centre offsets in mm. The mapping is:

```text
fx = W * f / sw
fy = W * f * a / sw
cx = W * (0.5 + ox / sw)
cy = W * (sh / 2 + oy) / sw

R = rows(right, -up, cross(right, -up))
C_metres = origin / 1000
t_metres = -R @ C_metres
X_camera = R @ X_world_metres + t_metres
```

Captury's commented intrinsic matrices normalize **both axes by image width**. Multiplying the second row by image height would be wrong. The multiplier on `fy` follows the vendor's numerical example with **non-unit** pixelAspect: `0.624420881 × 1.001912594 = 0.625615120`. This is a derivation checked against the reference matrices, not an assumption from the field name. See [fileformats.html:137](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:137), [fileformats.html:150](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:150), and [captury.py:85](multical/io/captury.py:85). Resolution is supplied explicitly because the sample format has no pixel-width/height fields. Do not infer resolution from physical sensor size.

For serial `56826091` in the September sample, at 1440×1080, the converted values are:

```text
K = [[1606.904065, 0, 676.792489],
     [0, 1606.904065, 590.113081],
     [0, 0, 1]]
C = [4.9477725, 2.0886904, -4.2725195] metres
```

Inputs and reference matrices: [seungki_xuelong_test_cal.calib:10](/home/zerospace/seungki_xuelong_test_cal.calib:10). Across both samples (18 camera entries), the conversion matched the commented rotations within `1.6e-7` per element, translations within `0.0027 mm`, and width-normalized intrinsics within `4.4e-8` (about `0.000063 px` at width 1440). These numbers measure **format-conversion agreement**, not physical calibration accuracy. Tiny rotation serialization errors are orthonormalized before use: [captury.py:90](multical/io/captury.py:90).

**Use the calibration before refining it.** Load the seed, display its camera centres immediately, then use direct PySpin images for board detection and cross-camera predicted-corner overlays. First collect validation-only poses and run the check: the seed stays unchanged. After that, separate training poses can refine camera extrinsics with the imported intrinsics fixed. The result records both the seed's and the refined solution's errors on the same validation captures. Code: [ui.py:253](multical/live/ui.py:253), [calibration.py:113](multical/live/calibration.py:113), [calibration.py:203](multical/live/calibration.py:203).

The new branch's commands, from this worktree, are:

```bash
# Native sensor images: 1440 wide, 1080 high; output must not already exist.
../../.venv/bin/python -m multical.io.interop captury \
  /path/to/current.calib seed.json --image-size 1440 1080

# Loads geometry without connecting until the operator presses Connect cameras.
./run-live --count 25 --seed seed.json

# Camera model export; this alone is not a training dataset.
../../.venv/bin/python -m multical.io.interop colmap seed.json export-4c4d

# Optional image conversion: rgb/<serial>/0000.png, 0001.png, ...
../../.venv/bin/python -m multical.io.interop colmap seed.json export-with-images \
  --image-root /path/to/rgb
```

The loader also accepts live calibration JSON. Stock multical JSON has no unit field; the export command requires `--units metres` when the operator knows that its board units were metres. Converted Captury files declare their units and use **absolute camera keys**, which preserves the Captury world frame. The GUI accepts declared-metre JSON and stops on roster/resolution/ROI/reversal mismatch. It records the seed alongside new captures. See [interop.py:21](multical/io/interop.py:21), [ui.py:647](multical/live/ui.py:647). Generated samples are under `live-sessions/captury-bootstrap/`; they are ignored by Git.

Seeded refinement fixes the first connected camera at its **imported**, potentially nonidentity pose, and fixes the board geometry. Board poses initialize from `inverse(C_camera) @ board_PnP`. The seed does not make disconnected new observations observable: refinement still requires a connected co-visibility graph. The minimum of three usable seeded views per camera is only an execution floor, not a quality guarantee. Validation-only checks have no such training requirement. See [calibration.py:134](multical/live/calibration.py:134), [calibration.py:151](multical/live/calibration.py:151), [calibration.py:165](multical/live/calibration.py:165).

**4C4D compatibility is an explicit conversion.** Its loader accepts undistorted `PINHOLE` or `SIMPLE_PINHOLE` cameras, and transposes the COLMAP rotation for internal storage. COLMAP and multical both represent world-to-camera poses with camera axes right/down/forward. The exporter converts `R` to Hamilton quaternion `qw,qx,qy,qz` and retains `t` in metres. Sources: [COLMAP format](https://colmap.github.io/format.html#images-txt), [4C4D dataset_readers.py:99](/home/zerospace/4C4D/scene/dataset_readers.py:99), [interop.py:129](multical/io/interop.py:129). The actual local 4C4D COLMAP parser successfully read all ten converted cameras and reproduced their poses.

There are several consequential downstream details:

- **Principal points are dropped on 4C4D's COLMAP path.** It reads focal lengths, but constructs `CameraInfo` without `cx,cy,fl_x,fl_y`; those keep their defaults. The renderer then chooses a symmetric projection. See [dataset_readers.py:103](/home/zerospace/4C4D/scene/dataset_readers.py:103), [dataset_readers.py:123](/home/zerospace/4C4D/scene/dataset_readers.py:123), [cameras.py:67](/home/zerospace/4C4D/scene/cameras.py:67). The sample above differs from that centre by roughly **43 px horizontally and 51 px vertically**. Simply copying its original K into a COLMAP file does not preserve its projection.
- **There is also a half-pixel convention.** The CUDA rasterizer maps NDC zero to `(W−1)/2,(H−1)/2`, as shown in [auxiliary.h:42](/home/zerospace/4C4D/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:42) and used in [forward.cu:466](/home/zerospace/4C4D/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:466). The adapter therefore undistorts and recentres images to **719.5,539.5** at 1440×1080, retaining the focal lengths. It writes that same K into COLMAP. The alternative is a coordinated loader/projection fix in 4C4D. The existing Depthkit converter uses an unconstrained optimal principal point, so it needs the same scrutiny: [convert_depthkit_to_4c4d.py:323](/home/zerospace/4C4D/preprocessing/depthkit/convert_depthkit_to_4c4d.py:323).
- **Lens coefficients cannot just be discarded.** `initUndistortRectifyMap` and `remap` transform the images to the exported camera model. Changing K without remapping images is wrong. Black borders can remain; further cropping/resizing must transform K consistently. Code and projection-equivalence test: [interop.py:74](multical/io/interop.py:74), [test_interop.py:102](tests/test_interop.py:102).
- **Calibration is not scene initialization.** 4C4D also reads `points3D.*`; multical board points are not a human/scene point cloud. Reconstruct scene points from training views using the fixed calibrated poses, or align an independently reconstructed cloud to this same frame and scale. Do not combine a MASt3R cloud in an unrelated gauge with the metric rig. See [dataset_readers.py:293](/home/zerospace/4C4D/scene/dataset_readers.py:293) and the [4C4D data preparation instructions](https://github.com/yangzf-1023/4C4D#2-data-preparation).
- **Time is inferred from filenames.** This loader reads the final four digits and maps the sequence to a roughly ten-unit interval, rather than reading hardware timestamps. Export contiguous synchronized `camXX_YYYY.png` sequences with at most 10,000 frames per chunk, and select the training time configuration deliberately. Equal file counts alone cannot establish synchronization. See [dataset_readers.py:183](/home/zerospace/4C4D/scene/dataset_readers.py:183), [dataset_readers.py:204](/home/zerospace/4C4D/scene/dataset_readers.py:204).
- **Choose training/test cameras explicitly.** The current reader's default training set uses specific N3V camera names. It can load more than four cameras, but the intended experiment's view selection and memory budget still need configuration: [dataset_readers.py:255](/home/zerospace/4C4D/scene/dataset_readers.py:255), [dataset_readers.py:280](/home/zerospace/4C4D/scene/dataset_readers.py:280). Export preserves all cameras and writes a serial-to-`camXX` mapping.

**Library gotchas that affect bootstrapping.** Stock multical already accepts `--calibration` and skips intrinsic fitting when given one, but its optimizer defaults to adjusting intrinsics unless `fix_intrinsic` is set: [config/workspace.py:26](multical/config/workspace.py:26), [config/workspace.py:49](multical/config/workspace.py:49), [config/arguments.py:65](multical/config/arguments.py:65). Its pose initializer estimates the camera tree before replacing it with the supplied seed: [tables.py:354](multical/tables.py:354). Its exporter can rebase the world frame to a master camera: [export_calib.py:81](multical/io/export_calib.py:81). None of these behaviors should be silently assumed to preserve Captury's floor frame and fixed lens parameters; the live seeded path handles those choices explicitly.

Captury's EXIF orientation must not be applied twice. The adapter matches the native reference matrices, including sample cameras with orientation 3; a Captury-exported movie might already be rotated or processed. For the direct capture path, keep the native sensor image geometry and verify the board overlay. The format stores no exposure, synchronization evidence, per-observation residual history, or parameter covariance sufficient to establish current metric accuracy; historical `intrinsicError` is not a current acceptance test. Geometric seed reuse also does not reproduce Captury's RGB color correction. Relevant fields: [fileformats.html:169](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:169), [fileformats.html:186](/home/zerospace/Downloads/CapturyLive-ubu22-269/doc/fileformats.html:186), [seungki_xuelong_test_cal.calib:28](/home/zerospace/seungki_xuelong_test_cal.calib:28).

Validation performed: 31 unit/integration tests, both actual calibration files against their embedded reference matrices, actual 4C4D camera/pose text parsing, a headless UI load of the Captury rig, and a seeded simulated live UI check confirming identity checks, predicted corners, and seed persistence. Tests cover nontrivial rotations, millimetre conversion, animated-frame selection, identity/model failures, undistortion/projection agreement including pixel centres, RGB rectification, validation without refitting, and seeded refinement preserving the world anchor and intrinsics. No hardware acquisition or 4C4D training was run. The current live calibration capture retains **grayscale** images; a production 4DGS recorder still needs synchronized RGB sequences and scene-point initialization. See [sources.py:258](multical/live/sources.py:258), [test_interop.py](tests/test_interop.py), [test_live.py:281](tests/test_live.py:281).
