Multical Live is a standalone Qt interface for direct FLIR acquisition, visual board feedback, retained coverage, and background calibration. It does not require Captury, exported video, or an image import step.

![Live simulation after calibration, with camera previews, detected and predicted corners, coverage, rig geometry, and validation residuals](screenshots/live_calibration.png)

Run from this worktree:

```bash
./run-live --demo --auto-capture
```

The simulator renders ChArUco images through the real detector and changes its held pose every 1.8 seconds. It is explicitly labelled in the UI, session manifest, and calibration output. It is for exercising acquisition, detection, retention, visualization, and fitting; its results are not hardware validation.

An existing Captury calibration can seed the live geometry and lens parameters. Convert its `.calib` file once, then pass `--seed seed.json` or use **Load calibration seed…** before connecting. With only validation captures, **Calibrate captures** checks the seed without changing it; with training captures it refines extrinsics while retaining the imported lenses and world anchor. The complete format mapping, commands, validation evidence, and COLMAP/4C4D conversion requirements are in [CALIBRATION_INTEROP.md](CALIBRATION_INTEROP.md). Direct acquisition remains independent of Captury.

For the physical rig, close applications holding the cameras and run:

```bash
./run-live --count 25 --boards example_boards/charuco_36x54.yaml
```

**Stationary board** is the default capture mode. Hold the board still during each capture; different fixed exposure times are allowed. Leave exposure/gain overrides blank to preserve each camera's settings. If needed, enter overrides before clicking **Connect cameras**. Acquisition freezes exposure/gain auto modes, uses PTP and scheduled Action0 commands on each populated interface, and restores readable changed settings when disconnected. `ActionDeviceKey` is write-only on this fleet: its previous value cannot be backed up, so the adapter uses the documented Captury/CaptureNet key 42 and leaves that key in place. The interface does not provision PTP or change network/packet settings; those remain the job of the existing CaptureNet setup.

An explicit expected roster is preferable once camera serials are known:

```bash
./run-live --count 3 --serials SERIAL_A SERIAL_B SERIAL_C
```

With only a count, discovery must find exactly that number and the discovered roster is frozen in the session. With serials, every requested serial must be present; other discovered cameras are not acquired. Both modes reject missing/incomplete images, repeated IDs, invalid image data, and stale frames that miss the scheduled action by more than 1 ms. Exposure is subtracted from the camera's documented end-of-exposure timestamp when checking frame freshness.

In **Stationary board** mode, exposure-length differences, microsecond start/midpoint spread, and PTP clock-quality warnings do not block connection or capture. Timing diagnostics remain available in the camera-wall tooltip and are saved with retained captures. They do not replace board guidance with clock warnings. Manual capture relies on the operator holding the board still; auto-capture retains its existing image-motion check. Sessions and captures record the selected mode.

Select **Moving board · strict timing**, or pass `--capture-mode motion`, to restore the stricter timing policy: require PTP Slave within 1000 ns at connection, and reject captures with PTP warnings or start/midpoint spread above 20 µs. During acquisition, clock-quality warnings block retention while previews continue. SDK communication failures still stop acquisition in either mode. These are acquisition tolerances, not metrology acceptance thresholds.

The launcher uses this checkout's virtualenv, or the main worktree's virtualenv. The existing local virtualenv includes Qt, OpenCV, and SciPy. On another installation, install the package's `live` extra and the vendor's matching Spinnaker Python SDK. `multical-live` is also registered as a package entry point. PySpin is imported only when hardware acquisition starts. A user-installed PySpin is discovered automatically; alternatively pass `--sdk-path /path/to/site-packages`. The SDK must match the running Python ABI. The hardware adapter uses the installed SDK's `ImageProcessor` conversion API.

Hardware bring-up on September 18, 2026 connected all 25 cameras and displayed their native grayscale previews on the desktop. It exposed and fixed the write-only action-key access and the installed SDK's `SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR` constant. With `--exposure-us 1400`, read-back exposures were 1397–1399 µs, observed start spread was approximately 6–7 µs, complete sets contained all 25 frames, and the ChArUco detector reported a usable view. Actual acquisition was approximately 2–2.3 sets/s with a requested 5 Hz; throughput still needs profiling. The initial strict policy rejected intermittent master-offset excursions. The operator then confirmed the board would be held still, so stationary mode became the default and the shared-exposure override was removed. No calibration solve or accuracy acceptance was performed. The automated suite passed 37 tests, including stationary-versus-motion acceptance, retained warning metadata, frame-integrity checks, write-only node access, and PTP-warning recovery.

**Use the visualizer to drive collection.** The camera wall displays the newest acquired frames. Select a camera to inspect a full-resolution detection result, with corner IDs and an analysis-age label. The inspection image and overlays always come from the same frame. Scroll to zoom, drag to pan, and double-click to reset. The green grid counts distinct retained training poses per image cell; repeated stationary frames do not inflate the counts. The overlap matrix counts shared retained poses when at least one camera adds novel image geometry. This is a geometric-diversity heuristic, not an information matrix.

Capture training poses with **Space**, and reserve independent validation poses with **V**. Auto-capture retains only training poses that add projected-board geometry and have consecutive detections moving less than one pixel. Manual capture can retain repeated training observations; it never bypasses frame-integrity, the selected timing policy, or board-usability checks. A queued validation capture waits until the board leaves previously retained training poses. Training also avoids reserved validation poses. This image-space duplicate check reduces leakage but does not replace an independently captured validation session or surveyed reference.

Click **Calibrate captures** or press **C** after collecting at least 12 usable views per camera. The solve runs in a separate process while previews and collection continue. It uses a frozen snapshot: later captures belong to the next solve. You can cancel the process without disconnecting cameras. The fit:

- Fits standard intrinsics from the selected training images without residual-based view deletion or a relaxed RMS target.
- Requires a connected camera co-visibility graph and initialization for every training frame and camera.
- Fixes the first camera pose and the single board transform to remove coordinate gauges.
- Optimizes camera and frame poses with fixed intrinsics and board geometry, using a soft-L1 loss and no corner deletion.
- Reports residuals over all retained training detections, including the robustly downweighted ones.
- Evaluates validation views by estimating board pose in a different camera and predicting the withheld camera, with no validation residual filtering. Missing predictions count as failed points.

The relative rig view appears after a solve. Drag to orbit and scroll to zoom. Camera centres and optical directions are in metres in the first camera's coordinate frame; the grid is a coordinate aid, not a surveyed ground plane. Blue points are training target origins. The live target and amber predicted corners are estimated using the currently best-covered reference camera. The reference camera's own residual is a pose-fit diagnostic, not independent validation. Other cameras are predicted from that reference. The metrics table separately displays the frozen solve's training and validation residuals; `Test n` is evaluated / expected detected points. New live coverage can exceed the solve snapshot's coverage.

Results remain **metric accuracy unverified**, even if the optimizer converges and pixel errors are low. The UI currently implements acquisition and image-consistency diagnostics. It does not implement surveyed scale checks, a physical-board uncertainty budget, full geometric conditioning/covariance, or certification of sub-millimetre camera positions. Twelve views is an execution floor, not a sufficiency claim. The actual 25-camera rig needs a capture route with varied board position, depth and normal, image-edge observations, and redundant cross-volume links. Large distances and foreshortened targets can require a different measured target.

Each connection creates a new directory under `live-sessions/` (override with `--output`). It contains the exact board YAML and its SHA-256 hash, camera serials, software versions, capture roles, frame IDs/timestamps/exposure/gain/image sizes, camera ROI offsets and source pixel format, lossless grayscale PNGs, and detected corner IDs/coordinates. The grayscale images are copied from SDK buffers before those buffers are released. Capture directories are staged and the manifest is atomically replaced only after successful writes. Memory retains detections and bounded current images, rather than loading the entire image dataset.

Calibration JSON files are versioned and record exact training/validation capture IDs, intrinsics, world-to-camera transforms, frame poses, per-camera residuals/counts, solver termination diagnostics, and an unverified accuracy status. Camera centres are `-R.T @ t`. This interface's calibration JSON is a documented live-session result schema, not a drop-in replacement for every downstream calibration format. Original images and manifest remain available for reprocessing and independent checks.

The current interface supports one ChArUco board and standard pinhole distortion. Multiple independently moving boards, multi-face targets, distortion-model selection, hardware throughput qualification at 25 cameras, and external metrology acceptance remain further work. Acquisition and detection rates are separate: `--fps` controls scheduled acquisition (default 5 Hz), and `--workers` controls detector threads. If detection falls behind, intermediate preview frames are skipped for analysis instead of queued indefinitely. Disk writes can slow analysis/retention while acquisition continues.

Run the automated checks from the worktree with its Python environment:

```bash
../../.venv/bin/python -m unittest discover -s tests -v
QT_QPA_PLATFORM=offscreen ./run-live --demo --auto-capture \
  --output /tmp/multical-ui-check --run-seconds 8 --screenshot /tmp/multical-live.png
QT_QPA_PLATFORM=offscreen ../../.venv/bin/python tests/live_ui_smoke.py
```

The unit/integration suite covers dropped and stale frames, exposure midpoint mismatches, SDK buffer lifetime, graph mutation, gauge anchoring, the ChArUco grid check, empty error reporting, intrinsic-fit provenance, session persistence, a complete solve from rendered images, and a validation failure induced by perturbing camera translation. The longer UI smoke test exercises automatic collection, separate validation capture, background calibration, live preview continuity, saving, and shutdown. Hardware calls require a separate test with the real rig; simulation cannot establish device behavior or throughput.

CaptureNet-derived behavior is attributed in `multical/live/sources.py` to `/home/zerospace/capturenet/scripts/fire3.py`: scheduled Action0 keys, multi-interface broadcasts, PTP checks, and the end-of-exposure timestamp convention. No CaptureNet script is executed by this application.
