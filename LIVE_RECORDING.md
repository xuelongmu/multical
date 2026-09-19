# Recording with Multical Live

Connect using **FLIR cameras · native recorder**. Close Captury before connecting.
The **● Record** button above the camera wall (shortcut **R**) starts a take; it
becomes **■ Stop**. Stopping early saves all already-scheduled frames and then
flushes the files. Wait for **Saved** before unplugging recording storage.

The adjacent gear menu selects frame rate, maximum duration, JPEG quality, output
folder, and whether previews pause automatically. Defaults: **60 fps, 120 seconds,
quality 85, previews paused**. The current limit is 25 cameras at native
1440 × 1080. Preview rate (`--fps`, default 5 Hz, native maximum 10 Hz) is separate
from recording rate. At recording time, preview acquisition is sampled at about
2 Hz; **P** controls rendering independently. Analysis, pose collection, and
calibration solving are suspended during the take. Existing calibration/session
data are retained, and previews resume after saving if the recorder paused them.

Each camera gets its own colour **MJPEG / Matroska** stream split into two-second
segments. Every frame is independently decodable; there are no P/B frames. This
uses CUDA NPP to debayer BayerRG8 and four nvJPEG workers on the GPU, not NVENC.
Quality 85 is lossy 4:2:0 JPEG, not RAW or lossless. No additional software white
balance, gamma, or colour matrix is applied. Frozen camera ISP settings are
recorded; check exposure and colour visually before important takes.

The native process owns acquisition, scheduling, encoders and disk writing.
Python reads replaceable shared-memory previews. Slow preview rendering cannot
block the recording queue. The recorder keeps 512 preallocated Bayer buffers
(about 0.8 GB), 64 SDK buffers per camera, and a compressed queue capped at both
8,192 frames and 1 GiB. Missing, repeated, incomplete, mistimed or unwritable
frames fail the take; the recorder never inserts replacement images to conceal
a gap. Queue overflow stops further scheduling and retains available frames.

## Saved files

```
take-<UTC>-<unique id>/
  context.json             # serial roster, session reference, calibration checksum
  calibration.json         # copy of current calibration, if one is loaded
  take.json                # complete flag, settings, counts, timing and errors
  cameras/<serial>/
    frames.csv             # one row per saved real frame
    segment_00000.mkv
    segment_00000.mkv.json  # finalized segment receipt
    ...
```

CSV records global take frame index, scheduled PTP time, exposure-end timestamp,
derived exposure start/midpoint, hardware frame ID, segment, packet index and JPEG
size. Hardware nanoseconds are authoritative: MKV playback timestamps have
millisecond precision and restart at zero in each segment. Frame IDs need not
match between cameras; their scheduled timestamps must match.

Completed segments and their ledger entries are flushed, followed by a durable
segment receipt. The final take manifest is replaced atomically only after all
videos and ledgers finish flushing. **Saved** means all scheduled frames were
received/encoded/written and files flushed; it does not mean a second decoder has
verified the media or that calibration accuracy passed a threshold.

A power cut/process kill can leave an unfinished current segment and a take still
marked `arming`. Retain the directory: finalized segments, receipts and ledger
entries are available for recovery. There is no automatic interrupted-take repair
or automatic retry that overwrites previous takes. A failed take is never suitable
for strict synchronized ingestion without examining its explicit frame gaps.

Verify a completed take independently (off the UI thread):

```bash
python -m multical.live.recording_verify /path/to/take
```

This fully decodes every segment and checks independent frames, geometry,
presentation timestamps, per-camera counts/order, shared schedule, exposure
arithmetic, file sizes and the calibration checksum. `--metadata-only` skips
video decoding and is a weaker check. `--workers 4` controls decode concurrency.

PTP must be Slave within 20 µs at arming. Exposure must fit within the selected
frame interval. Existing exposure/gain values are preserved unless overridden
when connecting. The UI marks **timing review** if observed start or midpoint
spread exceeds 20 µs; that is a diagnostic, not a universal motion-accuracy gate.
Different exposures can align starts while shifting exposure midpoints. A saved
take with that warning can be intact but unsuitable for some motion requirements.

Storage preflight reserves a conservative 35% of the uncompressed Bayer size plus
2 GiB; writing stops if the 2 GiB reserve is reached. Do not assume a fixed final
size: JPEG bitrate depends on scene detail and noise. Choose a dedicated fast
recording volume with sufficient free space. No background video recompression
runs during capture.

## Build on this workstation

The first implementation targets Linux x86-64 with Spinnaker 4.4, CUDA 12.8/NPP/
nvJPEG, and Ubuntu 22.04 FFmpeg 4.4 libraries. It is intentionally not a portable
replacement for the existing PySpin calibration source.

Required development headers: `libavformat-dev`, `libavcodec-dev`,
`libavutil-dev`, `nlohmann-json3-dev`; runtime FFmpeg/ffprobe must be installed.
CUDA and Spinnaker default to `/usr/local/cuda` and `/opt/spinnaker` (override with
`CUDA_PATH` and `SPINNAKER_PATH`). Install development packages normally, or extract
matching Ubuntu packages locally without changing system packages:

```bash
mkdir -p .native/debs .native/deps
(cd .native/debs && apt-get download libavformat-dev libavcodec-dev libavutil-dev nlohmann-json3-dev)
for package in .native/debs/*.deb; do dpkg-deb -x "$package" .native/deps; done
bash tools/build_recorder.sh
python -m multical.live --count 25 --autostart --recordings /path/to/recordings
```

Downloaded development package versions must match the installed FFmpeg ABI;
this build links `libavformat.so.58`, `libavcodec.so.58`, `libavutil.so.56`.
`.native/` and recording data are ignored by Git. If the native service is not
built, select **FLIR cameras · PySpin (calibration only)** or pass
`--camera-backend pyspin`.

Tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
QT_QPA_PLATFORM=offscreen python tests/recorder_ui_smoke.py
python tests/recorder_fault_smoke.py
```

The latter two require a working CUDA GPU/native build and use explicitly
simulated cameras. They do not open hardware cameras. Hardware qualification results and their limits are recorded below.

## Measured qualification — September 19, 2026

The 15-second physical rig test saved and independently decoded 22,500 frames.
A full 120-second test saved and independently decoded **180,000 frames**, exactly
7,200 from each of 25 cameras at 60 fps. Both passed schedule, exposure metadata,
frame order and full video decoding checks. The full take contained 1,500 finalized
two-second video segments, with 10.469 GB of JPEG payload (about 87.2 MB/s for the
current stage scene). Maximum raw queue: 23/512 frames; maximum encoded backlog:
37.8 MB/1 GiB. Exposure-start spread peaked at 9.12 µs. Different preserved exposure
times produced a 258.12 µs midpoint spread, correctly labelled for timing review.

The original 256-item encoded queue failed at the first multi-camera disk flush;
that failed take was retained and reported incomplete. The queue now has a byte
limit as well as capacity sufficient for measured flush bursts. Separately, a
real-UI trial encountered raw-buffer exhaustion around a GPU-saturation spike. A
subsequent monitored run identified a separate Python CUDA process alongside the
recorder. Recording needs available compute: other heavy GPU workloads can cause
an explicitly failed take. There is no silent frame substitution or automatic
rate reduction. Avoid concurrent GPU training/inference for production takes.

A second full **120-second take from the actual Qt UI** also passed complete
decoding and all metadata checks: another 180,000 frames, 10.423 GB JPEG payload,
100/512 maximum raw queue, 35.3 MB maximum encoded backlog, and 8.808 µs maximum
start spread. This take included the loaded calibration snapshot and checksum.
Preview pause/restore worked without reconnecting cameras. The original session
and its 60 validation poses remained loaded. The existing calibration UI regression
also passed collection, solving/cancellation, pause, save, new and resume.

The simulated Qt test exercised complete/early-stop takes, two takes in the same
native process, automatic preview pause/restore, calibration guards and RGB
correctness. Maximum observed event-loop gap was 61 ms. Native fault tests checked
a missing camera frame, an injected storage failure, and orderly shutdown during
recording. Failed takes remained incomplete; orderly shutdown produced a fully
verifiable shortened take. The Python suite passed 67 tests.

These results qualify the measured dim, mostly static scene and storage path, not all possible image
noise/detail, GPU workloads, disk conditions or multi-hour operation. See
[measured results](tools/recording_qualification_results.json) for take locations
and exact counters. This recorder does not turn a previously unvalidated camera
calibration into a validated one.
