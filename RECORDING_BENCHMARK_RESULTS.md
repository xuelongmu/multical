# Recording compression experiments — 2026-09-19

**Two promising paths on this machine: GPU JPEG quality 85 for independent frames, and CPU H.264 ultrafast CRF18 for compact takes.** Both sustained a two-minute paced encoding workload equivalent to 25 cameras at 1440×1080, 60 fps. This is a performance prototype, not a finished live recorder.

The executable prototypes are `tools/recording_benchmark.py` and `tools/recording_jpeg_benchmark.cpp`. Machine-readable measurements, excluding footage and local frame paths, are in `tools/recording_benchmark_results.json`.

## What was actually tested

- Xeon Gold 6246R, 16 physical cores / 32 threads; 93 GiB RAM; RTX A6000 48 GiB; NVIDIA driver 575.57.08; FFmpeg 4.4.2; CUDA nvJPEG 12.3.5.92; installed libjpeg-turbo.
- Three existing stage perspectives: Captury `test_003`, streams 00, 12 and 24, seeking to 120 seconds and decoding 360 frames each. The footage includes people moving around the stage. These clips repeat across 25 virtual cameras; they are not 25 different simultaneous scenes.
- Inputs were predecoded to full-range planar YUV420. FFmpeg tests launched 25 actual encoder processes with one codec thread each, one-second GOPs and no B frames. JPEG tests used native worker pools and interleaved jobs for 25 virtual cameras. CUDA measurements include staging copies, host-to-device transfer and compressed output retrieval.
- Ordinary desktop and live-viewer load remained present for encoder throughput tests. These are observed runs, not isolated hardware maxima or confidence intervals. The viewer was later restarted to deploy the preview pause control; the original encoding comparisons ran before that restart. A quality-check process overlapped the beginning of the paced GPU JPEG test.
- **Excluded:** GigE ingestion, PTP scheduling, Bayer debayer/colour processing, production queue management and synchronized frame ledgers. JPEG throughput tests discard timed bitstreams; video tests write MKV through the OS cache. Flush/full-decode checks and a storage test were performed separately.

The required aggregate rate is **1,500 frames/s**. For uncompressed Bayer8, the corresponding payload is 2.3328 GB/s and 279.936 GB per 120-second take. Encoded sizes below are measured on this footage and projected to 180,000 images; they are not bitrate limits or forecasts for arbitrary subjects, lighting or sensor noise.

## Throughput and size

Video rates below use the common interval during which all 25 encoders were active, excluding initial allocation and the tail after other streams finished. The full results also retain startup-inclusive rates. JPEG rates describe the warmed worker pool; dividing them by 25 would not establish per-camera fairness.

| Encoder / settings | Measured aggregate fps | Slowest video stream fps | MB/s at target rate | GB / 120 s |
|---|---:|---:|---:|---:|
| GPU HEVC P1, QP24 | 1,127 | 45.1 | 16.2 | 1.95 |
| GPU HEVC P3, QP28 | 1,129 | 45.1 | 4.75 | 0.57 |
| GPU H.264 P1, QP24 | 1,045 | 41.8 | 7.13 | 0.86 |
| GPU H.264 P1, QP30 | 1,068 | 42.7 | 2.93 | 0.35 |
| CPU H.264 ultrafast, CRF15 | 1,569 | 57.0 | 100.5 | 12.05 |
| **CPU H.264 ultrafast, CRF18** | **1,913** | **68.8** | **41.4** | **4.97** |
| CPU H.264 ultrafast, CRF23 | 2,744 | 93.2 | 6.08 | 0.73 |
| CPU H.264 ultrafast, CRF28 | 2,668 | 95.4 | 2.51 | 0.30 |
| CPU H.264 veryfast, CRF26 | 1,498 | 54.3 | 2.18 | 0.26 |
| **GPU JPEG Q85, 4 workers** | **2,931** | — | **243.5** | **29.22** |
| GPU JPEG Q85, 8 workers | 2,849 | — | 243.5 | 29.22 |
| GPU JPEG Q70, 8 workers | 2,864 | — | 182.7 | 21.92 |
| GPU JPEG Q55, 8 workers | 2,867 | — | 114.0 | 13.67 |
| CPU JPEG Q85, 8 workers | 2,483 | — | 242.9 | 29.15 |
| CPU JPEG Q85, 12 workers | 3,692 | — | 242.9 | 29.15 |
| CPU JPEG Q70, 12 workers | 3,863 | — | 187.2 | 22.46 |
| CPU JPEG Q55, 12 workers | 4,274 | — | 113.5 | 13.62 |

HEVC P1 QP30 produced a projected 0.36 GB take, but one stream finished so early that the short run lacked a usable common measurement interval. Its startup-inclusive aggregate was 951 fps; do not label that number steady throughput. This illustrates why multiplying a single stream's performance or averaging only the finish times can be misleading.

Increasing JPEG GPU workers from four to eight did not improve this implementation. Increasing compression also did not materially improve GPU JPEG throughput. The CPU H.264 preset mattered greatly. CRF15 and `veryfast` failed the **slowest-stream** criterion even where aggregate throughput approached or exceeded 1,500 fps.

NVENC utilization reached 100% during the GPU video tests. The [NVIDIA support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix) lists one NVENC engine for the RTX A6000; unrestricted session count does not imply unrestricted throughput. AV1 hardware encoding is not available on this GPU. nvJPEG uses GPU compute here, so its throughput should not be inferred from the NVENC ceiling. See the [nvJPEG documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html).

## Two-minute checks and storage

**CPU H.264 ultrafast CRF18:** 25 paced streams, 7,200 frames each, approximately 60 fps each throughout the common measurement interval. Total file size was **4,971,655,608 bytes**. After flushing the outputs, every stream fully decoded with the expected dimensions, rate and frame count: **180,000 decoded frames, no reported decode errors**. This verifies the replay outputs, not genuine camera exposures or PTP matching. The short unpaced test's worst-stream headroom was only about 15%; camera ingestion and colour conversion still have to fit.

**GPU JPEG Q85, four workers:** 180,000 jobs completed in **120.004 s**, with 25 jobs released together every 1/60 second. From each scheduled release to JPEG availability, p99 latency was **8.84 ms**, maximum **13.71 ms**. Individual encode/transfer time had median **1.27 ms**, p95 **1.82 ms**. No jobs were discarded; all completed before the next batch interval in this run. Encoded frames were counted and sized but not written to disk during this test. This measures an encoder stage, not a complete take writer.

**External SABRENT drive:** a new temporary 16 GiB file on the existing exFAT mount, using 8 MiB writes with `fsync` every 256 MiB, averaged **547.6 MB/s** over **31.37 s**. The slowest flushed 256 MiB interval was **490.8 MB/s**. Only this test file was created and then deleted. This gives useful margin over the measured JPEG rate, but it is a short sequential test; it does not certify multiple two-minute captures, fragmentation, low-free-space behavior or power-loss durability.

## Compression quality: a useful comparison, with a strong source bias

Quality was measured over all 360 frames from each of the three sources. PSNR is pooled by converting each equal-size clip's score back to mean squared error; SSIM is averaged across clips. A fixed lower-right region of camera 00 contains a moving subject for part of the clip and provides a second check beyond the mostly static floor. It is not a tracked foreground mask.

| Setting | Pooled PSNR, dB ↑ | Mean SSIM ↑ | Subject-region SSIM ↑ |
|---|---:|---:|---:|
| CPU H.264 ultrafast CRF15 | 46.65 | 0.98318 | 0.98011 |
| CPU H.264 ultrafast CRF18 | 45.28 | 0.97796 | 0.97329 |
| CPU H.264 ultrafast CRF23 | 43.35 | 0.96824 | 0.95815 |
| CPU H.264 ultrafast CRF28 | 41.57 | 0.95575 | 0.93966 |
| GPU HEVC P1 QP24 | 45.23 | 0.97808 | 0.97507 |
| GPU HEVC P1 QP30 | 43.72 | 0.97084 | 0.96386 |
| CPU JPEG Q70 | 45.23 | 0.98047 | 0.97838 |
| CPU JPEG Q55 | 44.05 | 0.97431 | 0.97145 |

JPEG Q85 scored implausibly close to lossless as a general camera-quality claim: approximately **65.5 dB / 0.99981 SSIM for CPU**, and **80.8 dB / 0.999995 SSIM for GPU**. The reference is itself decoded Captury JPEG. These numbers are consistent with reusing the source's quantization and therefore flatter JPEG at that setting. They establish very little additional damage on these particular inputs, not preservation of original sensor detail. Fresh Bayer-derived colour frames, including moving hair, clothing texture and noisy/dim exposures, are needed to select master quality and chroma subsampling.

The source was already 4:2:0, so this experiment cannot assess the benefit of 4:4:4 or the damage from the first chroma reduction. PSNR/SSIM also do not establish 4DGS reconstruction quality. The small CRF23/28 files are credible results on this footage, but they are aggressive candidates, not automatic master-quality defaults.

An implementation issue found during testing: assigning frame-index timestamps without first normalizing the timebase can misalign raw 1/60 s inputs, MKV millisecond timestamps and MJPEG's guessed frame rate. The harness now explicitly sets both timebases to 1/60 and both timestamps to the frame index. All retained quality measurements use that fix. A lossless control with deliberately different input frame rates produces infinite PSNR and SSIM 1; an incomplete comparison is rejected.

## Reproduce the experiments

Run from the worktree. Media and binaries go outside the repository. `prepare` currently targets this rig's 1440×1080 clips.

```bash
python3 tools/recording_benchmark.py prepare \
  --take /path/to/captury/take \
  --output /tmp/multical-recording-bench/dataset

python3 tools/recording_benchmark.py encode \
  --dataset /tmp/multical-recording-bench/dataset \
  --profile x264_ultrafast_crf18 --streams 25 --frames 1200 \
  --output /tmp/multical-recording-bench/cpu-capacity

python3 tools/recording_benchmark.py encode \
  --dataset /tmp/multical-recording-bench/dataset \
  --profile x264_ultrafast_crf18 --streams 25 --frames 7200 \
  --paced --timeout 180 --output /tmp/multical-recording-bench/cpu-soak

python3 tools/recording_benchmark.py verify \
  --run-dir /tmp/multical-recording-bench/cpu-soak

python3 tools/recording_benchmark.py quality \
  --dataset /tmp/multical-recording-bench/dataset \
  --encoded /tmp/multical-recording-bench/cpu-soak/stream00.mkv --source 0

g++ -O3 -std=c++17 -pthread tools/recording_jpeg_benchmark.cpp \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/local/cuda/lib64 -lnvjpeg -lcudart -lturbojpeg \
  -o /tmp/multical-recording-bench/jpeg-bench

/tmp/multical-recording-bench/jpeg-bench gpu 85 4 180000 \
  /tmp/multical-recording-bench/jpeg-soak \
  /tmp/multical-recording-bench/dataset/camera00.yuv \
  /tmp/multical-recording-bench/dataset/camera12.yuv \
  /tmp/multical-recording-bench/dataset/camera24.yuv --paced
```

For JPEG capacity, omit `--paced` and use 15,000 jobs. Choose `cpu` to test libjpeg-turbo. Add `--export` to write the source-length JPEG clips **after** the timed section for quality comparisons; those files are not part of the measured writer workload. Video profiles are listed in the Python script; comma-separated names cycle across streams for future mixed-backend experiments. No hybrid throughput result is claimed here.

Run one capacity experiment at a time. Both prototypes are bounded by finite frame counts; Python video workers have per-process watchdogs. Use an external timeout for standalone native experiments. Neither executable accesses cameras or changes their settings.

## Implementation decision

Start the production service with **GPU JPEG Q85 as the independent-frame candidate**, retaining the measured CPU JPEG fallback. Offer **CPU H.264 ultrafast CRF18 as a compact candidate**, subject to a quality comparison on fresh colour input. Do not select all-camera NVENC as the only backend on this GPU. Do not promise a fixed take size from these clips.

The next qualification must run the actual chain: pipelined scheduled triggers → all 25 Bayer receivers → debayer/colour conversion → bounded encoders → segmented per-camera files and timestamp ledger → durable completion. Require every expected frame, stable queue occupancy and measured headroom while the UI is active. Preview pause is available now; it reduces display work while acquisition, calibration detection and pose saving continue. Board detection should become optional in the future recording mode rather than silently changing the current calibration behavior.
