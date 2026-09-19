# Synchronized 25-camera recording — design proposal

Status: design plus measured replay/encoding prototype, 2026-09-19. See [benchmark results](RECORDING_BENCHMARK_RESULTS.md) and the executable tools. A production camera recording service is not implemented. Target confirmed by operator: **25 × 1440×1080 colour, 60 fps, takes up to 120 seconds**. Live camera acquisition was not switched to 60 fps for these tests. A bounded temporary-file storage test was performed and removed; existing takes were read only.

## Observed machine and calculated load

Read-only local inspection found Intel Xeon Gold 6246R (16 physical cores / 32 threads), 93 GiB RAM, NVIDIA RTX A6000 (48 GiB; driver 575.57.08), three active 10 Gb/s camera uplinks plus one active 1 Gb/s direct-camera link. `/` uses the internal Samsung MZVLB1T0HALR NVMe ext4 partition, about 92 GiB free. The external SABRENT volume is exFAT, about 722 GiB free, connected over USB 10 Gb/s. These are observed capacity/link figures, **not sustained throughput measurements**. GPU SM activity and available RAM fluctuate with other applications.

Installed media tools: FFmpeg 4.4.2 lists H.264/HEVC NVENC, libx264/libx265, FFV1 and MJPEG. GStreamer 1.20.3 has appsrc, queue, splitmuxsink, matroskamux and jpegenc, but inspection did not find nvh264enc, nvh265enc, nvjpegenc or cudaupload. A CUDA-capable media build/integration needs explicit qualification; modern documentation does not imply these plugins are installed.

Assuming uncompressed **BayerRG8**, excluding protocol/chunk overhead:

| Quantity | 25 cameras × 60 fps |
|---|---:|
| Images per second | 1,500 |
| Payload per frame | 1,555,200 bytes |
| Camera network payload | 18.6624 Gb/s total; 0.7465 Gb/s per camera |
| Host raw Bayer throughput | 2.3328 GB/s |
| Raw take, 120 s | 279.936 GB |
| RGB8 after debayer | 6.9984 GB/s |
| Images per camera / total per take | 7,200 / 180,000 |
| 8–16 GiB raw host ring | about 3.7–7.4 s of absorption |

Eight cameras on one 10 Gb/s uplink imply about 5.97 Gb/s image payload, before overhead. Aggregate NIC capacity is encouraging but says nothing about burst loss, switch queues or receive/CPU contention. Existing 20-camera tests are not proof for all 25 at 60 fps.

A 10 Gb/s USB link has only 1.25 GB/s theoretical line-rate capacity, already below the 2.33 GB/s raw requirement. The internal partition currently lacks room for a full raw take. RAM cannot hold a complete take. Raw recording requires enough storage space and **measured sustained** bandwidth with margin (proposed qualification target: at least 3 GB/s under realistic conditions), not an SSD's short cache-assisted peak.

## Current code is a calibration source, not a continuous recorder

- `multical/live/sources.py:309`: `_collect()` converts every image to Mono8 and copies it before releasing the SDK image. The current retained calibration images cannot provide original colour recording; branch off before this conversion.
- `multical/live/sources.py:323`: `read()` schedules one action 150 ms in the future, waits for all camera futures, then returns. Even ignoring all other work, this is structurally limited to under 6.7 sets/s, rather than a pipelined 60 fps clock. The periodic PTP/exposure/gain reads also live on this path.
- `multical/live/engine.py:71`: acquisition overwrites `latest_batch`; analysis deliberately samples the newest batch. This is suitable for preview but not a recording queue.
- `multical/live/session.py:85`: pose saving serially writes a PNG per camera and updates the manifest. Reuse its camera identity, calibration provenance and commit concepts, not per-frame PNG/JSON rewrites at 1,500 images/s.
- `multical/live/__main__.py:29`: the current CLI also rejects rates above 20 Hz. Raising that limit alone does not fix the acquisition design.

Reuse the discovery/serial-MAC mapping, scheduled Action0 keys, timestamp checks, stop/restore handling, UI camera wall and calibration snapshot formats. Replace the throughput-critical acquisition/writer path. Keep board detection and calibration optimization off the full-rate recording path.

## Proposed architecture

Keep the Qt application as controller/monitor. Put ownership of all cameras in a separate native C++ Spinnaker recording service, with explicit IPC commands, status and shared-memory preview buffers. Do not open the cameras simultaneously from separate calibration and recording processes; transition ownership cleanly, or let calibration subscribe to the service later.

```
PTP-domain scheduler -> 25 camera receive loops -> bounded Bayer buffer pools
                                             |-> recording workers -> segmented streams
                                             |-> sampled preview -> Qt / optional detection
                                             `-> timestamp ledger and integrity counters
```

1. **Scheduler:** choose a future take epoch only after all receivers, buffers, files and encoders report ready. Maintain fixed action times `T0 + n / 60` in camera/PTP time, independent of encoding or UI timing. Use integer/rational arithmetic to avoid accumulated truncation drift. Start with a bounded 4–6 frame scheduling horizon; discover actual queue limits and account for command latency. This camera family's documentation specifies a **10-action queue**. Never batch an entire take or blindly enqueue 150 ms repeatedly. Detect late/overflow/no-reference acknowledgements. Stop scheduling on an unrecoverable recording fault; drain already queued actions and mark any lost frames.
2. **Acquisition:** persistent native receive workers, preallocated buffers, no PNG encoding, filesystem calls, debayer, UI callbacks or camera discovery in the receive loop. Start with one bounded owned-buffer copy and prompt SDK release. Investigate user-provided/pinned SDK buffers only with proven ownership/lifetime behavior. Do not assume GigE/Spinnaker supplies GPUDirect or end-to-end zero copy.
3. **Compute:** upload original Bayer once; GPU debayer/colour conversion and preview scaling. NVENC is a separate engine from CUDA JPEG/processing. Use bounded CPU workers for JPEG or lossless compression as measured; start with 8–12 compute workers while leaving CPU time for acquisition, networking, PTP and the UI. A large number of blocking receive threads is different from saturating every core with encoding threads.
4. **Queues:** explicit byte/time budgets at every stage. Recording queues preserve frames; previews use a separate latest-frame/drop-old policy and cannot retain scarce recording buffers indefinitely. Growing queues trigger warnings and then a controlled take fault before exhaustion. Do not silently reduce to 30 fps, leak recording frames or substitute repeated images. Buffers absorb stalls; they do not fix a sustained throughput deficit.
5. **Files:** separate streams per camera, segmented into independently recoverable chunks, plus a shared frame ledger. Suggested starting segments: 2–5 seconds, with aligned keyframe boundaries for inter-frame codecs. Use Matroska or deliberately configured fragmented MP4 rather than ordinary MP4 that depends on final closure. Store PTP timestamps separately at integer-nanosecond precision; video-container timestamps are not the authoritative sync record.
6. **Stop:** stop at a shared schedule boundary, drain receive/encode/write queues, finalize and flush data, then mark the take complete. Distinguish recording, draining, saved and verified states. A crash leaves an incomplete manifest and recoverable finalized chunks. Durable completion requires data/index flushes, not just rename. UI exit/disconnection must have an explicit service policy; for a bounded armed take, finish it and retain the result even if the UI fails.

## Encoder/storage candidates

**Do not assume 25 simultaneous NVENC sessions means 25 real-time streams.** NVIDIA's matrix lists the RTX A6000 as one seventh-generation NVENC engine with unrestricted session count, H.264/HEVC and no AV1 encoding. Our load is pixel-equivalent to **1,125 full-HD frames/s**. NVIDIA's indicative Ampere table gives roughly **868 H.264 / 943 HEVC full-HD frames/s** at its fastest listed P1 low-latency setting, and less at slower presets. Those are RTX 3090/Windows/SDK13.1 reference measurements, not a benchmark of our Linux A6000; they are enough reason not to promise this workload fits.

| Path | Why evaluate it | Constraint / decision |
|---|---|---|
| Raw Bayer8 chunks, optional fast lossless compression | Original sensor values; minimal processing before persistence; defer colour/codec choices | Needs capacity and measured sustained writes; compression of noisy Bayer is uncertain; add dedicated fast storage if this is the quality master |
| GPU JPEG with nvJPEG, CPU JPEG with libjpeg-turbo | Independent frames suit random-access 4DGS processing; uses CUDA/CPU capacity instead of relying solely on one NVENC engine | Measure complete debayer + encode + mux + disk path at chosen chroma/quality; don't infer throughput from library support |
| H.264/HEVC NVENC | Compact video, low CPU encoding cost | Single-engine throughput likely a bottleneck at full target; benchmark; split work with CPU or add encoding capacity only if needed |
| CPU FFV1/lossless Bayer representation | Potential exact-data archival option | CPU and compression ratio unknown at 2.33 GB/s; verify byte-for-byte round trips and layout metadata before treating as raw master |

For JPEG sizing, **an assumption** of 200 kB per encoded frame gives 300 MB/s and 36 GB per take; 500 kB gives 750 MB/s and 90 GB. For video, an **assumed** 40 Mb/s per camera gives 125 MB/s and 15 GB per take. Neither bitrate nor quality is guaranteed; size and acceptable reconstruction quality must be measured on representative lit, moving subjects.

For 4DGS, compare 4:4:4/less aggressive compression with 4:2:0 using actual edges, hair and textures. Lossless encoding after debayer/chroma reduction does not restore original Bayer samples. Record native geometry; export undistorted images and adjusted intrinsics together downstream. Freeze exposure, gain, white balance/colour processing and calibration provenance during the take; record exact processing metadata. Moving subjects need an exposure/motion-blur budget and exposure-midpoint checks beyond stationary-board timing policy.

**Measured direction:** GPU JPEG quality 85 and CPU H.264 ultrafast CRF18 both passed two-minute paced encoder replays. GPU JPEG completed all 180,000 jobs; CPU H.264 wrote 25 streams whose 180,000 frames fully decoded. Their projected take sizes on the existing footage were about 29 GB and 5 GB respectively. GPU JPEG is the first independent-frame master candidate; CPU H.264 is the compact alternative. The single NVENC engine did not meet the full target in the tested configurations. Qualify camera transport, debayer and storage together before exposing a production Record button. Source footage was already JPEG-compressed, so these are not raw-camera quality or bitrate guarantees. For an uncompromised Bayer master, provision dedicated storage. A second GPU should follow evidence, not precede acquisition fixes.

The live UI now has a separate **Pause previews** button beside the camera wall, also bound to **P**. This skips image resizing/conversion and updates for the wall, single inspection view and four-up. Acquisition, detection, pose saving, sounds and status continue. It does not disable debayer or detection upstream and is not a recording throughput qualification. When the recorder is integrated, use sampled or paused previews independently of full-rate acquisition.

## Existing implementations worth borrowing from

- FLIR `AcquisitionMultipleThread`: per-camera acquisition threads; `AcquisitionUserBuffer`: explicit buffer ownership and release; `AcquisitionMultipleCamerasWriteToFile`: append raw buffers and convert later. The latter sample writes synchronously and is an example, not our full recording scheduler. Installed source examples are under `/home/zerospace/Downloads/docs/site/examples/cpp/`; public entry: https://softwareservices.flir.com/Spinnaker/latest/examples/cpp/AcquisitionMultipleCamerasWriteToFile.html
- GStreamer `appsrc`, explicit `queue`, encoders and `splitmuxsink`: reusable timestamped media plumbing and asynchronous fragment finalization. Default queue limits are only 200 buffers, 10 MiB or 1 second, whichever first; override deliberately. Current installed GStreamer lacks the NVIDIA plugins. https://gstreamer.freedesktop.org/documentation/coreelements/queue.html and https://gstreamer.freedesktop.org/documentation/multifile/splitmuxsink.html
- NVIDIA nvJPEG and libjpeg-turbo: GPU/CPU JPEG implementations instead of writing codecs. GStreamer's modern `nvjpegenc` accepts CUDA-memory inputs. https://developer.nvidia.com/nvjpeg ; https://gstreamer.freedesktop.org/documentation/nvcodec/nvjpegenc.html ; https://github.com/libjpeg-turbo/libjpeg-turbo
- OBS: use its recording-resilience lessons (Matroska / fragmented or hybrid MP4), not a composited camera-wall recording as the dataset. https://obsproject.com/kb/audio-video-formats-guide
- Existing CaptureNet investigations: `docs/03-captury-recording.md:45` records encoder late-start corruption, silent frame-rate caps and dropped queues; `:161` records a GPU-stage ceiling near 660 frames/s with tracking and NVENC idle; `:171` records historical USB/NTFS writer starvation. These explain why we must instrument stages and pre-arm recording. Historical Captury limits are not measurements of the proposed codec path or today's exFAT mount.

Hardware/reference sources:
- https://developer.nvidia.com/video-encode-decode-support-matrix
- https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html
- https://softwareservices.flir.com/BFS-PGE-16S2/latest/Model/public/ActionControl.html

## Build and qualification sequence

1. **Instrument and benchmark first.** Receive-only at full target; GPU debayer alone; JPEG CPU/GPU and NVENC on representative frames; disk sustained writes including flushes; then end-to-end with the real UI. Synthetic/replay benchmarks should avoid a repeated easy frame and separate input-generation cost from encoder cost. Benchmark all 25 contexts, not one stream multiplied by 25. Stress work must be scheduled independently of valuable live takes.
2. **Service + ledger + raw reference sink.** Establish full-colour data retention, frame ownership, pipelined triggering and explicit faults. Build the frame ledger with take ID, camera serial, scheduled global index, hardware frame ID, PTP/exposure timestamps, exposure/gain, pixel layout/stride and eventual file/packet location. Frame IDs may differ between cameras: match timestamps to scheduled indices, never pair by dequeue order.
3. **Choose the master writer from measured results.** Add GPU/CPU compression, segment finalization, disk reserve checks, checksums and recovery. Reject arming when expected take capacity or qualified throughput is insufficient. Keep codec/quality fixed during a take, even when work is distributed between resources.
4. **Integrate Record/Stop/review into Multical Live.** Show requested and received fps, retained/encoded/persisted counts, per-camera gaps, queue age/fill, PTP skew, CPU/GPU/NVENC usage, write throughput, free space/time left and finalization progress. Starting/stopping a take should not disconnect cameras or rebuild calibration coverage.
5. **Qualify repeatably.** Several full 120-second takes at 25 × 60 with the UI active; proposed 20–30% processing/write headroom under representative load. Require 7,200 real decoded frames per camera, 180,000 total, matching schedule/ledger entries, no hidden duplication or timing drift, and no increasing queue backlog. Verify start, middle and end plus full decode/frame-count checks. Inject a late frame, missing camera, stalled disk, encoder failure, UI crash and interrupted recording; preserve existing data and report the take as incomplete when appropriate.

Proposed initial same-frame timing budget: ≤20 µs maximum exposure-midpoint spread, logged per set and subject to a motion/reconstruction error budget. This is an operational target, not a claim of sub-millimetre reconstructed accuracy. Record expected gaps as gaps; optional preview placeholders must never be mistaken for real dataset images.
