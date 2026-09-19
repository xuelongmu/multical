"""Offline integrity check for a completed native recording (never run on the GUI thread)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Recording references a file outside its take directory')
    return path


def decode_segment(path, width, height, fps, count):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_streams', '-show_entries',
        'stream=codec_name,width,height:frame=key_frame,best_effort_timestamp_time',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=120)
    if result.returncode or result.stderr.strip():
        raise ValueError(f'{path.name}: decode failed: {result.stderr.strip()}')
    probe = json.loads(result.stdout)
    streams, frames = probe.get('streams', []), probe.get('frames', [])
    expected = dict(codec_name='mjpeg', width=width, height=height)
    if len(streams) != 1 or any(streams[0].get(k) != v for k, v in expected.items()):
        raise ValueError(f'{path.name}: unexpected codec or geometry')
    if len(frames) != count or any(f.get('key_frame') != 1 for f in frames):
        raise ValueError(f'{path.name}: decoded frame count or independent-frame check failed')
    for index, frame in enumerate(frames):
        # Matroska stores millisecond timestamps; hardware nanoseconds live in CSV.
        if abs(float(frame['best_effort_timestamp_time']) - index / fps) > .0011:
            raise ValueError(f'{path.name}: invalid presentation timestamp at frame {index}')
    return count


def verify_take(directory, *, decode=True, workers=4):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'take.json').read_text())
    if manifest.get('state') != 'saved' or manifest.get('complete') is not True or manifest.get('error'):
        raise ValueError('Take is not marked complete and saved')
    config = manifest['configuration']
    cameras = config['cameras']
    serials = [c['serial'] for c in cameras]
    expected, fps = manifest['scheduled_frames'], manifest['fps']
    if not serials or len(set(serials)) != len(serials) or expected <= 0:
        raise ValueError('Invalid camera roster or scheduled count')
    if set(manifest['cameras']) != set(serials) or set(manifest['segments']) != set(serials):
        raise ValueError('Camera roster differs between configuration and output')
    context = json.loads((root / 'context.json').read_text())
    if context['camera_serials'] != serials:
        raise ValueError('Camera roster differs from capture context')
    if context.get('calibration_file'):
        digest = hashlib.sha256(_inside(root, context['calibration_file']).read_bytes()).hexdigest()
        if digest != context.get('calibration_sha256'):
            raise ValueError('Calibration checksum mismatch')
    common_schedule = None
    starts, midpoints, jobs = [], [], []
    for camera in cameras:
        serial = camera['serial']
        counts = manifest['cameras'][serial]
        if any(counts.get(k) != expected for k in ('received', 'encoded', 'written')):
            raise ValueError(f'{serial}: frame counts differ from scheduled count')
        with _inside(root, f'cameras/{serial}/frames.csv').open(newline='') as stream:
            rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        if [r['frame_index'] for r in rows] != list(range(expected)):
            raise ValueError(f'{serial}: missing, duplicate or reordered frame indices')
        ids = [r['frame_id'] for r in rows]
        if any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError(f'{serial}: repeated or reordered camera frame IDs')
        schedule = [r['scheduled_ns'] for r in rows]
        if common_schedule is None:
            common_schedule = schedule
            if schedule != [schedule[0] + i * 1_000_000_000 // fps for i in range(expected)]:
                raise ValueError('Scheduled timestamps do not match the requested frame rate')
        elif schedule != common_schedule:
            raise ValueError(f'{serial}: scheduled timestamps differ between cameras')
        exposure = round(camera['exposure_us'] * 1000)
        for row in rows:
            if (row['exposure_start_ns'] != row['timestamp_ns'] - exposure or
                    row['exposure_midpoint_ns'] != row['timestamp_ns'] - exposure // 2 or
                    abs(row['exposure_start_ns'] - row['scheduled_ns']) > 1_000_000):
                raise ValueError(f'{serial}: invalid exposure timestamps')
        starts.append([r['exposure_start_ns'] for r in rows])
        midpoints.append([r['exposure_midpoint_ns'] for r in rows])
        covered = 0
        for segment in manifest['segments'][serial]:
            index, count = segment['segment'], segment['frames']
            subset = [r for r in rows if r['segment'] == index]
            if (not segment.get('finalized') or not count or len(subset) != count or
                    [r['packet'] for r in subset] != list(range(count)) or
                    [r['frame_index'] for r in subset] != list(range(covered, covered + count)) or
                    subset[0]['frame_index'] != segment['first_scheduled_index'] or
                    sum(r['bytes'] for r in subset) != segment['jpeg_bytes']):
                raise ValueError(f'{serial}: segment ledger mismatch')
            path = _inside(root, segment['file'])
            if not path.is_file() or path.stat().st_size < segment['jpeg_bytes']:
                raise ValueError(f'{serial}: missing or truncated video segment')
            jobs.append((path, camera['width'], camera['height'], fps, count))
            covered += count
        if covered != expected:
            raise ValueError(f'{serial}: video segments do not cover all frames')
    decoded = 0
    if decode:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded = sum(pool.map(lambda args: decode_segment(*args), jobs))
    return dict(directory=str(root), integrity='passed', decoded=bool(decode),
                decoded_frames=decoded, cameras=len(cameras), frames_per_camera=expected,
                fps=fps, seconds=expected / fps, segments=len(jobs),
                max_start_spread_us=max(max(v) - min(v) for v in zip(*starts)) / 1000,
                max_midpoint_spread_us=max(max(v) - min(v) for v in zip(*midpoints)) / 1000,
                simulated=config.get('simulated', False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take')
    parser.add_argument('--metadata-only', action='store_true', help='Skip video decoding; weaker integrity check')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        if not 1 <= args.workers <= 32:
            raise ValueError('Use 1–32 decoding workers')
        print(json.dumps(verify_take(args.take, decode=not args.metadata_only, workers=args.workers), indent=2))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')


if __name__ == '__main__':
    main()
