#!/usr/bin/env python3
"""Bounded replay benchmark: no camera access or production recording claims.

Predecode motion footage once, then measure actual concurrent encoder contexts.
Source Captury JPEG is already lossy; quality measures *additional* damage only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

PROFILES = {
 'hevc_p1_q24': ['-c:v','hevc_nvenc','-preset','p1','-rc','constqp','-qp','24'],
 'hevc_p1_q30': ['-c:v','hevc_nvenc','-preset','p1','-rc','constqp','-qp','30'],
 'hevc_p3_q28': ['-c:v','hevc_nvenc','-preset','p3','-rc','constqp','-qp','28'],
 'h264_p1_q24': ['-c:v','h264_nvenc','-preset','p1','-rc','constqp','-qp','24'],
 'h264_p1_q30': ['-c:v','h264_nvenc','-preset','p1','-rc','constqp','-qp','30'],
 'x264_ultrafast_crf23': ['-c:v','libx264','-preset','ultrafast','-crf','23','-tune','zerolatency'],
 'x264_ultrafast_crf18': ['-c:v','libx264','-preset','ultrafast','-crf','18','-tune','zerolatency'],
 'x264_ultrafast_crf15': ['-c:v','libx264','-preset','ultrafast','-crf','15','-tune','zerolatency'],
 'x264_ultrafast_crf28': ['-c:v','libx264','-preset','ultrafast','-crf','28','-tune','zerolatency'],
 'x264_veryfast_crf26': ['-c:v','libx264','-preset','veryfast','-crf','26','-tune','zerolatency'],
 'mjpeg_q5': ['-c:v','mjpeg','-q:v','5','-pix_fmt','yuvj420p'],
}


def run(args, timeout=120):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in args.cameras:
        original = args.take / f'stream{index:02d}.avi'
        destination = args.output / f'camera{index:02d}.yuv'
        run(['ffmpeg','-y','-v','warning','-threads','1','-ss',str(args.seek),'-i',str(original),
             '-an','-vf','scale=in_range=auto:out_range=full','-pix_fmt','yuvj420p',
             '-frames:v',str(args.frames),'-vsync','0','-f','rawvideo',str(destination)])
        frames = destination.stat().st_size // (1440*1080*3//2)
        if frames != args.frames or destination.stat().st_size % (1440*1080*3//2):
            raise ValueError(f'Incomplete source {destination}: {frames} frames')
        sources.append(dict(original=str(original), raw=str(destination), frames=frames))
    data = dict(width=1440,height=1080,fps=60,pixel_format='yuv420p',color_range='full',
                seek_seconds=args.seek,sources=sources,
                limitation='Decoded Captury JPEG; already lossy and may have original frame gaps. Replay is not live acquisition.')
    (args.output/'dataset.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)


def encode(args):
    dataset = json.loads((args.dataset/'dataset.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = args.profile.split(',')
    if not all(p in PROFILES for p in profiles): raise ValueError('Unknown codec profile')
    width,height = dataset['width'],dataset['height']
    origin = time.monotonic()
    def worker(index):
        profile=profiles[index % len(profiles)]
        src=dataset['sources'][index % len(dataset['sources'])]['raw']
        output=args.output/f'stream{index:02d}.mkv'
        cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-threads','1','-filter_threads','1',
             *(['-re'] if args.paced else []),
             '-stream_loop','-1','-f','rawvideo','-pixel_format','yuv420p','-color_range','pc',
             '-video_size',f'{width}x{height}','-framerate',str(args.rate),'-i',src,
             '-frames:v',str(args.frames),'-an',*PROFILES[profile],'-threads','1',
             '-g','60','-bf','0','-color_range','pc','-vsync','0','-stats_period','0.5','-progress','pipe:1',str(output)]
        events=[];started=time.monotonic()
        with (args.output/f'stream{index:02d}.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,text=True)
            # A separate watchdog bounds processes even if stdout blocks.
            import threading
            timer=threading.Timer(args.timeout,process.kill);timer.start()
            try:
                for line in process.stdout:
                    if line.startswith('frame='): events.append([time.monotonic()-origin,int(line.split('=')[1])])
                code=process.wait()
            finally:
                timer.cancel()
                if process.poll() is None: process.kill();process.wait()
        elapsed=time.monotonic()-started
        if code: raise RuntimeError(f'{profile} stream {index} failed; see {log.name}')
        probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-count_packets',
                              '-show_entries','stream=nb_read_packets','-of','json',str(output)]).stdout)
        packets=int(probe['streams'][0]['nb_read_packets'])
        if packets != args.frames:raise RuntimeError(f'Wrong packet count {packets}')
        return dict(stream=index,profile=profile,path=str(output),seconds=elapsed,fps=args.frames/elapsed,
                    frames=args.frames,packets=packets,bytes=output.stat().st_size,events=events)
    with ThreadPoolExecutor(max_workers=args.streams) as pool:
        rows=list(pool.map(worker,range(args.streams)))
    total_seconds=max(row['events'][-1][0] for row in rows)
    # Common interval during which every encoder is active; no tail with fewer sessions.
    low=max(row['events'][0][0] for row in rows)+1
    high=min(row['events'][-1][0] for row in rows)-.5
    steady=[]
    for row in rows:
        samples=[e for e in row['events'] if low <= e[0] <= high]
        if len(samples)>=3 and samples[-1][0]-samples[0][0] >= 1:
            steady.append((samples[-1][1]-samples[0][1])/(samples[-1][0]-samples[0][0]))
    total_bytes=sum(row['bytes'] for row in rows)
    summary=dict(profile=args.profile,streams=args.streams,frames_per_stream=args.frames,
        input_fps=args.rate,paced=args.paced,width=width,height=height,
        startup_inclusive_aggregate_fps=args.streams*args.frames/total_seconds,
        startup_inclusive_min_stream_fps=min(r['fps'] for r in rows),
        steady_aggregate_fps=sum(steady) if len(steady)==args.streams else None,
        steady_min_stream_fps=min(steady) if len(steady)==args.streams else None,
        steady_median_stream_fps=statistics.median(steady) if len(steady)==args.streams else None,
        projected_25cam_MB_s=total_bytes/(args.streams*args.frames)*1500/1e6,
        projected_120s_GB=total_bytes/(args.streams*args.frames)*180000/1e9,
        output_bytes=total_bytes,rows=rows,
        excludes='Camera transport, Bayer debayer, durable storage. OS cache and existing desktop/viewer load are present.')
    (args.output/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


def quality(args):
    """Compare full-range sample values; optionally crop to a foreground region."""
    dataset=json.loads((args.dataset/'dataset.json').read_text())
    reference=dataset['sources'][args.source]['raw']
    # Both inputs get identical range treatment and frame-index timestamps. In
    # particular, do not accidentally rescale full-range MJPEG to limited range.
    normalize='scale=in_range=full:out_range=full,format=yuv420p,setparams=range=full,settb=1/60,setpts=N'
    if args.crop: normalize+=',crop='+args.crop
    with tempfile.TemporaryDirectory(prefix='multical-quality-') as temp:
        stats=Path(temp)/'psnr.txt'
        filters=(f'[0:v]{normalize},split=2[rp][rs];[1:v]{normalize},split=2[dp][ds];'
                 f'[dp][rp]psnr=shortest=1:stats_file={stats}[p];[ds][rs]ssim=shortest=1[s]')
        result=run(['ffmpeg','-hide_banner','-loglevel','info','-xerror','-threads','1',
                '-f','rawvideo','-pixel_format','yuv420p','-color_range','pc',
                '-video_size',f"{dataset['width']}x{dataset['height']}",'-framerate','60','-i',reference,
                '-threads','1','-i',str(args.encoded),'-filter_complex_threads','1',
                '-filter_complex',filters,'-map','[p]','-map','[s]',
                '-frames:v:0',str(args.frames),'-frames:v:1',str(args.frames),'-f','null','-'])
        evaluated_frames=len(stats.read_text().splitlines())
    if evaluated_frames != args.frames:
        raise RuntimeError(f'Quality comparison evaluated {evaluated_frames}, expected {args.frames}')
    psnr=re.search(r'PSNR y:([\d.inf]+).*?average:([\d.inf]+)',result.stderr)
    ssim=re.search(r'SSIM Y:([\d.]+).*?All:([\d.]+)',result.stderr)
    if not psnr or not ssim: raise RuntimeError(result.stderr[-2500:])
    report=dict(encoded=str(args.encoded),source=args.source,evaluated_frames=evaluated_frames,crop=args.crop,
                alignment='frame_index_common_1_60_timebase',
                psnr_y_db=psnr[1],psnr_all_db=psnr[2],ssim_y=float(ssim[1]),ssim_all=float(ssim[2]),
                limitation='Additional loss relative to already-JPEG-compressed source; not reconstruction accuracy.')
    if args.output: args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def verify(args):
    """Flush completed replay files, then fully decode and count every stream."""
    summary=json.loads((args.run_dir/'result.json').read_text())
    rows=summary['rows']
    if len(rows)!=summary['streams']: raise ValueError('Incomplete stream list')
    started=time.monotonic()
    for row in rows:
        descriptor=os.open(row['path'],os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    flush_seconds=time.monotonic()-started
    def check(row):
        result=run(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames',
                    '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate',
                    '-of','json',row['path']],timeout=300)
        if result.stderr.strip(): raise RuntimeError(result.stderr)
        stream=json.loads(result.stdout)['streams'][0]
        if int(stream['nb_read_frames'])!=row['frames']:
            raise RuntimeError(f"Frame count mismatch: {row['path']}")
        expected=(summary.get('width',1440),summary.get('height',1080),f"{summary.get('input_fps',60)}/1")
        if (stream['width'],stream['height'],stream['avg_frame_rate'])!=expected:
            raise RuntimeError(f"Geometry or frame-rate mismatch: {row['path']}")
        return dict(stream=row['stream'],**stream)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked=list(pool.map(check,rows))
    report=dict(decoded_frames=sum(int(s['nb_read_frames']) for s in checked),cameras=len(checked),
                frames_each=summary['frames_per_stream'],flush_seconds_after_take=flush_seconds,
                seconds_with_full_decode=time.monotonic()-started,rows=checked,
                scope='Stored replay fully decoded and flushed; no live acquisition, PTP or power-loss verification.')
    (args.run_dir/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--take',type=Path,required=True)
    prep.add_argument('--cameras',type=int,nargs='+',default=[0,12,24]);prep.add_argument('--seek',type=float,default=120)
    prep.add_argument('--frames',type=int,default=360);prep.add_argument('--output',type=Path,required=True)
    enc=commands.add_parser('encode');enc.add_argument('--dataset',type=Path,required=True)
    enc.add_argument('--profile',required=True);enc.add_argument('--streams',type=int,default=25)
    enc.add_argument('--frames',type=int,default=600);enc.add_argument('--timeout',type=float,default=120)
    enc.add_argument('--rate',type=int,default=60);enc.add_argument('--paced',action='store_true')
    enc.add_argument('--output',type=Path,required=True)
    q=commands.add_parser('quality');q.add_argument('--dataset',type=Path,required=True)
    q.add_argument('--encoded',type=Path,required=True);q.add_argument('--source',type=int,default=0)
    q.add_argument('--frames',type=int,default=360);q.add_argument('--crop')
    q.add_argument('--output',type=Path)
    v=commands.add_parser('verify');v.add_argument('--run-dir',type=Path,required=True)
    v.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if getattr(a,'frames',1)<=0: p.error('frames must be positive')
    if a.command=='encode' and not 1<=a.streams<=25:p.error('streams must be 1–25')
    if a.command=='encode' and (a.rate<=0 or a.timeout<=0):p.error('rate and timeout must be positive')
    if a.command=='verify' and not 1<=a.workers<=25:p.error('workers must be 1–25')
    {'prepare':prepare,'encode':encode,'quality':quality,'verify':verify}[a.command](a)

if __name__=='__main__':main()
