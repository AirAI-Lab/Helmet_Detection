#!/usr/bin/env python3
import os, sys, subprocess, cv2, shutil
from pathlib import Path

if len(sys.argv) < 5:
    print("Usage: make_annotated_video.py <engine> <video> <input_w> <input_h> [out_video]")
    sys.exit(1)

engine = sys.argv[1]
video = sys.argv[2]
input_w = int(sys.argv[3])
input_h = int(sys.argv[4])
out_video = sys.argv[5] if len(sys.argv) > 5 else 'trt_result_video_from_frames.avi'

workdir = Path(__file__).resolve().parent
tmp_in = workdir / 'tmp_in_frames'
tmp_out = workdir / 'tmp_out_frames'
shutil.rmtree(tmp_in, ignore_errors=True)
shutil.rmtree(tmp_out, ignore_errors=True)
tmp_in.mkdir(parents=True, exist_ok=True)
tmp_out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print('Failed to open video:', video); sys.exit(2)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Video opened:', video, 'fps=', fps, 'size=', w, 'x', h)

frame_idx = 0
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    fname = tmp_in / f'frame_{frame_idx:06d}.png'
    cv2.imwrite(str(fname), frame)
    frames.append(fname)
    if frame_idx % 100 == 0:
        print('Saved', frame_idx, 'frames')
cap.release()
print('Total frames extracted:', len(frames))

# Run trt_infer_video_fix on each frame
bin_path = workdir / 'trt_infer_video_fix'
if not bin_path.exists():
    print('Binary not found:', bin_path); sys.exit(3)

for i, f in enumerate(frames, start=1):
    outf = tmp_out / f.name
    cmd = [str(bin_path), str(engine), str(f), str(input_w), str(input_h), '--names', 'names.txt', '--out', str(outf), '--conf', '0.25']
    # call and wait
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print('Frame', i, 'failed:', ret.returncode)
        print(ret.stdout.decode())
        print(ret.stderr.decode())
        sys.exit(4)
    if i % 50 == 0:
        print('Processed', i, 'frames')

# assemble annotated frames into video
annot_files = sorted(tmp_out.glob('frame_*.png'))
if not annot_files:
    print('No annotated frames found'); sys.exit(5)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
outp = cv2.VideoWriter(str(workdir / out_video), fourcc, fps, (w,h))
if not outp.isOpened():
    print('Failed to open output video writer for', out_video); sys.exit(6)

for i, af in enumerate(annot_files, start=1):
    img = cv2.imread(str(af))
    if img is None:
        print('Failed to read annotated frame', af); sys.exit(7)
    outp.write(img)
    if i % 100 == 0:
        print('Wrote', i, 'frames to video')
outp.release()
print('Annotated video written to', workdir / out_video)
print('Done')
