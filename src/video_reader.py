import cv2
import os
import sys
from pathlib import Path
import subprocess

# Handle imports for both direct execution and module import
try:
    from .utils import ensure_dirs
except ImportError:
    from utils import ensure_dirs

def extract_audio(input_video, out_audio):
    """Extract original audio track to out_audio (wav or m4a)."""
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", out_audio
    ]
    subprocess.run(cmd, check=True)

def read_and_write_batches(input_video, frames_out_dir, batch_size=30):
    """
    Read video, write frames in batches.
    Returns: list of tuples -> [(batch_idx, start_frame, end_frame, frames_paths_list), ...]
    """
    ensure_dirs(frames_out_dir)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    batches = []
    frame_idx = 0
    batch_idx = 0
    current_batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"frame_{frame_idx:06d}.png"
        outpath = os.path.join(frames_out_dir, fname)
        cv2.imwrite(outpath, frame)
        current_batch.append(outpath)
        frame_idx += 1
        if len(current_batch) >= batch_size:
            batches.append((batch_idx, frame_idx - len(current_batch), frame_idx-1, list(current_batch)))
            batch_idx += 1
            current_batch = []
    if current_batch:
        batches.append((batch_idx, frame_idx - len(current_batch), frame_idx-1, list(current_batch)))
    cap.release()
    return batches, fps, total, w, h
