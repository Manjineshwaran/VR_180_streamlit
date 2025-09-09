import os, re, subprocess

def _next_segment_start_number(stream_dir: str) -> int:
    pattern = re.compile(r"segment_(\d{5})\.ts$")
    max_idx = -1
    if os.path.isdir(stream_dir):
        for name in os.listdir(stream_dir):
            m = pattern.match(name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
    return (max_idx + 1) if max_idx >= 0 else 0

def add_batch_to_hls(batch_file, stream_dir="stream", fps=30):
    os.makedirs(stream_dir, exist_ok=True)

    # Append batch as .ts segment into HLS with audio and keyframe-aligned segments
    # Use zero-padded segment numbering for stable paths
    start_number = _next_segment_start_number(stream_dir)
    seg_template = os.path.join(stream_dir, "segment_%05d.ts").replace('\\', '/')
    playlist_path = os.path.join(stream_dir, "output.m3u8").replace('\\', '/')
    cmd = [
        "ffmpeg", "-y",
        "-i", batch_file,
        "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-hls_time", "2", "-hls_list_size", "0",
        "-hls_flags", "independent_segments+append_list+temp_file",
        "-hls_playlist_type", "event",
        "-start_number", str(start_number),
        "-hls_segment_filename", seg_template,
        "-f", "hls", playlist_path
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Added {batch_file} → {playlist_path} & segments")





