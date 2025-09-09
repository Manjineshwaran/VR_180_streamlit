import subprocess
import os

def extract_audio(input_video, out_audio):
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", out_audio]
    subprocess.run(cmd, check=True)

def slice_audio(audio_file, start_sec, duration_sec, out_audio):
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration_sec), "-i", audio_file, "-acodec", "pcm_s16le", out_audio]
    subprocess.run(cmd, check=True)

def mux_audio_to_video(video_file, audio_file, out_file):
    cmd = ["ffmpeg", "-y", "-i", video_file, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", out_file]
    subprocess.run(cmd, check=True)
