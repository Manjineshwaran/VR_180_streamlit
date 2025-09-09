import subprocess

def inject_vr180_metadata(input_file, output_file):
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-c", "copy",
        "-metadata:s:v:0", "stereo_mode=left_right",
        "-metadata:s:v:0", "projection=180",
        output_file
    ]
    subprocess.run(cmd, check=True)
