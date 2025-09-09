import os
import subprocess

# Handle imports for both direct execution and module import
try:
    from .utils import ensure_dirs
except ImportError:
    from utils import ensure_dirs

def frames_to_segment(frames_dir, out_segment_path, fps=30, codec="libx264", bitrate="6M"):
    """
    Use ffmpeg to convert frames (sorted alphabetically) to a mp4 segment with consistent settings.
    This makes it easy to later concat with -c copy.
    """
    ensure_dirs(os.path.dirname(out_segment_path))
    
    # Convert paths to absolute and normalize them
    frames_dir = os.path.abspath(frames_dir)
    out_segment_path = os.path.abspath(out_segment_path)
    
    # Get list of all PNG files in the directory
    png_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])
    
    if not png_files:
        raise ValueError(f"No PNG files found in {frames_dir}")
    
    # Create a temporary text file with the list of files
    temp_dir = os.path.dirname(out_segment_path)
    file_list = os.path.join(temp_dir, "file_list.txt")
    
    # Write file list with proper path formatting for Windows
    with open(file_list, 'w', encoding='utf-8') as f:
        for png in png_files:
            # Use forward slashes and double backslashes for Windows paths
            file_path = os.path.join(frames_dir, png).replace('\\', '\\\\')
            f.write(f"file '{file_path}'\n")
    
    try:
        # Use concat demuxer with the file list
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-r", str(fps),
            "-i", file_list,
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            "-b:v", bitrate,
            out_segment_path
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg command: {e}")
        print(f"Command was: {' '.join(cmd)}")
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(file_list):
            try:
                os.remove(file_list)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_list}: {e}")

def concat_segments(segment_paths, out_path):
    """
    Concatenate segments using ffmpeg concat demuxer (no re-encode).
    Requires segments to have same codec, format, profiles.
    """
    # write list file
    list_file = os.path.join(os.path.dirname(out_path), "concat_list.txt")
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path]
    subprocess.run(cmd, check=True)
