import os
import time
from .utils import load_config, ensure_dirs
from .video_reader import read_and_write_batches, extract_audio
from .midas_depth import Midas, ModelType
from .stereo import batch_generate_stereo
from .projection import batch_project
from .stitch import batch_stack
from .frames_to_video import frames_to_segment
from .audio_utils import extract_audio, slice_audio, mux_audio_to_video
from .metadata_inject import inject_vr180_metadata
from .streaming import add_batch_to_hls

import shutil

cfg = load_config()
BATCH_SIZE = cfg["video"]["batch_size"]
FPS = cfg["video"]["output_fps"]
SEG_DIR = cfg["paths"]["segments_dir"]
TMP = cfg["paths"]["tmp_dir"]
ensure_dirs(TMP, SEG_DIR)

def process_batch(batch_tuple, midas_obj, cfg):
    batch_idx, start_frame, end_frame, frames = batch_tuple
    print(f"> Processing batch {batch_idx} frames {start_frame}-{end_frame} ({len(frames)})")
    # output folder structure for this batch
    batch_prefix = f"batch_{batch_idx:03d}"
    depth_out = os.path.join(cfg["paths"]["depth_dir"], batch_prefix)
    left_out = os.path.join(cfg["paths"]["left_dir"], batch_prefix)
    right_out = os.path.join(cfg["paths"]["right_dir"], batch_prefix)
    vr_left = os.path.join(cfg["paths"]["vr_left"], batch_prefix)
    vr_right = os.path.join(cfg["paths"]["vr_right"], batch_prefix)
    stereo_out = os.path.join(cfg["paths"]["stereo_dir"], batch_prefix)
    frames_combined = os.path.join(TMP, batch_prefix, "frames")
    os.makedirs(frames_combined, exist_ok=True)

    # 1) Depth
    midas_obj.predict_batch(frames, depth_out)

    # 2) Stereo
    depth_map_dict = {os.path.basename(p): os.path.join(depth_out, os.path.basename(p)) for p in frames}
    batch_generate_stereo(frames, depth_map_dict, left_out, right_out, max_shift=cfg["processing"]["max_shift"])

    # 3) Projection: left & right -> vr
    batch_project(left_out, vr_left, output_width=2048, field_of_view=cfg["processing"]["field_of_view"])
    batch_project(right_out, vr_right, output_width=2048, field_of_view=cfg["processing"]["field_of_view"])

    # 4) Stack side-by-side into stereo frames
    batch_stack(vr_left, vr_right, stereo_out, side_by_side=True)

    # 5) Copy stereo frames to frames_combined (ensure zeros-based ordering)
    stereo_files = sorted([f for f in os.listdir(stereo_out) if f.lower().endswith(".png")])
    for i, f in enumerate(stereo_files):
        src = os.path.join(stereo_out, f)
        dst = os.path.join(frames_combined, f"frame_{i:06d}.png")
        shutil.copy(src, dst)

    # 6) Return frames folder for this batch (final video will be built from all frames)
    return frames_combined, (start_frame, end_frame)

def main(input_video, add_audio=True):
    print("Reading video and splitting into batches...")
    # Cleanup previous run's artifacts to avoid appending across runs
    try:
        out_root = os.path.dirname(cfg["paths"]["final_output"])
        tmp_root = cfg["paths"]["tmp_dir"]
        seg_dir = cfg["paths"]["segments_dir"]
        # Remove tmp and full output tree
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
        # Recreate expected structure
        os.makedirs(tmp_root, exist_ok=True)
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(os.path.join(out_root, "stream"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "final_hls"), exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to reset output/tmp dirs: {e}")
    frames_dir = cfg["paths"]["frames_dir"]
    ensure_dirs(frames_dir)
    batches, fps, total_frames, w, h = read_and_write_batches(input_video, frames_dir, batch_size=BATCH_SIZE)
    print(f"Total frames: {total_frames}, FPS detected: {fps}, total batches: {len(batches)}")

    # prepare midas
    midas = Midas(ModelType[cfg["processing"]["midas_model"] if cfg["processing"]["midas_model"] in ModelType.__members__ else "MIDAS_SMALL"])
    # ensure model on right device already in class

    # optionally extract full audio once
    audio_temp = os.path.join(TMP, "full_audio.wav")
    if add_audio:
        extract_audio(input_video, audio_temp)

    # Accumulate all frames from batches into one ordered folder
    all_frames_dir = os.path.join(TMP, "all_frames")
    if os.path.exists(all_frames_dir):
        shutil.rmtree(all_frames_dir)
    os.makedirs(all_frames_dir, exist_ok=True)

    next_index = 0
    for i, batch in enumerate(batches):
        frames_dir_batch, (sframe, eframe) = process_batch(batch, midas, cfg)
        batch_files = sorted([f for f in os.listdir(frames_dir_batch) if f.lower().endswith(".png")])
        for f in batch_files:
            src = os.path.join(frames_dir_batch, f)
            dst = os.path.join(all_frames_dir, f"frame_{next_index:06d}.png")
            shutil.copy(src, dst)
            next_index += 1
        print(f"Batch {sframe}-{eframe} appended. Total frames so far: {next_index}")

        # Update HLS after each batch by rendering a small MP4 for this batch and appending to HLS
        try:
            tmp_batch_mp4 = os.path.join(TMP, f"batch_{i:03d}.mp4")
            frames_to_segment(frames_dir_batch, tmp_batch_mp4, fps=FPS, codec=cfg["video"]["codec"], bitrate=cfg["ffmpeg"]["bitrate"])
            hls_dir = os.path.join(os.path.dirname(cfg["paths"]["final_output"]), "stream")
            add_batch_to_hls(tmp_batch_mp4, stream_dir=hls_dir, fps=FPS)
        except Exception as e:
            print(f"Warning: HLS append failed for batch {i}: {e}")

    # Render single final video from all frames
    combined_out = cfg["paths"]["final_output"]
    frames_to_segment(all_frames_dir, combined_out, fps=FPS, codec=cfg["video"]["codec"], bitrate=cfg["ffmpeg"]["bitrate"])

    # Mux original full audio to the final video
    if add_audio:
        combined_with_audio = combined_out.replace(".mp4", "_audio.mp4")
        mux_audio_to_video(combined_out, audio_temp, combined_with_audio)
        os.replace(combined_with_audio, combined_out)

    # Generate HLS assets for streaming under output/stream
    hls_dir = os.path.join(os.path.dirname(combined_out), "stream")
    add_batch_to_hls(combined_out, stream_dir=hls_dir, fps=FPS)

    # final metadata injection
    final_with_meta = combined_out.replace(".mp4", "_vr_180.mp4")
    inject_vr180_metadata(combined_out, final_with_meta)
    print("✅ Finished. Final VR file:", final_with_meta)

    # Also generate a dedicated final HLS playlist from the final MP4 for frontend playback
    try:
        final_hls_dir = os.path.join(os.path.dirname(combined_out), "final_hls")
        add_batch_to_hls(final_with_meta, stream_dir=final_hls_dir, fps=FPS)
        print(f"✅ Final HLS generated at: {final_hls_dir}")
    except Exception as e:
        print(f"Warning: final HLS generation failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video>")
        sys.exit(1)
    main(sys.argv[1], add_audio=True)
