import os
import cv2
import numpy as np
import shutil
import logging
import traceback
from .utils import load_config, ensure_dirs
from .video_reader import read_and_write_batches, extract_audio
from .midas_depth import Midas, ModelType
from .stereo import batch_generate_stereo
from .projection import batch_project
from .frames_to_video import frames_to_segment
from .audio_utils import extract_audio, mux_audio_to_video
from .streaming import add_batch_to_hls

# Configure logging
logger = logging.getLogger(__name__)

def create_anaglyph_from_stereo(left_path, right_path, output_path):
    """Create anaglyph image from left and right eye images"""
    logger.debug(f"Creating anaglyph from stereo: left={left_path}, right={right_path}, output={output_path}")
    try:
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        
        if left is None or right is None:
            logger.warning(f"Skipping {left_path} or {right_path}, could not read image.")
            return False

        logger.debug(f"Successfully loaded images: left shape={left.shape}, right shape={right.shape}")

        # Create anaglyph: Red from left eye, Green and Blue from right eye
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 2]   # Red channel from left eye
        anaglyph[:, :, 1] = right[:, :, 1]  # Green from right eye
        anaglyph[:, :, 2] = right[:, :, 0]  # Blue from right eye

        logger.debug(f"Created anaglyph with shape: {anaglyph.shape}")
        
        success = cv2.imwrite(output_path, anaglyph)
        if success:
            logger.debug(f"Successfully saved anaglyph: {output_path}")
        else:
            logger.error(f"Failed to save anaglyph: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error creating anaglyph from stereo: {e}")
        logger.error(traceback.format_exc())
        return False

def process_anaglyph_batch(batch_tuple, midas_obj, cfg):
    """Process a batch of frames for anaglyph generation"""
    batch_idx, start_frame, end_frame, frames = batch_tuple
    logger.info(f"Processing anaglyph batch {batch_idx} frames {start_frame}-{end_frame} ({len(frames)})")
    
    try:
        # Output folder structure for this batch
        batch_prefix = f"batch_{batch_idx:03d}"
        depth_out = os.path.join(cfg["paths"]["depth_dir"], batch_prefix)
        left_out = os.path.join(cfg["paths"]["left_dir"], batch_prefix)
        right_out = os.path.join(cfg["paths"]["right_dir"], batch_prefix)
        anaglyph_out = os.path.join(cfg["paths"]["anaglyph_dir"], batch_prefix)
        frames_combined = os.path.join(cfg["paths"]["tmp_dir"], batch_prefix, "frames")
        
        logger.debug(f"Batch directories: depth_out={depth_out}, left_out={left_out}, right_out={right_out}, anaglyph_out={anaglyph_out}, frames_combined={frames_combined}")
        
        os.makedirs(frames_combined, exist_ok=True)
        logger.debug("Created frames_combined directory")

        # 1) Depth estimation
        logger.info(f"Starting depth estimation for batch {batch_idx}")
        try:
            midas_obj.predict_batch(frames, depth_out)
            logger.info(f"Depth estimation completed for batch {batch_idx}")
        except Exception as e:
            logger.error(f"Depth estimation failed for batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            raise

        # 2) Stereo generation
        logger.info(f"Starting stereo generation for batch {batch_idx}")
        try:
            depth_map_dict = {os.path.basename(p): os.path.join(depth_out, os.path.basename(p)) for p in frames}
            batch_generate_stereo(frames, depth_map_dict, left_out, right_out, max_shift=cfg["processing"]["max_shift"])
            logger.info(f"Stereo generation completed for batch {batch_idx}")
        except Exception as e:
            logger.error(f"Stereo generation failed for batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            raise

        # 3) Create anaglyph images from left and right stereo pairs
        logger.info(f"Starting anaglyph creation for batch {batch_idx}")
        try:
            left_files = sorted([f for f in os.listdir(left_out) if f.lower().endswith(".png")])
            right_files = sorted([f for f in os.listdir(right_out) if f.lower().endswith(".png")])
            
            logger.debug(f"Found {len(left_files)} left files and {len(right_files)} right files")
            
            os.makedirs(anaglyph_out, exist_ok=True)
            
            anaglyph_count = 0
            for left_file, right_file in zip(left_files, right_files):
                left_path = os.path.join(left_out, left_file)
                right_path = os.path.join(right_out, right_file)
                anaglyph_path = os.path.join(anaglyph_out, f"anaglyph_{left_file}")
                
                if create_anaglyph_from_stereo(left_path, right_path, anaglyph_path):
                    anaglyph_count += 1
                else:
                    logger.warning(f"Failed to create anaglyph for {left_file}")
            
            logger.info(f"Created {anaglyph_count} anaglyph images for batch {batch_idx}")
        except Exception as e:
            logger.error(f"Anaglyph creation failed for batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            raise

        # 4) Copy anaglyph frames to frames_combined (ensure zeros-based ordering)
        logger.info(f"Copying anaglyph frames for batch {batch_idx}")
        try:
            anaglyph_files = sorted([f for f in os.listdir(anaglyph_out) if f.lower().endswith(".png")])
            logger.debug(f"Found {len(anaglyph_files)} anaglyph files to copy")
            
            for i, f in enumerate(anaglyph_files):
                src = os.path.join(anaglyph_out, f)
                dst = os.path.join(frames_combined, f"frame_{i:06d}.png")
                shutil.copy(src, dst)
            
            logger.info(f"Copied {len(anaglyph_files)} anaglyph frames to frames_combined")
        except Exception as e:
            logger.error(f"Failed to copy anaglyph frames for batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Successfully completed anaglyph batch {batch_idx}")
        return frames_combined, (start_frame, end_frame)
        
    except Exception as e:
        logger.error(f"Error processing anaglyph batch {batch_idx}: {e}")
        logger.error(traceback.format_exc())
        raise

def main_anaglyph(input_video, add_audio=True):
    """Main function for anaglyph processing"""
    cfg = load_config()
    BATCH_SIZE = cfg["video"]["batch_size"]
    FPS = cfg["video"]["output_fps"]
    TMP = cfg["paths"]["tmp_dir"]
    
    # Add anaglyph directory to config paths
    anaglyph_dir = os.path.join(TMP, "anaglyph")
    cfg["paths"]["anaglyph_dir"] = anaglyph_dir
    
    ensure_dirs(TMP, anaglyph_dir)
    
    print("Reading video and splitting into batches for anaglyph processing...")
    
    # Cleanup previous run's artifacts
    try:
        out_root = os.path.dirname(cfg["paths"]["final_output"])
        tmp_root = cfg["paths"]["tmp_dir"]
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
        os.makedirs(tmp_root, exist_ok=True)
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(os.path.join(out_root, "stream"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "final_hls"), exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to reset output/tmp dirs: {e}")
    
    frames_dir = cfg["paths"]["frames_dir"]
    ensure_dirs(frames_dir)
    batches, fps, total_frames, w, h = read_and_write_batches(input_video, frames_dir, batch_size=BATCH_SIZE)
    print(f"Total frames: {total_frames}, FPS detected: {fps}, total batches: {len(batches)}")

    # Prepare MiDaS model
    midas = Midas(ModelType[cfg["processing"]["midas_model"] if cfg["processing"]["midas_model"] in ModelType.__members__ else "MIDAS_SMALL"])

    # Extract audio if needed
    audio_temp = os.path.join(TMP, "full_audio.wav")
    if add_audio:
        extract_audio(input_video, audio_temp)

    # Accumulate all anaglyph frames from batches
    all_frames_dir = os.path.join(TMP, "all_anaglyph_frames")
    if os.path.exists(all_frames_dir):
        shutil.rmtree(all_frames_dir)
    os.makedirs(all_frames_dir, exist_ok=True)

    next_index = 0
    for i, batch in enumerate(batches):
        frames_dir_batch, (sframe, eframe) = process_anaglyph_batch(batch, midas, cfg)
        batch_files = sorted([f for f in os.listdir(frames_dir_batch) if f.lower().endswith(".png")])
        for f in batch_files:
            src = os.path.join(frames_dir_batch, f)
            dst = os.path.join(all_frames_dir, f"frame_{next_index:06d}.png")
            shutil.copy(src, dst)
            next_index += 1
        print(f"Anaglyph batch {sframe}-{eframe} appended. Total frames so far: {next_index}")

        # Update HLS after each batch by rendering a small MP4 for this batch and appending to HLS
        try:
            tmp_batch_mp4 = os.path.join(TMP, f"anaglyph_batch_{i:03d}.mp4")
            frames_to_segment(frames_dir_batch, tmp_batch_mp4, fps=FPS, codec=cfg["video"]["codec"], bitrate=cfg["ffmpeg"]["bitrate"])
            hls_dir = os.path.join(os.path.dirname(cfg["paths"]["final_output"]), "stream")
            add_batch_to_hls(tmp_batch_mp4, stream_dir=hls_dir, fps=FPS)
        except Exception as e:
            print(f"Warning: HLS append failed for anaglyph batch {i}: {e}")

    # Render final anaglyph video from all frames
    combined_out = cfg["paths"]["final_output"].replace(".mp4", "_anaglyph.mp4")
    frames_to_segment(all_frames_dir, combined_out, fps=FPS, codec=cfg["video"]["codec"], bitrate=cfg["ffmpeg"]["bitrate"])

    # Mux original audio to the final video
    if add_audio:
        combined_with_audio = combined_out.replace(".mp4", "_audio.mp4")
        mux_audio_to_video(combined_out, audio_temp, combined_with_audio)
        os.replace(combined_with_audio, combined_out)

    # Generate HLS assets for streaming under output/stream
    hls_dir = os.path.join(os.path.dirname(combined_out), "stream")
    add_batch_to_hls(combined_out, stream_dir=hls_dir, fps=FPS)

    print("✅ Anaglyph processing finished. Final file:", combined_out)

    # Also generate a dedicated final HLS playlist from the final MP4 for frontend playback
    try:
        final_hls_dir = os.path.join(os.path.dirname(combined_out), "final_hls")
        add_batch_to_hls(combined_out, stream_dir=final_hls_dir, fps=FPS)
        print(f"✅ Final anaglyph HLS generated at: {final_hls_dir}")
    except Exception as e:
        print(f"Warning: final anaglyph HLS generation failed: {e}")

    return combined_out
