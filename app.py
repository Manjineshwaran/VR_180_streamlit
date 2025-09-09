import os
import streamlit as st
import time
import subprocess
import uuid
from pathlib import Path
import shutil
import sys
import cv2

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import main as process_main
from src.anaglyph_processor import main_anaglyph

# Constants
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(APP_DIR, "input")
OUTPUTS_DIR = os.path.join(APP_DIR, "output")
STREAM_DIR = os.path.join(OUTPUTS_DIR, "stream")
FINAL_HLS_DIR = os.path.join(OUTPUTS_DIR, "final_hls")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)
os.makedirs(FINAL_HLS_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to input directory with a unique name"""
    if not uploaded_file:
        return None
    
    # Create a unique filename
    file_ext = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(INPUT_DIR, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def process_video(input_path, mode="vr180", add_audio=True):
    """Process video using the appropriate processor"""
    try:
        # Clear previous output files
        for f in os.listdir(OUTPUTS_DIR):
            if f.endswith(('.mp4', '.m3u8', '.ts')):
                try:
                    os.remove(os.path.join(OUTPUTS_DIR, f))
                except:
                    pass
        
        # Process the video
        if mode == "vr180":
            output_path = process_main(input_path, add_audio=add_audio)
        else:  # anaglyph
            output_path = main_anaglyph(input_path, add_audio=add_audio)
            
        # If output_path is None, try to find the output file
        if output_path is None or not os.path.exists(output_path):
            # Look for the output file in the output directory
            for f in os.listdir(OUTPUTS_DIR):
                if f.startswith('final_output') and f.endswith('.mp4'):
                    output_path = os.path.join(OUTPUTS_DIR, f)
                    break
        
        return output_path if output_path and os.path.exists(output_path) else None
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.set_page_config(
        page_title="VR 180 Video Processor",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 VR 180 Video Processor")
    st.markdown("---")

    # Sidebar for upload and settings
    with st.sidebar:
        st.header("Upload & Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", 
                                       type=["mp4", "avi", "mov", "mkv"])
        
        # Processing options
        st.subheader("Processing Options")
        processing_mode = st.radio(
            "Processing Mode:",
            ["VR 180", "Anaglyph 3D"]
        )
        
        add_audio = st.checkbox("Include Audio", value=True)
        
        process_button = st.button("Process Video", type="primary", 
                                 disabled=uploaded_file is None)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        with st.spinner('Processing video...'):
            try:
                # Save the uploaded file
                input_path = save_uploaded_file(uploaded_file)
                if not input_path or not os.path.exists(input_path):
                    st.error("Failed to save the uploaded file. Please try again.")
                    return
                    
                # Verify the video file
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened() or not cap.read()[0]:
                    st.error("Error: Could not read the video file. The file might be corrupted or in an unsupported format.")
                    cap.release()
                    return
                cap.release()
                
                # Process the video
                if process_button:
                    with st.spinner('Processing video... This may take a while...'):
                        output_path = process_video(
                            input_path, 
                            mode=processing_mode.lower().replace(" ", ""),
                            add_audio=add_audio
                        )
                        
                        if output_path and os.path.exists(output_path):
                            st.success("Video processing completed successfully!")
                            st.video(output_path)
                            
                            # Add download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=f,
                                    file_name=os.path.basename(output_path),
                                    mime="video/mp4"
                                )
                        else:
                            st.error("Failed to process the video. Please check the logs for more details.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check the console for more details.")
                import traceback
                traceback.print_exc()
    
    with col1:
        st.subheader("Original Video")
        if uploaded_file:
            # Save the uploaded file
            input_path = save_uploaded_file(uploaded_file)
            if input_path:
                st.video(input_path)
    
    with col2:
        st.subheader("Processed Video")
        
        if process_button and uploaded_file:
            with st.spinner("Processing video... This may take a few minutes..."):
                # Process the video
                output_path = process_video(
                    input_path,
                    mode=processing_mode.lower().replace(" ", ""),
                    add_audio=add_audio
                )
                
                if output_path and os.path.exists(output_path):
                    st.success("Processing complete!")
                    
                    # Display the processed video
                    try:
                        video_file = open(output_path, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        
                        # Add download button
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name=f"processed_{os.path.basename(uploaded_file.name)}",
                            mime="video/mp4"
                        )
                    except Exception as e:
                        st.error(f"Error displaying video: {str(e)}")
                        st.error(f"Output path: {output_path}")
                        if os.path.exists(output_path):
                            st.error(f"File exists: {os.path.getsize(output_path)} bytes")
                else:
                    st.error("Failed to process the video. The output file was not found.")
                    st.error(f"Expected output path: {output_path}")
                    if output_path:
                        st.error(f"Output path exists: {os.path.exists(output_path)}")
                    else:
                        st.error("No output path was returned from the processing function.")
                    
                    # List files in output directory for debugging
                    try:
                        output_files = os.listdir(OUTPUTS_DIR)
                        st.info(f"Files in output directory: {output_files}")
                    except Exception as e:
                        st.error(f"Could not list output directory: {str(e)}")
    
    # Add some information
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        ### VR 180 Video Processor
        
        This application processes videos to create VR 180 or Anaglyph 3D content.
        
        **Features:**
        - Upload and process standard videos
        - Convert to VR 180 format
        - Create Anaglyph 3D videos
        - Optional audio preservation
        
        **How to use:**
        1. Upload a video file
        2. Select processing mode (VR 180 or Anaglyph 3D)
        3. Click 'Process Video'
        4. Download the processed video
        """)

if __name__ == "__main__":
    main()
