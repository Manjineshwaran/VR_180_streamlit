from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import re
import httpx
import logging
import traceback
from typing import Optional, Iterator
from src.main import main as process_main
from src.anaglyph_processor import main_anaglyph
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Use environment variables for Render deployment
INPUT_DIR = os.getenv("INPUT_DIR", os.path.join(APP_DIR, "input"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", os.path.join(APP_DIR, "output"))
STREAM_DIR = os.path.join(OUTPUTS_DIR, "stream")

def _find_hls_dir() -> str:
    """Find HLS directory with comprehensive logging and error handling."""
    logger.info("Searching for HLS directory")
    try:
        # Prefer top-level output/stream; fall back to src/output/stream (used by pipeline)
        candidates = [
            os.path.join(APP_DIR, "output", "stream"),
            os.path.join(APP_DIR, "src", "output", "stream"),
        ]
        logger.debug(f"HLS directory candidates: {candidates}")
        
        for d in candidates:
            if os.path.isdir(d):
                logger.info(f"Found HLS directory: {d}")
                return d
        # Default to first path; created on demand later by pipeline
        logger.warning(f"No existing HLS directory found, using default: {candidates[0]}")
        return candidates[0]
    except Exception as e:
        logger.error(f"Error finding HLS directory: {e}")
        logger.error(traceback.format_exc())
        # Fallback to first candidate
        return os.path.join(APP_DIR, "output", "stream")

HLS_DIR = _find_hls_dir()


app = FastAPI(title="VR 180 Backend", version="0.1.0")

logger.info("Initializing FastAPI application")

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware added successfully")
except Exception as e:
    logger.error(f"Failed to add CORS middleware: {e}")
    logger.error(traceback.format_exc())

# Ensure base dirs exist at startup and mount HLS
try:
    logger.info("Creating base directories")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(STREAM_DIR, exist_ok=True)
    logger.info(f"Created directories: INPUT_DIR={INPUT_DIR}, OUTPUTS_DIR={OUTPUTS_DIR}, STREAM_DIR={STREAM_DIR}")
except Exception as e:
    logger.error(f"Failed to create base directories: {e}")
    logger.error(traceback.format_exc())

# Create storage directory for Render
try:
    STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(APP_DIR, "storage"))
    os.makedirs(STORAGE_DIR, exist_ok=True)
    logger.info(f"Created storage directory: {STORAGE_DIR}")
except Exception as e:
    logger.error(f"Failed to create storage directory: {e}")
    logger.error(traceback.format_exc())

try:
    if os.path.isdir(HLS_DIR):
        app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")
        logger.info(f"Mounted HLS directory: {HLS_DIR}")
    else:
        logger.warning(f"HLS directory does not exist: {HLS_DIR}")
except Exception as e:
    logger.error(f"Failed to mount HLS directory: {e}")
    logger.error(traceback.format_exc())

# Mount final HLS directory (created after finalize step)
try:
    FINAL_HLS_DIR = os.path.join(OUTPUTS_DIR, "final_hls")
    os.makedirs(FINAL_HLS_DIR, exist_ok=True)
    app.mount("/hls_final", StaticFiles(directory=FINAL_HLS_DIR), name="hls_final")
    logger.info(f"Mounted final HLS directory: {FINAL_HLS_DIR}")
except Exception as e:
    logger.error(f"Failed to mount final HLS directory: {e}")
    logger.error(traceback.format_exc())


@app.get("/health")
def health() -> dict:
    """Health check endpoint with logging."""
    logger.info("Health check endpoint called")
    try:
        result = {"status": "ok"}
        logger.info("Health check successful")
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


def ensure_dirs() -> None:
    """Ensure all required directories exist with logging."""
    logger.info("Ensuring directories exist")
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        os.makedirs(STREAM_DIR, exist_ok=True)
        logger.info(f"Directories ensured: INPUT_DIR={INPUT_DIR}, OUTPUTS_DIR={OUTPUTS_DIR}, STREAM_DIR={STREAM_DIR}")
    except Exception as e:
        logger.error(f"Failed to ensure directories: {e}")
        logger.error(traceback.format_exc())
        raise


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Accept a multipart video file and store it under input/"""
    logger.info(f"Upload endpoint called with file: {file.filename}")
    try:
        ensure_dirs()
        
        if not file.filename:
            logger.error("Upload failed: filename missing")
            raise HTTPException(status_code=400, detail="Filename missing")
        
        base_name = os.path.basename(file.filename)
        name, ext = os.path.splitext(base_name)
        unique_name = f"{name}_{uuid4().hex[:8]}{ext or ''}"
        dest_path = os.path.join(INPUT_DIR, unique_name)
        
        logger.info(f"Processing upload: original_name={file.filename}, unique_name={unique_name}, dest_path={dest_path}")
        
        try:
            with open(dest_path, "wb") as f:
                chunk_count = 0
                total_bytes = 0
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    if chunk_count % 10 == 0:  # Log every 10 chunks
                        logger.debug(f"Upload progress: {chunk_count} chunks, {total_bytes} bytes")
                
                logger.info(f"Upload completed: {chunk_count} chunks, {total_bytes} bytes written to {dest_path}")
        except Exception as e:
            logger.error(f"File write failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
        finally:
            try:
                await file.close()
                logger.debug("File handle closed successfully")
            except Exception as e:
                logger.warning(f"Failed to close file handle: {e}")
        
        result = {"filename": unique_name, "path": dest_path}
        logger.info(f"Upload successful: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


def iter_file_range(path: str, start: int, end: int, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Iterate over file range with logging and error handling."""
    logger.debug(f"Starting file range iteration: path={path}, start={start}, end={end}, chunk_size={chunk_size}")
    try:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            chunk_count = 0
            total_yielded = 0
            
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    logger.warning(f"No more data to read at position {f.tell()}")
                    break
                yield data
                remaining -= len(data)
                chunk_count += 1
                total_yielded += len(data)
                
                if chunk_count % 100 == 0:  # Log every 100 chunks
                    logger.debug(f"File range progress: {chunk_count} chunks, {total_yielded} bytes yielded")
            
            logger.debug(f"File range iteration completed: {chunk_count} chunks, {total_yielded} bytes total")
    except Exception as e:
        logger.error(f"Error in file range iteration: {e}")
        logger.error(traceback.format_exc())
        raise


range_re = re.compile(r"bytes=(\d+)-(\d*)")


@app.get("/stream")
def stream_file(filename: str = Query(..., description="Filename in input/ or output/")):
    """HTTP range streaming for a local file.
    Priority search in output/ then input/.
    """
    logger.info(f"Stream endpoint called with filename: {filename}")
    try:
        # Backward-compat: accept old name without underscore
        base = os.path.basename(filename)
        if base == "final_output_vr180.mp4":
            base = "final_output_vr_180.mp4"
            logger.debug(f"Applied backward compatibility: {filename} -> {base}")
        
        outputs_path = os.path.join(OUTPUTS_DIR, base)
        input_path = os.path.join(INPUT_DIR, base)
        path = outputs_path if os.path.exists(outputs_path) else input_path
        
        logger.debug(f"File search: outputs_path={outputs_path}, input_path={input_path}, selected={path}")
        
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise HTTPException(status_code=404, detail="File not found")

        file_size = os.path.getsize(path)
        content_type = "video/mp4" if path.lower().endswith(".mp4") else "application/octet-stream"
        logger.info(f"File found: {path}, size={file_size}, content_type={content_type}")

        # Manually parse Range header from the ASGI scope via dependency injection
        # FastAPI exposes headers on the request object; we can access it via request in a dependency
        from fastapi import Request

        async def _range_response(request: Request):
            logger.debug(f"Range response called for {path}")
            try:
                range_header = request.headers.get("range")
                logger.debug(f"Range header: {range_header}")
                
                if range_header is None:
                    logger.info("No range header, returning full file")
                    return FileResponse(path, media_type=content_type)

                match = range_re.match(range_header)
                if not match:
                    logger.warning(f"Invalid range header format: {range_header}")
                    return Response(status_code=416)
                
                start = int(match.group(1))
                end = match.group(2)
                end = int(end) if end else file_size - 1
                
                logger.debug(f"Parsed range: start={start}, end={end}, file_size={file_size}")
                
                if start >= file_size or end >= file_size or start > end:
                    logger.warning(f"Invalid range: start={start}, end={end}, file_size={file_size}")
                    return Response(status_code=416)

                response = StreamingResponse(
                    iter_file_range(path, start, end),
                    media_type=content_type,
                    status_code=206,
                )
                response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                response.headers["Accept-Ranges"] = "bytes"
                response.headers["Content-Length"] = str(end - start + 1)
                
                logger.info(f"Range response created: {start}-{end}/{file_size}")
                return response
                
            except Exception as e:
                logger.error(f"Error in range response: {e}")
                logger.error(traceback.format_exc())
                return Response(status_code=500)

        return _range_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in stream endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stream failed: {e}")


@app.get("/hls_refresh")
def hls_refresh() -> dict:
    """Ensure HLS static mount is attached if the directory now exists."""
    logger.info("HLS refresh endpoint called")
    try:
        global HLS_DIR
        new_dir = _find_hls_dir()
        mounted = any([r.path == "/hls" for r in getattr(app, "routes", [])])
        
        logger.debug(f"HLS directory check: new_dir={new_dir}, mounted={mounted}")
        
        if os.path.isdir(new_dir) and not mounted:
            logger.info(f"Mounting HLS directory: {new_dir}")
            app.mount("/hls", StaticFiles(directory=new_dir), name="hls")
            HLS_DIR = new_dir
        else:
            logger.debug(f"HLS directory not mounted: isdir={os.path.isdir(new_dir)}, mounted={mounted}")
        
        # Try mounting final hls as well
        global FINAL_HLS_DIR
        final_dir = os.path.join(OUTPUTS_DIR, "final_hls")
        final_mounted = any([r.path == "/hls_final" for r in getattr(app, "routes", [])])
        
        logger.debug(f"Final HLS directory check: final_dir={final_dir}, final_mounted={final_mounted}")
        
        if os.path.isdir(final_dir) and not final_mounted:
            logger.info(f"Mounting final HLS directory: {final_dir}")
            app.mount("/hls_final", StaticFiles(directory=final_dir), name="hls_final")
            FINAL_HLS_DIR = final_dir
        else:
            logger.debug(f"Final HLS directory not mounted: isdir={os.path.isdir(final_dir)}, final_mounted={final_mounted}")
        
        result = {
            "mounted": os.path.isdir(new_dir), 
            "final_mounted": os.path.isdir(final_dir), 
            "dir": new_dir, 
            "final_dir": final_dir
        }
        logger.info(f"HLS refresh result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in HLS refresh: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "mounted": False, "final_mounted": False, "dir": "", "final_dir": ""}


def _ensure_hls_mounted_and_path() -> str:
    """Ensure HLS is mounted and return playlist path if present, else empty string."""
    logger.debug("Ensuring HLS is mounted and getting playlist path")
    try:
        global HLS_DIR
        new_dir = _find_hls_dir()
        mounted = any([r.path == "/hls" for r in getattr(app, "routes", [])])
        
        logger.debug(f"HLS mount check: new_dir={new_dir}, mounted={mounted}")
        
        if os.path.isdir(new_dir) and not mounted:
            logger.info(f"Mounting HLS directory: {new_dir}")
            app.mount("/hls", StaticFiles(directory=new_dir), name="hls")
            HLS_DIR = new_dir
        
        playlist = os.path.join(HLS_DIR, "output.m3u8") if HLS_DIR else ""
        exists = playlist and os.path.exists(playlist)
        
        logger.debug(f"Playlist check: playlist={playlist}, exists={exists}")
        return playlist if exists else ""
        
    except Exception as e:
        logger.error(f"Error ensuring HLS mount: {e}")
        logger.error(traceback.format_exc())
        return ""


@app.get("/hls_manifest")
def hls_manifest() -> dict:
    """Return the HLS playlist URL for the frontend if available."""
    logger.info("HLS manifest endpoint called")
    try:
        playlist_fs = _ensure_hls_mounted_and_path()
        logger.debug(f"Incremental HLS playlist: {playlist_fs}")
        
        # Prefer final HLS when available
        final_playlist = os.path.join(FINAL_HLS_DIR, "output.m3u8")
        final_exists = os.path.exists(final_playlist)
        logger.debug(f"Final HLS playlist: {final_playlist}, exists: {final_exists}")
        
        if final_exists:
            result = {"ready": True, "url": "/hls_final/output.m3u8", "type": "final"}
            logger.info(f"Returning final HLS manifest: {result}")
            return result
        
        # Fall back to incremental HLS
        if not playlist_fs:
            result = {"ready": False, "url": None}
            logger.info(f"No HLS playlist available: {result}")
            return result
        
        result = {"ready": True, "url": "/hls/output.m3u8", "type": "incremental"}
        logger.info(f"Returning incremental HLS manifest: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in HLS manifest: {e}")
        logger.error(traceback.format_exc())
        return {"ready": False, "url": None, "error": str(e)}


@app.get("/stream_status")
def stream_status(mode: str = Query("vr180", description="Processing mode: vr180 or anaglyph")) -> dict:
    """Report availability of both HLS playlist and final MP4 for the frontend to poll.
    Frontend can call this every ~40 seconds.
    """
    logger.info(f"Stream status endpoint called with mode: {mode}")
    try:
        # Ensure dirs and possible HLS mount
        ensure_dirs()
        playlist_fs = _ensure_hls_mounted_and_path()
        hls_exists = bool(playlist_fs)
        
        logger.debug(f"HLS status: playlist_fs={playlist_fs}, hls_exists={hls_exists}")
        
        # Check for appropriate output file based on mode
        if mode == "anaglyph":
            mp4_path = os.path.join(OUTPUTS_DIR, "final_output_anaglyph.mp4")
        else:
            mp4_path = os.path.join(OUTPUTS_DIR, "final_output_vr_180.mp4")
        
        mp4_exists = os.path.exists(mp4_path)
        logger.debug(f"MP4 status: mp4_path={mp4_path}, mp4_exists={mp4_exists}")
        
        result = {
            "hls": {"exists": hls_exists, "url": "/hls/output.m3u8" if hls_exists else None},
            "mp4": {"exists": mp4_exists, "url": f"/stream?filename={os.path.basename(mp4_path)}" if mp4_exists else None},
            "mode": mode
        }
        
        logger.info(f"Stream status result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in stream status: {e}")
        logger.error(traceback.format_exc())
        return {
            "hls": {"exists": False, "url": None},
            "mp4": {"exists": False, "url": None},
            "mode": mode,
            "error": str(e)
        }


@app.get("/download")
def download_file(filename: str = Query(..., description="Filename in outputs/ or input/")):
    logger.info(f"Download endpoint called with filename: {filename}")
    try:
        outputs_path = os.path.join(OUTPUTS_DIR, os.path.basename(filename))
        input_path = os.path.join(INPUT_DIR, os.path.basename(filename))
        path = outputs_path if os.path.exists(outputs_path) else input_path
        
        logger.debug(f"File search: outputs_path={outputs_path}, input_path={input_path}, selected={path}")
        
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        media_type = "video/mp4" if path.lower().endswith(".mp4") else "application/octet-stream"
        logger.info(f"File found: {path}, media_type={media_type}")
        
        return FileResponse(path, media_type=media_type, filename=os.path.basename(path))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in download: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")


@app.get("/proxy")
async def proxy_stream(url: str = Query(..., description="Remote video URL to proxy")):
    """Stream or download a remote resource via proxy with chunked transfer."""
    logger.info(f"Proxy endpoint called with URL: {url}")
    try:
        try:
            client = httpx.AsyncClient(timeout=None, follow_redirects=True)
            logger.debug("HTTP client created successfully")
        except Exception as e:
            logger.error(f"Failed to create HTTP client: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        async def _gen():
            logger.debug(f"Starting proxy stream for: {url}")
            try:
                async with client.stream("GET", url) as resp:
                    logger.debug(f"Proxy response status: {resp.status_code}")
                    if resp.status_code >= 400:
                        logger.error(f"Upstream error: {resp.status_code}")
                        raise HTTPException(status_code=resp.status_code, detail="Upstream error")
                    
                    chunk_count = 0
                    total_bytes = 0
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                        if chunk:
                            chunk_count += 1
                            total_bytes += len(chunk)
                            if chunk_count % 100 == 0:  # Log every 100 chunks
                                logger.debug(f"Proxy progress: {chunk_count} chunks, {total_bytes} bytes")
                            yield chunk
                    
                    logger.info(f"Proxy stream completed: {chunk_count} chunks, {total_bytes} bytes")
            except Exception as e:
                logger.error(f"Error in proxy stream: {e}")
                logger.error(traceback.format_exc())
                raise

        headers = {}
        # Best-effort content-type passthrough
        try:
            logger.debug("Attempting to get content type from HEAD request")
            async with client.head(url) as head_resp:
                ctype = head_resp.headers.get("content-type")
                if ctype:
                    headers["Content-Type"] = ctype
                    logger.debug(f"Content type detected: {ctype}")
        except Exception as e:
            logger.warning(f"Failed to get content type: {e}")

        logger.info(f"Starting proxy response with headers: {headers}")
        return StreamingResponse(_gen(), headers=headers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in proxy: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Proxy failed: {e}")


@app.post("/process")
async def process_video(
    request: Request,
    background_tasks: BackgroundTasks,
    filename: Optional[str] = Query(None, description="Existing filename under input/"),
    add_audio: bool = Query(True),
):
    """Kick off processing using src.main.main without duplicating logic.
    Accepts either an uploaded file (multipart) or a filename already present in input/.
    Tolerates empty/absent file fields.
    """
    logger.info(f"Process endpoint called with filename: {filename}, add_audio: {add_audio}")
    try:
        ensure_dirs()

        # Try to read an optional file from multipart form without triggering validation errors
        upload: Optional[UploadFile] = None
        try:
            logger.debug("Attempting to read multipart form")
            form = await request.form()
            maybe_file = form.get("file")
            if isinstance(maybe_file, UploadFile) and getattr(maybe_file, "filename", None):
                upload = maybe_file
                logger.info(f"Found uploaded file: {upload.filename}")
            else:
                logger.debug("No valid file found in form")
        except Exception as e:
            logger.warning(f"Failed to read form data: {e}")
            upload = None

        if upload is None and not filename:
            logger.error("No file or filename provided")
            raise HTTPException(status_code=400, detail="Provide either file or filename")

        input_path: Optional[str] = None
        if upload is not None:
            safe_name = os.path.basename(upload.filename)
            input_path = os.path.join(INPUT_DIR, safe_name)
            logger.info(f"Processing uploaded file: {upload.filename} -> {input_path}")
            try:
                # Overwrite if exists
                if os.path.exists(input_path):
                    logger.debug(f"Removing existing file: {input_path}")
                    os.remove(input_path)
                
                chunk_count = 0
                total_bytes = 0
                with open(input_path, "wb") as f:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)
                        if chunk_count % 10 == 0:  # Log every 10 chunks
                            logger.debug(f"Upload progress: {chunk_count} chunks, {total_bytes} bytes")
                
                logger.info(f"File saved successfully: {chunk_count} chunks, {total_bytes} bytes")
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Save failed: {e}")
            finally:
                try:
                    await upload.close()
                    logger.debug("Upload file handle closed")
                except Exception as e:
                    logger.warning(f"Failed to close upload file handle: {e}")
        else:
            # filename may arrive quoted from some UIs; strip quotes if present
            assert filename is not None
            cleaned = filename.strip('"')
            safe_name = os.path.basename(cleaned)
            candidate = os.path.join(INPUT_DIR, safe_name)
            logger.info(f"Processing existing file: {filename} -> {candidate}")
            
            if not os.path.exists(candidate):
                logger.error(f"File not found: {candidate}")
                raise HTTPException(status_code=404, detail="Filename not found under input/")
            input_path = candidate

        # Delegate to pipeline in background
        assert input_path is not None
        logger.info(f"Starting background processing: {input_path}, add_audio: {add_audio}")
        background_tasks.add_task(process_main, input_path, add_audio)
        
        result = {"status": "accepted", "input": input_path, "add_audio": add_audio}
        logger.info(f"Process request accepted: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Process failed: {e}")


@app.post("/process_anaglyph")
async def process_anaglyph_video(
    request: Request,
    background_tasks: BackgroundTasks,
    filename: Optional[str] = Query(None, description="Existing filename under input/"),
    add_audio: bool = Query(True),
):
    """Process video for anaglyph 3D effect.
    Accepts either an uploaded file (multipart) or a filename already present in input/.
    """
    logger.info(f"Process anaglyph endpoint called with filename: {filename}, add_audio: {add_audio}")
    try:
        ensure_dirs()

        # Try to read an optional file from multipart form without triggering validation errors
        upload: Optional[UploadFile] = None
        try:
            logger.debug("Attempting to read multipart form for anaglyph")
            form = await request.form()
            maybe_file = form.get("file")
            if isinstance(maybe_file, UploadFile) and getattr(maybe_file, "filename", None):
                upload = maybe_file
                logger.info(f"Found uploaded file for anaglyph: {upload.filename}")
            else:
                logger.debug("No valid file found in form for anaglyph")
        except Exception as e:
            logger.warning(f"Failed to read form data for anaglyph: {e}")
            upload = None

        if upload is None and not filename:
            logger.error("No file or filename provided for anaglyph")
            raise HTTPException(status_code=400, detail="Provide either file or filename")

        input_path: Optional[str] = None
        if upload is not None:
            safe_name = os.path.basename(upload.filename)
            input_path = os.path.join(INPUT_DIR, safe_name)
            logger.info(f"Processing uploaded file for anaglyph: {upload.filename} -> {input_path}")
            try:
                # Overwrite if exists
                if os.path.exists(input_path):
                    logger.debug(f"Removing existing file for anaglyph: {input_path}")
                    os.remove(input_path)
                
                chunk_count = 0
                total_bytes = 0
                with open(input_path, "wb") as f:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)
                        if chunk_count % 10 == 0:  # Log every 10 chunks
                            logger.debug(f"Anaglyph upload progress: {chunk_count} chunks, {total_bytes} bytes")
                
                logger.info(f"File saved successfully for anaglyph: {chunk_count} chunks, {total_bytes} bytes")
            except Exception as e:
                logger.error(f"Failed to save uploaded file for anaglyph: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Save failed: {e}")
            finally:
                try:
                    await upload.close()
                    logger.debug("Anaglyph upload file handle closed")
                except Exception as e:
                    logger.warning(f"Failed to close anaglyph upload file handle: {e}")
        else:
            # filename may arrive quoted from some UIs; strip quotes if present
            assert filename is not None
            cleaned = filename.strip('"')
            safe_name = os.path.basename(cleaned)
            candidate = os.path.join(INPUT_DIR, safe_name)
            logger.info(f"Processing existing file for anaglyph: {filename} -> {candidate}")
            
            if not os.path.exists(candidate):
                logger.error(f"File not found for anaglyph: {candidate}")
                raise HTTPException(status_code=404, detail="Filename not found under input/")
            input_path = candidate

        # Delegate to anaglyph pipeline in background
        assert input_path is not None
        logger.info(f"Starting background anaglyph processing: {input_path}, add_audio: {add_audio}")
        background_tasks.add_task(main_anaglyph, input_path, add_audio)
        
        result = {"status": "accepted", "input": input_path, "add_audio": add_audio, "mode": "anaglyph"}
        logger.info(f"Anaglyph process request accepted: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in anaglyph process endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Anaglyph process failed: {e}")


if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "0") == "1"
        logger.info(f"Starting FastAPI server on port {port}, reload={reload}")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        raise


