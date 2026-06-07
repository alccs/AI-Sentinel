"""
Video Source Abstractions - Supports File and RTSP streams.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np
import logging
import time
import threading
import subprocess

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame. Returns (success, frame)."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get the source FPS."""
        pass
    
    @abstractmethod
    def get_frame_count(self) -> int:
        """Get total frame count (-1 for streams)."""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get (width, height) of the video."""
        pass
    
    @abstractmethod
    def release(self):
        """Release resources."""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is open and valid."""
        pass


class FileVideoSource(VideoSource):
    """Video source from local MP4/video file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()
    
    def _open(self):
        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.file_path}")
        logger.info(f"Opened video file: {self.file_path}")
        logger.info(f"  Resolution: {self.get_resolution()}, FPS: {self.get_fps():.2f}, Frames: {self.get_frame_count()}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None
    
    def get_fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
    
    def get_frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0
    
    def get_resolution(self) -> Tuple[int, int]:
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)
    
    def get_current_position(self) -> float:
        """Get current position in seconds."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return 0.0
    
    def get_current_frame_number(self) -> int:
        """Get current frame number."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0
    
    def seek(self, frame_number: int):
        """Seek to a specific frame."""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info(f"Released video file: {self.file_path}")
    
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    def reset(self):
        """Reset to beginning of video."""
        self.seek(0)


class RTSPVideoSource(VideoSource):
    """
    Video source from RTSP camera stream using FFmpeg pipe.
    Solves H.265/HEVC decoding artifacts by using robust FFmpeg process.
    Implements a background thread to always ensure the latest frame is read, preventing buffer lag.
    """
    
    def __init__(self, rtsp_url: str, buffer_size: int = 1024*1024):
        self.rtsp_url = rtsp_url
        self.process = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self._frame_size = 0
        
        # Threading support
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._error_count = 0
        
        # Probe stream info first
        self._probe_stream_info()
        self._open_pipeline()
        
    def _probe_stream_info(self):
        """Use ffprobe to get resolution and FPS."""
        import json
        
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
                "-of", "json",
                self.rtsp_url
            ]
            
            logger.info(f"Probing RTSP stream: {self.rtsp_url}")
            
            # Prevent console window on Windows
            startupinfo = None
            creationflags = 0
            import os
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, creationflags=creationflags)
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed with return code {result.returncode}")
                # Don't raise immediately, try default or retry logic could be handled by caller
                logger.warning("Using default 1920x1080 resolution due to probe failure.")
                self.width = 1920
                self.height = 1080
                self.fps = 25.0
                return
            
            info = json.loads(result.stdout)
            
            if not info.get("streams"):
                logger.warning("No video streams found in RTSP source. Using defaults.")
                self.width = 1920
                self.height = 1080
                self.fps = 25.0
                return
            
            stream = info["streams"][0]
            self.width = int(stream["width"])
            self.height = int(stream["height"])
            
            if self.width <= 0 or self.height <= 0:
                 self.width = 1920
                 self.height = 1080

            # Parse FPS (e.g., "25/1")
            fps_str = stream.get("r_frame_rate", "25/1")
            if fps_str and '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                self.fps = num / den if den > 0 else 25.0
            else:
                self.fps = 25.0
            
            logger.info(f"Stream info: {self.width}x{self.height} @ {self.fps:.2f}fps")
            
        except Exception as e:
            logger.error(f"Failed to probe stream: {e}. Using default 1920x1080.")
            self.width = 1920
            self.height = 1080
            self.fps = 25.0

    def _open_pipeline(self):
        """Open FFmpeg subprocess pipeline and start reader thread."""
        
        # FFmpeg command to read RTSP and output MJPEG stream to pipe
        # Added -stimeout (socket timeout) and -recv_buffer_size
        cmd = [
            "ffmpeg",
            "-loglevel", "warning",        # Show warnings/errors
            "-rtsp_transport", "tcp",      # Force TCP for reliability
            "-timeout", "5000000",         # Socket timeout (5s)
            "-i", self.rtsp_url,           # Input URL
            "-f", "mjpeg",                 # Output format: MJPEG stream
            "-q:v", "2",                   # Quality: 2 (high quality)
            "-an",                         # No audio
            "-"                            # Output to stdout
        ]
        
        logger.info(f"Starting optimized FFmpeg pipeline (MJPEG): {' '.join(cmd)}")
        
        try:
            # Start background process
            # Capture stderr for debugging
            # Prevent console window on Windows
            startupinfo = None
            creationflags = 0
            import os
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
                
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,    # Capture stderr
                bufsize=10**7,             # Large buffer to handle spikes
                creationflags=creationflags
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            
            logger.info("FFmpeg MJPEG pipeline started successfully.")
            
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install FFmpeg.")
            raise
        except Exception as e:
            logger.error(f"Failed to start FFmpeg pipeline: {e}")
            raise
    
    def _reader_loop(self):
        """Background loop to read frames continuously."""
        import numpy as np
        buffer = bytearray()
        
        while self._running:
            proc = self.process
            if proc is None:
                break
                
            if proc.poll() is not None:
                # Process died
                stderr_output = proc.stderr.read()
                logger.error(f"FFmpeg process died. Exit code: {proc.returncode}")
                if stderr_output:
                     logger.error(f"FFmpeg stderr: {stderr_output.decode('utf-8', errors='ignore')}")
                
                # Attempt to restart internal pipeline if running
                if self._running:
                    logger.info("Restarting FFmpeg pipeline internally...")
                    time.sleep(2)
                    try:
                        self._open_pipeline()
                        continue
                    except Exception as e:
                        logger.error(f"Internal restart failed: {e}")
                        break
                else:
                    break
                
            try:
                # Read a chunk from stream (Increased chunk size for performance)
                chunk = proc.stdout.read(65536)
                if not chunk:
                    # End of stream or process died
                    time.sleep(0.1)
                    continue
                
                buffer.extend(chunk)
                
                # Process all complete frames in buffer
                while True:
                    # Find SOI (Start of Image)
                    start_idx = buffer.find(b'\xff\xd8')
                    
                    if start_idx == -1:
                        # No frame start found yet
                        # If buffer is getting too large with no start, clear it to prevent memory issues
                        if len(buffer) > 10 * 1024 * 1024:
                            logger.warning("Buffer overflow with no SOI, clearing...")
                            buffer = bytearray()
                        break
                    
                    # Find EOI (End of Image) AFTER the SOI
                    end_idx = buffer.find(b'\xff\xd9', start_idx)
                    
                    if end_idx != -1:
                        # We have a complete JPEG frame
                        jpg_data = buffer[start_idx : end_idx + 2]
                        
                        # Advance buffer past this frame
                        buffer = buffer[end_idx + 2:]
                        
                        # Decode to BGR
                        try:
                            # Use flags to avoid warnings if possible, usually IMREAD_COLOR is fine
                            image = cv2.imdecode(np.frombuffer(jpg_data, dtype='uint8'), cv2.IMREAD_COLOR)
                            
                            if image is not None:
                                with self._lock:
                                    self._latest_frame = image
                                    self._error_count = 0 
                            else:
                                logger.debug("Decoded image is None")
                        except Exception as decode_err:
                            logger.error(f"Decode error: {decode_err}")
                            
                    else:
                        # SOI found but no EOI yet. 
                        # Discard garbage before SOI to save space
                        if start_idx > 0:
                            buffer = buffer[start_idx:]
                        break
                    
            except Exception as e:
                # Suppress noisy decode errors for incomplete frames
                error_msg = str(e)
                if "bad Huffman code" in error_msg or "premature end" in error_msg or "extraneous bytes" in error_msg:
                    pass # Quietly ignore common corruption
                else:
                    logger.debug(f"Frame decode warning: {e}")
                
                self._error_count += 1
                if self._error_count > 100:
                    logger.warning("High frame error rate, resetting connection logic may trigger...")
                    # logic to restart if needed, or just let strict timeout handle it
                    self._error_count = 0 
        
        self._running = False
        logger.info("Reader loop exited.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest available frame."""
        with self._lock:
            if self._latest_frame is not None:
                 return True, self._latest_frame
            
        if self._running:
             import time
             # Wait up to 2 seconds for the first frame
             for _ in range(20): 
                 time.sleep(0.1)
                 with self._lock:
                    if self._latest_frame is not None:
                        return True, self._latest_frame
                 if not self._running:
                     break
        
        return False, None
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_frame_count(self) -> int:
        return -1
    
    def get_resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def release(self):
        self._running = False
        # Do not join thread here if called from within the thread (unlikely but possible)
        # to avoid deadlock.
        if self._thread and self._thread is not threading.current_thread() and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
        proc = self.process
        if proc:
            self.process = None 
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except:
                proc.kill()
            logger.info("FFmpeg pipeline stopped.")
    
    def is_opened(self) -> bool:
        return self._running
    
    def reconnect(self):
        logger.info("Attempting to reconnect RTSP stream...")
        self.release()
        import time
        time.sleep(1)
        
        # Fast reconnect: Skip probe if we already have resolution
        # This significantly reduces the "lag" gap during glitches
        if self.width == 0 or self.height == 0:
             self._probe_stream_info() 
             
        self._open_pipeline()

class NativeRTSPVideoSource(VideoSource):
    """
    Direct RTSP source using OpenCV (cv2.VideoCapture).
    Uses a background thread to continuously read frames to keep buffer empty (low latency).
    """
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._width = 0
        self._height = 0
        self._fps = 25.0
        
        self._open()
        
    def _open(self):
        logger.info(f"Opening Native RTSP stream: {self.rtsp_url}")
        # Force ffmpeg backend for consistency if available, or any
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to small if possible (backend dependent)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        if not self._cap.isOpened():
             logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
             raise ValueError("Cannot open RTSP stream")
             
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        if self._fps <= 0: self._fps = 25.0
        
        logger.info(f"RTSP Stream opened: {self._width}x{self._height} @ {self._fps:.2f}fps")
        
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        
    def _reader_loop(self):
        """Continuously read frames to keep latency low."""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                # Stream issue
                # Check if opened
                if not self._cap.isOpened():
                     break
                time.sleep(0.1)
                continue
            
            # Small sleep not needed if we want max freshness, blocking read throttles us to FPS
            
        logger.info("Native RTSP reader loop exited")
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame
        
        # Wait a bit if just started
        if self._running:
             time.sleep(0.1)
             with self._lock:
                 if self._latest_frame is not None:
                     return True, self._latest_frame
                     
        return False, None
        
    def get_fps(self) -> float:
        return self._fps
        
    def get_frame_count(self) -> int:
        return -1
        
    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)
        
    def release(self):
        self._running = False
        # Avoid joining thread if called from within it
        if self._thread and threading.current_thread() != self._thread:
             self._thread.join(timeout=1.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
            
    def is_opened(self) -> bool:
         return self._cap is not None and self._cap.isOpened()
         
    def reconnect(self):
        logger.info("Reconnecting Native RTSP...")
        self.release()
        import time
        time.sleep(1)
        try:
             self._open()
        except:
             pass
