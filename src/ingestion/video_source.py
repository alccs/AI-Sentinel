"""
Video Source Abstractions - Supports File and RTSP streams.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np
import logging
import time

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
    """
    
    def __init__(self, rtsp_url: str, buffer_size: int = 1024*1024):
        self.rtsp_url = rtsp_url
        self.process = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self._frame_size = 0
        
        # Probe stream info first
        self._probe_stream_info()
        self._open_pipeline()
        
    def _probe_stream_info(self):
        """Use ffprobe to get resolution and FPS."""
        import subprocess
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed with return code {result.returncode}")
                logger.error(f"ffprobe stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            info = json.loads(result.stdout)
            
            if not info.get("streams"):
                raise ValueError("No video streams found in RTSP source")
            
            stream = info["streams"][0]
            self.width = int(stream["width"])
            self.height = int(stream["height"])
            
            if self.width <= 0 or self.height <= 0:
                raise ValueError(f"Invalid resolution detected: {self.width}x{self.height}")

            # Parse FPS (e.g., "25/1")
            fps_str = stream.get("r_frame_rate", "25/1")
            if fps_str and '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                self.fps = num / den if den > 0 else 25.0
            else:
                self.fps = 25.0
            
            logger.info(f"Stream info: {self.width}x{self.height} @ {self.fps:.2f}fps")
            
        except subprocess.TimeoutExpired:
            logger.error("ffprobe timeout - RTSP stream may be unreachable")
            raise
        except FileNotFoundError:
            logger.error("ffprobe not found. Please install FFmpeg.")
            raise
        except Exception as e:
            logger.error(f"Failed to probe stream: {e}. Using default 1920x1080.")
            # Don't raise here, use defaults
            self.width = 1920
            self.height = 1080
            self.fps = 25.0

    def _open_pipeline(self):
        """Open FFmpeg subprocess pipeline."""
        import subprocess
        
        # FFmpeg command to read RTSP and output raw BGR frames to pipe
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",      # Force TCP for reliability
            "-i", self.rtsp_url,           # Input URL
            "-f", "image2pipe",            # Output format: image pipe
            "-pix_fmt", "bgr24",           # Output pixel format: BGR (for OpenCV)
            "-vcodec", "rawvideo",         # Video codec: raw
            "-an",                         # No audio
            "-"                            # Output to stdout
        ]
        
        logger.info(f"Starting FFmpeg pipeline: {' '.join(cmd)}")
        
        try:
            # Start background process
            # bufsize set to hold a few frames
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for debugging
                bufsize=10**7
            )
            
            # Calculate size of one frame in bytes: width * height * 3 channels
            self._frame_size = self.width * self.height * 3
            if self._frame_size <= 0:
                # Fallback
                self._frame_size = 1920 * 1080 * 3
                
            logger.info(f"FFmpeg pipeline started successfully. Frame size: {self._frame_size} bytes")
            
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install FFmpeg.")
            raise
        except Exception as e:
            logger.error(f"Failed to start FFmpeg pipeline: {e}")
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Capture local reference for thread safety during release
        proc = self.process
        if proc is None or proc.poll() is not None:
            # Check if process ended with error
            if proc and proc.poll() is not None:
                # Process ended, check stderr for error info
                try:
                    stderr_output = proc.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        logger.error(f"FFmpeg process ended with error: {stderr_output}")
                except:
                    pass
            return False, None
        
        try:
            # Read exact number of bytes for one frame
            raw_frame = proc.stdout.read(self._frame_size)
            
            if len(raw_frame) != self._frame_size:
                # Determine if it's a real error or just shutdown
                if len(raw_frame) > 0:
                     logger.warning(f"Incomplete frame read from pipe. Expected {self._frame_size}, got {len(raw_frame)}")
                return False, None
            
            # Convert to numpy array
            image = np.frombuffer(raw_frame, dtype='uint8')
            image = image.reshape((self.height, self.width, 3))
            
            return True, image
            
        except Exception as e:
            logger.error(f"Error reading frame from pipe: {e}")
            return False, None
    
    def get_fps(self) -> float:
        return self.fps
    
    def get_frame_count(self) -> int:
        return -1
    
    def get_resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def release(self):
        proc = self.process
        if proc:
            self.process = None # Detach first
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except:
                proc.kill()
            logger.info("FFmpeg pipeline stopped.")
    
    def is_opened(self) -> bool:
        return self.process is not None and self.process.poll() is None
    
    def reconnect(self):
        logger.info("Attempting to reconnect RTSP stream...")
        self.release()
        time.sleep(1)
        # Re-probe to handle potential resolution changes or ensure valid state
        self._probe_stream_info() 
        self._open_pipeline()
