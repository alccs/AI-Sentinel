"""
Video Ingestion Service - Captures frames and manages decimation for AI analysis.
"""
import threading
import time
import os
import cv2
import numpy as np
import re
from typing import Optional, Generator, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .video_source import VideoSource, FileVideoSource, RTSPVideoSource
from ..common.types import Frame
from ..common.queue_manager import queue_manager

logger = logging.getLogger(__name__)

class TimeSynchronizer:
    """
    基于 VLM 的后台时间同步机制
    
    使用独立的后台线程每 10 秒调用一次 VLM 来识别视频中的时间戳，
    避免阻塞主视频流读取。
    """
    
    def __init__(self, vlm_client, roi_config: tuple):
        """
        初始化时间同步器
        
        Args:
            vlm_client: VLM 客户端实例
            roi_config: ROI 配置 (x_ratio, y_ratio, w_ratio, h_ratio)
        """
        self.vlm_client = vlm_client
        self.roi_config = roi_config
        
        # 核心时间基准变量
        self.base_video_time: Optional[datetime] = None  # VLM 读到的时间
        self.base_system_time: Optional[datetime] = None  # 读到该时间时的系统时间
        
        # 线程控制
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 当前帧缓存（用于 VLM 分析）
        self._current_frame: Optional[np.ndarray] = None
        
        # 统计信息
        self._stats = {
            "sync_attempts": 0,
            "sync_successes": 0,
            "sync_failures": 0,
            "last_sync_time": None,
            "last_parsed_time": "N/A"
        }
        
        logger.info("TimeSynchronizer initialized")
    
    def start(self):
        """启动后台同步线程"""
        if self._running:
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("TimeSynchronizer started")
    
    def stop(self):
        """停止后台同步线程"""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=2.0)
            self._sync_thread = None
        logger.info("TimeSynchronizer stopped")
    
    def update_frame(self, frame: np.ndarray):
        """更新当前帧（由主视频循环调用）"""
        with self._lock:
            self._current_frame = frame.copy() if frame is not None else None
    
    def _sync_loop(self):
        """后台同步循环，每 10 秒运行一次"""
        while self._running:
            try:
                # 等待 10 秒或直到停止
                for _ in range(100):  # 10秒 = 100 * 0.1秒
                    if not self._running:
                        return
                    time.sleep(0.1)
                
                if not self._running:
                    return
                
                # 执行同步
                self._perform_sync()
                
            except Exception as e:
                logger.error(f"TimeSynchronizer sync loop error: {e}")
                self._stats["sync_failures"] += 1
    
    def _perform_sync(self):
        """执行一次时间同步"""
        self._stats["sync_attempts"] += 1
        
        # 获取当前帧
        with self._lock:
            if self._current_frame is None:
                logger.debug("No frame available for sync")
                return
            frame = self._current_frame.copy()
        
        try:
            # 裁剪 ROI 区域
            roi_frame = self._extract_roi(frame)
            if roi_frame is None:
                logger.warning("Failed to extract ROI for sync")
                return
            
            # 保存临时图片用于 VLM 分析
            temp_path = f"temp_sync_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, roi_frame)
            
            try:
                # 调用 VLM
                prompt = "识别图片中的日期和时间，格式严格为 YYYY-MM-DD HH:MM:SS，只输出时间字符串，不要包含任何其他内容。"
                response = self.vlm_client.analyze_frame(temp_path, prompt)
                
                # 解析返回的时间字符串
                parsed_time = self._parse_time_response(response)
                
                if parsed_time:
                    # 成功解析，更新基准时间
                    current_system_time = datetime.now()
                    
                    with self._lock:
                        self.base_video_time = parsed_time
                        self.base_system_time = current_system_time
                    
                    self._stats["sync_successes"] += 1
                    self._stats["last_sync_time"] = current_system_time
                    self._stats["last_parsed_time"] = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    logger.info(f"Time sync successful: {self._stats['last_parsed_time']}")
                else:
                    # VLM 幻觉或解析失败，跳过本次
                    self._stats["sync_failures"] += 1
                    logger.debug(f"Time sync failed to parse: {response}")
                    
            finally:
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Time sync error: {e}")
            self._stats["sync_failures"] += 1
    
    def _extract_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """从帧中提取 ROI 区域"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        rx, ry, rw, rh = self.roi_config
        
        # 计算像素坐标
        x1 = int(w * rx)
        y1 = int(h * ry)
        w_px = int(w * rw)
        h_px = int(h * rh)
        
        # 边界检查
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = min(w, x1 + w_px)
        y2 = min(h, y1 + h_px)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2]
    
    def _parse_time_response(self, response: str) -> Optional[datetime]:
        """解析 VLM 返回的时间字符串"""
        if not response or not isinstance(response, str):
            return None
        
        # 清理文本
        text = re.sub(r'\s+', ' ', response.strip())
        
        # 尝试多种时间格式
        patterns = [
            # 标准格式: 2026-01-22 01:09:18
            r'(\d{4})[-/.](\d{2})[-/.](\d{2})\s+(\d{2}):(\d{2}):(\d{2})',
            # 紧凑格式
            r'(\d{4})[-/.](\d{2})[-/.](\d{2})[\s]*(\d{2}):(\d{2}):(\d{2})',
            # 分离的数字
            r'(\d{4})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})',
            # 混合分隔符
            r'(\d{4})[-/.](\d{2})[-/.](\d{2})[:\s](\d{2}):(\d{2}):(\d{2})',
        ]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, text)
                if match:
                    year, month, day, hour, minute, second = match.groups()
                    
                    # 验证数值范围
                    if (1900 <= int(year) <= 2100 and 
                        1 <= int(month) <= 12 and 
                        1 <= int(day) <= 31 and 
                        0 <= int(hour) <= 23 and 
                        0 <= int(minute) <= 59 and 
                        0 <= int(second) <= 59):
                        
                        dt_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, OverflowError):
                continue
        
        return None
    
    def get_current_video_time(self) -> datetime:
        """
        获取当前视频时间（主接口，零延迟）
        
        Returns:
            当前估算的视频时间
        """
        with self._lock:
            if self.base_video_time is None or self.base_system_time is None:
                # 系统刚启动还没完成第一次 VLM 同步，返回系统时间
                return datetime.now()
            
            # 计算时间偏移
            elapsed = datetime.now() - self.base_system_time
            return self.base_video_time + elapsed
    
    def get_stats(self) -> dict:
        """获取同步统计信息"""
        return self._stats.copy()

class VideoIngestionService:
    """
    Service that reads video frames and manages two output streams:
    1. Full-rate stream for UI display (via generator)
    2. Decimated frames for AI analysis (via queue)
    """
    
    def __init__(
        self,
        source: VideoSource,
        camera_id: str = "cam_01",
        analysis_interval: float = 1.0,  # Seconds between AI frames
        frame_save_dir: str = "./data/frames",
        max_queue_size: int = 100,
        loop_video: bool = True,  # Loop for file sources
        max_saved_frames: int = 50,  # Maximum number of saved frame images
        roi_config: tuple = (0.65, 0.85, 0.35, 0.15),  # Time ROI (x, y, w, h) ratios
        vlm_client=None,  # VLM client for time synchronization
        deletion_callback: Optional[Callable[[str], None]] = None, # Callback for sync deletion
    ):
        self.source = source
        self.camera_id = camera_id
        self.analysis_interval = analysis_interval
        self.frame_save_dir = Path(frame_save_dir)
        self.max_queue_size = max_queue_size
        self.loop_video = loop_video
        self.max_saved_frames = max_saved_frames
        self.deletion_callback = deletion_callback
        
        self._saved_files = [] # Track files for FIFO cleanup
        
        # Scan for existing files to enforce limit eventually
        if self.frame_save_dir.exists():
            try:
                # Find all jpgs, recursively
                # This might be slow if many files, but necessary for persistence
                existing = sorted(self.frame_save_dir.rglob("*.jpg"), key=os.path.getctime)
                for p in existing:
                    self._saved_files.append((str(p), p.name))
                logger.info(f"Discovered {len(self._saved_files)} existing frames.")
            except Exception as e:
                logger.warning(f"Failed to scan existing frames: {e}")
        
        if max_saved_frames > 2000:
             logger.warning(f"High max_saved_frames ({max_saved_frames}) may consume significant disk space.")
             
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Background processing frame queue
        import queue
        self._processing_queue = queue.Queue(maxsize=max_queue_size)
        
        # 初始化 VLM 时间同步器（替换 OCR）
        if vlm_client is not None:
            self.time_synchronizer = TimeSynchronizer(vlm_client, roi_config)
            logger.info("VLM-based time synchronization enabled")
        else:
            self.time_synchronizer = None
            logger.warning("No VLM client provided, using system time")
            
        # Frame state
        self._current_frame: Optional[np.ndarray] = None
        self._frame_number = 0
        self._last_analysis_time = 0.0
        self._frames_for_analysis = 0
        self._saved_frame_index = 0  # Circular buffer index
        
        # Statistics
        self._stats = {
            "total_frames": 0,
            "analysis_frames": 0,
            "dropped_frames": 0,
            "start_time": None,
            "pending_processing": 0, # Stats processing queue
        }
        
        # Ensure frame save directory exists
        self.frame_save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoIngestionService initialized for {camera_id}")
        logger.info(f"  Analysis interval: {analysis_interval}s, Save dir: {frame_save_dir}, Max frames: {max_saved_frames}")

    def get_current_video_time(self) -> datetime:
        """获取当前视频时间（通过 VLM 时间同步器）"""
        if self.time_synchronizer:
            return self.time_synchronizer.get_current_video_time()
        else:
            return datetime.now()

    def get_time_sync_stats(self) -> dict:
        """获取时间同步统计信息"""
        if self.time_synchronizer:
            return self.time_synchronizer.get_stats()
        else:
            return {"status": "disabled"}
    
    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> 'VideoIngestionService':
        """Factory method to create service from video file."""
        source = FileVideoSource(file_path)
        return cls(source, **kwargs)
    
    @classmethod
    def from_rtsp(cls, rtsp_url: str, **kwargs) -> 'VideoIngestionService':
        """Factory method to create service from RTSP stream."""
        source = RTSPVideoSource(rtsp_url)
        return cls(source, **kwargs)
    
    def start(self):
        """Start the ingestion service in a background thread."""
        if self._running:
            logger.warning("Service already running")
            return
        
        self._running = True
        self._stats["start_time"] = datetime.now()
        
        # 启动 VLM 时间同步器
        if self.time_synchronizer:
            self.time_synchronizer.start()
        
        # Start processing loop first
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        # Start capture loop
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"VideoIngestionService started for {self.camera_id}")
    
    def stop(self):
        """Stop the ingestion service."""
        self._running = False
        
        # 停止 VLM 时间同步器
        if self.time_synchronizer:
            self.time_synchronizer.stop()
        
        # Release source FIRST to unblock any pending reads in the thread
        if self.source:
             self.source.release()
             
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        # Wait for processing queue to drain or thread to exit
        if self._processing_thread:
             # Push a sentinel to unblock queue get if needed, or rely on _running check timeout
             self._processing_thread.join(timeout=2.0)
             self._processing_thread = None
            
        # Ensure cleanup (idempotent)
        if self.source:
             self.source.release()
             
        logger.info(f"VideoIngestionService stopped. Stats: {self._stats}")
    
    def is_running(self) -> bool:
        return self._running
    
    def _capture_loop(self):
        """Main capture loop running in background thread."""
        fps = self.source.get_fps()
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # Calculate how many frames to skip between analysis
        analysis_frame_interval = int(self.analysis_interval * fps)
        logger.info(f"Capture loop: FPS={fps:.1f}, Analysis every {analysis_frame_interval} frames")
        
        while self._running:
            loop_start = time.time()
            
            # Read frame
            success, frame = self.source.read_frame()
            
            if not success or frame is None:
                # Handle end of file or stream error
                if isinstance(self.source, FileVideoSource):
                    if self.loop_video:
                        logger.info("Video ended, looping...")
                        self.source.reset()
                        self._frame_number = 0
                        continue
                    else:
                        logger.info("Video ended")
                        break
                else:
                    # RTSP stream error
                    if not self._running:
                        break

                    # print("❌ 视频流读取中断，尝试重连...")
                    logger.warning("Stream read error, attempting reconnect...")
                    time.sleep(1) # Prevent busy loop
                    
                    if not self._running:
                        break

                    if hasattr(self.source, 'reconnect'):
                        # Try to reopen
                        try:
                            self.source.reconnect()
                            if not self._running:
                                self.source.release()
                                break
                            print("✅ 视频流重连成功！")
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e}")
                            print(f"⚠️ 重连失败: {e}")
                            time.sleep(2)
                    continue
            
            self._frame_number += 1
            self._stats["total_frames"] += 1
            
            # Update current frame for UI streaming (always)
            with self._lock:
                self._current_frame = frame.copy()
            
            # 更新时间同步器的当前帧
            if self.time_synchronizer:
                self.time_synchronizer.update_frame(frame)
            
            # Decimation: Check if this frame should be sent for AI analysis
            current_time = time.time()
            time_since_last = current_time - self._last_analysis_time
            
            if time_since_last >= self.analysis_interval:
                # Send copy to valid race condition on image buffer modification
                self._send_for_analysis(frame.copy(), current_time)
                self._last_analysis_time = current_time
            
            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._running = False
        logger.info("Capture loop ended")
    
    def _send_for_analysis(self, frame: np.ndarray, capture_time: float):
        """Put frame into processing queue to avoid blocking capture loop."""
        if not self._running:
            return

        import queue
        try:
             # Package needed data
             task = {
                 'frame': frame,
                 'timestamp': capture_time,
                 'frame_number': self._frame_number
             }
             self._processing_queue.put(task, block=False)
             self._stats['pending_processing'] = self._processing_queue.qsize()
        except queue.Full:
             self._stats["dropped_frames"] += 1
             # logger.warning(f"Processing queue full, dropped frame {self._frame_number}")

    def _processing_loop(self):
        """Background loop to handle heavy processing (OCR, Save, Enqueue)."""
        import queue
        logger.info("Processing loop started")
        
        while self._running:
            try:
                # Wait for task with timeout to check _running periodically
                task = self._processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            frame = task['frame']
            video_ts = task['timestamp']
            frame_num = task['frame_number']
            
            if frame is None or frame.size == 0:
                continue

            try:
                # 1. 使用 VLM 时间同步器获取当前视频时间
                video_time = self.get_current_video_time()
                final_ts = video_time.timestamp()

                # 2. Date-based Save
                date_str = video_time.strftime("%Y-%m-%d")
                save_dir = self.frame_save_dir / date_str
                save_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{self.camera_id}_{final_ts:.3f}.jpg"
                filepath = save_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                
                # 3. Track and Cleanup
                frame_id = filename
                
                self._saved_files.append((str(filepath), frame_id))
                
                while len(self._saved_files) > self.max_saved_frames:
                    item = self._saved_files.pop(0)
                    if isinstance(item, tuple):
                        old_path, old_frame_id = item
                    else:
                        old_path = item
                        old_frame_id = Path(old_path).name
                        
                    try:
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        
                        if self.deletion_callback:
                            doc_id = f"{self.camera_id}_{old_frame_id}"
                            self.deletion_callback(doc_id)
                    except Exception as e:
                        logger.warning(f"Cleanup error: {e}")
                
                # 4. Queue for Analysis Worker
                frame_obj = Frame(
                    id=frame_id,
                    timestamp=final_ts,
                    image_path=str(filepath),
                    camera_id=self.camera_id,
                    frame_number=frame_num,
                )
                
                # Put into analysis queue
                success = queue_manager.put("frame_queue", frame_obj, block=False)
                
                if success:
                    self._stats["analysis_frames"] += 1
                else:
                    self._stats["dropped_frames"] += 1
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
            finally:
                pass 
                # self._processing_queue.task_done()
        
        logger.info("Processing loop ended")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame (for single-frame access)."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def stream_frames(self, target_fps: float = 30.0) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames for UI display.
        This provides a smooth stream for Streamlit to consume.
        
        Args:
            target_fps: Target frame rate for streaming
            
        Yields:
            np.ndarray: BGR frames for display
        """
        frame_interval = 1.0 / target_fps
        last_frame_time = 0.0
        last_frame: Optional[np.ndarray] = None
        
        while self._running:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            # Get current frame
            with self._lock:
                if self._current_frame is not None:
                    frame = self._current_frame.copy()
                else:
                    # No frame yet, wait
                    time.sleep(0.01)
                    continue
            
            # Yield frame
            yield frame
            last_frame = frame
            last_frame_time = current_time
    
    def stream_frames_jpeg(self, target_fps: float = 30.0, quality: int = 80) -> Generator[bytes, None, None]:
        """
        Generator that yields JPEG-encoded frames for Streamlit.
        More efficient for web streaming.
        
        Args:
            target_fps: Target frame rate
            quality: JPEG quality (1-100)
            
        Yields:
            bytes: JPEG-encoded frame data
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        for frame in self.stream_frames(target_fps):
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if success:
                yield buffer.tobytes()
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        stats = self._stats.copy()
        if stats["start_time"]:
            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
            stats["elapsed_seconds"] = elapsed
            stats["avg_fps"] = stats["total_frames"] / elapsed if elapsed > 0 else 0
            stats["analysis_fps"] = stats["analysis_frames"] / elapsed if elapsed > 0 else 0
        stats["queue_size"] = queue_manager.qsize("frame_queue")
        
        # 添加时间同步统计
        if self.time_synchronizer:
            stats["time_sync"] = self.time_synchronizer.get_stats()
        else:
            stats["time_sync"] = {"status": "disabled"}
            
        return stats
