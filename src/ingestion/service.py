"""
Video Ingestion Service - Captures frames and manages decimation for AI analysis.
"""
import threading
import time
import os
import cv2
import numpy as np
import re
import gc
from typing import Optional, Generator, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .video_source import VideoSource, FileVideoSource, RTSPVideoSource, NativeRTSPVideoSource
from .object_detection import get_object_detector # New Import
from ..common.types import Frame
from ..common.queue_manager import queue_manager

logger = logging.getLogger(__name__)

class TimeSynchronizer:
    """
    基于 VLM 的后台时间同步机制
    
    使用独立的后台线程每 10 秒调用一次 VLM 来识别视频中的时间戳，
    避免阻塞主视频流读取。
    """
    
    def __init__(self, vlm_client, roi_config: tuple, frame_provider: Callable[[], Optional[np.ndarray]]):
        """
        初始化时间同步器
        
        Args:
            vlm_client: VLM 客户端实例
            roi_config: ROI 配置 (x_ratio, y_ratio, w_ratio, h_ratio)
            frame_provider: 获取当前帧的回调函数
        """
        self.vlm_client = vlm_client
        self.roi_config = roi_config
        self.frame_provider = frame_provider
        
        # 核心时间基准变量
        self.base_video_time: Optional[datetime] = None  # VLM 读到的时间
        self.base_system_time: Optional[datetime] = None  # 读到该时间时的系统时间
        
        # 线程控制
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Removed _current_frame cache to avoid copy overhead
        
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
    
    # update_frame removed to avoid copying every frame
    
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
        # 获取当前帧 (Pull on demand)
        frame = self.frame_provider()
        if frame is None:
            logger.debug("No frame available for sync")
            return
        
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
        max_queue_size: int = 60,
        loop_video: bool = True,  # Loop for file sources
        max_saved_frames: int = 50,  # Maximum number of saved frame images
        roi_config: tuple = (0.65, 0.85, 0.35, 0.15),  # Time ROI (x, y, w, h) ratios
        vlm_client=None,  # VLM client for time synchronization
        deletion_callback: Optional[Callable[[str], None]] = None, # Callback for sync deletion
        motion_detection_mode: str = "enhanced",  # Motion detection mode
        motion_heartbeat: float = 15.0,  # Heartbeat interval in seconds
        object_detection_enabled: bool = True, # Enable YOLO
        object_detection_classes: list = None, # Classes to detect
    ):
        self.source = source
        self.camera_id = camera_id
        self.analysis_interval = analysis_interval
        self.frame_save_dir = Path(frame_save_dir)
        self.max_queue_size = max_queue_size
        self.loop_video = loop_video
        self.max_saved_frames = max_saved_frames
        self.deletion_callback = deletion_callback
        self.motion_detection_mode = motion_detection_mode
        self.motion_heartbeat = motion_heartbeat
        
        # Object Detection Config
        self.object_detection_enabled = object_detection_enabled
        self.object_detection_classes = object_detection_classes if object_detection_classes else ["person", "vehicle", "animal"]
        
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
        
        # Stats processing queue
        import queue
        self._processing_queue = queue.Queue(maxsize=max_queue_size)
        
        # Buffering state for Best-Frame Selection
        self._is_buffering = False
        self._frame_buffer = [] # List of (frame, score, timestamp)
        self._buffer_start_time = 0.0
        self._buffer_duration = 0.4 # Buffer window in seconds
        
        # 初始化 VLM 时间同步器（替换 OCR）
        if vlm_client is not None:
            # Use self.get_current_frame as provider (thread-safe copy)
            self.time_synchronizer = TimeSynchronizer(
                vlm_client, 
                roi_config, 
                frame_provider=self.get_current_frame
            )
            logger.info("VLM-based time synchronization enabled")
        else:
            self.time_synchronizer = None
            logger.warning("No VLM client provided, using system time")
            
        # Frame state
        self._current_frame: Optional[np.ndarray] = None
        self._frame_number = 0
        self._last_analysis_time = 0.0
        self._frames_for_analysis = 0
        self.motion_detector = None # Expose for stats access
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
        # source = RTSPVideoSource(rtsp_url) 
        source = NativeRTSPVideoSource(rtsp_url) # Use optimized native source
        return cls(source, **kwargs)
    
    def get_motion_stats(self) -> dict:
        """Get motion detection statistics (if available)."""
        if self.motion_detector:
             return self.motion_detector.get_stats()
             
        # Fallback if not initialized yet
        return {
            "mode": getattr(self, "motion_detection_mode", "enhanced"),
            "threshold": getattr(self, "motion_threshold", 5.0),
            "heartbeat": getattr(self, "motion_heartbeat", 15.0),
            "obj_det_enabled": getattr(self, "object_detection_enabled", True),
            "status": "initializing"
        }
    
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
        """Main capture loop running in background thread with Enhanced Motion Gating."""
        fps = self.source.get_fps()
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # Enhanced Motion Gating Configuration
        from .motion import EnhancedMotionDetector
        
        # 动态读取配置
        motion_mode = getattr(self, "motion_detection_mode", "enhanced")
        self.motion_detector = EnhancedMotionDetector(mode=motion_mode, grid_size=6)
        motion_detector = self.motion_detector # Local alias for compatibility
        
        # 初始化时读取配置
        motion_threshold = getattr(self, "motion_threshold", 5.0)
        min_interval = getattr(self, "analysis_interval", 1.0)
        heartbeat_interval = getattr(self, "motion_heartbeat", 15.0)
        
        # 初始化 last_analysis_time 为当前时间减去心跳间隔，确保第一次会触发心跳
        last_analysis_time = time.time() - heartbeat_interval
        
        logger.info(f"Capture loop started: FPS={fps:.1f}, Mode={motion_mode}, Threshold={motion_threshold}%, Interval={min_interval}s, Heartbeat={heartbeat_interval}s")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [CAM] Capture loop started: FPS={fps:.1f}, Mode={motion_mode}, Threshold={motion_threshold}%, Interval={min_interval}s, Heartbeat={heartbeat_interval}s", flush=True)
        
        # Ensure flushing
        import sys
        
        while self._running:
            loop_start = time.time()
            
            # Read frame
            try:
                success, frame = self.source.read_frame()
            except cv2.error as e:
                if "Insufficient memory" in str(e):
                    logger.error(f"OpenCV OOM: {e}. Triggering GC.")
                    print("⚠️ System running low on memory, performing cleanup...", flush=True)
                    gc.collect()
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(f"OpenCV error: {e}")
                    # Allow non-fatal errors to continue unless critical? 
                    # For now, treat unknown cv2 errors as transient or check running
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logger.error(f"Unexpected error in read_frame: {e}")
                time.sleep(1)
                continue
            
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
                    logger.warning("Stream read error, attempting reconnect...")
                    time.sleep(1) 
                    if hasattr(self.source, 'reconnect'):
                        try:
                            self.source.reconnect()
                            if not self._running:
                                self.source.release()
                                break
                            print("✅ 视频流重连成功！")
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e}")
                            time.sleep(2)
                    continue
            
            self._frame_number += 1
            self._stats["total_frames"] += 1
            
            # Update current frame for UI streaming
            with self._lock:
                self._current_frame = frame
            
            # --- Enhanced Motion Gating Logic ---
            current_time = time.time()
            time_since_last = current_time - last_analysis_time
            
            # 动态读取配置（支持热更新）
            motion_threshold = getattr(self, "motion_threshold", 5.0)
            min_interval = getattr(self, "analysis_interval", 1.0)
            heartbeat_interval = getattr(self, "motion_heartbeat", 15.0)
            motion_mode = getattr(self, "motion_detection_mode", "enhanced")
            
            # Update detector mode if changed
            if motion_detector.mode != motion_mode:
                logger.info(f"Switching motion detection mode: {motion_detector.mode} -> {motion_mode}")
                motion_detector.mode = motion_mode
            
            # 1. Check Rate Limit (Must satisfy min interval)
            # Note: During buffering, last_analysis_time is NOT updated, so we keep entering this block
            if time_since_last >= min_interval:
                # 2. Calculate Motion Score
                motion_score = motion_detector.detect(frame, motion_threshold)
                
                # Check Buffering State
                if self._is_buffering:
                    # Collect frame for buffer
                    self._frame_buffer.append((frame.copy(), motion_score, current_time))
                    
                    # Check if buffer is full (time based)
                    if current_time - self._buffer_start_time >= self._buffer_duration:
                        # Select best frame (highest motion score)
                        if self._frame_buffer:
                            # Use max score. If scores equal, later frame is preferred (stable sort? No, simple max)
                            best_frame, best_score, best_ts = max(self._frame_buffer, key=lambda x: x[1])
                            
                            logger.info(f"[CAM] Best frame selected: Score={best_score:.1f}% (from {len(self._frame_buffer)} buffered frames)")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] [CAM] Best frame selected: Score={best_score:.1f}% (from {len(self._frame_buffer)} buffered frames)", flush=True)
                            
                            # Send best frame
                            self._send_for_analysis(best_frame, best_ts)
                            last_analysis_time = current_time # Update timer only after sending
                        
                        # Reset buffer
                        self._is_buffering = False
                        self._frame_buffer = []

                else:
                    # 3. Intelligent trigger decision
                    should_analyze, reason = motion_detector.should_trigger(
                        motion_score, 
                        motion_threshold, 
                        time_since_last, 
                        heartbeat_interval
                    )
                    
                    # 4. Enhanced Logging & Trigger Logic
                    if should_analyze:
                        # Start Buffering instead of sending immediately
                        self._is_buffering = True
                        self._buffer_start_time = current_time
                        self._frame_buffer = [(frame.copy(), motion_score, current_time)]
                        
                        logger.info(f"[CAM] Motion: {motion_score:.2f}% (Threshold: {motion_threshold:.1f}%) -> TRIGGER ({reason}). Buffering...")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [CAM] Motion: {motion_score:.2f}% (Threshold: {motion_threshold:.1f}%) -> ✅ TRIGGER ({reason}). Buffering {self._buffer_duration}s...", flush=True)
                        
                    elif motion_score > 0.1:
                        # Format: [CAM] Motion: XX.XX% (Threshold: YY.Y%) -> IGNORE
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [CAM] Motion: {motion_score:.2f}% (Threshold: {motion_threshold:.1f}%) -> ⏭️  IGNORE", flush=True)

            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Print final statistics
        stats = motion_detector.get_stats()
        logger.info(f"Motion detector stats: {stats}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [CAM] Motion detector final stats: {stats}", flush=True)
        
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

                # --- NEW: Object Detection (YOLO) ---
                # Load Shedding: Skip YOLO if queue is backing up (>5 frames pending)
                # This prevents VLM analysis from being starved by object detection lag
                if getattr(self, "object_detection_enabled", True):
                    # Check queue size (approximate)
                    current_qsize = self._processing_queue.qsize() if hasattr(self._processing_queue, 'qsize') else 0
                    
                    if current_qsize > 5:
                        logger.warning(f"High load (qsize={current_qsize}), skipping YOLO for frame {frame_num}")
                    else:
                        try:
                            classes = getattr(self, "object_detection_classes", ["person", "vehicle", "animal"])
                            detector = get_object_detector()
                            # detector handles lazy loading
                            annotated_frame, det_stats = detector.detect_and_annotate(
                                frame, 
                                conf=0.45, # Slightly lower threshold for recall
                                classes=classes
                            )
                            if "error" not in det_stats:
                                # Use the annotated frame for saving
                                frame = annotated_frame
                                if det_stats["count"] > 0:
                                    logger.info(f"YOLO Detected: {det_stats['objects']}")
                        except Exception as e:
                            logger.error(f"Object detection failed (non-fatal): {e}")
                # -------------------------------------

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
    
    def get_current_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """
        Get the most recent frame.
        
        Args:
            copy: If True, return a copy (safe for modification). 
                  If False, return direct reference (faster, read-only).
        """
        with self._lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy() if copy else self._current_frame
    
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
