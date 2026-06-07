"""
Analysis Worker - Consumes frames, runs VLM analysis, detects alerts, stores to vector DB.

Dual-Mode Embedding Pipeline:
  - API Mode: VLM description -> Text Embedding API -> Vector
  - Local Mode: Image -> Local Model -> Visual Vector (with text fallback)

Smart Fallback:
  1. Always generate VLM description (for alerts and metadata)
  2. Try embed_image() first (returns vector if Local mode)
  3. If embed_image() returns None (API mode), fallback to embed_text(description)
"""
import shutil
import os
from pathlib import Path
import threading
import time
import re
import logging
import cv2
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from .vlm import VLMAnalyzer
from .vector_db import VectorStore, VisualVectorStore
from .feature_extractor import BaseEmbedder, APIEmbedder
from .search import SearchEngine
from ..common.types import Frame, Alert, AnalysisResult
from ..common.queue_manager import queue_manager
from ..notification.manager import NotificationManager

logger = logging.getLogger(__name__)


# Alert type definitions
ALERT_PATTERNS = {
    "FALL_DETECTED": {
        "pattern": r"【ALERT:\s*FALL_DETECTED】",
        "risk_type": "跌倒检测",
        "severity": "Critical",
        "description_zh": "检测到人员摔倒",
    },
    "FIRE_DETECTED": {
        "pattern": r"【ALERT:\s*FIRE_DETECTED】",
        "risk_type": "火灾检测",
        "severity": "Critical",
        "description_zh": "检测到火灾/烟雾",
    },
    "VIOLENCE_DETECTED": {
        "pattern": r"【ALERT:\s*VIOLENCE_DETECTED】",
        "risk_type": "暴力检测",
        "severity": "High",
        "description_zh": "检测到暴力行为",
    },
    "INTRUSION_DETECTED": {
        "pattern": r"【ALERT:\s*INTRUSION_DETECTED】",
        "risk_type": "非法闯入",
        "severity": "High",
        "description_zh": "检测到可疑入侵",
    },
}


@dataclass
class AlertLog:
    """Thread-safe alert log."""
    alerts: List[Alert] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    max_size: int = 1000
    
    def add(self, alert: Alert):
        with self._lock:
            self.alerts.append(alert)
            # Trim if exceeds max size
            if len(self.alerts) > self.max_size:
                self.alerts = self.alerts[-self.max_size:]
    
    def get_recent(self, n: int = 20) -> List[Alert]:
        with self._lock:
            return list(reversed(self.alerts[-n:]))
    
    def get_unread(self, since: datetime) -> List[Alert]:
        with self._lock:
            return [a for a in self.alerts if a.timestamp > since]
    
    def count(self) -> int:
        with self._lock:
            return len(self.alerts)
    
    def clear(self):
        with self._lock:
            self.alerts.clear()


class AnalysisWorker:
    """
    Worker that consumes frames from queue, analyzes with VLM,
    detects alerts, and stores results in vector DB.
    
    Text-based Embedding Pipeline:
      Step 1: VLM generates detailed text description
      Step 2: Text description -> Embedding API -> Vector
      Step 3: Store vector + description + metadata to ChromaDB
    """
    
    def __init__(
        self,
        vlm_analyzer: Optional[VLMAnalyzer] = None,
        vector_store: Optional[VectorStore] = None,
        visual_embedder: Optional[BaseEmbedder] = None,
        visual_store: Optional[VisualVectorStore] = None,
        alert_log: Optional[AlertLog] = None,
        alert_callback: Optional[Callable[[Alert], None]] = None,
        batch_size: int = 1,
        poll_timeout: float = 1.0,
        enable_visual_index: bool = True,  # Enable text embedding indexing
        alarm_rules: Optional[Dict[str, Any]] = None,
        alert_mode: str = "keyword",  # "keyword" or "ai_tag"
        alert_save_dir: str = "./data/alerts",
        max_alert_images: int = 1000,
        context_size: int = 3,  # Number of previous frames to keep in context
    ):
        """
        Initialize AnalysisWorker.
        
        Args:
            vlm_analyzer: VLM analyzer instance (created if None)
            vector_store: Vector store instance for text search (created if None)
            visual_embedder: BaseEmbedder for embeddings (supports API or Local mode)
            visual_store: VisualVectorStore for semantic search (created if visual_embedder provided)
            alert_log: Alert log instance (created if None)
            alert_callback: Optional callback when alert is detected
            batch_size: Number of frames to process before yielding
            poll_timeout: Timeout for queue polling in seconds
            enable_visual_index: Whether to enable text embedding indexing
            alarm_rules: Dictionary of alarm rules from config
            alert_mode: "keyword" for keyword matching, "ai_tag" for AI ALERT tag detection
            alert_save_dir: Directory to save persistent alert images
            max_alert_images: Max number of alert images to keep
            context_size: Size of history context window
        """
        self.vlm = vlm_analyzer or VLMAnalyzer(backend="auto")
        self.vector_store = vector_store or VectorStore()
        self.alert_log = alert_log or AlertLog()
        self.alert_callback = alert_callback
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout
        self.enable_visual_index = enable_visual_index
        self.alarm_rules = alarm_rules or {}
        self.alert_mode = alert_mode
        self.context_size = context_size
        self.history = deque(maxlen=context_size) if context_size > 0 else None
        
        # Notification Manager
        self.notifier = NotificationManager()
        
        # Persistent Alert Storage
        self.alert_save_dir = Path(alert_save_dir)
        self.max_alert_images = max_alert_images
        self.alert_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Text embedding components
        self.visual_embedder = visual_embedder
        self.visual_store = visual_store
        
        # Lazy init visual store if embedder is provided but store is not
        if self.visual_embedder and not self.visual_store:
            self.visual_store = VisualVectorStore()
            
        # Initialize Search Engine
        self.search_engine = SearchEngine(
            vector_store=self.vector_store,
            visual_store=self.visual_store,
            visual_embedder=self.visual_embedder
        )
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            "frames_processed": 0,
            "alerts_detected": 0,
            "visual_indexed": 0,
            "errors": 0,
            "start_time": None,
            "last_analysis_time": None,
        }
        
        logger.info(f"AnalysisWorker initialized with VLM backend: {self.vlm.get_backend_name()}")
        if self.visual_embedder:
            logger.info(f"  Text embedding indexing enabled: {self.enable_visual_index}")
        if self.alarm_rules:
            logger.info(f"  Loaded {len(self.alarm_rules)} alarm rules")
    
    def start(self):
        """Start the worker in a background thread."""
        if self._running:
            logger.warning("Worker already running")
            return
        
        self._running = True
        self._stats["start_time"] = datetime.now()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("AnalysisWorker started")
    
    def stop(self):
        """Stop the worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"AnalysisWorker stopped. Stats: {self._stats}")
    
    def is_running(self) -> bool:
        return self._running
    
    def _process_loop(self):
        """Main processing loop with latest-frame-only strategy."""
        while self._running:
            try:
                # Get the LATEST frame from queue (skip all older frames)
                # This prevents cumulative latency when VLM processing time >= capture interval
                frame = self._drain_queue_get_latest()
                
                if frame is None:
                    time.sleep(0.1)  # No frames available, wait briefly
                    continue
                
                # Process the frame
                result = self._process_frame(frame)
                
                self._stats["frames_processed"] += 1
                self._stats["last_analysis_time"] = datetime.now()
                
                # If VLM returned an error, add cooldown to avoid hammering the API
                if result and result.description and result.description.startswith("[API Error"):
                    time.sleep(3.0)  # Wait before next attempt
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}", exc_info=True)
                self._stats["errors"] += 1
                time.sleep(1.0)  # Brief pause on error
        
        logger.info("Analysis loop ended")
    
    def _drain_queue_get_latest(self) -> Optional[Frame]:
        """
        取出队列中的最新帧，丢弃所有旧帧。
        这确保我们始终处理最接近实时的帧，防止累积延迟。
        
        Returns:
            最新的帧对象，如果队列为空则返回 None
        """
        latest_frame = None
        dropped_count = 0
        
        # 阻塞等待第一帧（带超时）
        try:
            latest_frame = queue_manager.get("frame_queue", block=True, timeout=self.poll_timeout)
        except Exception:
            return None
        
        if latest_frame is None:
            return None
        
        # 持续取出所有可用帧，只保留最新的
        while True:
            try:
                frame = queue_manager.get_nowait("frame_queue")
                if frame is not None:
                    latest_frame = frame
                    dropped_count += 1
                else:
                    break
            except Exception:
                break
        
        if dropped_count > 0:
            logger.debug(f"Skipped {dropped_count} stale frames, processing latest")
            self._stats["skipped_frames"] = self._stats.get("skipped_frames", 0) + dropped_count
        
        return latest_frame
    
    def _process_frame(self, frame: Frame) -> AnalysisResult:
        """
        Process a single frame with text-based embedding pipeline:
          Step 1: VLM generates detailed text description
          Step 2: Text description -> Embedding API -> Vector
          Step 3: Store vector + description + metadata to ChromaDB
        
        Args:
            frame: Frame to process
            
        Returns:
            AnalysisResult with description and alert status
        """
        logger.debug(f"Processing frame: {frame}")
        
        # ===== STEP 1: VLM Analysis - Generate Text Description =====
        try:
            # Convert history deque to list if available
            history_list = list(self.history) if self.history else None
            description = self.vlm.analyze_frame(frame.image_path, history=history_list)
            
            # Update history with new result (clean description preferred usually, but raw is fine)
            if self.history is not None:
                # Store simplified history to save tokens: "Time: Description"
                time_str = datetime.fromtimestamp(frame.timestamp).strftime("%H:%M:%S")
                # Remove [Lx_TYPE] tags from history to keep it clean for model context
                clean_desc = self._clean_description(description)
                self.history.append(f"[{time_str}] {clean_desc}")
                
        except Exception as e:
            logger.error(f"VLM analysis failed for frame {frame.frame_number}: {e}")
            # Return empty result on VLM failure
            return AnalysisResult(
                frame_id=frame.id,
                description="VLM分析失败",
                risk_score=0.0,
                is_alert=False,
                alert=None,
            )

        # Print VLM result (简洁格式)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 📹 Frame {frame.frame_number}: {description[:100]}{'...' if len(description) > 100 else ''}")

        logger.info(f"[{frame.camera_id}] Frame {frame.frame_number}: {description[:150]}")
        
        logger.info(f"[{frame.camera_id}] Frame {frame.frame_number}: {description[:150]}")
        
        # Implement OSD Timestamp Extraction (Early extraction for Alert)
        # Try to find "监控时间：YYYY-MM-DD HH:MM:SS"
        osd_timestamp = self._extract_osd_timestamp(description)
        
        # Check for alerts
        alert = self._detect_alert(description, frame)
        is_alert = alert is not None
        
        if is_alert:
            # Sync alert timestamp with OSD time if available
            if osd_timestamp:
                alert.timestamp = datetime.fromtimestamp(osd_timestamp)
            
            self._stats["alerts_detected"] += 1
            
            # Add to alert log
            self.alert_log.add(alert)
            
            # Put alert in queue for WebUI
            queue_manager.put("alert_queue", alert, block=False)
            
            # Call callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
            
            # Send Push Notification
            # Deduplication key combines alert type (to prevent spam)
            dedup_key = f"alert_{alert.risk_type}"
            
            # Use clean description for notification to avoid tags like [L2_WARNING] taking up space
            notif_desc = self._clean_description(alert.description)
            
            self.notifier.send_alert(
                title=f"🚨 {alert.risk_type}",
                # Send raw description as content, let manager format it
                content=notif_desc,
                dedup_key=dedup_key,
                level="active" if alert.severity != "Critical" else "timeSensitive",
                image_path=alert.image_path,
                # Pass metadata for template population
                risk_type=alert.risk_type,
                severity=alert.severity,
                # Use full datetime if OSD available, otherwise just time is fine but full is better for log
                alert_time=alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            )
        
        # Clean description (remove alert tags)
        clean_description = self._clean_description(description)
        
        # osd_timestamp already extracted above
        index_timestamp = osd_timestamp if osd_timestamp else frame.timestamp
        
        if osd_timestamp:
            logger.info(f"Using OSD timestamp for index: {datetime.fromtimestamp(index_timestamp)} (Frame {frame.id})")
        
        # ===== STEP 2 & 3: Smart Embedding + Storage =====
        # Dual-mode smart fallback: try image embedding first, fallback to text
        if self.enable_visual_index and self.visual_embedder and self.visual_store:
            try:
                embedding = None
                embedding_type = "none"
                
                # Try to load the actual image for visual embedding
                frame_image = None
                if frame.image_path:
                    try:
                        frame_image = cv2.imread(frame.image_path)
                    except Exception as img_e:
                        logger.debug(f"Could not load image for visual embedding: {img_e}")
                
                # Step 2a: Try visual embedding (Local mode)
                if frame_image is not None:
                    visual_vector = self.visual_embedder.embed_image(frame_image)
                    if visual_vector is not None:
                        embedding = visual_vector
                        embedding_type = "visual"
                        logger.debug(f"Using visual embedding from Local mode")
                
                # Step 2b: Fallback to text embedding (API mode or Local mode fallback)
                if embedding is None:
                    embedding = self.visual_embedder.embed_text(clean_description)
                    embedding_type = "text"
                    logger.debug(f"Using text embedding (fallback)")
                
                # Determine correct image path (Persistent if alert)
                final_image_path = alert.image_path if (is_alert and alert) else frame.image_path
                
                # Step 3: Store in visual vector store with embedding
                self.visual_store.add(
                    frame_id=frame.id,
                    embedding=embedding,
                    timestamp=index_timestamp,
                    camera_id=frame.camera_id,
                    image_path=final_image_path,
                    description=clean_description,
                    alert_info=alert.risk_type if alert else "",
                    metadata={
                        "frame_number": frame.frame_number,
                        "has_alert": is_alert,
                        "embedding_type": embedding_type,
                        "severity": alert.severity if alert else "Info", # "Critical", "Warning", etc.
                        "alert_level": 3 if (alert and alert.severity == "Critical") else (2 if (alert and alert.severity == "Warning") else (1 if is_alert else 0)),
                        "real_time": datetime.fromtimestamp(index_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        "time_source": "ai_osd" if osd_timestamp else "system"
                    }
                )
                self._stats["visual_indexed"] += 1
                logger.debug(f"Embedding indexed: frame {frame.id}, type={embedding_type}, dim={len(embedding)}")
                
            except Exception as e:
                logger.error(f"Embedding/storage failed for frame {frame.frame_number}: {e}")
                # Don't store to vector DB if embedding fails (avoid dirty data)
                logger.warning(f"Skipping vector storage due to embedding failure")
        
        # Helper to define final image path for all stores
        final_image_path = alert.image_path if (is_alert and alert) else frame.image_path

        # Also store in text vector DB (for backward compatibility)
        try:
            self.vector_store.add(
                frame_id=frame.id,
                description=clean_description,
                timestamp=index_timestamp,
                camera_id=frame.camera_id,
                image_path=final_image_path,
                metadata={
                    "frame_number": frame.frame_number,
                    "has_alert": is_alert,
                    "severity": alert.severity if alert else "Info",
                    "alert_level": 3 if (alert and alert.severity == "Critical") else (2 if (alert and alert.severity == "Warning") else (1 if is_alert else 0)),
                    "real_time": datetime.fromtimestamp(index_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "time_source": "ai_osd" if osd_timestamp else "system"
                }
            )
        except Exception as e:
            logger.error(f"Text vector store failed: {e}")
        
        # Build and return result
        result = AnalysisResult(
            frame_id=frame.id,
            description=clean_description,
            risk_score=1.0 if is_alert else 0.0,
            is_alert=is_alert,
            alert=alert,
        )
        
        # Put result in result queue
        queue_manager.put("result_queue", result, block=False)
        
        return result
    
    def _save_alert_image(self, frame_image_path: str, alert_type: str) -> str:
        """
        Save a persistent copy of the alert frame image.
        Returns the path to the new persistent image.
        """
        if not frame_image_path or not Path(frame_image_path).exists():
            return frame_image_path
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize alert type for filename
            safe_type = re.sub(r'[^a-zA-Z0-9_]', '_', alert_type)
            filename = f"alert_{timestamp}_{safe_type}.jpg"
            target_path = self.alert_save_dir / filename
            
            # Avoid overwriting if multiple alerts in same second
            counter = 1
            while target_path.exists():
                filename = f"alert_{timestamp}_{safe_type}_{counter}.jpg"
                target_path = self.alert_save_dir / filename
                counter += 1
            
            shutil.copy2(frame_image_path, target_path)
            
            # Explicitly update mtime to now so it's not treated as old
            # (copy2 preserves original mtime which might be old if looping video)
            os.utime(target_path, None)
            
            # Enforce limit after saving, protecting the new file
            self._enforce_alert_limit(exclude_path=target_path)
            
            return str(target_path.absolute())
        except Exception as e:
            logger.error(f"Failed to save persistent alert image: {e}")
            return frame_image_path

    def _enforce_alert_limit(self, exclude_path: Optional[Path] = None):
        """Delete oldest alert images if limit exceeded."""
        try:
            if self.max_alert_images <= 0:
                return # No limit
                
            # List all jpg files in alert dir
            files = list(self.alert_save_dir.glob("*.jpg"))
            
            # Filter out the file we just saved
            if exclude_path:
                files = [f for f in files if f.resolve() != exclude_path.resolve()]
            
            if len(files) > self.max_alert_images:
                # Sort by modification time (oldest first)
                files.sort(key=lambda p: p.stat().st_mtime)
                
                # Delete oldest
                to_delete_count = len(files) - self.max_alert_images
                for i in range(to_delete_count):
                    try:
                        files[i].unlink()
                        logger.info(f"Deleted old alert image: {files[i].name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old alert image {files[i]}: {e}")
        except Exception as e:
            logger.error(f"Error enforcing alert image limit: {e}")

    def _extract_osd_timestamp(self, description: str) -> Optional[float]:
        """
        Extract OSD timestamp from description.
        Format: 监控时间：YYYY-MM-DD HH:MM:SS
        """
        try:
            # Match standard format: YYYY-MM-DD HH:MM:SS
            pattern = r"监控时间[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
            match = re.search(pattern, description)
            if match:
                time_str = match.group(1)
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
        except Exception as e:
            logger.debug(f"Failed to extract OSD timestamp: {e}")
        return None

    def _detect_alert(self, description: str, frame: Frame) -> Optional[Alert]:
        """
        Detect alerts from VLM description using multiple strategies.
        Now supports:
        1. Level Tags: [L3_CRITICAL], [L2_WARNING]
        2. Keyword Match: User defined keywords
        3. Legacy AI Tags: 【ALERT: ...】
        """
        alert = None
        
        # 1. Try Level Tag Detection (Priority)
        alert = self._detect_by_level_tag(description, frame)
        
        # 2. If no level tag triggered, check configured keyword rules
        # (Only if not already an alert, or to upgrade severity?)
        # For simplicity: Level tag takes precedence for now as it implies AI "understanding"
        if not alert and self.alert_mode == "keyword":
            alert = self._detect_by_keyword(description, frame)
            
        # 3. Fallback to Legacy AI Tag
        if not alert:
             alert = self._detect_by_ai_tag(description, frame)
            
        # If alert detected, save persistent image
        if alert:
            persistent_path = self._save_alert_image(alert.image_path, alert.risk_type)
            alert.image_path = persistent_path
            
        return alert
    
    def _detect_by_level_tag(self, description: str, frame: Frame) -> Optional[Alert]:
        """
        Parse [Ln_TYPE] tags from description.
        L3 -> Critical Alert
        L2 -> Warning Alert
        L1, L0 -> Info (No Alert)
        """
        pattern = r"\[L(\d)_([A-Z_]+)\]"
        match = re.search(pattern, description)
        
        if match:
            level = int(match.group(1))
            type_str = match.group(2)
            
            # Only L3 and L2 trigger system alerts
            if level >= 2:
                severity = "Critical" if level >= 3 else "Warning"
                
                # Try to map type_str to Chinese for better UI display
                type_map = {
                    "CRITICAL": "紧急警报",
                    "WARNING": "异常关注",
                    "FALL_DETECTED": "摔倒检测",
                    "FIRE_DETECTED": "火灾检测",
                    "VIOLENCE_DETECTED": "暴力行为",
                    "INTRUSION_DETECTED": "非法闯入"
                }
                
                risk_type = type_map.get(type_str, f"检测到 {type_str}")
                
                return Alert(
                    frame_id=frame.id,
                    risk_type=risk_type,
                    description=description, # Keep raw desc for UI badges (app.py cleans it visually)
                    severity=severity,
                    timestamp=datetime.now(),
                    image_path=frame.image_path,
                )
                
        return None

    def _detect_by_keyword(self, description: str, frame: Frame) -> Optional[Alert]:
        """关键词匹配模式检测报警 (Enhanced with robust negation check)"""
        if not self.alarm_rules:
            return None
        
        for rule_id, rule in self.alarm_rules.items():
            if not rule.get("enabled", True):
                continue
            
            # 获取关键词列表
            keywords = rule.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            
            # 检查是否有任何关键词匹配 (需过滤否定语境)
            for kw in keywords:
                if not kw: continue
                
                # 使用正则查找所有匹配项的位置
                # re.escape 确保关键词中的特殊字符被转义
                matches = re.finditer(re.escape(kw), description)
                
                for match in matches:
                    start_idx = match.start()
                    
                    # 使用增强的否定检测逻辑
                    if not self._is_negated_context(description, start_idx):
                        # 只有当不是否定语境时，才触发报警
                        return Alert(
                            frame_id=frame.id,
                            risk_type=rule.get("description", f"检测到 {rule_id}"),
                            description=description,
                            severity=rule.get("severity", "Medium"),
                            timestamp=datetime.now(),
                            image_path=frame.image_path,
                        )
        
        return None

    def _is_negated_context(self, text: str, target_idx: int) -> bool:
        """
        Check if the keyword at target_idx is in a negative context.
        Searches backwards until a 'blocker' (sentence boundary) is found.
        Handles lists like "无A、B或C" correctly.
        """
        # 1. Define Blockers: Punctuation or contrastive conjunctions that break negation scope
        # 注意：不要包含 '、' 或 '或'，因为它们是在 lists 中传递否定的
        blockers = [
            "，", "。", "；", "！", "？", 
            ",", ".", ";", "!", "?", "\n",
            "但", "但是", "然而", "不过"
        ]
        
        # 2. Find the nearest preceding blocker
        start_bound = 0
        for i in range(target_idx - 1, -1, -1):
            char = text[i]
            # Check single char blockers
            if char in blockers:
                start_bound = i + 1
                break
            # Check 2-char blockers (e.g., 但是) - simplified check
            if i > 0 and text[i-1:i+1] in blockers:
                start_bound = i + 1
                break
        
        # 3. Extract the context segment (from blocker to keyword)
        context_segment = text[start_bound:target_idx]
        
        # 4. Check for negation words in this segment
        # 扩展的否定词列表
        negatives = [
            "无", "没有", "没", 
            "未", "未见", "未发现", "暂未",
            "不包含", "不是", "非", 
            "no ", "No ", "not ", "neither ", "nor "
        ]
        
        for neg in negatives:
            if neg in context_segment:
                return True
                
        return False
    
    def _detect_by_ai_tag(self, description: str, frame: Frame) -> Optional[Alert]:
        """AI标签模式检测报警（原逻辑）"""
        # Generic pattern for dynamic tags: 【ALERT: TYPE_DETECTED】
        pattern = r"【ALERT:\s*([A-Z0-9_]+)_DETECTED】"
        matches = re.findall(pattern, description)
        
        if matches:
            alert_key = matches[0]
            
            if self.alarm_rules and alert_key in self.alarm_rules:
                rule = self.alarm_rules[alert_key]
                if rule.get("enabled", True):
                    return Alert(
                        frame_id=frame.id,
                        risk_type=rule.get("description", f"检测到 {alert_key}"),
                        description=description,
                        severity=rule.get("severity", "Medium"),
                        timestamp=datetime.now(),
                        image_path=frame.image_path,
                    )
            elif not self.alarm_rules:
                return Alert(
                    frame_id=frame.id,
                    risk_type=f"Alert: {alert_key}",
                    description=description,
                    severity="Medium",
                    timestamp=datetime.now(),
                    image_path=frame.image_path,
                )
        
        return None
    
    def _clean_description(self, description: str) -> str:
        """Remove alert tags from description for clean storage."""
        # 1. Clean the generic tag pattern 【ALERT:...】
        d = re.sub(r"【ALERT:\s*[A-Z0-9_]+_DETECTED】", "", description)
        # 2. Clean the new level tag pattern [Ln_TYPE]
        d = re.sub(r"\[L\d_[A-Z_]+\]", "", d)
        return d.strip()
    
    def process_single(self, frame: Frame) -> AnalysisResult:
        """
        Process a single frame synchronously (for testing).
        
        Args:
            frame: Frame to process
            
        Returns:
            AnalysisResult
        """
        return self._process_frame(frame)
    
    def search(self, query: str, n_results: int = 10, time_range: Optional[tuple] = None, sort_mode: str = "relevance", search_mode: str = "hybrid", min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Unified search method using SearchEngine.
        """
        return self.search_engine.search(
            query=query,
            n_results=n_results,
            sort_mode=sort_mode,
            time_range=time_range,
            search_mode=search_mode,
            min_score=min_score
        )
    
    # Maintain backwards compatibility aliases if needed, or just redirect
    def search_frames(self, query: str, n_results: int = 10, time_range: Optional[tuple] = None, sort_mode: str = "relevance", min_score: float = 0.0) -> List[Dict[str, Any]]:
        return self.search(query, n_results, time_range, sort_mode, search_mode="text", min_score=min_score)

    def search_visual(self, query: str, n_results: int = 10, time_range: Optional[tuple] = None, sort_mode: str = "relevance", min_score: float = 0.0) -> List[Dict[str, Any]]:
        return self.search(query, n_results, time_range, sort_mode, search_mode="hybrid", min_score=min_score)
    
    def get_alerts(self, n: int = 20) -> List[Alert]:
        """Get recent alerts."""
        return self.alert_log.get_recent(n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self._stats.copy()
        
        # Calculate Worker FPS (Actual processed rate)
        if stats.get("start_time"):
            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
            stats["worker_fps"] = stats["frames_processed"] / elapsed if elapsed > 0 else 0.0
        else:
            stats["worker_fps"] = 0.0
            
        stats["vector_store_count"] = self.vector_store.count()
        stats["alert_log_count"] = self.alert_log.count()
        stats["queue_frame_size"] = queue_manager.qsize("frame_queue")
        stats["queue_alert_size"] = queue_manager.qsize("alert_queue")
        
        # Add visual store stats if available
        if self.visual_store:
            stats["visual_store_count"] = self.visual_store.count()
            stats["visual_embedding_dim"] = self.visual_store._actual_dim
        return stats
