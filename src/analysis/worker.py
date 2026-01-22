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
import threading
import time
import re
import logging
import cv2
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field

from .vlm import VLMAnalyzer
from .vector_db import VectorStore, VisualVectorStore
from .feature_extractor import BaseEmbedder, APIEmbedder
from ..common.types import Frame, Alert, AnalysisResult
from ..common.queue_manager import queue_manager

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
        """
        self.vlm = vlm_analyzer or VLMAnalyzer(backend="auto")
        self.vector_store = vector_store or VectorStore()
        self.alert_log = alert_log or AlertLog()
        self.alert_callback = alert_callback
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout
        self.enable_visual_index = enable_visual_index
        
        # Text embedding components
        self.visual_embedder = visual_embedder
        self.visual_store = visual_store
        
        # Lazy init visual store if embedder is provided but store is not
        if self.visual_embedder and not self.visual_store:
            self.visual_store = VisualVectorStore()
        
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
        """Main processing loop."""
        while self._running:
            try:
                # Get frame from queue
                frame = queue_manager.get("frame_queue", block=True, timeout=self.poll_timeout)
                
                if frame is None:
                    continue
                
                # Process the frame
                result = self._process_frame(frame)
                
                self._stats["frames_processed"] += 1
                self._stats["last_analysis_time"] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}", exc_info=True)
                self._stats["errors"] += 1
                time.sleep(0.5)  # Brief pause on error
        
        logger.info("Analysis loop ended")
    
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
            description = self.vlm.analyze_frame(frame.image_path)
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
        
        # Print VLM result clearly to console
        print(f"\n👁️ [Frame {frame.frame_number}] VLM 识别结果:")
        print(f"   └── {description}")
        
        logger.info(f"[{frame.camera_id}] Frame {frame.frame_number}: {description}")
        
        # Check for alerts
        alert = self._detect_alert(description, frame)
        is_alert = alert is not None
        
        if is_alert:
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
            
            logger.warning(f"🚨 ALERT: {alert.risk_type} - {alert.description}")
        
        # Clean description (remove alert tags)
        clean_description = self._clean_description(description)
        
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
                
                # Step 3: Store in visual vector store with embedding
                self.visual_store.add(
                    frame_id=frame.id,
                    embedding=embedding,
                    timestamp=frame.timestamp,
                    camera_id=frame.camera_id,
                    image_path=frame.image_path,
                    description=clean_description,
                    alert_info=alert.risk_type if alert else "",
                    metadata={
                        "frame_number": frame.frame_number,
                        "has_alert": is_alert,
                        "embedding_type": embedding_type,
                    }
                )
                self._stats["visual_indexed"] += 1
                logger.debug(f"Embedding indexed: frame {frame.id}, type={embedding_type}, dim={len(embedding)}")
                
            except Exception as e:
                logger.error(f"Embedding/storage failed for frame {frame.frame_number}: {e}")
                # Don't store to vector DB if embedding fails (avoid dirty data)
                logger.warning(f"Skipping vector storage due to embedding failure")
        
        # Also store in text vector DB (for backward compatibility)
        try:
            self.vector_store.add(
                frame_id=frame.id,
                description=clean_description,
                timestamp=frame.timestamp,
                camera_id=frame.camera_id,
                image_path=frame.image_path,
                metadata={
                    "frame_number": frame.frame_number,
                    "has_alert": is_alert,
                    "alert_type": alert.risk_type if alert else "",
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
    
    def _detect_alert(self, description: str, frame: Frame) -> Optional[Alert]:
        """
        Detect alerts from VLM description.
        
        Args:
            description: VLM output text
            frame: Source frame
            
        Returns:
            Alert object if detected, None otherwise
        """
        for alert_key, alert_info in ALERT_PATTERNS.items():
            if re.search(alert_info["pattern"], description):
                return Alert(
                    frame_id=frame.id,
                    risk_type=alert_info["risk_type"],
                    description=description,
                    severity=alert_info["severity"],
                    timestamp=datetime.now(),
                    image_path=frame.image_path,
                )
        return None
    
    def _clean_description(self, description: str) -> str:
        """Remove alert tags from description for clean storage."""
        cleaned = description
        for alert_info in ALERT_PATTERNS.values():
            cleaned = re.sub(alert_info["pattern"], "", cleaned)
        return cleaned.strip()
    
    def process_single(self, frame: Frame) -> AnalysisResult:
        """
        Process a single frame synchronously (for testing).
        
        Args:
            frame: Frame to process
            
        Returns:
            AnalysisResult
        """
        return self._process_frame(frame)
    
    def search_frames(self, query: str, n_results: int = 10, time_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for frames matching a natural language query.
        Convenience method wrapping vector_store.search.
        
        Args:
            query: Search query (e.g., "穿红衣服的人")
            n_results: Max results
            time_range: Optional filter by timestamp range (start_ts, end_ts)
            
        Returns:
            List of matching frame results
        """
        return self.vector_store.search(query, n_results=n_results, time_range=time_range)
    
    def search_visual(self, query: str, n_results: int = 10, time_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for frames using text embedding (semantic search).
        Converts text query to embedding, then searches the visual vector store.
        
        Args:
            query: Text search query
            n_results: Max results
            time_range: Optional filter by timestamp range (start_ts, end_ts)
            
        Returns:
            List of matching frame results with similarity scores
        """
        if not self.visual_embedder or not self.visual_store:
            logger.warning("Semantic search not available: embedder or store not initialized")
            return []
        
        try:
            # Get text embedding for query
            query_embedding = self.visual_embedder.embed_text(query)
            
            # Search visual store
            return self.visual_store.search_by_embedding(
                query_embedding=query_embedding,
                n_results=n_results,
                time_range=time_range
            )
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def get_alerts(self, n: int = 20) -> List[Alert]:
        """Get recent alerts."""
        return self.alert_log.get_recent(n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self._stats.copy()
        stats["vector_store_count"] = self.vector_store.count()
        stats["alert_log_count"] = self.alert_log.count()
        stats["queue_frame_size"] = queue_manager.qsize("frame_queue")
        stats["queue_alert_size"] = queue_manager.qsize("alert_queue")
        
        # Add visual store stats if available
        if self.visual_store:
            stats["visual_store_count"] = self.visual_store.count()
            stats["visual_embedding_dim"] = self.visual_store._actual_dim
        return stats
