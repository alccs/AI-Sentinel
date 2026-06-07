"""
Thread-safe Global Queue Manager for inter-module communication.
"""
import threading
from queue import Queue, Empty, Full
from typing import Optional, Any, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class GlobalQueueManager:
    """
    Singleton manager for all application queues.
    Provides thread-safe access to named queues.
    """
    _instance: Optional['GlobalQueueManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._queues: Dict[str, Queue] = {}
        self._queue_locks: Dict[str, threading.Lock] = {}
        self._initialized = True
        
        # Pre-create standard queues
        # Pre-create standard queues
        self.create_queue("frame_queue", maxsize=500)      # Frames for AI analysis (Increased)
        self.create_queue("alert_queue", maxsize=500)      # Detected alerts (Increased)
        self.create_queue("result_queue", maxsize=500)     # Analysis results (Increased)
        
        logger.info("GlobalQueueManager initialized with standard queues")
    
    def create_queue(self, name: str, maxsize: int = 0) -> Queue:
        """Create a new named queue."""
        with self._lock:
            if name not in self._queues:
                self._queues[name] = Queue(maxsize=maxsize)
                self._queue_locks[name] = threading.Lock()
                logger.debug(f"Created queue: {name} (maxsize={maxsize})")
            return self._queues[name]
    
    def get_queue(self, name: str) -> Optional[Queue]:
        """Get a queue by name."""
        return self._queues.get(name)
    
    def put(self, queue_name: str, item: Any, block: bool = True, timeout: float = None) -> bool:
        """Put an item into a named queue."""
        queue = self._queues.get(queue_name)
        if queue is None:
            logger.error(f"Queue '{queue_name}' not found")
            return False
        try:
            queue.put(item, block=block, timeout=timeout)
            return True
        except Full:
            # If queue is full, discard the *oldest* item to make room for the new one
            # This ensures we always process the latest frames
            try:
                queue.get_nowait()
                queue.put(item, block=block, timeout=timeout)
                logger.debug(f"Queue '{queue_name}' full: dropped oldest item to add new one.")
                return True
            except (Empty, Full):
                # Extremely rare race condition or still full
                logger.error(f"Queue '{queue_name}' full and failed to cycle items.")
                return False
    
    def get(self, queue_name: str, block: bool = True, timeout: float = None) -> Optional[Any]:
        """Get an item from a named queue."""
        queue = self._queues.get(queue_name)
        if queue is None:
            logger.error(f"Queue '{queue_name}' not found")
            return None
        try:
            return queue.get(block=block, timeout=timeout)
        except Empty:
            return None
    
    def get_nowait(self, queue_name: str) -> Optional[Any]:
        """Non-blocking get from a named queue."""
        return self.get(queue_name, block=False)
    
    def qsize(self, queue_name: str) -> int:
        """Get approximate size of a queue."""
        queue = self._queues.get(queue_name)
        return queue.qsize() if queue else 0
    
    def clear_queue(self, queue_name: str):
        """Clear all items from a queue."""
        queue = self._queues.get(queue_name)
        if queue:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
            logger.info(f"Cleared queue: {queue_name}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get size of all queues."""
        return {name: q.qsize() for name, q in self._queues.items()}


# Global singleton instance
queue_manager = GlobalQueueManager()
