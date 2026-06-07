import logging
import cv2
import numpy as np
import threading
from typing import List, Optional, Union, Dict, Tuple

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Wrapper for YOLO object detection.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.ready = False
        self._load_lock = threading.Lock()
        
        # COCO Class Mappings
        self.CLASS_GROUPS = {
            "person": [0], # Person
            "vehicle": [1, 2, 3, 5, 7], # Bicycle, Car, Motorcycle, Bus, Truck
            "animal": [15, 16, 17, 18, 19, 20, 21, 22, 23], # Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe
        }
        
    def load_model(self):
        """Lazy load the model."""
        if self.ready:
            return

        with self._load_lock:
            if self.ready:
                return
            
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model from {self.model_path}...")
                # verbose=False to reduce noise
                self.model = YOLO(self.model_path) 
                self.ready = True
                logger.info("YOLO model loaded successfully.")
            except ImportError:
                logger.error("Failed to import ultralytics. Please run 'pip install ultralytics'.")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")

    def detect_and_annotate(
        self, 
        frame: np.ndarray, 
        conf: float = 0.5, 
        classes: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run detection and draw boxes on the frame.
        
        Args:
            frame: BGR image
            conf: Confidence threshold (0.0 - 1.0)
            classes: List of class groups to detect (e.g., ["person", "vehicle"]). 
                     If None or empty, valid detections depend on internal logic (default to person?).
                     If ["all"], detects everything.
                     
        Returns:
            (annotated_frame, stats)
        """
        if not self.ready:
            self.load_model()
            if not self.ready:
                return frame, {"error": "Model not loaded"}

        # Determine target class IDs
        target_ids = []
        if classes:
            for c in classes:
                c = c.lower()
                if c == "all":
                    target_ids = None # None means all classes for YOLO
                    break
                if c in self.CLASS_GROUPS:
                    target_ids.extend(self.CLASS_GROUPS[c])
        
        # If classes is provided but didn't match groups, maybe they are raw names?
        # For simplicity, if classes is explicit but empty match, we might detect nothing.
        # Default behavior: if classes is None/Empty, maybe default to Person?
        if classes is None or (isinstance(classes, list) and len(classes) == 0):
             target_ids = self.CLASS_GROUPS["person"] # Default to person only

        try:
            # Run inference
            # stream=True is not needed for single frame
            # classes=target_ids filters results inside YOLO
            results = self.model(frame, conf=conf, classes=target_ids, verbose=False)
            
            result = results[0]
            # Customizing plot:
            # line_width=2 (Thinner)
            # labels=False (No class name)
            # conf=False (No confidence score)
            annotated_frame = result.plot(line_width=2, labels=False, conf=False) 
            
            # Extract basic stats
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf_score = float(box.conf[0])
                detections.append(f"{cls_name} ({conf_score:.2f})")
            
            stats = {
                "count": len(detections),
                "objects": detections
            }
            
            return annotated_frame, stats
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, {"error": str(e)}

# Singleton accessor
def get_object_detector():
    with ObjectDetector._lock:
        if ObjectDetector._instance is None:
            ObjectDetector._instance = ObjectDetector()
        return ObjectDetector._instance
