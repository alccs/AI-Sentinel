"""
Data types for cross-module communication.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class Frame:
    """Represents a video frame for analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = 0.0           # Video timestamp in seconds
    capture_time: datetime = field(default_factory=datetime.now)
    image_path: str = ""             # Path to saved frame image
    camera_id: str = "default"
    frame_number: int = 0
    
    def __repr__(self):
        return f"Frame(id={self.id}, ts={self.timestamp:.2f}s, cam={self.camera_id})"


@dataclass
class Alert:
    """Represents a detected risk/alert."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    frame_id: str = ""
    risk_type: str = ""              # e.g., "Fire", "Fall", "Intrusion"
    description: str = ""
    severity: str = "Medium"         # "Low", "Medium", "High", "Critical"
    timestamp: datetime = field(default_factory=datetime.now)
    image_path: str = ""
    
    def __repr__(self):
        return f"Alert({self.risk_type}, severity={self.severity})"


@dataclass
class AnalysisResult:
    """Result from VLM analysis."""
    frame_id: str = ""
    description: str = ""
    risk_score: float = 0.0
    is_alert: bool = False
    alert: Optional[Alert] = None
