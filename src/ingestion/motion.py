"""
Enhanced Motion Detection Module - Multi-strategy detection for VLM gating.

Features:
- Illumination compensation (光照补偿)
- Cumulative change tracking (累积变化)
- Grid-based detection (分块检测)
- Motion trend analysis (运动趋势)
- Adaptive thresholding (自适应阈值)
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class EnhancedMotionDetector:
    """
    Enhanced motion detector with multiple detection strategies.
    
    Detection Modes:
    - standard: Basic frame differencing (fast, low accuracy)
    - enhanced: + illumination compensation + grid detection
    - intelligent: + cumulative tracking + trend analysis (best accuracy)
    """
    
    def __init__(
        self,
        mode: str = "enhanced",
        history_size: int = 10,
        grid_size: int = 4,
        cumulative_window: int = 5,
    ):
        """
        Initialize enhanced motion detector.
        
        Args:
            mode: Detection mode ("standard", "enhanced", "intelligent")
            history_size: Number of frames for moving average
            grid_size: Grid divisions for block detection (e.g., 4 = 4x4 grid)
            cumulative_window: Frames to track for cumulative change
        """
        self.mode = mode
        self.history_size = history_size
        self.grid_size = grid_size
        self.cumulative_window = cumulative_window
        
        # Basic detection state
        self.prev_gray = None
        self.blur_ksize = (21, 21)
        self._history = deque(maxlen=history_size)
        
        # Illumination tracking
        self.prev_brightness = None
        self.brightness_history = deque(maxlen=10)
        
        # Cumulative change tracking
        self.cumulative_changes = deque(maxlen=cumulative_window)
        
        # Motion trend analysis
        self.motion_scores = deque(maxlen=10)
        
        # Consecutive minor motion (for continuous small movements)
        self.consecutive_minor_motion = 0
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "illumination_compensations": 0,
            "grid_triggers": 0,
            "cumulative_triggers": 0,
            "cumulative_triggers": 0,
            "trend_triggers": 0,
            "continuous_triggers": 0,
        }
        
        logger.info(f"EnhancedMotionDetector initialized: mode={mode}, grid={grid_size}x{grid_size}")
    
    def detect(self, frame: np.ndarray, threshold: float = 5.0) -> float:
        """
        Detect motion with selected strategy.
        
        Args:
            frame: Input BGR frame
            threshold: Base threshold percentage
            
        Returns:
            float: Motion score (0-100), adjusted by detection mode
        """
        if frame is None:
            return 0.0
        
        self.stats["total_detections"] += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_brightness = np.mean(gray)
            return 0.0
        
        # === Mode: Standard ===
        if self.mode == "standard":
            return self._detect_standard(gray)
        
        # === Mode: Enhanced ===
        elif self.mode == "enhanced":
            return self._detect_enhanced(gray, frame, threshold)
        
        # === Mode: Intelligent ===
        elif self.mode == "intelligent":
            return self._detect_intelligent(gray, frame, threshold)
        
        else:
            logger.warning(f"Unknown mode: {self.mode}, falling back to standard")
            return self._detect_standard(gray)
    
    def _detect_standard(self, gray: np.ndarray) -> float:
        """Standard frame differencing (original logic)."""
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        score = (changed_pixels / total_pixels) * 100.0
        
        self.prev_gray = gray
        
        # Moving average
        self._history.append(score)
        return sum(self._history) / len(self._history)
    
    def _detect_enhanced(self, gray: np.ndarray, frame: np.ndarray, threshold: float) -> float:
        """
        Enhanced detection with:
        - Illumination compensation
        - Grid-based detection for small objects
        """
        # 1. Check illumination change
        current_brightness = np.mean(gray)
        brightness_delta = abs(current_brightness - self.prev_brightness)
        self.brightness_history.append(brightness_delta)
        
        # Compensate for global illumination changes
        compensated_gray = gray.copy()
        if brightness_delta > 10:  # Significant brightness change
            self.stats["illumination_compensations"] += 1
            # Apply histogram equalization to normalize
            compensated_gray = cv2.equalizeHist(gray)
            logger.debug(f"Illumination compensation: delta={brightness_delta:.1f}")
        
        # 2. Standard detection on compensated frame
        # Safety Fix: Ensure shapes match (handle resolution changes)
        if self.prev_gray.shape != compensated_gray.shape:
             logger.warning(f"Frame size changed from {self.prev_gray.shape} to {compensated_gray.shape}. Resizing stored frame.")
             self.prev_gray = cv2.resize(self.prev_gray, (compensated_gray.shape[1], compensated_gray.shape[0]))

        frame_delta = cv2.absdiff(self.prev_gray, compensated_gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        global_score = (changed_pixels / total_pixels) * 100.0
        
        # 3. Grid-based detection for small objects
        grid_score = self._detect_grid(self.prev_gray, compensated_gray, threshold)
        
        # 4. Combine scores intelligently
        # Only use grid_score if it's higher than global_score
        # Relaxed logic: Allow grid score if > 0.4 * threshold to catch small moving objects
        if grid_score > global_score * 1.05 and grid_score > threshold * 0.4:
            # Boost the score more aggressively for small concentrated motion
            final_score = grid_score * 1.5
            # Log valid grid trigger
            self.stats["grid_triggers"] += 1
            if final_score > threshold: 
                logger.debug(f"Grid trigger (boosted): raw={grid_score:.1f}% -> final={final_score:.1f}%")
        else:
            final_score = global_score
        
        # Update state
        self.prev_gray = compensated_gray
        self.prev_brightness = current_brightness
        
        # Moving average smoothing
        self._history.append(final_score)
        avg_score = sum(self._history) / len(self._history)
        
        return avg_score
    
    def _detect_intelligent(self, gray: np.ndarray, frame: np.ndarray, threshold: float) -> float:
        """
        Intelligent detection with all features:
        - Illumination compensation
        - Grid detection
        - Cumulative change tracking
        - Motion trend analysis
        """
        # Start with enhanced detection
        base_score = self._detect_enhanced(gray, frame, threshold)
        
        # 5. Cumulative change tracking (for slow motion)
        self.cumulative_changes.append(base_score)
        if len(self.cumulative_changes) >= self.cumulative_window:
            cumulative_score = sum(self.cumulative_changes)
            # If cumulative change is significant, boost score
            if cumulative_score > threshold * self.cumulative_window * 0.6:
                self.stats["cumulative_triggers"] += 1
                logger.debug(f"Cumulative trigger: {cumulative_score:.1f} over {self.cumulative_window} frames")
                base_score = max(base_score, threshold * 1.2)  # Boost above threshold
        
        # 6. Motion trend analysis (predict acceleration)
        self.motion_scores.append(base_score)
        if len(self.motion_scores) >= 5:
            # Calculate trend (linear regression slope)
            x = np.arange(5)
            y = np.array(list(self.motion_scores)[-5:])
            trend = np.polyfit(x, y, 1)[0]  # Slope
            
            # If motion is accelerating, trigger early
            if trend > 0.5 and base_score > threshold * 0.5:
                self.stats["trend_triggers"] += 1
                logger.debug(f"Trend trigger: slope={trend:.2f}, score={base_score:.1f}")
                base_score = max(base_score, threshold * 1.1)
        
        return base_score
    
    def _detect_grid(self, prev_gray: np.ndarray, curr_gray: np.ndarray, threshold: float) -> float:
        """
        Grid-based detection to catch small objects.
        Divides frame into grid and checks each block.
        
        Returns:
            float: Maximum block score (to catch small changes)
        """
        h, w = prev_gray.shape
        block_h = h // self.grid_size
        block_w = w // self.grid_size
        
        max_block_score = 0.0
        significant_blocks = 0  # Count blocks with significant change
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract block
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                prev_block = prev_gray[y1:y2, x1:x2]
                curr_block = curr_gray[y1:y2, x1:x2]
                
                # Calculate block difference
                block_delta = cv2.absdiff(prev_block, curr_block)
                _, block_thresh = cv2.threshold(block_delta, 25, 255, cv2.THRESH_BINARY)
                
                block_pixels = block_thresh.shape[0] * block_thresh.shape[1]
                if block_pixels == 0:
                    continue
                    
                block_changed = cv2.countNonZero(block_thresh)
                block_score = (block_changed / block_pixels) * 100.0
                
                # Lower threshold to detect small movement within block
                if block_score > threshold * 0.6:
                    significant_blocks += 1
                
                max_block_score = max(max_block_score, block_score)
        
        # Relaxed: If any block has significant motion, return the max score
        # Even if only 1 block triggers, it might be a small object (like a chicken)
        if significant_blocks >= 1:
            return max_block_score
        else:
            return 0.0
    
    def should_trigger(self, score: float, threshold: float, time_since_last: float, heartbeat: float) -> Tuple[bool, str]:
        """
        Intelligent trigger decision with multiple criteria.
        
        Args:
            score: Current motion score
            threshold: Base threshold
            time_since_last: Seconds since last analysis
            heartbeat: Heartbeat interval
            
        Returns:
            (should_trigger, reason)
        """
        # 1. Direct threshold trigger
        if score > threshold:
            self.consecutive_minor_motion = 0 # Reset counter on major trigger
            return True, f"Motion={score:.1f}%"
        
        # 1.5 Consecutive Minor Motion Trigger
        # Detects sustained movement (e.g. 0.5% - Threshold) that doesn't breach peak threshold
        if score > 0.5 and score <= threshold:
            self.consecutive_minor_motion += 1
            if self.consecutive_minor_motion >= 5:
                self.consecutive_minor_motion = 0
                self.stats["continuous_triggers"] += 1
                return True, f"Continuous Motion ({score:.1f}%)"
        else:
            # Reset if motion stops (silence)
            self.consecutive_minor_motion = 0
        
        # 2. Heartbeat trigger (force periodic check)
        if heartbeat > 0 and time_since_last >= heartbeat:
            return True, f"Heartbeat ({time_since_last:.0f}s)"
        
        # 3. Intelligent mode: Check for sudden stop (possible fall/incident)
        if self.mode == "intelligent" and len(self.motion_scores) >= 3:
            recent = list(self.motion_scores)[-3:]
            # If motion suddenly dropped to near zero
            if recent[0] > threshold * 0.8 and recent[-1] < threshold * 0.2:
                return True, "Sudden stop detected"
        
        return False, ""
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            **self.stats,
            "mode": self.mode,
            "grid_size": f"{self.grid_size}x{self.grid_size}",
            "avg_brightness_delta": np.mean(self.brightness_history) if self.brightness_history else 0,
            "recent_scores": list(self.motion_scores)[-5:] if self.motion_scores else [],
            "consecutive_minor": self.consecutive_minor_motion,
        }
    
    def reset(self):
        """Reset detector state."""
        self.prev_gray = None
        self.prev_brightness = None
        self._history.clear()
        self.brightness_history.clear()
        self.cumulative_changes.clear()
        self.motion_scores.clear()
        self.consecutive_minor_motion = 0
        logger.info("Motion detector reset")


# Backward compatibility alias
MotionDetector = EnhancedMotionDetector
