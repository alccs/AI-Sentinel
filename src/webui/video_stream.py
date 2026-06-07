"""
视频流服务 - 用于实时视频播放优化
避免Streamlit页面刷新导致的卡顿
"""
import cv2
import time
from threading import Lock
from typing import Optional
import numpy as np


class VideoStreamService:
    """视频流服务,提供缓存帧访问"""

    def __init__(self):
        self._current_frame: Optional[np.ndarray] = None
        self._lock = Lock()
        self._last_update = 0
        self._frame_count = 0

    def update_frame(self, frame: np.ndarray):
        """更新当前帧(由ingestion service调用)"""
        with self._lock:
            self._current_frame = frame.copy() if frame is not None else None
            self._last_update = time.time()
            self._frame_count += 1

    def get_jpeg_frame(self, max_width: int = 1280) -> Optional[bytes]:
        """获取JPEG编码的帧数据"""
        with self._lock:
            if self._current_frame is None:
                return None

            frame = self._current_frame.copy()

        # 缩放
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)))

        # JPEG编码(更高效的网络传输)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def get_stats(self) -> dict:
        """获取流统计信息"""
        with self._lock:
            return {
                "frame_count": self._frame_count,
                "last_update": self._last_update,
                "has_frame": self._current_frame is not None
            }


# 全局单例
_stream_service = VideoStreamService()

def get_stream_service() -> VideoStreamService:
    """获取视频流服务单例"""
    return _stream_service
