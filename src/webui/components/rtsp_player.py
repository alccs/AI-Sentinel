"""
独立 RTSP 视频播放器
使用 OpenCV + Threading 实现稳定的视频流显示
完全独立于 Streamlit 的刷新机制
"""
import streamlit as st
import cv2
import threading
import time
import base64
from typing import Optional
import numpy as np


class RTSPStreamPlayer:
    """
    独立的 RTSP 流播放器
    使用后台线程持续读取帧，前端只需获取最新帧
    """
    
    _instances = {}  # 单例模式，避免重复创建线程
    
    def __new__(cls, rtsp_url: str):
        if rtsp_url not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[rtsp_url] = instance
        return cls._instances[rtsp_url]
    
    def __init__(self, rtsp_url: str):
        if hasattr(self, '_initialized'):
            return
        
        self.rtsp_url = rtsp_url
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._initialized = True
        self._last_frame_time = 0
        self._fps = 25.0
        
    def start(self):
        """启动流读取"""
        if self._running:
            return
            
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # 设置缓冲区大小为 1，确保总是获取最新帧
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        if not self._cap.isOpened():
            raise ValueError(f"无法打开 RTSP 流: {self.rtsp_url}")
        
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """停止流读取"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
            
    def _reader_loop(self):
        """后台持续读取帧"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
            else:
                time.sleep(0.05)
                
    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
            
    def get_frame_base64(self, quality: int = 80) -> Optional[str]:
        """获取 Base64 编码的帧，用于 HTML 显示"""
        frame = self.get_frame()
        if frame is None:
            return None
            
        # 压缩为 JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
        
    @property
    def is_running(self) -> bool:
        return self._running
        
    @property
    def fps(self) -> float:
        return self._fps


def render_independent_rtsp_player(rtsp_url: str, key: str = "rtsp_player"):
    """
    渲染独立的 RTSP 播放器
    使用 JavaScript 自动刷新，完全独立于 Streamlit
    
    Args:
        rtsp_url: RTSP 流地址
        key: 组件唯一标识
    """
    import streamlit.components.v1 as components
    
    # 初始化或获取播放器实例
    player_key = f"_rtsp_player_{key}"
    if player_key not in st.session_state:
        try:
            player = RTSPStreamPlayer(rtsp_url)
            player.start()
            st.session_state[player_key] = player
        except Exception as e:
            st.error(f"RTSP 连接失败: {e}")
            return
    
    player = st.session_state[player_key]
    
    # 获取当前帧
    frame_b64 = player.get_frame_base64(quality=85)
    
    if frame_b64:
        # 使用 HTML img 标签显示，避免 Streamlit 刷新
        html_content = f"""
        <div style="width: 100%; background: #000; border-radius: 8px; overflow: hidden;">
            <img id="rtsp_frame_{key}" 
                 src="data:image/jpeg;base64,{frame_b64}" 
                 style="width: 100%; height: auto; display: block;"
                 alt="RTSP Stream">
        </div>
        <style>
            #rtsp_frame_{key} {{
                transition: opacity 0.05s ease;
            }}
        </style>
        """
        components.html(html_content, height=480)
    else:
        st.info("⏳ 等待视频信号...")


def render_mjpeg_stream_player(mjpeg_url: str, key: str = "mjpeg_player"):
    """
    渲染 MJPEG 流播放器
    如果摄像头支持 MJPEG 输出，这是最稳定的方案
    
    Args:
        mjpeg_url: MJPEG 流地址（如 http://camera/video.mjpg）
        key: 组件唯一标识
    """
    import streamlit.components.v1 as components
    
    html_content = f"""
    <div style="width: 100%; background: #000; border-radius: 8px; overflow: hidden;">
        <img src="{mjpeg_url}" 
             style="width: 100%; height: auto; display: block;"
             alt="MJPEG Stream"
             onerror="this.src=''; this.alt='连接失败';">
    </div>
    """
    components.html(html_content, height=480)


def get_rtsp_player(rtsp_url: str, key: str = "default") -> Optional[RTSPStreamPlayer]:
    """
    获取 RTSP 播放器实例
    
    Args:
        rtsp_url: RTSP 流地址
        key: 实例标识
        
    Returns:
        播放器实例或 None
    """
    player_key = f"_rtsp_player_{key}"
    if player_key in st.session_state:
        return st.session_state[player_key]
    return None


def cleanup_rtsp_player(key: str = "default"):
    """
    清理 RTSP 播放器资源
    """
    player_key = f"_rtsp_player_{key}"
    if player_key in st.session_state:
        player = st.session_state[player_key]
        if hasattr(player, 'stop'):
            player.stop()
        del st.session_state[player_key]
