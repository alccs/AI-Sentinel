"""
独立 MJPEG 流服务器
在后台运行 Flask 服务器，提供 MJPEG 流
Streamlit 前端通过 img 标签直接连接，完全避免闪烁
"""
import threading
import time
import cv2
import logging
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class MJPEGStreamServer:
    """
    独立的 MJPEG 流服务器
    使用后台线程运行 HTTP 服务器，提供 /video_feed 端点
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._running = False
        self._server_thread = None
        self._port = 8765
        self._frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._app = None
        self._server = None
        
    def set_frame_provider(self, provider: Callable[[], Optional[np.ndarray]]):
        """设置帧提供器函数"""
        self._frame_provider = provider
        
    def start(self, port: int = 8765):
        """启动 MJPEG 服务器"""
        if self._running:
            return self._port
            
        self._port = port
        
        try:
            from flask import Flask, Response
            
            self._app = Flask(__name__)
            
            @self._app.route('/video_feed')
            def video_feed():
                def generate():
                    last_frame_time = time.time()
                    target_interval = 1.0 / 30.0  # 目标30 FPS
                    
                    while self._running:
                        if self._frame_provider:
                            frame = self._frame_provider()
                            if frame is not None:
                                # 降低JPEG质量以减少带宽和编码时间
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                frame_bytes = buffer.tobytes()
                                
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                                
                                # 动态帧率控制：仅在需要时sleep
                                current_time = time.time()
                                elapsed = current_time - last_frame_time
                                sleep_time = target_interval - elapsed
                                if sleep_time > 0:
                                    time.sleep(sleep_time)
                                last_frame_time = time.time()
                            else:
                                # 无帧时等待
                                time.sleep(0.01)
                        else:
                            time.sleep(0.01)
                        
                return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
            
            @self._app.route('/health')
            def health():
                return "OK"
            
            # 在后台线程启动服务器
            def run_server():
                from werkzeug.serving import make_server
                self._server = make_server('127.0.0.1', self._port, self._app, threaded=True)
                logger.info(f"MJPEG 服务器启动在 http://127.0.0.1:{self._port}/video_feed")
                self._server.serve_forever()
            
            self._running = True
            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            
            # 等待服务器启动
            time.sleep(0.5)
            
            return self._port
            
        except ImportError:
            logger.error("Flask 未安装，无法启动 MJPEG 服务器")
            raise
        except Exception as e:
            logger.error(f"启动 MJPEG 服务器失败: {e}")
            raise
            
    def stop(self):
        """停止服务器"""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
            
    @property
    def is_running(self) -> bool:
        return self._running
        
    @property
    def stream_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/video_feed"


def render_mjpeg_iframe(stream_url: str, height: int = 600):
    """
    渲染 MJPEG 流 iframe
    完全独立于 Streamlit 刷新机制
    """
    import streamlit as st
    import streamlit.components.v1 as components
    
    html = f'''
    <div style="width:100%; background:#0E1117; border-radius:8px; overflow:hidden; 
                border: 1px solid #41444C;">
        <div style="padding:8px 16px; background:#262730; border-bottom: 1px solid #41444C;
                    display:flex; justify-content:space-between; align-items:center;">
            <span style="color:white; font-weight:600; font-size:14px;">🔴 实时监控</span>
            <span style="color:rgba(255,255,255,0.7); font-size:12px;" id="stream_status">连接中...</span>
        </div>
        <div style="width:100%; aspect-ratio: 16/9; background:#000; position:relative;">
            <img id="mjpeg_stream" 
                 src="{stream_url}" 
                 style="width:100%; height:100%; object-fit:contain; display:block;"
                 alt="实时视频流"
                 onload="document.getElementById('stream_status').textContent='🟢 已连接'"
                 onerror="document.getElementById('stream_status').textContent='🔴 连接失败'">
        </div>
    </div>
    '''
    components.html(html, height=height)


def get_mjpeg_server() -> MJPEGStreamServer:
    """获取 MJPEG 服务器单例"""
    return MJPEGStreamServer()


def start_mjpeg_server_for_ingestion(ingestion_service, port: int = 8765) -> str:
    """
    为 IngestionService 启动 MJPEG 服务器
    
    Args:
        ingestion_service: VideoIngestionService 实例
        port: 服务器端口
        
    Returns:
        MJPEG 流 URL
    """
    server = get_mjpeg_server()
    server.set_frame_provider(ingestion_service.get_current_frame)
    server.start(port)
    return server.stream_url
