"""
WebUI Components Package
优化的前端组件,支持独立视频播放和异步更新
"""

from .video_player import render_video_player, render_webrtc_player
from .video_section import (
    render_video_section_optimized,
    render_rtsp_player,
    render_file_player,
    handle_pending_alert_dialog
)

__all__ = [
    "render_video_player",
    "render_webrtc_player",
    "render_video_section_optimized",
    "render_rtsp_player",
    "render_file_player",
    "handle_pending_alert_dialog"
]

