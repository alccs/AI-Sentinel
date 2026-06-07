"""
优化的视频显示区块 - 使用直连流播放器

替换原app.py中1272-1385行的视频显示逻辑
"""
import streamlit as st
from pathlib import Path
from datetime import datetime
import time
from src.common.queue_manager import queue_manager
import queue
import cv2
import base64
import re

@st.dialog("📸 报警抓拍", width="large")
def show_alert_image_dialog(image_path, caption):
    """Show details and image of an alert."""
    import os
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=caption, output_format="JPEG", width="stretch")
    else:
        st.warning(f"图片路径缺失或文件不存在: {image_path}")
    
    if st.button("关闭", key="btn_close_opt_alert"):
        # st.rerun() # Remove rerun to prevent conflict
        pass

def handle_pending_alert_dialog():
    """
    Handle pending alert image dialog requests.
    Must be called OUTSIDE of any fragment to avoid conflicts.
    Only opens if no other dialogs are active.
    """
    # 检查是否有其他对话框打开
    has_other_dialog = (
        st.session_state.get('show_settings_dialog', False) or
        st.session_state.get('show_db_browser', False) or
        st.session_state.get('show_alarm_dialog', False) or
        st.session_state.get('show_search_detail_dialog', False)
    )
    
    # 只有在没有其他对话框时才处理待处理的报警图片
    if not has_other_dialog and st.session_state.get('_pending_alert_image'):
        pending = st.session_state._pending_alert_image
        st.session_state._pending_alert_image = None  # Clear immediately
        show_alert_image_dialog(pending['path'], pending['caption'])


@st.fragment(run_every=3.0)  # 降低刷新频率，避免与视频流冲突
def render_alert_consumer(services):
    """
    独立运行的报警消费与渲染组件
    """
    # 初始化 toast 计时器 & 渲染计数
    if 'last_toast_time' not in st.session_state:
        st.session_state.last_toast_time = 0
    if '_alert_render_count' not in st.session_state:
        st.session_state._alert_render_count = 0
    st.session_state._alert_render_count += 1
    
    # Poll alerts
    has_new_alert = False
    latest_important_alert = None
    
    # Max alerts to fetch per cycle to avoid blocking
    for _ in range(10): 
        try:
            alert = queue_manager.get_nowait("alert_queue")
            if alert:
                if 'alerts' not in st.session_state:
                    st.session_state.alerts = []
                st.session_state.alerts.insert(0, alert)
                if len(st.session_state.alerts) > 50:
                    st.session_state.alerts.pop()
                has_new_alert = True
                
                if not latest_important_alert:
                    latest_important_alert = alert
                elif alert.severity in ["Critical", "High"] and (not latest_important_alert or latest_important_alert.severity not in ["Critical", "High"]):
                        latest_important_alert = alert
            else:
                break
        except queue.Empty:
            break
    
    # Toast logic
    if latest_important_alert:
        import time as py_time
        current_ts = py_time.time()
        # Rate limit toasts to avoid flooding
        if current_ts - st.session_state.last_toast_time > 2.0:
            try:
                st.toast(f"🚨 [{latest_important_alert.severity}] {latest_important_alert.risk_type}\n{latest_important_alert.description[:30]}...", icon="🚨")
            except Exception:
                pass
            st.session_state.last_toast_time = current_ts
    
    # Render Alerts List
    if not st.session_state.get('alerts'):
        st.caption("暂无异常报警")
    else:
        # 只显示前5条
        # Custom CSS to ensure st.error has the desired background color
        # Custom CSS to force styling on the alert box
        # Custom CSS to force styling on the alert box
        # We target both possible test-ids to be safe across versions
        st.markdown("""
        <style>
        /* Global Alert Override */
        div[data-testid="stAlert"] {
            padding: 0.5rem 1rem !important;
        }
        
        /* Custom Alert Card */
        .alert-card {
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 10px;
            border-left: 5px solid #ccc;
            background-color: rgba(255, 255, 255, 0.05); /* Adaptive background */
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Severity Variants */
        .alert-critical {
            border-left-color: #ff4b4b; /* Red */
            background-color: rgba(255, 75, 75, 0.08);

        }
        .alert-warning {
            border-left-color: #ffa421; /* Orange */
            background-color: rgba(255, 164, 33, 0.08);
        }
        .alert-info {
            border-left-color: #21c354; /* Green */
            background-color: rgba(33, 195, 84, 0.08);
        }
        
        .alert-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .alert-title {
            font-weight: 700;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .alert-body {
            font-size: 0.95rem;
            line-height: 1.4;
            color: inherit; /* Inherit for dark/light mode */
            margin-bottom: 8px;
        }
        
        /* Badges for L1/L2 tags */
        .tech-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-family: "SF Mono", "Roboto Mono", monospace;
            font-weight: 600;
            margin-right: 6px;
            vertical-align: text-bottom;
            user-select: all;
        }
        .badge-l1 {
            background-color: #e0e0e0;
            color: #333;
            border: 1px solid #ccc;
        }
        /* Dark mode adjustment for L1 */
        @media (prefers-color-scheme: dark) {
            .badge-l1 {
                background-color: #444;
                color: #ddd;
                border-color: #555;
            }
        }
        
        .badge-l2 {
            background-color: #ffe0b2;
            color: #e65100;
            border: 1px solid #ffb74d;
        }
        @media (prefers-color-scheme: dark) {
            .badge-l2 {
                background-color: #5a3000;
                color: #ffb74d;
                border-color: #7c4000;
            }
        }
        
        .stButton>button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for i, alert in enumerate(st.session_state.alerts[:50]):
            # Check severity & L1 override
            sev = alert.severity.lower()
            desc_preview = alert.description
            
            # Logic update: Downgrade L1_INFO to prevent Red alerts (Robust Regex)
            is_l1 = bool(re.search(r"\[L1[_\s]", desc_preview, re.IGNORECASE))
            
            if is_l1:
                 severity_class = "alert-info"
            else:
                if sev == "critical":
                    severity_class = "alert-critical"
                elif sev in ["high", "warning"]:
                    severity_class = "alert-warning"
                else:
                    severity_class = "alert-info"
            
            rt = alert.risk_type
            type_icon = "🔔"
            if "人" in rt: type_icon = "👤"
            elif "狗" in rt or "犬" in rt: type_icon = "🐕"
            elif "车" in rt: type_icon = "🚗"
            elif "火" in rt or "烟" in rt: type_icon = "🔥"
            elif "摔倒" in rt or "倒地" in rt: type_icon = "⚠️"

            # Process description to badge tags
            desc_html = alert.description
            desc_html = re.sub(
                r'(\[L1_[A-Za-z0-9_]+\])', 
                r'<span class="tech-badge badge-l1">\1</span>', 
                desc_html
            )
            desc_html = re.sub(
                r'(\[L2_[A-Za-z0-9_]+\])', 
                r'<span class="tech-badge badge-l2">\1</span>', 
                desc_html
            )

            # Create Card HTML
            card_html = f"""
            <div class="alert-card {severity_class}" style="position: relative; padding-bottom: 45px;">
                <div class="alert-header">
                    <div class="alert-title">{type_icon} {rt}</div>
                    <div class="alert-time">{alert.timestamp.strftime('%H:%M:%S')}</div>
                </div>
                <div class="alert-body">
                    {desc_html}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Position button "inside" the card
            st.markdown(f'<div style="margin-top: -50px; margin-bottom: 15px; padding-right: 15px; text-align: right; position: relative; z-index: 5;">', unsafe_allow_html=True)
            if st.button("查看图片", key=f"btn_frag_view_{alert.frame_id}_{i}", help="查看抓拍画面"):
                # Explicitly close other dialogs to prevent conflicts/stuck states
                st.session_state.show_settings_dialog = False
                st.session_state.show_db_browser = False
                st.session_state.show_alarm_dialog = False
                
                st.session_state._pending_alert_image = {
                    'path': alert.image_path,
                    'caption': f"[{alert.timestamp.strftime('%H:%M:%S')}] {alert.risk_type}"
                }
                # Rerun to trigger the dialog handler (outside fragment)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_video_section_optimized(services, col_video, alert_placeholder=None):
    """
    优化的视频显示区块 - 使用 MJPEG 服务器实现流畅播放
    
    Args:
        services: 服务管理器
        col_video: Streamlit列容器
        alert_placeholder: 用于显示报警的st.empty占位符
    """
    with col_video:
        st.subheader("📷 实时画面")

        # 状态栏
        status_cols = st.columns([0.7, 0.3])
        with status_cols[0]:
            if st.session_state.monitoring:
                st.caption("🔴 直播中")
            else:
                st.caption("⚪ 已停止")

        with status_cols[1]:
            # Use fragment for auto-refreshing time
            render_time_display_fragment(services)

        if st.session_state.monitoring and services.ingestion:
            # 尝试使用 MJPEG 服务器（最流畅）
            try:
                from .mjpeg_server import get_mjpeg_server, render_mjpeg_iframe
                
                server = get_mjpeg_server()
                
                # 如果服务器未运行，启动它
                if not server.is_running:
                    # 使用零拷贝模式获取帧，减少内存开销
                    server.set_frame_provider(lambda: services.ingestion.get_current_frame(copy=False))
                    server.start(port=8765)
                    time.sleep(0.5)  # 等待服务器启动
                
                # 渲染 MJPEG iframe - 完全独立于 Streamlit
                render_mjpeg_iframe(server.stream_url, height=600)
                
            except Exception as e:
                # 回退到传统模式
                st.warning(f"MJPEG 服务器不可用: {e}")
                render_snapshot_mode(services)

            # 自定义渲染报警区块 (独立 Fragment)
            if alert_placeholder:
                with alert_placeholder:
                    render_alert_consumer(services)

            # 关键: 仅更新时间显示
            # Time display is handled by independent fragment above
            pass

        else:
            st.info("监控已停止。请点击侧边栏「启动监控」按钮。")


def render_rtsp_player(rtsp_url: str, time_placeholder, services, alert_placeholder=None):
    """
    渲染RTSP直连播放器 - 使用独立 HTML 播放器避免闪烁
    """
    import streamlit.components.v1 as components
    
    # 尝试使用独立播放器
    try:
        from .rtsp_player import RTSPStreamPlayer
        
        # 初始化或获取播放器实例
        player_key = "_independent_rtsp_player"
        if player_key not in st.session_state:
            player = RTSPStreamPlayer(rtsp_url)
            player.start()
            st.session_state[player_key] = player
        
        player = st.session_state[player_key]
        
        # 获取当前帧并以 Base64 编码显示
        frame = player.get_frame()
        if frame is not None:
            # 转换为 JPEG Base64
            import cv2
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 使用 HTML img 标签，完全独立于 Streamlit 刷新
            html = f'''
            <div style="width:100%; background:#0E1117; border-radius:12px; overflow:hidden; border: 1px solid #41444C; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                <img src="data:image/jpeg;base64,{frame_b64}" 
                     style="width:100%; height:auto; display:block;" 
                     alt="实时画面">
            </div>
            '''
            components.html(html, height=500)
        else:
            st.info("⏳ 等待视频信号...")
            
    except ImportError:
        # 回退到快照模式
        render_snapshot_mode(services)


def render_snapshot_mode(services):
    """
    快照模式 - 实时刷新 (仅视频)
    """
    # Define the fragment rendering function
    @st.fragment(run_every=0.2)  # 5 FPS - 更稳定
    def render_content():
        try:
            # Check if we should still be running
            if not (services.ingestion and services.ingestion.is_running() and st.session_state.monitoring):
                st.info("监控已停止")
                return

            # 获取当前帧
            frame = services.ingestion.get_current_frame()
            if frame is not None and frame.size > 0:
                # 将 BGR 转换为 RGB 以正确显示颜色
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
                st.session_state.video_persistence_counter = 0
            else:
                # Persistence Logic
                if 'video_persistence_counter' not in st.session_state:
                    st.session_state.video_persistence_counter = 0
                
                if st.session_state.video_persistence_counter > 10:
                    st.info("⏳ 等待视频信号...")
                
                st.session_state.video_persistence_counter += 1
        except Exception as e:
            # Suppress websocket closed errors which are benign on page refresh
            # tornado.websocket.WebSocketClosedError or StreamClosedError
            error_str = str(e)
            if "WebSocketClosedError" in error_str or "StreamClosedError" in error_str:
                pass
            else:
                # Log other real errors
                # st.error(f"Render error: {e}")
                pass
            
    # Call the fragment function to start the loop
    render_content()


def render_file_player(services, time_placeholder, alert_placeholder=None):
    """
    本地文件播放器
    """
    # 回退到快照模式 (最稳健的分析显示方式)
    render_snapshot_mode(services)


@st.fragment(run_every=1.0)
def render_time_display_fragment(services):
    """
    更新时间显示 (独立刷新片段)
    """
    if services.ingestion and services.ingestion.time_synchronizer:
        try:
            sync_stats = services.ingestion.get_time_sync_stats()
            last_time = sync_stats.get("last_parsed_time", "N/A")
            sync_successes = sync_stats.get("sync_successes", 0)
            sync_failures = sync_stats.get("sync_failures", 0)

            if last_time and last_time != "N/A":
                st.markdown(f"🕒 **VLM时间**: {last_time}")
                st.caption(f"同步: 成功 {sync_successes}, 失败 {sync_failures}")
            else:
                st.markdown("🕒 **VLM时间**: 同步中...")

        except Exception as e:
            st.markdown(f"🕒 **错误**: {str(e)}")
    else:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"🕒 **系统时间**: {current_time}")
