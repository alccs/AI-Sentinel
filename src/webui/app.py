"""
AI-Sentinel Dashboard - Streamlit Web Interface
Run with: streamlit run src/webui/app.py
"""
import streamlit as st
import cv2
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import sys
import os
import queue
from PIL import Image
from loguru import logger
import re
import logging

# Suppress annoying Streamlit fragment warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.fragment").setLevel(logging.ERROR)

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion import VideoIngestionService
from src.analysis import AnalysisWorker, VLMAnalyzer, VectorStore
from src.analysis.vlm import DEFAULT_SYSTEM_PROMPT
from src.common.queue_manager import queue_manager
from src.common.types import Alert
from src.common.config import get_config

# Utility function to safely display images with error handling
# Utility function to safely display images with error handling
# Use cache to stabilize media IDs and prevent "Missing file" errors on rerun
@st.cache_data(ttl=60, show_spinner=False)
def load_image_from_path(path_str):
    """Load image from path and return PIL object. Cached to prevent media ID churn."""
    if not os.path.exists(path_str):
        return None
    try:
        with Image.open(path_str) as img:
            return img.copy()
    except Exception:
        return None

def safe_image_display(image_data, caption=None, **kwargs):
    """Safely display an image (path, array, or PIL) with proper error handling."""
    try:
        # Convert path to PIL Image using cached loader
        # This ensures the same PIL object (same data hash) is returned for the same path
        # preventing Streamlit from generating new transient media IDs that get GC'd
        if isinstance(image_data, (str, Path)):
            path_str = str(image_data)
            # Use cached loader
            loaded_img = load_image_from_path(path_str)
            
            if loaded_img is None:
                st.caption("⚠️ 图片文件已被清理")
                return False
            image_data = loaded_img
        
        st.image(image_data, caption=caption, **kwargs)
        return True
    except Exception as e:
        # Suppress but log transient media storage errors
        error_msg = str(e)
        if "MediaFileStorageError" in error_msg or "KeyError" in error_msg:
            # Only show simplified message to user, don't clear cache aggressively
            st.caption("⚠️ 图片加载中...")
            # st.cache_data.clear() # Avoid clearing cache as it might cause more churn
        else:
            st.caption(f"⚠️ 图片显示失败: {error_msg}")
        return False

# Page Config
st.set_page_config(
    page_title="AI-Sentinel 智能监控",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# --- Core Service Initialization ---

def generate_system_prompt(rules, alert_mode="keyword"):
    """
    Generate VLM system prompt based on alert mode.
    
    Args:
        rules: Alarm rules dictionary
        alert_mode: "keyword" (纯描述) or "ai_tag" (带ALERT标签)
    """
    # 直接使用用户在 vlm.py 中定义的 DEFAULT_SYSTEM_PROMPT
    # 这样用户修改 vlm.py 后可以直接生效
    return DEFAULT_SYSTEM_PROMPT


@st.cache_resource
def init_system_singletons(version=1.0):
    """
    Initialize background services as singletons.
    Uses st.cache_resource to ensure only one instance exists across sessions/refreshes.
    Args:
        version: Cache invalidation key. Increment to force reload.
    """
    from src.analysis import get_embedder, BaseEmbedder
    from src.analysis.vector_db import VisualVectorStore
    from src.notification.manager import NotificationManager
    
    class SingletonServiceManager:
        def __init__(self):
            self.ingestion: VideoIngestionService = None
            self.worker: AnalysisWorker = None
            self.visual_embedder: BaseEmbedder = None
            self.visual_store: VisualVectorStore = None
            self.notifier: NotificationManager = None
            self.lock = threading.Lock()
            
            # Pre-load config or resources if needed here
            logger.info("Initializing SingletonServiceManager")
            
            # Initialize Notifier
            self.notifier = NotificationManager()
        
        def _init_text_embedder(
            self, 
            provider: str = "api",
            api_key: str = "", 
            base_url: str = "https://api.openai.com/v1", 
            model_name: str = "text-embedding-ada-002",
            local_model_path: str = ""
        ):
            """Lazy initialize Embedder based on provider mode."""
            if self.visual_embedder is None:
                try:
                    if provider == "local":
                        print(f"🔄 正在初始化本地嵌入模型: {local_model_path}...")
                    else:
                        print(f"🔄 正在初始化API嵌入模型: {model_name}...")
                    
                    self.visual_embedder = get_embedder(
                        provider=provider,
                        api_key=api_key,
                        api_url=base_url,
                        model_name=model_name,
                        local_model_path=local_model_path,
                    )
                    self.visual_store = VisualVectorStore()
                    
                    # Test connection
                    if self.visual_embedder.test_connection():
                        print(f"✅ 嵌入模型初始化成功 (模式: {provider})")
                    else:
                        print("⚠️ 嵌入模型连接测试失败")
                        self.visual_embedder = None
                        self.visual_store = None
                except Exception as e:
                    print(f"⚠️ 嵌入模型初始化失败: {e}")
                    self.visual_embedder = None
                    self.visual_store = None
        
        def start_system(self, input_source, is_file=True, enable_visual=False, visual_model_path=None):
            with self.lock:
                # 1. Stop existing services if running to avoid leaks
                self.stop_system()
                
                # 2. Clear queues (start fresh)
                queue_manager.clear_queue("frame_queue")
                queue_manager.clear_queue("alert_queue")
                
                # 3. Load Config
                conf = get_config()
                ocr_roi = conf.get("ocr.roi", [0.65, 0.85, 0.35, 0.15])
                max_frames = conf.get("storage.max_saved_frames", 50)
                analysis_interval = st.session_state.get("analysis_interval", conf.get("analysis.interval", 3.0))  # 从配置读取
                enable_time_sync = conf.get("ocr.enable_time_sync", True)
                
                # 4. 初始化 VLM 分析器（用于时间同步和内容分析）
                vlm_api_url = st.session_state.get("vlm_api_url", "http://127.0.0.1:1234/v1")
                vlm_api_key = st.session_state.get("vlm_api_key", "")
                vlm_model_name = st.session_state.get("vlm_model_name", "qwen/qwen3-vl-4b")
                
                # Motion Threshold
                motion_threshold = float(conf.get("analysis.motion_threshold", 5.0))
                
                # Get alarm rules and generate dynamic prompt
                alarm_conf = conf.get("alarms.rules", {})
                alert_mode = conf.get("alarms.mode", "keyword")  # "keyword" or "ai_tag"
                max_alert_images = int(conf.get("alarms.max_images", 1000))
                
                # Configure Notification Manager
                notif_conf = conf.get("notification", {})
                if notif_conf:
                    self.notifier.configure(notif_conf)
                
                # 根据报警模式生成对应的提示词
                vlm_system_prompt = generate_system_prompt(alarm_conf, alert_mode)
                
                # Get context size from config/session
                vlm_context_size = st.session_state.get("vlm_context_size", config.get("analysis.context_size", 3))
                
                # 日志记录当前使用的报警规则和模式
                enabled_rules = [k for k, v in alarm_conf.items() if v.get("enabled", True)]
                logger.info(f"Alert mode: {alert_mode}, active rules: {enabled_rules}")
                
                vlm_analyzer = None
                if vlm_api_url and vlm_api_url.strip():
                    vlm_analyzer = VLMAnalyzer(
                        backend="custom",
                        model_name=vlm_model_name,
                        api_base_url=vlm_api_url,
                        api_key=vlm_api_key or "EMPTY",
                        system_prompt=vlm_system_prompt
                    )
                else:
                    vlm_analyzer = VLMAnalyzer(backend='auto', system_prompt=vlm_system_prompt)
                
                # 5. Start Ingestion Service with VLM time sync
                logger.info(f"Starting ingestion with source: {input_source}")
                # Define deletion callback
                def deletion_cb(doc_id):
                    # Run in try-except to avoid crashing ingestion thread
                    try:
                        if self.worker:
                            # 1. Check Vector Store (Text/Meta)
                            should_delete_text = True
                            if hasattr(self.worker, 'vector_store') and self.worker.vector_store:
                                try:
                                    # Safe check for alerts before deleting
                                    # Accessing underlying collection to avoid ID parsing issues
                                    # doc_id passed is "camera_id_frame_id"
                                    res = self.worker.vector_store._collection.get(ids=[doc_id], include=['metadatas'])
                                    if res and res['ids'] and res['metadatas'][0]:
                                        meta = res['metadatas'][0]
                                        # Convert boolean stored as generic (Chroma sometimes issues with bools, verify)
                                        # Usually stored as bool or int.
                                        has_alert = meta.get('has_alert', False)
                                        # Also check alert_type existence
                                        if has_alert or meta.get('alert_type'):
                                            should_delete_text = False
                                            # logger.debug(f"Skipping deletion of alert frame: {doc_id}")
                                            
                                    if should_delete_text:
                                        self.worker.vector_store.delete(doc_id)
                                except Exception as e:
                                    logger.warning(f"Error checking text store deletion: {e}")

                            # 2. Check Visual Store
                            if hasattr(self.worker, 'visual_store') and self.worker.visual_store:
                                try:
                                    # Visual store ID has prefix "semantic_" or "semantic_default_" (handled by delete logic fix)
                                    # But to check metadata we need the correct ID.
                                    # Based on our fix, it is "semantic_{doc_id}"
                                    visual_id = f"semantic_{doc_id}"
                                    res = self.worker.visual_store._collection.get(ids=[visual_id], include=['metadatas'])
                                    should_delete_visual = True
                                    
                                    if res and res['ids'] and res['metadatas'][0]:
                                        meta = res['metadatas'][0]
                                        has_alert = meta.get('has_alert', False)
                                        if has_alert or meta.get('alert_info'):
                                            should_delete_visual = False
                                    
                                    if should_delete_visual:
                                        self.worker.visual_store.delete(doc_id) # delete method handles prefix logic 
                                except Exception as e:
                                    logger.warning(f"Error checking visual store deletion: {e}")

                    except Exception as e:
                        logger.error(f"Deletion callback failed: {e}")

                # Determine if VLM time sync should be enabled
                vlm_client_for_sync = vlm_analyzer if enable_time_sync else None

                if is_file:
                    self.ingestion = VideoIngestionService.from_file(
                        input_source,
                        camera_id="cam_main",
                        analysis_interval=analysis_interval,
                        loop_video=True,
                        max_saved_frames=max_frames,
                        roi_config=ocr_roi,
                        vlm_client=vlm_client_for_sync,
                        deletion_callback=deletion_cb,
                        motion_detection_mode=st.session_state.get("motion_detection_mode", "enhanced"),
                        motion_heartbeat=st.session_state.get("motion_heartbeat", 15.0),
                    )
                else:
                    self.ingestion = VideoIngestionService.from_rtsp(
                        str(input_source),
                        camera_id="cam_main",
                        analysis_interval=analysis_interval,  
                        max_saved_frames=max_frames,
                        roi_config=ocr_roi,  
                        vlm_client=vlm_client_for_sync, 
                        deletion_callback=deletion_cb,
                        motion_detection_mode=st.session_state.get("motion_detection_mode", "enhanced"),
                        motion_heartbeat=st.session_state.get("motion_heartbeat", 15.0),
                    )

                # Inject motion threshold and object detection settings
                self.ingestion.motion_threshold = motion_threshold
                self.ingestion.object_detection_enabled = st.session_state.get("obj_det_enabled", True)
                self.ingestion.object_detection_classes = st.session_state.get("obj_det_classes", ["person", "vehicle", "animal"])
                self.ingestion.start()
                
                # 6. Initialize embedding components if needed
                if enable_visual:
                    # Get embedding config from session state
                    emb_provider = st.session_state.get("embedding_provider", "api")
                    emb_api_url = st.session_state.get("embedding_api_url", "https://api.openai.com/v1")
                    emb_api_key = st.session_state.get("embedding_api_key", "")
                    emb_model_name = st.session_state.get("embedding_model_name", "text-embedding-ada-002")
                    emb_local_path = st.session_state.get("embedding_local_model_path", "")
                    
                    self._init_text_embedder(
                        provider=emb_provider,
                        api_key=emb_api_key,
                        base_url=emb_api_url,
                        model_name=emb_model_name,
                        local_model_path=emb_local_path
                    )
                
                # 7. Start Analysis Worker (使用已创建的 VLM 分析器)
                logger.info("Starting analysis worker...")
                self.worker = AnalysisWorker(
                    vlm_analyzer=vlm_analyzer,  # 使用已创建的 VLM 分析器
                    visual_embedder=self.visual_embedder,
                    visual_store=self.visual_store,
                    enable_visual_index=enable_visual,
                    batch_size=1,
                    alarm_rules=alarm_conf,  # Pass alarm rules to worker
                    alert_mode=alert_mode,  # Pass alert mode to worker
                    alert_save_dir="./data/alerts",
                    max_alert_images=max_alert_images,
                    context_size=vlm_context_size
                )
                    
                self.worker.start()
                return True

        def stop_system(self):
            """Stop all background services safely."""
            if self.ingestion and self.ingestion.is_running():
                logger.info("Stopping Ingestion Service...")
                self.ingestion.stop()
            if self.worker and self.worker.is_running():
                logger.info("Stopping Analysis Worker...")
                self.worker.stop()
            
            # Allow components to be GC'd if needed, though we reuse the manager
            # self.ingestion = None
            # self.worker = None

    return SingletonServiceManager()

# Initialize Singletons
# Initialize Singletons
services = init_system_singletons(version=1.2)


# --- Session State Management ---

# Load persistent config
config = get_config()

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

# 确保对话框状态初始化为 False
if 'show_db_browser' not in st.session_state:
    st.session_state.show_db_browser = False
if 'show_settings_dialog' not in st.session_state:
    st.session_state.show_settings_dialog = False
if 'show_alarm_dialog' not in st.session_state:
    st.session_state.show_alarm_dialog = False

# Initialize settings from saved config
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = True
    st.session_state.video_source_type = config.get("video_source.type", "file")
    st.session_state.video_source_path = config.get("video_source.path", str(ROOT_DIR / "data/videos/sample.mp4"))
    st.session_state.vlm_api_url = config.get("vlm.api_url", "http://127.0.0.1:1234/v1")
    st.session_state.vlm_api_key = config.get("vlm.api_key", "")
    st.session_state.vlm_model_name = config.get("vlm.model_name", "qwen/qwen3-vl-4b")
    st.session_state.vlm_model_name = config.get("vlm.model_name", "qwen/qwen3-vl-4b")
    st.session_state.vlm_system_prompt = config.get("vlm.system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.vlm_context_size = config.get("analysis.context_size", 3)
    # Embedding settings (dual mode)
    st.session_state.embedding_provider = config.get("embedding.provider", "api")
    st.session_state.embedding_api_url = config.get("embedding.api_url", "https://api.openai.com/v1")
    st.session_state.embedding_api_key = config.get("embedding.api_key", "")
    st.session_state.embedding_model_name = config.get("embedding.model_name", "text-embedding-3-small")
    st.session_state.embedding_local_model_path = config.get("embedding.local_model_path", "")
    st.session_state.analysis_interval = config.get("analysis.interval", 1.0)
    st.session_state.max_saved_frames = config.get("storage.max_saved_frames", 50)
    
    # Motion detection settings
    st.session_state.motion_threshold = config.get("analysis.motion_threshold", 5.0)
    st.session_state.motion_detection_mode = config.get("analysis.motion_detection_mode", "enhanced")
    st.session_state.motion_heartbeat = config.get("analysis.motion_heartbeat", 15.0)
    
    # 初始化 ROI 配置到 session_state，确保框选区域持久化
    saved_roi = config.get("ocr.roi", [0.65, 0.85, 0.35, 0.15])
    st.session_state.new_roi_ratios = saved_roi  # 将配置文件中的 ROI 设为当前选择
    st.session_state.enable_time_sync = config.get("ocr.enable_time_sync", True)
    
    # Notification Settings
    st.session_state.notif_enabled = config.get("notification.enabled", False)
    st.session_state.notif_provider = config.get("notification.provider", "wechat")
    st.session_state.notif_wechat_appid = config.get("notification.wechat_appid", "")
    st.session_state.notif_wechat_secret = config.get("notification.wechat_secret", "")
    st.session_state.notif_wechat_openid = config.get("notification.wechat_openid", "")
    st.session_state.notif_wechat_openid = config.get("notification.wechat_openid", "")
    st.session_state.notif_wechat_template = config.get("notification.wechat_template_id", "")
    st.session_state.notif_bark_key = config.get("notification.bark_key", "")
    st.session_state.notif_bark_key = config.get("notification.bark_key", "")
    st.session_state.notif_cooldown = config.get("notification.cooldown", 60)

    # Object Detection Settings
    st.session_state.obj_det_enabled = config.get("analysis.obj_det_enabled", True)
    st.session_state.obj_det_classes = config.get("analysis.obj_det_classes", ["person", "vehicle", "animal"])

# Initialize separate paths for File and RTSP to remember inputs when switching
if 'video_source_path_file' not in st.session_state:
    # If starting in file mode, use current path, else default
    # Try separate config key first, then fallback to generic
    st.session_state.video_source_path_file = config.get("video_source.path_file", 
        config.get("video_source.path", str(ROOT_DIR / "data/videos/sample.mp4")))

if 'video_source_path_rtsp' not in st.session_state:
    # Try separate config key first, then fallback to generic
    st.session_state.video_source_path_rtsp = config.get("video_source.path_rtsp", 
        config.get("video_source.path", "0"))


# --- Sidebar: Configuration ---

with st.sidebar:
    st.title("⚙️ 系统配置")
    
    # Input Source
    source_type_index = 0 if st.session_state.get("video_source_type", "file") == "file" else 1
    source_type = st.radio("视频源类型", ["本地文件", "RTSP / 摄像头"], index=source_type_index)
    
    if source_type == "本地文件":
        # Support both manual path and file upload
        tab_manual, tab_upload = st.tabs(["📄 手动输入路径", "📤 上传视频文件"])
        
        with tab_manual:
             # Use specific file path state
             default_path = st.session_state.video_source_path_file
             
             def update_file_path():
                 st.session_state.video_source_path_file = st.session_state.input_path_manual
                 
             input_path = st.text_input("文件路径", value=default_path, key="input_path_manual", on_change=update_file_path)
             # Sync back if manual input wasn't triggered but state changed elsewhere (rare but safe)
             if input_path != st.session_state.video_source_path_file:
                 st.session_state.video_source_path_file = input_path
             
        with tab_upload:
            uploaded_file = st.file_uploader("选择视频文件", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file is not None:
                # Create uploads directory if not exists (checked at startup but good to be safe)
                upload_dir = ROOT_DIR / "data/uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the file
                save_path = upload_dir / uploaded_file.name
                
                # Only write if it doesn't exist or if we want to overwrite (st.file_uploader triggers on every run if file present)
                # To avoid re-writing large files on every rerun, we could check size/name, but simple write is safer for updates
                if not save_path.exists():
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"已上传: {uploaded_file.name}")
                
                # If a file is uploaded, it overrides the manual input for the final start
                # We can visually indicate this or just set a session state
                # Define callback to update state safely before next run
                def update_path_state(new_path):
                    st.session_state.video_source_path = new_path
                    st.session_state.video_source_path_file = new_path # Update specific file path
                    st.session_state.input_path_manual = new_path
                
                # If a file is uploaded, it overrides the manual input for the final start
                # We use on_click to update the state BEFORE the script re-runs to avoid "widget already instantiated" error
                if st.button("使用此文件", type="secondary", on_click=update_path_state, args=(str(save_path),)):
                    st.toast(f"已选择上传文件: {uploaded_file.name}", icon="✅")
                    # No explicit rerun needed as button click triggers it, and callback handles state
                    pass
            
            st.caption("提示: 上传的文件将保存在 data/uploads/ 目录下")

        # Logic to determine final effective path
        # If user explicitly clicked "Use this file", session_state.video_source_path is updated.
        # But 'input_path' variable usually binds to the text_input. 
        # So we should respect what's currently in session_state if it differs from default?
        # Actually simpler: Let the manual input show the current effective path always.
        
        # Refined Logic:
        # The 'input_path' variable is driven by the manual text input widget.
        # If the user uploads a file and clicks 'Use this file', we update the session state for the manual input widget
        # and rerun, so the manual input box shows the new path.
        
        # Ensure input_path reflects the chosen one
        # Because we used key="input_path_manual", the value comes from widget state.
        # If we updated session_state["input_path_manual"] in the button callback, it will reflect here.
        
    else:
        # RTSP Mode
        default_rtsp = st.session_state.video_source_path_rtsp
        
        def update_rtsp_path():
            st.session_state.video_source_path_rtsp = st.session_state.input_path_rtsp
            
        input_path = st.text_input("RTSP地址 / URL (或0)", value=default_rtsp, key="input_path_rtsp", on_change=update_rtsp_path)
        # Sync
        if input_path != st.session_state.video_source_path_rtsp:
             st.session_state.video_source_path_rtsp = input_path
        
    # Standardize VLM Configuration box

    st.subheader("🤖 AI 模型配置")
    vlm_choice = st.selectbox(
        "AI 模型后端", 
        ["自动检测 (Auto)", "自定义 API (OpenAI/vLLM)"], 
        index=0
    )
    
    if vlm_choice == "自定义 API (OpenAI/vLLM)":
        st.session_state.vlm_backend_choice = "custom"
        st.session_state.custom_api_url = st.text_input("API 地址 (Base URL)", "https://api.openai.com/v1")
        st.session_state.custom_api_key = st.text_input("API 密钥 (Key)", type="password")
        st.session_state.custom_model_name = st.text_input("模型名称", "gpt-4o")
    else:
        st.session_state.vlm_backend_choice = "auto"

    st.divider()
    # Semantic indexing option
    st.caption("🔍 语义搜索设置")
    enable_visual = st.checkbox(
        "启用语义搜索 (文本嵌入)",
        value=False,
        help="启用后可使用文本嵌入进行语义搜索，需要配置 Embedding API"
    )
    
    if enable_visual:
        st.info("💡 语义搜索使用 VLM 描述 + 文本 Embedding API 的方案，无需加载大模型到本地")
        # No need for visual_model_path anymore

    col_btn1, col_btn2 = st.columns(2)
    
    # Callback for Start
    def start_monitoring_cb():
        try:
            # Clear all dialogs immediately
            st.session_state.show_settings_dialog = False
            st.session_state.show_alarm_dialog = False
            st.session_state.show_db_browser = False
            
            # 1. Update Config/State
            is_file_mode = (source_type == "本地文件")
            st.session_state.video_source_type = "file" if is_file_mode else "rtsp"
            
            # Use current input_path value attached to the widget key
            # We need to grab it from session state directly as arguments are evaluated at definition time
            # but callback runs at click time. 
            # However, simpler to pass current input_path as arg to callback? 
            # Streamlit widgets update session_state before callback.
            # We will use the values passed in args to be safe.
            pass

        except Exception as e:
            st.error(f"Error in start prep: {e}")

    with col_btn1:
        # We need to capture the current input_path from the widgets above
        # The widgets write to 'input_path_manual' or 'input_path_rtsp' in session_state
        final_input_path = st.session_state.get("input_path_manual") if source_type == "本地文件" else st.session_state.get("input_path_rtsp")
        
        # Fallback if keys missing (e.g. first run defaults)
        if not final_input_path:
             final_input_path = st.session_state.video_source_path_file if source_type == "本地文件" else st.session_state.video_source_path_rtsp

        def handle_start_click():
            try:
                # 1. Clear UI Flags
                st.session_state.show_settings_dialog = False
                st.session_state.show_alarm_dialog = False
                st.session_state.show_db_browser = False
                
                # 2. Save Config
                is_file = (source_type == "本地文件")
                st.session_state.video_source_type = "file" if is_file else "rtsp"
                st.session_state.video_source_path = final_input_path
                
                config.set("video_source.type", "file" if is_file else "rtsp")
                config.set("video_source.path", final_input_path)
                
                if is_file:
                    st.session_state.video_source_path_file = final_input_path
                    config.set("video_source.path_file", final_input_path)
                else:
                    st.session_state.video_source_path_rtsp = final_input_path
                    config.set("video_source.path_rtsp", final_input_path)
                config.save()
                
                # 3. Restart Services
                services.stop_system()
                services.start_system(
                    final_input_path, 
                    is_file, 
                    enable_visual=enable_visual,
                    visual_model_path=None
                )
                
                st.session_state.monitoring = True
                st.toast("监控系统已启动", icon="✅")
                
            except Exception as e:
                st.error(f"启动失败: {e}")

        st.button("▶️ 启动监控", type="primary", on_click=handle_start_click)
            
    with col_btn2:
        def handle_stop_click():
            services.stop_system()
            st.session_state.monitoring = False
            st.session_state.show_settings_dialog = False
            st.session_state.show_alarm_dialog = False
            st.session_state.show_db_browser = False
            st.toast("监控系统已停止", icon="🛑")
            
        st.button("⏹️ 停止监控", on_click=handle_stop_click)

    # Advanced Settings Button
    st.divider()
    def open_settings_dialog():
        st.session_state.show_db_browser = False
        st.session_state.show_alarm_dialog = False
        st.session_state.show_settings_dialog = True
    
    st.button("⚙️ 高级设置", on_click=open_settings_dialog, key="btn_open_settings_sidebar")

# --- Settings Dialog (Modal Popup) ---
@st.dialog("⚙️ 高级设置", width="large")
def settings_dialog():
    """Settings dialog with close button."""
    
    # Initialize session state for settings if not exists
    if 'vlm_api_url' not in st.session_state:
        st.session_state.vlm_api_url = "http://127.0.0.1:1234/v1"
    if 'vlm_api_key' not in st.session_state:
        st.session_state.vlm_api_key = ""
    if 'vlm_model_name' not in st.session_state:
        st.session_state.vlm_model_name = "qwen/qwen3-vl-4b"
    if 'vlm_system_prompt' not in st.session_state:
        st.session_state.vlm_system_prompt = config.get("vlm.system_prompt", DEFAULT_SYSTEM_PROMPT)
    if 'embedding_api_url' not in st.session_state:
        st.session_state.embedding_api_url = "https://api.openai.com/v1"
    if 'embedding_api_key' not in st.session_state:
        st.session_state.embedding_api_key = ""
    if 'embedding_model_name' not in st.session_state:
        st.session_state.embedding_model_name = "text-embedding-3-small"
    if 'embedding_provider' not in st.session_state:
        st.session_state.embedding_provider = "api"
    if 'embedding_local_model_path' not in st.session_state:
        st.session_state.embedding_local_model_path = ""
    if 'analysis_interval' not in st.session_state:
        st.session_state.analysis_interval = 1.0
    
    # Two columns for VLM and Embedding settings
    col1, col2 = st.columns(2)
    
    with col1:
        # Header with Custom Prompt Button
        c1_head, c1_btn = st.columns([0.65, 0.35])
        with c1_head:
            st.subheader("🤖 VLM 视觉模型")
        
        with c1_btn:
            with st.popover("📝 自定义提示词", width='stretch'):
                st.markdown("**系统提示词 (System Prompt)**")
                st.caption("控制AI的行为模式、检测规则和报警格式。")
                
                # Input for custom prompt
                vlm_system_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.get("vlm_system_prompt", DEFAULT_SYSTEM_PROMPT),
                    height=300,
                    key="dialog_vlm_system_prompt",
                    label_visibility="collapsed"
                )
                
                # Show default for reference
                with st.expander("查看默认提示词"):
                    st.code(DEFAULT_SYSTEM_PROMPT, language="text")
                
                if st.button("重置为默认"):
                    st.session_state.dialog_vlm_system_prompt = DEFAULT_SYSTEM_PROMPT
                    st.rerun()
        
        # VLM Context Size Slider
        vlm_context_size = st.slider(
            "上下文窗口大小 (Context Size)",
            min_value=0,
            max_value=10,
            value=st.session_state.get("vlm_context_size", 3),
            help="包含前 N 帧的分析历史作为上下文。(0 表示无上下文)",
            key="slider_vlm_context_size"
        )
        st.caption(f"当前设置: {'无记忆' if vlm_context_size == 0 else f'包含前 {vlm_context_size} 帧历史'}")

        # --- VLM Presets (New Feature) ---
        presets = config.get("vlm.presets", {})
        if not presets:
            presets = {} # Handle empty/none
            
        preset_names = ["-- 选择预设 --"] + list(presets.keys())
        
        # Preset Selection with Columns
        c_p1, c_p2 = st.columns([0.7, 0.3])
        with c_p1:
            selected_preset = st.selectbox(
                "⚡ 快速切换配置",
                preset_names,
                index=0,
                key="vlm_preset_selector",
                help="选择已保存的 API 配置"
            )
        
        with c_p2:
            # Management Popover (Save/Delete)
            with st.popover("💾 管理预设"):
                st.markdown("##### 保存当前配置")
                new_preset_name = st.text_input("预设名称", placeholder="如: DeepSeek", key="input_new_preset_name")
                
                if st.button("保存/覆盖", use_container_width=True, key="btn_save_preset"):
                    if new_preset_name:
                        # Grab current values from session state (synced below)
                        # We use the values bound to the input widgets later, stored in session state
                        preset_data = {
                            "api_url": st.session_state.get("dialog_vlm_url", st.session_state.vlm_api_url),
                            "api_key": st.session_state.get("dialog_vlm_key", st.session_state.vlm_api_key),
                            "model_name": st.session_state.get("dialog_vlm_model", st.session_state.vlm_model_name)
                        }
                        presets[new_preset_name] = preset_data
                        config.set("vlm.presets", presets)
                        config.save()
                        st.toast(f"预设 '{new_preset_name}' 已保存!", icon="💾")
                        st.rerun()
                    else:
                        st.error("请输入名称")
                
                st.divider()
                if selected_preset != "-- 选择预设 --":
                    if st.button(f"🗑️ 删除 '{selected_preset}'", type="primary", use_container_width=True, key="btn_del_preset"):
                        del presets[selected_preset]
                        config.set("vlm.presets", presets)
                        config.save()
                        st.toast(f"预设 '{selected_preset}' 已删除", icon="🗑️")
                        st.rerun()

        # Auto-load logic when preset changes
        # We check if selected preset corresponds to a known state change
        if selected_preset != "-- 选择预设 --" and selected_preset in presets:
            # Check if we just switched to this preset (Store last loaded in session to avoid infinite reload loop if user edits)
            # Actually simplest verification: If Selectbox value != Last Known Selectbox Value
            if st.session_state.get("last_loaded_preset_name") != selected_preset:
                p_data = presets[selected_preset]
                
                # Update backing state
                st.session_state.vlm_api_url = p_data.get("api_url", "")
                st.session_state.vlm_api_key = p_data.get("api_key", "")
                st.session_state.vlm_model_name = p_data.get("model_name", "")
                
                # Update Widget State Keys (Critical for UI update)
                st.session_state["dialog_vlm_url"] = st.session_state.vlm_api_url
                st.session_state["dialog_vlm_key"] = st.session_state.vlm_api_key
                st.session_state["dialog_vlm_model"] = st.session_state.vlm_model_name
                
                # Update last loaded tracker
                st.session_state.last_loaded_preset_name = selected_preset
                st.toast(f"已加载预设: {selected_preset}", icon="⚡")
                st.rerun()
        else:
             # Reset tracker if back to default
             if st.session_state.get("last_loaded_preset_name") and selected_preset == "-- 选择预设 --":
                 st.session_state.last_loaded_preset_name = None

        def _auto_save_vlm_url():
            val = st.session_state.get("dialog_vlm_url", "")
            st.session_state.vlm_api_url = val
            config.set("vlm.api_url", val)
            config.save()

        def _auto_save_vlm_key():
            val = st.session_state.get("dialog_vlm_key", "")
            st.session_state.vlm_api_key = val
            config.set("vlm.api_key", val)
            config.save()

        def _auto_save_vlm_model():
            val = st.session_state.get("dialog_vlm_model", "")
            st.session_state.vlm_model_name = val
            config.set("vlm.model_name", val)
            config.save()

        vlm_url = st.text_input(
            "API 地址", 
            value=st.session_state.vlm_api_url,
            key="dialog_vlm_url",
            on_change=_auto_save_vlm_url
        )
        vlm_key = st.text_input(
            "API 密钥", 
            value=st.session_state.vlm_api_key,
            type="password",
            key="dialog_vlm_key",
            on_change=_auto_save_vlm_key
        )
        vlm_model = st.text_input(
            "模型名称", 
            value=st.session_state.vlm_model_name,
            key="dialog_vlm_model",
            on_change=_auto_save_vlm_model
        )
        
        # Test VLM connection
        vlm_test_col1, vlm_test_col2 = st.columns([1, 1])
        with vlm_test_col1:
            if st.button("🔗 测试连接", key="test_vlm"):
                with st.spinner("测试中..."):
                    try:
                        import requests
                        headers = {"Authorization": f"Bearer {vlm_key}"}
                        test_url = vlm_url.rstrip("/") + "/models"
                        resp = requests.get(test_url, headers=headers, timeout=10)
                        if resp.status_code == 200:
                            st.session_state.vlm_test_result = "success"
                        else:
                            st.session_state.vlm_test_result = "fail"
                    except Exception as e:
                        st.session_state.vlm_test_result = "fail"
        with vlm_test_col2:
            if st.session_state.get("vlm_test_result") == "success":
                st.markdown('<span style="color: green; font-weight: bold;">✅ 连接成功</span>', unsafe_allow_html=True)
            elif st.session_state.get("vlm_test_result") == "fail":
                st.markdown('<span style="color: red; font-weight: bold;">❌ 连接失败</span>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Embedding 向量模型")
        
        # Provider selection (API vs Local)
        current_provider = st.session_state.get("embedding_provider", "api")
        provider_options = ["api", "local"]
        provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0
        
        emb_provider = st.radio(
            "模式选择",
            provider_options,
            index=provider_index,
            format_func=lambda x: "🌐 API 模式" if x == "api" else "💻 本地模式",
            horizontal=True,
            key="dialog_emb_provider",
            help="API模式使用远程Embedding API；本地模式使用本地transformers模型"
        )
        
        if emb_provider == "api":
            # API Mode settings
            emb_url = st.text_input(
                "API 地址", 
                value=st.session_state.embedding_api_url,
                key="dialog_emb_url"
            )
            emb_key = st.text_input(
                "API 密钥", 
                value=st.session_state.embedding_api_key,
                type="password",
                key="dialog_emb_key"
            )
            emb_model = st.text_input(
                "模型名称", 
                value=st.session_state.embedding_model_name,
                key="dialog_emb_model",
                help="本地服务(如LM Studio/Ollama)请加载专用embedding模型(如nomic-embed-text)并填写其ID"
            )
            emb_local_path = ""  # Not used in API mode
            
            # Test Embedding connection
            emb_test_col1, emb_test_col2 = st.columns([1, 1])
            with emb_test_col1:
                if st.button("🔗 测试连接", key="test_emb"):
                    with st.spinner("测试中..."):
                        try:
                            import requests
                            headers = {"Authorization": f"Bearer {emb_key}"}
                            test_url = emb_url.rstrip("/") + "/models"
                            resp = requests.get(test_url, headers=headers, timeout=10)
                            if resp.status_code == 200:
                                st.session_state.emb_test_result = "success"
                            else:
                                st.session_state.emb_test_result = "fail"
                        except Exception as e:
                            st.session_state.emb_test_result = "fail"
            with emb_test_col2:
                if st.session_state.get("emb_test_result") == "success":
                    st.markdown('<span style="color: green; font-weight: bold;">✅ 连接成功</span>', unsafe_allow_html=True)
                elif st.session_state.get("emb_test_result") == "fail":
                    st.markdown('<span style="color: red; font-weight: bold;">❌ 连接失败</span>', unsafe_allow_html=True)
        else:
            # Local Mode settings
            st.info("💡 本地模式使用 transformers 加载模型，支持图像+文本双模态 Embedding")
            emb_local_path = st.text_input(
                "本地模型路径",
                value=st.session_state.get("embedding_local_model_path", ""),
                key="dialog_emb_local_path",
                help="模型目录的绝对路径，如 D:/models/Qwen3-VL-Embedding"
            )
            # Keep API settings for potential fallback, but hide them
            emb_url = st.session_state.embedding_api_url
            emb_key = st.session_state.embedding_api_key
            emb_model = st.session_state.embedding_model_name
            
            st.warning("⚠️ 本地模式需要安装: `pip install transformers torch`")
    
    
    st.divider()
    st.subheader("⚡ 检测频率与灵敏度")
    
    # 1. Large Model Detection Frequency
    analysis_interval = st.slider(
        "大模型检测频率 (秒/次)", 
        min_value=0.5, 
        max_value=10.0, 
        value=st.session_state.analysis_interval,
        step=0.5,
        help="控制调用VLM大模型进行分析的频率。数值越小，反应越快，但消耗资源越多。"
    )

    # 2. Motion Detection Mode
    st.caption("🎯 运动检测模式")
    mode_options = {
        "standard": "标准模式 - 基础帧差检测（最快）",
        "enhanced": "增强模式 - 光照补偿 + 分块检测（推荐）",
        "intelligent": "智能模式 - 累积跟踪 + 趋势分析（最准确）"
    }
    
    current_mode = st.session_state.get("motion_detection_mode", "enhanced")
    mode_index = list(mode_options.keys()).index(current_mode) if current_mode in mode_options else 1
    
    motion_mode = st.radio(
        "检测模式",
        list(mode_options.keys()),
        index=mode_index,
        format_func=lambda x: mode_options[x],
        help="标准：快速但可能漏检\n增强：平衡性能和准确度\n智能：最高准确度，适合复杂场景"
    )
    
    # Update session state immediately for UI feedback
    if motion_mode != st.session_state.get("motion_detection_mode"):
        st.session_state.motion_detection_mode = motion_mode

    # 3. OpenCV Motion Threshold
    motion_threshold = st.slider(
        "OpenCV 运动检测阈值 (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state.get("motion_threshold", 5.0)),
        step=0.1,
        help="基于OpenCV的像素变化检测。只有超过此阈值才会触发大模型分析。0表示关闭检测（强制分析）。"
    )
    # Automatically update session state for immediate feedback/config saving
    if motion_threshold != st.session_state.get("motion_threshold"):
        st.session_state.motion_threshold = motion_threshold
        config.set("analysis.motion_threshold", motion_threshold)
    
    # 4. Heartbeat Interval
    motion_heartbeat = st.slider(
        "心跳检测间隔 (秒)",
        min_value=0.0,
        max_value=60.0,
        value=float(st.session_state.get("motion_heartbeat", 15.0)),
        step=5.0,
        help="即使画面静止，也会定期强制检测一次。防止漏检长时间静止的异常（如摔倒）。设置为 0 表示关闭心跳检测（纯运动触发）。"
    )
    
    if motion_heartbeat != st.session_state.get("motion_heartbeat"):
        st.session_state.motion_heartbeat = motion_heartbeat

    col_frames, col_space = st.columns([1, 1])
    with col_frames:
        st.caption("最大本地保存图片 (张)")
        saved_frames_options = [50, 300, 500, "Custom"]
        current_max_frames = st.session_state.get("max_saved_frames", 50)
        
        # Determine safe index
        if current_max_frames in [50, 300, 500]:
            sel_idx = saved_frames_options.index(current_max_frames)
        else:
            sel_idx = 3 # Custom
            
        saved_frames_choice = st.radio(
            "Max Saved Frames",
            saved_frames_options,
            index=sel_idx,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if saved_frames_choice == "Custom":
            max_saved_frames = st.number_input(
                "自定义数量", 
                min_value=10, 
            max_value=10000, 
                value=int(current_max_frames) if isinstance(current_max_frames, int) else 50
            )
        else:
            max_saved_frames = saved_frames_choice
    
    st.divider()

    # --- Object Detection Settings ---
    st.subheader("📦 报警目标检测 (YOLO)")
    st.caption("在报警触发时，自动标记画面中的特定目标。")
    
    obj_det_enabled = st.toggle("启用目标检测框 (Event Bounding Box)", value=st.session_state.get("obj_det_enabled", True))
    
    obj_det_options = ["person", "vehicle", "animal"]
    current_b_classes = st.session_state.get("obj_det_classes", ["person", "vehicle", "animal"])
    # Handle if stored as single string or list
    if isinstance(current_b_classes, str):
        current_b_classes = [current_b_classes]
        
    obj_det_classes = st.multiselect(
        "检测目标类型",
        obj_det_options,
        default=current_b_classes,
        format_func=lambda x: {"person": "👤 人员", "vehicle": "🚗 车辆", "animal": "🐕 动物 (猫/狗/鸟)"}.get(x, x),
        disabled=not obj_det_enabled
    )
    # Sync immediate state
    st.session_state.obj_det_enabled = obj_det_enabled
    st.session_state.obj_det_classes = obj_det_classes
    
    st.divider()
    
    # --- Notification Settings ---
    st.subheader("📱 手机通知推送")
    
    notif_enabled = st.checkbox("启用通知推送", value=st.session_state.get("notif_enabled", False))
    
    # initialize vars for saving even if disabled
    n_provider = st.session_state.get("notif_provider", "wechat")
    n_wx_appid = st.session_state.get("notif_wechat_appid", "")
    n_wx_secret = st.session_state.get("notif_wechat_secret", "")
    n_wx_openid = st.session_state.get("notif_wechat_openid", "")
    n_wx_template = st.session_state.get("notif_wechat_template", "")
    n_bark_key = st.session_state.get("notif_bark_key", "")
    n_cooldown = st.session_state.get("notif_cooldown", 60)
    
    if notif_enabled:
        col_n1, col_n2 = st.columns(2)
        with col_n1:
             n_provider = st.radio(
                "推送服务商", 
                ["wechat", "bark"], 
                index=0 if st.session_state.get("notif_provider") == "wechat" else 1,
                format_func=lambda x: "微信测试号 (Official Sandbox)" if x == "wechat" else "iOS (Bark)",
                horizontal=True
             )
        
        with col_n2:
            if n_provider == "wechat":
                st.info("💡 使用微信公众平台测试号 (无需认证)")
                st.markdown("[📋 点击获取测试号信息](https://mp.weixin.qq.com/debug/cgi-bin/sandboxinfo?action=showinfo&t=sandbox/index)")
                n_wx_appid = st.text_input("AppID", value=n_wx_appid)
                n_wx_secret = st.text_input("AppSecret", value=n_wx_secret, type="password")
                n_wx_openid = st.text_input("用户 OpenID", value=n_wx_openid, help="扫码关注测试号后获取")
                n_wx_template = st.text_input("模板 ID", value=n_wx_template, help="需新增模板，标题: AI报警，内容: {{title.DATA}} ...")
            else:
                st.info("💡 在 App Store 下载「Bark」，复制 APP 内的链接即可。")
                n_bark_key = st.text_input("Bark Key / URL后缀", value=n_bark_key, help="例如: https://api.day.app/YOUR_KEY/ 中 YOUR_KEY 的部分")

        # Cooldown Slider
        n_cooldown = st.slider(
            "⏳ 报警冷却时间 (秒)",
            min_value=10,
            max_value=300,
            value=int(st.session_state.get("notif_cooldown", 60)),
            help="同一种类型的报警在设定时间内只会发送一次，防止被轰炸。"
        )

        # Test Button
        if st.button("📨 发送测试通知"):
             # Temporarily configure notifier for test
             test_conf = {
                 "enabled": True,
                 "provider": n_provider,
                 "wechat_appid": n_wx_appid,
                 "wechat_secret": n_wx_secret,
                 "wechat_openid": n_wx_openid,
                 "wechat_template_id": n_wx_template,
                 "bark_key": n_bark_key,
                 "cooldown": n_cooldown
             }
             services.notifier.configure(test_conf)
             if services.notifier.test_connection(n_provider):
                 st.toast("✅ 测试通知发送成功！请检查手机。", icon="📱")
             else:
                 st.error("❌ 发送失败，请检查配置是否正确。")
    
    st.divider()
    
    # Action buttons
    col_save, col_close = st.columns(2)
    with col_save:
        if st.button("💾 保存全部设置", type="primary", key="btn_save_all_settings"):
            # Save all settings to session state
            st.session_state.vlm_api_url = vlm_url
            st.session_state.vlm_api_key = vlm_key
            st.session_state.vlm_api_key = vlm_key
            st.session_state.vlm_model_name = vlm_model
            # Since system prompt is in a popover with its own key, we read it from session state key
            # But the text_area writes to 'dialog_vlm_system_prompt', we need to sync it to 'vlm_system_prompt'
            if 'dialog_vlm_system_prompt' in st.session_state:
                st.session_state.vlm_system_prompt = st.session_state.dialog_vlm_system_prompt
            
            st.session_state.embedding_provider = emb_provider
            st.session_state.embedding_api_url = emb_url
            st.session_state.embedding_api_key = emb_key
            st.session_state.embedding_model_name = emb_model
            st.session_state.embedding_local_model_path = emb_local_path
            st.session_state.analysis_interval = analysis_interval
            st.session_state.max_saved_frames = max_saved_frames
            st.session_state.motion_detection_mode = motion_mode
            st.session_state.motion_heartbeat = motion_heartbeat
            # Removed time sync switch, defaulting to config or True
            # st.session_state.enable_time_sync = enable_time_sync 
            
            # Save to persistent config file
            config.set("vlm.api_url", vlm_url)
            config.set("vlm.api_key", vlm_key)
            config.set("vlm.model_name", vlm_model)
            if 'vlm_system_prompt' in st.session_state:
                config.set("vlm.system_prompt", st.session_state.vlm_system_prompt)
                
            config.set("embedding.provider", emb_provider)
            config.set("embedding.api_url", emb_url)
            config.set("embedding.api_key", emb_key)
            config.set("embedding.model_name", emb_model)
            config.set("embedding.local_model_path", emb_local_path)
            
            config.set("analysis.interval", analysis_interval)
            config.set("storage.max_saved_frames", max_saved_frames)
            config.set("analysis.motion_threshold", motion_threshold)
            config.set("analysis.motion_detection_mode", motion_mode)
            config.set("analysis.motion_detection_mode", motion_mode)
            config.set("analysis.motion_detection_mode", motion_mode)
            config.set("analysis.motion_heartbeat", motion_heartbeat)
            config.set("analysis.context_size", vlm_context_size)
            
            # Save Object Detection Settings
            st.session_state.obj_det_enabled = obj_det_enabled
            st.session_state.obj_det_classes = obj_det_classes
            config.set("analysis.obj_det_enabled", obj_det_enabled)
            config.set("analysis.obj_det_classes", obj_det_classes)
            
            # Removed ROI saving
            
            # Save Notification Settings
            st.session_state.notif_enabled = notif_enabled
            st.session_state.notif_provider = n_provider
            st.session_state.notif_wechat_appid = n_wx_appid
            st.session_state.notif_wechat_secret = n_wx_secret
            st.session_state.notif_wechat_openid = n_wx_openid
            st.session_state.notif_wechat_template = n_wx_template
            st.session_state.notif_bark_key = n_bark_key
            st.session_state.notif_cooldown = n_cooldown
            
            config.set("notification.enabled", notif_enabled)
            config.set("notification.provider", n_provider)
            config.set("notification.wechat_appid", n_wx_appid)
            config.set("notification.wechat_secret", n_wx_secret)
            config.set("notification.wechat_openid", n_wx_openid)
            config.set("notification.wechat_template_id", n_wx_template)
            config.set("notification.bark_key", n_bark_key)
            config.set("notification.cooldown", n_cooldown)
            
            config.save()
            
            # Update Notifier immediately
            services.notifier.configure({
                "enabled": notif_enabled,
                "provider": n_provider,
                "wechat_appid": n_wx_appid,
                "wechat_secret": n_wx_secret,
                "wechat_openid": n_wx_openid,
                "wechat_template_id": n_wx_template,
                "bark_key": n_bark_key,
                "cooldown": n_cooldown 
            })
            
            # Hot-reload settings into running service
            if services.ingestion and services.ingestion.is_running():
                # 更新运动检测阈值（立即生效）
                services.ingestion.motion_threshold = motion_threshold
                # 更新分析间隔（立即生效）
                services.ingestion.analysis_interval = analysis_interval
                # 更新运动检测模式
                services.ingestion.motion_detection_mode = motion_mode
                # 更新心跳间隔
                services.ingestion.motion_heartbeat = motion_heartbeat
                # 同时更新 motion detector 的最小间隔
                # Update Object Detection
                services.ingestion.object_detection_enabled = obj_det_enabled
                services.ingestion.object_detection_classes = obj_det_classes
                
                if hasattr(services.ingestion, '_capture_loop'):
                    logger.info(f"Hot-reload: mode={motion_mode}, threshold={motion_threshold}%, interval={analysis_interval}s, heartbeat={motion_heartbeat}s, obj_det={obj_det_enabled}")
                st.toast(f"✅ 设置即时生效: 模式={motion_mode}, 频率={analysis_interval}s, 阈值={motion_threshold}%, 心跳={motion_heartbeat}s, YOLO={obj_det_enabled}", icon="⚡")
            else:
                st.toast("✅ 设置已保存！(重启监控后生效)", icon="💾")
            
            # Close dialog
            
            # Close dialog
            st.session_state.show_settings_dialog = False
            st.rerun()

    with col_close:
        if st.button("❌ 关闭"):
            st.session_state.show_settings_dialog = False
            st.rerun()


# --- Alarm Settings Dialog ---
@st.dialog("🚨 报警类型配置", width="large")
def alarm_settings_dialog():
    """Configure alarm rules with dual mode support."""
    st.caption("配置报警规则和检测模式。")
    
    # Load config
    config = get_config()
    current_rules = config.get("alarms.rules", {})
    current_mode = config.get("alarms.mode", "keyword")
    
    # 初始化默认规则（如果为空）
    DEFAULT_RULES = {
        "PERSON": {"enabled": True, "description": "检测到人", "severity": "Low", "keywords": "人,行人,有人", "icon": "👤"},
        "DOG": {"enabled": True, "description": "检测到狗", "severity": "Medium", "keywords": "狗,犬,宠物狗", "icon": "🐕"},
        "CAR": {"enabled": False, "description": "检测到车辆", "severity": "Low", "keywords": "车,汽车,轿车,卡车", "icon": "🚗"},
        "FIRE": {"enabled": True, "description": "检测到火灾", "severity": "Critical", "keywords": "火,烟,火焰,着火", "icon": "🔥"},
        "FALL": {"enabled": True, "description": "检测到摔倒", "severity": "Critical", "keywords": "摔倒,倒地,躺", "icon": "⚠️"},
    }
    
    if not current_rules:
        current_rules = DEFAULT_RULES.copy()
    else:
        # 兼容性修复：如果现有规则缺少关键词（旧配置），自动填充默认关键词
        for key, def_rule in DEFAULT_RULES.items():
            if key in current_rules:
                if "keywords" not in current_rules[key] or not current_rules[key]["keywords"]:
                    current_rules[key]["keywords"] = def_rule["keywords"]
            # 同时也确保图标存在
            if key in current_rules and "icon" not in current_rules[key]:
                 current_rules[key]["icon"] = def_rule["icon"]
    
    if 'temp_rules' not in st.session_state:
        st.session_state.temp_rules = current_rules.copy()
    if 'temp_alert_mode' not in st.session_state:
        st.session_state.temp_alert_mode = current_mode
        
    rules = st.session_state.temp_rules
    
    # --- 报警模式选择 ---
    st.subheader("⚙️ 报警检测模式")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        mode_options = ["关键词匹配", "AI标签识别"]
        mode_index = 0 if st.session_state.temp_alert_mode == "keyword" else 1
        selected_mode = st.radio(
            "选择检测模式",
            mode_options,
            index=mode_index,
            horizontal=True,
            help="关键词匹配：AI只描述画面，系统根据关键词触发报警\nAI标签识别：要求AI输出 【ALERT: XXX】 标签"
        )
        st.session_state.temp_alert_mode = "keyword" if selected_mode == "关键词匹配" else "ai_tag"
    
    with mode_col2:
        if st.session_state.temp_alert_mode == "keyword":
            st.info("🏷️ **关键词模式**：AI 只需描述画面，系统根据描述文字中的关键词自动触发报警。更稳定。")
        else:
            st.info("🤖 **AI标签模式**：要求 AI 在检测到事件时输出特定标签（如 【ALERT: DOG_DETECTED】）。更精准但依赖AI遵守格式。")
    
    st.divider()
    
    # --- 现有规则列表 ---
    st.subheader("📋 报警规则")
    
    cols = st.columns(3)
    for idx, key in enumerate(sorted(rules.keys())):
        rule = rules[key]
        col_idx = idx % 3
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"### {rule.get('icon', '🔔')} {rule.get('description', key)}")
                
                # 严重程度
                sev = rule.get('severity', 'Info')
                color = "red" if sev == "Critical" else ("orange" if sev == "High" else "blue")
                st.markdown(f":{color}[{sev}]")
                
                # 关键词（仅关键词模式显示）
                if st.session_state.temp_alert_mode == "keyword":
                    kw = rule.get("keywords", "")
                    new_kw = st.text_input("关键词", value=kw, key=f"kw_{key}", help="逗号分隔，如：狗,犬")
                    rule["keywords"] = new_kw
                
                # 启用/禁用
                is_on = rule.get("enabled", True)
                btn_type = "primary" if is_on else "secondary"
                btn_text = "✅ 已启用" if is_on else "❌ 已禁用"
                
                if st.button(btn_text, key=f"toggle_{key}", type=btn_type):
                    rule["enabled"] = not is_on
                    st.rerun()
    
    st.divider()
    
    # --- 添加自定义规则 ---
    with st.expander("➕ 添加自定义报警", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            new_key = st.text_input("唯一标识 (ID)", placeholder="如: BIKE, HAT").upper()
            new_desc = st.text_input("显示名称", placeholder="检测到XXX")
        with c2:
            new_kw = st.text_input("关键词 (逗号分隔)", placeholder="关键词1,关键词2")
            new_sev = st.selectbox("严重程度", ["Info", "Low", "Medium", "High", "Critical"])
            
        if st.button("添加规则"):
            if new_key and new_desc:
                if new_key in rules:
                    st.error("规则ID已存在")
                else:
                    rules[new_key] = {
                        "enabled": True, 
                        "description": new_desc, 
                        "severity": new_sev,
                        "keywords": new_kw,
                        "icon": "🔔"
                    }
                    st.success(f"已添加 {new_key}")
                    st.rerun()
            else:
                st.warning("请填写ID和显示名称")

    st.divider()
    
    # --- Advanced Settings (Image Persistence Limit) ---
    st.markdown("#### 🖼️ 报警抓拍存储")
    current_limit = config.get("alarms.max_images", 1000)
    
    limit_options = [50, 300, 500, "自定义"]
    
    # Determine index based on current value
    try:
        if current_limit in limit_options:
            idx = limit_options.index(current_limit)
        else:
            idx = 3 # "自定义"
    except ValueError:
        idx = 3
        
    selected_limit_opt = st.selectbox(
        "最大报警图片数量 (超出自动删除旧图片)",
        limit_options,
        index=idx,
        help="建议设置为 500 左右，过多会占用磁盘空间"
    )
    
    if selected_limit_opt == "自定义":
        final_limit = st.number_input("请输入数量", min_value=10, max_value=10000, value=current_limit if isinstance(current_limit, int) else 1000)
    else:
        final_limit = selected_limit_opt
    
    st.divider()
    
    # Action buttons
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 保存生效", type="primary"):
            # Update config
            config.set("alarms.rules", rules)
            config.set("alarms.mode", st.session_state.temp_alert_mode)
            config.set("alarms.max_images", final_limit)
            config.save()
            
            st.session_state.show_alarm_dialog = False
            
            # Restart system to apply new prompt if running
            if st.session_state.get('monitoring', False):
                st.toast("正在重启监控以应用新规则...", icon="🔄")
                st.session_state.restart_required = True
                
            st.toast("报警规则已保存！", icon="✅")
            st.rerun()
            
    with col_cancel:
        if st.button("取消"):
            st.session_state.show_alarm_dialog = False
            if 'temp_rules' in st.session_state:
                del st.session_state.temp_rules
            st.rerun()

# --- Database Browser Dialog ---

@st.dialog("📂 数据库浏览器", width="large")
def database_browser_dialog():
    """Browse vector database records by date."""
    st.caption("查看历史分析记录 (按日期归档)")

    # Explicit Close Button to prevent zombie state
    if st.button("✖️ 关闭窗口", key="btn_close_db_browser_main"):
        st.session_state.show_db_browser = False
        st.rerun()
    
    # 1. First Pass: Get simplified list of all contents (lightweight)
    @st.cache_data(ttl=60)
    def fetch_date_stats():
        """Fetch all metadata to aggregate available dates and counts."""
        date_stats = {} # "YYYY-MM-DD": count
        try:
            from src.analysis import VectorStore
            vs = VectorStore()
            # Just get metadatas to prevent loading all text
            all_meta = vs.get_all_metadata()
            
            for item in all_meta:
                 meta = item.get("metadata", {})
                 ts_float = meta.get('timestamp', 0)
                 # fallback to capture_time string parsing if timestamp missing
                 if ts_float <= 0:
                     ts_str = meta.get('capture_time', '')
                     if ts_str:
                         try:
                             from datetime import datetime as dt_parse
                             ts_float = dt_parse.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                         except:
                             ts_float = 0
                 
                 if ts_float > 0:
                     dt_obj = datetime.fromtimestamp(ts_float)
                     date_key = dt_obj.strftime('%Y-%m-%d')
                     date_stats[date_key] = date_stats.get(date_key, 0) + 1
            
        except Exception as e:
            st.error(f"Error fetching stats: {e}")
        
        # Return sorted list of tuples
        return sorted(date_stats.items(), key=lambda x: x[0], reverse=True)

    # 2. Lazy load records for specific date
    def fetch_records_for_date(date_str):
        try:
            from src.analysis import VectorStore
            vs = VectorStore()
            # Use new optimized backend method
            if hasattr(vs, "get_documents_by_date"):
                return vs.get_documents_by_date(date_str)
            else:
                st.error("VectorStore missing get_documents_by_date method.")
                return [] 
        except Exception as e:
            st.error(f"Error fetching records for {date_str}: {e}")
            return []

    # UI Logic
    available_dates = fetch_date_stats()
    
    if not available_dates:
        st.info("数据库为空或暂无记录。")
        if st.button("关闭"):
            st.session_state.show_db_browser = False
            st.rerun()
        return

    # Date Selector
    # Format options: "YYYY-MM-DD (N items)"
    date_options = [d[0] for d in available_dates]
    date_labels = {d[0]: f"{d[0]} ({d[1]} 条记录)" for d in available_dates}
    
    # Grid layout for controls
    c_ctl1, c_ctl2 = st.columns([0.7, 0.3])
    
    with c_ctl1:
        # Use session state to persist selection
        if 'db_browser_selected_date' not in st.session_state:
            st.session_state.db_browser_selected_date = date_options[0] if date_options else None
            
        # Ensure selected is valid
        if st.session_state.db_browser_selected_date not in date_options:
             st.session_state.db_browser_selected_date = date_options[0] if date_options else None
             
        selected_date = st.selectbox(
            "选择日期", 
            date_options, 
            index=date_options.index(st.session_state.db_browser_selected_date) if st.session_state.db_browser_selected_date else 0,
            format_func=lambda x: date_labels.get(x, x),
            key="db_browser_date_select"
        )
        st.session_state.db_browser_selected_date = selected_date
    
    with c_ctl2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True) # Spacer
        if st.button("🗑️ 删除该日数据", type="primary", use_container_width=True):
            if selected_date:
                try:
                    from src.analysis import VectorStore
                    vs = VectorStore()
                    vs.delete_by_date(selected_date)
                    try:
                        from src.analysis.vector_db import VisualVectorStore
                        vvs = VisualVectorStore()
                        vvs.delete_by_date(selected_date)
                    except:
                        pass
                    st.toast(f"已删除 {selected_date}", icon="🗑️")
                    fetch_date_stats.clear() # Clear cache
                    # Reset selection
                    st.session_state.db_browser_selected_date = None 
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")

    st.divider()

    # Initialize View Mode
    if 'db_browser_view_mode' not in st.session_state:
        st.session_state.db_browser_view_mode = 'list'

    # --- LIST VIEW ---
    if st.session_state.db_browser_view_mode == 'list':
        if selected_date:
            records = fetch_records_for_date(selected_date)
            
            # Group by Hour
            grouped_by_hour = {}
            for r in records:
                ts = r['metadata'].get('timestamp', 0)
                if ts > 0:
                    h = datetime.fromtimestamp(ts).strftime('%H:00')
                else:
                    h = "Unknown"
                if h not in grouped_by_hour:
                    grouped_by_hour[h] = []
                grouped_by_hour[h].append(r)
                
            sorted_hours = sorted(grouped_by_hour.keys(), reverse=True)
            
            # Render Hours
            st.markdown(f"##### {selected_date} 详细记录")
            
            if not records:
                st.info("该日期无记录")
            
            for hour in sorted_hours:
                hour_records = grouped_by_hour[hour]
                # Sort by timestamp desc within hour
                hour_records.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
                
                with st.expander(f"🕒 {hour} ({len(hour_records)} 条)", expanded=False):
                    # Header Actions for Hour
                    c_h1, c_h2 = st.columns([0.8, 0.2])
                    with c_h2:
                        # Extract int hour from "HH:00"
                        try:
                            hour_int = int(hour.split(':')[0])
                            if st.button("🗑️ 删除该小时", key=f"del_hour_{selected_date}_{hour}", help="删除该小时内的所有记录"):
                                try:
                                    # Hot-reload check: if method missing, reload module
                                    from src.analysis import VectorStore
                                    vs = VectorStore()
                                    
                                    if not hasattr(vs, 'delete_by_hour'):
                                        import importlib
                                        import src.analysis.vector_db
                                        importlib.reload(src.analysis.vector_db)
                                        # Re-import after reload
                                        from src.analysis.vector_db import VectorStore
                                        vs = VectorStore()
                                    
                                    vs.delete_by_hour(selected_date, hour_int)
                                    
                                    # Also handle Visual Store
                                    try:
                                        from src.analysis.vector_db import VisualVectorStore
                                        vvs = VisualVectorStore()
                                        if not hasattr(vvs, 'delete_by_hour'):
                                             # Module already reloaded above, just re-instantiate
                                             vvs = VisualVectorStore()
                                        vvs.delete_by_hour(selected_date, hour_int)
                                    except Exception as ev:
                                        pass # Visual store might fail if not configured
                                        
                                    st.toast(f"已删除 {selected_date} {hour} 数据", icon="🗑️")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"删除失败: {e}. 请尝试重启程序。")
                        except:
                            pass

                    # Grid Layout: 6 columns
                    cols_per_row = 6
                    rows = [hour_records[i:i + cols_per_row] for i in range(0, len(hour_records), cols_per_row)]
                    
                    for row_items in rows:
                        cols = st.columns(cols_per_row)
                        for idx, rec in enumerate(row_items):
                            with cols[idx]:
                                meta = rec['metadata']
                                
                                # Time Label
                                t_str = meta.get('capture_time', '')
                                if 'T' in t_str:
                                    # Extract HH:MM:SS
                                    time_label = t_str.split('T')[1][:8]
                                else:
                                    time_label = "N/A"
                                
                                # Alert Icon
                                prefix = "🚨 " if meta.get('has_alert') else ""
                                
                                # Button
                                if st.button(f"{prefix}{time_label}", key=f"btn_v_{rec['id']}", help="点击查看详情"):
                                    st.session_state.selected_record = rec
                                    st.session_state.db_browser_view_mode = 'detail'
                                    st.rerun()

    # --- DETAIL VIEW ---
    elif st.session_state.db_browser_view_mode == 'detail':
        # Back Button
        if st.button("⬅️ 返回列表", type="secondary"):
            st.session_state.db_browser_view_mode = 'list'
            if 'selected_record' in st.session_state:
                del st.session_state.selected_record
            st.rerun()
            
        if 'selected_record' in st.session_state:
            sel = st.session_state.selected_record
            st.subheader("📝 记录详情")
            
            meta = sel['metadata']
            
            # Image Display (Large)
            img_path = meta.get('image_path')
            
            # Check if file exists to give better feedback
            if img_path and os.path.exists(img_path):
                safe_image_display(img_path, caption=f"原始画面 - {meta.get('camera_id', '')}", width='stretch')
            else:
                # Try to fix path if it's relative
                if img_path and not os.path.isabs(img_path):
                     # naive fix assuming ROOT_DIR
                     possible_path = str(ROOT_DIR / img_path)
                     if os.path.exists(possible_path):
                         safe_image_display(possible_path, caption=f"原始画面 (Rel) - {meta.get('camera_id', '')}", width='stretch')
                     else:
                         st.warning(f"图片文件未找到: {img_path}")
                else:
                     st.warning(f"图片路径失效或不存在: {img_path}")
            
            st.divider()
            
            # Info Columns
            info_c1, info_c2 = st.columns(2)
            with info_c1:
                st.caption("🔍 基本信息")
                st.write(f"**ID**: `{sel['id']}`")
                st.write(f"**时间**: `{meta.get('capture_time')}`")
                st.write(f"**摄像头**: `{meta.get('camera_id')}`")
                if meta.get('has_alert'):
                    st.error(f"🚨 触发报警: {meta.get('alert_info', 'Unknown')}")
            
            with info_c2:
                st.caption("📊 AI 分析描述")
                st.info(sel.get('description', '无描述内容'))

            with st.expander("查看完整元数据 (JSON)"):
                st.json(meta)
            
            st.divider()
            
            # --- Delete Section ---
            if st.button("🗑️ 删除此记录 (不可恢复)", type="primary", key="btn_delete_record_detail"):
                try:
                    record_id = sel['id']
                    
                    # 1. Delete from VectorStore (Text)
                    from src.analysis import VectorStore
                    vs = VectorStore()
                    vs.delete(record_id)
                    
                    # 2. Delete from VisualVectorStore (Semantic)
                    try:
                        from src.analysis.vector_db import VisualVectorStore
                        vvs = VisualVectorStore()
                        vvs.delete(record_id)
                    except Exception:
                        pass 
                    
                    st.toast(f"记录 {record_id} 已删除", icon="🗑️")
                    
                    # Clear cache and return
                    fetch_date_stats.clear()
                    st.session_state.db_browser_view_mode = 'list'
                    del st.session_state.selected_record
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"删除失败: {e}")



@st.dialog("🔍 搜索结果详情", width="large")
def search_detail_dialog():
    """Show details for a specific search result."""
    if 'selected_search_record' not in st.session_state:
        st.error("未选择记录")
        if st.button("关闭"):
            st.session_state.show_search_detail_dialog = False
            st.rerun()
        return

    sel = st.session_state.selected_search_record
    meta = sel['metadata']
    
    # Image Display
    img_path = meta.get('image_path')
    if img_path:
        safe_image_display(img_path, caption=f"相关性: {st.session_state.get('last_search_relevance', 'N/A')}", width='stretch')
    else:
        st.warning("图片路径缺失")
    
    st.divider()
    
    # Description
    st.markdown("**📝 场景描述:**")
    st.info(sel.get('description', '无描述'))
    
    st.divider()
    
    # Info Columns
    info_c1, info_c2 = st.columns(2)
    with info_c1:
        st.caption("基本信息")
        st.json({
            "ID": sel['id'],
            "时间戳": meta.get('timestamp'),
            "摄像头": meta.get('camera_id'),
            "报警类型": meta.get('alert_type', '无') if meta.get('has_alert') else '无'
        })
    
    with info_c2:
        st.caption("元数据")
        st.json(meta)

    st.divider()
    
    # Actions
    c_vid, c_del = st.columns([0.7, 0.3])
    with c_vid:
        if st.button("▶️ 跳转到视频位置", type="primary", key="btn_search_jump_video"):
             video_path = st.session_state.get("video_source_path", "")
             timestamp = meta.get('timestamp', 0)
             if video_path and Path(video_path).exists():
                 st.session_state.playback_video = video_path
                 st.session_state.playback_timestamp = timestamp
                 st.session_state.show_search_detail_dialog = False # Close dialog to see video
                 st.toast(f"已跳转到视频位置", icon="📍")
                 st.rerun()
             else:
                 st.warning("视频文件不存在")

    with c_del:
        if st.button("🗑️ 删除此记录", type="secondary", key="btn_search_del_record"):
            try:
                record_id = sel['id']
                from src.analysis import VectorStore
                vs = VectorStore()
                vs.delete(record_id)
                
                # Delete from VisualVectorStore if likely there
                try:
                    from src.analysis.vector_db import VisualVectorStore
                    vvs = VisualVectorStore()
                    vvs.delete(record_id)
                except Exception:
                    pass
                
                st.toast(f"记录已删除", icon="🗑️")
                st.session_state.show_search_detail_dialog = False
                st.rerun()
            except Exception as e:
                st.error(f"删除失败: {e}")

if st.session_state.get('show_settings_dialog', False):
    settings_dialog()
elif st.session_state.get('show_db_browser', False):
    database_browser_dialog()
elif st.session_state.get('show_search_detail_dialog', False):
    search_detail_dialog()
elif st.session_state.get('show_alarm_dialog', False):
    alarm_settings_dialog()

@st.dialog("📸 报警抓拍", width="large")
def show_alert_image_dialog(image_path, caption):
    """Show details and image of an alert."""
    if image_path:
        safe_image_display(image_path, caption=caption, width='stretch')
    else:
        st.warning("图片路径缺失")
    
    if st.button("关闭", key="btn_close_alert_img"):
        st.rerun()


# --- Main Layout ---

# Header
col_header, col_status = st.columns([0.6, 0.4])
with col_header:
    st.title("🛡️ AI-Sentinel 智能监控中心")

@st.fragment(run_every=2.0)
def render_system_stats(services):
    """Auto-refreshing system statistics"""
    if services.ingestion and services.ingestion.is_running():
        stats = services.ingestion.get_stats()
        worker_stats = services.worker.get_stats() if services.worker else {}
        
        # Display small metrics in a row
        c1, c2, c3 = st.columns(3)
        # Use Worker stats for true AI processing speed/count
        c1.metric("处理速度", f"{worker_stats.get('worker_fps', 0.0):.1f} 帧/秒")
        c2.metric("已分析帧数", f"{worker_stats.get('frames_processed', 0)}")
        c3.metric("报警数量", f"{worker_stats.get('alerts_detected', 0)}")
        
        # Motion Debug Stats
        motion_stats = services.ingestion.get_motion_stats()
        if motion_stats:
            mode = motion_stats.get('mode', 'N/A')
            grid = motion_stats.get('grid_size', '4x4 (Default)')
            with st.expander(f"⚙️ 运动检测状态 (Grid: {grid})", expanded=False):
                d1, d2 = st.columns(2)
                d1.caption(f"模式: {mode}")
                d1.caption(f"网格: {grid}")
                d2.caption(f"光照补偿: {motion_stats.get('illumination_compensations', 0)}")
                d2.caption(f"网格触发: {motion_stats.get('grid_triggers', 0)}")
    else:
        st.caption("🔴 系统未启动")

with col_status:
    render_system_stats(services)

# Layout: Video (Left) - 70%, Sidebar/Info (Right) - 30%
col_video, col_info = st.columns([0.7, 0.3], gap="medium")


# --- Right Column: Alerts & Search ---
with col_info:
    # 1. Alert Log
    # Header with Settings Button

    # Clear All Button in the same row/area or just below
    # Let's put it in a small column next to settings if possible, or just add logic.
    # The user asked for "Next to real-time alarm"
    # Current columns: [0.7, 0.3] -> Header, Settings Btn
    # We can change to [0.5, 0.25, 0.25]
    
    # Redefine columns for header to accommodate Clear button
    a_col1, a_col2, a_col3 = st.columns([0.6, 0.25, 0.15])
    with a_col1:
        st.subheader("🚨 实时报警")
    
    with a_col2:
        def clear_all_alerts():
            st.session_state.alerts = []
            st.toast("已清除所有报警通知", icon="🗑️")
            
        st.button("清空", help="清除所有报警记录", on_click=clear_all_alerts, type="secondary", key="btn_clear_alerts")
        
    with a_col3:
        def open_alarm_dialog():
            st.session_state.show_settings_dialog = False
            st.session_state.show_db_browser = False
            st.session_state.show_alarm_dialog = True
            
        st.button("⚙️", help="配置报警类型和规则", on_click=open_alarm_dialog, key="btn_open_alarm_settings")
            
    # Alert Container with fixed height
    alert_scroll_container = st.container(height=300)
    with alert_scroll_container:
        alert_placeholder = st.empty()


    
    # 2. Semantic Search
    st.divider()
    
    # Row for Search Title and DB Browser Button
    s_col1, s_col2 = st.columns([0.6, 0.4])
    with s_col1:
        st.subheader("🔍 智能搜索")
    with s_col2:
        def open_db_browser():
            # Close any other open dialogs first
            st.session_state.show_settings_dialog = False
            st.session_state.show_alarm_dialog = False
            st.session_state.show_db_browser = True
            
        st.button("📂 历史归档", help="查看历史记录归档（数据库浏览器）", on_click=open_db_browser, key="btn_open_db_browser", type="secondary")
    
    # 搜索设置区域
    with st.expander("⚙️ 搜索设置", expanded=False):
        col_mode, col_sort = st.columns(2)
        with col_mode:
            search_mode = st.selectbox(
                "搜索模式",
                ["文本搜索", "语义搜索"],
                help="文本搜索：基于VLM生成的中文描述匹配\n语义搜索：基于Embedding向量匹配（更精准）",
                key="select_search_mode"
            )
        with col_sort:
            # Change default sort order to Relevance First by placing it first in list
            sort_order = st.selectbox(
                "排序方式",
                ["相关性优先", "最新优先"],
                help="最新优先：按时间倒序排列\n相关性优先：按匹配度排序",
                key="select_sort_order"
            )
        
        col_num, col_threshold = st.columns(2)
        with col_num:
            n_results = st.slider("结果数量", 5, 30, 10, help="返回的最大结果数量", key="slider_n_results")
        with col_threshold:
            min_similarity = st.slider("最低相关性", 0.0, 1.0, 0.25, 0.05, 
                                      help="过滤掉相关性低于此阈值的结果，避免不相关内容", key="slider_min_similarity")
        
        # Level Filter
        level_filter = st.multiselect(
            "报警等级筛选",
            ["Critical (紧急)", "Warning (警告)", "Info (一般)"],
            default=[],
            help="只显示特定等级的事件。留空则显示所有。",
            key="multiselect_level_filter"
        )
        
        # Map UI selection to metadata values
        selected_severities = []
        if "Critical (紧急)" in level_filter: selected_severities.append("Critical")
        if "Warning (警告)" in level_filter: selected_severities.append("Warning")
        if "Info (一般)" in level_filter: selected_severities.append("Info")
        
        # Time Filter
        use_time_filter = st.checkbox("启用时间筛选", value=False)
        time_range = None
        
        if use_time_filter:
            # 预设时间范围快捷选项
            time_preset = st.selectbox(
                "快捷选择",
                ["自定义", "过去1小时", "过去6小时", "今天", "过去24小时", "过去7天"],
            )
            
            if time_preset == "自定义":
                tf_col1, tf_col2 = st.columns(2)
                with tf_col1:
                    start_date = st.date_input("开始日期", value=datetime.now())
                    start_time = st.time_input("开始时间", value=datetime.strptime("00:00:00", "%H:%M:%S").time())
                with tf_col2:
                    end_date = st.date_input("结束日期", value=datetime.now())
                    end_time = st.time_input("结束时间", value=datetime.strptime("23:59:59", "%H:%M:%S").time())
                    
                # Combine to timestamp
                try:
                    start_dt = datetime.combine(start_date, start_time)
                    end_dt = datetime.combine(end_date, end_time)
                    time_range = (start_dt.timestamp(), end_dt.timestamp())
                except Exception as e:
                    st.error(f"无效的时间范围: {e}")
            else:
                # 计算预设时间范围
                from datetime import timedelta
                now = datetime.now()
                if time_preset == "过去1小时":
                    start_dt = now - timedelta(hours=1)
                elif time_preset == "过去6小时":
                    start_dt = now - timedelta(hours=6)
                elif time_preset == "今天":
                    start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif time_preset == "过去24小时":
                    start_dt = now - timedelta(hours=24)
                elif time_preset == "过去7天":
                    start_dt = now - timedelta(days=7)
                else:
                    start_dt = now - timedelta(hours=1)
                time_range = (start_dt.timestamp(), now.timestamp())
                st.caption(f"📅 时间范围: {start_dt.strftime('%m-%d %H:%M')} 至 {now.strftime('%m-%d %H:%M')}")

    with st.form(key="search_form"):
        search_query = st.text_input("🔎 输入描述（如：穿红衣服的人、白色狗）", placeholder="输入搜索内容并回车...")
        search_btn = st.form_submit_button("🔍 搜索", type="primary", use_container_width=True)
    
    # Persist search state
    if search_btn:
        st.session_state.search_active = True
        st.session_state.last_search_query = search_query
    
    # Check if we should show results (Active AND query matches current input to avoid stale results if user types but doesn't search)
    # Actually, allow showing old results even if query changed, until new search? 
    # Better: Use the stored query for search logic.
    
    if st.session_state.get("search_active", False):
        # Use the query that was actually submitted
        active_query = st.session_state.get("last_search_query", "")
        
        if active_query:
            with st.spinner(f"🔍 正在搜索: {active_query}..."):
                if services.worker:
                    # Calculate limit for filtering - Request more candidates (10x) to allow better "Latest First" sorting
                    # This ensures recent events that are slightly less relevant still get retrieved
                    search_limit = int(n_results * 10)

                    # Determine sort mode for backend
                    sort_mode_param = "time" if sort_order == "最新优先" else "relevance"

                    # Choose search method based on mode
                    # Pass min_score=0.0 to get broad candidates, then filter in UI
                    if search_mode == "语义搜索":
                        results = services.worker.search_visual(active_query, n_results=search_limit, time_range=time_range, sort_mode=sort_mode_param, min_score=0.0)
                        if not results:
                            st.warning("语义搜索未返回结果或未启用，尝试切换到普通搜索")
                            results = services.worker.search_frames(active_query, n_results=search_limit, time_range=time_range, sort_mode=sort_mode_param, min_score=0.0)
                    else:
                        results = services.worker.search_frames(active_query, n_results=search_limit, time_range=time_range, sort_mode=sort_mode_param, min_score=0.0)
                    
                    # Filter by Severity Level
                    if selected_severities and results:
                        filtered = []
                        for r in results:
                            meta = r.get("metadata", {})
                            sev = meta.get("severity")
                            if sev in selected_severities:
                                filtered.append(r)
                        results = filtered
                    
                    if results:
                        # Helper function to calculate relevance score
                        def get_relevance(r):
                            if 'similarity' in r and r['similarity'] is not None:
                                return r['similarity']  # Higher is better
                            elif 'distance' in r and r['distance'] is not None:
                                return max(0, 1 - r['distance'] / 2)  # Convert distance to similarity
                            return 0
                        
                        # 过滤低相关性结果
                        filtered_results = [r for r in results if get_relevance(r) >= min_similarity]
                    
                        # 根据排序方式排序
                        if sort_order == "最新优先":
                            # 严格按时间倒序
                            def get_time_sort_key(r):
                                try:
                                    return float(r['metadata'].get('timestamp', 0))
                                except:
                                    return 0.0
                                    
                            sorted_results = sorted(filtered_results, 
                                                    key=get_time_sort_key, 
                                                    reverse=True)
                        else:
                            sorted_results = sorted(filtered_results, key=get_relevance, reverse=True)
                        
                        # 重要：最后必须截断到用户请求的数量
                        # (Previously this was missing if severity filter was skipped)
                        sorted_results = sorted_results[:n_results]
                        
                        # 显示统计信息
                        filtered_count = len(results) - len(filtered_results)
                        if filtered_count > 0:
                            st.success(f"找到 {len(sorted_results)} 个结果 (已智能排序 & 过滤 {filtered_count} 个低相关性结果)")
                        else:
                            st.success(f"找到 {len(sorted_results)} 个结果")
                    
                        for idx, r in enumerate(sorted_results):
                            timestamp = r['metadata'].get('timestamp', 0)
                            
                            # Format timestamp: if > 86400 (1 day in seconds), treat as unix timestamp
                            if timestamp > 86400:
                                time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # Treat as video position in seconds
                                mins, secs = divmod(int(timestamp), 60)
                                time_str = f"{mins:02d}:{secs:02d}"
                            
                            # Calculate relevance score for display
                            # Calculate relevance score for display
                            if 'similarity' in r and r['similarity'] is not None:
                                relevance = r['similarity']
                                relevance_pct = f"{relevance * 100:.1f}%"
                            elif 'distance' in r and r['distance'] is not None:
                                # Convert distance to percentage (smaller = better, max around 2 for cosine)
                                relevance = max(0, 1 - r['distance'] / 2)
                                relevance_pct = f"{relevance * 100:.1f}%"
                            else:
                                relevance_pct = "N/A"
                            
                            # Create clean description preview
                            full_desc = r.get('description', '') or ''
                            desc_preview = full_desc[:50].replace('\n', ' ') if full_desc else f"帧 #{idx+1}"
                            
                            # Expander with rank, time, and preview
                            with st.expander(f"#{idx+1} 🎯 {relevance_pct} | ⏱️ {time_str} | {desc_preview}..."):
                                # Show full description in a styled box
                                if full_desc:
                                    st.markdown(f"**📝 场景描述:**")
                                    st.info(full_desc)
                                
                                # Show frame image if exists
                                if 'image_path' in r['metadata']:
                                    img_path = r['metadata']['image_path']
                                    safe_image_display(img_path, caption="帧图像")
                                
                                # Show metadata in columns
                                meta_col1, meta_col2 = st.columns(2)
                                with meta_col1:
                                    camera_id = r['metadata'].get('camera_id', 'unknown')
                                    st.caption(f"📹 摄像头: {camera_id}")
                                with meta_col2:
                                    st.caption(f"🕒 时间戳: {time_str}")
                                
                                # Button to open video at timestamp
                                btn_col1, btn_col2 = st.columns(2)
                                with btn_col1:
                                    if st.button(f"▶️ 播放视频", key=f"play_video_{idx}"):
                                        video_path = st.session_state.get("video_source_path", "")
                                        if video_path and Path(video_path).exists():
                                            st.session_state.playback_video = video_path
                                            st.session_state.playback_timestamp = timestamp
                                            st.toast(f"已标记视频位置: {time_str}", icon="📍")
                                        else:
                                            st.warning("视频文件不存在")
                                
                                with btn_col2:
                                     if st.button(f"🔍 查看详情", key=f"search_view_detail_{idx}"):
                                         st.session_state.selected_search_record = r
                                         # Save relevance for display
                                         st.session_state.last_search_relevance = relevance_pct
                                         st.session_state.show_search_detail_dialog = True
                                         st.rerun()
                        
                        if not sorted_results:
                            st.info("所有结果相关性过低，请尝试调整「最低相关性」阈值或修改搜索词")
                    else:
                        st.info("未找到匹配的结果")
                else:
                    # Even without worker running, try to search existing DB
                    try:
                        from src.analysis import VectorStore
                        vs = VectorStore()
                        if vs.count() > 0:
                            results = vs.search(active_query, n_results=n_results, time_range=time_range)
                            if results:
                                # 过滤低相关性结果
                                def get_relevance_offline(r):
                                    if r.get('distance') is not None:
                                        return max(0, 1 - r['distance'] / 2)
                                    return 0
                                
                                filtered_results = [r for r in results if get_relevance_offline(r) >= min_similarity]
                                
                                if filtered_results:
                                    # 根据排序方式排序
                                    if sort_order == "最新优先":
                                        def get_ts_offline(r):
                                            try:
                                                return float(r['metadata'].get('timestamp', 0))
                                            except:
                                                return 0.0
                                        sorted_results = sorted(filtered_results, key=get_ts_offline, reverse=True)
                                    else:
                                        sorted_results = sorted(filtered_results, key=get_relevance_offline, reverse=True)
                                    
                                    st.success(f"找到 {len(sorted_results)} 个历史结果 (离线搜索)")
                                    for idx, r in enumerate(sorted_results):
                                        timestamp = r['metadata'].get('timestamp', 0)
                                        
                                        # Format timestamp consistently
                                        if timestamp > 86400:
                                            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                        else:
                                            mins, secs = divmod(int(timestamp), 60)
                                            time_str = f"{mins:02d}:{secs:02d}"
                                        
                                        # Calculate relevance
                                        if r.get('distance') is not None:
                                            relevance_pct = f"{max(0, (1 - r['distance'] / 2)) * 100:.1f}%"
                                        else:
                                            relevance_pct = "N/A"
                                        
                                        full_desc = r.get('description', '') or ''
                                        desc_preview = full_desc[:50].replace('\n', ' ') if full_desc else f"帧 #{idx+1}"
                                        
                                        with st.expander(f"#{idx+1} 🎯 {relevance_pct} | ⏱️ {time_str} | {desc_preview}..."):
                                            if full_desc:
                                                st.info(full_desc)
                                            if 'image_path' in r['metadata']:
                                                img_path = r['metadata']['image_path']
                                                safe_image_display(img_path)
                                else:
                                    st.info("未找到相关视频片段")
                            else:
                                st.info("暂无历史数据，请先分析视频")
                        else:
                            st.info("暂无历史数据，请先分析视频")
                    except Exception as e:
                        st.warning(f"搜索失败: {e}")


# --- Left Column: Video Monitoring ---

# --- Left Column: Video Monitoring ---

# 使用优化的视频组件 (Required)
from src.webui.components.video_section import render_video_section_optimized, handle_pending_alert_dialog

# Handle pending alert image dialog OUTSIDE of fragment to avoid conflicts
handle_pending_alert_dialog()

# 使用优化的视频区块 (前后端解耦)
render_video_section_optimized(services, col_video, alert_placeholder)

