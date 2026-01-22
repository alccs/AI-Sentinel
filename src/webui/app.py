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
import queue
from PIL import Image
from loguru import logger

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion import VideoIngestionService
from src.analysis import AnalysisWorker, VLMAnalyzer, VectorStore
from src.common.queue_manager import queue_manager
from src.common.types import Alert
from src.common.config import get_config

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
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ff4b4b;
        background-color: #ff4b4b26;
    }
    .alert-title {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Core Service Initialization ---

@st.cache_resource
def init_system_singletons():
    """
    Initialize background services as singletons.
    Uses st.cache_resource to ensure only one instance exists across sessions/refreshes.
    """
    from src.analysis import get_embedder, BaseEmbedder
    from src.analysis.vector_db import VisualVectorStore
    
    class SingletonServiceManager:
        def __init__(self):
            self.ingestion: VideoIngestionService = None
            self.worker: AnalysisWorker = None
            self.visual_embedder: BaseEmbedder = None
            self.visual_store: VisualVectorStore = None
            self.lock = threading.Lock()
            
            # Pre-load config or resources if needed here
            logger.info("Initializing SingletonServiceManager")
        
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
                
                # 4. 初始化 VLM 分析器（用于时间同步和内容分析）
                vlm_api_url = st.session_state.get("vlm_api_url", "http://127.0.0.1:1234/v1")
                vlm_api_key = st.session_state.get("vlm_api_key", "")
                vlm_model_name = st.session_state.get("vlm_model_name", "qwen/qwen3-vl-4b")
                
                vlm_analyzer = None
                if vlm_api_url and vlm_api_url.strip():
                    vlm_analyzer = VLMAnalyzer(
                        backend="custom",
                        model_name=vlm_model_name,
                        api_base_url=vlm_api_url,
                        api_key=vlm_api_key or "EMPTY"
                    )
                else:
                    vlm_analyzer = VLMAnalyzer(backend='auto')
                
                # 5. Start Ingestion Service with VLM time sync
                logger.info(f"Starting ingestion with source: {input_source}")
                # Define deletion callback
                def deletion_cb(doc_id):
                    # Run in try-except to avoid crashing ingestion thread
                    try:
                        if self.worker:
                            if hasattr(self.worker, 'vector_store') and self.worker.vector_store:
                                self.worker.vector_store.delete(doc_id)
                            if hasattr(self.worker, 'visual_store') and self.worker.visual_store:
                                self.worker.visual_store.delete(doc_id)
                    except Exception as e:
                        logger.error(f"Deletion callback failed: {e}")

                if is_file:
                    self.ingestion = VideoIngestionService.from_file(
                        input_source, 
                        camera_id="cam_main",
                        analysis_interval=1.0,
                        loop_video=False,
                        max_saved_frames=max_frames,
                        roi_config=ocr_roi,  # 改名为 roi_config
                        vlm_client=vlm_analyzer,  # 传入 VLM 客户端
                        deletion_callback=deletion_cb
                    )
                else:
                    self.ingestion = VideoIngestionService.from_rtsp(
                        str(input_source),
                        camera_id="cam_main",
                        analysis_interval=1.0,
                        max_saved_frames=max_frames,
                        roi_config=ocr_roi,  # 改名为 roi_config
                        vlm_client=vlm_analyzer,  # 传入 VLM 客户端
                        deletion_callback=deletion_cb
                    )
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
                    batch_size=1
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
services = init_system_singletons()


# --- Session State Management ---

# Load persistent config
config = get_config()

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

# Initialize settings from saved config
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = True
    st.session_state.video_source_type = config.get("video_source.type", "file")
    st.session_state.video_source_path = config.get("video_source.path", str(ROOT_DIR / "data/videos/sample.mp4"))
    st.session_state.vlm_api_url = config.get("vlm.api_url", "http://127.0.0.1:1234/v1")
    st.session_state.vlm_api_key = config.get("vlm.api_key", "")
    st.session_state.vlm_model_name = config.get("vlm.model_name", "qwen/qwen3-vl-4b")
    # Embedding settings (dual mode)
    st.session_state.embedding_provider = config.get("embedding.provider", "api")
    st.session_state.embedding_api_url = config.get("embedding.api_url", "https://api.openai.com/v1")
    st.session_state.embedding_api_key = config.get("embedding.api_key", "")
    st.session_state.embedding_model_name = config.get("embedding.model_name", "text-embedding-3-small")
    st.session_state.embedding_local_model_path = config.get("embedding.local_model_path", "")
    st.session_state.analysis_interval = config.get("analysis.interval", 1.0)
    st.session_state.max_saved_frames = config.get("storage.max_saved_frames", 50)
    
    # 初始化 ROI 配置到 session_state，确保框选区域持久化
    saved_roi = config.get("ocr.roi", [0.65, 0.85, 0.35, 0.15])
    st.session_state.new_roi_ratios = saved_roi  # 将配置文件中的 ROI 设为当前选择


# --- Sidebar: Configuration ---

with st.sidebar:
    st.title("⚙️ 系统配置")
    
    # Input Source
    source_type_index = 0 if st.session_state.get("video_source_type", "file") == "file" else 1
    source_type = st.radio("视频源类型", ["本地文件", "RTSP / 摄像头"], index=source_type_index)
    
    if source_type == "本地文件":
        input_path = st.text_input("文件路径", value=st.session_state.get("video_source_path", str(ROOT_DIR / "data/videos/sample.mp4")))
    else:
        input_path = st.text_input("RTSP地址 / URL (或0)", value=st.session_state.get("video_source_path", "0"))
    
    # VLM Configuration
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
    with col_btn1:
        if st.button("▶️ 启动监控", type="primary"):
            try:
                is_file = (source_type == "本地文件")
                # Save video source to config
                st.session_state.video_source_type = "file" if is_file else "rtsp"
                st.session_state.video_source_path = input_path
                
                config.set("video_source.type", "file" if is_file else "rtsp")
                config.set("video_source.path", input_path)
                config.save()
                
                services.start_system(
                    input_path, 
                    is_file, 
                    enable_visual=enable_visual,
                    visual_model_path=None  # No longer needed
                )
                st.session_state.monitoring = True
                if enable_visual:
                    st.toast("监控系统已启动（含语义搜索）", icon="✅")
                else:
                    st.toast("监控系统已启动", icon="✅")
            except Exception as e:
                st.error(f"启动失败: {e}")
            
    with col_btn2:
        if st.button("⏹️ 停止监控"):
            services.stop_system()
            st.session_state.monitoring = False
            st.toast("监控系统已停止", icon="🛑")

    # Advanced Settings Button
    st.divider()
    if st.button("⚙️ 高级设置"):
        # Close any other open dialogs first
        st.session_state.show_db_browser = False
        st.session_state.show_settings_dialog = True

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
        st.subheader("🤖 VLM 视觉模型")
        vlm_url = st.text_input(
            "API 地址", 
            value=st.session_state.vlm_api_url,
            key="dialog_vlm_url"
        )
        vlm_key = st.text_input(
            "API 密钥", 
            value=st.session_state.vlm_api_key,
            type="password",
            key="dialog_vlm_key"
        )
        vlm_model = st.text_input(
            "模型名称", 
            value=st.session_state.vlm_model_name,
            key="dialog_vlm_model"
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
    st.subheader("⚡ 其他设置")
    
    analysis_interval = st.slider(
        "分析间隔 (秒)", 
        min_value=0.5, 
        max_value=10.0, 
        value=st.session_state.analysis_interval,
        step=0.5,
        help="每隔多少秒抽取一帧进行AI分析"
    )

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
    st.subheader("📐 区域设置 (VLM 时间识别)")
    
    # VLM ROI Configuration
    # Load current ROI from config or session
    current_roi = config.get("ocr.roi", [0.65, 0.85, 0.35, 0.15]) # x, y, w, h ratios
    
    roi_expander = st.expander("设置时间识别区域 (ROI)", expanded=False)
    with roi_expander:
        # Try to get frame from running ingestion OR use cached frame
        frame = None
        if services.ingestion:
            frame = services.ingestion.get_current_frame()
        
        # Check if we have a frame (from ingestion or cached)
        has_live_frame = frame is not None
        has_cached_frame = "ocr_setup_frame" in st.session_state
        
        if has_live_frame or has_cached_frame:
            if has_live_frame:
                # Store frame in session state for button callbacks
                st.session_state.ocr_current_frame = frame
                
                # Resize for display
                fh, fw = frame.shape[:2]
                canvas_width = 600
                scale = canvas_width / fw
                canvas_height = int(fh * scale)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized_pil = Image.fromarray(cv2.resize(frame_rgb, (canvas_width, canvas_height)))
                
                # Store canvas size for ratio calculations
                st.session_state.roi_canvas_size = (canvas_width, canvas_height)
            else:
                # Use cached frame dimensions
                canvas_size = st.session_state.get("roi_canvas_size", (600, 400))
                canvas_width, canvas_height = canvas_size
            
            st.caption("请框选包含时间戳的区域（双击方框内部可重置）：")
            
            # Use a session state variable to hold the frame for cropping
            # This prevents the cropper from resetting due to frame updates
            if has_live_frame:
                # Always show the refresh button
                refresh_clicked = st.button("📸 刷新/截取当前帧", key="refresh_ocr_frame_btn")
                
                # Auto-capture first frame or refresh if button clicked
                if "ocr_setup_frame" not in st.session_state or refresh_clicked:
                    st.session_state.ocr_setup_frame = frame_resized_pil
                    if refresh_clicked:
                        st.toast("✅ 已刷新画面", icon="📸")
            elif not has_cached_frame:
                st.warning("⚠️ 无可用帧，请先启动监控获取画面")
            
            # Check if we have a frame to crop
            if "ocr_setup_frame" in st.session_state:
                setup_img = st.session_state.ocr_setup_frame
                
                from streamlit_cropper import st_cropper
                
                # Prepare default box if valid (st_cropper expects left, top, width, height)
                # 优先使用用户的最新选择，然后是配置文件中的值
                roi_to_use = st.session_state.get("new_roi_ratios") or current_roi
                
                if roi_to_use:
                    ix = int(roi_to_use[0] * canvas_width)
                    iy = int(roi_to_use[1] * canvas_height)
                    iw = int(roi_to_use[2] * canvas_width)
                    ih = int(roi_to_use[3] * canvas_height)
                    default_coords = (ix, iy, iw, ih)
                else:
                    default_coords = (0, 0, canvas_width//2, canvas_height//5)
                
                # Render cropper
                # Set realtime_update=True so we see what we drag
                # Use a stable key. 
                cropped_box = st_cropper(
                    setup_img,
                    realtime_update=True,
                    box_color='#FF0000',
                    aspect_ratio=None,
                    return_type='box',
                    default_coords=default_coords,
                    key="roi_cropper_widget"
                )
                
                # Calculate new ratios and store in session state
                # Only update if we have a valid cropped_box with non-zero dimensions
                if cropped_box:
                    left = cropped_box['left']
                    top = cropped_box['top']
                    width = cropped_box['width']
                    height = cropped_box['height']
                    
                    if width > 0 and height > 0:
                        rx = left / canvas_width
                        ry = top / canvas_height
                        rw = width / canvas_width
                        rh = height / canvas_height
                        st.session_state.new_roi_ratios = [rx, ry, rw, rh]
                        # Store canvas dimensions for later use
                        st.session_state.roi_canvas_size = (canvas_width, canvas_height)
                    # Don't reset to None - keep the last valid value
            
            # Display current ROI selection (outside the if block to always show)
            current_selection = st.session_state.get("new_roi_ratios")
            if current_selection:
                st.caption(f"📍 当前选区: x={current_selection[0]:.3f}, y={current_selection[1]:.3f}, w={current_selection[2]:.3f}, h={current_selection[3]:.3f}")
            
            # Show any pending OCR feedback messages
            if "ocr_feedback" in st.session_state:
                feedback = st.session_state.ocr_feedback
                if feedback["type"] == "success":
                    st.success(feedback["message"])
                elif feedback["type"] == "warning":
                    st.warning(feedback["message"])
                elif feedback["type"] == "error":
                    st.error(feedback["message"])
                elif feedback["type"] == "info":
                    st.info(feedback["message"])
                # Show image if available
                if "image" in feedback and feedback["image"] is not None:
                    st.image(feedback["image"], caption="测试区域", channels="BGR", width=300)
                # Clear feedback after showing (only once)
                del st.session_state.ocr_feedback
            
            col_test, col_apply = st.columns(2)
            
            with col_test:
                if st.button("🤖 测试 VLM", key="test_vlm_btn"):
                    new_roi_ratios = st.session_state.get("new_roi_ratios")
                    current_frame = st.session_state.get("ocr_current_frame")
                    
                    if new_roi_ratios and current_frame is not None:
                        # Extract ROI from current frame
                        full_h, full_w = current_frame.shape[:2]
                        tx = int(full_w * new_roi_ratios[0])
                        ty = int(full_h * new_roi_ratios[1])
                        tw = int(full_w * new_roi_ratios[2])
                        th = int(full_h * new_roi_ratios[3])
                        
                        # Ensure coordinates are valid
                        tx = max(0, min(tx, full_w - 1))
                        ty = max(0, min(ty, full_h - 1))
                        tw = max(1, min(tw, full_w - tx))
                        th = max(1, min(th, full_h - ty))
                        
                        roi_crop = current_frame[ty:ty+th, tx:tx+tw]
                        
                        # 直接使用系统配置的 VLM API（与时间同步共用）
                        vlm_api_url = st.session_state.get("vlm_api_url", "http://127.0.0.1:1234/v1")
                        vlm_api_key = st.session_state.get("vlm_api_key", "")
                        vlm_model_name = st.session_state.get("vlm_model_name", "qwen/qwen3-vl-4b")
                        
                        try:
                            # 使用与时间同步相同的 VLM 配置
                            vlm_analyzer = VLMAnalyzer(
                                backend="custom",
                                model_name=vlm_model_name,
                                api_base_url=vlm_api_url,
                                api_key=vlm_api_key or "EMPTY"
                            )
                            st.info(f"🤖 使用配置的 VLM: {vlm_model_name} @ {vlm_api_url}")
                            
                        except Exception as e:
                            st.session_state.ocr_feedback = {
                                "type": "error",
                                "message": f"❌ VLM 初始化失败: {str(e)}\n💡 请检查 VLM 配置是否正确",
                                "image": None
                            }
                            vlm_analyzer = None
                        
                        if vlm_analyzer:
                            try:
                                with st.spinner("正在进行 VLM 时间识别..."):
                                    # 保存 ROI 区域为临时图片
                                    import tempfile
                                    import os
                                    
                                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                        temp_path = tmp_file.name
                                        cv2.imwrite(temp_path, roi_crop)
                                    
                                    try:
                                        # 使用专门的时间识别提示词
                                        prompt = "识别图片中的日期和时间，格式严格为 YYYY-MM-DD HH:MM:SS，只输出时间字符串，不要包含任何其他内容。"
                                        response = vlm_analyzer.analyze_frame(temp_path, prompt)
                                        
                                        # 解析返回的时间字符串
                                        import re
                                        time_patterns = [
                                            r'(\d{4})[-/.](\d{2})[-/.](\d{2})\s+(\d{2}):(\d{2}):(\d{2})',
                                            r'(\d{4})[-/.](\d{2})[-/.](\d{2})[\s]*(\d{2}):(\d{2}):(\d{2})',
                                            r'(\d{4})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})',
                                            r'(\d{4})[-/.](\d{2})[-/.](\d{2})[:\s](\d{2}):(\d{2}):(\d{2})',
                                        ]
                                        
                                        parsed_time = None
                                        for pattern in time_patterns:
                                            match = re.search(pattern, response)
                                            if match:
                                                year, month, day, hour, minute, second = match.groups()
                                                if (1900 <= int(year) <= 2100 and 
                                                    1 <= int(month) <= 12 and 
                                                    1 <= int(day) <= 31 and 
                                                    0 <= int(hour) <= 23 and 
                                                    0 <= int(minute) <= 59 and 
                                                    0 <= int(second) <= 59):
                                                    parsed_time = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                                                    break
                                        
                                        if parsed_time:
                                            st.session_state.ocr_feedback = {
                                                "type": "success",
                                                "message": f"✅ VLM 识别成功!\n🤖 原始回复: {response}\n🕒 解析时间: {parsed_time}",
                                                "image": roi_crop
                                            }
                                        elif response and response.strip():
                                            st.session_state.ocr_feedback = {
                                                "type": "warning",
                                                "message": f"⚠️ VLM 有回复但无法解析为时间格式\n🤖 回复内容: {response}\n💡 请确保区域包含完整清晰的时间戳",
                                                "image": roi_crop
                                            }
                                        else:
                                            st.session_state.ocr_feedback = {
                                                "type": "warning",
                                                "message": "⚠️ VLM 未返回有效内容\n💡 请调整区域位置或大小，确保时间戳清晰可见",
                                                "image": roi_crop
                                            }
                                    finally:
                                        # 清理临时文件
                                        try:
                                            os.unlink(temp_path)
                                        except:
                                            pass
                                            
                            except Exception as e:
                                st.session_state.ocr_feedback = {
                                    "type": "error",
                                    "message": f"❌ VLM 测试失败: {str(e)}",
                                    "image": None
                                }
                        else:
                            st.session_state.ocr_feedback = {
                                "type": "error",
                                "message": "❌ VLM 分析器初始化失败",
                                "image": None
                            }
                    elif not new_roi_ratios:
                        st.session_state.ocr_feedback = {
                            "type": "warning",
                            "message": "⚠️ 请先框选区域",
                            "image": None
                        }
                    else:
                        st.session_state.ocr_feedback = {
                            "type": "warning",
                            "message": "⚠️ 无法获取当前帧",
                            "image": None
                        }
                    st.rerun()

            with col_apply:
                if st.button("💾 保存区域配置", key="save_ocr_btn", type="primary"):
                    new_roi_ratios = st.session_state.get("new_roi_ratios")
                    if new_roi_ratios:
                        try:
                            config.set("ocr.roi", new_roi_ratios)
                            config.save()
                            
                            # 立即更新当前 ROI，避免下次需要重新框选
                            # 这样下次打开时就会使用新保存的值作为默认值
                            st.session_state.ocr_feedback = {
                                "type": "success",
                                "message": f"✅ 时间识别区域已保存并生效！\n📍 坐标: x={new_roi_ratios[0]:.3f}, y={new_roi_ratios[1]:.3f}, w={new_roi_ratios[2]:.3f}, h={new_roi_ratios[3]:.3f}\n💡 下次打开时将自动使用此区域",
                                "image": None
                            }
                        except Exception as e:
                            st.session_state.ocr_feedback = {
                                "type": "error",
                                "message": f"❌ 保存失败: {e}",
                                "image": None
                            }
                    else:
                        st.session_state.ocr_feedback = {
                            "type": "warning",
                            "message": "⚠️ 无有效区域，请先框选",
                            "image": None
                        }
                    st.rerun()
        else:
            # 独立帧捕获功能 - 支持未启动监控时配置时间识别区域
            st.info("💡 监控未启动。您可以独立捕获一帧用于配置时间识别区域：")
            
            # 显示当前配置的视频源
            current_source = config.get("video_source.path", "")
            source_type = config.get("video_source.type", "file")
            st.caption(f"📹 当前视频源: {current_source}")
            
            if current_source:
                if st.button("📸 捕获预览帧", key="capture_preview_frame_btn"):
                    try:
                        with st.spinner("正在连接视频源..."):
                            if source_type == "rtsp" or current_source.startswith("rtsp://"):
                                from src.ingestion.video_source import RTSPVideoSource
                                temp_source = RTSPVideoSource(current_source)
                            else:
                                from src.ingestion.video_source import FileVideoSource
                                temp_source = FileVideoSource(current_source)
                            
                            if temp_source.is_opened():
                                # 读取几帧让画面稳定
                                captured_frame = None
                                for _ in range(10):
                                    success, temp_frame = temp_source.read_frame()
                                    if success and temp_frame is not None:
                                        captured_frame = temp_frame
                                
                                temp_source.release()
                                
                                if captured_frame is not None:
                                    # 处理并保存到session
                                    fh, fw = captured_frame.shape[:2]
                                    canvas_width = 600
                                    scale = canvas_width / fw
                                    canvas_height = int(fh * scale)
                                    
                                    frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                                    frame_pil = Image.fromarray(cv2.resize(frame_rgb, (canvas_width, canvas_height)))
                                    
                                    st.session_state.ocr_setup_frame = frame_pil
                                    st.session_state.ocr_current_frame = captured_frame
                                    st.session_state.roi_canvas_size = (canvas_width, canvas_height)
                                    st.toast("✅ 预览帧捕获成功！", icon="📸")
                                    st.rerun()
                                else:
                                    st.error("❌ 无法读取视频帧，请检查视频源")
                            else:
                                st.error("❌ 无法连接视频源，请检查配置")
                                temp_source.release()
                    except Exception as e:
                        st.error(f"❌ 捕获失败: {e}")
            else:
                st.warning("⚠️ 请先在侧边栏配置视频源")

    st.divider()
    
    # Action buttons
    col_save, col_close = st.columns(2)
    with col_save:
        if st.button("💾 保存全部设置", type="primary"):
            # Save all settings to session state
            st.session_state.vlm_api_url = vlm_url
            st.session_state.vlm_api_key = vlm_key
            st.session_state.vlm_model_name = vlm_model
            st.session_state.embedding_provider = emb_provider
            st.session_state.embedding_api_url = emb_url
            st.session_state.embedding_api_key = emb_key
            st.session_state.embedding_model_name = emb_model
            st.session_state.embedding_local_model_path = emb_local_path
            st.session_state.analysis_interval = analysis_interval
            st.session_state.max_saved_frames = max_saved_frames
            
            # Save to persistent config file
            config.set("vlm.api_url", vlm_url)
            config.set("vlm.api_key", vlm_key)
            config.set("vlm.model_name", vlm_model)
            config.set("embedding.provider", emb_provider)
            config.set("embedding.api_url", emb_url)
            config.set("embedding.api_key", emb_key)
            config.set("embedding.model_name", emb_model)
            config.set("embedding.local_model_path", emb_local_path)
            config.set("analysis.interval", analysis_interval)
            config.set("storage.max_saved_frames", max_saved_frames)
            config.save()
            
            st.toast("设置已保存到本地！", icon="✅")
            st.rerun()
    
    with col_close:
        if st.button("❌ 关闭"):
            st.session_state.show_settings_dialog = False
            st.rerun()

# --- Database Browser Dialog ---
@st.dialog("📂 数据库浏览器", width="large")
def database_browser_dialog():
    """Browse vector database records by date."""
    st.caption("查看历史分析记录 (按日期归档)")
    
    # Helper to load data
    @st.cache_data(ttl=60)
    def fetch_database_records():
        records = []
        try:
            # 1. Fetch from standard vector store
            from src.analysis import VectorStore
            vs = VectorStore()
            # Use lightweight metadata fetch if possible, else fetch recent
            # Assuming get_all_metadata is implemented
            if hasattr(vs, "get_all_metadata"):
                metas = vs.get_all_metadata()
                for m in metas:
                    records.append({
                        "id": m['id'],
                        "metadata": m['metadata'],
                        "source": "Text DB"
                    })
            else:
                 # Fallback
                 pass
                 
            # 2. Fetch from visual store if enabled
            # (Optional: merge logic here)
        except Exception as e:
            st.error(f"Error fetching records: {e}")
        return records

    records = fetch_database_records()
    
    if not records:
        st.info("数据库为空或暂无记录。")
        if st.button("关闭"):
            st.rerun()
        return

    # Group by Date (YYYY-MM-DD)
    grouped = {}
    for r in records:
        # Try to parse timestamp or capture_time
        ts_str = r['metadata'].get('capture_time', '')
        if not ts_str:
            # Try float timestamp
            ts_float = r['metadata'].get('timestamp', 0)
            if ts_float > 0:
                ts_str = datetime.fromtimestamp(ts_float).isoformat()
        
        # Extract date part
        date_key = "Unknown Date"
        if ts_str:
            try:
                date_key = ts_str.split('T')[0]
            except:
                pass
        
        if date_key not in grouped:
            grouped[date_key] = []
        grouped[date_key].append(r)
    
    # Sort dates desc
    sorted_dates = sorted(grouped.keys(), reverse=True)
    
    # UI: Date Expanders
    for date in sorted_dates:
        day_records = grouped[date]
        # Sort records within day by time descending
        day_records.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
        
        with st.expander(f"📅 {date} ({len(day_records)} 条记录)"):
            # List items with pagination or limit?
            # For now, list top 50 in expander to avoid lag
            for rec in day_records[:20]:
                meta = rec['metadata']
                ts = meta.get('timestamp', 0)
                time_label = datetime.fromtimestamp(ts).strftime('%H:%M:%S') if ts > 0 else "N/A"
                
                # Clickable row? Streamlit doesn't support easy clickable divs, use button
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"⏱️ **{time_label}**")
                
                if st.button("查看", key=f"btn_view_{rec['id']}"):
                    st.session_state.selected_record = rec
    
    st.divider()
    
    # Detail View Area
    if 'selected_record' in st.session_state:
        sel = st.session_state.selected_record
        st.subheader("📝 记录详情")
        
        meta = sel['metadata']
        
        # Image
        img_path = meta.get('image_path')
        if img_path and Path(img_path).exists():
            st.image(img_path, caption="原始画面")
        else:
            st.warning("图片文件不存在 (可能已被清理)")
            
        # Description is not in metadata_only fetch, might need to re-fetch or assume it wasn't returned
        # Detailed fetch
        if st.button("🔄 加载完整描述 & 详情"):
             # Logic to fetch full doc by ID
             from src.analysis import VectorStore
             vs = VectorStore()
             full_doc = vs.get_by_frame_id(meta.get('frame_id', ''), meta.get('camera_id', 'default'))
             if full_doc:
                 st.info(full_doc.get('description', '无描述'))
                 st.json(full_doc.get('metadata', {}))
        
    if st.button("关闭", key="close_db_dialog_btn"):
        st.session_state.show_db_browser = False
        st.rerun()

# Show dialog if triggered (keep state True while dialog is open)
# Only allow one dialog at a time
if st.session_state.get('show_settings_dialog', False):
    settings_dialog()
elif st.session_state.get('show_db_browser', False):
    database_browser_dialog()


# --- Main Layout ---

# Header
col_header, col_status = st.columns([0.6, 0.4])
with col_header:
    st.title("🛡️ AI-Sentinel 智能监控中心")

with col_status:
    if services.ingestion and services.ingestion.is_running():
        stats = services.ingestion.get_stats()
        worker_stats = services.worker.get_stats() if services.worker else {}
        
        # Display small metrics in a row
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS", f"{stats.get('analysis_fps', 0):.1f}")
        c2.metric("Frames", f"{stats.get('analysis_frames', 0)}")
        c3.metric("Alerts", f"{worker_stats.get('alerts_detected', 0)}")
    else:
        st.caption("🔴 系统未启动")

# Layout: Video (Left) - 70%, Sidebar/Info (Right) - 30%
col_video, col_info = st.columns([0.7, 0.3], gap="medium")


# --- Right Column: Alerts & Search ---
with col_info:
    # 1. Alert Log
    st.subheader("🚨 实时报警")
    alert_container = st.container(height=300)
    
    # 2. Semantic Search
    st.divider()
    
    # Row for Search Title and DB Browser Button
    s_col1, s_col2 = st.columns([0.6, 0.4])
    with s_col1:
        st.subheader("🔍 智能搜索")
    with s_col2:
        if st.button("📂 浏览数据库", help="查看历史记录归档"):
            # Close any other open dialogs first
            st.session_state.show_settings_dialog = False
            st.session_state.show_db_browser = True
            st.rerun()
    
    # Search mode toggle
    search_mode = st.radio(
        "搜索模式",
        ["📝 文本语义搜索 (ChromaDB)", "🔍 语义搜索 (Embedding API)"],
        horizontal=True,
        help="文本搜索基于VLM描述，语义搜索基于文本嵌入向量"
    )
    
    # Time Filter
    use_time_filter = st.checkbox("启用时间筛选", value=False)
    time_range = None
    
    if use_time_filter:
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
            st.error(f"Invalid time range: {e}")

    search_query = st.text_input("输入描述（如：穿红衣服的人）", placeholder="输入搜索内容并回车...")
    search_btn = st.button("🔍 搜索")
    
    if search_btn and search_query:
        if services.worker:
            # Choose search method based on mode
            if search_mode == "🔍 语义搜索 (Embedding API)":
                results = services.worker.search_visual(search_query, n_results=5, time_range=time_range)
                if not results:
                    st.warning("语义搜索未启用，请在启动时勾选「启用语义搜索」并配置 Embedding API")
                    results = services.worker.search_frames(search_query, n_results=5, time_range=time_range)
            else:
                results = services.worker.search_frames(search_query, n_results=5, time_range=time_range)
            
            if results:
                st.success(f"找到 {len(results)} 个结果")
                
                # Sort results by relevance (similarity desc or distance asc)
                def get_relevance(r):
                    if 'similarity' in r and r['similarity'] is not None:
                        return r['similarity']  # Higher is better
                    elif 'distance' in r and r['distance'] is not None:
                        return -r['distance']  # Lower distance is better (negate for desc sort)
                    return 0
                
                sorted_results = sorted(results, key=get_relevance, reverse=True)
                
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
                            if Path(img_path).exists():
                                st.image(img_path, caption=f"帧图像", width="stretch")
                            else:
                                st.caption("⚠️ 图片文件已被清理")
                        
                        # Show metadata in columns
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            camera_id = r['metadata'].get('camera_id', 'unknown')
                            st.caption(f"📹 摄像头: {camera_id}")
                        with meta_col2:
                            st.caption(f"🕒 时间戳: {time_str}")
                        
                        # Button to open video at timestamp
                        if st.button(f"▶️ 从此处播放视频", key=f"play_video_{idx}"):
                            video_path = st.session_state.get("video_source_path", "")
                            if video_path and Path(video_path).exists():
                                st.session_state.playback_video = video_path
                                st.session_state.playback_timestamp = timestamp
                                st.toast(f"已标记视频位置: {time_str}", icon="📍")
                            else:
                                st.warning("视频文件不存在")
            else:
                st.info("未找到相关视频片段")
        else:
            # Even without worker running, try to search existing DB
            try:
                from src.analysis import VectorStore
                vs = VectorStore()
                if vs.count() > 0:
                    results = vs.search(search_query, n_results=5, time_range=time_range)
                    if results:
                        st.success(f"找到 {len(results)} 个历史结果 (离线搜索)")
                        for idx, r in enumerate(results):
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
                                    if Path(img_path).exists():
                                        st.image(img_path, width="stretch")
                                    else:
                                        st.caption("⚠️ 图片文件已被清理")
                    else:
                        st.info("未找到相关视频片段")
                else:
                    st.info("暂无历史数据，请先分析视频")
            except Exception as e:
                st.warning(f"搜索失败: {e}")


# --- Left Column: Video Monitoring ---

with col_video:
    st.subheader("📷 实时画面")
    
    # Status bar above video
    status_cols = st.columns([0.7, 0.3])
    with status_cols[0]:
        st.caption("🔴 直播中")
    with status_cols[1]:
        ocr_time_placeholder = st.empty()
        
    video_placeholder = st.empty()
    
    # Pull any pending alerts from queue into session state
    alerts_updated = False
    while queue_manager.qsize("alert_queue") > 0:
        try:
            alert = queue_manager.get_nowait("alert_queue")
            if alert:
                st.session_state.alerts.insert(0, alert)
                alerts_updated = True
                print(f"📥 收到新警报: {alert.risk_type} - {alert.description[:50]}...")
                # Keep valid size
                if len(st.session_state.alerts) > 50:
                    st.session_state.alerts.pop()
        except queue.Empty:
            break

    # Render alerts in container
    with alert_container:
        if not st.session_state.alerts:
            st.caption("暂无异常报警")
        else:
            for alert in st.session_state.alerts[:10]:
                risk_color = "red" if alert.severity == "Critical" else "orange"
                st.markdown(
                    f"""
                    <div class="alert-box" style="border-color: {risk_color}; background-color: {risk_color}1A;">
                        <div class="alert-title">🚨 {alert.risk_type}</div>
                        <div style="font-size: 0.9em;">{alert.description[:100]}</div>
                        <div style="font-size: 0.8em; color: gray; margin-top: 5px;">
                            {alert.timestamp.strftime('%H:%M:%S')}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # Non-blocking video display with auto-refresh
    if st.session_state.monitoring:
        # Display current frame (single refresh, non-blocking)
        if services.ingestion:
            frame = services.ingestion.get_current_frame()
            if frame is not None and frame.size > 0:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Check if frame has changed to reduce unnecessary updates
                    frame_hash = hash(frame_rgb.tobytes())
                    if not hasattr(st.session_state, 'last_frame_hash') or st.session_state.last_frame_hash != frame_hash:
                        video_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                        st.session_state.last_frame_hash = frame_hash
                    
                    # Update time sync display (替换 OCR 时间显示)
                    if services.ingestion and services.ingestion.time_synchronizer:
                        sync_stats = services.ingestion.get_time_sync_stats()
                        last_time = sync_stats.get("last_parsed_time", "N/A")
                        if last_time and last_time != "N/A":
                            ocr_time_placeholder.markdown(f"🕒 **VLM时间**: {last_time}")
                        else:
                            ocr_time_placeholder.markdown(f"🕒 **VLM时间**: 同步中...")
                    else:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ocr_time_placeholder.markdown(f"🕒 **系统时间**: {current_time}")
                        
                except Exception as e:
                    print(f"Error drawing frame: {e}")
            else:
                video_placeholder.info("等待视频信号...")
        
        # Auto-refresh mechanism (non-blocking)
        # Only auto-refresh if no dialogs are open
        dialog_open = st.session_state.get('show_settings_dialog', False) or st.session_state.get('show_db_browser', False)
        if not dialog_open:
            time.sleep(0.5)  # Reduced refresh rate to 2 FPS to prevent flickering
            st.rerun()
    else:
        video_placeholder.info("监控已停止。请点击侧边栏「启动监控」按钮。")
