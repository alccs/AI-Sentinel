"""
VLM Analyzer - Qwen2-VL Integration for Frame Analysis
Supports both Transformers (local GPU) and Ollama (API) backends.
"""
import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)

# System prompt for safety-aware scene description
DEFAULT_SYSTEM_PROMPT = """你是一个专业的安防监控分析AI。请你对画面进行分级并描述。
背景信息：这是一个家庭院落监控。画面左侧是院子和铁门，右侧是一条乡村小路，画面上方（远处）有一条公路和鸡笼。
注意：请确保输出为单行，不要包含换行符（\n）。
请按照以下格式返回结果：[L{level}_{type}] {description}。

我们将采用 4 级分级标准:
[L3_CRITICAL]: 紧急危险 (有人摔倒、火灾、暴力行为、非法闯入、烟雾)
[L2_WARNING]: 警告关注 (有人在院子里、狗或猫（描述其颜色）、车移动)
[L1_INFO]: 画面静止 (无人员或动物活动，画面静止)

务必在描述末尾加上监控时间，格式：监控时间：YYYY-MM-DD HH:MM:SS"""


# DEFAULT_SYSTEM_PROMPT = """你是一个专业的安防监控分析AI。请你对画面进行分级并描述。
# 注意：请确保输出为单行，不要包含换行符（\n）。
# 请按照以下格式返回结果：[L{level}_{type}] {description}。

# 我们将采用 4 级分级标准:
# [L3_CRITICAL]: 紧急危险 (有人摔倒、火灾、暴力行为、非法闯入)
# [L2_WARNING]: 警告关注 (有人在院子里、狗或猫、车移动)
# [L1_INFO]: 画面静止 (无人员或动物活动，画面静止)

# 务必在描述末尾加上监控时间，格式：监控时间：YYYY-MM-DD HH:MM:SS"""


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def analyze(self, image_path: str, prompt: str, system_prompt: str, history: List[str] = None) -> str:
        """Analyze an image and return description."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class TransformersBackend(VLMBackend):
    """Qwen2-VL via Transformers (local GPU inference)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            logger.info(f"Loading Qwen2-VL model: {self.model_name}...")
            
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if self.device != "cuda":
                self._model = self._model.to(self.device)
            
            self._loaded = True
            logger.info("Qwen2-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            raise
    
    def analyze(self, image_path: str, prompt: str, system_prompt: str, history: List[str] = None) -> str:
        self._load_model()
        
        from qwen_vl_utils import process_vision_info
        
        # Format prompt with history if available
        final_prompt = prompt
        if history and len(history) > 0:
            context_str = "\n".join([f"- {h}" for h in history])
            final_prompt = f"Previous Context:\n{context_str}\n\nCurrent Request: {prompt}"
        
        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # Prepare inputs
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        
        # Generate
        import torch
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=256)
        
        # Decode
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def is_available(self) -> bool:
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration
            return torch.cuda.is_available()
        except ImportError:
            return False


class OllamaBackend(VLMBackend):
    """Qwen2-VL via Ollama API (local or remote server)."""
    
    def __init__(self, model_name: str = "qwen2-vl:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
    
    def analyze(self, image_path: str, prompt: str, system_prompt: str) -> str:
        import requests
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Build request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "model": self.model_name,
            "prompt": final_prompt,
            "system": system_prompt,
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def is_available(self) -> bool:
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class OpenAICompatibleBackend(VLMBackend):
    """Generic OpenAI-compatible API backend (e.g. vLLM, DeepSeek, OpenAI)."""
    
    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
    def analyze(self, image_path: str, prompt: str, system_prompt: str, history: List[str] = None) -> str:
        import requests
        import time as _time
        
        # Format prompt with history if available
        final_prompt = prompt
        if history and len(history) > 0:
            context_str = "\n".join([f"- {h}" for h in history])
            final_prompt = f"Previous Context:\n{context_str}\n\nCurrent Request: {prompt}"
        
        # Read, resize, and encode image to prevent payload size errors with smaller models
        from PIL import Image
        import io
        
        try:
            with Image.open(image_path) as img:
                # Resize if image is too large (max 1920px on longest side to save tokens/payload size)
                max_size = 1280
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to RGB (in case of PNG with alpha)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return f"[API Error: Image Processing Failed - {str(e)}]"
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        # Retry with exponential backoff for transient errors
        # LM Studio can only process one request at a time;
        # concurrent requests return 400/Channel Error
        max_retries = 3
        base_delay = 2.0  # seconds
        
        url = f"{self.base_url}/chat/completions"
        if "chat/completions" in self.base_url:
            url = self.base_url

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response is not None else 0
                
                # Retry on 400 (LM Studio busy/Channel Error), 429 (rate limit), 5xx (server error)
                if status_code in (400, 429, 502, 503) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                    logger.warning(
                        f"VLM API {status_code} error (model={self.model_name}, attempt {attempt+1}/{max_retries}), "
                        f"retrying in {delay:.0f}s..."
                    )
                    _time.sleep(delay)
                    continue
                else:
                    logger.error(f"VLM API error (model={self.model_name}): {e}")
                    return f"[API Error: {str(e)}]"
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"VLM API timeout (attempt {attempt+1}/{max_retries}), retrying in {delay:.0f}s...")
                    _time.sleep(delay)
                    continue
                else:
                    logger.error(f"VLM API timeout after {max_retries} attempts")
                    return f"[API Error: 请求超时]"
            except Exception as e:
                logger.error(f"VLM API unexpected error (model={self.model_name}): {e}")
                return f"[API Error: {str(e)}]"
        
        return f"[API Error: {str(last_error)}]"

    def is_available(self) -> bool:
        return True


class MockBackend(VLMBackend):
    """Mock backend for testing without GPU/Ollama."""
    
    def __init__(self):
        self._call_count = 0
    
    def analyze(self, image_path: str, prompt: str, system_prompt: str, history: List[str] = None) -> str:
        self._call_count += 1
        
        # Simulate occasional alerts for testing
        if self._call_count % 10 == 0:
            return "【ALERT: FALL_DETECTED】画面中有一人倒在地上，周围无其他人员。"
        elif self._call_count % 15 == 0:
            return "【ALERT: FIRE_DETECTED】画面右侧出现明火和浓烟。"
        else:
            return f"画面正常。办公区域，有{self._call_count % 5}人正在工作，无异常情况。"
    
    def is_available(self) -> bool:
        return True


class VLMAnalyzer:
    """
    High-level VLM Analyzer with automatic backend selection.
    """
    
    def __init__(
        self,
        backend: str = "auto",  # "auto", "transformers", "ollama", "mock", "openai"
        model_name: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        ollama_url: str = "http://localhost:11434",
        # Custom API args
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        # Store config for potential delayed init
        self.config = {
            "model_name": model_name,
            "ollama_url": ollama_url,
            "api_base_url": api_base_url,
            "api_key": api_key,
        }
        self._backend: VLMBackend = self._init_backend(backend, **self.config)
        import threading
        self._lock = threading.Lock()
        logger.info(f"VLMAnalyzer initialized with backend: {type(self._backend).__name__}")
    
    def _init_backend(self, backend: str, **kwargs) -> VLMBackend:
        """Initialize the appropriate backend."""
        
        if backend == "mock":
            return MockBackend()
            
        if backend == "openai" or backend == "custom":
            return OpenAICompatibleBackend(
                model_name=kwargs.get("model_name") or "gpt-4o",
                base_url=kwargs.get("api_base_url") or "https://api.openai.com/v1",
                api_key=kwargs.get("api_key") or "EMPTY"
            )
        
        if backend == "transformers":
            return TransformersBackend(kwargs.get("model_name") or "Qwen/Qwen2-VL-7B-Instruct")
        
        if backend == "ollama":
            return OllamaBackend(kwargs.get("model_name") or "qwen2-vl:7b", kwargs.get("ollama_url"))
        
        # Auto-detect
        if backend == "auto":
            # Try Ollama first
            ollama_model = kwargs.get("model_name") or "qwen2-vl:7b"
            ollama_url = kwargs.get("ollama_url") or "http://localhost:11434"
            ollama = OllamaBackend(ollama_model, ollama_url)
            if ollama.is_available():
                logger.info("Auto-selected Ollama backend")
                return ollama
            
            # Try Transformers
            tf_model = kwargs.get("model_name") or "Qwen/Qwen2-VL-7B-Instruct"
            tf = TransformersBackend(tf_model)
            if tf.is_available():
                logger.info("Auto-selected Transformers backend")
                return tf
            
            # Fallback to mock
            logger.warning("No VLM backend available, using MockBackend for testing")
            return MockBackend()
        
        raise ValueError(f"Unknown backend: {backend}")
    
    def analyze_frame(self, image_path: str, custom_prompt: str = None, history: List[str] = None) -> str:
        """
        Analyze a single frame and return description.
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom prompt (uses default if None)
            history: Optional list of previous analysis results strings
            
        Returns:
            str: Description of the frame, possibly with ALERT tags
        """
        prompt = custom_prompt or "请分析这张监控画面。"
        
        try:
            with self._lock:
                result = self._backend.analyze(image_path, prompt, self.system_prompt, history)
            logger.debug(f"VLM analysis result: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return f"[分析失败: {str(e)}]"
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend."""
        return type(self._backend).__name__
