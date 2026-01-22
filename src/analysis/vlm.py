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
DEFAULT_SYSTEM_PROMPT = """你是一个专业的安防监控分析AI。请简要描述画面内容（50字以内）。
如果检测到以下紧急情况，请在描述开头加上对应标记：
- 有人摔倒或躺在地上：【ALERT: FALL_DETECTED】
- 出现明火或烟雾：【ALERT: FIRE_DETECTED】
- 有人打架或暴力行为：【ALERT: VIOLENCE_DETECTED】
- 有可疑人员闯入或徘徊：【ALERT: INTRUSION_DETECTED】
如果画面正常，直接描述即可，无需添加标记。"""


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def analyze(self, image_path: str, prompt: str, system_prompt: str) -> str:
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
    
    def analyze(self, image_path: str, prompt: str, system_prompt: str) -> str:
        self._load_model()
        
        from qwen_vl_utils import process_vision_info
        
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
            "prompt": prompt,
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
        
    def analyze(self, image_path: str, prompt: str, system_prompt: str) -> str:
        import requests
        
        # Read and encode image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            
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
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        try:
            # Try chat completions endpoint
            url = f"{self.base_url}/chat/completions"
            # Handle full URLs vs base URLs
            if "chat/completions" in self.base_url:
                url = self.base_url
                
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[API Error: {str(e)}]"

    def is_available(self) -> bool:
        return True


class MockBackend(VLMBackend):
    """Mock backend for testing without GPU/Ollama."""
    
    def __init__(self):
        self._call_count = 0
    
    def analyze(self, image_path: str, prompt: str, system_prompt: str) -> str:
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
    
    def analyze_frame(self, image_path: str, custom_prompt: str = None) -> str:
        """
        Analyze a single frame and return description.
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom prompt (uses default if None)
            
        Returns:
            str: Description of the frame, possibly with ALERT tags
        """
        prompt = custom_prompt or "请分析这张监控画面。"
        
        try:
            result = self._backend.analyze(image_path, prompt, self.system_prompt)
            logger.debug(f"VLM analysis result: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return f"[分析失败: {str(e)}]"
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend."""
        return type(self._backend).__name__
