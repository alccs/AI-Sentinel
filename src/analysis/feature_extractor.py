"""
Feature Extractor - Dual Mode Embedding Support
Supports both API mode (OpenAI-compatible) and Local mode (transformers-based).
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import time

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Base Embedder (Abstract)
# ============================================================================

class BaseEmbedder(ABC):
    """
    Abstract base class for embedding providers.
    Defines the common interface for both API and Local modes.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a text string into a vector representation.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        pass
    
    @abstractmethod
    def embed_image(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Embed an image into a vector representation.
        
        Args:
            image: OpenCV image (BGR numpy array)
            
        Returns:
            List[float]: Embedding vector, or None if not supported
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> Optional[int]:
        """Get the embedding dimension (None if not yet determined)."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection/model loading."""
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings. Default implementation calls embed_text for each.
        Subclasses can override for batch optimization.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]


# ============================================================================
# API Embedder (OpenAI-compatible)
# ============================================================================

class APIEmbedder(BaseEmbedder):
    """
    API-based Embedder using OpenAI-compatible API.
    Converts VLM-generated descriptions into embeddings for semantic search.
    
    Note: embed_image() returns None as API mode doesn't support direct image embedding.
    """
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "text-embedding-ada-002",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the APIEmbedder with API configuration.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API (OpenAI-compatible)
            model_name: Model name for embeddings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts on failure
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._client = None
        self._embedding_dim = None
        
        logger.info(f"APIEmbedder initialized (model: {model_name}, base_url: {base_url})")
    
    def _init_client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is not None:
            return
        
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            
            logger.info(f"OpenAI client initialized: {self.base_url}")
            
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def embed_image(self, image: np.ndarray) -> Optional[List[float]]:
        """
        API mode does not support direct image embedding.
        Returns None to signal fallback to text embedding.
        
        Args:
            image: OpenCV image (ignored)
            
        Returns:
            None (API doesn't support direct image embedding)
        """
        return None
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a text string into a vector representation using the API.
        
        Args:
            text: Text to embed (VLM description or search query)
                
        Returns:
            List[float]: Embedding vector
            
        Raises:
            Exception: If API call fails after all retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self._init_client()
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI-compatible embeddings API
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=text,
                )
                
                # Extract embedding
                embedding = response.data[0].embedding
                
                # Cache embedding dimension on first successful call
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)
                    logger.info(f"Embedding dimension detected: {self._embedding_dim}")
                
                logger.debug(f"Embedded text (len={len(text)}, dim={len(embedding)})")
                return embedding
                
            except Exception as e:
                last_error = e
                err_val = str(e)
                
                # Check for specific "Model is not embedding" error (common in LM Studio)
                if "Model is not embedding" in err_val:
                    logger.error(f"❌ Server rejected model '{self.model_name}' for embeddings. This model might be a chat model.")
                    logger.error("👉 Solution: Load a dedicated Text Embedding model (e.g., 'nomic-embed-text') in your local server.")
                    raise ValueError(f"Model '{self.model_name}' is not an embedding model.") from e
                
                logger.warning(f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        # All retries failed
        error_msg = f"Failed to get embedding after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings using the API.
        
        Args:
            texts: List of text strings to embed
                
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        self._init_client()
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Call API with batch
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                
                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]
                
                # Cache dimension
                if self._embedding_dim is None and embeddings:
                    self._embedding_dim = len(embeddings[0])
                    logger.info(f"Embedding dimension detected: {self._embedding_dim}")
                
                logger.debug(f"Embedded {len(texts)} texts")
                return embeddings
                
            except Exception as e:
                last_error = e
                logger.warning(f"Batch embedding failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
        
        error_msg = f"Failed to get batch embeddings after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    @property
    def embedding_dim(self) -> Optional[int]:
        """Get the embedding dimension (None if not yet determined)."""
        return self._embedding_dim
    
    def test_connection(self) -> bool:
        """
        Test the API connection with a simple embedding request.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.embed_text("test")
            logger.info("API connection test successful")
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False


# ============================================================================
# Local Embedder (Transformers-based)
# ============================================================================

class LocalEmbedder(BaseEmbedder):
    """
    Local Embedder using transformers library.
    Supports both text and image embedding using multimodal models like Qwen-VL.
    
    Note: Requires transformers and torch to be installed.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        """
        Initialize the LocalEmbedder with a local model.
        
        Args:
            model_path: Absolute path to the local model directory
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype for model ("auto", "float16", "bfloat16", "float32")
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        self._model = None
        self._processor = None
        self._embedding_dim = None
        self._torch = None
        self._cv2 = None
        
        logger.info(f"LocalEmbedder initialized (model_path: {model_path})")
    
    def _init_model(self):
        """Lazy initialize the transformers model."""
        if self._model is not None:
            return
        
        # Try to import required libraries
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError(
                "torch is required for Local mode. "
                "Install with: pip install torch"
            )
        
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for Local mode. "
                "Install with: pip install transformers"
            )
        
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.warning("cv2 not available, image embedding may not work")
        
        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Determine device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        logger.info(f"Loading local model from {self.model_path} (device={device}, dtype={dtype})")
        
        try:
            # Try loading as a generic embedding model first
            self._model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=device if device == "auto" else None,
                trust_remote_code=True,
            )
            
            if device != "auto":
                self._model = self._model.to(device)
            
            self._model.eval()
            
            # Try to load processor or tokenizer
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                logger.info("Loaded AutoProcessor")
            except Exception:
                try:
                    self._processor = AutoTokenizer.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                    )
                    logger.info("Loaded AutoTokenizer (no image support)")
                except Exception as e:
                    logger.warning(f"Could not load processor/tokenizer: {e}")
            
            logger.info(f"Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def embed_image(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Embed an image into a vector representation.
        
        Args:
            image: OpenCV image (BGR numpy array)
            
        Returns:
            List[float]: Embedding vector, or None if not supported
        """
        if image is None:
            return None
        
        try:
            self._init_model()
        except Exception as e:
            logger.error(f"Failed to initialize local model for image embedding: {e}")
            return None
        
        if self._processor is None:
            logger.warning("No processor available for image embedding")
            return None
        
        try:
            # Check if processor supports image input
            if not hasattr(self._processor, 'image_processor') and not hasattr(self._processor, '__call__'):
                logger.warning("Processor does not support image input")
                return None
            
            # Convert BGR to RGB
            if self._cv2 is not None and len(image.shape) == 3:
                image_rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            from PIL import Image
            if isinstance(image_rgb, np.ndarray):
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image_rgb
            
            # Process image
            with self._torch.no_grad():
                # Try multimodal processing
                try:
                    inputs = self._processor(
                        images=pil_image,
                        return_tensors="pt",
                    )
                except Exception:
                    # Fallback: try image-only processing
                    inputs = self._processor(
                        pil_image,
                        return_tensors="pt",
                    )
                
                # Move to device
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self._model(**inputs)
                
                # Extract embedding (try different output formats)
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling over sequence
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                elif hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output.squeeze()
                elif hasattr(outputs, 'image_embeds'):
                    embedding = outputs.image_embeds.squeeze()
                else:
                    # Just use the first output
                    embedding = outputs[0].mean(dim=1).squeeze()
                
                embedding_list = embedding.cpu().numpy().tolist()
                
                # Cache dimension
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding_list)
                    logger.info(f"Local embedding dimension detected: {self._embedding_dim}")
                
                logger.debug(f"Image embedded (dim={len(embedding_list)})")
                return embedding_list
                
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return None
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a text string into a vector representation.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self._init_model()
        
        if self._processor is None:
            raise RuntimeError("No processor/tokenizer available for text embedding")
        
        try:
            with self._torch.no_grad():
                # Tokenize text
                inputs = self._processor(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                
                # Move to device
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self._model(**inputs)
                
                # Extract embedding
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling over sequence
                    if 'attention_mask' in inputs:
                        mask = inputs['attention_mask'].unsqueeze(-1)
                        embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                    else:
                        embedding = outputs.last_hidden_state.mean(dim=1)
                    embedding = embedding.squeeze()
                elif hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output.squeeze()
                else:
                    embedding = outputs[0].mean(dim=1).squeeze()
                
                embedding_list = embedding.cpu().numpy().tolist()
                
                # Cache dimension
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding_list)
                    logger.info(f"Local embedding dimension detected: {self._embedding_dim}")
                
                logger.debug(f"Text embedded (len={len(text)}, dim={len(embedding_list)})")
                return embedding_list
                
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise
    
    @property
    def embedding_dim(self) -> Optional[int]:
        """Get the embedding dimension (None if not yet determined)."""
        return self._embedding_dim
    
    def test_connection(self) -> bool:
        """
        Test the model loading by performing a simple embedding.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.embed_text("test")
            logger.info("Local model test successful")
            return True
        except Exception as e:
            logger.error(f"Local model test failed: {e}")
            return False


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Keep TextEmbedder as alias for APIEmbedder
TextEmbedder = APIEmbedder

# Keep VisualEmbedder as alias for backward compatibility
VisualEmbedder = APIEmbedder


# ============================================================================
# Factory Functions
# ============================================================================

def get_embedder(
    provider: str = "api",
    # API mode params
    api_key: str = "",
    api_url: str = "https://api.openai.com/v1",
    model_name: str = "text-embedding-ada-002",
    # Local mode params
    local_model_path: str = "",
    device: str = "auto",
    torch_dtype: str = "auto",
) -> BaseEmbedder:
    """
    Factory function to get an embedder based on provider type.
    
    Args:
        provider: "api" or "local"
        api_key: API key for authentication (API mode)
        api_url: Base URL for the API (API mode)
        model_name: Model name for embeddings (API mode)
        local_model_path: Absolute path to local model (Local mode)
        device: Device to load model on (Local mode)
        torch_dtype: Torch dtype for model (Local mode)
        
    Returns:
        BaseEmbedder instance (APIEmbedder or LocalEmbedder)
        
    Raises:
        ValueError: If provider is invalid or required params are missing
    """
    provider = provider.lower().strip()
    
    if provider == "api":
        return APIEmbedder(
            api_key=api_key,
            base_url=api_url,
            model_name=model_name,
        )
    
    elif provider == "local":
        if not local_model_path:
            raise ValueError("local_model_path is required for Local mode")
        
        return LocalEmbedder(
            model_path=local_model_path,
            device=device,
            torch_dtype=torch_dtype,
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'api' or 'local'.")


def get_text_embedder(
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model_name: str = "text-embedding-ada-002",
) -> APIEmbedder:
    """
    Get an APIEmbedder instance (backward compatibility).
    
    Args:
        api_key: API key for authentication
        base_url: Base URL for the API
        model_name: Model name for embeddings
        
    Returns:
        APIEmbedder instance
    """
    return APIEmbedder(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )
