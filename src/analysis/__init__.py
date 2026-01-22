# Analysis Module
from .vlm import VLMAnalyzer
from .vector_db import VectorStore
from .worker import AnalysisWorker
from .feature_extractor import (
    BaseEmbedder, 
    APIEmbedder, 
    LocalEmbedder, 
    get_embedder,
    # Backward compatibility aliases
    TextEmbedder, 
    VisualEmbedder,
)
