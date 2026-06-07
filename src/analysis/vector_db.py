"""
Vector Store - ChromaDB Integration for Semantic Search
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for frame descriptions.
    Supports semantic search over video content.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "frame_descriptions",
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        
        self._init_db()
    
    def _init_db(self):
        """Initialize ChromaDB client and collection."""
        import chromadb
        from chromadb.config import Settings
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI-Sentinel frame analysis results"}
        )
        
        logger.info(f"VectorStore initialized: {self.persist_directory}/{self.collection_name}")
        logger.info(f"  Existing documents: {self._collection.count()}")
    
    def add(
        self,
        frame_id: str,
        description: str,
        timestamp: float,
        camera_id: str = "default",
        image_path: str = "",
        real_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a frame description to the vector store.
        
        Args:
            frame_id: Unique identifier for the frame
            description: Text description of the frame content
            timestamp: Video timestamp in seconds (or unix timestamp)
            camera_id: Camera identifier
            image_path: Path to the saved frame image
            real_time: Human-readable timestamp string (from OCR)
            metadata: Additional metadata
            
        Returns:
            str: Document ID
        """
        # Generate unique document ID
        doc_id = f"{camera_id}_{frame_id}"
        
        # Build metadata
        doc_metadata = {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": float(timestamp),
            "image_path": image_path,
            "capture_time": datetime.now().isoformat(),
        }
        
        if real_time:
            doc_metadata["real_time"] = real_time
            
        if metadata:
            doc_metadata.update(metadata)
        
        # Filter out None values (ChromaDB doesn't accept None)
        doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
        
        # Add to collection (ChromaDB handles embedding automatically)
        self._collection.add(
            ids=[doc_id],
            documents=[description],
            metadatas=[doc_metadata],
        )
        
        logger.debug(f"Added document: {doc_id}, desc: {description[:50]}...")
        return doc_id

    def delete(self, doc_id: str):
        """Delete a document by ID."""
        try:
            self._collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")

    def delete_by_date(self, date_str: str):
        """
        Delete all documents for a specific date.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
        """
        try:
            # Parse date and create timestamp range
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Safe timestamp calculation for Windows pre-epoch support
            try:
                start_ts = target_date.timestamp()
            except OSError:
                # Fallback: calculate using safe offset (e.g. from 2000-01-01)
                safe_base = datetime(2000, 1, 1)
                safe_ts_base = safe_base.timestamp()
                delta = (safe_base - target_date).total_seconds()
                start_ts = safe_ts_base - delta
                
            # End of day (next day - 1 microsecond) or just < next day
            end_ts = start_ts + 86400.0
            
            # Delete using where filter on timestamp
            self._collection.delete(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lt": end_ts}}
                    ]
                }
            )
            logger.info(f"Deleted documents for date: {date_str}")
        except Exception as e:
            logger.error(f"Error deleting documents for date {date_str}: {e}")
            raise e


    def delete_by_hour(self, date_str: str, hour: int):
        """
        Delete all documents for a specific hour on a specific date.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            hour: Hour (0-23)
        """
        try:
            # Parse date
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Safe timestamp calculation
            try:
                base_ts = target_date.timestamp()
            except OSError:
                safe_base = datetime(2000, 1, 1)
                safe_ts_base = safe_base.timestamp()
                delta = (safe_base - target_date).total_seconds()
                base_ts = safe_ts_base - delta
            
            # Calculate range for the specific hour
            start_ts = base_ts + (hour * 3600.0)
            end_ts = start_ts + 3600.0
            
            # Delete using where filter
            self._collection.delete(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lt": end_ts}}
                    ]
                }
            )
            logger.info(f"Deleted documents for {date_str} hour {hour}")
        except Exception as e:
            logger.error(f"Error deleting documents for {date_str} hour {hour}: {e}")
            raise e

    def get_documents_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a specific date.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            
        Returns:
            List of documents with details
        """
        try:
            # Parse date and create timestamp range
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Safe timestamp calculation
            try:
                start_ts = target_date.timestamp()
            except OSError:
                safe_base = datetime(2000, 1, 1)
                safe_ts_base = safe_base.timestamp()
                delta = (safe_base - target_date).total_seconds()
                start_ts = safe_ts_base - delta
                
            end_ts = start_ts + 86400.0
            
            # Query
            results = self._collection.get(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lt": end_ts}}
                    ]
                },
                include=["documents", "metadatas"]
            )
            
            entries = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    entries.append({
                        "id": doc_id,
                        "description": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            
            # Sort by capture time descending
            entries.sort(key=lambda x: x["metadata"].get("capture_time", ""), reverse=True)
            return entries
            
        except Exception as e:
            logger.error(f"Error fetching documents for date {date_str}: {e}")
            return []

    
    def search(
        self,
        query: str,
        n_results: int = 10,
        camera_id: Optional[str] = None,
        time_range: Optional[tuple] = None,  # (start_ts, end_ts)
        min_score: Optional[float] = None,  # Minimum similarity score (0-1)
        sort_by: str = "similarity",  # "similarity" or "time"
    ) -> List[Dict[str, Any]]:
        """
        Search for frames matching the query.
        
        Args:
            query: Natural language search query
            n_results: Maximum number of results
            camera_id: Optional filter by camera
            time_range: Optional filter by timestamp range (start_ts, end_ts)
            min_score: Optional minimum similarity score (0-1), filters out low-relevance results
            sort_by: "similarity" (default) or "time" (prioritize latest)
            
        Returns:
            List of results with description, metadata, and distance
        """
        # Build where filter
        filters = []
        if camera_id:
            filters.append({"camera_id": camera_id})
        
        if time_range:
            filters.append({
                "$and": [
                    {"timestamp": {"$gte": time_range[0]}},
                    {"timestamp": {"$lte": time_range[1]}}
                ]
            })
            
        if len(filters) > 1:
            where_filter = {"$and": filters}
        elif len(filters) == 1:
            where_filter = filters[0]
        else:
            where_filter = None
        
        results_map = {}
        
        # Determine fetch limit
        # If sorting by time, we need to fetch more candidates to ensure we capture recent ones
        # even if their similarity score isn't in the global top N.
        # We fetch 5x requested results to form a pool.
        fetch_limit = n_results * 5 if sort_by == "time" else n_results
        
        # 1. Vector Search
        try:
            vec_results = self._collection.query(
                query_texts=[query],
                n_results=fetch_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            
            if vec_results and vec_results["ids"] and vec_results["ids"][0]:
                for i, doc_id in enumerate(vec_results["ids"][0]):
                    dist = vec_results["distances"][0][i] if vec_results.get("distances") else 1.0
                    sim = max(0, 1 - dist / 2)
                    
                    results_map[doc_id] = {
                        "id": doc_id,
                        "description": vec_results["documents"][0][i],
                        "metadata": vec_results["metadatas"][0][i],
                        "distance": dist,
                        "similarity": sim,
                    }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # 2. Multi-Keyword Search with Chinese Segmentation
        # Use jieba for Chinese tokenization to improve matching
        # Important single-char nouns that should be preserved
        IMPORTANT_SINGLE_CHARS = {'猫', '狗', '人', '车', '鸟', '树', '门', '窗', '狼', '熊', '牛', '羊', '马', '鸡', '鸭', '鹅'}
        
        try:
            import jieba
            keywords = [k.strip() for k in jieba.cut(query) 
                       if k.strip() and (len(k.strip()) > 1 or k.strip() in IMPORTANT_SINGLE_CHARS)]
        except ImportError:
            # Fallback: split by space or use whole query
            keywords = [k.strip() for k in query.split() if k.strip()]
            if not keywords and query.strip():
                keywords = [query.strip()]
        
        logger.debug(f"Keyword search with terms: {keywords}")
        
        # Search for each keyword and merge results
        keyword_hits = {}  # doc_id -> hit_count
        keyword_docs = {}  # doc_id -> doc_data
        
        for kw in keywords[:5]:  # Limit to first 5 keywords for performance
            try:
                kw_results = self._collection.get(
                    where=where_filter,
                    where_document={"$contains": kw},
                    include=["documents", "metadatas"],
                    limit=fetch_limit
                )
                
                if kw_results and kw_results["ids"]:
                    for i, doc_id in enumerate(kw_results["ids"]):
                        keyword_hits[doc_id] = keyword_hits.get(doc_id, 0) + 1
                        if doc_id not in keyword_docs:
                            keyword_docs[doc_id] = {
                                "id": doc_id,
                                "description": kw_results["documents"][i],
                                "metadata": kw_results["metadatas"][i],
                            }
            except Exception as e:
                logger.warning(f"Keyword search for '{kw}' failed: {e}")
        
        # Merge keyword matches into results
        total_keywords = len(keywords) if keywords else 1
        for doc_id, hit_count in keyword_hits.items():
            # Calculate similarity based on how many keywords matched
            keyword_similarity = 0.5 + (hit_count / total_keywords) * 0.45  # Range: 0.5 ~ 0.95
            
            if doc_id not in results_map:
                doc_data = keyword_docs[doc_id]
                results_map[doc_id] = {
                    "id": doc_id,
                    "description": doc_data["description"],
                    "metadata": doc_data["metadata"],
                    "distance": 2 * (1 - keyword_similarity),
                    "similarity": keyword_similarity,
                    "_keyword_hits": hit_count,
                }
            else:
                # Boost existing entry if keywords matched
                boost = 0.1 + (hit_count / total_keywords) * 0.1  # 0.1 ~ 0.2 boost
                results_map[doc_id]["similarity"] = min(0.99, results_map[doc_id]["similarity"] + boost)
                results_map[doc_id]["distance"] = 2 * (1 - results_map[doc_id]["similarity"])
                results_map[doc_id]["_keyword_hits"] = hit_count

        # Convert map to list
        formatted = list(results_map.values())
        
        # Filter by min_score if specified
        if min_score is not None:
            formatted = [r for r in formatted if r["similarity"] >= min_score]
            
        # Final Sort
        if sort_by == "time":
            # Sort by timestamp descending
            formatted.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        else:
            # Sort by similarity desc
            formatted.sort(key=lambda x: x["similarity"], reverse=True)
            
        logger.info(f"Hybrid Search '{query}' returned {len(formatted)} results (sort={sort_by})")
        return formatted[:n_results]
    
    def get_by_frame_id(self, frame_id: str, camera_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get a specific frame by ID."""
        doc_id = f"{camera_id}_{frame_id}"
        
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        
        if result and result["ids"]:
            return {
                "id": result["ids"][0],
                "description": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None
    
    def get_recent(self, n: int = 20, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the most recent N entries."""
        where_filter = {"camera_id": camera_id} if camera_id else None
        
        # Get all with metadata, then sort by capture_time
        result = self._collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        
        if not result or not result["ids"]:
            return []
        
        # Build list and sort
        entries = []
        for i, doc_id in enumerate(result["ids"]):
            entries.append({
                "id": doc_id,
                "description": result["documents"][i],
                "metadata": result["metadatas"][i],
            })
        
        # Sort by capture_time descending
        entries.sort(key=lambda x: x["metadata"].get("capture_time", ""), reverse=True)
        
        return entries[:n]
    
    def count(self) -> int:
        """Get total number of documents."""
        return self._collection.count()
    
    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "AI-Sentinel frame analysis results"}
        )
        logger.info("VectorStore cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": self._collection.count(),
            "persist_directory": str(self.persist_directory),
        }
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all documents (optimized for browser).
        Returns a list of dicts: {'id': doc_id, 'metadata': {...}}
        """
        result = self._collection.get(
            include=["metadatas"]
        )
        
        entries = []
        if result and result["ids"]:
            for i, doc_id in enumerate(result["ids"]):
                entries.append({
                    "id": doc_id,
                    "metadata": result["metadatas"][i]
                })
        return entries


class VisualVectorStore:
    """
    ChromaDB-based vector store for text-based embeddings.
    Uses custom embeddings from Text Embedding API for semantic search.
    
    Strategy: VLM description -> Text Embedding API -> Vector storage
    
    ⚠️ BREAKING CHANGE: If the embedding dimension changes, the old database
    must be deleted and rebuilt. The system will detect dimension mismatches
    and prompt for action.
    """
    
    COLLECTION_NAME = "video_semantic_search"
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        embedding_dim: Optional[int] = None,  # Auto-detect from first embedding if None
    ):
        """
        Initialize semantic vector store.
        
        Args:
            persist_directory: Path to ChromaDB storage
            embedding_dim: Expected embedding dimension (auto-detect if None)
        """
        self.persist_directory = Path(persist_directory)
        self._expected_dim = embedding_dim
        self._actual_dim = None
        self._client = None
        self._collection = None
        
        self._init_db()
    
    def _init_db(self):
        """Initialize ChromaDB client and collection."""
        import chromadb
        from chromadb.config import Settings
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Check if collection exists and get its metadata
        existing_collections = [c.name for c in self._client.list_collections()]
        
        if self.COLLECTION_NAME in existing_collections:
            self._collection = self._client.get_collection(
                name=self.COLLECTION_NAME,
            )
            
            # Try to get stored dimension from collection metadata
            metadata = self._collection.metadata or {}
            stored_dim = metadata.get("embedding_dim")
            
            if stored_dim:
                self._actual_dim = int(stored_dim)
                logger.info(f"VisualVectorStore loaded existing collection (dim: {self._actual_dim})")
                
                # Check dimension mismatch
                if self._expected_dim and self._expected_dim != self._actual_dim:
                    logger.warning(
                        f"⚠️ DIMENSION MISMATCH: Expected {self._expected_dim}, "
                        f"but collection has {self._actual_dim}. "
                        f"You may need to rebuild the database."
                    )
            else:
                logger.info(f"VisualVectorStore loaded existing collection (dim: unknown)")
        else:
            # Create new collection
            collection_metadata = {
                "description": "AI-Sentinel text embeddings for semantic search",
                "hnsw:space": "cosine",  # Use cosine similarity
            }
            
            if self._expected_dim:
                collection_metadata["embedding_dim"] = str(self._expected_dim)
                self._actual_dim = self._expected_dim
            
            self._collection = self._client.create_collection(
                name=self.COLLECTION_NAME,
                metadata=collection_metadata,
            )
            logger.info(f"VisualVectorStore created new collection: {self.COLLECTION_NAME}")
        
        logger.info(f"VisualVectorStore initialized: {self.persist_directory}/{self.COLLECTION_NAME}")
        logger.info(f"  Existing documents: {self._collection.count()}")
    
    def add(
        self,
        frame_id: str,
        embedding: "List[float]",
        timestamp: float,
        camera_id: str = "default",
        image_path: str = "",
        description: str = "",
        alert_info: str = "",
        real_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a frame embedding to the vector store.
        
        Args:
            frame_id: Unique identifier for the frame
            embedding: Embedding vector from TextEmbedder (List[float])
            timestamp: Video timestamp in seconds
            camera_id: Camera identifier
            image_path: Path to the saved frame image
            description: Text description of the frame (from VLM)
            alert_info: Alert information if any
            real_time: Human-readable timestamp
            metadata: Additional metadata
            
        Returns:
            str: Document ID
        """
        # Validate and record dimension
        emb_dim = len(embedding)
        
        if self._actual_dim is None:
            # First embedding - record dimension
            self._actual_dim = emb_dim
            # Update collection metadata
            try:
                # ChromaDB doesn't support updating collection metadata directly
                # Store in first document's metadata as fallback
                logger.info(f"First embedding dimension recorded: {emb_dim}")
            except Exception:
                pass
        elif self._actual_dim != emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch! Expected {self._actual_dim}, got {emb_dim}. "
                f"Please rebuild the database with: visual_store.rebuild()"
            )
        
        # Generate unique document ID
        doc_id = f"semantic_{camera_id}_{frame_id}"
        
        # Build metadata
        doc_metadata = {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": float(timestamp),
            "image_path": image_path,
            "description": description,
            "alert_info": alert_info,
            "capture_time": datetime.now().isoformat(),
            "embedding_dim": emb_dim,
        }
        
        if real_time:
            doc_metadata["real_time"] = real_time
            
        if metadata:
            doc_metadata.update(metadata)
        
        # Filter out None/empty values
        doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None and v != ""}
        
        # Ensure embedding is a list
        if not isinstance(embedding, list):
            embedding_list = list(embedding)
        else:
            embedding_list = embedding
        
        # Add to collection with custom embedding
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding_list],
            documents=[description] if description else [f"Frame {frame_id}"],
            metadatas=[doc_metadata],
        )
        
        logger.debug(f"Added text embedding: {doc_id}, dim: {emb_dim}")
        return doc_id
    
    def search_by_embedding(
        self,
        query_embedding: "List[float]",
        n_results: int = 10,
        camera_id: Optional[str] = None,
        time_range: Optional[tuple] = None,
        min_score: Optional[float] = None,  # Minimum similarity score (0-1)
        sort_by: str = "similarity",  # "similarity" or "time"
    ) -> List[Dict[str, Any]]:
        """
        Search for frames using an embedding vector.
        
        Args:
            query_embedding: Query embedding from TextEmbedder.embed_text()
            n_results: Maximum number of results
            camera_id: Optional filter by camera
            time_range: Optional filter by timestamp range (start_ts, end_ts)
            min_score: Optional minimum similarity score (0-1), filters out low-relevance results
            sort_by: "similarity" or "time"
            
        Returns:
            List of results with metadata and similarity scores
        """
        # Build where filter
        filters = []
        if camera_id:
            filters.append({"camera_id": camera_id})
        
        if time_range:
            filters.append({
                "$and": [
                    {"timestamp": {"$gte": time_range[0]}},
                    {"timestamp": {"$lte": time_range[1]}}
                ]
            })
            
        if len(filters) > 1:
            where_filter = {"$and": filters}
        elif len(filters) == 1:
            where_filter = filters[0]
        else:
            where_filter = None
        
        # Ensure embedding is a list
        if not isinstance(query_embedding, list):
            query_list = list(query_embedding)
        else:
            query_list = query_embedding
            
        # Determine fetch limit based on sort logic
        fetch_limit = n_results * 5 if sort_by == "time" else n_results
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[query_list],
            n_results=fetch_limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else None
                # Convert distance to similarity (cosine distance to similarity)
                similarity = 1 - distance if distance is not None else None
                
                result = {
                    "id": doc_id,
                    "description": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": distance,
                    "similarity": similarity,
                }
                
                # Filter by min_score if specified
                if min_score is not None and similarity is not None:
                    if similarity < min_score:
                        continue
                
                # Apply time range filter
                if time_range:
                    ts = result["metadata"].get("timestamp", 0)
                    if not (time_range[0] <= ts <= time_range[1]):
                        continue
                
                formatted.append(result)
        
        # Sort if needed (Query already returns sorted by distance/similarity, so only time sort needed)
        if sort_by == "time":
            formatted.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
            
        logger.info(f"Semantic search returned {len(formatted)} results (sort={sort_by})")
        return formatted[:n_results]
    
    def search_by_text(
        self,
        embedder: "TextEmbedder",
        query_text: str,
        n_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: Search using text query.
        
        Args:
            embedder: TextEmbedder instance
            query_text: Text search query
            n_results: Maximum results
            **kwargs: Additional filters (camera_id, time_range, min_score, sort_by)
            
        Returns:
            List of search results
        """
        # Get text embedding
        query_embedding = embedder.embed_text(query_text)
        
        return self.search_by_embedding(
            query_embedding=query_embedding,
            n_results=n_results,
            **kwargs
        )
    
    def get_by_frame_id(self, frame_id: str, camera_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get a specific frame by ID."""
        doc_id = f"semantic_{camera_id}_{frame_id}"
        
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"],
        )
        
        if result and result["ids"]:
            return {
                "id": result["ids"][0],
                "description": result["documents"][0] if result.get("documents") else "",
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                "embedding": result["embeddings"][0] if result.get("embeddings") else None,
            }
        return None
    
    def count(self) -> int:
        """Get total number of documents."""
        return self._collection.count()
    
    def delete(self, doc_id: str):
        """Delete a document by ID."""
        try:
            # Try direct ID first
            self._collection.delete(ids=[doc_id])
            logger.info(f"Deleted document from visual store: {doc_id}")
        except Exception as e:
            # Try with semantic prefix
            try:
                # Fix: Handle doc_id that might already include camera_id but lack semantic prefix
                # If doc_id is "cam_01_123", we want "semantic_cam_01_123"
                semantic_id = f"semantic_{doc_id}" if not doc_id.startswith("semantic_") else doc_id
                self._collection.delete(ids=[semantic_id])
                logger.info(f"Deleted document from visual store: {semantic_id}")
            except Exception as e2:
                logger.debug(f"Document not found in visual store: {doc_id} - {e2}")

    def delete_by_date(self, date_str: str):
        """
        Delete all documents for a specific date from visual store.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
        """
        try:
            from datetime import datetime
            # Parse date and create timestamp range
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Safe timestamp calculation for Windows pre-epoch support
            try:
                start_ts = target_date.timestamp()
            except OSError:
                # Fallback: calculate using safe offset (e.g. from 2000-01-01)
                safe_base = datetime(2000, 1, 1)
                safe_ts_base = safe_base.timestamp()
                delta = (safe_base - target_date).total_seconds()
                start_ts = safe_ts_base - delta
                
            end_ts = start_ts + 86400.0
            
            # Delete using where filter on timestamp
            self._collection.delete(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lt": end_ts}}
                    ]
                }
            )
            logger.info(f"Deleted visual documents for date: {date_str}")
        except Exception as e:
            logger.error(f"Error deleting visual documents for date {date_str}: {e}")
            raise e


    def delete_by_hour(self, date_str: str, hour: int):
        """
        Delete all documents for a specific hour on a specific date from visual store.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            hour: Hour (0-23)
        """
        try:
            from datetime import datetime
            
            # Parse date
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Safe timestamp calculation
            try:
                base_ts = target_date.timestamp()
            except OSError:
                safe_base = datetime(2000, 1, 1)
                safe_ts_base = safe_base.timestamp()
                delta = (safe_base - target_date).total_seconds()
                base_ts = safe_ts_base - delta
            
            # Calculate range for the specific hour
            start_ts = base_ts + (hour * 3600.0)
            end_ts = start_ts + 3600.0
            
            # Delete using where filter
            self._collection.delete(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lt": end_ts}}
                    ]
                }
            )
            logger.info(f"Deleted visual documents for {date_str} hour {hour}")
        except Exception as e:
            logger.error(f"Error deleting visual documents for {date_str} hour {hour}: {e}")
            raise e
    
    def clear(self):
        """Clear all documents from the collection."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._actual_dim = None
        self._init_db()
        logger.info("VisualVectorStore cleared")
    
    def rebuild(self, new_embedding_dim: Optional[int] = None):
        """
        Rebuild the collection with a new embedding dimension.
        
        ⚠️ WARNING: This will DELETE all existing data!
        
        Args:
            new_embedding_dim: New expected dimension (or None to auto-detect)
        """
        logger.warning("⚠️ REBUILDING VisualVectorStore - All existing data will be deleted!")
        
        # Delete existing collection
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        
        # Reset state
        self._expected_dim = new_embedding_dim
        self._actual_dim = None
        self._collection = None
        
        # Reinitialize
        self._init_db()
        
        logger.info(f"VisualVectorStore rebuilt (expected dim: {new_embedding_dim or 'auto'})")
    
    def check_dimension_compatibility(self, embedding_dim: int) -> bool:
        """
        Check if an embedding dimension is compatible with the store.
        
        Args:
            embedding_dim: Dimension to check
            
        Returns:
            bool: True if compatible, False if mismatch
        """
        if self._actual_dim is None:
            return True  # No data yet, any dimension is fine
        return self._actual_dim == embedding_dim
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "collection_name": self.COLLECTION_NAME,
            "document_count": self._collection.count(),
            "persist_directory": str(self.persist_directory),
            "embedding_dim": self._actual_dim,
        }

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all documents (optimized for browser).
        Returns a list of dicts: {'id': doc_id, 'metadata': {...}}
        """
        result = self._collection.get(
            include=["metadatas"]
        )
        
        entries = []
        if result and result["ids"]:
            for i, doc_id in enumerate(result["ids"]):
                entries.append({
                    "id": doc_id,
                    "metadata": result["metadatas"][i]
                })
        return entries
