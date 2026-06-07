import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from .vector_db import VectorStore, VisualVectorStore
from .feature_extractor import BaseEmbedder

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Dedicated search engine for AI-Sentinel.
    Handles complex search strategies including Time-Windowed retrieval for "Latest First" sorting.
    """
    
    def __init__(self, 
                 vector_store: VectorStore, 
                 visual_store: Optional[VisualVectorStore] = None,
                 visual_embedder: Optional[BaseEmbedder] = None):
        self.vector_store = vector_store
        self.visual_store = visual_store
        self.visual_embedder = visual_embedder
        
    def search(self, 
               query: str, 
               n_results: int = 10, 
               sort_mode: str = "relevance", 
               time_range: Optional[tuple] = None,
               search_mode: str = "hybrid",
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Unified search entry point.
        
        Args:
            query: Search text
            n_results: Max results to return
            sort_mode: "relevance" or "time"
            time_range: Global time range constraint (start_ts, end_ts)
            search_mode: "text" (legacy), "semantic" (visual), or "hybrid"
            min_score: Minimum similarity score (0.0 - 1.0)
            
        Returns:
            List of unique, sorted results.
        """
        # logger.info(f"Search: '{query}', mode={search_mode}, sort={sort_mode}")
        
        candidates = []
        
        # Dispatch to specific strategy
        if sort_mode == "time":
            candidates = self._search_time_priority(query, n_results, time_range, search_mode, min_score)
        else:
            candidates = self._search_relevance_priority(query, n_results, time_range, search_mode, min_score)
            
        return candidates

    def _search_relevance_priority(self, query: str, n_results: int, time_range: Optional[tuple], search_mode: str, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Standard relevance-based search. Fetches globally best matches.
        """
        # Fetch more candidates for re-ranking
        fetch_limit = n_results * 5
        
        results = self._fetch_candidates(query, fetch_limit, time_range, search_mode)
        
        # Rank by hybrid score (Similarity + Keyword Bonus)
        ranked = self._rank_candidates(results, query, sort_by="score")
        
        # Apply min_score filter
        if min_score > 0:
            ranked = [r for r in ranked if r.get("similarity", 0) >= min_score]
        
        return ranked[:n_results]

    def _search_time_priority(self, query: str, n_results: int, global_time_range: Optional[tuple], search_mode: str, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Time-Windowed Search Strategy (Merged Approach).
        Fetches a mix of "Globally Best" and "Recently Best" items to ensure
        we don't miss high-quality older matches while still favoring freshness.
        """
        final_results = []
        seen_ids = set()
        
        candidates = []
        
        # 1. Fetch "Globally Best" (Relevance)
        # We fetch a larger pool to ensure we find good matches from the past
        global_pool = self._fetch_candidates(
            query, 
            n_results * 3, 
            global_time_range, 
            search_mode
        )
        candidates.extend(global_pool)
        
        # 2. Fetch "Recent Best" (Freshness)
        # Specifically target the last 7 days (or 24h) to ensure recent events are included
        # even if their semantic score is slightly lower than historical bests.
        current_ts = datetime.now().timestamp()
        recent_window = (current_ts - 7 * 86400, current_ts)
        
        # Only if global range allows (or is unset)
        range_valid = True
        if global_time_range:
             # Check if recent window overlaps with user range
             if global_time_range[1] < recent_window[0] or global_time_range[0] > recent_window[1]:
                 range_valid = False
        
        if range_valid:
            recent_pool = self._fetch_candidates(
                query,
                n_results * 5,  # Increased multiplier for safety
                recent_window,
                search_mode,
                sort_by="time" # CRITICAL: Ask DB to sort by time to find recent stuff first
            )
            candidates.extend(recent_pool)
            
        # 3. Merge and Deduplicate
        unique_candidates = []
        for r in candidates:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_candidates.append(r)
        
        # 4. Rank/Score (Calculate hybrid scores, but don't filter just yet)
        # Note: sort_by="time" here essentially just prepares the list, 
        # actual sorting happens at the end of this block.
        ranked_candidates = self._rank_candidates(unique_candidates, query, sort_by="time")
        
        # 5. Filter by Min Score with Smart Relevance Detection
        # For time priority mode, we need to ensure results are TRULY RELEVANT to the query.
        # Strategy: 
        #   - MUST have keyword match (at least one query term appears in description)
        #   - AND similarity >= floor (to avoid very low quality matches)
        # This strictly prevents returning irrelevant items even if semantically similar.
        
        # Get keywords for relevance checking
        if JIEBA_AVAILABLE:
            query_keywords = [k.strip().lower() for k in jieba.cut(query) 
                            if k.strip() and (len(k.strip()) > 1 or k.strip() in {'猫', '狗', '人', '车', '鸟'})]
        else:
            query_keywords = [query.strip().lower()] if query.strip() else []
        
        # Floor threshold for quality
        floor_score = max(min_score, 0.25)  # Lowered to 0.25 since we now require keyword match
        
        filtered = []
        for r in ranked_candidates:
            sim = r.get("similarity", 0)
            desc = r.get("description", "").lower()
            
            # Check keyword hits - REQUIRED for time priority mode
            has_keyword_match = any(kw in desc for kw in query_keywords) if query_keywords else True
            
            # MUST have keyword match AND meet minimum similarity
            if has_keyword_match and sim >= floor_score:
                filtered.append(r)
                
        # 6. Final Sort by Time
        def get_ts(r):
            try: return float(r.get("metadata", {}).get("timestamp", 0))
            except: return 0.0
            
        filtered.sort(key=get_ts, reverse=True)
        
        return filtered[:n_results]

    def _fetch_candidates(self, query: str, limit: int, time_range: Optional[tuple], search_mode: str, sort_by: str = "similarity") -> List[Dict[str, Any]]:
        """
        Helper to fetch raw candidates from stores (Semantic + Text).
        """
        combined_candidates = []
        merged_map = {}
        
        # Helper to normalize ID
        def get_base_id(doc_id):
            return doc_id.replace("semantic_", "")
        
        # 1. Semantic Search
        if search_mode in ["semantic", "hybrid"] and self.visual_store and self.visual_embedder:
            try:
                # Use search_by_text convenience
                sem_results = self.visual_store.search_by_text(
                    embedder=self.visual_embedder,
                    query_text=query,
                    n_results=limit,
                    time_range=time_range,
                    sort_by=sort_by
                )
                for r in sem_results:
                    r['_source'] = 'semantic'
                    merged_map[get_base_id(r['id'])] = r
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
                
        # 2. Text/Keyword Search
        # Always run text search if mode is text or hybrid
        # Text search (ChromaDB default) is good for exact keyword matching
        if search_mode in ["text", "hybrid"]:
            try:
                text_results = self.vector_store.search(
                    query=query,
                    n_results=limit,
                    time_range=time_range,
                    sort_by=sort_by
                )
                for r in text_results:
                    base_id = get_base_id(r['id'])
                    if base_id in merged_map:
                        # Merge/Boost
                        existing = merged_map[base_id]
                        score_sem = existing.get('similarity', 0)
                        score_text = r.get('similarity', 0)
                        
                        # Boost logic
                        new_score = max(score_sem, score_text) + 0.1
                        existing['similarity'] = min(0.99, new_score)
                        existing['_source'] = 'hybrid'
                    else:
                        r['_source'] = 'text'
                        merged_map[base_id] = r
                        
            except Exception as e:
                logger.error(f"Text search error: {e}")
        
        combined_candidates = list(merged_map.values())
        return combined_candidates

    def _rank_candidates(self, results: List[Dict[str, Any]], query: str, sort_by: str = "score") -> List[Dict[str, Any]]:
        """
        Apply scoring and sort with improved Chinese keyword matching.
        """
        # Enhanced Chinese keyword extraction using jieba
        # Important single-char nouns that should be preserved
        IMPORTANT_SINGLE_CHARS = {'猫', '狗', '人', '车', '鸟', '树', '门', '窗', '狼', '熊', '牛', '羊', '马', '鸡', '鸭', '鹅'}
        
        if JIEBA_AVAILABLE:
            # Use jieba for Chinese text segmentation
            raw_keywords = list(jieba.cut(query))
            # Filter: keep words with length > 1 OR important single-char nouns
            keywords = [k.strip().lower() for k in raw_keywords 
                       if k.strip() and (len(k.strip()) > 1 or k.strip() in IMPORTANT_SINGLE_CHARS)]
            # If jieba produced no useful keywords, fallback to original query
            if not keywords:
                keywords = [query.strip().lower()] if query.strip() else []
        else:
            # Fallback: simple space-based split
            raw_keywords = [k.strip().lower() for k in query.split() if k.strip()]
            keywords = raw_keywords
            # For Chinese without spaces, treat entire query as one keyword
            if not keywords and query.strip():
                keywords = [query.strip().lower()]
        
        logger.debug(f"Search keywords extracted: {keywords}")
        
        current_ts = datetime.now().timestamp()
        
        reranked = []
        for r in results:
            desc = r.get("description", "").lower()
            
            # Base score
            sim = r.get("similarity", 0)
            if sim is None: sim = 0
            
            # 1. Keyword Bonus (Increased for better Chinese matching)
            kw_bonus = 0.0
            if keywords:
                hits = sum(1 for k in keywords if k in desc)
                ratio = hits / len(keywords)
                # Increased from 0.08 to 0.15 for better keyword relevance
                kw_bonus = ratio * 0.15
            
            # 2. Time Sorting / Decay
            ts = 0.0
            try:
                ts = float(r.get("metadata", {}).get("timestamp", 0))
            except: 
                pass
                
            # Time Bonus (Reduced logic)
            # Only applied for 'score' sorting to break ties or slight boost.
            # For 'time' sorting, this bonus is less relevant as we sort by TS anyway.
            time_bonus = 0.0
            if ts > 0:
                age_seconds = max(0, current_ts - ts)
                if age_seconds < 86400: # Last 24h
                    time_bonus = 0.05 * (1 - (age_seconds / 86400))
            
            # 3. Global Time Tie-Breaker
            tie_breaker = ts / 1e12 
            
            # Final Score Calculation
            final_score = sim + kw_bonus + time_bonus + tie_breaker
            
            # Clone and Annotate
            new_r = r.copy()
            new_r["_hybrid_score"] = final_score
            
            # IMPORTANT: Do NOT overwrite raw similarity with boosted score.
            # The UI shows "distance" based on similarity.
            # We want to show the USER how well it matched content, not how well it matched our algorithm.
            # However, for 'sort by relevance', we use _hybrid_score.
            
            reranked.append(new_r)
            
        if sort_by == "time":
            def get_ts(r):
                try: return float(r.get("metadata", {}).get("timestamp", 0))
                except: return 0.0
            reranked.sort(key=get_ts, reverse=True)
        else:
            # Sort by hybrid score
            reranked.sort(key=lambda x: x["_hybrid_score"], reverse=True)
            
        return reranked
