
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- Mock Search Engine Logic (Copied from src/analysis/search.py) ---

class SearchEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def search(self, query, n_results=10, sort_mode="relevance", time_range=None, min_score=0.0):
        # Dispatch
        if sort_mode == "time":
            return self._search_time_priority(query, n_results, time_range, "hybrid", min_score)
        else:
            return self._search_relevance_priority(query, n_results, time_range, "hybrid", min_score)

    def _fetch_candidates(self, query, limit, time_range, search_mode, sort_by="similarity"):
        # Mock fetch from store
        return self.vector_store.search(query, limit, time_range, sort_by)

    def _rank_candidates(self, results, query, sort_by="score"):
        # Mock ranking - minimal logic used in real code
        # In real code, it adds bonuses. Here we just sort.
        reranked = []
        for r in results:
            new_r = r.copy()
            # Simplified score just equals similarity for this test
            new_r["_hybrid_score"] = r["similarity"]
            reranked.append(new_r)
            
        if sort_by == "time":
             reranked.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        else:
             reranked.sort(key=lambda x: x["_hybrid_score"], reverse=True)
        return reranked

    def _search_relevance_priority(self, query: str, n_results: int, time_range: Optional[tuple], search_mode: str, min_score: float = 0.0) -> List[Dict[str, Any]]:
        fetch_limit = n_results * 5
        results = self._fetch_candidates(query, fetch_limit, time_range, search_mode)
        ranked = self._rank_candidates(results, query, sort_by="score")
        
        # Logic from codebase:
        if min_score > 0:
            ranked = [r for r in ranked if r.get("similarity", 0) >= min_score]
        
        return ranked[:n_results]

    def _search_time_priority(self, query: str, n_results: int, global_time_range: Optional[tuple], search_mode: str, min_score: float = 0.0) -> List[Dict[str, Any]]:
        candidates = []
        seen_ids = set()
        
        # 1. Fetch "Globally Best"
        global_pool = self._fetch_candidates(query, n_results * 3, global_time_range, search_mode)
        candidates.extend(global_pool)
        
        # 2. Fetch "Recent Best"
        # We mock this by just fetching more, typically it fetches by time
        recent_pool = self._fetch_candidates(query, n_results * 5, global_time_range, search_mode, sort_by="time")
        candidates.extend(recent_pool)
            
        # 3. Merge and Deduplicate
        unique_candidates = []
        for r in candidates:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_candidates.append(r)
        
        # 4. Rank/Score
        ranked_candidates = self._rank_candidates(unique_candidates, query, sort_by="time")
        
        # 5. Filter by Min Score
        # --- THE BUGGY LOGIC ---
        effective_min_score = min_score if min_score > 0 else 0.25
        
        filtered = []
        for r in ranked_candidates:
            if r.get("similarity", 0) >= effective_min_score:
                filtered.append(r)
                
        # 6. Final Sort by Time
        filtered.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        
        return filtered[:n_results]

# --- Test ---

class MockVectorStore:
    def __init__(self):
        self.data = []
        
    def search(self, query, n_results, time_range=None, sort_by="similarity"):
        # Return all data
        return [d.copy() for d in self.data]

def test_search_filtering():
    print("--- Starting Reproduction Test (Standalone) ---")
    
    store = MockVectorStore()
    
    # Target item: Relevant but low score (0.22)
    doc_target = {
        "id": "item_relevant_low_score",
        "distance": 0,
        "similarity": 0.22,
        "metadata": {"timestamp": 1000}
    }
    store.data.append(doc_target)
    
    engine = SearchEngine(store)
    
    # 1. Relevance Search (min_score=0.0)
    print("\n[Test 1] Search Mode: Relevance, min_score=0.0")
    res1 = engine.search("q", sort_mode="relevance", min_score=0.0)
    found1 = any(r['id'] == doc_target['id'] for r in res1)
    print(f"Found Target? {found1}")
    
    # 2. Time Search (min_score=0.0)
    print("\n[Test 2] Search Mode: Time, min_score=0.0")
    res2 = engine.search("q", sort_mode="time", min_score=0.0)
    found2 = any(r['id'] == doc_target['id'] for r in res2)
    print(f"Found Target? {found2}")
    
    if found1 and not found2:
        print("\nSUCCESS: Issue Reproduced! Item found in Relevance but hidden in Time mode.")
    else:
        print("\nFailed to reproduce issue.")

if __name__ == "__main__":
    test_search_filtering()
