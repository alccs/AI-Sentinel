
# Standalone Verification Script for Search Logic Fix
# This script uses the FIXED logic for _search_time_priority

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

class SearchEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def search(self, query, n_results=10, sort_mode="relevance", time_range=None, min_score=0.0):
        if sort_mode == "time":
            return self._search_time_priority(query, n_results, time_range, "hybrid", min_score)
        else:
            return self._search_relevance_priority(query, n_results, time_range, "hybrid", min_score)
            
    def _fetch_candidates(self, query, limit, time_range, search_mode, sort_by="similarity"):
        return self.vector_store.search(query, limit, time_range, sort_by)

    def _rank_candidates(self, results, query, sort_by="score"):
        reranked = []
        for r in results:
            new_r = r.copy()
            new_r["_hybrid_score"] = r["similarity"]
            reranked.append(new_r)
        if sort_by == "time":
             reranked.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        else:
             reranked.sort(key=lambda x: x["_hybrid_score"], reverse=True)
        return reranked

    def _search_relevance_priority(self, query, n_results, time_range, search_mode, min_score=0.0):
        # ... relevance logic (simplified) ...
        # Standard logic
        fetch_limit = n_results * 5
        results = self._fetch_candidates(query, fetch_limit, time_range, search_mode)
        ranked = self._rank_candidates(results, query, sort_by="score")
        if min_score > 0:
            ranked = [r for r in ranked if r.get("similarity", 0) >= min_score]
        return ranked[:n_results]

    def _search_time_priority(self, query, n_results, global_time_range, search_mode, min_score=0.0):
        # ... time logic ...
        candidates = []
        seen_ids = set()
        
        # 1. Global
        candidates.extend(self._fetch_candidates(query, n_results*3, global_time_range, search_mode))
        # 2. Recent
        candidates.extend(self._fetch_candidates(query, n_results*5, global_time_range, search_mode, sort_by="time"))
        
        # 3. Dedup
        unique_candidates = []
        for r in candidates:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_candidates.append(r)
        
        # 4. Rank
        ranked = self._rank_candidates(unique_candidates, query, sort_by="time")
        
        # 5. Filter (FIXED)
        # OLD: effective_min_score = min_score if min_score > 0 else 0.25
        # NEW: effective_min_score = min_score
        effective_min_score = min_score
        
        filtered = []
        for r in ranked:
            if r.get("similarity", 0) >= effective_min_score:
                filtered.append(r)
                
        # 6. Final Sort
        filtered.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        return filtered[:n_results]

class MockStore:
    def __init__(self):
        self.data = []
    def search(self, q, limit, t, sort_by):
        return [d.copy() for d in self.data]

def verify():
    print("--- Verifying Search Fix ---")
    store = MockStore()
    
    # Target: 0.22 similarity (Low but Relevant)
    target = {"id": "target", "similarity": 0.22, "metadata": {"timestamp": 1000}}
    store.data.append(target)
    
    engine = SearchEngine(store)
    
    # Search Time Priority with min_score=0.0 (UI default param)
    print("Searching with sort_mode='time', min_score=0.0")
    results = engine.search("cat", sort_mode="time", min_score=0.0)
    
    found = any(r['id'] == "target" for r in results)
    if found:
        print("SUCCESS: Target found (0.22 similarity >= 0.0 effective threshold)")
    else:
        print("FAILURE: Target filtered out")

if __name__ == "__main__":
    verify()
