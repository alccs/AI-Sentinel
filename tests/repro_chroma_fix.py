import os
import shutil
import time
from typing import List, Dict, Any, Optional

# Mock or Minimal implementation of the VectorStore query logic to test syntax
# We need to install chromadb to test this properly, or rely on the existing environment.
# Assuming the environment has chromadb as per the traceback.

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed, cannot verify fix directly.")
    exit(1)

TEST_DIR = "./data/test_chroma"

def setup_db():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    
    client = chromadb.PersistentClient(
        path=TEST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    
    collection = client.get_or_create_collection(name="test_collection")
    
    # Add some dummy data
    collection.add(
        ids=["1", "2", "3"],
        documents=["doc1", "doc2", "doc3"],
        metadatas=[
            {"timestamp": 100},
            {"timestamp": 200},
            {"timestamp": 300}
        ]
    )
    return collection

def test_query_old_broken(collection):
    print("\n--- Testing Old Broken Syntax ---")
    time_range = (150, 250)
    try:
        # This mirrors the old code that caused the error
        where_filter = {
            "timestamp": {
                "$gte": time_range[0],
                "$lte": time_range[1]
            }
        }
        
        results = collection.query(
            query_texts=["doc"],
            n_results=1,
            where=where_filter
        )
        print("Scary! Old syntax actually worked? (It shouldn't if chroma updated)")
    except Exception as e:
        print(f"Caught expected error: {e}")

def test_query_new_fixed(collection):
    print("\n--- Testing New Fixed Syntax ---")
    time_range = (150, 250)
    try:
        # This mirrors the new code
        where_filter = {
            "$and": [
                {"timestamp": {"$gte": time_range[0]}},
                {"timestamp": {"$lte": time_range[1]}}
            ]
        }
        
        results = collection.query(
            query_texts=["doc"],
            n_results=1,
            where=where_filter
        )
        print(f"Success! Query returned {len(results['ids'][0])} results.")
        # Expecting doc 2 (timestamp 200)
        print(f"Result IDs: {results['ids']}")
    except Exception as e:
        print(f"FAILED: New syntax caused error: {e}")
        raise

if __name__ == "__main__":
    try:
        col = setup_db()
        test_query_old_broken(col)
        test_query_new_fixed(col)
        print("\nVerification Complete.")
    finally:
        if os.path.exists(TEST_DIR):
            try:
                shutil.rmtree(TEST_DIR)
            except:
                pass
