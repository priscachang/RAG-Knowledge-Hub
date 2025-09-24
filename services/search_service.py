import numpy as np
from typing import List, Dict

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    try:
        a, b = np.array(vec_a), np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    except Exception as e:
        print(f"Error in cosine_similarity: {e}")
        return 0.0

def hybrid_search(query: str, kb_entries: List[Dict], top_k: int = 5) -> List[Dict]:
    """Combine semantic and keyword search."""
    try:
        from utils import get_embedding
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Semantic search
        semantic_scores = []
        try:
            query_emb = get_embedding(query)
            for entry in kb_entries:
                if "embedding" in entry:
                    score = cosine_similarity(query_emb, entry["embedding"])
                    semantic_scores.append((score, entry))
            semantic_scores.sort(key=lambda x: x[0], reverse=True)
        except Exception as e:
            print(f"Error in semantic search: {e}")
            semantic_scores = []
        
        # Keyword search
        keyword_scores = []
        for entry in kb_entries:
            text_lower = entry.get("text", "").lower()
            text_words = set(text_lower.split())
            
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(text_words) if text_words else 0
                keyword_scores.append((score, entry))
        
        keyword_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Combine scores (weighted average)
        combined_scores = {}
        for score, entry in semantic_scores[:top_k*2]:
            chunk_id = entry.get("chunk_id", "")
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score * 0.7
        
        for score, entry in keyword_scores[:top_k*2]:
            chunk_id = entry.get("chunk_id", "")
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score * 0.3
        
        # Get top results
        all_entries = {entry.get("chunk_id", ""): entry for entry in kb_entries}
        top_entries = []
        for chunk_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if chunk_id in all_entries:
                entry = all_entries[chunk_id].copy()
                entry["combined_score"] = score
                top_entries.append(entry)
        
        return top_entries
    except Exception as e:
        print(f"Error in hybrid_search: {e}")
        return []
