import re
from typing import Dict, Any, List

def check_sensitive_content(query: str) -> Dict[str, Any]:
    """Check for sensitive content in queries."""
    try:
        query_lower = query.lower()
        
        # PII patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        ]
        
        pii_found = []
        for pattern in pii_patterns:
            if re.search(pattern, query):
                pii_found.append(pattern)
        
        # Legal/medical keywords
        sensitive_keywords = ["legal advice", "medical advice", "diagnosis", "treatment", "lawsuit", "court"]
        sensitive_found = [kw for kw in sensitive_keywords if kw in query_lower]
        
        return {
            "has_pii": len(pii_found) > 0,
            "has_sensitive": len(sensitive_found) > 0,
            "pii_patterns": pii_found,
            "sensitive_keywords": sensitive_found,
            "should_refuse": len(pii_found) > 0 or len(sensitive_found) > 0
        }
    except Exception as e:
        print(f"Error in check_sensitive_content: {e}")
        return {"has_pii": False, "has_sensitive": False, "pii_patterns": [], "sensitive_keywords": [], "should_refuse": False}

def check_evidence(answer: str, context_chunks: List[str]) -> Dict[str, Any]:
    """Check if answer is supported by evidence."""
    try:
        answer_sentences = re.split(r'[.!?]+', answer)
        context_text = " ".join(context_chunks).lower()
        
        unsupported_claims = []
        for sentence in answer_sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 10:
                continue
            
            sentence_words = set(sentence.split())
            if len(sentence_words) > 3:
                context_words = set(context_text.split())
                unique_words = sentence_words - context_words
                if len(unique_words) / len(sentence_words) > 0.5:
                    unsupported_claims.append(sentence)
        
        evidence_score = max(0, 1 - len(unsupported_claims) / len(answer_sentences)) if answer_sentences else 0
        
        return {
            "evidence_score": evidence_score,
            "unsupported_claims": unsupported_claims,
            "is_reliable": evidence_score > 0.7
        }
    except Exception as e:
        print(f"Error in check_evidence: {e}")
        return {"evidence_score": 0.0, "unsupported_claims": [], "is_reliable": False}
