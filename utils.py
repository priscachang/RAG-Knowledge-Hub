import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is required")

client = Mistral(api_key=MISTRAL_API_KEY)


def clean_text(text: str) -> str:
    # Remove excessive whitespace and line breaks
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """Enhanced chunking that respects sentence boundaries."""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": chunk_id,
                "word_count": len(current_chunk.split()),
                "char_count": len(current_chunk)
            })
            chunk_id += 1
            # Overlap handling
            overlap_text = " ".join(current_chunk.split()[-overlap//4:])
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_id": chunk_id,
            "word_count": len(current_chunk.split()),
            "char_count": len(current_chunk)
        })
    
    return chunks

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding
