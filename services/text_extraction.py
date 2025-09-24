import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from typing import Dict, Any

def extract_text_from_pdf(file) -> Dict[str, Any]:
    """Extract text from PDF with metadata."""
    text = ""
    metadata = {"pages": 0, "extraction_method": "pdfplumber"}
    
    try:
        with pdfplumber.open(file) as pdf:
            metadata["pages"] = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return {"text": "", "metadata": metadata}
    
    return {"text": text, "metadata": metadata}

def extract_text_with_ocr(file_bytes: bytes) -> Dict[str, Any]:
    """Extract text from scanned PDF using OCR."""
    try:
        images = convert_from_bytes(file_bytes)
        text = ""
        metadata = {"pages": len(images), "extraction_method": "ocr"}
        
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        
        return {"text": text, "metadata": metadata}
    except Exception as e:
        print(f"Error in OCR: {e}")
        return {"text": "", "metadata": {"pages": 0, "extraction_method": "ocr"}}
