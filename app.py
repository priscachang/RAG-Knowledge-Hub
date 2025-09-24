import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from models import QueryRequest, QueryResponse, IngestionResponse
from services.text_extraction import extract_text_from_pdf, extract_text_with_ocr
from services.intent_detection import detect_query_intent, enhance_query
from services.search_service import hybrid_search, cosine_similarity
from services.llm_service import get_prompt_template
from services.security_service import check_sensitive_content, check_evidence
from utils import clean_text, smart_chunk_text, get_embedding
from mistralai import Mistral

app = FastAPI(title="RAG Knowledge Hub", description="Enterprise-grade document intelligence")

# Load environment variables
load_dotenv()

# Configuration
KB_FILE = "knowledge_base.json"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is required")

llm_client = Mistral(api_key=MISTRAL_API_KEY)

# Initialize knowledge base
if not os.path.exists(KB_FILE):
    with open(KB_FILE, "w") as f:
        json.dump([], f)

# Ingest PDFs
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """Enhanced PDF ingestion with better error handling and metadata."""
    kb_entries = []
    processed_files = []
    
    for file in files:
        try:
            file.file.seek(0)
            
            # Try pdfplumber
            text_data = extract_text_from_pdf(file.file)
            text = text_data["text"]
            metadata = text_data["metadata"]
            
            print(f"[DEBUG] Extracted text length from {file.filename}: {len(text)}")
            
            # check for OCR text (additional supplement)
            file.file.seek(0)
            ocr_data = extract_text_with_ocr(file.file.read())
            
            # Combine both sources
            if ocr_data["text"].strip():
                text += "\n" + ocr_data["text"]
                metadata["ocr_extracted"] = True
                print(f"[DEBUG] OCR added {len(ocr_data['text'])} characters from {file.filename}")
            else:
                metadata["ocr_extracted"] = False
                print(f"[DEBUG] No OCR text found in {file.filename}")
            
            if len(text.strip()) == 0:
                print(f"[WARNING] No text extracted from {file.filename}")
                continue
            
            # Clean and chunk
            text = clean_text(text)
            chunks = smart_chunk_text(text)
            
            # Generate embeddings
            for chunk_data in chunks:
                try:
                    embedding = get_embedding(chunk_data["text"])
                    entry = {
                        "file_name": file.filename,
                        "chunk_id": f"{file.filename}_{chunk_data['chunk_id']}",
                        "text": chunk_data["text"],
                        "embedding": embedding,
                        "metadata": {
                            **metadata,
                            "word_count": chunk_data["word_count"],
                            "char_count": chunk_data["char_count"]
                        }
                    }
                    kb_entries.append(entry)
                except Exception as e:
                    print(f"[ERROR] Failed to generate embedding for chunk: {str(e)}")
                    continue
            
            processed_files.append(file.filename)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file.filename}: {str(e)}")
            continue
    
    if not kb_entries:
        raise HTTPException(status_code=400, detail="No valid content extracted from uploaded files")
    
    # Append new entries into the local knowledge base
    with open(KB_FILE, "r+") as f:
        existing = json.load(f)
        existing.extend(kb_entries)
        f.seek(0)
        json.dump(existing, f, indent=2)
    
    return IngestionResponse(
        status="success",
        ingested_chunks=len(kb_entries),
        files_processed=processed_files,
        total_chunks=len(existing)
    )

# Query system
@app.post("/query", response_model=QueryResponse)
async def query_system(query_request: QueryRequest):
    """Enhanced query processing with intent detection, hybrid search, and evidence checking."""
    import time
    start_time = time.time()
    
    try:
        # Security check
        security_check = check_sensitive_content(query_request.query)
        if security_check["should_refuse"]:
            return QueryResponse(
                answer="I cannot process this request as it may contain sensitive information or requests for legal/medical advice. Please consult appropriate professionals.",
                citations=[],
                confidence=0.0,
                evidence_score=0.0,
                query_type="refused",
                processing_time=time.time() - start_time
            )
        
        # Intent detection
        intent = detect_query_intent(query_request.query, llm_client)
        # Handle greetings
        if intent == "greeting":
            return QueryResponse(
                answer="Hello! I'm here to help you find information from your uploaded documents. What would you like to know?",
                citations=[],
                confidence=1.0,
                evidence_score=1.0,
                query_type="greeting",
                processing_time=time.time() - start_time
            )
        
        # Handle finish intent
        if intent == "finish":
            return QueryResponse(
                answer="Thank you for using RAG Knowledge Hub! I'm glad I could help you find the information you needed. Have a great day and feel free to come back anytime! 👋",
                citations=[],
                confidence=1.0,
                evidence_score=1.0,
                query_type="finish",
                processing_time=time.time() - start_time
            )
        
        # Query enhancement
        enhanced_query = enhance_query(query_request.query, intent)

        # Get embedding for query
        try:
            query_emb = get_embedding(enhanced_query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

        # Load knowledge base
        if not os.path.exists(KB_FILE):
            raise HTTPException(status_code=400, detail="Knowledge base is empty. Please ingest PDFs first.")
        
        with open(KB_FILE, "r") as f:
            kb_entries = json.load(f)
        
        if not kb_entries:
            raise HTTPException(status_code=400, detail="Knowledge base is empty. Please ingest PDFs first.")
        
        # Hybrid search
        if query_request.use_hybrid:
            top_chunks = hybrid_search(enhanced_query, kb_entries, query_request.top_k)
        else:
            # Pure semantic search
            scored_chunks = []
            for entry in kb_entries:
                if "embedding" in entry:
                    score = cosine_similarity(query_emb, entry["embedding"])
                    scored_chunks.append((score, entry))
            
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [entry for score, entry in scored_chunks[:query_request.top_k] if score >= query_request.threshold]
        
        if not top_chunks:
            return QueryResponse(
                answer="I couldn't find sufficient evidence in the knowledge base to answer your question. Please try rephrasing or upload more relevant documents.",
                citations=[],
                confidence=0.0,
                evidence_score=0.0,
                query_type=intent,
                processing_time=time.time() - start_time
            )
        
        # Build context
        context = "\n\n".join([chunk.get("text", "") for chunk in top_chunks])
        citations = [chunk.get("chunk_id", "") for chunk in top_chunks]
        
        # Get appropriate prompt template
        prompt = get_prompt_template(intent, context, query_request.query)
        print(f"Prompt: {prompt}")
        # Call Mistral for answer generation
        try:
            response = llm_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            print(f"Response: {response}")
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")
        
        # Evidence checking
        evidence_check = check_evidence(answer, [chunk.get("text", "") for chunk in top_chunks])
        
        # Calculate confidence based on evidence score and similarity scores
        avg_similarity = sum(chunk.get("combined_score", 0.5) for chunk in top_chunks) / len(top_chunks)
        confidence = (evidence_check["evidence_score"] + avg_similarity) / 2
        
        # Add disclaimer if evidence is low
        if evidence_check["evidence_score"] < 0.3:
            answer += "\n\n Note: This answer may not be fully supported by the available evidence."
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            evidence_score=evidence_check["evidence_score"],
            query_type=intent,
            processing_time=time.time() - start_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in query_system: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# UI endpoint
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve a beautiful and user-friendly chat UI with Markdown support."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Knowledge Hub</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                min-height: 100vh;
                padding: 20px;
                color: #ffffff;
            }
            
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: #2c2c2c;
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: calc(100vh - 40px);
            }
            
            .header { 
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                color: white; 
                padding: 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
                border-bottom: 3px solid #6c63ff;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(108, 99, 255, 0.1) 0%, transparent 70%);
                animation: float 6s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }
            
            .header h1 { 
                font-size: 2.5em; 
                margin-bottom: 10px;
                position: relative;
                z-index: 1;
                color: #ffffff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p { 
                font-size: 1.1em; 
                opacity: 0.9;
                position: relative;
                z-index: 1;
                color: #e0e0e0;
                font-weight: 300;
            }
            
            .main-content {
                display: flex;
                flex: 1;
                overflow: hidden;
            }
            
            .sidebar {
                width: 300px;
                background: #1e1e1e;
                border-right: 1px solid #404040;
                padding: 20px;
                overflow-y: auto;
            }
            
            .chat-area {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            
            .upload-section {
                background: #2c2c2c;
                border-radius: 15px;
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                border: 2px dashed #404040;
                transition: all 0.3s ease;
            }
            
            .upload-section:hover {
                border-color: #6c63ff;
                transform: translateY(-2px);
            }
            
            .upload-section h3 {
                color: #ffffff;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .upload-box {
                width: 100%;
                padding: 12px;
                border: 2px dashed #6c63ff;
                border-radius: 6px;
                background: #1e1e1e;
                text-align: center;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3);
                transition: all 0.3s ease-in-out;
                cursor: pointer;
                margin: 8px 0;
            }
            
            .upload-box:hover {
                border-color: #4e49c9;
                background: #2c2c2c;
                transform: translateY(-1px);
            }
            
            .upload-icon {
                font-size: 24px;
                color: #6c63ff;
                margin-bottom: 8px;
            }
            
            .upload-text {
                font-size: 12px;
                color: #ffffff;
                margin-bottom: 10px;
                font-weight: 500;
            }
            
            .browse-btn {
                display: inline-block;
                padding: 6px 12px;
                background: #6c63ff;
                color: white;
                border-radius: 4px;
                cursor: pointer;
                text-decoration: none;
                font-size: 11px;
                transition: background 0.3s ease;
                border: none;
            }
            
            .browse-btn:hover {
                background: #4e49c9;
                transform: translateY(-1px);
            }
            
            .upload-btn {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #27ae60, #2ecc71);
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                margin-top: 10px;
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
            }
            
            .parameters-section { 
                background: #2c2c2c;
                border-radius: 15px;
                padding: 20px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .parameters-section h3 {
                color: #ffffff;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .parameter-group { 
                margin-bottom: 20px;
            }
            
            .parameter-group label { 
                display: block;
                font-weight: 600; 
                color: #ffffff; 
                margin-bottom: 8px;
                font-size: 0.9em;
            }
            
            .slider-container {
                position: relative;
                margin: 10px 0;
            }
            
            .slider-container input[type="range"] { 
                width: 100%; 
                height: 6px;
                border-radius: 3px;
                background: #e9ecef;
                outline: none;
                -webkit-appearance: none;
            }
            
            .slider-container input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: linear-gradient(135deg, #3498db, #2980b9);
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            }
            
            .parameter-value { 
                font-size: 0.85em; 
                color: #cccccc;
                font-weight: 500;
                margin-top: 5px;
            }
            
            .toggle-section { 
                display: flex; 
                align-items: center; 
                gap: 12px; 
                margin-top: 20px;
                padding: 15px;
                background: #1e1e1e;
                border-radius: 10px;
            }
            
            .toggle-section input[type="checkbox"] { 
                transform: scale(1.3);
                accent-color: #3498db;
            }
            
            .toggle-section label { 
                font-weight: 500; 
                color: #ffffff;
                cursor: pointer;
            }
            
            .chat-container { 
                flex: 1;
                overflow-y: auto; 
                padding: 20px;
                background: #1a1a1a;
                scroll-behavior: smooth;
            }
            
            .message { 
                margin: 15px 0; 
                padding: 20px; 
                border-radius: 20px;
                max-width: 80%;
                word-wrap: break-word;
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user-message { 
                background: linear-gradient(135deg, #6c63ff, #4e49c9);
                color: white; 
                margin-left: auto;
                border-bottom-right-radius: 5px;
            }
            
            .bot-message { 
                background: #ffffff;
                color: #2c3e50;
                margin-right: auto;
                border-bottom-left-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 3px solid #6c63ff;
            }
            
            .system-message {
                background: linear-gradient(135deg, #404040, #2c2c2c);
                color: #e0e0e0;
                margin: 10px auto;
                max-width: 90%;
                text-align: center;
                font-size: 0.9em;
                border: 1px solid #555555;
            }
            
            .input-container { 
                padding: 20px; 
                background: #2c2c2c;
                border-top: 1px solid #404040;
            }
            
            .input-row { 
                display: flex; 
                gap: 15px; 
                align-items: center;
            }
            
            .input-row input { 
                flex: 1; 
                padding: 15px 20px; 
                border: 2px solid #404040;
                border-radius: 25px;
                font-size: 1em;
                outline: none;
                transition: all 0.3s ease;
                background: #1e1e1e;
                color: #ffffff;
            }
            
            .input-row input:focus {
                border-color: #6c63ff;
                box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1);
            }
            
            .input-row input::placeholder {
                color: #888888;
            }
            
            .send-btn {
                padding: 15px 25px;
                background: linear-gradient(135deg, #6c63ff, #4e49c9);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .send-btn:hover {
                background: linear-gradient(135deg, #4e49c9, #3d3a9e);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(108, 99, 255, 0.4);
            }
            
            .refresh-btn {
                padding: 15px;
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                margin-right: 10px;
            }
            
            .refresh-btn:hover {
                background: linear-gradient(135deg, #34495e, #2c3e50);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(44, 62, 80, 0.4);
            }
            
            .citations { 
                font-size: 0.85em; 
                color: #6c757d; 
                margin-top: 15px; 
                padding: 12px; 
                background: #f8f9fa; 
                border-radius: 10px;
                border-left: 4px solid #3498db;
            }
            
            .metrics {
                font-size: 0.8em;
                color: #6c757d;
                margin-top: 10px;
                padding: 10px;
                background: #e8f4fd;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }
            
            .metric-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .markdown-content h1, .markdown-content h2, .markdown-content h3 { 
                color: #2c3e50; 
                margin: 20px 0 15px 0;
                font-weight: 600;
            }
            
            .markdown-content p { 
                margin: 15px 0; 
                line-height: 1.7;
                color: #2c3e50;
            }
            
            .markdown-content strong { 
                color: #2c3e50; 
                font-weight: 600;
            }
            
            .markdown-content code { 
                background: #f1f2f6; 
                padding: 3px 6px; 
                border-radius: 4px; 
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #e74c3c;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .main-content {
                    flex-direction: column;
                }
                
                .sidebar {
                    width: 100%;
                    order: 2;
                }
                
                .chat-area {
                    order: 1;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .message {
                    max-width: 95%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-brain"></i> RAG Knowledge Hub</h1>
                <p>Enterprise-grade document intelligence with adaptive search and reliable AI-generated responses</p>
            </div>
            
            <div class="main-content">
                <div class="sidebar">
                    <div class="upload-section">
                        <h3><i class="fas fa-upload"></i> Upload Documents</h3>
                        <div class="upload-box" id="uploadArea">
                            <div class="upload-icon">☁️</div>
                            <div class="upload-text">Drag & Drop your PDF files here</div>
                            <label class="browse-btn" for="fileInput">Browse Files</label>
                            <input type="file" id="fileInput" multiple accept=".pdf" style="display: none;">
                        </div>
                        <button class="upload-btn" onclick="uploadFiles()" id="uploadBtn" disabled>
                            <i class="fas fa-cloud-upload-alt"></i> Upload & Process
                        </button>
                        <div id="uploadStatus"></div>
                    </div>
                    
                    <div class="parameters-section">
                        <h3><i class="fas fa-cogs"></i> Search Parameters</h3>
                        
                        <div class="parameter-group">
                            <label for="topK"><i class="fas fa-list-ol"></i> Top K Results</label>
                            <div class="slider-container">
                                <input type="range" id="topK" min="1" max="10" value="5" oninput="updateTopK()">
                            </div>
                            <div class="parameter-value">Value: <span id="topKValue">5</span></div>
                        </div>
                        
                        <div class="parameter-group">
                            <label for="threshold"><i class="fas fa-bullseye"></i> Similarity Threshold</label>
                            <div class="slider-container">
                                <input type="range" id="threshold" min="0.1" max="1.0" step="0.1" value="0.6" oninput="updateThreshold()">
                            </div>
                            <div class="parameter-value">Value: <span id="thresholdValue">0.6</span></div>
                        </div>
                        
                        <div class="toggle-section">
                            <input type="checkbox" id="useHybrid" checked>
                            <label for="useHybrid">
                                <i class="fas fa-search-plus"></i> Hybrid Search
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="chat-area">
                    <div class="chat-container" id="chatContainer">
                        <div class="message system-message">
                            <i class="fas fa-info-circle"></i> Welcome! Upload some PDF documents and start asking questions. Adjust the search parameters to fine-tune your results.
                        </div>
                    </div>
                    
                    <div class="input-container">
                        <div class="input-row">
                            <input type="text" id="messageInput" placeholder="Ask a question about your documents..." onkeypress="handleKeyPress(event)">
                            <button class="refresh-btn" onclick="refreshChat()" title="Clear chat history">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                            <button class="send-btn" onclick="sendMessage()">
                                <i class="fas fa-paper-plane"></i> Send
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Configure marked for better rendering
            marked.setOptions({
                breaks: true,
                gfm: true
            });
            
            // Parameter update functions
            function updateTopK() {
                const slider = document.getElementById('topK');
                const value = document.getElementById('topKValue');
                value.textContent = slider.value;
            }
            
            function updateThreshold() {
                const slider = document.getElementById('threshold');
                const value = document.getElementById('thresholdValue');
                value.textContent = slider.value;
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                const uploadBtn = document.getElementById('uploadBtn');
                const uploadStatus = document.getElementById('uploadStatus');
                
                let selectedFiles = [];
                
                // Click to select files
                uploadArea.addEventListener('click', () => {
                    fileInput.click();
                });
                
                // File input change
                fileInput.addEventListener('change', (e) => {
                    handleFiles(e.target.files);
                });
                
                // Drag and drop functionality
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
                
                uploadArea.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                });
                
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    handleFiles(e.dataTransfer.files);
                });
                
                function handleFiles(files) {
                    selectedFiles = Array.from(files).filter(file => file.type === 'application/pdf');
                    
                    if (selectedFiles.length > 0) {
                        uploadBtn.disabled = false;
                        displayFileList();
                    } else {
                        uploadBtn.disabled = true;
                        uploadStatus.innerHTML = '<div class="status-message status-error"><i class="fas fa-exclamation-triangle"></i> Please select PDF files only</div>';
                    }
                }
                
                function displayFileList() {
                    const fileListHtml = selectedFiles.map((file, index) => `
                        <div class="file-item">
                            <i class="fas fa-file-pdf"></i>
                            <span>${file.name}</span>
                            <button class="remove-file" onclick="removeFile(${index})">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    `).join('');
                    
                    uploadStatus.innerHTML = `
                        <div class="file-list">
                            <strong><i class="fas fa-check-circle"></i> Selected Files:</strong>
                            ${fileListHtml}
                        </div>
                    `;
                }
                
                // Make removeFile function global
                window.removeFile = function(index) {
                    selectedFiles.splice(index, 1);
                    if (selectedFiles.length > 0) {
                        displayFileList();
                    } else {
                        uploadBtn.disabled = true;
                        uploadStatus.innerHTML = '';
                    }
                };
                
                // Update uploadFiles function
                window.uploadFiles = async function() {
                    if (selectedFiles.length === 0) {
                        uploadStatus.innerHTML = '<div class="status-message status-error"><i class="fas fa-exclamation-triangle"></i> Please select files first</div>';
                        return;
                    }
                    
                    const formData = new FormData();
                    selectedFiles.forEach(file => {
                        formData.append('files', file);
                    });
                    
                    uploadStatus.innerHTML = '<div class="status-message status-info"><div class="loading"></div> Uploading and processing files...</div>';
                    
                    try {
                        const response = await fetch('/ingest', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            uploadStatus.innerHTML = `<div class="status-message status-success"><i class="fas fa-check-circle"></i> Successfully processed ${result.ingested_chunks} chunks from ${result.files_processed.length} files</div>`;
                            addMessage('System', `Successfully processed ${result.ingested_chunks} chunks from ${result.files_processed.length} files. You can now ask questions!`, 'system');
                            
                            // Clear selected files
                            selectedFiles = [];
                            uploadBtn.disabled = true;
                            fileInput.value = '';
                        } else {
                            uploadStatus.innerHTML = `<div class="status-message status-error"><i class="fas fa-times-circle"></i> Error: ${result.error}</div>`;
                        }
                    } catch (error) {
                        uploadStatus.innerHTML = `<div class="status-message status-error"><i class="fas fa-times-circle"></i> Upload failed: ${error.message}</div>`;
                    }
                };
            });
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Get current parameter values
                const topK = parseInt(document.getElementById('topK').value);
                const threshold = parseFloat(document.getElementById('threshold').value);
                const useHybrid = document.getElementById('useHybrid').checked;
                
                addMessage('You', message, 'user');
                
                // Show parameters used
                const paramsText = `Top K: ${topK} | Threshold: ${threshold} | Hybrid: ${useHybrid ? 'Yes' : 'No'}`;
                addMessage('System', `Searching with parameters: ${paramsText}`, 'system');
                
                input.value = '';
                
                // Show loading message
                const loadingId = 'loading-' + Date.now();
                addMessage('Assistant', '<div class="loading"></div> Processing your query...', 'bot');
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            query: message,
                            top_k: topK,
                            threshold: threshold,
                            use_hybrid: useHybrid
                        })
                    });
                    
                    const result = await response.json();
                    
                    // Remove loading message
                    const loadingMessage = document.getElementById(loadingId);
                    if (loadingMessage) {
                        loadingMessage.remove();
                    }
                    
                    if (response.ok) {
                        let citationsText = '';
                        if (result.citations && result.citations.length > 0) {
                            citationsText = `<div class="citations"><i class="fas fa-book"></i> <strong>Sources:</strong> ${result.citations.join(', ')}</div>`;
                        }
                        
                        // Show confidence and evidence scores
                        let metricsText = '';
                        if (result.confidence !== undefined && result.evidence_score !== undefined) {
                            metricsText = `<div class="metrics">
                                <div class="metric-item"><i class="fas fa-chart-line"></i> Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                                <div class="metric-item"><i class="fas fa-shield-alt"></i> Evidence: ${(result.evidence_score * 100).toFixed(1)}%</div>
                                <div class="metric-item"><i class="fas fa-clock"></i> Time: ${result.processing_time.toFixed(2)}s</div>
                            </div>`;
                        }
                        
                        // Render markdown content
                        const markdownContent = marked.parse(result.answer);
                        
                        addMessage('Assistant', markdownContent + citationsText + metricsText, 'bot', true);
                    } else {
                        addMessage('System', `<i class="fas fa-exclamation-triangle"></i> Error: ${result.error}`, 'system');
                    }
                } catch (error) {
                    addMessage('System', `<i class="fas fa-exclamation-triangle"></i> Error: ${error.message}`, 'system');
                }
            }
            
            function addMessage(sender, message, type, isMarkdown = false) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                if (isMarkdown) {
                    messageDiv.innerHTML = `<strong>${sender}:</strong> <div class="markdown-content">${message}</div>`;
                } else {
                    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
                }
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function refreshChat() {
                const chatContainer = document.getElementById('chatContainer');
                
                // Clear all messages except the welcome message
                chatContainer.innerHTML = `
                    <div class="message system-message">
                        <i class="fas fa-info-circle"></i> Welcome! Upload some PDF documents and start asking questions. Adjust the search parameters to fine-tune your results.
                    </div>
                `;
                
                // Clear the input field
                document.getElementById('messageInput').value = '';
                
                // Show a brief confirmation
                addMessage('System', '<i class="fas fa-check-circle"></i> Chat history cleared. Ready for new questions!', 'system');
            }
            
            // Initialize parameter displays
            updateTopK();
            updateThreshold();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
