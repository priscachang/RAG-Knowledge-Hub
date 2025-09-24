# RAG Knowledge Hub

RAG Knowledge Hub is a sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent document analysis and question-answering. Built with FastAPI and Mistral AI, it provides enterprise-grade document intelligence with customizable search parameters, intent detection, and evidence-based responses.

<p align="center">
<img width="800" height="777" alt="截圖 2025-09-24 凌晨1 12 46" src="https://github.com/user-attachments/assets/9070ba55-5b55-46c1-9fcd-7e65e096dea2" />
</p>

## System Architecture

```

                   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                   │ Frontend UI     │    │ FastAPI App     │    │ Services        │
                   │                 │    │                 │    │ • Text Extract  │
                   │ • Chat Interface│◄──►│ • Routes        │◄──►│ • Intent Detect │
                   │ • File Upload   │    │ • Models        │    │ • Search Engine │
                   │ • Parameters    │    │ • Validation    │    │ • LLM Service   │
                   │ • Markdown UI   │    │                 │    │ • Security      │
                   └─────────────────┘    └─────────────────┘    └─────────────────┘    
                                                  │               
                                                  ▼
                                          ┌─────────────────┐
                                          │ Knowledge Base  │
                                          │                 │
                                          │ • JSON Storage  │
                                          │ • Embeddings    │
                                          │ • Metadata      │
                                          └─────────────────┘

```

## 🔧 System Design

### Core Components

#### 1. **Document Ingestion Pipeline**

- **PDF Text Extraction**: Uses `pdfplumber` for regular PDFs
- **OCR Fallback**: Uses `pytesseract` for scanned documents
- **Smart Chunking**: Respects sentence boundaries with overlap
- **Embedding Generation**: Creates vector representations using Mistral AI

#### 2. **Query Processing Engine**

- **Intent Detection**: LLM-powered classification (greeting, question, list, summary, finish)
- **Query Enhancement**: Context-aware query transformation
- **Hybrid Search**: Combines semantic and keyword search
- **Evidence Checking**: Validates response against source material

#### 3. **Response Generation**

- **Template-based Prompts**: Intent-specific prompt templates
- **LLM Integration**: Mistral AI for answer generation
- **Citation Support**: Source tracking and references
- **Confidence Scoring**: Reliability metrics for responses

### Data Flow

```

PDF Upload → Text Extraction → Chunking → Embedding → Knowledge Base
│
User Query → Intent Detection → Search → Context → LLM → Response

```

## 📁 Project Structure

```

RAG_Knowledge_Hub/
├── app.py                  # Main FastAPI application (routes, endpoints)
├── models.py          
├── utils.py           
├── requirements.txt   
├── .env.example               
├── services/           
│   ├── text_extraction.py   # PDF text + OCR extraction
│   ├── intent_detection.py  # Query intent classification
│   ├── search_service.py    # Hybrid semantic + keyword search
│   ├── llm_service.py       # LLM interactions (Mistral API)
│   ├── security_service.py  # Security checks and validation
└── README.md 

```

## 🚀 Features

### Core Features

- **Multi-format PDF Support**: Handles both text and scanned PDFs
- **Intelligent Chunking**: Semantic boundary-aware text splitting
- **Hybrid Search**: Combines semantic and keyword matching
- **Intent Detection**: LLM-powered query classification
- **Evidence Validation**: Response verification against sources

### Advanced Features

- **Security Checks**: PII detection and sensitive content filtering
- **Confidence Scoring**: Reliability metrics for responses
- **Citation Tracking**: Source attribution and references
- **Parameter Tuning**: Adjustable search parameters
- **Real-time Processing**: Streaming responses with loading indicators

### UI Features

- **Modern Dark Theme**: Professional, enterprise-ready interface
- **Drag & Drop Upload**: Intuitive file handling
- **Parameter Controls**: Real-time search parameter adjustment
- **Markdown Rendering**: Rich text response formatting
- **Responsive Design**: Mobile-friendly interface

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Mistral AI API key

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-knowledge-hub.git
cd rag-knowledge-hub
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Run the Application

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application

Open your browser and navigate to: `http://localhost:8000`

## 📊 API Endpoints

### Core Endpoints

| Endpoint  | Method | Description                  |
| --------- | ------ | ---------------------------- |
| `/`       | GET    | Main chat interface          |
| `/ingest` | POST   | Upload and process PDF files |
| `/query`  | POST   | Query the knowledge base     |

### Request/Response Examples

#### Upload Documents

```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "files=@document1.pdf" \
     -F "files=@document2.pdf"
```

#### Query System

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the main topic of the document?",
       "top_k": 5,
       "threshold": 0.6,
       "use_hybrid": true
     }'
```

## 🔍 Usage Guide

### 1. Upload Documents

- Drag and drop PDF files into the upload area
- Or click "Browse Files" to select files
- Click "Upload & Process" to ingest documents

### 2. Adjust Search Parameters

- **Top K Results**: Number of chunks to retrieve (1-10)
- **Similarity Threshold**: Minimum similarity score (0.1-1.0)
- **Hybrid Search**: Enable/disable hybrid search mode

### 3. Ask Questions

- Type your question in the chat input
- The system will automatically detect intent
- View confidence scores and citations
- Responses support Markdown formatting

## Intent Detection

The system automatically classifies queries into categories:

| Intent         | Description          | Example                                         |
| -------------- | -------------------- | ----------------------------------------------- |
| `greeting`     | Simple greetings     | "Hello", "Hi there"                             |
| `question`     | Information requests | "What is RAG?", "How does it work?"             |
| `list_request` | List/table requests  | "Show me all topics", "List everything"         |
| `summary`      | Summary requests     | "Summarize the document", "Give me an overview" |
| `finish`       | Goodbye messages     | "Thank you", "Bye", "That's all"                |
| `general`      | Other queries        | General information requests                    |

## 🔒 Security Features

### Content Filtering

- **PII Detection**: Automatically detects and blocks personal information
- **Sensitive Content**: Filters legal/medical advice requests
- **Evidence Validation**: Ensures responses are supported by source material

### Response Quality

- **Confidence Scoring**: Measures response reliability
- **Evidence Checking**: Validates claims against source documents
- **Citation Tracking**: Provides source attribution

## Performance Metrics

The system provides real-time metrics:

- **Confidence Score**: Response reliability (0-100%)
- **Evidence Score**: Source support level (0-100%)
- **Processing Time**: Query response time
- **Citation Count**: Number of source references

## 🛡️ Error Handling

### Robust Error Management

- **Graceful Degradation**: Fallback mechanisms for API failures
- **Input Validation**: Comprehensive request validation
- **Error Messages**: User-friendly error descriptions
- **Logging**: Detailed system logging for debugging

### Common Issues

- **Empty Knowledge Base**: Upload documents before querying
- **API Key Issues**: Ensure MISTRAL_API_KEY is set correctly
- **File Format**: Only PDF files are supported
- **Memory Usage**: Large documents may require more memory

## Configuration

### Environment Variables

```bash
MISTRAL_API_KEY=your_api_key_here    # Required: Mistral AI API key
```

### Search Parameters

- **Chunk Size**: 500 characters (configurable)
- **Overlap**: 100 characters (configurable)
- **Embedding Model**: mistral-embed
- **LLM Model**: mistral-small-latest

## Acknowledgments

- **Mistral AI** for the embedding and language models
- **FastAPI** for the web framework
- **pdfplumber** and **pytesseract** for PDF processing

---
