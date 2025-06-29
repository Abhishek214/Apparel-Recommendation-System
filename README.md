# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
import httpx
import asyncio
import json
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Document Processing API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple security (optional)
security = HTTPBearer(auto_error=False)

# Configuration
RAG_SERVICE = "https://hsbc-multi-wcs-nonprod-use-chat-01.azurewebsites.net/rag"
DOCUMENT_UPLOADER = "https://hsbc-multi-wcs-nonprod-use-upload-01.azurewebsites.net/upload-files"
SESSION_TOKEN_URL = "https://hsbc-multi-wcs-nonprod-use-session-01.azurewebsites.net/session"
EXCHANGE_TOKEN_URL = "https://hsbc-multi-wcs-nonprod-use-session-01.azurewebsites.net/exchange"

# System prompts for different extraction types
SYSTEM_PROMPTS = {
    "key_value": """You are an intelligent document analysis system. Extract key information from the document context in JSON format. 
    Return the response as: {"field1": "value1", "field2": "value2", ...}
    Return "N/A" for any information not found.""",
    
    "table": """You are an intelligent document analysis system. Extract table information and return in JSON format.
    Return the response as: {"table_name": [["header1", "header2"], ["row1_col1", "row1_col2"], ...]}""",
    
    "default": "You are an intelligent document analysis assistant. Extract the requested information from the document."
}

# In-memory storage for demo purposes
uploaded_documents = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Simple Document Processing API"}

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    category: str = Form(default="general"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Upload documents for processing"""
    try:
        uploaded_files = []
        
        # Process each uploaded file
        for file in files:
            # Read file content
            content = await file.read()
            
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Store file info (in real app, you'd upload to external service)
            uploaded_documents[doc_id] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "upload_time": datetime.now().isoformat(),
                "category": category
            }
            
            # Simulate external document upload
            files_data = [("files", (file.filename, content, file.content_type))]
            
            try:
                async with httpx.AsyncClient() as client:
                    upload_response = await client.post(
                        DOCUMENT_UPLOADER,
                        files=files_data,
                        data={
                            "business_line": "General",
                            "classification": category,
                            "user_id": "demo_user"
                        },
                        timeout=30.0
                    )
                
                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    if "successful_upload_document_messages" in upload_result:
                        for uploaded_doc in upload_result["successful_upload_document_messages"]:
                            uploaded_documents[uploaded_doc["doc_id"]] = {
                                **uploaded_documents[doc_id],
                                "external_doc_id": uploaded_doc["doc_id"],
                                "title": uploaded_doc.get("title", file.filename)
                            }
                            uploaded_files.append({
                                "doc_id": uploaded_doc["doc_id"],
                                "filename": file.filename,
                                "status": "uploaded"
                            })
                else:
                    # Fallback for demo
                    uploaded_files.append({
                        "doc_id": doc_id,
                        "filename": file.filename,
                        "status": "local_only"
                    })
            except Exception as e:
                logger.warning(f"External upload failed: {e}")
                # Fallback for demo
                uploaded_files.append({
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "status": "local_only"
                })
        
        return JSONResponse({
            "status": "success",
            "uploaded_files": uploaded_files
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """List all uploaded documents"""
    documents = []
    for doc_id, doc_info in uploaded_documents.items():
        documents.append({
            "doc_id": doc_id,
            "filename": doc_info.get("filename"),
            "upload_time": doc_info.get("upload_time"),
            "category": doc_info.get("category"),
            "status": "available"
        })
    
    return {"documents": documents}

async def get_session_token():
    """Get session token for RAG service"""
    try:
        session_data = {
            "name": "demo_user",
            "user_email": "demo@example.com",
            "app_id": "demo-app",
            "task": "Conversation",
            "settings": {"model": "GPT 3.5"}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SESSION_TOKEN_URL,
                json=session_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json().get("session_id")
    except Exception as e:
        logger.warning(f"Session token failed: {e}")
    
    return "demo_session"

async def get_exchange_token(session_id):
    """Get exchange token for RAG service"""
    try:
        exchange_data = {
            "session_id": session_id,
            "question": "Demo question",
            "answer": "Demo answer",
            "app_id": "demo-app"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EXCHANGE_TOKEN_URL,
                json=exchange_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json().get("exchange_id")
    except Exception as e:
        logger.warning(f"Exchange token failed: {e}")
    
    return "demo_exchange"

async def query_rag(client, doc_id, question, system_prompt, model_type="gpt-4o-mini", temperature=0.4):
    """Query RAG service for document analysis"""
    try:
        # Get tokens
        session_id = await get_session_token()
        exchange_id = await get_exchange_token(session_id)
        
        rag_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "model": model_type,
            "temperature": temperature,
            "vector_store_id": "stitt-embeddings",
            "search_text": question,
            "filter_query": f"parent_document_id eq '{doc_id}'",
            "stream": False,
            "entity_id": exchange_id,
            "entity_type": "exchange",
            "app_id": "demo-app"
        }
        
        response = await client.post(
            RAG_SERVICE,
            json=rag_data,
            timeout=60.0
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return f"Error querying document: {str(e)}"

@app.post("/extract")
async def extract_information(
    doc_ids: List[str] = Form(...),
    question: str = Form(...),
    extraction_type: str = Form(default="default"),
    model_type: str = Form(default="gpt-4o-mini"),
    temperature: float = Form(default=0.4),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Extract information from documents using AI"""
    try:
        # Handle comma-separated doc_ids from form
        if len(doc_ids) == 1 and ',' in doc_ids[0]:
            doc_ids = [doc_id.strip() for doc_id in doc_ids[0].split(',')]
        
        # Get system prompt based on extraction type
        system_prompt = SYSTEM_PROMPTS.get(extraction_type, SYSTEM_PROMPTS["default"])
        
        # Process each document
        results = []
        async with httpx.AsyncClient() as client:
            tasks = []
            for doc_id in doc_ids:
                if doc_id in uploaded_documents or doc_id.startswith("doc_"):
                    tasks.append(query_rag(client, doc_id, question, system_prompt, model_type, temperature))
                else:
                    logger.warning(f"Document {doc_id} not found")
            
            if tasks:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, (doc_id, response) in enumerate(zip(doc_ids, responses)):
                    if isinstance(response, Exception):
                        response = f"Error processing document: {str(response)}"
                    
                    # Try to parse JSON response for structured extraction
                    parsed_response = response
                    if extraction_type in ["key_value", "table"] and isinstance(response, str):
                        try:
                            # Look for JSON in response
                            start = response.find('{')
                            end = response.rfind('}') + 1
                            if start >= 0 and end > start:
                                json_str = response[start:end]
                                parsed_response = json.loads(json_str)
                        except:
                            # Keep original response if JSON parsing fails
                            pass
                    
                    doc_info = uploaded_documents.get(doc_id, {})
                    results.append({
                        "doc_id": doc_id,
                        "filename": doc_info.get("filename", "Unknown"),
                        "question": question,
                        "response": parsed_response,
                        "extraction_type": extraction_type,
                        "model_used": model_type
                    })
        
        return JSONResponse({
            "status": "success",
            "extraction_results": results,
            "total_documents": len(results)
        })
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/bulk_extract")
async def bulk_extract(
    extraction_config: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Bulk extraction with multiple questions and document sets
    
    Expected format:
    {
        "extractions": [
            {
                "doc_ids": ["doc1", "doc2"],
                "question": "What is the company name?",
                "extraction_type": "key_value"
            },
            {
                "doc_ids": ["doc1"],
                "question": "Extract the financial table",
                "extraction_type": "table"
            }
        ]
    }
    """
    try:
        extractions = extraction_config.get("extractions", [])
        all_results = []
        
        for extraction in extractions:
            doc_ids = extraction.get("doc_ids", [])
            question = extraction.get("question", "")
            extraction_type = extraction.get("extraction_type", "default")
            model_type = extraction.get("model_type", "gpt-4o-mini")
            temperature = extraction.get("temperature", 0.4)
            
            if doc_ids and question:
                # Use the existing extract function logic
                system_prompt = SYSTEM_PROMPTS.get(extraction_type, SYSTEM_PROMPTS["default"])
                
                async with httpx.AsyncClient() as client:
                    tasks = [query_rag(client, doc_id, question, system_prompt, model_type, temperature) 
                            for doc_id in doc_ids if doc_id in uploaded_documents or doc_id.startswith("doc_")]
                    
                    if tasks:
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for doc_id, response in zip(doc_ids, responses):
                            if isinstance(response, Exception):
                                response = f"Error: {str(response)}"
                            
                            doc_info = uploaded_documents.get(doc_id, {})
                            all_results.append({
                                "doc_id": doc_id,
                                "filename": doc_info.get("filename", "Unknown"),
                                "question": question,
                                "response": response,
                                "extraction_type": extraction_type
                            })
        
        return JSONResponse({
            "status": "success",
            "bulk_extraction_results": all_results,
            "total_extractions": len(all_results)
        })
        
    except Exception as e:
        logger.error(f"Bulk extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk extraction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Simple Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload documents",
            "documents": "GET /documents - List uploaded documents", 
            "extract": "POST /extract - Extract information from documents",
            "bulk_extract": "POST /bulk_extract - Bulk extraction operations",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# ============================================================================
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
httpx==0.25.2

# ============================================================================
# README.md

# Simple Document Processing API

A lightweight FastAPI application for document upload and AI-powered information extraction.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📋 API Endpoints

### 1. Upload Documents
```bash
POST /upload
```
Upload one or more documents for processing.

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf" \
  -F "category=financial"
```

### 2. List Documents
```bash
GET /documents
```
Get list of all uploaded documents.

### 3. Extract Information
```bash
POST /extract
```
Extract information from documents using AI.

**Parameters:**
- `doc_ids`: List of document IDs (comma-separated)
- `question`: What information to extract
- `extraction_type`: Type of extraction (`key_value`, `table`, `default`)
- `model_type`: AI model to use (`gpt-4o-mini`, `gpt-3.5-turbo`)
- `temperature`: AI response creativity (0.0-1.0)

**Example:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -F "doc_ids=doc_123" \
  -F "question=What is the company name and revenue?" \
  -F "extraction_type=key_value"
```

### 4. Bulk Extract
```bash
POST /bulk_extract
```
Perform multiple extractions with different questions and document sets.

**Example:**
```bash
curl -X POST "http://localhost:8000/bulk_extract" \
  -H "Content-Type: application/json" \
  -d '{
    "extractions": [
      {
        "doc_ids": ["doc_123"],
        "question": "Extract company information",
        "extraction_type": "key_value"
      },
      {
        "doc_ids": ["doc_123"],
        "question": "Extract financial table",
        "extraction_type": "table"
      }
    ]
  }'
```

### 5. Health Check
```bash
GET /health
```
Check if the service is running.

## 🔧 Extraction Types

### `key_value`
Extracts key-value pairs in JSON format:
```json
{
  "company_name": "Example Corp",
  "revenue": "$1.2M",
  "year": "2023"
}
```

### `table`
Extracts tabular data:
```json
{
  "financial_data": [
    ["Year", "Revenue", "Profit"],
    ["2021", "$1.0M", "$100K"],
    ["2022", "$1.1M", "$120K"]
  ]
}
```

### `default`
General text extraction for any question.

## 🏗️ Architecture

This is a simplified, stateless application that:

- **No Database**: Uses in-memory storage for demo purposes
- **No File Persistence**: Documents processed through external services
- **No Complex Auth**: Optional simple bearer token
- **No Project Management**: Direct document-to-extraction workflow

## 🔄 Workflow

1. **Upload** documents via `/upload` endpoint
2. **List** available documents with `/documents`
3. **Extract** information using `/extract` with specific questions
4. **Process** multiple extractions with `/bulk_extract`

## ⚙️ Configuration

Key settings in `main.py`:
- `RAG_SERVICE`: AI processing service URL
- `DOCUMENT_UPLOADER`: Document upload service URL
- `SESSION_TOKEN_URL`: Authentication service URL

## 🧪 Testing

### Test with Sample Document
```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "files=@sample.pdf"

# 2. Extract information (use doc_id from upload response)
curl -X POST "http://localhost:8000/extract" \
  -F "doc_ids=YOUR_DOC_ID" \
  -F "question=What is this document about?" \
  -F "extraction_type=default"
```

## 🚀 Production Considerations

For production use, consider adding:
- **Database**: For persistent document metadata
- **File Storage**: AWS S3, Azure Blob, etc.
- **Authentication**: Proper user management
- **Rate Limiting**: API usage controls
- **Monitoring**: Logging and metrics
- **Error Handling**: Comprehensive error responses

## 🎯 Use Cases

- **Document Analysis**: Extract key information from reports
- **Data Mining**: Pull structured data from unstructured documents  
- **Content Summarization**: Generate summaries and insights
- **Form Processing**: Extract form field values
- **Research**: Analyze academic papers or research documents

This simple API provides a clean interface for AI-powered document processing without the complexity of enterprise features.
