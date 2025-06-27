#!/usr/bin/env python3
"""
Simple DC Automation with RAG - Document Upload and RAG Search
"""

import os
import json
import tempfile
from typing import Dict, List, Optional, Literal
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from simple_rag_client import SimpleRAGClient
from simple_utils import extract_pdf_text, extract_mt700_rules_rag, analyze_documents_rag

# Configuration
RAG_SERVICE_URL = "https://hsbc-multi-dcrest-nonprod01-chat-service.azurewebsites.net/rag"
DOCUMENT_UPLOADER_URL = "https://hsbc-multi-dcrest-nonprod01-ingest-service.azurewebsites.net/ingest"
INGEST_STATUS_URL = "https://hsbc-multi-dcrest-nonprod01-ingest-service.azurewebsites.net/ingest"
APP_ID = "lydia"

# Initialize RAG client
rag_client = SimpleRAGClient(
    rag_service_url=RAG_SERVICE_URL,
    document_uploader_url=DOCUMENT_UPLOADER_URL,
    ingest_status_url=INGEST_STATUS_URL,
    app_id=APP_ID
)

# FastAPI Application
app = FastAPI(
    title="DC Automation with RAG",
    description="Simple Documentary Credit automation with RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    azure_token: str = Form(...),
    multimodal: bool = Form(default=True),
    chunking_strategy: Literal["BY_PAGE", "BY_SECTION", "BY_PARAGRAPH", "SEMANTIC"] = Form(default="BY_PAGE")
):
    """Upload document and process with RAG indexing"""
    temp_file_path = None
    
    try:
        # Validate file type
        allowed_extensions = [".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"]
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Upload document
        result = await rag_client.upload_document(
            file_path=temp_file_path,
            file_name=file.filename,
            azure_token=azure_token,
            multimodal=multimodal,
            chunking_strategy=chunking_strategy
        )
        
        if not result or "documentID" not in result:
            raise HTTPException(status_code=500, detail="Document upload failed")
        
        return {
            "status": "success",
            "document_id": result["documentID"],
            "filename": file.filename,
            "chunking_strategy": chunking_strategy,
            "multimodal": multimodal,
            "message": "Document uploaded and indexed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/document-status/{document_id}")
async def get_document_status(
    document_id: str,
    azure_token: str = Header()
):
    """Check document processing status"""
    try:
        status = await rag_client.get_document_status(document_id, azure_token)
        return {
            "document_id": document_id,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/search-documents")
async def search_documents(
    query: str = Form(...),
    document_id: str = Form(...),
    azure_token: str = Form(...),
    max_documents: int = Form(default=10),
    multimodal: bool = Form(default=False),
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = Form(default="SEARCH")
):
    """Search documents using RAG"""
    try:
        result = await rag_client.search_documents(
            query=query,
            document_id=document_id,
            azure_token=azure_token,
            max_documents=max_documents,
            multimodal=multimodal,
            search_type=search_type
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Search failed")
        
        return {
            "status": "success",
            "query": query,
            "answer": result.get("response", ""),
            "documents_found": len(result.get("documents_used", [])),
            "document_content": result.get("document_content", ""),
            "documents_used": result.get("documents_used", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/extract-rules")
async def extract_rules_from_mt700(
    mt700_text: str = Form(...),
    document_id: str = Form(...),
    azure_token: str = Form(...),
    use_rag: bool = Form(default=True),
    custom_prompt: str = Form(default="")
):
    """Extract business rules from MT700 with RAG enhancement"""
    try:
        result = await extract_mt700_rules_rag(
            rag_client=rag_client,
            mt700_text=mt700_text,
            document_id=document_id,
            azure_token=azure_token,
            use_rag=use_rag,
            custom_prompt=custom_prompt if custom_prompt.strip() else None
        )
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        return {
            "status": "success",
            "extracted_fields": result.get("extracted_fields", {}),
            "business_rules": result.get("business_rules", []),
            "dc_metadata": result.get("dc_metadata", {}),
            "rules_count": len(result.get("business_rules", [])),
            "rag_used": use_rag
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rule extraction failed: {str(e)}")

@app.post("/analyze-documents")
async def analyze_documents(
    business_rules: str = Form(...),
    document_id: str = Form(...),
    azure_token: str = Form(...),
    custom_prompt: str = Form(default=""),
    multimodal: bool = Form(default=False),
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = Form(default="SEARCH")
):
    """Analyze documents against business rules using RAG"""
    try:
        # Parse business rules
        try:
            rules_list = json.loads(business_rules)
            if not isinstance(rules_list, list):
                raise ValueError("Business rules must be a list")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for business rules")
        
        result = await analyze_documents_rag(
            rag_client=rag_client,
            business_rules=rules_list,
            document_id=document_id,
            azure_token=azure_token,
            custom_prompt=custom_prompt if custom_prompt.strip() else None,
            multimodal=multimodal,
            search_type=search_type
        )
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        # Calculate simple metrics
        verification_results = result.get("verification_results", [])
        analysis_metrics = {}
        if verification_results:
            analysis_metrics = {
                "total_rules_checked": len(verification_results),
                "rules_passed": len([r for r in verification_results if r.get("verification_result") == "Passed"]),
                "rules_need_review": len([r for r in verification_results if r.get("verification_result") == "Need Review"])
            }
        
        return {
            "status": "success",
            "verification_results": verification_results,
            "overall_compliance": result.get("overall_compliance", "Unknown"),
            "discrepancies": result.get("discrepancies", []),
            "analysis_metrics": analysis_metrics,
            "rules_processed": len(rules_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@app.delete("/delete-document/{document_id}")
async def delete_document(
    document_id: str,
    azure_token: str = Header()
):
    """Delete uploaded document"""
    try:
        success = await rag_client.delete_document(document_id, azure_token)
        
        if success:
            return {"status": "deleted", "document_id": document_id}
        else:
            raise HTTPException(status_code=500, detail="Delete operation failed")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/chunking-strategies")
async def get_chunking_strategies():
    """Get available chunking strategies"""
    strategies = [
        {"name": "BY_PAGE", "description": "Chunk document by pages"},
        {"name": "BY_SECTION", "description": "Chunk document by sections/headings"},
        {"name": "BY_PARAGRAPH", "description": "Chunk document by paragraphs"},
        {"name": "SEMANTIC", "description": "Semantic chunking based on content similarity"}
    ]
    
    return {"strategies": strategies, "default": "BY_PAGE"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "DC Automation API with RAG is running",
        "features": ["Document Upload", "RAG Search", "Rule Extraction", "Document Analysis"]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DC Automation API with RAG",
        "version": "1.0.0",
        "description": "Simple Documentary Credit automation with RAG capabilities",
        "workflow": {
            "1": "Upload documents with /upload-document",
            "2": "Check processing status with /document-status/{document_id}",
            "3": "Search documents with /search-documents",
            "4": "Extract rules with /extract-rules",
            "5": "Analyze documents with /analyze-documents"
        },
        "endpoints": {
            "upload": "/upload-document",
            "status": "/document-status/{document_id}",
            "search": "/search-documents", 
            "extract": "/extract-rules",
            "analyze": "/analyze-documents",
            "delete": "/delete-document/{document_id}",
            "strategies": "/chunking-strategies"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
