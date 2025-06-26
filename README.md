#!/usr/bin/env python3
"""
application.py - Main FastAPI Application
Simple API for DC Automation with prompt optimization
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
from pathlib import Path

# Import our utility functions
from utils import SimpleLLMClient, extract_pdf_text, extract_mt700_rules, analyze_combined_documents

# FastAPI Application
app = FastAPI(
    title="DC Automation API",
    description="API for Documentary Credit rule extraction and document verification with prompt optimization",
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

# Initialize LLM Client
llm_client = SimpleLLMClient(
    rag_service_url="https://hsbc-multi-dcrest-nonprod01-chat-service.azurewebsites.net/rag",
    app_id="lydia"
)

@app.post("/extract-rules")
async def extract_rules_from_mt700(request: dict):
    """
    Extract business rules from MT700 text
    Supports custom prompts for optimization
    
    Body:
    {
        "mt700_text": "MT700 content...",
        "azure_token": "your_token",
        "custom_prompt": "optional custom prompt with {mt700_text} placeholder",
        "temperature": 0.1 (optional)
    }
    """
    try:
        # Extract parameters from request
        mt700_text = request.get("mt700_text", "")
        azure_token = request.get("azure_token", "")
        custom_prompt = request.get("custom_prompt", "")
        temperature = request.get("temperature", 0.1)
        
        # Validate inputs
        if not mt700_text.strip():
            return {"error": "MT700 text cannot be empty"}
        
        if not azure_token.strip():
            return {"error": "Azure token is required"}
        
        # Extract rules using utility function
        result = await extract_mt700_rules(
            llm_client=llm_client,
            mt700_text=mt700_text,
            azure_token=azure_token,
            custom_prompt=custom_prompt if custom_prompt.strip() else None
        )
        
        if "error" in result:
            return {"status": "error", "message": result["error"], "raw_response": result.get("raw_response", "")}
        
        # Structure successful response
        response_data = {
            "status": "success",
            "extracted_fields": result.get("extracted_fields", {}),
            "business_rules": result.get("business_rules", []),
            "dc_metadata": result.get("dc_metadata", {}),
            "rules_count": len(result.get("business_rules", [])),
            "prompt_optimization": {
                "custom_prompt_used": bool(custom_prompt.strip()),
                "temperature_used": temperature
            }
        }
        
        return response_data
        
    except Exception as e:
        return {"status": "error", "message": f"Internal server error: {str(e)}"}

@app.post("/analyze-documents")
async def analyze_combined_documents_endpoint(
    combined_documents: UploadFile = File(..., description="Combined PDF containing all trade documents"),
    business_rules: str = Form(..., description="JSON string of business rules to check against"),
    azure_token: str = Form(..., description="Azure authentication token"),
    custom_prompt: str = Form("", description="Optional custom prompt with {business_rules}, {documents}, {document_text} placeholders"),
    temperature: float = Form(0.1, description="Temperature for LLM response generation"),
    model_type: str = Form("gpt-4o-mini", description="LLM model to use")
):
    """
    Analyze combined documents PDF against business rules
    Supports custom prompts for verification optimization
    """
    temp_pdf_path = None
    
    try:
        # Validate inputs
        if not azure_token.strip():
            return {"error": "Azure token is required"}
        
        if not combined_documents.filename.lower().endswith('.pdf'):
            return {"error": "File must be a PDF"}
        
        # Parse business rules JSON
        try:
            rules_list = json.loads(business_rules)
            if not isinstance(rules_list, list):
                raise ValueError("Business rules must be a list")
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for business rules"}
        
        # Save uploaded PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await combined_documents.read()
            temp_file.write(content)
            temp_pdf_path = temp_file.name
        
        # Extract text from PDF
        document_text = extract_pdf_text(temp_pdf_path)
        
        if document_text.startswith("Error"):
            return {"error": f"PDF extraction failed: {document_text}"}
        
        # Analyze documents using utility function
        result = await analyze_combined_documents(
            llm_client=llm_client,
            combined_pdf_text=document_text,
            business_rules=rules_list,
            azure_token=azure_token,
            custom_prompt=custom_prompt if custom_prompt.strip() else None
        )
        
        if "error" in result:
            return {"status": "error", "message": result["error"], "raw_response": result.get("raw_response", "")}
        
        # Structure successful response
        response_data = {
            "status": "success",
            "document_summary": result.get("document_summary", {}),
            "verification_results": result.get("verification_results", []),
            "overall_compliance": result.get("overall_compliance", "Unknown"),
            "compliance_summary": result.get("compliance_summary", {}),
            "discrepancies": result.get("discrepancies", []),
            "separated_documents": result.get("separated_documents", {}),
            "prompt_optimization": {
                "custom_prompt_used": bool(custom_prompt.strip()),
                "temperature_used": temperature,
                "model_used": model_type,
                "document_length": len(document_text),
                "rules_processed": len(rules_list)
            }
        }
        
        # Add analysis metrics for quick assessment
        verification_results = result.get("verification_results", [])
        if verification_results:
            response_data["analysis_metrics"] = {
                "total_rules_checked": len(verification_results),
                "rules_passed": len([r for r in verification_results if r.get("verification_result") == "Passed"]),
                "rules_need_review": len([r for r in verification_results if r.get("verification_result") == "Need Review"]),
                "rules_with_evidence": len([r for r in verification_results if r.get("evidence_found")]),
                "high_confidence_results": len([r for r in verification_results if r.get("confidence_level") == "High"])
            }
        
        return response_data
        
    except Exception as e:
        return {"status": "error", "message": f"Internal server error: {str(e)}"}
    
    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "DC Automation API is running",
        "endpoints": ["/extract-rules", "/analyze-documents"]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DC Automation API with Prompt Optimization",
        "version": "1.0.0",
        "description": "Extract business rules from MT700 and verify documents with customizable prompts",
        "endpoints": {
            "extract_rules": {
                "url": "/extract-rules",
                "method": "POST",
                "description": "Extract business rules from MT700 text",
                "supports_custom_prompts": True
            },
            "analyze_documents": {
                "url": "/analyze-documents", 
                "method": "POST",
                "description": "Verify documents against business rules",
                "supports_custom_prompts": True
            },
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "API health check"
            }
        },
        "prompt_optimization": {
            "placeholders": {
                "extract_rules": ["{mt700_text}"],
                "analyze_documents": ["{business_rules}", "{documents}", "{document_text}"]
            },
            "example_custom_prompt": "Verify these rules carefully: {business_rules} against documents: {documents}"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
