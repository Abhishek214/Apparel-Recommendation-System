#!/usr/bin/env python3
"""
Simple FastAPI for Prompt Optimization - Business Rule Verification
No Pydantic, focused on prompt testing and optimization
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import httpx
import asyncio
from typing import Dict, List, Optional, Any
import PyPDF2
from pathlib import Path

# Your existing SimpleLLMClient class
class SimpleLLMClient:
    def __init__(self, rag_service_url: str, app_id: str):
        self.rag_service_url = rag_service_url
        self.app_id = app_id

    async def ask_llm(
        self,
        azure_token: str,
        prompt: str,
        system_prompt: str = "You are an intelligent AI assistant",
        model_type: str = "gpt-4o-mini",
        temperature: float = 0.1,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a simple prompt to LLM without document search"""
        
        try:
            headers = {
                "accept": "application/json",
                "Authorization": azure_token
            }

            json_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "model": model_type,
                "temperature": temperature,
                "search_type": "SEARCH",  # No document search
                "use_tools": False,
                "save_response": False,
                "multimodal": False
            }

            if session_id:
                json_data["session_id"] = session_id

            url = f"{self.rag_service_url}/{self.app_id}"

            async with httpx.AsyncClient(timeout=60, verify=False) as client:
                response = await client.post(url, headers=headers, json=json_data)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# FastAPI Application
app = FastAPI(
    title="Prompt Optimization API - Business Rule Verification",
    description="Simple API for testing and optimizing prompts for business rule verification",
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
    Simple endpoint for prompt optimization
    """
    try:
        # Extract parameters from request
        mt700_text = request.get("mt700_text", "")
        azure_token = request.get("azure_token", "")
        custom_prompt = request.get("custom_prompt", "")  # For prompt optimization
        system_prompt = request.get("system_prompt", "You are an expert Documentary Credit examiner.")
        temperature = request.get("temperature", 0.1)
        
        if not mt700_text.strip():
            return {"error": "MT700 text cannot be empty"}
        
        if not azure_token.strip():
            return {"error": "Azure token is required"}
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = custom_prompt.replace("{mt700_text}", mt700_text)
        else:
            # Default prompt
            prompt = f"""
            Analyze this MT700 Documentary Credit message and extract business rules.

            MT700 Content:
            {mt700_text}

            Extract and return as JSON:
            {{
                "business_rules": [
                    {{
                        "rule_id": 1,
                        "rule_text": "specific requirement",
                        "document_type": "document type to check",
                        "requirement_type": "exact_match/contains/presence_check",
                        "expected_value": "value to look for"
                    }}
                ],
                "dc_metadata": {{
                    "dc_number": "from field 20",
                    "beneficiary": "from field 59", 
                    "applicant": "from field 50"
                }}
            }}
            """
        
        # Call LLM
        response = await llm_client.ask_llm(
            azure_token=azure_token,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        if not response:
            return {"error": "No response from LLM service"}
        
        # Try to extract JSON from response
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        
        # Find JSON in response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        result = {
            "status": "success",
            "raw_response": response_text,
            "prompt_used": prompt,
            "system_prompt_used": system_prompt,
            "temperature_used": temperature
        }
        
        if start_idx != -1 and end_idx != -1:
            try:
                json_str = response_text[start_idx:end_idx]
                parsed_json = json.loads(json_str)
                result["parsed_result"] = parsed_json
                result["rules_count"] = len(parsed_json.get("business_rules", []))
            except json.JSONDecodeError:
                result["parsing_error"] = "Failed to parse JSON from response"
        else:
            result["parsing_error"] = "No JSON found in response"
        
        return result
        
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/test-prompt")
async def test_prompt_for_verification(
    documents: UploadFile = File(...),
    business_rules: str = Form(...),
    custom_prompt: str = Form(...),
    azure_token: str = Form(...),
    system_prompt: str = Form("You are an expert Documentary Credit examiner."),
    temperature: float = Form(0.1),
    model_type: str = Form("gpt-4o-mini")
):
    """
    Test custom prompts for verifying business rules against documents
    Main endpoint for prompt optimization
    """
    temp_pdf_path = None
    
    try:
        # Validate inputs
        if not azure_token.strip():
            return {"error": "Azure token is required"}
        
        if not custom_prompt.strip():
            return {"error": "Custom prompt is required"}
        
        if not documents.filename.lower().endswith('.pdf'):
            return {"error": "File must be a PDF"}
        
        # Parse business rules
        try:
            rules_list = json.loads(business_rules)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for business rules"}
        
        # Save uploaded PDF and extract text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await documents.read()
            temp_file.write(content)
            temp_pdf_path = temp_file.name
        
        # Extract text from PDF
        document_text = extract_pdf_text(temp_pdf_path)
        
        if document_text.startswith("Error"):
            return {"error": f"PDF extraction failed: {document_text}"}
        
        # Replace placeholders in custom prompt
        final_prompt = custom_prompt
        final_prompt = final_prompt.replace("{business_rules}", json.dumps(rules_list, indent=2))
        final_prompt = final_prompt.replace("{documents}", document_text)
        final_prompt = final_prompt.replace("{document_text}", document_text)
        
        # Call LLM with custom prompt
        response = await llm_client.ask_llm(
            azure_token=azure_token,
            prompt=final_prompt,
            system_prompt=system_prompt,
            model_type=model_type,
            temperature=temperature
        )
        
        if not response:
            return {"error": "No response from LLM service"}
        
        # Extract response text
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        
        # Try to parse JSON if present
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        result = {
            "status": "success",
            "raw_response": response_text,
            "prompt_used": final_prompt,
            "system_prompt_used": system_prompt,
            "temperature_used": temperature,
            "model_used": model_type,
            "document_length": len(document_text),
            "rules_provided": len(rules_list)
        }
        
        if start_idx != -1 and end_idx != -1:
            try:
                json_str = response_text[start_idx:end_idx]
                parsed_json = json.loads(json_str)
                result["parsed_result"] = parsed_json
                
                # Add some analysis metrics
                if "verification_results" in parsed_json:
                    verification_results = parsed_json["verification_results"]
                    result["analysis_metrics"] = {
                        "total_rules_checked": len(verification_results),
                        "rules_passed": len([r for r in verification_results if r.get("verification_result") == "Passed"]),
                        "rules_need_review": len([r for r in verification_results if r.get("verification_result") == "Need Review"]),
                        "overall_compliance": parsed_json.get("overall_compliance", "Unknown")
                    }
                    
            except json.JSONDecodeError:
                result["parsing_error"] = "Failed to parse JSON from response"
        else:
            result["parsing_error"] = "No JSON found in response"
        
        return result
        
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}
    
    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

@app.post("/compare-prompts")
async def compare_multiple_prompts(
    documents: UploadFile = File(...),
    business_rules: str = Form(...),
    prompts: str = Form(...),  # JSON array of prompts to test
    azure_token: str = Form(...),
    system_prompt: str = Form("You are an expert Documentary Credit examiner."),
    temperature: float = Form(0.1)
):
    """
    Compare multiple prompts side by side for optimization
    """
    temp_pdf_path = None
    
    try:
        # Parse inputs
        try:
            rules_list = json.loads(business_rules)
            prompts_list = json.loads(prompts)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        
        if not isinstance(prompts_list, list):
            return {"error": "Prompts must be a list"}
        
        # Extract document text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await documents.read()
            temp_file.write(content)
            temp_pdf_path = temp_file.name
        
        document_text = extract_pdf_text(temp_pdf_path)
        
        if document_text.startswith("Error"):
            return {"error": f"PDF extraction failed: {document_text}"}
        
        # Test each prompt
        results = []
        
        for i, prompt_template in enumerate(prompts_list):
            # Replace placeholders
            final_prompt = prompt_template
            final_prompt = final_prompt.replace("{business_rules}", json.dumps(rules_list, indent=2))
            final_prompt = final_prompt.replace("{documents}", document_text)
            final_prompt = final_prompt.replace("{document_text}", document_text)
            
            # Call LLM
            response = await llm_client.ask_llm(
                azure_token=azure_token,
                prompt=final_prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            response_text = response.get("response", "") if response else "No response"
            
            # Try to parse JSON
            parsed_result = None
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    json_str = response_text[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            results.append({
                "prompt_index": i + 1,
                "prompt_template": prompt_template,
                "final_prompt": final_prompt,
                "raw_response": response_text,
                "parsed_result": parsed_result,
                "response_length": len(response_text),
                "json_parseable": parsed_result is not None
            })
        
        return {
            "status": "success",
            "comparison_results": results,
            "total_prompts_tested": len(prompts_list),
            "document_length": len(document_text),
            "rules_count": len(rules_list)
        }
        
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}
    
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

@app.get("/sample-prompts")
async def get_sample_prompts():
    """
    Get sample prompts for testing and optimization
    """
    return {
        "rule_extraction_prompts": [
            {
                "name": "basic_extraction",
                "prompt": """Analyze this MT700 and extract business rules as JSON:
{mt700_text}

Return: {{"business_rules": [list], "dc_metadata": {{}}}}"""
            },
            {
                "name": "detailed_extraction", 
                "prompt": """You are a Documentary Credit expert. Analyze this MT700 message and extract specific business rules from field 47A:

{mt700_text}

For each condition in 47A, create a specific rule with:
- rule_text: exact requirement
- document_type: which documents to check
- requirement_type: how to verify
- expected_value: what to look for

Return as JSON with business_rules array and dc_metadata object."""
            }
        ],
        "verification_prompts": [
            {
                "name": "basic_verification",
                "prompt": """Check these documents against the business rules:

RULES: {business_rules}
DOCUMENTS: {documents}

Return JSON with verification_results array."""
            },
            {
                "name": "detailed_verification",
                "prompt": """As a Documentary Credit examiner, verify these trade documents against business rules:

BUSINESS RULES TO CHECK:
{business_rules}

DOCUMENTS TO ANALYZE:
{documents}

For each rule:
1. Check if documents contain required information
2. Verify values match requirements  
3. Provide specific evidence
4. Return "Passed" or "Need Review"

Output JSON:
{{
  "verification_results": [
    {{
      "rule_id": 1,
      "verification_result": "Passed/Need Review", 
      "verification_reasoning": "detailed explanation",
      "evidence_found": "exact text or null"
    }}
  ],
  "overall_compliance": "Passed/Need Review",
  "discrepancies": []
}}"""
            },
            {
                "name": "conservative_verification",
                "prompt": """CONSERVATIVE APPROACH: Flag any uncertain items for review.

Check documents against rules and be strict about compliance:

RULES: {business_rules}
DOCUMENTS: {documents}

- Use "Passed" only when completely certain
- Use "Need Review" for any ambiguity
- Provide specific evidence quotes
- Classify discrepancies as Major/Minor

Return detailed JSON verification results."""
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Prompt Optimization API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Prompt Optimization API for Business Rule Verification",
        "version": "1.0.0",
        "endpoints": {
            "extract_rules": "/extract-rules - Test prompts for rule extraction",
            "test_prompt": "/test-prompt - Test custom verification prompts", 
            "compare_prompts": "/compare-prompts - Compare multiple prompts",
            "sample_prompts": "/sample-prompts - Get sample prompts",
            "health": "/health",
            "docs": "/docs"
        },
        "usage": "Upload documents and test different prompts to optimize accuracy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
