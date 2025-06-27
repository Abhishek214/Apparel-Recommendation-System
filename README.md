#!/usr/bin/env python3
"""
Simple Utils - Basic business logic for DC automation with RAG
"""

import json
import PyPDF2
from typing import Dict, List, Optional, Any, Literal
from simple_rag_client import SimpleRAGClient


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


def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text"""
    try:
        # Find JSON boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return None


async def extract_mt700_rules_rag(
    rag_client: SimpleRAGClient,
    mt700_text: str,
    document_id: str,
    azure_token: str,
    use_rag: bool = True,
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Extract business rules from MT700 using RAG"""
    
    system_prompt = """You are an expert Documentary Credit examiner specializing in trade finance compliance.
    
Extract business rules from MT700 fields and provide structured output in JSON format.
Focus on fields 45A (Description of Goods), 46A (Documents Required), and 47A (Additional Conditions).
Be precise and create actionable validation rules."""

    if custom_prompt:
        prompt = custom_prompt.replace("{mt700_text}", mt700_text)
    else:
        prompt = f"""
        Analyze this MT700 Documentary Credit message and extract detailed business rules.

        MT700 Content:
        {mt700_text}

        Tasks:
        1. Extract all relevant DC fields (20, 31D, 32B, 40A, 44A, 44C, 45A, 46A, 47A, 50, 59)
        2. Convert conditions from fields 45A, 46A, and 47A into specific business rules
        3. Create actionable validation rules for document verification

        Output as valid JSON:
        {{
            "extracted_fields": {{
                "20": "DC reference number",
                "31D": "expiry date",
                "32B": "amount and currency",
                "40A": "form of DC",
                "44A": "port of loading", 
                "44C": "latest date of shipment",
                "45A": "description of goods",
                "46A": "documents required",
                "47A": "additional conditions",
                "50": "applicant details",
                "59": "beneficiary details"
            }},
            "business_rules": [
                {{
                    "rule_id": 1,
                    "rule_text": "Clear, specific requirement statement",
                    "document_type": "All Documents",
                    "requirement_type": "exact_match",
                    "field_name": "beneficiary_address",
                    "expected_value": "specific value to check for",
                    "validation_note": "additional context for validation"
                }}
            ],
            "dc_metadata": {{
                "dc_number": "extracted from field 20",
                "beneficiary": "extracted from field 59",
                "applicant": "extracted from field 50",
                "amount": "extracted from field 32B",
                "expiry_date": "extracted from field 31D",
                "latest_shipment": "extracted from field 44C"
            }}
        }}

        Make each rule specific and actionable for document verification.
        """
    
    try:
        if use_rag:
            # Use RAG to search for relevant documentation/examples
            search_query = f"MT700 documentary credit business rules extraction {mt700_text[:200]}..."
            search_result = await rag_client.search_documents(
                query=search_query,
                document_id=document_id,
                azure_token=azure_token,
                system_prompt=system_prompt
            )
            
            if search_result:
                response_text = search_result.get("response", "")
            else:
                return {"error": "RAG search failed"}
        else:
            # Direct query without RAG
            search_result = await rag_client.search_documents(
                query=prompt,
                document_id=document_id,
                azure_token=azure_token,
                system_prompt=system_prompt,
                search_type="NO-SEARCH"
            )
            
            if search_result:
                response_text = search_result.get("response", "")
            else:
                return {"error": "LLM query failed"}
        
        # Extract JSON from response
        result = extract_json_from_response(response_text)
        
        if result:
            return result
        else:
            return {"error": "No valid JSON found in response", "raw_response": response_text}
            
    except Exception as e:
        return {"error": f"Failed to extract rules: {str(e)}"}


async def analyze_documents_rag(
    rag_client: SimpleRAGClient,
    business_rules: List[Dict],
    document_id: str,
    azure_token: str,
    custom_prompt: Optional[str] = None,
    multimodal: bool = False,
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = "SEARCH"
) -> Dict[str, Any]:
    """Analyze documents against business rules using RAG"""
    
    system_prompt = """You are an expert Documentary Credit examiner specializing in trade finance compliance.

Verify trade documents against specified business rules and provide detailed compliance analysis.
Use "Passed" or "Need Review" for verification results.
Provide specific evidence from documents and detailed reasoning."""

    if custom_prompt:
        verification_prompt = custom_prompt
        verification_prompt = verification_prompt.replace("{business_rules}", json.dumps(business_rules, indent=2))
    else:
        verification_prompt = f"""
        As a Documentary Credit examiner, verify trade documents against these business rules:

        BUSINESS RULES TO CHECK:
        {json.dumps(business_rules, indent=2)}

        For each business rule:
        1. Search the documents for relevant information
        2. Check if the requirements are met
        3. Provide specific evidence from the documents
        4. Determine Pass/Fail status with detailed reasoning

        Output as JSON:
        {{
            "verification_results": [
                {{
                    "rule_id": 1,
                    "rule_text": "the business rule being checked",
                    "verification_result": "Passed" or "Need Review",
                    "verification_reasoning": "detailed explanation with specific evidence",
                    "evidence_found": "exact text/data found in documents or null",
                    "document_source": "which document(s) contained the evidence",
                    "confidence_level": "High" or "Medium" or "Low"
                }}
            ],
            "overall_compliance": "Passed" or "Need Review",
            "discrepancies": [
                {{
                    "rule_id": 1,
                    "issue_description": "clear description of the problem",
                    "severity": "Major" or "Minor",
                    "suggested_action": "recommendation for resolving the issue"
                }}
            ]
        }}
        """
    
    try:
        # Use RAG to search documents and verify rules
        search_result = await rag_client.search_documents(
            query=verification_prompt,
            document_id=document_id,
            azure_token=azure_token,
            system_prompt=system_prompt,
            multimodal=multimodal,
            search_type=search_type,
            max_documents=15  # More documents for comprehensive analysis
        )
        
        if not search_result:
            return {"error": "Document search failed"}
        
        response_text = search_result.get("response", "")
        
        # Extract JSON from response
        result = extract_json_from_response(response_text)
        
        if result:
            # Add search metadata
            result["search_metadata"] = {
                "documents_searched": len(search_result.get("documents_used", [])),
                "document_content_length": len(search_result.get("document_content", "")),
                "multimodal_used": multimodal,
                "search_type": search_type
            }
            return result
        else:
            return {"error": "Failed to parse verification response", "raw_response": response_text}
            
    except Exception as e:
        return {"error": f"Failed to analyze documents: {str(e)}"}
