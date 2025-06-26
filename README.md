#!/usr/bin/env python3
"""
Utils.py - Utility functions for DC Automation
Contains PDF processing, LLM client, and core processing functions
"""

import json
import httpx
import PyPDF2
from typing import Dict, List, Optional, Any

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

def split_pdf_in_two(input_pdf_path, output_pdf1_path, output_pdf2_path):
    """Split PDF into two parts"""
    reader = PyPDF2.PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    midpoint = total_pages // 2
    writer1 = PyPDF2.PdfWriter()
    writer2 = PyPDF2.PdfWriter()

    # First half
    for i in range(midpoint):
        writer1.add_page(reader.pages[i])
    with open(output_pdf1_path, 'wb') as f:
        writer1.write(f)

    # Second half
    for i in range(midpoint, total_pages):
        writer2.add_page(reader.pages[i])
    with open(output_pdf2_path, 'wb') as f:
        writer2.write(f)

    print(f"PDF split into two parts: {output_pdf1_path}, {output_pdf2_path}")

async def extract_mt700_rules(llm_client: SimpleLLMClient, mt700_text: str, azure_token: str, custom_prompt: str = None) -> Dict:
    """Extract business rules from MT700 using LLM client"""
    
    system_prompt = """You are an expert Documentary Credit examiner specializing in trade finance compliance.

EXPERTISE:
- SWIFT MT700 message analysis
- UCP 600 and ISBP standards
- Trade document examination
- Banking compliance requirements

CORE TASKS:
- Extract business rules from MT700 fields (45A, 46A, 47A)
- Verify trade documents against DC requirements
- Generate professional compliance reports

OUTPUT STANDARDS:
- Return valid JSON when requested
- Provide specific evidence and reasoning
- Use "Passed" or "Need Review" for compliance status
- Classify discrepancies as "Major" or "Minor"
- Be precise and conservative in assessments

COMPLIANCE APPROACH:
- Flag potential issues rather than assume compliance
- Quote exact text as evidence
- Reference specific documents
- Follow banking best practices
"""
    
    if custom_prompt:
        prompt = custom_prompt.replace("{mt700_text}", mt700_text)
    else:
        prompt = f"""
        Analyze this MT700 Documentary Credit message and extract business rules from fields 45A, 46A, and 47A.

        MT700 Content:
        {mt700_text}

        Tasks:
        1. Extract the 47A: Additional Conditions section
        2. Convert each condition into specific, numbered business rules
        3. Extract other relevant DC information (59, 58A, etc.)
        4. Categorize rules by document type and requirement type

        Output as valid JSON:
        {{
            "extracted_fields": {{
                "45A": "description of goods text...",
                "46A": "documents required text...", 
                "47A": "additional conditions text..."
            }},
            "business_rules": [
                {{
                    "rule_id": 1,
                    "rule_text": "Clear, specific requirement statement",
                    "document_type": "All Documents" or "Bill of Lading" or "Commercial Invoice" etc,
                    "requirement_type": "exact_match" or "contains" or "tolerance_check" or "presence_check",
                    "field_name": "beneficiary_address" or "quantity" etc,
                    "expected_value": "specific value to check for or null",
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
        Ensure all JSON is properly formatted and valid.
        """
    
    try:
        response = await llm_client.ask_llm(
            azure_token=azure_token,
            prompt=prompt,
            system_prompt=system_prompt,
            model_type="gpt-4o-mini",
            temperature=0.1
        )
        
        if not response:
            return {"error": "No response from LLM service"}
        
        # Extract JSON from response
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        
        # Try to find JSON in the response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {"error": "No valid JSON found in response", "raw_response": response_text}
            
    except Exception as e:
        return {"error": f"Failed to extract rules: {str(e)}", "raw_response": str(response)}

async def analyze_combined_documents(llm_client: SimpleLLMClient, combined_pdf_text: str, business_rules: List[Dict], azure_token: str, custom_prompt: str = None) -> Dict:
    """Analyze single combined PDF containing all trade documents"""
    
    system_prompt = """You are an expert Documentary Credit examiner specializing in trade finance compliance.

EXPERTISE:
- SWIFT MT700 message analysis
- UCP 600 and ISBP standards
- Trade document examination
- Banking compliance requirements

CORE TASKS:
- Extract business rules from MT700 fields (45A, 46A, 47A)
- Verify trade documents against DC requirements
- Generate professional compliance reports

OUTPUT STANDARDS:
- Return valid JSON when requested
- Provide specific evidence and reasoning
- Use "Passed" or "Need Review" for compliance status
- Classify discrepancies as "Major" or "Minor"
- Be precise and conservative in assessments

COMPLIANCE APPROACH:
- Flag potential issues rather than assume compliance
- Quote exact text as evidence
- Reference specific documents
- Follow banking best practices
"""
    
    try:
        # First, identify and separate different documents within the combined PDF
        document_separation_prompt = f"""
        This is a combined PDF containing multiple trade documents for a Documentary Credit transaction.
        Please identify and separate the different documents.

        Combined Document Text:
        {combined_pdf_text}

        Tasks:
        1. Identify the different document types (e.g., Bill of Lading, Commercial Invoice, Packing List, Certificate of Origin, etc.)
        2. Extract the content for each document type
        3. Provide the extracted content for each document

        Output as JSON:
        {{
            "identified_documents": [
                {{
                    "document_type": "Bill of Lading",
                    "document_content": "extracted content for this document...",
                    "page_reference": "approximate location/page in combined PDF"
                }},
                {{
                    "document_type": "Commercial Invoice", 
                    "document_content": "extracted content for this document...",
                    "page_reference": "approximate location/page in combined PDF"
                }}
            ],
            "total_documents_found": 3,
            "extraction_notes": "any issues or observations during extraction"
        }}
        """
        
        # Step 1: Separate documents
        separation_response = await llm_client.ask_llm(
            azure_token=azure_token,
            prompt=document_separation_prompt,
            system_prompt=system_prompt,
            model_type="gpt-4o-mini",
            temperature=0.1
        )
        
        if not separation_response:
            return {"error": "No response from document separation"}
        
        # Parse separation response
        sep_text = separation_response.get("response", "") if isinstance(separation_response, dict) else str(separation_response)
        start_idx = sep_text.find("{")
        end_idx = sep_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = sep_text[start_idx:end_idx]
            separated_docs = json.loads(json_str)
        else:
            # Fallback: treat entire content as single document set
            separated_docs = {
                "identified_documents": [
                    {"document_type": "Combined Documents", "document_content": combined_pdf_text}
                ]
            }
        
        # Step 2: Verify against business rules
        if custom_prompt:
            verification_prompt = custom_prompt
            verification_prompt = verification_prompt.replace("{business_rules}", json.dumps(business_rules, indent=2))
            verification_prompt = verification_prompt.replace("{documents}", json.dumps(separated_docs.get('identified_documents', []), indent=2))
            verification_prompt = verification_prompt.replace("{document_text}", combined_pdf_text)
        else:
            verification_prompt = f"""
            As a Documentary Credit examiner, verify these trade documents against the specified business rules.

            BUSINESS RULES TO CHECK:
            {json.dumps(business_rules, indent=2)}

            IDENTIFIED TRADE DOCUMENTS:
            {json.dumps(separated_docs.get('identified_documents', []), indent=2)}

            For each business rule:
            1. Check if the relevant document(s) contain the required information
            2. Verify if values match expected requirements
            3. Provide specific evidence from the documents
            4. Determine Pass/Fail status with detailed reasoning

            Output as JSON:
            {{
                "document_summary": {{
                    "total_documents": {len(separated_docs.get('identified_documents', []))},
                    "document_types": ["list of document types found"]
                }},
                "verification_results": [
                    {{
                        "rule_id": 1,
                        "rule_text": "the business rule being checked",
                        "verification_result": "Passed" or "Need Review",
                        "verification_reasoning": "detailed explanation with specific evidence or why it failed",
                        "evidence_found": "exact text/data found in documents or null if not found",
                        "document_source": "which specific document(s) contained the evidence",
                        "confidence_level": "High" or "Medium" or "Low"
                    }}
                ],
                "overall_compliance": "Passed" or "Need Review",
                "compliance_summary": {{
                    "rules_passed": 0,
                    "rules_failed": 0,
                    "rules_need_review": 0
                }},
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
        
        verification_response = await llm_client.ask_llm(
            azure_token=azure_token,
            prompt=verification_prompt,
            system_prompt=system_prompt,
            model_type="gpt-4o-mini",
            temperature=0.1
        )
        
        if not verification_response:
            return {"error": "No response from document verification"}
        
        # Parse verification response
        ver_text = verification_response.get("response", "") if isinstance(verification_response, dict) else str(verification_response)
        start_idx = ver_text.find("{")
        end_idx = ver_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = ver_text[start_idx:end_idx]
            verification_result = json.loads(json_str)
            verification_result["separated_documents"] = separated_docs
            return verification_result
        else:
            return {"error": "Failed to parse verification response", "raw_response": ver_text}
            
    except Exception as e:
        return {"error": f"Failed to analyze documents: {str(e)}"}
