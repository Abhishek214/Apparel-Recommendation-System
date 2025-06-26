#!/usr/bin/env python3
"""
Documentary Credit Automation - Modified for Single Combined PDF
Uses your SimpleLLMClient class for all LLM interactions
"""

import json
import PyPDF2
import tempfile
from typing import Dict, List, Optional
from pathlib import Path

class DCAutomationSinglePDF:
    def __init__(self, rag_service_url: str, app_id: str, azure_token: str):
        """Initialize with your SimpleLLMClient parameters"""
        # Import your SimpleLLMClient class here
        from your_module import SimpleLLMClient  # Update with actual import
        
        self.llm_client = SimpleLLMClient(
            rag_service_url=rag_service_url,
            app_id=app_id
        )
        self.azure_token = azure_token
        self.system_prompt = "You are an expert Documentary Credit examiner with deep knowledge of UCP 600, ISBP guidelines, and trade finance regulations."
    
    def extract_pdf_text(self, pdf_path: str) -> str:
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
    
    def extract_mt700_rules(self, mt700_text: str) -> Dict:
        """Extract business rules from MT700 using your LLM client"""
        
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
            response = self.llm_client.ask_llm(
                azure_token=self.azure_token,
                prompt=prompt,
                system_prompt=self.system_prompt,
                model_type="gpt-4o-mini",
                temperature=0.1
            )
            
            # Extract JSON from response if it's wrapped in text
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
            return {"error": f"Failed to extract rules: {str(e)}", "raw_response": response}
    
    def analyze_combined_documents(self, combined_pdf_text: str, business_rules: List[Dict]) -> Dict:
        """Analyze single combined PDF containing all trade documents"""
        
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
        
        try:
            # Step 1: Separate documents
            separation_response = self.llm_client.ask_llm(
                azure_token=self.azure_token,
                prompt=document_separation_prompt,
                system_prompt=self.system_prompt,
                model_type="gpt-4o-mini",
                temperature=0.1
            )
            
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
            
            verification_response = self.llm_client.ask_llm(
                azure_token=self.azure_token,
                prompt=verification_prompt,
                system_prompt=self.system_prompt,
                model_type="gpt-4o-mini",
                temperature=0.1
            )
            
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
            return {"error": f"Failed to analyze documents: {str(e)}", "raw_response": str(separation_response)}
    
    def generate_compliance_report(self, verification_results: Dict, dc_metadata: Dict) -> str:
        """Generate final compliance report"""
        
        prompt = f"""
        Generate a comprehensive, professional Documentary Credit examination report.

        DC INFORMATION:
        {json.dumps(dc_metadata, indent=2)}

        VERIFICATION RESULTS:
        {json.dumps(verification_results, indent=2)}

        Create a detailed report with:
        1. EXECUTIVE SUMMARY (overall status and key findings)
        2. DOCUMENTARY CREDIT DETAILS (DC number, parties, amount, etc.)
        3. DOCUMENTS EXAMINED (list of documents found and analyzed)
        4. COMPLIANCE EXAMINATION RESULTS 
           - Rules that passed (✓)
           - Discrepancies identified (⚠)
           - Detailed findings for each rule
        5. DISCREPANCY ANALYSIS (if any)
           - Major vs Minor issues
           - Impact assessment
           - Evidence from documents
        6. RECOMMENDATION (Accept/Reject/Request Waiver)
        7. NEXT STEPS
        8. CHARGES (if applicable per DC terms)

        Use professional banking language, clear formatting, and provide specific evidence for all findings.
        The report should be suitable for bank management review and customer communication.
        """
        
        try:
            response = self.llm_client.ask_llm(
                azure_token=self.azure_token,
                prompt=prompt,
                system_prompt=self.system_prompt,
                model_type="gpt-4o-mini",
                temperature=0.2
            )
            
            return response.get("response", "") if isinstance(response, dict) else str(response)
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def process_documentary_credit(self, mt700_pdf_path: str, combined_documents_pdf_path: str) -> Dict:
        """Main method to process complete DC automation"""
        
        try:
            # Step 1: Extract text from PDFs
            print("📄 Extracting MT700 text...")
            mt700_text = self.extract_pdf_text(mt700_pdf_path)
            
            print("📄 Extracting combined documents text...")
            combined_docs_text = self.extract_pdf_text(combined_documents_pdf_path)
            
            # Step 2: Extract business rules from MT700
            print("🔍 Extracting business rules from MT700...")
            rules_data = self.extract_mt700_rules(mt700_text)
            
            if "error" in rules_data:
                return {"status": "error", "message": rules_data["error"], "step": "rule_extraction"}
            
            business_rules = rules_data.get("business_rules", [])
            dc_metadata = rules_data.get("dc_metadata", {})
            
            print(f"✅ Extracted {len(business_rules)} business rules")
            
            # Step 3: Analyze combined documents against rules
            print("📋 Analyzing documents against business rules...")
            verification_results = self.analyze_combined_documents(combined_docs_text, business_rules)
            
            if "error" in verification_results:
                return {"status": "error", "message": verification_results["error"], "step": "document_verification"}
            
            # Step 4: Generate final compliance report
            print("📊 Generating compliance report...")
            final_report = self.generate_compliance_report(verification_results, dc_metadata)
            
            # Step 5: Compile final results
            final_results = {
                "status": "success",
                "processing_summary": {
                    "mt700_processed": True,
                    "rules_extracted": len(business_rules),
                    "documents_analyzed": verification_results.get("document_summary", {}).get("total_documents", 0),
                    "overall_compliance": verification_results.get("overall_compliance", "Unknown")
                },
                "dc_information": dc_metadata,
                "extracted_rules": business_rules,
                "verification_results": verification_results,
                "final_report": final_report,
                "discrepancies_summary": {
                    "total_discrepancies": len(verification_results.get("discrepancies", [])),
                    "major_issues": len([d for d in verification_results.get("discrepancies", []) if d.get("severity") == "Major"]),
                    "minor_issues": len([d for d in verification_results.get("discrepancies", []) if d.get("severity") == "Minor"])
                }
            }
            
            print("✅ Documentary Credit processing completed!")
            return final_results
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Unexpected error: {str(e)}", 
                "step": "processing"
            }

# Usage Example
def main():
    """Example usage with your LLM client"""
    
    # Initialize with your Azure/LLM service details
    dc_processor = DCAutomationSinglePDF(
        rag_service_url="https://hsbc-multi-dcrest-nonprod01-chat-service.azurewebsites.net/rag",
        app_id="lydia",  # or your actual app_id
        azure_token="your_azure_token_here"
    )
    
    # Process documentary credit
    results = dc_processor.process_documentary_credit(
        mt700_pdf_path="path/to/mt700.pdf",
        combined_documents_pdf_path="path/to/combined_trade_documents.pdf"
    )
    
    if results["status"] == "success":
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Rules Extracted: {results['processing_summary']['rules_extracted']}")
        print(f"Documents Analyzed: {results['processing_summary']['documents_analyzed']}")
        print(f"Overall Compliance: {results['processing_summary']['overall_compliance']}")
        print(f"Total Discrepancies: {results['discrepancies_summary']['total_discrepancies']}")
        
        print("\n=== FINAL REPORT ===")
        print(results["final_report"])
        
        # Save results to file
        with open("dc_processing_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    else:
        print(f"Error: {results['message']} at step: {results['step']}")

# FastAPI Integration (optional)
def create_fastapi_app():
    """Create FastAPI app for DC automation service"""
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="DC Automation Service - Single PDF")
    
    @app.post("/process-dc-single-pdf")
    async def process_dc_single_pdf(
        mt700: UploadFile = File(...),
        combined_documents: UploadFile = File(...),
        azure_token: str = Form(...)
    ):
        try:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as mt700_file:
                mt700_content = await mt700.read()
                mt700_file.write(mt700_content)
                mt700_path = mt700_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as docs_file:
                docs_content = await combined_documents.read()
                docs_file.write(docs_content)
                docs_path = docs_file.name
            
            # Process with DC automation
            dc_processor = DCAutomationSinglePDF(
                rag_service_url="https://hsbc-multi-dcrest-nonprod01-chat-service.azurewebsites.net/rag",
                app_id="lydia",
                azure_token=azure_token
            )
            
            results = dc_processor.process_documentary_credit(mt700_path, docs_path)
            
            # Clean up temp files
            Path(mt700_path).unlink()
            Path(docs_path).unlink()
            
            return JSONResponse(results)
            
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": f"API error: {str(e)}"
            }, status_code=500)
    
    return app

if __name__ == "__main__":
    main()
