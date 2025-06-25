#!/usr/bin/env python3
"""
Documentary Credit Automation - Working Example
Uses your actual MT700 document to demonstrate the LLM-centric approach
"""

import json
import openai
from typing import Dict, List
import PyPDF2
from pathlib import Path

class DCAutomationExample:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4"
    
    def llm_call(self, prompt: str, max_tokens: int = 4000) -> str:
        """Make LLM API call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def extract_rules_from_mt700(self, mt700_text: str) -> Dict:
        """Extract business rules from MT700 using LLM"""
        prompt = f"""
        You are a Documentary Credit expert. Analyze this MT700 message and extract business rules from field 47A.

        MT700 Content:
        {mt700_text}

        Tasks:
        1. Find and extract the 47A: Additional Conditions section
        2. Convert each condition into a specific business rule
        3. Categorize rules by document type and requirement type

        Output as JSON:
        {{
            "extracted_47A": "full 47A text here...",
            "business_rules": [
                {{
                    "rule_id": 1,
                    "rule_text": "BENEFICIARY'S FULL NAME AND ADDRESS must be 'STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM'",
                    "document_type": "All Documents",
                    "requirement_type": "exact_match",
                    "field_name": "beneficiary_address",
                    "expected_value": "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
                }}
            ],
            "dc_metadata": {{
                "dc_number": "extracted from 20 field",
                "beneficiary": "extracted from 59 field",
                "applicant": "extracted from 50 field"
            }}
        }}

        Make sure each rule is specific and actionable for document verification.
        """
        
        response = self.llm_call(prompt)
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse LLM response", "raw_response": response}
    
    def verify_documents_against_rules(self, documents: List[str], rules: List[Dict]) -> Dict:
        """Verify trade documents against extracted business rules"""
        
        # Prepare documents text
        docs_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([
            f"DOCUMENT {i+1}:\n{doc}" for i, doc in enumerate(documents)
        ])
        
        prompt = f"""
        You are a Documentary Credit examiner. Verify these trade documents against the business rules.

        BUSINESS RULES TO CHECK:
        {json.dumps(rules, indent=2)}

        TRADE DOCUMENTS:
        {docs_text}

        For each rule, determine:
        1. Does the document(s) contain the required information?
        2. Does it match the expected value (if specified)?
        3. Provide specific evidence from the documents

        Output as JSON:
        {{
            "verification_results": [
                {{
                    "rule_id": 1,
                    "rule_text": "the rule being checked...",
                    "verification_result": "Passed" or "Need Review",
                    "verification_reasoning": "detailed explanation with specific evidence",
                    "evidence_found": "exact text found in documents or null if not found",
                    "document_source": "which document contained the evidence"
                }}
            ],
            "overall_compliance": "Passed" or "Need Review",
            "discrepancies": [
                {{
                    "rule_id": X,
                    "issue": "description of the problem",
                    "severity": "Minor" or "Major"
                }}
            ]
        }}
        """
        
        response = self.llm_call(prompt, max_tokens=6000)
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse verification response", "raw_response": response}
    
    def generate_compliance_report(self, verification_results: Dict, dc_metadata: Dict) -> str:
        """Generate final compliance report"""
        prompt = f"""
        Generate a professional Documentary Credit examination report based on this verification:

        DC INFORMATION:
        {json.dumps(dc_metadata, indent=2)}

        VERIFICATION RESULTS:
        {json.dumps(verification_results, indent=2)}

        Create a comprehensive report with:
        1. EXECUTIVE SUMMARY
        2. DOCUMENT EXAMINATION RESULTS
        3. COMPLIANCE STATUS
        4. DISCREPANCIES IDENTIFIED (if any)
        5. RECOMMENDATION (Accept/Reject documents)
        6. NEXT STEPS

        Use professional banking language and clear formatting.
        """
        
        return self.llm_call(prompt, max_tokens=4000)

def demo_with_sample_documents():
    """Demo using the provided MT700 and sample trade documents"""
    
    # Sample MT700 text (from your document)
    mt700_sample = """
    SWIFT MT700 ISSUE OF A DOCUMENTARY CREDIT
    
    :20: DOCUMENTARY CREDIT NUMBER
    : : DC UK1071320
    
    :50: APPLICANT
    : : ABC FOODS LIMITED
    : : FOX STREET, SUFFEX, UK
    
    :59: BENEFICIARY  
    : : DEF CO LTD
    : : INDUSTRIAL ZONE,
    : : SHANGHAI, CHINA
    
    :47A: ADDITIONAL CONDITIONS
    : +BENEFICIARY'S FULL NAME AND ADDRESS:
    : +STEVE POWER LTD.
    : :NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM
    : +BILLS OF LADING, SUPPLIER'S CERTIFICATE OF PRODUCTION MUST SHOW
    : :QUANTITY, FORM M NUMBER AND DESCRIPTION OF GOODS
    : :AS 'SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR' ONLY.
    : +BILLS OF LADING MUST MENTION H.S. CODE: 8503.0000.00
    : +ALL DOCUMENTS EXCEPT DRAFT AND COMMERCIAL INVOICE MUST NOT SHOW
    : :THE UNIT PRICE, TOTAL VALUE AND THIS D.C NO.
    : +ALL DOCUMENTS EXCEPT DRAFT MUST SHOW FORM M NUMBER.
    : +SHIPMENT MUST BE EFFECTED BY HAPAG LLOYD OR MAERSK LINE OR PILL
    : :AND BILLS OF LADING MUST BE ISSUED BY HAPAG LLOYD OR MAERSK LINE OR PILL
    : +BILLS OF LADING MUST MENTION 'FREE TIME 21 DAYS COMBINED
    : :(DETENTION AND DEMURRAGE) AT PORT OF DISCHARGE'
    """
    
    # Sample trade documents (based on your images)
    sample_documents = [
        """
        BILL OF LADING
        Shipper: DEF CO. LTD
        INDUSTRIAL ZONE, SHANGHAI, CHINA
        
        Consignee: TO ORDER OF ABC FOODS LIMITED
        
        Description of Goods: FROZEN TIGER DELIGHT ROAST DUCK
        Quantity: 2300 CARTONS
        
        Port of Loading: QINGDAO, CHINA
        Port of Discharge: FELIXSTOWE, UK
        
        Carrier: CMA CGM
        """,
        
        """
        COMMERCIAL INVOICE
        From: DEF CO.,LTD
        INDUSTRIAL ZONE, SHANGHAI, CHINA
        
        To: ABC FOODS LIMITED
        FOX STREET, SUFFEX, UNITED KINGDOM
        
        Invoice No: VFL015
        Date: JUL 05, 2024
        
        Description: FROZEN TIGER DELIGHT ROAST DUCK
        Quantity: 2300 CARTONS
        Unit Price: USD43.5/CARTON
        Amount: USD100050.00
        """,
        
        """
        PACKING LIST
        From: DEF CO.,LTD
        INDUSTRIAL ZONE, SHANGHAI, CHINA
        
        To: ABC FOODS LIMITED
        
        Description: FROZEN TIGER DELIGHT ROAST DUCK
        Total Packed: 2300 CARTONS
        Net Weight: 25000 KGS
        Gross Weight: 25100 KGS
        """
    ]
    
    # Initialize system (you would use your actual OpenAI API key)
    dc_system = DCAutomationExample(api_key="your-openai-api-key-here")
    
    print("=== Step 1: Extracting Business Rules from MT700 ===")
    rules_data = dc_system.extract_rules_from_mt700(mt700_sample)
    print(json.dumps(rules_data, indent=2))
    
    if "business_rules" in rules_data:
        print(f"\n=== Extracted {len(rules_data['business_rules'])} Business Rules ===")
        
        print("\n=== Step 2: Verifying Documents Against Rules ===")
        verification = dc_system.verify_documents_against_rules(
            sample_documents, 
            rules_data['business_rules']
        )
        print(json.dumps(verification, indent=2))
        
        print("\n=== Step 3: Generating Final Report ===")
        final_report = dc_system.generate_compliance_report(
            verification, 
            rules_data.get('dc_metadata', {})
        )
        print(final_report)
    
    return {
        "rules": rules_data,
        "verification": verification if 'verification' in locals() else None,
        "report": final_report if 'final_report' in locals() else None
    }

# Simple FastAPI integration
def create_api():
    """
    Simple API setup - install: pip install fastapi uvicorn python-multipart
    Run with: uvicorn filename:app --reload
    """
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="DC Automation API")
    dc_system = DCAutomationExample(api_key="your-openai-api-key")
    
    @app.post("/analyze-dc")
    async def analyze_documentary_credit(
        mt700: UploadFile = File(...),
        documents: List[UploadFile] = File(...)
    ):
        try:
            # Extract MT700 text
            mt700_content = await mt700.read()
            mt700_text = mt700_content.decode('utf-8')
            
            # Extract rules
            rules_data = dc_system.extract_rules_from_mt700(mt700_text)
            
            # Extract document texts
            doc_texts = []
            for doc in documents:
                content = await doc.read()
                doc_texts.append(content.decode('utf-8'))
            
            # Verify compliance
            verification = dc_system.verify_documents_against_rules(
                doc_texts, 
                rules_data.get('business_rules', [])
            )
            
            # Generate report
            report = dc_system.generate_compliance_report(
                verification, 
                rules_data.get('dc_metadata', {})
            )
            
            return JSONResponse({
                "status": "success",
                "rules_extracted": len(rules_data.get('business_rules', [])),
                "overall_compliance": verification.get('overall_compliance', 'Unknown'),
                "discrepancies_count": len(verification.get('discrepancies', [])),
                "full_report": report,
                "detailed_results": verification
            })
            
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": str(e)
            }, status_code=500)
    
    return app

# Run the demo
if __name__ == "__main__":
    # For demo purposes - replace with actual API key to run
    print("DC Automation Demo")
    print("Replace 'your-openai-api-key-here' with your actual API key to run")
    
    # Uncomment to run demo:
    # demo_with_sample_documents()
    
    # To run API:
    # app = create_api()
    # Then run: uvicorn filename:app --reload
