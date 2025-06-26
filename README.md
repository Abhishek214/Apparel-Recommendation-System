prompt = f"""
Analyze this MT700 Documentary Credit message and extract ALL business rules from field 47A: Additional Conditions.

MT700 Content:
{mt700_text}

CRITICAL: Extract EVERY SINGLE condition from the 47A section as a separate business rule. Do not summarize or combine conditions.

For the 47A section:
- Each "+" symbol indicates a new condition - extract as separate rule
- Each ":" continuation should be part of the rule above it
- Each specific requirement, instruction, or condition must be a separate rule
- Include all charges, fees, banking instructions, and procedural requirements
- Include all document requirements and format specifications
- Extract each distinct obligation or condition mentioned

Be extremely detailed and comprehensive. Extract each individual requirement as a separate rule.

Output as valid JSON:
{{
    "extracted_fields": {{
        "45A": "full text from 45A section",
        "46A": "full text from 46A section", 
        "47A": "full text from 47A section"
    }},
    "business_rules": [
        {{
            "rule_id": 1,
            "rule_text": "All required data, field headings, and any pre-printed text required in order to determine facial compliance must be in English",
            "document_type": "All Documents",
            "requirement_type": "language_check",
            "field_name": "language",
            "expected_value": "English",
            "validation_note": "Check that all text in documents is in English language"
        }},
        {{
            "rule_id": 2,
            "rule_text": "Documents must be presented within 15 days of transport document date and within DC validity",
            "document_type": "All Documents",
            "requirement_type": "time_limit",
            "field_name": "presentation_date",
            "expected_value": "within 15 days of transport document date",
            "validation_note": "Ensure timely presentation of documents"
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

IMPORTANT: 
- Extract MINIMUM 10-15 rules from the 47A section (more if conditions exist)
- Each specific condition, requirement, or instruction should be a separate rule
- Do not combine multiple conditions into one rule
- Be very detailed and specific for each requirement
- Ensure all JSON is properly formatted and valid
"""
```
