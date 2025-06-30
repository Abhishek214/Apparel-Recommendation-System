async def extract_rules_from_dc_field(field_text, field_name=""):
    prompt = f"""
TASK: Extract ALL business rules and requirements from Documentary Credit field content.

FIELD CONTENT:
{field_text}

EXTRACTION APPROACH:
1. Read EVERY sentence and clause completely
2. Extract ALL requirements (mandatory, conditional, optional)
3. Include ALL charges, fees, instructions, waivers, and conditions
4. Capture multi-part and nested conditions
5. Don't skip lengthy paragraphs or complex clauses

RULE TYPES TO CAPTURE:
- Document requirements (what must be shown/mentioned)
- Conditional requirements (IF-THEN scenarios)
- Charges and fees (amounts, who pays, when deducted)
- Banking instructions and procedures
- Waiver and refusal provisions
- Shipment and transport requirements
- Beneficiary/Applicant obligations
- Timeline and validity conditions

OUTPUT FORMAT:
{{
    "business_rules": [
        {{
            "rule_id": 1,
            "rule_text": "Complete, clear requirement statement",
            "category": "document_requirement|charge|procedure|condition|waiver",
            "applies_to": "specific document type or 'All Documents'",
            "requirement_type": "must_show|must_mention|must_not_show|conditional|procedure",
            "field_name": "specific field if applicable",
            "expected_value": "required value or format",
            "conditions": "any IF-THEN conditions",
            "enforcement": "mandatory|conditional|optional"
        }}
    ]
}}

CRITICAL: Process ENTIRE text systematically. Extract each distinct obligation, even from long paragraphs."""

    return prompt
