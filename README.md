As a Documentary Credit examiner, verify these trade documents against the specified business rules.

BUSINESS RULES TO CHECK:
{json.dumps(business_rules, indent=2)}

For each business rule:
1. Check if the relevant document(s) contain the required information
2. Verify if values match expected requirements
3. Provide specific evidence from the documents
4. Identify the PAGE NUMBER where evidence was found
5. Determine "Matched"/"Partial Match (with % match)"/"Not Found" status with detailed reasoning
6. If specific document type is not mentioned in rule, then document_source is All Documents.

ADDITIONAL VERIFICATION CRITERIA:
1. Partial Match Logic:
   -Exact Match: Calculate similarity percentage based on common words, character matching, and semantic similarity
   -Partial Match: Calculate similarity percentage based on common words, character matching, and semantic similarity
   -Apply fuzzy matching for text fields and tolerance ranges for numerical values

2. Cross-reference Rules:
   -For rules with interdependent requirements, cross-reference other rules to establish baseline values
   -Apply tolerance ranges and consistency checks across document types
   -Use successfully matched rules as reference points for ambiguous requirements

3. Matching Strategies:
   -Use appropriate matching method: exact, fuzzy text, numerical range, or cross-reference
   -Apply case-insensitive and format-flexible comparisons

STRING MATCH PERCENTAGE CALCULATION EXAMPLES:

**Example 1: Exact Match**
- Required: "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
- Found: "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
- string_match_percentage: 100.0
- verification_result: "Matched"

**Example 2: High Partial Match**
- Required: "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
- Found: "STEVE POWER LIMITED NORTH ROAD BUSINESS PARK SUSSEX L33 7RR UK"
- string_match_percentage: 87.5 (missing "UNITED KINGDOM", "LTD" vs "LIMITED")
- verification_result: "Partial Match (87.5%)"

**Example 3: Medium Partial Match**
- Required: "SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR"
- Found: "SPARE PARTS FOR INDUSTRIAL GENERATOR"
- string_match_percentage: 71.4 (missing "GAS POWERED")
- verification_result: "Partial Match (71.4%)"

**Example 4: Low Partial Match**
- Required: "HAPAG LLOYD OR MAERSK LINE OR PILL"
- Found: "COSCO SHIPPING"
- string_match_percentage: 15.2 (completely different shipping line)
- verification_result: "Not Found"

**Example 5: Numerical Match**
- Required: "8503.0000.00"
- Found: "8503.0000.00"
- string_match_percentage: 100.0
- verification_result: "Matched"

**Example 6: Numerical Partial Match**
- Required: "8503.0000.00"
- Found: "8503.0000"
- string_match_percentage: 83.3 (missing ".00" suffix)
- verification_result: "Partial Match (83.3%)"

MATCHING THRESHOLDS:
- 95-100%: "Matched"
- 70-94%: "Partial Match (X%)"
- Below 70%: "Not Found"

Output as JSON:
{
    "verification_results": [
        {
            "rule_id": 1,
            "rule_text": "the business rule being checked",
            "verification_result": "Matched" or "Partial Match (with % match)" or "Not Found",
            "verification_reasoning": "detailed explanation with specific evidence or why it failed",
            "evidence_found": "exact text/data found in documents or null if not found",
            "document_source": "which specific document(s) contained the evidence",
            "page_location": "specific page where evidence was found",
            "string_match_percentage": 85.5,
            "confidence_level": "High" or "Medium" or "Low"
        }
    ],
    "overall_compliance": "Matched" or "Partial Match (with % match)" or "Not Found",
    "compliance_summary": {
        "rules_matched": 0,
        "rules_partial_match": 0,
        "rules_not_found": 0
    },
    "discrepancies": [
        {
            "rule_id": 1,
            "issue_description": "clear description of the problem",
            "severity": "Major" or "Minor",
            "page_location": "Page where issue found",
            "suggested_action": "recommendation for resolving the issue"
        }
    ]
}

EXAMPLE OUTPUT WITH STRING MATCH PERCENTAGES:
```json
{
    "verification_results": [
        {
            "rule_id": 1,
            "rule_text": "BENEFICIARY'S FULL NAME AND ADDRESS must be 'STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM'",
            "verification_result": "Partial Match (87.5%)",
            "verification_reasoning": "Found beneficiary name and address with minor variations: 'LTD' instead of 'LIMITED' and 'UK' instead of 'UNITED KINGDOM'",
            "evidence_found": "STEVE POWER LIMITED NORTH ROAD BUSINESS PARK SUSSEX L33 7RR UK",
            "document_source": "Commercial Invoice",
            "page_location": "9",
            "string_match_percentage": 87.5,
            "confidence_level": "High"
        },
        {
            "rule_id": 4,
            "rule_text": "Bill of Lading must show field DESCRIPTION OF GOODS as 'SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR'",
            "verification_result": "Matched",
            "verification_reasoning": "Exact match found for description of goods in Bill of Lading",
            "evidence_found": "SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR",
            "document_source": "Bill of Lading",
            "page_location": "12",
            "string_match_percentage": 100.0,
            "confidence_level": "High"
        },
        {
            "rule_id": 6,
            "rule_text": "Bill of Lading must mention field HS CODE as '8503.0000.00'",
            "verification_result": "Not Found",
            "verification_reasoning": "HS Code not found in any Bill of Lading documents",
            "evidence_found": null,
            "document_source": "Bill of Lading",
            "page_location": null,
            "string_match_percentage": 0.0,
            "confidence_level": "High"
        }
    ]
}
```
