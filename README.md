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

COMPLIANCE_MATCH_SCORE CALCULATION METHOD:
**CRITICAL: Always calculate compliance_match_score based on word overlap, regardless of verification_result**

**Step-by-Step Calculation:**
1. Extract all significant words from both expected_value and evidence_found (ignore punctuation, case)
2. Count matching words between both texts
3. Calculate percentage: (matching_words / total_unique_words_in_expected) × 100
4. Apply this calculation INDEPENDENT of verification_result classification

**Example 1: Beneficiary Address Partial Match**
- Expected: "DEF CO LTD, YANDIAN INDUSTRIAL ZONE, YANZHOU, JINING, SHANDONG PROVINCE, CHINA"
- Found: "DEF CO., LTD INDUSTRIAL ZONE, SHANGHAI, CHINA"
- Matching words: ["DEF", "CO", "LTD", "INDUSTRIAL", "ZONE", "CHINA"] = 6 words
- Total words in expected: ["DEF", "CO", "LTD", "YANDIAN", "INDUSTRIAL", "ZONE", "YANZHOU", "JINING", "SHANDONG", "PROVINCE", "CHINA"] = 11 words
- compliance_match_score: (6/11) × 100 = 54.5%
- verification_result: "Not Found" (due to wrong location: SHANGHAI vs required YANDIAN/YANZHOU/JINING/SHANDONG)

**Example 2: Description Match**
- Expected: "SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR"
- Found: "SPARE PARTS FOR INDUSTRIAL GENERATOR HIGH POWER"
- Matching words: ["SPARE", "PARTS", "FOR", "INDUSTRIAL", "GENERATOR"] = 5 words
- Total words in expected: ["SPARE", "PARTS", "FOR", "INDUSTRIAL", "GAS", "POWERED", "GENERATOR"] = 7 words
- compliance_match_score: (5/7) × 100 = 71.4%
- verification_result: "Partial Match (71.4%)"

**Example 3: Exact Match**
- Expected: "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
- Found: "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM"
- compliance_match_score: 100.0%
- verification_result: "Matched"

**Example 4: No Match**
- Expected: "HAPAG LLOYD OR MAERSK LINE OR PILL"
- Found: "COSCO SHIPPING"
- Matching words: [] = 0 words
- compliance_match_score: 0.0%
- verification_result: "Not Found"

**Example 5: Bank Details Exact Match**
- Expected: "Bank of Rizhao, Jining, China, SWIFT CODE: RZCBCNBDJ1"
- Found: "Bank of Rizhao, Jining, China, SWIFT CODE: RZCBCNBDJ1"
- compliance_match_score: 100.0%
- verification_result: "Matched"

**IMPORTANT RULES:**
- compliance_match_score MUST be calculated for ALL rules, even "Not Found" results
- If any matching words exist, compliance_match_score cannot be 0%
- Only use 0% when there are absolutely no common significant words
- Consider variations like "LTD" vs "LIMITED", "CO" vs "COMPANY" as matches

VERIFICATION RESULT CLASSIFICATION THRESHOLDS:
- 95-100%: "Matched"
- 70-94%: "Partial Match (X%)"
- Below 70%: "Not Found"

**Note:** compliance_match_score is calculated independently and may show word overlap even for "Not Found" results when business requirements aren't met despite some text similarities.

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
            "compliance_match_score": 85.5,
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

EXAMPLE OUTPUT WITH COMPLIANCE_MATCH_SCORES:
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
            "compliance_match_score": 87.5,
            "confidence_level": "High"
        },
        {
            "rule_id": 2,
            "rule_text": "Beneficiary on DC must be: DEF CO LTD, YANDIAN INDUSTRIAL ZONE, YANZHOU, JINING, SHANDONG PROVINCE, CHINA",
            "verification_result": "Not Found",
            "verification_reasoning": "Found similar company name but wrong location: SHANGHAI instead of required YANDIAN/YANZHOU/JINING/SHANDONG area",
            "evidence_found": "DEF CO., LTD INDUSTRIAL ZONE, SHANGHAI, CHINA",
            "document_source": "Commercial Invoice, Packing List",
            "page_location": "9",
            "compliance_match_score": 54.5,
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
            "compliance_match_score": 100.0,
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
            "compliance_match_score": 0.0,
            "confidence_level": "High"
        }
    ]
}
```
