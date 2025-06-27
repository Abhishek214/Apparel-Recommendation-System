# BUSINESS RULES VERIFICATION PROMPT

## BUSINESS RULES TO CHECK:
{json.dumps(business_rules, indent=2)}

## IDENTIFIED TRADE DOCUMENTS:
{json.dumps(separated_docs.get('identified_documents', []), indent=2)}

## For each business rule:
1. Check if the relevant document(s) contain the required information
2. Verify if values match expected requirements
3. Provide specific evidence from the documents
4. Identify the PAGE/WHERE where document is present
5. Determine "Matched"/"Partial Match (with % match)"/"Not Found" status with reasoning
6. If specific document type is not mentioned in rule, then document_source is All Documents

## ADDITIONAL VERIFICATION CRITERIA:

### Partial Match Logic:
- **Partial Match** will be determined when the string we need to find in documents using the rules matches with a percentage threshold
- If the found value is within an acceptable range but not exact, mark as "Partial Match (with X% match)"
- Calculate percentage match based on how close the found value is to the required value

### Cross-Reference Rules for Ambiguous Quantities:
- For rules like "Plus or minus 5 percent in quantity and value is acceptable" where the specific quantity to refer is not mentioned:
  - Check other related rules that may contain the base quantity/value references
  - Cross-reference information present in other rules to determine the baseline values
  - If baseline quantity/value is found in other rules, apply the percentage tolerance to those values
  - Document which rule provided the baseline reference in the verification_reasoning

## Output as JSON:
```json
{
  "document_summary": {
    "total_documents": {len(separated_docs.get('identified_documents', []))},
    "document_types": ["list of document types found"]
  },
  "verification_results": [
    {
      "rule_id": 1,
      "rule_text": "the business rule being checked",
      "verification_result": "Matched" or "Partial Match (with X% match)" or "Conflicting" or "Not Found",
      "verification_reasoning": "detailed explanation with specific evidence or why it failed, including cross-referenced rules if applicable",
      "evidence_found": "exact text/data found in documents or null if not found",
      "document_source": "which specific document(s) contained the evidence",
      "page_location": "specific page where evidence was found",
      "confidence_level": "High" or "Medium" or "Low",
      "percentage_match": "X%" (only for partial matches),
      "cross_referenced_rules": ["list of rule IDs that provided baseline values"] (if applicable)
    }
  ],
  "overall_compliance": "Matched" or "Partial Match (with X% match)" or "Conflicting" or "Not Found",
  "compliance_summary": {
    "rules_matched": 0,
    "rules_partial_match": 0,
    "rules_conflicting": 0,
    "rules_not_found": 0
  },
  "discrepancies": [
    {
      "rule_id": 1,
      "issue_description": "clear description of the problem",
      "severity": "Major" or "Minor",
      "suggested_action": "recommendation for resolving the issue"
    }
  ]
}
```
