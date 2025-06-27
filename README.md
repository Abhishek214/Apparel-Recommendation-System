# BUSINESS RULES VERIFICATION PROMPT

## BUSINESS RULES TO CHECK:
{json.dumps(business_rules, indent=2)}

## IDENTIFIED TRADE DOCUMENTS:
{json.dumps(separated_docs.get('identified_documents', []), indent=2)}

## VERIFICATION INSTRUCTIONS:
For each business rule, perform the following analysis:

1. **Document Analysis**: Check if the relevant document(s) contain the required information
2. **Value Verification**: Compare found values against expected requirements using intelligent matching
3. **Evidence Collection**: Provide specific evidence from the documents with exact quotes
4. **Location Identification**: Identify the PAGE/SECTION where evidence is present
5. **Status Determination**: Classify as "Matched"/"Partial Match"/"Conflicting"/"Not Found"
6. **Document Source**: If specific document type is not mentioned in rule, then document_source is "All Documents"

## ENHANCED VERIFICATION CRITERIA:

### Partial Match Logic:
- **Partial Match**: Calculate similarity percentage based on common words, character matching, and semantic similarity
- Apply fuzzy matching for text fields and tolerance ranges for numerical values
- Account for variations in formatting, punctuation, and common abbreviations

### Cross-Reference Rules:
- For rules with interdependent requirements, cross-reference other rules to establish baseline values
- Apply tolerance ranges and consistency checks across document types
- Use successfully matched rules as reference points for ambiguous requirements

### Matching Strategies:
- Use appropriate matching method: exact, fuzzy text, numerical range, or cross-reference
- Apply case-insensitive and format-flexible comparisons

## OUTPUT FORMAT:
```json
{
  "document_summary": {
    "total_documents": {len(separated_docs.get('identified_documents', []))},
    "document_types": ["list of document types found"],
    "analysis_timestamp": "current timestamp"
  },
  "verification_results": [
    {
      "rule_id": 1,
      "rule_text": "the complete business rule being checked",
      "verification_result": "Matched|Partial Match|Conflicting|Not Found",
      "verification_reasoning": "detailed explanation including specific evidence, matching logic used, and cross-referenced information",
      "evidence_found": "exact text/data found in documents with quotes, or null if not found",
      "document_source": "specific document name(s) that contained the evidence",
      "page_location": "specific page/section where evidence was found",
      "confidence_level": "High|Medium|Low",
      "percentage_match": "X%",
      "matching_strategy": "exact|fuzzy_text|numerical_range|cross_reference|etc",
      "cross_referenced_rules": ["list of rule IDs that provided baseline values or supporting context"],
      "similarity_breakdown": {
        "common_elements": ["shared words/phrases"],
        "match_percentage": "X%"
      }
    }
  ],
  "overall_compliance": "Matched|Partial Match|Conflicting|Not Found",
  "compliance_summary": {
    "total_rules": 0,
    "rules_matched": 0,
    "rules_partial_match": 0,
    "rules_conflicting": 0,
    "rules_not_found": 0,
    "average_match_percentage": "X%"
  },
  "discrepancies": [
    {
      "rule_id": 1,
      "issue_description": "clear description of the compliance issue",
      "severity": "Critical|Major|Minor",
      "impact": "description of business impact",
      "suggested_action": "specific recommendation for resolving the issue",
      "alternative_evidence": "any related information that might be relevant"
    }
  ],
  "cross_reference_analysis": {
    "interdependent_rules": [
      {
        "primary_rule_id": 1,
        "dependent_rule_ids": [2, 3],
        "relationship_type": "baseline_value|tolerance_application|consistency_check",
        "analysis_result": "description of cross-reference findings"
      }
    ]
  }
}
```

## QUALITY ASSURANCE CHECKLIST:
- [ ] Every rule has been evaluated with appropriate matching strategy
- [ ] Partial matches include detailed similarity breakdown
- [ ] Cross-references are identified and analyzed
- [ ] Evidence quotes are exact and traceable
- [ ] Confidence levels reflect the quality of evidence
- [ ] Discrepancies include actionable recommendations
- [ ] Overall compliance accurately reflects individual rule results
