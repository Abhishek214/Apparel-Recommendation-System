# ENHANCED INVESTMENT MANAGEMENT AGREEMENT (IMA) DATA EXTRACTION PROMPT

## OBJECTIVE
Extract comprehensive data attributes from Investment Management Agreement documents with multiple supporting references and exact page citations for each attribute.

## INSTRUCTIONS

### Step 1: Comprehensive Attribute Extraction
For each of the following data attributes, extract ALL relevant information found throughout the document. Do not stop at the first occurrence - scan the entire document for multiple mentions, variations, or related clauses.

### Step 2: Multiple Reference Documentation  
For each extracted attribute, provide:
- **Summary**: A concise summary of the key information
- **Multiple Supporting Quotes**: ALL relevant direct quotes from the document that support this attribute
- **Page References**: Exact page number for each supporting quote
- **Context**: Brief explanation of how each quote relates to the attribute

### Required Data Attributes to Extract:

1. **Term/Duration** - Contract duration, start/end dates, renewal provisions, termination conditions
2. **Title** - Formal document name, agreement type, any alternative titles
3. **Access** - Access rights, information sharing permissions, data usage rights
4. **Client Name** - All entity names that are clients (partnerships, funds, subsidiaries)
5. **Investment Manager Name** - Investment advisor/manager entity names and details
6. **Notices** - Notice requirements, delivery methods, addresses, communication protocols
7. **Indemnity** - Indemnification clauses, protection provisions, liability limitations
8. **Liability** - Liability allocation, limitations, exclusions, standard of care
9. **Arbitration** - Dispute resolution mechanisms, arbitration clauses, legal proceedings
10. **Agreement Date** - Execution date, effective date, signature dates
11. **Execution Date** - When agreement was signed/executed
12. **Confidentiality** - Confidentiality provisions, information protection clauses
13. **Representations** - Representations and warranties by all parties
14. **Scope of Authorization** - Investment authority, decision-making powers, discretionary rights
15. **Permissible Actions** - Allowed activities, investment restrictions, prohibited actions
16. **Exclusive Jurisdiction** - Governing law, jurisdiction clauses, legal venue
17. **Delegation of Authority** - Authority delegation provisions, sub-delegation rights
18. **Governing Law/Jurisdiction** - Applicable law, jurisdiction, venue specifications

### Output Format Requirements:

Return results in the following JSON structure:

```json
{
  "document_analysis": {
    "document_title": "extracted title",
    "total_pages": "number",
    "extraction_date": "current date"
  },
  "extracted_attributes": {
    "attribute_name": {
      "summary": "Concise summary of the key findings for this attribute",
      "supporting_references": [
        {
          "quote": "Exact text from document",
          "page_number": "X",
          "context": "Brief explanation of relevance"
        },
        {
          "quote": "Another relevant quote",
          "page_number": "Y", 
          "context": "How this quote supports the attribute"
        }
      ],
      "key_details": {
        "detail_1": "specific extracted detail",
        "detail_2": "another specific detail"
      }
    }
  }
}
```

### Critical Extraction Guidelines:

1. **Completeness**: Scan the ENTIRE document - do not stop after finding the first reference
2. **Accuracy**: Use exact quotes - no paraphrasing in the supporting references
3. **Multiple Sources**: Find ALL sections that mention or relate to each attribute
4. **Page Precision**: Provide exact page numbers for every quote
5. **Contextual Relevance**: Explain how each quote specifically supports the attribute
6. **Comprehensive Coverage**: Extract information even if it appears in multiple forms (e.g., main clauses, definitions, schedules, annexes)

### Special Instructions:

- If an attribute appears in multiple sections (e.g., definitions, main clauses, schedules), capture ALL instances
- Include cross-references and related provisions
- Capture both explicit statements and implicit provisions
- Note any conflicts or variations in how the same attribute is described
- Include information from signatures, annexes, and exhibits
- Pay attention to amendments, modifications, or qualifications to standard provisions

### Quality Check:
Before finalizing, ensure:
- All 18 attributes have been addressed
- Each attribute has multiple supporting quotes where available
- Page numbers are accurate
- No relevant information has been missed
- JSON format is valid and complete
