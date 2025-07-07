# COMPREHENSIVE INVESTMENT MANAGEMENT AGREEMENT (IMA) DATA EXTRACTION PROMPT

## OBJECTIVE
Extract comprehensive data attributes from any Investment Management Agreement document with multiple supporting references and exact page citations for each attribute.

## INSTRUCTIONS

### Step 1: Document Analysis
First, identify the document title, total pages, and current extraction date.

### Step 2: Comprehensive Attribute Extraction
For each of the 17 defined attributes below, extract ALL relevant information found throughout the document. Scan the entire document including main text, definitions, schedules, annexes, exhibits, and signature pages.

### Step 3: Multiple Reference Documentation  
For each extracted attribute, provide:
- **Summary**: A comprehensive summary of all key findings for this attribute
- **Supporting References**: ALL relevant direct quotes from the document with exact page numbers and context
- **Key Details**: Specific extracted details in structured format

## ATTRIBUTE DEFINITIONS

### 1. Term
Extract all information about contract duration, effective dates, renewal provisions, termination conditions, and notice requirements for ending the agreement.

### 2. Title 
Extract the formal document name, agreement type, and any alternative titles or defined terms for the agreement.

### 3. Access to Records
Extract all provisions about record access rights, reporting obligations, record maintenance duties, information sharing permissions, and data usage rights.

### 4. Client Name
Extract all entity names that are clients, including primary partnerships/funds, subsidiary entities, general partners, and any entities listed in annexes or schedules.

### 5. Investment Manager Name
Extract the full legal name and details of the investment management company or advisor entity.

### 6. Notices
Extract all notice requirements, delivery methods, addresses, communication protocols, and procedures for providing notices between parties.

### 7. Indemnity
Extract all indemnification clauses, protection provisions, and circumstances where one party agrees to protect another from liabilities.

### 8. Liability
Extract liability allocation provisions, limitations, exclusions, standard of care requirements, and definitions of conduct that creates or limits liability.

### 9. Agreement Date
Extract the date when the agreement was made, entered into, or became effective.

### 10. Execution Date
Extract dates when the agreement was signed, executed, or formally completed by the parties.

### 11. Confidentiality
Extract confidentiality provisions, information protection clauses, non-disclosure obligations, and restrictions on information sharing.

### 12. Representations (Company)
Extract all representations and warranties made by the client/partnership about their authority, legal status, compliance, and ability to enter into the agreement.

### 13. Scope of Authorization
Extract all provisions granting discretionary authority, investment powers, decision-making rights, and specific authorizations to the investment manager.

### 14. Impermissible Actions
Extract all restrictions, prohibited activities, limitations on authority, and actions requiring specific consent or approval.

### 15. Exclusive Jurisdiction
Extract any clauses specifying exclusive jurisdiction, court venues, or dispute resolution forums.

### 16. Delegation of Authority
Extract provisions allowing delegation of authority, sub-delegation rights, assignment of responsibilities to affiliates or third parties.

### 17. Governing Law/Jurisdiction
Extract applicable law specifications, jurisdiction clauses, legal venue requirements, and conflict of laws provisions.

## EXTRACTION GUIDELINES

### Comprehensive Coverage Requirements:
1. **Complete Document Scan**: Read the ENTIRE document for each attribute - do not stop at first occurrence
2. **Multiple Instances**: Find ALL sections that mention or relate to each attribute throughout the document
3. **Exact Quotations**: Use exact text from the document - no paraphrasing in supporting references
4. **Page Precision**: Provide exact page number for every quote
5. **Contextual Relevance**: Explain how each quote specifically supports the attribute
6. **Cross-References**: Include related provisions and cross-referenced sections

### Special Attention Areas:
- Main agreement clauses
- Definitions sections
- Schedules and annexes
- Signature pages and execution details
- Amendment or modification provisions
- Exhibits and attachments

### Quality Requirements:
- If an attribute appears in multiple forms, capture ALL instances
- Include both explicit statements and implicit provisions
- Note any conflicts or variations in how the same attribute is described
- Capture information even if spread across different document sections

## OUTPUT FORMAT

Return results in the following JSON structure:

```json
{
  "document_analysis": {
    "document_title": "extracted title",
    "total_pages": "number",
    "extraction_date": "current date"
  },
  "extracted_attributes": {
    "term": {
      "summary": "Comprehensive summary of all findings related to agreement duration and termination",
      "supporting_references": [
        {
          "quote": "Exact text from document",
          "page_number": "X",
          "context": "Brief explanation of relevance"
        }
      ],
      "key_details": {
        "effective_date": "specific date if found",
        "termination_notice": "notice period required",
        "renewal_provisions": "any renewal terms"
      }
    },
    "title": {
      "summary": "Document title and formal name information",
      "supporting_references": [
        {
          "quote": "Exact title text from document",
          "page_number": "X",
          "context": "Main document title"
        }
      ],
      "key_details": {
        "formal_title": "full formal title",
        "abbreviated_reference": "any defined abbreviation"
      }
    },
    "access_to_records": {
      "summary": "All provisions related to record access, maintenance, and reporting",
      "supporting_references": [
        {
          "quote": "Relevant quote about record access",
          "page_number": "X",
          "context": "How this relates to record access rights"
        }
      ],
      "key_details": {
        "access_rights": "who has access to what",
        "reporting_obligations": "required reports",
        "record_ownership": "who owns the records"
      }
    },
    "client_name": {
      "summary": "All client entities and related parties",
      "supporting_references": [
        {
          "quote": "Text identifying client entities",
          "page_number": "X",
          "context": "Primary client identification"
        }
      ],
      "key_details": {
        "primary_entity": "main client name",
        "entity_type": "legal structure",
        "subsidiaries": "any subsidiary information"
      }
    },
    "investment_manager_name": {
      "summary": "Investment manager entity details",
      "supporting_references": [
        {
          "quote": "Investment manager identification text",
          "page_number": "X",
          "context": "Manager entity details"
        }
      ],
      "key_details": {
        "entity_name": "full legal name",
        "entity_type": "legal structure",
        "registration": "regulatory registration details"
      }
    },
    "notices": {
      "summary": "Notice requirements and communication procedures",
      "supporting_references": [
        {
          "quote": "Notice provision text",
          "page_number": "X",
          "context": "Notice requirements"
        }
      ],
      "key_details": {
        "delivery_method": "how notices must be sent",
        "address_requirements": "address specifications"
      }
    },
    "indemnity": {
      "summary": "Indemnification provisions and protections",
      "supporting_references": [
        {
          "quote": "Indemnification clause text",
          "page_number": "X",
          "context": "Indemnity obligations"
        }
      ],
      "key_details": {
        "indemnifying_party": "who provides indemnity",
        "scope": "what is covered",
        "exceptions": "any limitations"
      }
    },
    "liability": {
      "summary": "Liability limitations, exclusions, and standards",
      "supporting_references": [
        {
          "quote": "Liability provision text",
          "page_number": "X",
          "context": "Liability limitations"
        }
      ],
      "key_details": {
        "limitation_scope": "what is limited",
        "exceptions": "conduct that creates liability",
        "standard_of_care": "applicable standards"
      }
    },
    "agreement_date": {
      "summary": "Date agreement was made or entered into",
      "supporting_references": [
        {
          "quote": "Date reference text",
          "page_number": "X",
          "context": "Agreement date"
        }
      ],
      "key_details": {
        "effective_date": "when agreement takes effect"
      }
    },
    "execution_date": {
      "summary": "Date agreement was signed or executed",
      "supporting_references": [
        {
          "quote": "Execution date text",
          "page_number": "X",
          "context": "Signature/execution details"
        }
      ],
      "key_details": {
        "signature_date": "when signed"
      }
    },
    "confidentiality": {
      "summary": "Confidentiality and information protection provisions",
      "supporting_references": [
        {
          "quote": "Confidentiality provision text",
          "page_number": "X",
          "context": "Information protection requirements"
        }
      ],
      "key_details": {
        "scope": "what information is protected",
        "obligations": "confidentiality duties"
      }
    },
    "representations_company": {
      "summary": "Representations and warranties made by the client/company",
      "supporting_references": [
        {
          "quote": "Representation text",
          "page_number": "X",
          "context": "Company representations"
        }
      ],
      "key_details": {
        "authority": "authority to execute",
        "compliance": "legal compliance representations",
        "status": "entity status confirmations"
      }
    },
    "scope_of_authorization": {
      "summary": "Investment manager's authorized powers and discretionary authority",
      "supporting_references": [
        {
          "quote": "Authorization text",
          "page_number": "X",
          "context": "Granted authority"
        }
      ],
      "key_details": {
        "discretionary_authority": "level of discretion",
        "specific_powers": "particular authorizations",
        "limitations": "any restrictions"
      }
    },
    "impermissible_actions": {
      "summary": "Restrictions and prohibited activities",
      "supporting_references": [
        {
          "quote": "Restriction text",
          "page_number": "X",
          "context": "Prohibited actions"
        }
      ],
      "key_details": {
        "prohibited_activities": "what cannot be done",
        "consent_required": "actions requiring approval"
      }
    },
    "exclusive_jurisdiction": {
      "summary": "Exclusive jurisdiction and venue specifications",
      "supporting_references": [
        {
          "quote": "Jurisdiction clause text",
          "page_number": "X",
          "context": "Exclusive jurisdiction provisions"
        }
      ],
      "key_details": {
        "exclusive_venue": "specified court or jurisdiction"
      }
    },
    "delegation_of_authority": {
      "summary": "Authority delegation and assignment provisions",
      "supporting_references": [
        {
          "quote": "Delegation provision text",
          "page_number": "X",
          "context": "Delegation rights"
        }
      ],
      "key_details": {
        "delegation_scope": "what can be delegated",
        "permitted_delegates": "who can receive delegated authority"
      }
    },
    "governing_law_jurisdiction": {
      "summary": "Applicable law and jurisdiction provisions",
      "supporting_references": [
        {
          "quote": "Governing law text",
          "page_number": "X",
          "context": "Applicable law specification"
        }
      ],
      "key_details": {
        "governing_law": "applicable state/country law",
        "conflict_of_laws": "conflict provisions"
      }
    }
  }
}
```

## FINAL VALIDATION

Before submitting results, ensure:
- All 17 attributes have been addressed
- Each attribute has comprehensive supporting references where information exists
- Page numbers are accurate for all quotes
- No relevant information has been missed from any section of the document
- JSON format is valid and complete
- If no information is found for an attribute, still include it with "summary": "No relevant information found in document"
