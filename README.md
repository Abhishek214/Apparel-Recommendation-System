# Written Confirmation Extraction Prompt

For each document containing confirmatory language ("we confirm", "we declare", "we certify", "we undertake", etc.):

## Extract:
- Document subject/title (e.g., "Re: Company Name", "DECLARATION OF TRUST")
- All paragraphs with confirmatory statements
- Numbered/lettered declarative clauses
- Structured confirmatory information (including highlighted/boxed content)
- Witness clauses and execution statements
- Dates within substantive content

## Exclude:
- Letterheads, addresses, contact details
- Signature blocks and handwritten signatures
- Legal disclaimers and certification stamps

## Requirements:
- Only include documents with visible signatures or formal execution sections
- For multiple documents, use keys "WCL1", "WCL2", "WCL3", etc.
- Include all document types: legal opinions, trust confirmations, due diligence letters, declarations, attestations, corporate confirmations

## Content Scope:
Start from document title/subject through final substantive paragraph before signatures.
