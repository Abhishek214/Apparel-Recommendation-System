## Comprehensive Written Confirmation Letter Extraction Prompt

For each written confirmation letter visible in the document:

1. **Extract the complete factual content** of this business document for documentation purposes.

2. **Include the ENTIRE letter content from opening to closing**:
   - Start from the first line after any letterhead/company header
   - Include the date, reference numbers, and recipient details  
   - Include subject line (e.g., "Re: Trust Name" or "Re: Company Name")
   - Include ALL body paragraphs containing confirmations, declarations, statements, or undertakings
   - Include any numbered/lettered clauses, bullet points, or highlighted sections
   - Include closing statements and formal undertakings
   - End with the final substantive paragraph before signature blocks

3. **Content boundaries**:
   - INCLUDE: Date, reference, recipient, subject line, complete letter body, numbered clauses, highlighted sections, closing statements
   - EXCLUDE: Company letterheads, sender addresses, signature blocks, stamps, and footer disclaimers

4. **Multiple documents**: If multiple written confirmation letters are present, return all using keys like "WCL1", "WCL2", "WCL3", etc.

5. **Signature requirement**: Only include letters that contain hand-drawn visible signatures or signature blocks. Exclude any letters that are unsigned or lack signature sections.
