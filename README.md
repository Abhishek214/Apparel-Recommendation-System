# MT700 BUSINESS RULES EXTRACTION

## TASK:
Extract ALL business rules from MT700 Documentary Credit fields 45A, 46A, and 47A.

## MT700 CONTENT:
{mt700_text}

## EXTRACTION RULES:
- Each "+" symbol = new condition (separate rule)
- Each "-" continuation = part of same rule
- Extract every requirement, instruction, condition as separate rule
- Include charges, fees, banking instructions, document requirements
- Be comprehensive - extract each distinct obligation

## OUTPUT FORMAT:
```json
{
  "extracted_fields": {
    "45A": "text...",
    "46A": "text...", 
    "47A": "text..."
  },
  "business_rules": [
    {
      "rule_id": 1,
      "rule_text": "clear requirement statement",
      "document_type": "Bill of Lading|Commercial Invoice|All Documents|etc",
      "requirement_type": "exact_match|contains|tolerance_check|presence_check",
      "field_name": "beneficiary_address|quantity|etc",
      "expected_value": "specific value or null",
      "validation_note": "additional context"
    }
  ]
}
```

## DOCUMENT TYPES:
Bill of Lading, Airway Bill, Forwarder Cargo Receipt, Commercial Invoice, Draft, Bank Schedule, Certificate of Origin, Insurance Certificate, Letter of Credit, Packing List, Collection Order, Loan Application, Purchase Order, Proforma Invoice, Beneficiary Certificate, Certificate of Analysis, Fumigation Certificate, Inspection Certificate, Non Wood Certificate, Quality Certificate, Quantity Weight Certificate, Shipping Company Certificate, Test Certificate Report, Courier Receipt, Custom Declaration, Delivery Note, Shipment Advice, Weight List, Other Certificate, Other Document
