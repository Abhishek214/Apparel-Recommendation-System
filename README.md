# Sample Compliant Trade Documents

Based on the extracted business rules, here are sample documents that would **PASS** the verification checks:

## Document 1: Compliant Bill of Lading

```
ORIGINAL BILL OF LADING

Shipper: STEVE POWER LTD.
NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM

Consignee: TO ORDER OF ABC FOODS LIMITED
FOX STREET, SUFFEX, UK

Notify Party: W. SAUNDERS (SHIPPING AND FORWARDING) LTD
8 VICTORIA STREET, FELIXSTOWE, IP11 7ER
TEL +44 1394 672 244, FAX +44 1394 672 266

Description of Goods: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR
H.S. CODE: 8503.0000.00
FORM M NUMBER: FM2024001234

Quantity: 500 PIECES
Net Weight: 5000 KGS
Gross Weight: 5200 KGS

Port of Loading: QINGDAO, CHINA
Port of Discharge: FELIXSTOWE, UK

Vessel: HAPAG LLOYD VESSEL
Voyage Number: HL240801

FREE TIME 21 DAYS COMBINED (DETENTION AND DEMURRAGE) AT PORT OF DISCHARGE

Freight: PREPAID
Date of Issue: 15 JUL 2024

Signed for HAPAG LLOYD
Master's Signature: [Signature]
```

## Document 2: Compliant Supplier's Certificate of Production

```
SUPPLIER'S CERTIFICATE OF PRODUCTION

Certificate No: SCP-2024-0789
Date: 15 JUL 2024

From: STEVE POWER LTD.
NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM

To: ABC FOODS LIMITED
FOX STREET, SUFFEX, UK

We hereby certify that we have manufactured and supplied the following goods:

Description of Goods: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR
Quantity: 500 PIECES
FORM M NUMBER: FM2024001234

Manufacturing Date: 10-14 JUL 2024
Quality Standard: ISO 9001:2015
Country of Origin: UNITED KINGDOM

This certificate is issued based on our production records and quality control procedures.

STEVE POWER LTD.
[Company Seal]
Authorized Signature: [Signature]
Name: John Smith, Production Manager
Date: 15 JUL 2024
```

## Document 3: Compliant Packing List (No Unit Price/Total Value)

```
PACKING LIST

From: STEVE POWER LTD.
NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM

To: ABC FOODS LIMITED
FOX STREET, SUFFEX, UK

Packing List No: PL-2024-0567
Date: 15 JUL 2024

Description of Goods: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR
FORM M NUMBER: FM2024001234

Packing Details:
- Total Pieces: 500 PIECES
- Total Packages: 50 CARTONS
- Net Weight: 5000 KGS
- Gross Weight: 5200 KGS
- Dimensions: 100cm x 80cm x 60cm per carton

Container Details:
- Container No: HLLU1234567
- Seal No: SL789456
- Container Type: 20' DRY

Port of Loading: QINGDAO, CHINA
Port of Discharge: FELIXSTOWE, UK

STEVE POWER LTD.
[Company Seal]
Authorized by: [Signature]
Date: 15 JUL 2024
```

## Document 4: Compliant Commercial Invoice (Can show Unit Price)

```
COMMERCIAL INVOICE

From: STEVE POWER LTD.
NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM

To: ABC FOODS LIMITED
FOX STREET, SUFFEX, UK

Invoice No: CI-2024-0890
Date: 15 JUL 2024

Purchase Order Ref: ABC PO 10696
Documentary Credit No: DC UK1071320

Description of Goods: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR
Quantity: 500 PIECES
Unit Price: USD 200.10 per PIECE
Total Amount: USD 100,050.00

FORM M NUMBER: FM2024001234
Country of Origin: UNITED KINGDOM
Terms of Payment: L/C at Sight

Total Invoice Value: USD 100,050.00
(Say: US Dollars One Hundred Thousand and Fifty Only)

STEVE POWER LTD.
[Company Seal]
Authorized Signature: [Signature]
Name: Sarah Johnson, Export Manager
Date: 15 JUL 2024
```

## Document 5: Compliant Certificate of Origin (No Unit Price/Total Value)

```
CERTIFICATE OF ORIGIN

Certificate No: CO-2024-1122
Date: 15 JUL 2024

Exporter: STEVE POWER LTD.
NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM

Consignee: ABC FOODS LIMITED
FOX STREET, SUFFEX, UK

Description of Goods: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR
Quantity: 500 PIECES
FORM M NUMBER: FM2024001234

Country of Origin: UNITED KINGDOM
Harmonized System Code: 8503.0000.00

Transport Details:
Port of Loading: QINGDAO, CHINA
Port of Discharge: FELIXSTOWE, UK
Vessel: HAPAG LLOYD VESSEL

I hereby certify that the goods described above originate from the United Kingdom.

UK CHAMBER OF COMMERCE
[Official Seal]
Authorized Officer: [Signature]
Name: Michael Brown
Title: Certification Officer
Date: 15 JUL 2024
```

## Expected Verification Results with These Documents

With these compliant documents, the verification results would be:

```json
{
  "verification_results": [
    {
      "rule_id": 1,
      "rule_text": "BENEFICIARY'S FULL NAME AND ADDRESS must be 'STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM'",
      "verification_result": "Passed",
      "verification_reasoning": "All documents show the correct beneficiary name and address as required",
      "evidence_found": "STEVE POWER LTD. NORTH ROAD, BUSINESS PARK, SUSSEX L33 7RR UNITED KINGDOM",
      "document_source": "All Documents"
    },
    {
      "rule_id": 2,
      "rule_text": "BILLS OF LADING and SUPPLIER'S CERTIFICATE OF PRODUCTION must show QUANTITY, FORM M NUMBER and DESCRIPTION OF GOODS as 'SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR' ONLY",
      "verification_result": "Passed",
      "verification_reasoning": "Both documents show correct description, quantity (500 PIECES), and Form M Number (FM2024001234)",
      "evidence_found": "Description: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR, Quantity: 500 PIECES, FORM M NUMBER: FM2024001234",
      "document_source": "Bill of Lading and Supplier's Certificate"
    },
    {
      "rule_id": 3,
      "rule_text": "BILLS OF LADING must mention H.S. CODE: 8503.0000.00",
      "verification_result": "Passed",
      "verification_reasoning": "Bill of Lading clearly shows the required H.S. Code",
      "evidence_found": "H.S. CODE: 8503.0000.00",
      "document_source": "Bill of Lading"
    },
    {
      "rule_id": 4,
      "rule_text": "ALL DOCUMENTS EXCEPT DRAFT AND COMMERCIAL INVOICE must NOT show the UNIT PRICE, TOTAL VALUE and THIS D.C NO.",
      "verification_result": "Passed",
      "verification_reasoning": "Non-commercial documents (Bill of Lading, Packing List, Certificate of Origin, Supplier's Certificate) do not show unit prices or total values. Only Commercial Invoice shows these details, which is allowed",
      "evidence_found": "Commercial Invoice shows pricing (allowed), other documents do not show pricing",
      "document_source": "All Documents Verified"
    },
    {
      "rule_id": 5,
      "rule_text": "ALL DOCUMENTS EXCEPT DRAFT must show FORM M NUMBER",
      "verification_result": "Passed",
      "verification_reasoning": "All submitted documents show the required Form M Number",
      "evidence_found": "FORM M NUMBER: FM2024001234",
      "document_source": "All Documents"
    }
  ],
  "overall_compliance": "Passed",
  "discrepancies": []
}
```

## Key Compliance Features in These Sample Documents

1. **✅ Correct Beneficiary**: STEVE POWER LTD. with exact address
2. **✅ Correct Description**: SPARE PARTS FOR INDUSTRIAL GAS POWERED GENERATOR  
3. **✅ H.S. Code Present**: 8503.0000.00 in Bill of Lading
4. **✅ Form M Number**: FM2024001234 in all required documents
5. **✅ Pricing Rules**: Only Commercial Invoice shows unit price/total value
6. **✅ Shipping Line**: HAPAG LLOYD (one of the approved carriers)
7. **✅ Free Time Clause**: 21 days combined detention/demurrage mentioned

## How to Test These Documents

You can use these sample documents in your LLM system to verify that it correctly identifies **compliant** documents and returns "Passed" results instead of "Need Review". This demonstrates the system's ability to distinguish between compliant and non-compliant documentation.
