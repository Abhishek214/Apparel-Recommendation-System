
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


{
  "formTitle": "Passport",
  "documentInformation": {
    "documentType": "P",
    "passportNo": "",
    "personalDetails": {
      "surname": "",
      "givenNames": "",
      "sex": "",
      "nationality": "",
      "dateOfExpiry": "",
      "dateOfBirth": ""
    }
  },
  "outputColumn": {
    "header": "Output",
    "values": {}
  }
}




{
  "formTitle": "HKID",
  "documentInformation": {
    "documentType": "I",
    "personalDetails": {
      "nameInEnglish": "",
      "nameInChinese": "",
      "idNumber": "",
      "sex": "",
      "dateOfBirth": "",
      "chineseCommercialCode": ""
    }
  },
  "outputColumn": {
    "header": "Output",
    "values": {}
  },
  "interfaceDetails": {
    "autoSave": "Off",
    "currentView": "Chart1",
    "status": "Ready",
    "accessibility": "Investigate"
  }
}




import requests
import json
import uuid
from datetime import datetime
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.constants import DIOS_meta, dios_sso_headers, sso_headers, translation_headers, DIOS_PUBLISHER_URL, DSP_TRANSLATE_TOKEN_URL
from auth.dsp_auth import get_token

def get_current_time():
    """Format the current time in ISO 8601 format"""
    current_time = datetime.utcnow()
    formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return formatted_time

def send_sample_notification():
    """Send a sample notification to the broker service"""
    
    # Create a unique ID for correlation
    unique_id = str(uuid.uuid4())
    
    # Sample request IDs (would normally come from your application)
    request_ids = {
        "wcs_id": "SAMPLE123456",
        "document-id": "DOC987654321"
    }
    
    # Format request IDs as expected by the service
    id_dict = {}
    for _id in request_ids:
        if _id == "wcs_id":
            id_dict["wcscaseid"] = request_ids[_id]
        else:
            id_dict[_id.replace("-","").replace("id","Id")] = request_ids[_id]
    
    # Sample of document classification response
    sample_status = "DOCTAGGING_COMPLETED"
    sample_doctype = "UTILITY_BILL"
    processing_time = "0.75secs"
    
    # Sample of extracted entities (simplified for example)
    sample_entities = {
        "values": [
            {
                "kv_pairs": {
                    "values": {
                        "address.fullAddress": "123 Sample Street, London",
                        "pincode": "EC1A 1BB",
                        "accountNumber": "ABC123456789",
                        "customerName": "John Smith",
                        "billDate": "2025-05-15"
                    }
                }
            }
        ]
    }
    
    # Sample authentication results (all passed)
    sample_auth_results = {
        "address_verification": {"checkStatus": True, "details": "Address verified successfully"},
        "metadata_check": {"checkStatus": True, "details": "Metadata validation passed"},
        "quality_score": {"checkStatus": True, "score": 0.92, "details": "Image quality acceptable"}
    }
    
    # Prepare DIOS metadata
    DIOS_meta['id'] = unique_id
    DIOS_meta['correlationid'] = unique_id
    DIOS_meta['time'] = get_current_time()
    
    # Format data based on status (in this example, using DOCTAGGING_COMPLETED)
    docType = sample_doctype if sample_status == "DOCTAGGING_COMPLETED" else ""
    processingTime = processing_time if sample_status == "DOCTAGGING_COMPLETED" else ""
    final_entities = sample_entities if sample_status == "OCR_COMPLETED" else ""
    
    # Convert authentication results to JSON format if they're strings
    # (In this example they're already dictionaries, but including for completeness)
    def convert_to_json(data):
        if isinstance(data, str):
            return json.loads(data)
        return data
    
    addr_verification = convert_to_json(sample_auth_results['address_verification'])
    metadata_verification = convert_to_json(sample_auth_results['metadata_check'])
    image_verification = convert_to_json(sample_auth_results['quality_score'])
    
    # Determine if there's any authentication failure
    reasonForauthFailure = ""
    if not (addr_verification['checkStatus'] and metadata_verification['checkStatus'] and image_verification['checkStatus']):
        if not addr_verification['checkStatus']:
            reasonForauthFailure = {"checkStatus": False, "reason": "Address verification Failed"}
        elif not metadata_verification['checkStatus']:
            reasonForauthFailure = {"checkStatus": False, "reason": "Metadata verification Failed"}
        else:
            reasonForauthFailure = {"checkStatus": False, "reason": "Image quality check Failed"}
    
    # Construct the final response payload
    dcrest_data = {
        "data": {
            "status": sample_status,
            "tagResult": {
                "docType": docType,
                "processingTime": processingTime
            },
            "ocrResult": {
                "ocrData": final_entities,
                "processingTime": processingTime,
            },
            "authResult": {
                "metadataCheck": metadata_verification,
                "councilAddressCheck": addr_verification,
                "imageQualityCheck": image_verification
            },
            "reasonForauthFailure": reasonForauthFailure
        }
    }
    
    # Add request IDs to the data
    dcrest_data["data"].update({"id_dict": id_dict})
    
    # Add metadata to the response
    dcrest_data["meta"] = DIOS_meta
    
    # Get authentication token
    print("Requesting authentication token...")
    dios_sso_headers['X-HSBC-Request-Correlation-Id'] = str(uuid.uuid4())
    dios_sso_headers['X-HSBC-E2E-Trust-Token'] = get_token(DSP_TRANSLATE_TOKEN_URL, sso_headers, translation_headers)
    
    if not dios_sso_headers['X-HSBC-E2E-Trust-Token']:
        print("Failed to obtain authentication token")
        return False
    
    # Print the data being sent (for debugging)
    print("\nSending the following data to broker:")
    print(json.dumps(dcrest_data, indent=2))
    
    # Send the request to the broker service
    print(f"\nSending request to: {DIOS_PUBLISHER_URL}")
    
    try:
        # In a real scenario, you would use verify=True in production
        res = requests.post(
            DIOS_PUBLISHER_URL, 
            headers=dios_sso_headers, 
            data=json.dumps(dcrest_data), 
            verify=False
        )
        
        # Print the response
        print(f"Response status: {res.status_code}")
        if res.status_code == 200:
            print(f"Notification successfully sent: {sample_status}")
            print(f"Response: {res.json()}")
            return True
        else:
            print(f"Error sending notification: {res.text}")
            return False
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    # When running this script directly, send a sample notification
    send_sample_notification()
