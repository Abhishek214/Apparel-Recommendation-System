
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import uuid
import requests
import json

# Pydantic model for input validation
class NotificationRequest(BaseModel):
    status: str
    entities: Dict[str, Any]
    topic: str
    
    @validator('status')
    def validate_status(cls, v):
        if not v:
            raise ValueError('Status cannot be empty')
        return v
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v:
            raise ValueError('Topic cannot be empty')
        return v

def send_sample_notification(request: NotificationRequest):
    DIOS_PUBLISHER_URL = f"https://dios-sit.hsbc-1211793-dios-dev.dev.gcp.cloud.uk.hsbc/mcs-events-broker/api/v1/topics/{request.topic}/publish"
    
    unique_id = str(uuid.uuid4())
    
    DIOS_meta['id'] = unique_id
    DIOS_meta['correlationId'] = unique_id
    DIOS_meta['time'] = get_current_time()
    
    dcrest_data = {
        "data": {
            "status": {
                "type": request.status
            },
            "ocnResult": {
                "ocnData": request.entities
            }
        }
    }
    
    dcrest_data["meta"] = DIOS_meta
    
    print("Requesting authentication token...")
    dios_sso_headers['X-HSBC-Request-Correlation-Id'] = str(uuid.uuid4())
    dios_sso_headers['X-HSBC-E2E-Trust-Token'] = get_token(DSP_TRANSLATE_TOKEN_URL, sso_headers, translation_headers)
    
    if not dios_sso_headers['X-HSBC-E2E-Trust-Token']:
        print("Failed to obtain authentication token")
        return False
    
    try:
        res = requests.post(
            DIOS_PUBLISHER_URL,
            headers=dios_sso_headers,
            data=json.dumps(dcrest_data),
            verify=False
        )
        
        print(f"Response status: {res.status_code}")
        if res.status_code == 200:
            print("Notification successfully sent: {status}")
            print(f"Response: {res.json()}")
            return True
        else:
            print(f"Error sending notification: {res.text}")
            return False
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

# Example usage:
if __name__ == "__main__":
    # Create notification request with validation
    try:
        notification = NotificationRequest(
            status="success",
            entities={"key": "value", "another_key": "another_value"},
            topic="my-topic"
        )
        
        # Send notification
        send_sample_notification(notification)
        
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Example of validation error
    try:
        invalid_notification = NotificationRequest(
            status="",  # This will fail validation
            entities={},
            topic="my-topic"
        )
    except ValueError as e:
        print(f"Expected validation error: {e}")
