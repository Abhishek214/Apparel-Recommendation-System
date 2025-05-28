
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, Dict, Any
from datetime import datetime
import requests
import json
from uuid import uuid4

# Pydantic Models for Input Validation
class NotificationStatus(BaseModel):
    """Status information for a notification"""
    type: str
    ocnResult: Optional[str] = None
    ocnData: Optional[Dict[str, Any]] = None

class NotificationData(BaseModel):
    """Main notification data structure"""
    data: Dict[str, Any]
    status: NotificationStatus

class NotificationRequest(BaseModel):
    """Request model for sending notifications"""
    status: str
    entities: Dict[str, Any]
    topic: str
    
    # Optional fields with defaults
    id: str = Field(default_factory=lambda: str(uuid4()))
    correlationId: str = Field(default_factory=lambda: str(uuid4()))
    time: datetime = Field(default_factory=datetime.now)
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['success', 'failed', 'pending', 'processing']
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of: {allowed_statuses}')
        return v
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v.strip():
            raise ValueError('Topic cannot be empty')
        return v

class DIOSConfig(BaseModel):
    """Configuration for DIOS connection"""
    publisher_url: HttpUrl
    auth_token: str
    sso_headers: Dict[str, str]
    
    @validator('auth_token')
    def validate_token(cls, v):
        if not v.startswith('X-HSBC-E2E-Trust-Token'):
            raise ValueError('Invalid token format')
        return v

# Refactored notification sender class
class DIOSNotificationSender:
    def __init__(self, config: DIOSConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.sso_headers)
    
    def send_notification(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Send a notification to DIOS"""
        # Set DIOS metadata
        dios_meta = {
            'id': notification.id,
            'correlationId': notification.correlationId,
            'time': notification.time.isoformat()
        }
        
        # Prepare notification data
        notification_data = NotificationData(
            data={
                "status": {
                    "type": notification.status,
                    "ocnResult": {
                        "ocnData": notification.entities
                    }
                }
            },
            status=NotificationStatus(
                type=notification.status,
                ocnData=notification.entities
            )
        )
        
        # Prepare request payload
        payload = {
            'DIOS_PUBLISHER_URL': str(self.config.publisher_url),
            'headers': self.config.sso_headers,
            'data': json.dumps(notification_data.dict()),
            'verify': False
        }
        
        try:
            response = self.session.post(
                str(self.config.publisher_url),
                json=notification_data.dict(),
                headers={'X-DIOS-Meta': json.dumps(dios_meta)},
                verify=False
            )
            
            response.raise_for_status()
            
            return {
                'status': 'success',
                'response': response.json() if response.text else None,
                'status_code': response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e),
                'details': 'Failed to send notification'
            }

# Example usage
def main():
    # Configure DIOS connection with Pydantic validation
    config = DIOSConfig(
        publisher_url="https://dios-sit.hsbc-1211793-dios-dev.dev.gcp.cloud.uk.hsbc/mcs-events-broker/api/v1/topics/{topic}/publish",
        auth_token="X-HSBC-E2E-Trust-Token:your-token-here",
        sso_headers={
            'X-HSBC-Request-Correlation-Id': str(uuid4()),
            'X-HSBC-E2E-Trust-Token': 'your-token-here',
            'DSP_TRANSLATE_TOKEN_URL': 'sso_headers',
            'translation_headers': 'headers'
        }
    )
    
    # Create notification request with validation
    notification = NotificationRequest(
        status="success",
        entities={
            "entity1": "value1",
            "entity2": "value2"
        },
        topic="my-topic"
    )
    
    # Send notification
    sender = DIOSNotificationSender(config)
    result = sender.send_notification(notification)
    
    print(f"Notification sent: {result}")

# Example with error handling
def send_with_validation():
    try:
        # This will raise validation error
        invalid_notification = NotificationRequest(
            status="invalid_status",  # This will fail validation
            entities={},
            topic=""  # This will also fail
        )
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Correct usage
    try:
        valid_notification = NotificationRequest(
            status="success",
            entities={"key": "value"},
            topic="valid-topic"
        )
        print(f"Valid notification created: {valid_notification.dict()}")
    except ValueError as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
    send_with_validation()
