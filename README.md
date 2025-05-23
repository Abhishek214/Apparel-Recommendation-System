
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF



import os
import uuid
import platform
import uvicorn
import logging
import traceback
import json
import requests
from typing import Dict

from fastapi import FastAPI, status, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Constants from constants.py 
try:
    SVC_AC_ID = os.getenv("HKBBPM_SVC_ID")
    SVC_AC_PWD = os.getenv("HKBBPM_SVC_PWD")
except Exception as e:
    logging.exception("Environment variables not found")

# CDG API endpoints
DSP_FQDN = "https://cmb-1b2b-dsp-pprod-ap.hk.hsbc:8443"
PROD_DSP_FQDN = "https://cmb-1b2b-dsp-ap.hk.hsbc:8443"
DSP_GET_TOKEN_URL = PROD_DSP_FQDN + "/dsp/json/realms/root/realms/DSP_1B2B/authenticate?authIndexType=service&authIndexValue=1B2B_ldapService"
DSP_TRANSLATE_TOKEN_URL = PROD_DSP_FQDN + "/dsp/rest-sts/DSP_1B2B/1B2B_tokenTranslator?_action=translate"

# CDG API download endpoints
download_api_dev = "https://digitaldev-int-cmb.hk.hsbc/cmb-cdg-service-sherlock-cert-dev-internal-proxy/v1/cdg/sherlock/download/"
download_api_uat = "https://wsit-dev-int-dbbhk-api.lkp301x.cloud.hk.hsbc/cmb-cdg-service-hkbpm-dev-internal-proxy/v1/cdg/hkbpm/document?doc_id="
download_api_prod = "https://wsit-int-dbbhk-api.lkp201.cloud.hk.hsbc/cmb-cdg-service-hkbpm-prod-internal-proxy/v1/cdg/hkbpm/document?doc_id="

# Headers for CDG API requests
sso_headers = {
    "Accept-API-Version": "resource=2.0, protocol=1.0",
    "Content-Type": "application/json",
    "X-OpenAM-Username": SVC_AC_ID,
    "X-OpenAM-Password": SVC_AC_PWD,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Connection": "close"
}

translation_headers = {"Content-Type": "application/json"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/dcrest_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dcrest_api")

# Initialize FastAPI app
app = FastAPI(
    title="DCREST Intelligent Document Processing API",
    description="API for document processing, classification, and entity extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demonstration
# In a real implementation, you would use a proper database
db = {
    "documents": {},
    "downloads": {}
}

# CDG Helper Functions
def get_token(DSP_GET_TOKEN_URL, DSP_TRANSLATE_TOKEN_URL, sso_headers, translation_headers):
    """
    Get authentication token for CDG API
    
    Args:
        DSP_GET_TOKEN_URL: URL to get SSO token
        DSP_TRANSLATE_TOKEN_URL: URL to translate SSO token to JWT
        sso_headers: Headers for SSO token request
        translation_headers: Headers for token translation request
        
    Returns:
        str: JWT token or False if error
    """
    try:
        logging.info("Requesting token")
        sso_resp = requests.post(DSP_GET_TOKEN_URL, headers=sso_headers, verify=False)
        if sso_resp.status_code == 200:
            sso_resp = sso_resp.json()
            SSOTOKEN = str(sso_resp["tokenId"])
            translation_body = {
                "input_token_state": {
                    "token_type": "SSOTOKEN",
                    "tokenId": SSOTOKEN
                },
                "output_token_state": {
                    "token_type": "JWT"
                }
            }
            translation_body = json.dumps(translation_body)
            logging.info("Translating token")
            jwt_token_resp = requests.post(DSP_TRANSLATE_TOKEN_URL, headers=translation_headers, data=translation_body, verify=False)
            if jwt_token_resp.status_code == 200:
                jwt_token_resp = jwt_token_resp.json()
                logging.info("JWT token created successfully")
                return jwt_token_resp["issuedToken"]
            else:
                logging.info("Error translating token")
        return False
    except Exception as e:
        logging.error(f"Error getting token: {e}")
        return False

def get_document(doc_id, download_api, file_name):
    """
    Download document from CDG API
    
    Args:
        doc_id: Document ID
        download_api: CDG API download endpoint
        file_name: Name to save the document as
        
    Returns:
        int: HTTP status code
    """
    e2e_token = get_token(DSP_GET_TOKEN_URL, DSP_TRANSLATE_TOKEN_URL, sso_headers, translation_headers)
    api_url = download_api + str(doc_id)
    header = {
        "X-HSBC-E2E-Trust-Token": str(e2e_token),
        "X-HSBC-CUSTID": "CNBAPCDN220004002",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        "Connection": "close"
    }
    try:
        logging.info(f"Downloading document {doc_id}")
        response = requests.get(api_url, headers=header, verify=False)
        file_location = str(file_name)
        with open(file_location, mode="wb") as file:
            file.write(response.content)
        logging.info("Document downloaded successfully")
        return response.status_code
    except Exception as e:
        logging.error(f"Error downloading document: {e}")
        return 500

# API Endpoints

@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "server": platform.node(),
        "service_name": "DCREST Intelligent Document Processing API",
        "timestamp": str(datetime.now())
    }

@app.post("/dcrest/document/process", status_code=201)
async def process_document_request(request: Request, background_tasks: BackgroundTasks):
    """
    Process document request from external system
    
    This endpoint implements steps 1.3-1.4 from the diagram:
    - Receive document request
    - Acknowledge receipt
    - Download document in the background
    """
    try:
        # Parse request body
        body = await request.json()
        
        # Validate required fields
        case_id = body.get('caseID')
        document_id = body.get('documentID')
        
        if not case_id or not document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="caseID and documentID are required",
            )
            
        # Store additional request data if needed
        document_name = body.get('documentName', f"{document_id}.pdf")
        
        # Prepare document data
        doc_data = {
            'caseID': case_id,
            'documentID': document_id,
            'documentName': document_name,
            'received_at': str(datetime.now())
        }
        
        # Store in memory DB (would be a real DB in production)
        db["documents"][document_id] = doc_data
        db["downloads"][document_id] = "pending"
            
        # Download the document in background
        background_tasks.add_task(download_document, doc_data)
        
        # Return acknowledgment per diagram
        return {
            "caseID": case_id,
            "documentID": document_id,
            "status": "Request Acknowledged"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

async def download_document(doc_data: Dict):
    """
    Download document from CDG
    
    Args:
        doc_data: Document data
    """
    doc_id = doc_data.get('documentID')
    doc_name = doc_data.get('documentName')
    
    try:
        # Update status to downloading
        db["downloads"][doc_id] = "downloading"
        
        # Download document from CDG
        status_code = get_document(doc_id, download_api_uat, doc_name)
        
        if status_code == 200:
            logger.info(f"Document {doc_id} downloaded successfully to {doc_name}")
            db["downloads"][doc_id] = "downloaded"
        else:
            logger.error(f"Failed to download document {doc_id}, status code: {status_code}")
            db["downloads"][doc_id] = "download_failed"
            
    except Exception as e:
        logger.error(f"Error downloading document {doc_id}: {str(e)}")
        db["downloads"][doc_id] = "error"

# Disable SSL warnings for development
# In production, proper SSL certificates should be used
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import datetime now that we need it
from datetime import datetime

# Main entry point
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "dcrest_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
