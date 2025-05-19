
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF
Please help me identify and transcribe the main informational sections of this document. I need to copy the central content that appears between the opening (after the date and recipient) and the closing signature area. Include any lists, structured data, or detailed information present in the body. This is for creating a plain text version of the document's main content for accessibility and archival purposes.



Please transcribe the main body paragraphs of this letter, starting after the "To:" and "Date:" lines and ending before "Yours faithfully" or similar closing. Include all the informational content in between, such as declarations, lists, and details about the trust. I need this text for my records.


Extract the factual content of this business document for documentation purposes. Include all paragraphs from the first substantive content after the salutation (e.g., "Dear Sirs", "Dear Sir/Madam"). If the document has a subject line starting with "Re:", begin after that. Otherwise, include everything after the salutation, including any trust/entity names or titles that appear before the main paragraphs. Continue through all content until you reach closing phrases like "Yours faithfully", "Sincerely", or the end of the visible page if no closing is present. Include all numbered points, bullet points, and paragraphs. Exclude only letterhead, recipient address, date, reference numbers, and signature blocks. This is for accurate record-keeping of the document's informational contents only.


import os
import uuid
import platform
import uvicorn
import gc
from datetime import datetime
import time
import json
import dotenv
import logging
import traceback
import sys
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, status, HTTPException, Request, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
dotenv.load_dotenv(dotenv_path="./env_variables/dcrest.env")

# Constants - these would typically come from environment variables
CDG_API_ENDPOINT = os.getenv("CDG_API_ENDPOINT", "https://cdg-api.example.com")
DOWNLOAD_API_ENDPOINT = os.getenv("DOWNLOAD_API_ENDPOINT", "https://cdg-api.example.com/download")
BROKER_SERVICE_ENDPOINT = os.getenv("BROKER_SERVICE_ENDPOINT", "https://broker.example.com/publish")
CDG_API_KEY = os.getenv("CDG_API_KEY", "your-api-key-here")

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
    "processing_status": {}
}

# Helper functions for document processing
async def download_document_from_cdg(doc_id: str, doc_name: str) -> tuple:
    """
    Download document from CDG API
    
    This function makes a real HTTP request to the CDG API to download a document
    and saves it to a local file.
    
    Args:
        doc_id: Document ID
        doc_name: Document name
        
    Returns:
        tuple: (status_code, file_path)
    """
    file_path = f"./temp/{doc_id}_{doc_name}"
    
    try:
        logger.info(f"Downloading document {doc_id} from CDG API")
        
        # Ensure the temp directory exists
        os.makedirs("./temp", exist_ok=True)
        
        # Construct the API URL for document download
        # Example: https://cdg-api.example.com/download/documents/123456
        download_url = f"{DOWNLOAD_API_ENDPOINT}/documents/{doc_id}"
        
        # Set up headers for the API request
        headers = {
            "X-API-Key": CDG_API_KEY,
            "Accept": "application/octet-stream",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload if needed
        # Some APIs might require additional parameters
        payload = {
            "documentId": doc_id,
            "fileName": doc_name,
            "requestTimestamp": datetime.now().isoformat()
        }
        
        # Make the HTTP request to download the document
        async with aiohttp.ClientSession() as session:
            # First, make a POST request to initiate the download
            async with session.post(
                download_url, 
                headers=headers, 
                json=payload,
                timeout=60  # Set a reasonable timeout
            ) as response:
                
                # Check if the request was successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"CDG API returned error: {response.status}, {error_text}")
                    return response.status, None
                
                # Get the content disposition to verify filename if needed
                content_disposition = response.headers.get("Content-Disposition", "")
                logger.debug(f"Content-Disposition: {content_disposition}")
                
                # Read the file content
                file_content = await response.read()
                
                # Check if we got any content
                if not file_content:
                    logger.error(f"No content received for document {doc_id}")
                    return 204, None  # No content
                
                # Save the file content to disk
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                # Verify the file was created and has content
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    logger.error(f"Failed to save document {doc_id} to {file_path}")
                    return 500, None
                
                logger.info(f"Document {doc_id} downloaded successfully to {file_path} ({os.path.getsize(file_path)} bytes)")
                return 200, file_path
                
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error downloading document {doc_id}: {str(e)}")
        return 500, None
    except asyncio.TimeoutError:
        logger.error(f"Timeout downloading document {doc_id}")
        return 504, None  # Gateway Timeout
    except Exception as e:
        logger.error(f"Error downloading document {doc_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return 500, None

def store_document_metadata(doc_data: Dict) -> str:
    """
    Store document metadata in database
    
    Args:
        doc_data: Document metadata
        
    Returns:
        str: Operation result
    """
    try:
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Prepare record for database
        db_record = {
            'user_id': doc_data.get('user_id', 'system'),
            'case_id': doc_data.get('caseID'),
            'cdg_unit_id': doc_data.get('documentID'),
            'session_id': session_id,
            'created_timestamp': current_time,
            'doc_name': doc_data.get('documentName', ''),
            'status': 'received',
            'last_updated': current_time
        }
        
        # In a real implementation, you would insert this into your database
        # For demonstration, storing in memory
        doc_id = doc_data.get('documentID', str(uuid.uuid4()))
        db["documents"][doc_id] = db_record
        db["processing_status"][doc_id] = "pending"
        
        return "success"
    except Exception as e:
        logger.error(f"Error storing document metadata: {str(e)}")
        return "error"

async def process_document(doc_data: Dict):
    """
    Process document through the DCREST pipeline
    
    This function implements the document processing flow up to OCR
    
    Args:
        doc_data: Document data
    """
    doc_id = doc_data.get('documentID')
    doc_name = doc_data.get('documentName', f"{doc_id}.pdf")
    case_id = doc_data.get('caseID')
    
    try:
        # Update status to processing
        db["processing_status"][doc_id] = "processing"
        logger.info(f"Starting document processing flow for document {doc_id}")
        
        # Step 1: Retrieve document from CDG
        status_code, file_path = await download_document_from_cdg(doc_id, doc_name)
        
        if status_code != 200 or not file_path:
            logger.error(f"Failed to download document {doc_id}")
            db["processing_status"][doc_id] = "download_failed"
            return
        
        # Document retrieved successfully
        # Update status to downloaded
        db["processing_status"][doc_id] = "downloaded"
        logger.info(f"Document {doc_id} downloaded to {file_path}")
        
        # Record download time
        download_time = datetime.now()
        db["documents"][doc_id]["download_time"] = download_time
        
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist after download")
            db["processing_status"][doc_id] = "file_missing"
            return
            
        # Next step would be OCR, but as requested, we'll stop here
        # In a complete implementation, the next steps would be:
        # - OCR processing
        # - Classification
        # - Named entity extraction
        # - Publishing results to broker
        
        logger.info(f"Document {doc_id} ready for OCR processing")
        db["processing_status"][doc_id] = "ready_for_ocr"
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        logger.error(traceback.format_exc())
        db["processing_status"][doc_id] = "processing_error"
        
        # Cleanup - remove temporary file if it exists
        if 'file_path' in locals() and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Temporary file {file_path} removed")
            except Exception as cleanup_error:
                logger.error(f"Error removing temporary file {file_path}: {str(cleanup_error)}")

# API Endpoints

@app.post("/dcrest/document/process", status_code=201)
async def process_document_request(request: Request, background_tasks: BackgroundTasks):
    """
    Process document request from external system
    
    This endpoint implements steps 1.3-1.4 from the diagram:
    - Receive document request
    - Acknowledge receipt
    - Trigger document processing in the background
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
            'user_id': 'system',  # Default user since we removed authentication
            'received_at': datetime.now().isoformat()
        }
        
        # Store document metadata
        result = store_document_metadata(doc_data)
        
        if result != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document information",
            )
            
        # Start document processing in background
        background_tasks.add_task(process_document, doc_data)
        
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

@app.get("/dcrest/document/{document_id}/status", status_code=200)
async def get_document_status(document_id: str):
    """Get document processing status"""
    if document_id not in db["processing_status"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
        
    status = db["processing_status"][document_id]
    return {
        "documentID": document_id,
        "status": status,
        "lastUpdated": db["documents"][document_id].get("last_updated", "").isoformat() 
                       if isinstance(db["documents"][document_id].get("last_updated"), datetime) else None
    }

# Main entry point
if __name__ == "__main__":
    # Create temporary directory if it doesn't exist
    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "dcrest_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
