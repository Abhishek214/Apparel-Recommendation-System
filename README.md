import requests
import os

def PVT():
    """
    Function to test the upload file functionality using REAL tokens and APIs.
    Returns True if the upload is successful, otherwise returns False.
    """
    
    try:
        # REAL TOKENS - Replace these with actual tokens from your system
        # You need to get these from your actual authentication system
        real_jwt_token = "PUT_REAL_JWT_TOKEN_HERE"
        real_correlation_id = "correlation-id-12345"  # Or generate UUID
        real_azure_token = "PUT_REAL_AZURE_TOKEN_HERE"
        
        # Check if tokens are placeholder values
        if "PUT_REAL" in real_jwt_token or "PUT_REAL" in real_azure_token:
            print("ERROR: Please replace placeholder tokens with real tokens")
            print("You need to get actual JWT and Azure tokens from your system")
            print('PVT Failed - No real tokens provided')
            return False
        
        base_url = "http://localhost:8000"  # Your running application
        
        print("Starting REAL upload functionality test...")
        print(f"Using Real JWT Token: {real_jwt_token[:30]}...")
        print(f"Using Correlation ID: {real_correlation_id}")
        print(f"Using Real Azure Token: {real_azure_token[:30]}...")
        
        # Step 1: Test session creation with REAL tokens
        print("\n1. Testing session creation with real APIs...")
        session_headers = {
            "DCREST_JWT_TOKEN": real_jwt_token,
            "X_HSBC_Request_Correlation_Id": real_correlation_id,
            "azure_token": real_azure_token
        }
        
        session_response = requests.post(
            f"{base_url}/extraction/session", 
            headers=session_headers,
            timeout=30
        )
        
        print(f"Session response status: {session_response.status_code}")
        print(f"Session response: {session_response.text}")
        
        if session_response.status_code != 201:
            print(f"Session creation failed: {session_response.text}")
            print('PVT Failed - Session creation failed')
            return False
            
        session_data = session_response.json()
        session_token = session_data.get("session")
        print(f"✓ Session token created: {session_token}")
        
        # Step 2: Create a real test file
        print("\n2. Creating test document...")
        test_file_path = "test_upload_document.pdf"  # Use PDF as it's commonly supported
        test_content = b"Sample PDF content for upload testing"  # Binary content for PDF
        
        with open(test_file_path, "wb") as f:
            f.write(test_content)
        print(f"✓ Test file created: {test_file_path}")
        
        # Step 3: Test REAL file upload
        print("\n3. Testing REAL file upload...")
        upload_headers = {
            "DCREST_JWT_TOKEN": real_jwt_token,
            "X_HSBC_Request_Correlation_Id": real_correlation_id,
            "azure_token": real_azure_token
        }
        
        upload_data = {
            "session_token": session_token,
            "multimodal": True,
            "chunking_strategy": "BY_PAGE"
        }
        
        with open(test_file_path, "rb") as f:
            files = {"files": (test_file_path, f, "application/pdf")}
            upload_response = requests.post(
                f"{base_url}/extraction/upload-file",
                headers=upload_headers,
                data=upload_data,
                files=files,
                timeout=60  # Longer timeout for file upload
            )
        
        print(f"Upload response status: {upload_response.status_code}")
        print(f"Upload response: {upload_response.text}")
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print("✓ Test file cleaned up")
        
        # Check upload result
        if upload_response.status_code == 201:
            upload_result = upload_response.json()
            document_id = upload_result.get("documentID")
            if document_id:
                print(f"✓ Upload successful! Document ID: {document_id}")
                print('PVT PASS - Real upload completed successfully')
                return True
            else:
                print("✗ Upload response missing documentID")
                print('PVT Failed - Invalid upload response')
                return False
        else:
            print(f"✗ Upload failed with status {upload_response.status_code}")
            print('PVT Failed - Upload request failed')
            return False
    
    # Handle specific HTTP errors
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("Message: Unable to upload file - HTTP error")
        return False
    
    # Handle connection errors
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        print("Message: Unable to connect to server - ensure Application.py is running")
        return False
    
    # Handle timeout errors
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error: {e}")
        print("Message: Request timed out - server may be slow")
        return False
    
    # Handle other request errors
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        print("Message: Unable to upload file - request failed")
        return False
    
    # Handle any other exceptions
    except Exception as e:
        print(f"Unexpected Error: {e}")
        print("Message: Unable to upload file - unexpected error")
        return False

def get_tokens_info():
    """Helper function to show how to get real tokens"""
    print("="*60)
    print("HOW TO GET REAL TOKENS:")
    print("="*60)
    print("1. JWT TOKEN (DCREST_JWT_TOKEN):")
    print("   - Login to your application's frontend")
    print("   - Check browser developer tools > Network tab")
    print("   - Look for 'DCREST_JWT_TOKEN' in request headers")
    print("")
    print("2. AZURE TOKEN:")
    print("   - Get from Azure AD authentication")
    print("   - Or check existing API calls for 'azure_token' header")
    print("")
    print("3. CORRELATION ID:")
    print("   - Can be any unique string")
    print("   - Format: 'correlation-id-' + timestamp or UUID")
    print("="*60)

if __name__ == "__main__":
    print("="*50)
    print("TESTING REAL UPLOAD FUNCTIONALITY")
    print("="*50)
    
    # Show token info first
    get_tokens_info()
    
    print("\nStarting real API tests...")
    result = PVT()
    
    print("\n" + "="*50)
    if result:
        print("OVERALL TEST RESULT: PASS")
        print("✓ Real authentication and upload working!")
    else:
        print("OVERALL TEST RESULT: FAIL")
        print("✗ Check tokens and ensure server is running")
    print("="*50)
