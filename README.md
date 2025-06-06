import requests
import os
from source.auth import create_access_token

def PVT():
    """
    Function to test the upload file functionality.
    Returns True if the upload is successful, otherwise returns False.
    """
    
    try:
        # Create test JWT token
        test_token = create_access_token(data={"user": "testuser"})
        
        # Test session creation first
        session_url = "http://localhost:8000/extraction/session"
        session_headers = {
            "DCREST_JWT_TOKEN": test_token,
            "X_HSBC_Request_Correlation_Id": "test-correlation-id",
            "azure_token": "test-azure-token"
        }
        
        session_response = requests.post(session_url, headers=session_headers)
        
        if session_response.status_code == 201:
            session_data = session_response.json()
            session_token = session_data.get("session")
            
            # Create a test file
            test_file_path = "test_document.txt"
            with open(test_file_path, "w") as f:
                f.write("This is a test document for upload testing.")
            
            # Test file upload
            upload_url = "http://localhost:8000/extraction/upload-file"
            upload_headers = {
                "DCREST_JWT_TOKEN": test_token,
                "X_HSBC_Request_Correlation_Id": "test-upload-id",
                "azure_token": "test-azure-token"
            }
            
            upload_data = {
                "session_token": session_token,
                "multimodal": True,
                "chunking_strategy": "BY_PAGE"
            }
            
            with open(test_file_path, "rb") as f:
                files = {"files": ("test_document.txt", f, "text/plain")}
                upload_response = requests.post(
                    upload_url, 
                    headers=upload_headers, 
                    data=upload_data, 
                    files=files
                )
            
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            
            if upload_response.status_code == 201:
                print('PVT PASS')  # Log success message
                return True
            else:
                print(f'Upload failed with status: {upload_response.status_code}')
                print('PVT Failed')  # Log failure message
                return False
        else:
            print(f'Session creation failed with status: {session_response.status_code}')
            print('PVT Failed')
            return False
    
    # Handle HTTP errors
    except requests.exceptions.HTTPError as e:
        print("Message: Unable to upload file",
              "Exception: " + str(e))
        return False
    
    # Handle attribute-related errors
    except AttributeError as e:
        print("Message: Unable to upload file",
              "Exception: " + str(e))
        return False
    
    # Handle value-related errors
    except ValueError as e:
        print("Message: Unable to upload file",
              "Exception: " + str(e))
        return False
    
    # Handle any other exceptions
    except Exception as e:
        print("Message: Unable to upload file",
              "Exception: " + str(e))
        return False

if __name__ == "__main__":
    PVT()
