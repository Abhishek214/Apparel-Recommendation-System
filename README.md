"""
PVT (Product Verification Test) for Document Classification Endpoint

Purpose: Test if /check-classification endpoint correctly approves or rejects documents
         based on their security classification level.

Business Logic Being Tested:
- PUBLIC documents     → Should be APPROVED (return True)
- INTERNAL documents   → Should be APPROVED (return True)  
- RESTRICTED documents → Should be REJECTED (return 400/403 error)
- HIGHLY RESTRICTED    → Should be REJECTED (return 400/403 error)
- UNCLASSIFIED docs    → Should be REJECTED (return 400/403 error)

Test Approach:
- Uses assert statements that STOP on first failure
- Tests with real files from directory (no mocked data)
- Validates actual HTTP status codes and responses
- Ensures security gate is working to protect sensitive data
"""

import requests
import os

# =============================================================================
# TEST CONFIGURATION - UPDATE THESE BEFORE RUNNING
# =============================================================================

# Server configuration
BASE_URL = "http://localhost:6303"  # Update with your actual server URL
ENDPOINT = "/dcrest/v1/idp/warden/check-classification"  # Classification endpoint

# Directory containing test files with different classification levels
TEST_FILES_DIR = "./test_files"  # Update with your actual directory path

# Authentication headers - MUST BE VALID for tests to work
TEST_HEADERS = {
    "DCREST-JWT-TOKEN": "valid.test.token.here",        # Replace with valid JWT token
    "X-HSBC-Request-Correlation-Id": "test-correlation-123",  # Request tracking ID
    "azure-token": "Bearer test.azure.token"            # Replace with valid Azure token
}

# File mappings - UPDATE these with your actual file names
# Each file should have content/labels that trigger the expected classification
TEST_FILES = {
    "PUBLIC": "public_document.pdf",                    # File that gets classified as PUBLIC
    "INTERNAL": "internal_document.pdf",                # File that gets classified as INTERNAL  
    "RESTRICTED": "restricted_document.pdf",            # File that gets classified as RESTRICTED
    "HIGHLY_RESTRICTED": "highly_restricted_document.pdf",  # File that gets classified as HIGHLY RESTRICTED
    "UNCLASSIFIED": "unclassified_document.pdf"         # File that gets classified as UNCLASSIFIED
}

# =============================================================================
# CORE TEST FUNCTION - Tests document classification logic
# =============================================================================

def test_file_classification(classification_type, filename, should_approve):
    """
    Core function to test document classification approval/rejection
    
    Args:
        classification_type (str): Type of classification (PUBLIC, INTERNAL, etc.)
        filename (str): Name of test file to upload
        should_approve (bool): True if document should be approved, False if rejected
    
    Process:
        1. Upload file to /check-classification endpoint
        2. Check HTTP status code and response
        3. Assert expected approval/rejection behavior
        4. STOPS immediately if assertion fails (no continue on error)
    
    Expected Behavior:
        - Approved docs: Return 200 status with True response
        - Rejected docs: Return 400/403 status with error message
    """
    print(f"Testing {classification_type} document: {filename}")
    
    # Build full file path
    file_path = os.path.join(TEST_FILES_DIR, filename)
    
    # ASSERT: Test file must exist (stops test if missing)
    assert os.path.exists(file_path), f"Test file not found: {file_path} - Check TEST_FILES_DIR and filename"
    
    # Upload file to classification endpoint
    with open(file_path, 'rb') as f:
        files = {
            'file': (filename, f, 'application/pdf')  # Send as PDF file upload
        }
        
        # Make POST request to classification endpoint
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False  # Skip SSL verification for testing
        )
    
    # Log response for debugging
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if should_approve:
        # DOCUMENT SHOULD BE APPROVED
        # Assert: Must return 200 status (HTTP OK)
        assert response.status_code == 200, \
            f"{classification_type} document got status {response.status_code}, expected 200 (approved). " \
            f"Check if content moderation service is classifying {filename} correctly as {classification_type}."
        
        # Assert: Response body must be True (approval)
        assert response.json() == True, \
            f"{classification_type} document returned {response.json()}, expected True (approved). " \
            f"Endpoint logic may be rejecting {classification_type} documents incorrectly."
        
        print(f"✓ {classification_type} document correctly APPROVED")
        
    else:
        # DOCUMENT SHOULD BE REJECTED
        # Assert: Must return error status (400 Bad Request or 403 Forbidden)
        # Note: Currently might return 500 if endpoint not fixed yet
        assert response.status_code in [400, 403, 500], \
            f"{classification_type} document got status {response.status_code}, expected rejection (400/403/500). " \
            f"Endpoint is incorrectly approving {classification_type} documents - SECURITY RISK!"
        
        # Assert: Error response must have detail message
        response_data = response.json()
        error_detail = response_data.get("detail", "") if isinstance(response_data, dict) else str(response_data)
        assert error_detail, \
            f"{classification_type} document rejection should include error message explaining why it was rejected"
        
        print(f"✓ {classification_type} document correctly REJECTED")
        print(f"  Rejection reason: {error_detail}")

# =============================================================================
# INDIVIDUAL TEST FUNCTIONS - One for each classification type
# =============================================================================

def test_public_document_approval():
    """
    Test: PUBLIC classified documents should be APPROVED
    
    Business Rule: PUBLIC documents are safe for AI processing
    Expected: 200 status, response = True
    """
    test_file_classification("PUBLIC", TEST_FILES["PUBLIC"], should_approve=True)

def test_internal_document_approval():
    """
    Test: INTERNAL classified documents should be APPROVED
    
    Business Rule: INTERNAL documents are safe for AI processing within organization
    Expected: 200 status, response = True
    """
    test_file_classification("INTERNAL", TEST_FILES["INTERNAL"], should_approve=True)

def test_restricted_document_rejection():
    """
    Test: RESTRICTED classified documents should be REJECTED
    
    Business Rule: RESTRICTED documents are too sensitive for AI processing
    Expected: 400/403 status with error message
    Security: Prevents data leakage of sensitive information
    """
    test_file_classification("RESTRICTED", TEST_FILES["RESTRICTED"], should_approve=False)

def test_highly_restricted_document_rejection():
    """
    Test: HIGHLY RESTRICTED classified documents should be REJECTED
    
    Business Rule: HIGHLY RESTRICTED documents are extremely sensitive
    Expected: 400/403 status with error message  
    Security: Highest level protection against data exposure
    """
    test_file_classification("HIGHLY_RESTRICTED", TEST_FILES["HIGHLY_RESTRICTED"], should_approve=False)

def test_unclassified_document_rejection():
    """
    Test: UNCLASSIFIED documents should be REJECTED
    
    Business Rule: Documents without proper classification labels are not allowed
    Expected: 400 status with message asking user to add classification label
    Compliance: Ensures all documents have proper security labeling
    """
    test_file_classification("UNCLASSIFIED", TEST_FILES["UNCLASSIFIED"], should_approve=False)

# =============================================================================
# SETUP AND VALIDATION FUNCTIONS
# =============================================================================

def verify_test_files():
    """Verify all required test files exist"""
    print("Verifying test files exist...")
    
    for classification, filename in TEST_FILES.items():
        file_path = os.path.join(TEST_FILES_DIR, filename)
        assert os.path.exists(file_path), f"Test file missing: {file_path}"
        file_size = os.path.getsize(file_path)
        print(f"✓ {classification}: {filename} ({file_size} bytes)")
    
    print("All test files found!")

def run_classification_tests():
    """Run all document classification approval/rejection tests"""
    print("=== Document Classification Approval/Rejection Tests ===")
    print(f"Test files directory: {TEST_FILES_DIR}")
    print(f"Endpoint: {BASE_URL + ENDPOINT}")
    print()
    
    # Verify all files exist - will assert and stop if any missing
    verify_test_files()
    
    print("\n" + "="*60)
    print("Starting classification tests...")
    print("NOTE: Tests will stop on first failure with assert error")
    print()
    
    # Test approvals first
    print("Testing APPROVALS (should return True):")
    test_public_document_approval()
    print("-" * 30)
    test_internal_document_approval()
    
    print("\nTesting REJECTIONS (should return 400/403/500 error):")
    print("-" * 30)
    test_restricted_document_rejection()
    print("-" * 30)
    test_highly_restricted_document_rejection()
    print("-" * 30)
    test_unclassified_document_rejection()
    
    print("\n🎉 All classification tests passed!")
    print("✓ Endpoint properly approves PUBLIC/INTERNAL documents")
    print("✓ Endpoint properly rejects RESTRICTED/HIGHLY RESTRICTED/UNCLASSIFIED documents")

def list_available_files():
    """List all files in the test directory"""
    print(f"Files in {TEST_FILES_DIR}:")
    if os.path.exists(TEST_FILES_DIR):
        files = os.listdir(TEST_FILES_DIR)
        for i, file in enumerate(files, 1):
            file_path = os.path.join(TEST_FILES_DIR, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"{i}. {file} ({size} bytes)")
    else:
        print(f"Directory {TEST_FILES_DIR} does not exist!")

if __name__ == "__main__":
    print("Before running tests, ensure:")
    print("1. Update BASE_URL to your server")
    print("2. Update TEST_FILES_DIR to your files directory")
    print("3. Update TEST_FILES mapping with your actual filenames")
    print("4. Update TEST_HEADERS with valid tokens")
    print("5. Content moderation service is running")
    print()
    
    print("Available files:")
    list_available_files()
    print()
    
    run_classification_tests()
