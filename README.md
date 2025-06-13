import requests
import os

# Test configuration
BASE_URL = "http://localhost:6303"  # Update with your server URL
ENDPOINT = "/dcrest/v1/idp/warden/check-classification"

# Directory containing test files
TEST_FILES_DIR = "./test_files"  # Update with your directory path

# Test headers
TEST_HEADERS = {
    "DCREST-JWT-TOKEN": "valid.test.token.here",  # Replace with valid JWT
    "X-HSBC-Request-Correlation-Id": "test-correlation-123",
    "azure-token": "Bearer test.azure.token"  # Replace with valid Azure token
}

# File mappings - update these with your actual file names
TEST_FILES = {
    "PUBLIC": "public_document.pdf",           # File that should be classified as PUBLIC
    "INTERNAL": "internal_document.pdf",       # File that should be classified as INTERNAL  
    "RESTRICTED": "restricted_document.pdf",   # File that should be classified as RESTRICTED
    "HIGHLY_RESTRICTED": "highly_restricted_document.pdf",  # File classified as HIGHLY RESTRICTED
    "UNCLASSIFIED": "unclassified_document.pdf"  # File that should be classified as UNCLASSIFIED
}

def test_file_classification(classification_type, filename, should_approve):
    """Test file classification for approval/rejection"""
    print(f"Testing {classification_type} document: {filename}")
    
    file_path = os.path.join(TEST_FILES_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (filename, f, 'application/pdf')
            }
            
            response = requests.post(
                BASE_URL + ENDPOINT,
                headers=TEST_HEADERS,
                files=files,
                verify=False
            )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if should_approve:
            # Should return True (200 status) for approved documents
            if response.status_code == 200 and response.json() == True:
                print(f"✓ {classification_type} document correctly APPROVED")
                return True
            else:
                print(f"✗ {classification_type} document incorrectly REJECTED")
                return False
        else:
            # Should return 500 error for rejected documents
            if response.status_code == 500:
                error_detail = response.json().get("detail", "")
                print(f"✓ {classification_type} document correctly REJECTED: {error_detail}")
                return True
            else:
                print(f"✗ {classification_type} document incorrectly APPROVED")
                return False
                
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_public_document_approval():
    """Test that PUBLIC classified document is APPROVED"""
    return test_file_classification("PUBLIC", TEST_FILES["PUBLIC"], should_approve=True)

def test_internal_document_approval():
    """Test that INTERNAL classified document is APPROVED"""
    return test_file_classification("INTERNAL", TEST_FILES["INTERNAL"], should_approve=True)

def test_restricted_document_rejection():
    """Test that RESTRICTED classified document is REJECTED"""
    return test_file_classification("RESTRICTED", TEST_FILES["RESTRICTED"], should_approve=False)

def test_highly_restricted_document_rejection():
    """Test that HIGHLY RESTRICTED classified document is REJECTED"""
    return test_file_classification("HIGHLY_RESTRICTED", TEST_FILES["HIGHLY_RESTRICTED"], should_approve=False)

def test_unclassified_document_rejection():
    """Test that UNCLASSIFIED document is REJECTED"""
    return test_file_classification("UNCLASSIFIED", TEST_FILES["UNCLASSIFIED"], should_approve=False)

def verify_test_files():
    """Verify all required test files exist"""
    print("Verifying test files exist...")
    missing_files = []
    
    for classification, filename in TEST_FILES.items():
        file_path = os.path.join(TEST_FILES_DIR, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✓ {classification}: {filename} ({file_size} bytes)")
        else:
            print(f"✗ {classification}: {filename} - NOT FOUND")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Please ensure all test files are present before running tests.")
        return False
    
    print("All test files found!")
    return True

def run_classification_tests():
    """Run all document classification approval/rejection tests"""
    print("=== Document Classification Approval/Rejection Tests ===")
    print(f"Test files directory: {TEST_FILES_DIR}")
    print(f"Endpoint: {BASE_URL + ENDPOINT}")
    print()
    
    # First verify all files exist
    if not verify_test_files():
        return
    
    print("\n" + "="*60)
    print("Starting classification tests...")
    print()
    
    tests = [
        ("APPROVE", test_public_document_approval),
        ("APPROVE", test_internal_document_approval),
        ("REJECT", test_restricted_document_rejection),
        ("REJECT", test_highly_restricted_document_rejection),
        ("REJECT", test_unclassified_document_rejection)
    ]
    
    passed = 0
    total = len(tests)
    
    for expected_result, test_func in tests:
        print(f"Expected: {expected_result}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
        print("-" * 50)
    
    print(f"=== Classification Test Results ===")
    print(f"Correctly Handled: {passed}/{total}")
    print(f"Incorrectly Handled: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All classifications working correctly!")
        print("✓ Endpoint properly approves PUBLIC/INTERNAL documents")
        print("✓ Endpoint properly rejects RESTRICTED/HIGHLY RESTRICTED/UNCLASSIFIED documents")
    else:
        print("⚠️  Classification logic has issues!")
        print("Check if:")
        print("- Content moderation service is returning correct classifications")
        print("- Endpoint logic matches expected approval/rejection rules")
        print("- Test files have correct classification labels/content")

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
