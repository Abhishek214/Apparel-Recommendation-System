import requests
from io import BytesIO

# Test configuration
BASE_URL = "http://localhost:6303"  # Update with your server URL
ENDPOINT = "/dcrest/v1/idp/warden/check-classification"

# Test headers
TEST_HEADERS = {
    "DCREST-JWT-TOKEN": "valid.test.token.here",  # Replace with valid JWT
    "X-HSBC-Request-Correlation-Id": "test-correlation-123",
    "azure-token": "Bearer test.azure.token"  # Replace with valid Azure token
}

def create_test_file(filename="test_document.pdf"):
    """Create a simple test file"""
    file_content = b"Test document content for classification testing"
    return BytesIO(file_content)

def test_public_document_approval():
    """Test that PUBLIC classified document is APPROVED"""
    print("Testing PUBLIC document approval...")
    
    # Create test file that should be classified as PUBLIC
    test_file = create_test_file("PUBLIC_document.pdf")
    
    files = {
        'file': ('PUBLIC_document.pdf', test_file, 'application/pdf')
    }
    
    try:
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return True (200 status) for PUBLIC documents
        if response.status_code == 200 and response.json() == True:
            print("✓ PUBLIC document correctly APPROVED")
            return True
        else:
            print("✗ PUBLIC document incorrectly REJECTED")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_internal_document_approval():
    """Test that INTERNAL classified document is APPROVED"""
    print("\nTesting INTERNAL document approval...")
    
    # Create test file that should be classified as INTERNAL
    test_file = create_test_file("INTERNAL_document.pdf")
    
    files = {
        'file': ('INTERNAL_document.pdf', test_file, 'application/pdf')
    }
    
    try:
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return True (200 status) for INTERNAL documents
        if response.status_code == 200 and response.json() == True:
            print("✓ INTERNAL document correctly APPROVED")
            return True
        else:
            print("✗ INTERNAL document incorrectly REJECTED")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_restricted_document_rejection():
    """Test that RESTRICTED classified document is REJECTED"""
    print("\nTesting RESTRICTED document rejection...")
    
    # Create test file that should be classified as RESTRICTED
    test_file = create_test_file("RESTRICTED_document.pdf")
    
    files = {
        'file': ('RESTRICTED_document.pdf', test_file, 'application/pdf')
    }
    
    try:
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return 500 error for RESTRICTED documents
        if response.status_code == 500:
            error_detail = response.json().get("detail", "")
            if "RESTRICTED" in error_detail and "classification level INTERNAL" in error_detail:
                print("✓ RESTRICTED document correctly REJECTED")
                return True
        
        print("✗ RESTRICTED document incorrectly APPROVED")
        return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_highly_restricted_document_rejection():
    """Test that HIGHLY RESTRICTED classified document is REJECTED"""
    print("\nTesting HIGHLY RESTRICTED document rejection...")
    
    # Create test file that should be classified as HIGHLY RESTRICTED
    test_file = create_test_file("HIGHLY_RESTRICTED_document.pdf")
    
    files = {
        'file': ('HIGHLY_RESTRICTED_document.pdf', test_file, 'application/pdf')
    }
    
    try:
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return 500 error for HIGHLY RESTRICTED documents
        if response.status_code == 500:
            error_detail = response.json().get("detail", "")
            if "HIGHLY RESTRICTED" in error_detail and "classification level INTERNAL" in error_detail:
                print("✓ HIGHLY RESTRICTED document correctly REJECTED")
                return True
        
        print("✗ HIGHLY RESTRICTED document incorrectly APPROVED")
        return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_unclassified_document_rejection():
    """Test that UNCLASSIFIED document is REJECTED"""
    print("\nTesting UNCLASSIFIED document rejection...")
    
    # Create test file that should be classified as UNCLASSIFIED
    test_file = create_test_file("unclassified_document.pdf")
    
    files = {
        'file': ('unclassified_document.pdf', test_file, 'application/pdf')
    }
    
    try:
        response = requests.post(
            BASE_URL + ENDPOINT,
            headers=TEST_HEADERS,
            files=files,
            verify=False
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Should return 500 error for UNCLASSIFIED documents
        if response.status_code == 500:
            error_detail = response.json().get("detail", "")
            if "add a classification label" in error_detail and "PUBLIC, INTERNAL, RESTRICTED, HIGHLY RESTRICTED" in error_detail:
                print("✓ UNCLASSIFIED document correctly REJECTED")
                return True
        
        print("✗ UNCLASSIFIED document incorrectly APPROVED")
        return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def run_classification_tests():
    """Run all document classification approval/rejection tests"""
    print("=== Document Classification Approval/Rejection Tests ===")
    print("Testing if endpoint correctly approves/rejects based on classification")
    print(f"Endpoint: {BASE_URL + ENDPOINT}")
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

if __name__ == "__main__":
    print("Before running tests, ensure:")
    print("1. Update BASE_URL to your server")
    print("2. Update TEST_HEADERS with valid tokens")
    print("3. Content moderation service is running")
    print("4. Test files trigger correct classifications")
    print()
    
    run_classification_tests()
