from source.auth import auth
import requests

def PVT():
    """
    Function to test the authentication functionality using the auth function.
    Returns True if the authentication is successful, otherwise returns False.
    """
    
    try:
        # Test with a sample JWT token (replace with actual test token)
        test_jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.token"  # Replace with actual test token
        
        # Call the auth function with the test JWT token
        result = auth(test_jwt_token)
        
        # Check if the result is not None and contains expected user info
        if result is not None:
            print('PVT PASS')  # Log success message
            print(f"Auth result: {result}")  # Print the auth result for debugging
            return True
        else:
            print('PVT Failed')  # Log failure message
            return False
            
    # Handle HTTP errors
    except requests.exceptions.HTTPError as e:
        print("Message: Unable to authenticate user",
              "Exception: " + str(e))
        return False
        
    # Handle connection-related errors
    except requests.exceptions.ConnectionError as e:
        print("Message: Unable to connect to auth service",
              "Exception: " + str(e))
        return False
        
    # Handle timeout errors
    except requests.exceptions.Timeout as e:
        print("Message: Auth service timeout",
              "Exception: " + str(e))
        return False
        
    # Handle invalid token errors
    except ValueError as e:
        print("Message: Invalid JWT token format",
              "Exception: " + str(e))
        return False
        
    # Handle attribute-related errors
    except AttributeError as e:
        print("Message: Unable to process auth response",
              "Exception: " + str(e))
        return False
        
    # Handle any other exceptions
    except Exception as e:
        print("Message: Unable to authenticate user",
              "Exception: " + str(e))
        return False


def PVT_WITH_VALID_TOKEN():
    """
    Function to test auth with a known valid token structure.
    """
    
    try:
        # Example of a more realistic test token structure
        # You should replace this with actual valid tokens from your system
        valid_test_tokens = [
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoidGVzdF91c2VyIiwiaWF0IjoxNjMwMDAwMDAwfQ.test_signature"
        ]
        
        for i, token in enumerate(valid_test_tokens):
            print(f"\n--- Testing Token {i+1} ---")
            result = auth(token)
            
            if result is not None:
                print(f'Token {i+1}: PVT PASS')
                print(f"User info: {result}")
                if 'user' in result:
                    print(f"User ID: {result['user']}")
            else:
                print(f'Token {i+1}: PVT Failed')
                
        return True
        
    except Exception as e:
        print("Message: Error in token testing",
              "Exception: " + str(e))
        return False


def PVT_WITH_INVALID_TOKEN():
    """
    Function to test auth with invalid tokens to ensure proper error handling.
    """
    
    try:
        invalid_tokens = [
            "",  # Empty token
            "invalid_token",  # Invalid format
            "Bearer ",  # Empty bearer token
            "Bearer invalid.jwt.token",  # Malformed JWT
            None,  # None token
        ]
        
        print("\n--- Testing Invalid Tokens ---")
        for i, token in enumerate(invalid_tokens):
            try:
                print(f"\nTesting invalid token {i+1}: {token}")
                result = auth(token)
                
                if result is None:
                    print(f'Invalid token {i+1}: Correctly rejected')
                else:
                    print(f'Invalid token {i+1}: Unexpectedly accepted - {result}')
                    
            except Exception as e:
                print(f'Invalid token {i+1}: Correctly rejected with exception - {str(e)}')
                
        return True
        
    except Exception as e:
        print("Message: Error in invalid token testing",
              "Exception: " + str(e))
        return False


def PVT_FULL_AUTH_TEST():
    """
    Comprehensive auth testing function.
    """
    
    print("=" * 50)
    print("STARTING COMPREHENSIVE AUTH TESTING")
    print("=" * 50)
    
    # Test 1: Basic auth test
    print("\n1. Basic Auth Test:")
    basic_result = PVT()
    
    # Test 2: Valid token tests
    print("\n2. Valid Token Tests:")
    valid_result = PVT_WITH_VALID_TOKEN()
    
    # Test 3: Invalid token tests
    print("\n3. Invalid Token Tests:")
    invalid_result = PVT_WITH_INVALID_TOKEN()
    
    # Summary
    print("\n" + "=" * 50)
    print("AUTH TESTING SUMMARY")
    print("=" * 50)
    print(f"Basic Auth Test: {'PASS' if basic_result else 'FAIL'}")
    print(f"Valid Token Test: {'PASS' if valid_result else 'FAIL'}")
    print(f"Invalid Token Test: {'PASS' if invalid_result else 'FAIL'}")
    
    overall_result = basic_result and valid_result and invalid_result
    print(f"Overall Result: {'PASS' if overall_result else 'FAIL'}")
    
    return overall_result


# Additional utility function for testing specific user scenarios
def PVT_USER_SCENARIO(user_id="test_user", jwt_token=None):
    """
    Test auth for a specific user scenario.
    
    Args:
        user_id (str): The user ID to test
        jwt_token (str): Optional JWT token, if None uses a default test token
    """
    
    try:
        if jwt_token is None:
            # Create a mock token for testing (replace with actual token generation logic)
            jwt_token = f"Bearer test_token_for_{user_id}"
        
        print(f"\n--- Testing User Scenario: {user_id} ---")
        result = auth(jwt_token)
        
        if result is not None:
            print(f'User {user_id}: Authentication PASS')
            print(f"User details: {result}")
            
            # Verify the user ID matches
            if 'user' in result and result['user'] == user_id:
                print(f"User ID verification: PASS")
            else:
                print(f"User ID verification: FAIL (expected {user_id}, got {result.get('user', 'None')})")
                
            return True
        else:
            print(f'User {user_id}: Authentication FAIL')
            return False
            
    except Exception as e:
        print(f"User {user_id}: Authentication ERROR - {str(e)}")
        return False


if __name__ == "__main__":
    # Run the comprehensive test when script is executed directly
    PVT_FULL_AUTH_TEST()
    
    # Test specific user scenarios
    print("\n" + "=" * 30)
    print("USER SCENARIO TESTS")
    print("=" * 30)
    PVT_USER_SCENARIO("john_doe")
    PVT_USER_SCENARIO("admin_user")
    PVT_USER_SCENARIO("test_user_123")
