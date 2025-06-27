#!/usr/bin/env python3
"""
Simple Auth Module - Basic JWT token validation
"""

import jwt
from fastapi import HTTPException, status


def auth(token: str) -> dict:
    """
    Simple auth function - validates JWT token
    For demo purposes, you can either:
    1. Implement actual JWT validation with your secret key
    2. Return a mock user for testing
    """
    
    try:
        # For production, use actual JWT validation:
        # SECRET_KEY = "your-secret-key"
        # ALGORITHM = "HS256"
        # payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # return payload
        
        # For demo/testing purposes, return mock user:
        if token:  # Just check if token exists
            return {
                "user": "demo_user",
                "sub": "demo_subject",
                "exp": 9999999999  # Far future expiry
            }
        else:
            raise ValueError("No token provided")
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
