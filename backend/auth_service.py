"""
Auth0 Integration Service
Handles JWT token verification and user authentication
"""

import os
from typing import Dict
from jose import jwt, JWTError
from fastapi import HTTPException, status
import requests
from functools import lru_cache


@lru_cache()
def get_auth0_public_key():
    """
    Fetch Auth0 public key for JWT verification
    Cached to avoid repeated API calls
    """
    domain = os.getenv("AUTH0_DOMAIN")
    if not domain:
        raise ValueError("AUTH0_DOMAIN not configured")
    
    jwks_url = f"https://{domain}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Auth0 JWKS: {e}")
        return None


def verify_token(token: str) -> Dict:
    """
    Verify Auth0 JWT token and return payload
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Dict containing token payload (user info)
        
    Raises:
        ValueError: If token is invalid
    """
    try:
        # Get configuration
        domain = os.getenv("AUTH0_DOMAIN")
        audience = os.getenv("AUTH0_API_AUDIENCE")
        issuer = os.getenv("AUTH0_ISSUER", f"https://{domain}/")
        algorithms = os.getenv("AUTH0_ALGORITHMS", "RS256").split(",")
        
        if not all([domain, audience]):
            raise ValueError("Auth0 configuration incomplete")
        
        # For development, allow skipping verification if explicitly disabled
        if os.getenv("SKIP_AUTH_VERIFICATION") == "true":
            print("⚠️  WARNING: Auth verification is disabled (development mode)")
            return {"sub": "dev_user", "email": "dev@example.com"}
        
        # Get JWKS (JSON Web Key Set) from Auth0
        jwks = get_auth0_public_key()
        if not jwks:
            raise ValueError("Unable to fetch Auth0 public keys")
        
        # Decode and verify the token
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        
        # Find the correct key from JWKS
        for key in jwks.get("keys", []):
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        
        if not rsa_key:
            raise ValueError("Unable to find appropriate key")
        
        # Verify and decode
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=algorithms,
            audience=audience,
            issuer=issuer,
        )
        
        return payload
        
    except JWTError as e:
        raise ValueError(f"Invalid token: {str(e)}")
    except Exception as e:
        raise ValueError(f"Token verification failed: {str(e)}")


def get_user_info(token: str) -> Dict:
    """
    Get user information from Auth0 using the access token
    
    Args:
        token: Auth0 access token
        
    Returns:
        Dict containing user profile information
    """
    domain = os.getenv("AUTH0_DOMAIN")
    
    try:
        response = requests.get(
            f"https://{domain}/userinfo",
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching user info: {e}")
        return {}


def check_user_permissions(payload: Dict, required_permission: str) -> bool:
    """
    Check if user has required permission
    
    Args:
        payload: JWT payload containing permissions
        required_permission: Permission string to check
        
    Returns:
        bool: True if user has permission
    """
    permissions = payload.get("permissions", [])
    return required_permission in permissions


# Example usage and testing
if __name__ == "__main__":
    print("Auth0 Service Configuration:")
    print(f"Domain: {os.getenv('AUTH0_DOMAIN')}")
    print(f"Audience: {os.getenv('AUTH0_API_AUDIENCE')}")
    print(f"Issuer: {os.getenv('AUTH0_ISSUER')}")
