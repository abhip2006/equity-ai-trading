"""
Security utilities for handling sensitive data.

This module provides functions to safely mask API keys and sanitize
sensitive data before logging.
"""

from typing import Any, Dict, Union


def mask_api_key(key: str) -> str:
    """
    Mask API key for safe logging.

    Shows first 8 and last 4 characters, masks the rest.

    Args:
        key: API key or secret to mask

    Returns:
        Masked string safe for logging

    Examples:
        >>> mask_api_key("sk-1234567890abcdefghijklmnopqrstuvwxyz")
        'sk-12345...wxyz'
        >>> mask_api_key("short")
        '***'
        >>> mask_api_key("")
        'None'
    """
    if not key:
        return "None"

    if len(key) <= 12:
        return "***"

    return f"{key[:8]}...{key[-4:]}"


def sanitize_for_log(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or mask sensitive data from dictionary for safe logging.

    Recursively processes nested dictionaries and masks values for keys
    that commonly contain sensitive information.

    Args:
        data: Dictionary potentially containing sensitive data

    Returns:
        New dictionary with sensitive values masked

    Examples:
        >>> sanitize_for_log({"api_key": "secret123", "name": "test"})
        {'api_key': '***', 'name': 'test'}
        >>> sanitize_for_log({"config": {"password": "pass", "count": 5}})
        {'config': {'password': '***', 'count': 5}}
    """
    # Keys that commonly contain sensitive information
    sensitive_keys = {
        'api_key', 'apikey', 'api-key',
        'secret', 'secret_key', 'secretkey', 'secret-key',
        'password', 'passwd', 'pwd',
        'token', 'access_token', 'refresh_token',
        'private_key', 'privatekey', 'private-key',
        'auth', 'authorization',
        'credentials', 'credential'
    }

    if not isinstance(data, dict):
        return data

    sanitized = {}
    for key, value in data.items():
        # Check if key name suggests sensitive data (case-insensitive)
        if key.lower() in sensitive_keys:
            # Mask the value
            if isinstance(value, str):
                sanitized[key] = mask_api_key(value)
            else:
                sanitized[key] = "***"
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_for_log(value)
        elif isinstance(value, list):
            # Sanitize list items if they are dicts
            sanitized[key] = [
                sanitize_for_log(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Safe to log as-is
            sanitized[key] = value

    return sanitized


def mask_url_credentials(url: str) -> str:
    """
    Mask credentials in URLs for safe logging.

    Args:
        url: URL that may contain credentials

    Returns:
        URL with credentials masked

    Examples:
        >>> mask_url_credentials("https://user:pass@api.example.com/data")
        'https://***:***@api.example.com/data'
        >>> mask_url_credentials("https://api.example.com/data")
        'https://api.example.com/data'
    """
    if '@' in url and '://' in url:
        # URL contains credentials (user:pass@host)
        parts = url.split('://', 1)
        protocol = parts[0]
        rest = parts[1]

        if '@' in rest:
            # Split at @ to separate credentials from host
            credentials, host_path = rest.split('@', 1)
            return f"{protocol}://***:***@{host_path}"

    return url
