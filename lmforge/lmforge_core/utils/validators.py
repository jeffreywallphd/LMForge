"""
Input validation utilities for LMForge
"""
import re
from django.core.exceptions import ValidationError


def validate_file_size(file, max_size_mb=100):
    """
    Validate file size
    Args:
        file: Django file object
        max_size_mb: Maximum allowed size in MB
    """
    max_size = max_size_mb * 1024 * 1024
    if file.size > max_size:
        raise ValidationError(f"File size exceeds {max_size_mb}MB limit")


def validate_file_extension(filename, allowed_extensions):
    """
    Validate file extension
    Args:
        filename: Name of file
        allowed_extensions: List of allowed extensions (e.g., ['pdf', 'txt'])
    """
    ext = filename.split('.')[-1].lower()
    if ext not in allowed_extensions:
        raise ValidationError(f"File type .{ext} not allowed. Allowed: {', '.join(allowed_extensions)}")


def validate_session_id(session_id):
    """Validate session ID format"""
    if not session_id or len(session_id) > 100:
        raise ValidationError("Invalid session ID")


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")


def validate_url(url):
    """Validate URL format"""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(pattern, url, re.IGNORECASE):
        raise ValidationError("Invalid URL format")


def sanitize_filename(filename):
    """Remove potentially dangerous characters from filename"""
    # Keep only alphanumeric, dash, underscore, and dot
    sanitized = re.sub(r'[^\w\-\.]', '_', filename)
    return sanitized


def validate_json_string(json_str):
    """Validate JSON string format"""
    import json
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")
