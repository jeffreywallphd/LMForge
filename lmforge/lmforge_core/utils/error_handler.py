"""
Centralized error handling and logging utilities for LMForge
"""
import logging
from django.http import JsonResponse
from functools import wraps
import traceback

# Configure logger
logger = logging.getLogger(__name__)

class AppError(Exception):
    """Base exception for application errors"""
    def __init__(self, message, status_code=500, user_message=None):
        self.message = message
        self.status_code = status_code
        self.user_message = user_message or message
        super().__init__(self.message)


def handle_errors(view_func):
    """
    Decorator to handle errors consistently across views
    Usage: @handle_errors
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        try:
            return view_func(request, *args, **kwargs)
        except AppError as e:
            logger.error(f"AppError in {view_func.__name__}: {e.message}")
            return JsonResponse(
                {"error": e.user_message}, 
                status=e.status_code
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in {view_func.__name__}: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            return JsonResponse(
                {"error": "An unexpected error occurred. Please try again later."},
                status=500
            )
    return wrapper


def log_info(message, extra_data=None):
    """Log info with optional extra data"""
    if extra_data:
        logger.info(f"{message} | Data: {extra_data}")
    else:
        logger.info(message)


def log_error(message, exception=None, extra_data=None):
    """Log errors with optional exception traceback"""
    if exception:
        logger.error(f"{message} | Exception: {str(exception)} | Traceback: {traceback.format_exc()}")
    else:
        logger.error(message)
    if extra_data:
        logger.error(f"Extra data: {extra_data}")


def log_warning(message, extra_data=None):
    """Log warnings"""
    if extra_data:
        logger.warning(f"{message} | Data: {extra_data}")
    else:
        logger.warning(message)
