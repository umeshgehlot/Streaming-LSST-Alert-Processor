from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger("astronomy_api")


class BaseAPIException(Exception):
    """Base exception for API errors"""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)


class DatabaseException(BaseAPIException):
    """Database related errors"""
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(status_code=500, detail=detail, error_code="DATABASE_ERROR")


class ModelTrainingException(BaseAPIException):
    """Model training related errors"""
    def __init__(self, detail: str = "Model training failed"):
        super().__init__(status_code=500, detail=detail, error_code="TRAINING_ERROR")


class DataProcessingException(BaseAPIException):
    """Data processing related errors"""
    def __init__(self, detail: str = "Data processing failed"):
        super().__init__(status_code=400, detail=detail, error_code="DATA_PROCESSING_ERROR")


async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    logger.error(f"API Exception: {exc.error_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code or "UNKNOWN_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": exc.errors()
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "status_code": 500
            }
        }
    )


def setup_exception_handlers(app: FastAPI):
    """Setup all exception handlers for the app"""
    app.add_exception_handler(BaseAPIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)