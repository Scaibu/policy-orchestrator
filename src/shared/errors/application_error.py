from typing import Any, Optional
from ..constants.error_constants import (
    ERROR_CODE_VALIDATION_FAILED,
    ERROR_CODE_UNAUTHORIZED,
    ERROR_CODE_FORBIDDEN,
    ERROR_CODE_RESOURCE_NOT_FOUND,
    ERROR_CODE_CONFLICT,
    ERROR_CODE_PRECONDITION_FAILED,
    ERROR_CODE_TIMEOUT,
    ERROR_CODE_INTERNAL_SERVER_ERROR,
)

class ApplicationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        canonical_code: str = ERROR_CODE_INTERNAL_SERVER_ERROR,
        retryable: bool = False,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.canonical_code = canonical_code
        self.retryable = retryable
        self.details = details

class ValidationError(ApplicationError):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=400,
            canonical_code=ERROR_CODE_VALIDATION_FAILED,
            retryable=False,
            details=details,
        )

class UnauthorizedError(ApplicationError):
    def __init__(self, message: str = "Unauthorized", details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=401,
            canonical_code=ERROR_CODE_UNAUTHORIZED,
            retryable=False,
            details=details,
        )

class ForbiddenError(ApplicationError):
    def __init__(self, message: str = "Forbidden", details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=403,
            canonical_code=ERROR_CODE_FORBIDDEN,
            retryable=False,
            details=details,
        )

class NotFoundError(ApplicationError):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=404,
            canonical_code=ERROR_CODE_RESOURCE_NOT_FOUND,
            retryable=False,
            details=details,
        )

class ConflictError(ApplicationError):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=409,
            canonical_code=ERROR_CODE_CONFLICT,
            retryable=False,
            details=details,
        )

class PreconditionFailedError(ApplicationError):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=412,
            canonical_code=ERROR_CODE_PRECONDITION_FAILED,
            retryable=False,
            details=details,
        )

class TimeoutError(ApplicationError):
    def __init__(self, message: str = "Operation timed out", details: Optional[Any] = None):
        super().__init__(
            message=message,
            status_code=504,
            canonical_code=ERROR_CODE_TIMEOUT,
            retryable=True,
            details=details,
        )
