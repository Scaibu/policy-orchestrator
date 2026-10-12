from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ErrorDetail(BaseModel):
    field: Optional[str] = None
    issue: str
    rule: Optional[str] = None
    rejectedValue: Optional[Any] = None

class ErrorContainer(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: Optional[List[Dict[str, Any]]] = None

class PaginationMeta(BaseModel):
    page: int
    pageSize: int
    totalItems: int
    totalPages: int
    hasNext: bool
    hasPrev: bool

class ResponseMeta(BaseModel):
    requestId: str
    correlationId: str
    causationId: Optional[str] = None
    timestamp: str
    executionTimeMs: int = 0
    apiVersion: str = "v1"
    status: str = "success"
    trace_id: str
    auditLogId: Optional[str] = None
    operationId: Optional[str] = None
    deprecatedFields: Optional[List[str]] = None
    supportedVersions: Optional[List[str]] = None
    staleness: Optional[Dict[str, Any]] = None
    pagination: Optional[Dict[str, Any]] = None
    links: Optional[Dict[str, Any]] = None

class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    statusCode: int = 200
    data: Optional[T] = None
    error: Optional[ErrorContainer] = None
    errors: Optional[List[Dict[str, Any]]] = None
    meta: ResponseMeta

class ApiErrorResponse(BaseModel):
    success: bool = False
    statusCode: int = 400
    data: Optional[Any] = None
    error: ErrorContainer
    errors: Optional[List[Dict[str, Any]]] = None
    meta: ResponseMeta

class TraceContext(BaseModel):
    trace_id: str
    request_id: str
    correlation_id: str
    causation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
