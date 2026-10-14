"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 SYSTEM ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Encapsulates HTTP REST endpoints for system-level capabilities:
   - System Health Probe (/health)

2. ARCHITECTURAL LAYOUT & DESIGN PILLARS:
   - Zero-Inline-Comment Doctrine: Docblock serves as full architectural specification.
   - Strict Protocol Envelope: All handlers return success/error envelopes with trace IDs.
================================================================================
"""

from typing import Dict, Any
from fastapi import APIRouter, Request

from src.api.rest.envelope import build_success_envelope

router = APIRouter()


@router.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    trace_id = request.headers.get("x-trace-id") or request.headers.get("traceparent")
    return build_success_envelope(
        data={"status": "healthy", "service": "policy-orchestrator", "version": "0.1.0"},
        trace_id=trace_id,
    )
