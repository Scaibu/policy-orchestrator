"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 UNIFIED FACADE ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Unified facade aggregator composing all modular role-based routers for REST API V1:
   - System Router (System health)
   - Folder Scaffold Router (File scaffolding & relationship modeling)

2. ZERO-INLINE-COMMENT DOCTRINE:
   Document the entire algorithmic workflow in top docblock.
================================================================================
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

from .routers.system_router import router as system_router
router.include_router(system_router)

from .routers.folder_scaffold_router import router as folder_scaffold_router
router.include_router(folder_scaffold_router)

# -----------------------------------------------------------------------------
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# -----------------------------------------------------------------------------

from .routers.system_router import (
    health_check,
)
