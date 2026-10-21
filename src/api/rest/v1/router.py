"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 UNIFIED FACADE ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Unified facade aggregator composing all modular role-based routers for REST API V1:
   - System Router (System health)
   - Folder Scaffold Router (File scaffolding & relationship modeling)
   - File Search Router (Directory tree traversal & inode caching)
   - Code Search Router (Aho-Corasick & Boyer-Moore pattern matching)
   - Module Resolver Router (Python PEP 302/451 import resolution)
   - Git Verifier Router (Merkle tree & submodule verification)

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

from .routers.file_search_router import router as file_search_router
router.include_router(file_search_router)

from .routers.code_search_router import router as code_search_router
router.include_router(code_search_router)

from .routers.module_resolver_router import router as module_resolver_router
router.include_router(module_resolver_router)

from .routers.git_verifier_router import router as git_verifier_router
router.include_router(git_verifier_router)

from .routers.system_router import (
    health_check,
)
