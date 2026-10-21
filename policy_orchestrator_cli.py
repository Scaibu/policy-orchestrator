"""
================================================================================
BLUEPRINT: GLOBAL CLI BOOTSTRAPPER (POLICY-ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Global executable entrypoint installed into system/user PATH.
   Automatically adds policy-orchestrator root to sys.path and invokes main().
================================================================================
"""
import sys
from pathlib import Path

PKG_ROOT = str(Path(__file__).resolve().parent)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.api.cli.main import main


if __name__ == "__main__":
    main()
