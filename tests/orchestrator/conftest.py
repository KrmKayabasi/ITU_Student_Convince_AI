"""
Test setup for the realtime orchestrator.

The orchestrator modules (backend/orchestrator/*.py) use flat imports
(`import config`, `from cv_hints import ...`) after inserting their own
directory on sys.path, so we do the same here. These tests require the
orchestrator deps (numpy/websockets/google-genai) and are intended to run in
the orchestrator venv; individual tests importorskip what they need so a plain
`pytest` in the CV env skips them gracefully.
"""

import os
import sys

_ORCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backend", "orchestrator")
)
if _ORCH_DIR not in sys.path:
    sys.path.insert(0, _ORCH_DIR)
