"""
app.py - DND Video Pipeline - FastAPI application entry point.

This file is intentionally slim: it creates the FastAPI app, adds middleware,
mounts the three APIRouter modules, and serves static files.

All business logic lives in the router modules and core.py:
  Web/core.py              - shared state (jobs_db, ConnectionManager, paths)
  Web/routers/upload.py    - pipeline workflow endpoints (upload, transcript, ...)
  Web/routers/settings.py  - API key read/write endpoints
  Web/routers/history.py   - past-session history endpoint

Run with:
    cd Web && uvicorn app:app --reload
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

# Ensure the project root (parent of this Web/ directory) is on sys.path so
# that `from src.orchestrator.pipeline import ...` works regardless of the
# working directory when uvicorn is launched.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
	sys.path.insert(0, _root)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core import FRONTEND_DIR, OUTPUTS_DIR, load_jobs_from_disk
from routers import upload, settings, history

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Restore persisted job metadata from disk so interrupted sessions can be resumed."""
	logger.info("Starting up: loading persisted jobs from disk...")
	load_jobs_from_disk()
	yield


app = FastAPI(title="DND-Video-Pipeline API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static file mounts (must come last - the "/" mount catches everything else)
# ---------------------------------------------------------------------------
if os.path.exists(OUTPUTS_DIR):
	# Serve generated videos and transcripts at /outputs/<session>/<file>
	app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
else:
	logger.warning("%s not found - generated outputs will not be served.", OUTPUTS_DIR)

if os.path.exists(FRONTEND_DIR):
	# Serve the frontend SPA; html=True enables index.html fallback routing
	app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
	logger.warning("%s not found - frontend will not be served.", FRONTEND_DIR)
