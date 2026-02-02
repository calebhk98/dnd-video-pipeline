"""
routers/upload.py - Re-exports the router from upload_routes.py.

Routes are defined in:
    upload_routes.py - All HTTP and WebSocket route handlers.
    upload_tasks.py  - Background task helpers (_run_stage1 ... _run_stage4).
"""

from routers.upload_routes import router  # noqa: F401
