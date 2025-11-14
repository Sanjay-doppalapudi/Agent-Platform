"""
FastAPI application for Agent Platform.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_platform.config import get_settings
from agent_platform.sandbox.manager import SandboxManager
from agent_platform.utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

# Global state
app_state = {
    "start_time": time.time(),
    "sandbox_manager": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agent Platform API")
    settings = get_settings()
    app_state["sandbox_manager"] = SandboxManager()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Platform API")
    if app_state["sandbox_manager"]:
        # Cleanup sandboxes
        for sandbox_id in list(app_state["sandbox_manager"].active_sandboxes.keys()):
            app_state["sandbox_manager"].destroy_sandbox(sandbox_id)


# Create FastAPI app
app = FastAPI(
    title="AI Agent Platform",
    description="Open-source AI developer agent platform with secure code execution",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration
        }
    )
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Health endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    from agent_platform.api.models import HealthResponse
    
    uptime = time.time() - app_state["start_time"]
    active_sandboxes = len(app_state["sandbox_manager"].active_sandboxes) if app_state["sandbox_manager"] else 0
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime=uptime,
        active_sandboxes=active_sandboxes
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Agent Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }
