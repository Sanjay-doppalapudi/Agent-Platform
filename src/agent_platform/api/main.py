"""
FastAPI application for Agent Platform.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_platform.config import get_settings
from agent_platform.sandbox.manager import SandboxManager
from agent_platform.sandbox.mock_manager import MockSandboxManager
from agent_platform.utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

def create_sandbox_manager(settings):
    """
    Create appropriate sandbox manager based on configuration and environment.
    
    Tries to use Docker-based SandboxManager first, falls back to MockSandboxManager
    if Docker is not available or sandbox is disabled.
    """
    # Check if sandbox is explicitly disabled
    if not settings.sandbox.enabled:
        logger.info("Sandbox functionality is disabled in configuration")
        return None
    
    # Try Docker-based manager first
    try:
        # Test if Docker is available
        import docker
        docker_client = docker.from_env()
        docker_client.ping()
        
        logger.info("Using Docker-based sandbox manager")
        return SandboxManager()
        
    except ImportError:
        logger.warning("Docker library not available, using mock sandbox manager")
    except Exception as e:
        logger.warning(f"Docker not available or not running: {e}, using mock sandbox manager")
    
    # Use mock manager as fallback
    if settings.sandbox.mock_mode:
        logger.warning("Using MOCK sandbox manager - NO SECURITY ISOLATION")
        return MockSandboxManager()
    else:
        logger.warning("Sandbox disabled - no sandbox functionality available")
        return None

# Global state
app_state = {
    "start_time": time.time(),
    "sandbox_manager": None,
    "sandbox_mode": "unknown"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agent Platform API")
    settings = get_settings()
    
    # Create sandbox manager based on environment
    app_state["sandbox_manager"] = create_sandbox_manager(settings)
    
    # Determine sandbox mode for health endpoint
    if app_state["sandbox_manager"] is None:
        app_state["sandbox_mode"] = "disabled"
    elif isinstance(app_state["sandbox_manager"], MockSandboxManager):
        app_state["sandbox_mode"] = "mock"
    else:
        app_state["sandbox_mode"] = "docker"
    
    logger.info(
        "Sandbox manager initialized",
        extra={
            "sandbox_mode": app_state["sandbox_mode"],
            "enabled": settings.sandbox.enabled
        }
    )
    
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
        active_sandboxes=active_sandboxes,
        sandbox_mode=app_state["sandbox_mode"]
    )


# Sandbox endpoints
@app.get("/sandbox/status")
async def sandbox_status():
    """Get sandbox status and configuration."""
    from agent_platform.api.models import SandboxStatusResponse
    
    settings = get_settings()
    sandbox_manager = app_state["sandbox_manager"]
    
    if sandbox_manager is None:
        status = "disabled"
        active_sandboxes = 0
    elif isinstance(sandbox_manager, MockSandboxManager):
        status = "mock"
        active_sandboxes = len(sandbox_manager.active_sandboxes)
    else:
        status = "docker"
        active_sandboxes = len(sandbox_manager.active_sandboxes)
    
    return SandboxStatusResponse(
        status=status,
        enabled=settings.sandbox.enabled,
        mock_mode=settings.sandbox.mock_mode,
        active_sandboxes=active_sandboxes,
        docker_available=isinstance(sandbox_manager, SandboxManager)
    )
@app.get("/sandbox/docker-info")
async def docker_info_endpoint():
    """Get detailed Docker daemon information."""
    sandbox_manager = app_state["sandbox_manager"]
    
    if sandbox_manager is None:
        return {
            "available": False,
            "status": "no_manager",
            "message": "No sandbox manager available"
        }
    
    if isinstance(sandbox_manager, MockSandboxManager):
        return {
            "available": False,
            "status": "mock_mode",
            "message": "Running in mock mode - Docker not available"
        }
    
    # For Docker-based SandboxManager
    try:
        docker_info = sandbox_manager.get_docker_info()
        return {
            "available": docker_info["available"],
            "status": docker_info["status"],
            "version": docker_info.get("version", "unknown"),
            "api_version": docker_info.get("api_version", "unknown"),
            "containers_running": docker_info.get("containers_running", 0),
            "images_count": docker_info.get("images_count", 0),
            "message": "Docker daemon is running" if docker_info["available"] else "Docker daemon is not available"
        }
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "message": f"Failed to get Docker info: {str(e)}"
        }


@app.post("/sandbox/create")
async def create_sandbox_endpoint(workspace_path: str):
    """Create a new sandbox."""
    if app_state["sandbox_manager"] is None:
        raise HTTPException(status_code=503, detail="Sandbox is disabled")
    
    try:
        sandbox_id = app_state["sandbox_manager"].create_sandbox(workspace_path)
        return {"sandbox_id": sandbox_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create sandbox: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create sandbox: {str(e)}")


@app.post("/sandbox/{sandbox_id}/execute")
async def execute_code_endpoint(sandbox_id: str, code: str, language: str = "python"):
    """Execute code in a sandbox."""
    if app_state["sandbox_manager"] is None:
        raise HTTPException(status_code=503, detail="Sandbox is disabled")
    
    try:
        result = app_state["sandbox_manager"].execute_code(sandbox_id, code, language)
        return result
    except Exception as e:
        logger.error(f"Failed to execute code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute code: {str(e)}")


@app.delete("/sandbox/{sandbox_id}")
async def destroy_sandbox_endpoint(sandbox_id: str):
    """Destroy a sandbox."""
    if app_state["sandbox_manager"] is None:
        raise HTTPException(status_code=503, detail="Sandbox is disabled")
    
    try:
        app_state["sandbox_manager"].destroy_sandbox(sandbox_id)
        return {"status": "destroyed"}
    except Exception as e:
        logger.error(f"Failed to destroy sandbox: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to destroy sandbox: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Agent Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "sandbox_status": "/sandbox/status"
    }