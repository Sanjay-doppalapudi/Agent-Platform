# Docker Integration Guide

This guide describes the enhanced Docker integration for the Agent Platform, including automatic detection, graceful fallback, and comprehensive error handling.

## Overview

The Agent Platform now seamlessly integrates Docker support with automatic detection and graceful fallback mechanisms. The system intelligently detects Docker availability and provides appropriate sandbox implementations based on the environment.

## Key Features

### 1. Automatic Docker Detection
- Detects Docker daemon availability on startup
- Provides detailed status information about Docker connectivity
- Handles various failure scenarios gracefully

### 2. Graceful Fallback Mechanism
- Falls back to MockSandboxManager when Docker is unavailable
- Maintains the same API interface regardless of backend
- Provides clear logging and status reporting

### 3. Enhanced Error Handling
- Comprehensive error handling for Docker connection failures
- Detailed logging for troubleshooting
- Graceful degradation without breaking functionality

### 4. Configuration Options
- Flexible configuration for Docker settings
- Validation of Docker-related parameters
- Environment variable support

## Architecture

### Core Components

#### SandboxManager
The enhanced `SandboxManager` class now includes:

```python
from agent_platform.sandbox.manager import SandboxManager

# Initialize with automatic Docker detection
manager = SandboxManager()

# Check Docker availability
if manager.is_docker_available():
    print(f"Docker version: {manager.get_docker_info()['version']}")
else:
    print("Docker not available - using fallback")
```

**New Methods:**
- `is_docker_available()` - Check Docker connectivity
- `get_docker_info()` - Get Docker daemon information
- `assert_docker_available()` - Assert Docker availability (raises exception if not)

#### Configuration
Enhanced configuration with Docker-specific settings:

```python
from agent_platform.config import get_settings

settings = get_settings()

# Docker-related configuration
print(f"Auto-detect Docker: {settings.sandbox.docker_auto_detect}")
print(f"Docker required: {settings.sandbox.docker_required}")
print(f"Connection timeout: {settings.sandbox.docker_connection_timeout}")
print(f"Max retries: {settings.sandbox.docker_max_retries}")
```

#### API Integration
Enhanced API endpoints provide Docker status information:

- `GET /sandbox/status` - General sandbox status
- `GET /sandbox/docker-info` - Detailed Docker information
- `POST /sandbox/create` - Create sandbox (auto-selects backend)
- `POST /sandbox/{id}/execute` - Execute code in sandbox

## Usage Examples

### Basic Usage

```python
from agent_platform.sandbox.manager import SandboxManager

# Automatic Docker detection and initialization
manager = SandboxManager()

# Check what's available
if manager.is_docker_available():
    print("Using Docker-based sandbox")
    docker_info = manager.get_docker_info()
    print(f"Docker version: {docker_info['version']}")
else:
    print("Docker not available - platform will use mock sandbox")
```

### Configuration-Based Usage

```python
from agent_platform.config import get_settings
from agent_platform.sandbox.manager import SandboxManager
from agent_platform.sandbox.mock_manager import MockSandboxManager

settings = get_settings()

if settings.sandbox.enabled and not settings.sandbox.docker_required:
    try:
        manager = SandboxManager()
        if manager.is_docker_available():
            print("Docker sandbox available")
        else:
            print("Falling back to mock sandbox")
            manager = MockSandboxManager()
    except Exception as e:
        if settings.sandbox.mock_mode:
            manager = MockSandboxManager()
        else:
            raise e
else:
    manager = None
```

### API Usage

```python
import httpx

async with httpx.AsyncClient() as client:
    # Check Docker status
    response = await client.get("http://localhost:8000/sandbox/docker-info")
    docker_info = response.json()
    
    if docker_info["available"]:
        print(f"Docker is running: {docker_info['version']}")
    else:
        print(f"Docker status: {docker_info['status']}")
    
    # Create sandbox (auto-selects backend)
    response = await client.post("http://localhost:8000/sandbox/create", 
                                json={"workspace_path": "/tmp/workspace"})
    sandbox_data = response.json()
```

## Configuration Reference

### Environment Variables

```bash
# Enable/disable sandbox functionality
SANDBOX__ENABLED=true

# Mock mode (use when Docker unavailable)
SANDBOX__MOCK_MODE=false

# Docker auto-detection
SANDBOX__DOCKER_AUTO_DETECT=true

# Require Docker (fail startup if not available)
SANDBOX__DOCKER_REQUIRED=false

# Docker connection timeout (seconds)
SANDBOX__DOCKER_CONNECTION_TIMEOUT=5

# Maximum retry attempts
SANDBOX__DOCKER_MAX_RETRIES=3

# Docker image for sandbox
SANDBOX__IMAGE=agent-sandbox:latest

# Resource limits
SANDBOX__MEMORY_LIMIT=512m
SANDBOX__CPU_LIMIT=1.0
SANDBOX__TIMEOUT=30
```

### Configuration File (.env)

```env
# Agent Platform Configuration

# API Settings
API__HOST=0.0.0.0
API__PORT=8000

# Sandbox Configuration
SANDBOX__ENABLED=true
SANDBOX__MOCK_MODE=false
SANDBOX__DOCKER_AUTO_DETECT=true
SANDBOX__DOCKER_REQUIRED=false
SANDBOX__DOCKER_CONNECTION_TIMEOUT=5
SANDBOX__DOCKER_MAX_RETRIES=3

# Docker Image and Resources
SANDBOX__IMAGE=agent-sandbox:latest
SANDBOX__MEMORY_LIMIT=512m
SANDBOX__CPU_LIMIT=1.0
SANDBOX__TIMEOUT=30
SANDBOX__NETWORK_DISABLED=true

# Security
SECURITY__SECRET_KEY=your-super-secret-key-change-this-in-production
```

## Docker Integration States

The platform can operate in several states:

### 1. Docker Available
```json
{
    "status": "docker",
    "docker_available": true,
    "active_sandboxes": 2
}
```

### 2. Mock Mode (Docker Unavailable)
```json
{
    "status": "mock",
    "docker_available": false,
    "active_sandboxes": 1,
    "warning": "Running in MOCK mode - NO SECURITY ISOLATION"
}
```

### 3. Disabled
```json
{
    "status": "disabled",
    "docker_available": false,
    "active_sandboxes": 0
}
```

## Error Handling

### Docker Connection Failures

The platform handles various Docker failure scenarios:

1. **Docker daemon not running**
   ```
   ERROR: Docker daemon not available or not running
   INFO: Falling back to mock sandbox manager
   ```

2. **Docker library not available**
   ```
   ERROR: Docker library not available
   INFO: Checking mock mode preference
   ```

3. **Connection timeout**
   ```
   WARNING: Docker connectivity check failed (Connection timeout)
   INFO: Using mock sandbox manager
   ```

### Graceful Degradation

All operations continue to function even when Docker is unavailable:

- Code execution returns informative error messages
- Sandbox creation fails with clear explanations
- Existing functionality remains intact

## Security Considerations

### Docker Mode
- Full container isolation
- Resource limits enforced
- Network isolation by default
- Security profiles applied

### Mock Mode
- ⚠️ **WARNING**: No security isolation
- Code executes directly on host system
- Use only for development/testing
- Consider container alternatives for production

## Troubleshooting

### Common Issues

1. **Docker daemon not running**
   ```bash
   # Start Docker daemon
   sudo systemctl start docker
   
   # Check Docker status
   docker ps
   ```

2. **Permission issues**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Docker not installed**
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
```

### Health Checks

```bash
# Check platform health
curl http://localhost:8000/health

# Check Docker status
curl http://localhost:8000/sandbox/docker-info

# Check sandbox status
curl http://localhost:8000/sandbox/status
```

## Best Practices

### Development
- Use mock mode for local development
- Enable detailed logging for troubleshooting
- Test both Docker and non-Docker environments

### Production
- Require Docker for security isolation
- Monitor Docker daemon health
- Set appropriate resource limits
- Use environment-specific configurations

### Monitoring
- Monitor Docker daemon availability
- Track sandbox creation/destruction metrics
- Log Docker connectivity issues
- Set up alerts for Docker failures

## Migration Guide

### From Previous Version

If upgrading from a version without Docker integration:

1. **No code changes required** - Existing code continues to work
2. **Add configuration** - Optionally configure Docker settings
3. **Test both modes** - Verify functionality with and without Docker
4. **Update monitoring** - Add Docker-specific health checks

### Backward Compatibility

- All existing APIs remain unchanged
- MockSandboxManager provides the same interface
- Configuration is backward compatible
- No breaking changes to existing functionality

## Performance Considerations

### Docker Mode
- Container startup overhead (~500ms)
- Resource limits protect system stability
- Network isolation improves security

### Mock Mode
- Faster execution (no container overhead)
- No resource isolation
- Suitable for development/testing only

### Recommendations
- Use Docker mode for production
- Monitor container startup times
- Optimize Docker image size
- Configure appropriate resource limits