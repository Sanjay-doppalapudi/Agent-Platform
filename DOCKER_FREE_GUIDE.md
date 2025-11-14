# Running Agent Platform Without Docker

This guide explains how to run the Agent Platform on Windows systems without Docker, providing immediate functionality while preserving the codebase structure for when Docker becomes available.

## 🚀 Quick Start (Docker-Free Mode)

1. **Copy the Docker-free configuration**:
   ```bash
   cp .env.docker-free.example .env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the platform**:
   ```bash
   python -m agent_platform.api.main
   ```

4. **Check status**:
   - API will start at `http://localhost:8000`
   - Health check: `http://localhost:8000/health`
   - Sandbox status: `http://localhost:8000/sandbox/status`

## ⚠️ Important Security Notice

**Mock sandbox mode provides NO security isolation!** Code executes directly on your host system with full privileges. Use this mode only for:
- Development and testing
- Local experimentation
- Learning and exploration

**Never use mock mode in production or with untrusted code!**

## 🔧 Configuration Options

### Environment Variables for Docker-Free Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_ENABLED` | `true` | Enable sandbox functionality |
| `SANDBOX_MOCK_MODE` | `true` | Use mock implementation when Docker unavailable |
| `SANDBOX_TIMEOUT` | `30` | Maximum execution time (seconds) |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Memory limit (mock mode ignores this) |
| `SANDBOX_CPU_LIMIT` | `1.0` | CPU limit (mock mode ignores this) |

### Configuration Modes

1. **Full Docker Mode** (Recommended when Docker is available):
   ```bash
   SANDBOX_ENABLED=true
   SANDBOX_MOCK_MODE=false
   ```

2. **Mock Mode** (No Docker, limited functionality):
   ```bash
   SANDBOX_ENABLED=true
   SANDBOX_MOCK_MODE=true
   ```

3. **Disabled Mode** (No sandbox functionality):
   ```bash
   SANDBOX_ENABLED=false
   ```

## 🐳 Docker Desktop Installation (Windows)

If you want full functionality with security isolation, install Docker Desktop:

### Prerequisites
- Windows 10/11 (64-bit)
- WSL 2 enabled
- Virtualization enabled in BIOS
- 4GB+ RAM available for Docker

### Installation Steps

1. **Enable WSL 2**:
   ```powershell
   # Run as Administrator in PowerShell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
   # Restart your computer
   ```

2. **Download and install Linux kernel update**:
   - Download from: https://aka.ms/wsl2kernel
   - Run the installer

3. **Set WSL 2 as default**:
   ```powershell
   wsl --set-default-version 2
   ```

4. **Install Docker Desktop**:
   - Download from: https://desktop.docker.com/win/main/amd64/Docker Desktop Installer.exe
   - Run installer with WSL 2 backend enabled

5. **Restart and verify**:
   ```bash
   docker --version
   docker run hello-world
   ```

### Post-Installation Configuration

1. **Build the sandbox image**:
   ```bash
   cd docker
   docker build -f Dockerfile.sandbox -t agent-sandbox:latest .
   ```

2. **Test Docker functionality**:
   ```bash
   # Should automatically use Docker-based sandbox
   python -m agent_platform.api.main
   ```

## 🔍 Understanding Sandbox Modes

### Docker Mode (Full Security)
- **Pros**: Complete security isolation, resource limits, network isolation
- **Cons**: Requires Docker, additional system resources
- **Use**: Production, untrusted code execution

### Mock Mode (No Security)
- **Pros**: No additional dependencies, faster startup
- **Cons**: NO security isolation, full host access
- **Use**: Development, testing, trusted code only

### Disabled Mode
- **Pros**: Minimal resource usage
- **Cons**: No code execution capability
- **Use**: API-only functionality, file operations

## 🛠️ API Endpoints

### Health and Status
- `GET /health` - Application health with sandbox mode info
- `GET /sandbox/status` - Detailed sandbox configuration and status

### Sandbox Operations
- `POST /sandbox/create` - Create new sandbox
- `POST /sandbox/{id}/execute` - Execute code in sandbox
- `DELETE /sandbox/{id}` - Destroy sandbox

### Example API Usage

```bash
# Check sandbox status
curl http://localhost:8000/sandbox/status

# Create a sandbox
curl -X POST "http://localhost:8000/sandbox/create?workspace_path=/tmp/workspace"

# Execute code
curl -X POST "http://localhost:8000/sandbox/{sandbox_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello, World!\")", "language": "python"}'
```

## 📁 File Structure Changes

The Docker-free implementation adds these key files:

```
src/agent_platform/
├── sandbox/
│   ├── manager.py          # Original Docker-based manager
│   └── mock_manager.py     # New mock implementation
└── api/
    ├── main.py            # Updated with Docker detection
    └── models.py          # Added sandbox status models
```

## 🔄 Transitioning to Docker

When Docker becomes available:

1. **Update configuration**:
   ```bash
   SANDBOX_MOCK_MODE=false
   ```

2. **Restart the application**:
   ```bash
   python -m agent_platform.api.main
   ```

3. **Verify Docker mode**:
   - Check `/sandbox/status` endpoint
   - Should show `"status": "docker"`

## 🚨 Troubleshooting

### Common Issues

1. **"Docker library not available"**:
   ```bash
   pip install docker
   ```

2. **"Mock mode enabled" warnings**:
   - Normal when Docker is not available
   - Set `SANDBOX_MOCK_MODE=false` to suppress

3. **Port already in use**:
   ```bash
   # Change API port
   API_PORT=8001
   ```

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG
DEBUG=true
```

## 📚 Best Practices

### For Development
- Use mock mode for quick iteration
- Keep mock mode warnings visible
- Test in Docker mode before production

### For Production
- Always use Docker mode when possible
- Monitor resource usage
- Implement proper logging and monitoring

### Security Considerations
- Never run untrusted code in mock mode
- Use Docker mode for any code execution
- Implement proper input validation
- Monitor sandbox resource usage

## 🆘 Getting Help

### Log Analysis
Check application logs for detailed error information:
```bash
# View logs
tail -f logs/agent_platform.log

# Debug mode
LOG_LEVEL=DEBUG python -m agent_platform.api.main
```

### API Testing
Use the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Configuration Validation
Test your configuration:
```python
from agent_platform.config import get_settings
settings = get_settings()
print(f"Sandbox enabled: {settings.sandbox.enabled}")
print(f"Mock mode: {settings.sandbox.mock_mode}")
```

---

**Remember**: Mock sandbox mode is a development aid, not a security solution. Always use Docker mode for production or untrusted code execution.