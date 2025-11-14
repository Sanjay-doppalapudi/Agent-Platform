@echo off
REM Startup script for Agent Platform in Docker-free mode (Windows)

echo 🤖 Agent Platform - Docker-Free Startup (Windows)
echo ================================================
echo.

REM Check if we're in the right directory
if not exist "src\agent_platform" (
    echo ❌ Error: Run this script from the project root directory
    echo    Current directory should contain 'src\agent_platform'
    pause
    exit /b 1
)

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo 📦 Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing basic dependencies...
    pip install fastapi uvicorn pydantic pydantic-settings
)

echo ✅ Dependencies checked
echo.

echo 🔧 Setting up configuration...
if not exist ".env" (
    if exist ".env.docker-free.example" (
        copy ".env.docker-free.example" ".env" >nul
        echo ✅ Configuration created from docker-free example
    ) else (
        echo Creating basic configuration...
        (
            echo # Agent Platform Configuration ^(Docker-Free Mode^)
            echo ENVIRONMENT=development
            echo LOG_LEVEL=INFO
            echo DEBUG=true
            echo.
            echo # API Configuration
            echo API_HOST=0.0.0.0
            echo API_PORT=8000
            echo API_RELOAD=true
            echo.
            echo # Sandbox Configuration ^(Docker-Free^)
            echo SANDBOX_ENABLED=true
            echo SANDBOX_MOCK_MODE=true
            echo SANDBOX_TIMEOUT=30
            echo.
            echo # Security
            echo SECRET_KEY=change-this-secret-key-in-production
        ) > .env
        echo ✅ Basic configuration created
    )
) else (
    echo ✅ Configuration file exists
)

echo.
echo 🚀 Starting Agent Platform...
echo.
echo 📝 IMPORTANT SECURITY NOTICE:
echo    Mock sandbox mode provides NO security isolation!
echo    Code executes directly on your system with full privileges.
echo    Use only for development and trusted code.
echo.

REM Run the Python startup script
python scripts\start-docker-free.py

pause