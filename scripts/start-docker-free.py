#!/usr/bin/env python3
"""
Startup script for Agent Platform in Docker-free mode.

This script provides an easy way to start the platform without Docker,
with automatic configuration for mock sandbox functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pydantic-settings',
        'docker'  # Optional, for Docker mode
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_docker_free_config():
    """Setup Docker-free configuration."""
    env_file = Path(".env")
    docker_free_env = Path(".env.docker-free.example")
    
    if not env_file.exists() and docker_free_env.exists():
        print("📋 Creating .env from Docker-free example...")
        docker_free_env.copy(env_file)
        print("✅ Configuration created. You can edit .env to customize settings.")
    elif not env_file.exists():
        print("⚠️  No .env file found. Creating basic configuration...")
        with open(env_file, 'w') as f:
            f.write("""# Agent Platform Configuration (Docker-Free Mode)
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Sandbox Configuration (Docker-Free)
SANDBOX_ENABLED=true
SANDBOX_MOCK_MODE=true
SANDBOX_TIMEOUT=30

# Security
SECRET_KEY=change-this-secret-key-in-production
""")
        print("✅ Basic configuration created.")
    else:
        print("✅ Configuration file exists.")

def start_platform():
    """Start the Agent Platform."""
    print("🚀 Starting Agent Platform in Docker-free mode...")
    print()
    print("📝 IMPORTANT SECURITY NOTICE:")
    print("   Mock sandbox mode provides NO security isolation!")
    print("   Code executes directly on your system with full privileges.")
    print("   Use only for development and trusted code.")
    print()
    
    try:
        # Import and run the application
        from agent_platform.api.main import app
        import uvicorn
        
        settings = get_settings()
        
        print(f"🔧 Configuration:")
        print(f"   Sandbox enabled: {settings.sandbox.enabled}")
        print(f"   Mock mode: {settings.sandbox.mock_mode}")
        print(f"   API port: {settings.api.port}")
        print()
        
        # Start the server
        uvicorn.run(
            app,
            host=settings.api.host,
            port=settings.api.port,
            reload=settings.api.reload and settings.environment == "development"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start platform: {e}")
        sys.exit(1)

def get_settings():
    """Import and return settings."""
    from agent_platform.config import get_settings
    return get_settings()

def main():
    """Main entry point."""
    print("🤖 Agent Platform - Docker-Free Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src/agent_platform").exists():
        print("❌ Error: Run this script from the project root directory")
        print("   Current directory should contain 'src/agent_platform'")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup configuration
    setup_docker_free_config()
    
    # Start the platform
    start_platform()

if __name__ == "__main__":
    main()