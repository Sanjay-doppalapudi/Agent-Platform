#!/usr/bin/env python3
"""
Test script to validate Docker integration enhancements.

This script tests the key Docker integration features including:
- Automatic Docker detection
- Graceful fallback to mock mode
- Enhanced error handling
- Configuration validation
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_sandbox_manager_docker_detection():
    """Test SandboxManager Docker detection capabilities."""
    print("🔍 Testing SandboxManager Docker detection...")
    
    try:
        from agent_platform.sandbox.manager import SandboxManager
        
        # Initialize manager
        manager = SandboxManager()
        
        # Test Docker availability check
        is_available = manager.is_docker_available()
        print(f"   Docker available: {is_available}")
        
        # Test Docker info retrieval
        docker_info = manager.get_docker_info()
        print(f"   Docker info: {docker_info}")
        
        # Test assert method (should not raise if properly handled)
        try:
            manager.assert_docker_available()
            print("   ✓ Docker assertion passed")
        except Exception as e:
            if "Docker is not available" in str(e):
                print("   ✓ Docker assertion properly failed (expected)")
            else:
                print(f"   ❌ Unexpected error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_mock_sandbox_manager():
    """Test MockSandboxManager functionality."""
    print("\n🔍 Testing MockSandboxManager...")
    
    try:
        from agent_platform.sandbox.mock_manager import MockSandboxManager
        
        # Initialize mock manager
        manager = MockSandboxManager()
        
        # Test basic functionality
        workspace_path = "/tmp/test_workspace"
        sandbox_id = manager.create_sandbox(workspace_path)
        
        print(f"   Created sandbox: {sandbox_id[:12]}")
        
        # Test code execution
        result = manager.execute_code(sandbox_id, "print('Hello from mock')")
        print(f"   Execution result: {result['success']}")
        print(f"   Warning included: {'warning' in result}")
        
        # Cleanup
        manager.destroy_sandbox(sandbox_id)
        print("   ✓ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation."""
    print("\n🔍 Testing configuration validation...")
    
    try:
        from agent_platform.config import get_settings, SandboxConfig
        
        # Test settings loading
        settings = get_settings()
        print(f"   Settings loaded successfully")
        
        # Test Docker configuration fields
        assert hasattr(settings.sandbox, 'docker_auto_detect')
        assert hasattr(settings.sandbox, 'docker_required')
        assert hasattr(settings.sandbox, 'docker_connection_timeout')
        assert hasattr(settings.sandbox, 'docker_max_retries')
        print("   ✓ All Docker config fields present")
        
        # Test SandboxConfig validation
        config = SandboxConfig(timeout=30, memory_limit="512m")
        print("   ✓ SandboxConfig validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_api_docker_endpoints():
    """Test API Docker endpoints."""
    print("\n🔍 Testing API Docker endpoints...")
    
    try:
        from agent_platform.api.main import app
        from fastapi.testclient import TestClient
        
        # Test the new Docker info endpoint
        client = TestClient(app)
        
        # Test Docker info endpoint
        response = client.get("/sandbox/docker-info")
        print(f"   Docker info endpoint status: {response.status_code}")
        
        data = response.json()
        print(f"   Docker available: {data.get('available', 'unknown')}")
        print(f"   Docker status: {data.get('status', 'unknown')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_docker_import_handling():
    """Test Docker import error handling."""
    print("\n🔍 Testing Docker import handling...")
    
    try:
        # Test import of Docker-related modules
        try:
            import docker
            print("   ✓ Docker library available")
        except ImportError:
            print("   ⚠️  Docker library not available (expected in some environments)")
        
        # Test configuration import
        from agent_platform.sandbox.manager import SandboxManager, SandboxConfig
        from agent_platform.sandbox.mock_manager import MockSandboxManager
        print("   ✓ All modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all Docker integration tests."""
    print("🚀 Starting Docker Integration Validation\n")
    
    tests = [
        test_docker_import_handling,
        test_sandbox_manager_docker_detection,
        test_mock_sandbox_manager,
        test_configuration_validation,
        test_api_docker_endpoints,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("   ✅ All tests passed!")
        return 0
    else:
        print("   ❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())