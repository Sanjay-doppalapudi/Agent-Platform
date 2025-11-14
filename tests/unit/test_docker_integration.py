"""
Tests for Docker integration and connectivity detection.

This module tests the Docker detection, fallback mechanisms,
and integration with both SandboxManager and MockSandboxManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_platform.sandbox.manager import SandboxManager, SandboxConfig
from agent_platform.sandbox.mock_manager import MockSandboxManager
from agent_platform.config import get_settings
import docker
from docker.errors import DockerException


class TestDockerConnectivity:
    """Test Docker connectivity detection and handling."""

    def test_docker_available_success(self):
        """Test successful Docker connection."""
        with patch('docker.from_env') as mock_docker, \
             patch.object(SandboxManager, '_initialize_docker_client') as mock_init:
            
            # Mock successful Docker connection
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.version.return_value = {"Version": "24.0.0"}
            mock_docker.return_value = mock_client
            
            # Initialize manager
            manager = SandboxManager()
            
            # Verify Docker is detected as available
            assert manager.is_docker_available() is True
            
            # Verify Docker info is accessible
            docker_info = manager.get_docker_info()
            assert docker_info["available"] is True
            assert docker_info["version"] == "24.0.0"

    def test_docker_not_available_daemon_not_running(self):
        """Test Docker detection when daemon is not running."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock Docker daemon not running
            mock_docker.side_effect = DockerException("Connection refused")
            
            # Initialize manager
            manager = SandboxManager()
            
            # Verify Docker is not available
            assert manager.is_docker_available() is False
            assert manager.docker_available is False
            
            # Verify Docker info reflects unavailability
            docker_info = manager.get_docker_info()
            assert docker_info["available"] is False
            assert docker_info["status"] == "unavailable"

    def test_docker_library_not_available(self):
        """Test when Docker library is not installed."""
        with patch('docker.from_env', side_effect=ImportError("No module named 'docker'")):
            
            # Initialize manager
            manager = SandboxManager()
            
            # Verify Docker is not available
            assert manager.is_docker_available() is False
            assert manager.docker_available is False

    def test_docker_connection_timeout(self):
        """Test Docker connection timeout handling."""
        with patch('docker.from_env') as mock_docker, \
             patch('time.sleep'):
            
            # Mock Docker connection timeout
            mock_docker.side_effect = DockerException("Connection timeout")
            
            # Initialize manager
            manager = SandboxManager()
            
            # Verify Docker is not available
            assert manager.is_docker_available() is False

    def test_docker_info_with_error(self):
        """Test Docker info retrieval when Docker fails during info call."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock successful initial connection but failed version call
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.version.side_effect = DockerException("API error")
            mock_docker.return_value = mock_client
            
            # Initialize manager
            manager = SandboxManager()
            
            # Verify Docker info reflects error
            docker_info = manager.get_docker_info()
            assert docker_info["available"] is False
            assert docker_info["status"] == "error"


class TestSandboxManagerOperations:
    """Test SandboxManager operations with Docker detection."""

    def test_create_sandbox_requires_docker(self):
        """Test that create_sandbox requires Docker to be available."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock Docker not available
            mock_docker.side_effect = DockerException("Docker not running")
            
            manager = SandboxManager()
            
            # Attempting to create sandbox should raise DockerException
            with pytest.raises(DockerException, match="Docker is not available"):
                manager.create_sandbox("/tmp/workspace")

    def test_execute_code_graceful_docker_unavailable(self):
        """Test execute_code handles Docker unavailability gracefully."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock Docker not available
            mock_docker.side_effect = DockerException("Docker not running")
            
            manager = SandboxManager()
            
            # Execute code should return graceful error
            result = manager.execute_code("fake_id", "print('hello')")
            
            assert result["success"] is False
            assert result["exit_code"] == -1
            assert "Docker is not available" in result["stderr"]

    def test_destroy_sandbox_handles_docker_unavailable(self):
        """Test destroy_sandbox handles Docker unavailability gracefully."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock Docker not available
            mock_docker.side_effect = DockerException("Docker not running")
            
            manager = SandboxManager()
            
            # Destroy should not raise exception
            manager.destroy_sandbox("fake_id")  # Should not raise

    def test_get_sandbox_info_with_docker_unavailable(self):
        """Test get_sandbox_info handles Docker unavailability gracefully."""
        with patch('docker.from_env') as mock_docker:
            
            # Mock Docker not available
            mock_docker.side_effect = DockerException("Docker not running")
            
            manager = SandboxManager()
            
            # Add fake sandbox to test
            manager.active_sandboxes["fake_id"] = MagicMock()
            
            # Get sandbox info should handle gracefully
            info = manager.get_sandbox_info("fake_id")
            assert "docker_available" in info
            assert info["docker_available"] is False


class TestConfigurationValidation:
    """Test Docker-related configuration validation."""

    def test_sandbox_config_docker_settings(self):
        """Test that sandbox config accepts Docker-related settings."""
        settings = get_settings()
        
        # Verify Docker configuration fields exist
        assert hasattr(settings.sandbox, 'docker_auto_detect')
        assert hasattr(settings.sandbox, 'docker_required')
        assert hasattr(settings.sandbox, 'docker_connection_timeout')
        assert hasattr(settings.sandbox, 'docker_max_retries')
        
        # Verify defaults
        assert settings.sandbox.docker_auto_detect is True
        assert settings.sandbox.docker_required is False
        assert settings.sandbox.docker_connection_timeout == 5
        assert settings.sandbox.docker_max_retries == 3

    def test_sandbox_config_timeout_validation(self):
        """Test sandbox config timeout validation."""
        from pydantic import ValidationError
        from agent_platform.config import SandboxConfig
        
        # Valid timeout
        config = SandboxConfig(timeout=30)
        assert config.timeout == 30
        
        # Invalid timeout (too low)
        with pytest.raises(ValidationError):
            SandboxConfig(timeout=0)
        
        # Invalid timeout (too high)
        with pytest.raises(ValidationError):
            SandboxConfig(timeout=400)

    def test_sandbox_config_memory_limit_validation(self):
        """Test sandbox config memory limit validation."""
        from pydantic import ValidationError
        from agent_platform.config import SandboxConfig
        
        # Valid memory limits
        config = SandboxConfig(memory_limit="512m")
        assert config.memory_limit == "512m"
        
        # Invalid memory limits
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit="invalid")
        
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit="512")


class TestMockSandboxManager:
    """Test MockSandboxManager behavior."""

    def test_mock_manager_initialization(self):
        """Test MockSandboxManager initializes properly."""
        manager = MockSandboxManager()
        
        assert manager.config is not None
        assert len(manager.active_sandboxes) == 0
        assert manager.temp_dir.exists()

    def test_mock_create_sandbox(self):
        """Test MockSandboxManager create_sandbox functionality."""
        manager = MockSandboxManager()
        
        sandbox_id = manager.create_sandbox("/tmp/test_workspace")
        
        assert sandbox_id is not None
        assert len(sandbox_id) == 12  # MD5 hash truncated
        assert sandbox_id in manager.active_sandboxes

    def test_mock_execute_code(self):
        """Test MockSandboxManager execute_code functionality."""
        manager = MockSandboxManager()
        
        # Create sandbox
        sandbox_id = manager.create_sandbox("/tmp/test_workspace")
        
        # Execute code
        result = manager.execute_code(sandbox_id, "print('Hello from mock')")
        
        assert result["success"] is True
        assert "Hello from mock" in result["stdout"]
        assert "mock mode" in result.get("warning", "")

    def test_mock_execute_code_security_warning(self):
        """Test that mock execution includes security warnings."""
        manager = MockSandboxManager()
        
        sandbox_id = manager.create_sandbox("/tmp/test_workspace")
        
        result = manager.execute_code(sandbox_id, "print('test')")
        
        assert "warning" in result
        assert "mock mode" in result["warning"]
        assert "no isolation" in result["warning"]


class TestDockerManagerSelection:
    """Test Docker manager selection logic in API."""

    def test_api_manager_creation_with_docker(self):
        """Test API creates Docker manager when available."""
        from agent_platform.api.main import create_sandbox_manager
        from agent_platform.config import get_settings
        
        settings = get_settings()
        settings.sandbox.enabled = True
        settings.sandbox.mock_mode = False
        
        with patch('docker.from_env') as mock_docker:
            # Mock successful Docker connection
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.version.return_value = {"Version": "24.0.0"}
            mock_docker.return_value = mock_client
            
            manager = create_sandbox_manager(settings)
            
            assert manager is not None
            assert isinstance(manager, SandboxManager)

    def test_api_manager_creation_fallback_mock(self):
        """Test API falls back to mock manager when Docker unavailable."""
        from agent_platform.api.main import create_sandbox_manager
        from agent_platform.config import get_settings
        
        settings = get_settings()
        settings.sandbox.enabled = True
        settings.sandbox.mock_mode = True
        
        with patch('docker.from_env') as mock_docker:
            # Mock Docker not available
            mock_docker.side_effect = DockerException("Docker not running")
            
            manager = create_sandbox_manager(settings)
            
            assert manager is not None
            assert isinstance(manager, MockSandboxManager)

    def test_api_manager_creation_disabled(self):
        """Test API returns None when sandbox disabled."""
        from agent_platform.api.main import create_sandbox_manager
        from agent_platform.config import get_settings
        
        settings = get_settings()
        settings.sandbox.enabled = False
        
        manager = create_sandbox_manager(settings)
        
        assert manager is None


class TestDockerAPIEndpoints:
    """Test Docker-related API endpoints."""

    @pytest.mark.asyncio
    async def test_docker_info_endpoint_no_manager(self):
        """Test Docker info endpoint when no manager available."""
        from agent_platform.api.main import app, app_state
        
        # Mock no sandbox manager
        app_state["sandbox_manager"] = None
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/sandbox/docker-info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["available"] is False
        assert data["status"] == "no_manager"

    @pytest.mark.asyncio
    async def test_docker_info_endpoint_mock_mode(self):
        """Test Docker info endpoint in mock mode."""
        from agent_platform.api.main import app, app_state
        
        # Mock mock sandbox manager
        app_state["sandbox_manager"] = MockSandboxManager()
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/sandbox/docker-info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["available"] is False
        assert data["status"] == "mock_mode"

    @pytest.mark.asyncio
    async def test_docker_info_endpoint_docker_mode(self):
        """Test Docker info endpoint in Docker mode."""
        from agent_platform.api.main import app, app_state
        
        # Mock Docker sandbox manager with successful connection
        with patch('docker.from_env') as mock_docker:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.version.return_value = {"Version": "24.0.0"}
            mock_client.containers.list.return_value = []
            mock_client.images.list.return_value = []
            mock_docker.return_value = mock_client
            
            manager = SandboxManager()
            app_state["sandbox_manager"] = manager
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/sandbox/docker-info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["available"] is True
        assert data["status"] == "connected"
        assert data["version"] == "24.0.0"