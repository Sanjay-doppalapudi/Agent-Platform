"""
Mock sandbox manager for environments without Docker.

This module provides a fallback implementation that simulates sandbox
functionality when Docker is not available, allowing the platform
to run in a limited capacity.
"""

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

from agent_platform.sandbox.manager import SandboxConfig, Sandbox
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class MockSandboxManager:
    """
    Mock implementation of SandboxManager for environments without Docker.
    
    This class provides the same interface as SandboxManager but executes
    code directly on the host system with limited security measures.
    
    WARNING: This is for development/testing only and should NOT be used
    in production environments as it provides NO security isolation.
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize mock sandbox manager.
        
        Args:
            config: Sandbox configuration (ignored in mock mode).
        """
        self.config = config or SandboxConfig()
        self.active_sandboxes: Dict[str, Sandbox] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "agent_platform_mock_sandboxes"
        self.temp_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        
        self.logger.warning(
            "Running in MOCK sandbox mode - NO SECURITY ISOLATION",
            extra={
                "mode": "mock",
                "warning": "Code execution is NOT sandboxed - use only for testing!"
            }
        )
    
    def create_sandbox(self, workspace_path: str) -> str:
        """
        Create a mock sandbox (creates a temporary directory).
        
        Args:
            workspace_path: Path to mount as workspace in the mock container.
        
        Returns:
            Mock sandbox ID (directory path hash).
        
        Raises:
            OSError: If sandbox creation fails.
        """
        try:
            # Create workspace directory
            safe_workspace = Path(workspace_path).resolve()
            safe_workspace.mkdir(parents=True, exist_ok=True)
            
            # Generate sandbox ID based on workspace path
            sandbox_id = hashlib.md5(str(safe_workspace).encode()).hexdigest()[:12]
            
            # Create mock sandbox directory
            mock_dir = self.temp_dir / sandbox_id
            mock_dir.mkdir(exist_ok=True)
            
            # Copy workspace to mock directory (simulating container mount)
            import shutil
            workspace_copy = mock_dir / "workspace"
            if safe_workspace.exists():
                if workspace_copy.exists():
                    shutil.rmtree(workspace_copy)
                shutil.copytree(safe_workspace, workspace_copy, dirs_exist_ok=True)
            
            # Store mock sandbox
            from docker.models.containers import Container
            from unittest.mock import MagicMock
            
            # Create a mock container object
            mock_container = MagicMock()
            mock_container.id = sandbox_id
            mock_container.status = "running"
            mock_container.exec_run = lambda cmd, **kwargs: self._mock_exec_run(cmd, workspace_copy, **kwargs)
            
            sandbox = Sandbox(
                container_id=sandbox_id,
                container=mock_container,
                created_at=time.time(),
                workspace_path=str(workspace_copy),
                config=self.config
            )
            
            self.active_sandboxes[sandbox_id] = sandbox
            
            self.logger.info(
                f"Created mock sandbox",
                extra={
                    "sandbox_id": sandbox_id[:12],
                    "workspace": str(safe_workspace),
                    "mock_dir": str(mock_dir)
                }
            )
            
            return sandbox_id
            
        except Exception as e:
            self.logger.error(f"Failed to create mock sandbox: {e}", exc_info=True)
            raise OSError(f"Failed to create sandbox: {e}")
    
    def execute_code(
        self,
        sandbox_id: str,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Execute code in mock sandbox (direct execution with limited safety).
        
        Args:
            sandbox_id: ID of the mock sandbox.
            code: Code to execute.
            language: Programming language (currently only "python" supported).
            timeout: Execution timeout in seconds (uses config default if None).
        
        Returns:
            Dictionary with execution results.
        
        WARNING: This executes code directly on the host system!
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        timeout = timeout or self.config.timeout
        
        # Prepare command based on language
        if language == "python":
            cmd = ["python", "-c", code]
        else:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "success": False,
                "execution_time": 0.0
            }
        
        start_time = time.time()
        
        try:
            self.logger.warning(
                f"Executing code in MOCK sandbox",
                extra={
                    "sandbox_id": sandbox_id[:12],
                    "warning": "Code executes with full host privileges!",
                    "language": language
                }
            )
            
            # Execute command directly with timeout
            result = subprocess.run(
                cmd,
                cwd=sandbox.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONPATH": sandbox.workspace_path,
                    "WORKSPACE_PATH": sandbox.workspace_path
                }
            )
            
            execution_time = time.time() - start_time
            
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "success": False,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Code execution failed: {e}",
                extra={"sandbox_id": sandbox_id[:12]},
                exc_info=True
            )
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
    
    def execute_file(
        self,
        sandbox_id: str,
        file_path: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Execute a file in mock sandbox.
        
        Args:
            sandbox_id: ID of the mock sandbox.
            file_path: Path to file relative to workspace.
            args: Optional command-line arguments.
            timeout: Execution timeout in seconds.
        
        Returns:
            Execution result dictionary.
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        timeout = timeout or self.config.timeout
        
        # Build command
        cmd = ["python", str(Path(sandbox.workspace_path) / file_path)]
        if args:
            cmd.extend(args)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=sandbox.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONPATH": sandbox.workspace_path,
                    "WORKSPACE_PATH": sandbox.workspace_path
                }
            )
            
            execution_time = time.time() - start_time
            
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "success": False,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "execution_time": execution_time,
                "warning": "Executed in mock mode - no isolation!"
            }
    
    def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        Destroy a mock sandbox (remove temporary directory).
        
        Args:
            sandbox_id: ID of the sandbox to destroy.
        """
        if sandbox_id not in self.active_sandboxes:
            self.logger.warning(f"Mock sandbox not found: {sandbox_id[:12]}")
            return
        
        try:
            # Remove mock directory
            mock_dir = self.temp_dir / sandbox_id
            if mock_dir.exists():
                import shutil
                shutil.rmtree(mock_dir)
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            self.logger.info(f"Destroyed mock sandbox", extra={"sandbox_id": sandbox_id[:12]})
            
        except Exception as e:
            self.logger.error(
                f"Failed to destroy mock sandbox: {e}",
                extra={"sandbox_id": sandbox_id[:12]},
                exc_info=True
            )
    
    def cleanup_old_sandboxes(self, max_age: int = 3600) -> int:
        """
        Clean up mock sandboxes older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds before cleanup.
        
        Returns:
            Number of sandboxes cleaned up.
        """
        current_time = time.time()
        to_destroy = []
        
        for sandbox_id, sandbox in self.active_sandboxes.items():
            age = current_time - sandbox.created_at
            if age > max_age:
                to_destroy.append(sandbox_id)
        
        for sandbox_id in to_destroy:
            self.destroy_sandbox(sandbox_id)
        
        if to_destroy:
            self.logger.info(
                f"Cleaned up {len(to_destroy)} old mock sandboxes",
                extra={"count": len(to_destroy)}
            )
        
        return len(to_destroy)
    
    def get_sandbox_info(self, sandbox_id: str) -> Dict:
        """
        Get information about a mock sandbox.
        
        Args:
            sandbox_id: ID of the sandbox.
        
        Returns:
            Dictionary with sandbox metadata.
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        
        return {
            "sandbox_id": sandbox_id,
            "workspace": sandbox.workspace_path,
            "age": time.time() - sandbox.created_at,
            "status": "mock-running",
            "created_at": sandbox.created_at,
            "mode": "mock"
        }
    
    def _validate_sandbox(self, sandbox_id: str) -> None:
        """
        Validate that mock sandbox exists.
        
        Raises:
            ValueError: If sandbox doesn't exist.
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Mock sandbox not found: {sandbox_id}")
    
    def _mock_exec_run(self, cmd: List[str], workspace_path: Path, **kwargs) -> 'MockExecResult':
        """
        Mock exec_run implementation for compatibility.
        
        Returns a mock execution result object.
        """
        class MockExecResult:
            def __init__(self, exit_code: int, stdout: str, stderr: str):
                self.exit_code = exit_code
                self.output = (stdout.encode() if stdout else b"", stderr.encode() if stderr else b"")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PYTHONPATH": str(workspace_path)}
            )
            return MockExecResult(result.returncode, result.stdout, result.stderr)
        except Exception as e:
            return MockExecResult(-1, "", str(e))
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all mock sandboxes."""
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.destroy_sandbox(sandbox_id)
        
        # Cleanup temp directory
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)