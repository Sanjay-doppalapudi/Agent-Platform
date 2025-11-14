"""
Sandbox manager for orchestrating secure container-based code execution.

This module manages Docker containers for executing untrusted code with
security constraints including resource limits, network isolation, and
syscall filtering.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox containers."""
    
    image: str = "agent-sandbox:latest"
    memory_limit: str = "512m"
    cpu_quota: int = 100000  # 1 CPU = 100000 (100% of one core)
    timeout: int = 30  # seconds
    network_enabled: bool = False
    enable_gvisor: bool = False
    seccomp_profile: Optional[str] = None
    
    def __post_init__(self):
        """Set default seccomp profile path if not provided."""
        if self.seccomp_profile is None:
            # Default to the profile in the docker directory
            profile_path = Path(__file__).parent.parent.parent.parent / "docker" / "seccomp-profile.json"
            if profile_path.exists():
                self.seccomp_profile = str(profile_path)


@dataclass
class Sandbox:
    """Represents an active sandbox container."""
    
    container_id: str
    container: docker.models.containers.Container
    created_at: float
    workspace_path: str
    config: SandboxConfig = field(default_factory=SandboxConfig)


class SandboxManager:
    """
    Manages lifecycle of secure sandbox containers.
    
    Provides methods for creating, executing code in, and destroying
    isolated container environments for untrusted code execution.
    
    Example:
        >>> manager = SandboxManager()
        >>> sandbox_id = manager.create_sandbox("/workspace")
        >>> result = manager.execute_code(sandbox_id, "print('Hello')")
        >>> print(result["stdout"])  # "Hello\\n"
        >>> manager.destroy_sandbox(sandbox_id)
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox manager.
        
        Args:
            config: Sandbox configuration (uses defaults if not provided).
        """
        self.config = config or SandboxConfig()
        self.client = docker.from_env()
        self.active_sandboxes: Dict[str, Sandbox] = {}
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initialized SandboxManager",
            extra={"image": self.config.image, "memory_limit": self.config.memory_limit}
        )
    
    def create_sandbox(self, workspace_path: str) -> str:
        """
        Create a new sandbox container.
        
        Args:
            workspace_path: Path to mount as /workspace in the container.
        
        Returns:
            Sandbox ID (container ID).
        
        Raises:
            DockerException: If container creation fails.
        
        Example:
            >>> manager = SandboxManager()
            >>> sandbox_id = manager.create_sandbox("/tmp/myworkspace")
        """
        try:
            # Ensure image exists
            self._ensure_image_exists()
            
            # Prepare container configuration
            container_config = self._build_container_config(workspace_path)
            
            # Create container
            self.logger.info(f"Creating sandbox container with workspace: {workspace_path}")
            container = self.client.containers.create(**container_config)
            
            # Start container
            container.start()
            
            # Wait for container to be ready
            time.sleep(0.5)
            
            # Store sandbox
            sandbox = Sandbox(
                container_id=container.id,
                container=container,
                created_at=time.time(),
                workspace_path=workspace_path,
                config=self.config
            )
            self.active_sandboxes[container.id] = sandbox
            
            self.logger.info(
                f"Created sandbox",
                extra={"sandbox_id": container.id[:12], "workspace": workspace_path}
            )
            
            return container.id
            
        except DockerException as e:
            self.logger.error(f"Failed to create sandbox: {e}", exc_info=True)
            raise
    
    def execute_code(
        self,
        sandbox_id: str,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Execute code in a sandbox.
        
        Args:
            sandbox_id: ID of the sandbox container.
            code: Code to execute.
            language: Programming language (currently only "python" supported).
            timeout: Execution timeout in seconds (uses config default if None).
        
        Returns:
            Dictionary with:
            - exit_code: int
            - stdout: str
            - stderr: str
            - success: bool
            - execution_time: float
        
        Example:
            >>> result = manager.execute_code(
            ...     sandbox_id,
            ...     "print('Hello, World!')"
            ... )
            >>> print(result["stdout"])  # "Hello, World!\\n"
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        timeout = timeout or self.config.timeout
        
        # Prepare command based on language
        if language == "python":
            cmd = ["python", "-c", code]
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        start_time = time.time()
        
        try:
            self.logger.debug(
                f"Executing code in sandbox",
                extra={"sandbox_id": sandbox_id[:12], "language": language}
            )
            
            # Execute command
            exec_result = sandbox.container.exec_run(
                cmd,
                demux=True,  # Separate stdout and stderr
                user="sandbox",
                workdir="/workspace"
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            exit_code = exec_result.exit_code
            stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ""
            
            success = exit_code == 0
            
            self.logger.info(
                f"Code execution completed",
                extra={
                    "sandbox_id": sandbox_id[:12],
                    "success": success,
                    "execution_time": execution_time
                }
            )
            
            return {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": success,
                "execution_time": execution_time
            }
            
        except Exception as e:
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
                "execution_time": time.time() - start_time
            }
    
    def execute_file(
        self,
        sandbox_id: str,
        file_path: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Execute a file in the sandbox.
        
        Args:
            sandbox_id: ID of the sandbox container.
            file_path: Path to file relative to workspace.
            args: Optional command-line arguments.
            timeout: Execution timeout in seconds.
        
        Returns:
            Execution result dictionary (same format as execute_code).
        
        Example:
            >>> result = manager.execute_file(
            ...     sandbox_id,
            ...     "script.py",
            ...     args=["--verbose"]
            ... )
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        timeout = timeout or self.config.timeout
        
        # Build command
        cmd = ["python", f"/workspace/{file_path}"]
        if args:
            cmd.extend(args)
        
        start_time = time.time()
        
        try:
            exec_result = sandbox.container.exec_run(
                cmd,
                demux=True,
                user="sandbox",
                workdir="/workspace"
            )
            
            execution_time = time.time() - start_time
            
            exit_code = exec_result.exit_code
            stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ""
            
            return {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": exit_code == 0,
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "execution_time": time.time() - start_time
            }
    
    def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        Destroy a sandbox container.
        
        Args:
            sandbox_id: ID of the sandbox to destroy.
        
        Example:
            >>> manager.destroy_sandbox(sandbox_id)
        """
        if sandbox_id not in self.active_sandboxes:
            self.logger.warning(f"Sandbox not found: {sandbox_id[:12]}")
            return
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        try:
            # Stop and remove container
            sandbox.container.stop(timeout=5)
            sandbox.container.remove()
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            self.logger.info(f"Destroyed sandbox", extra={"sandbox_id": sandbox_id[:12]})
            
        except Exception as e:
            self.logger.error(
                f"Failed to destroy sandbox: {e}",
                extra={"sandbox_id": sandbox_id[:12]},
                exc_info=True
            )
    
    def cleanup_old_sandboxes(self, max_age: int = 3600) -> int:
        """
        Clean up sandboxes older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds before cleanup.
        
        Returns:
            Number of sandboxes cleaned up.
        
        Example:
            >>> # Cleanup sandboxes older than 1 hour
            >>> count = manager.cleanup_old_sandboxes(3600)
            >>> print(f"Cleaned up {count} sandboxes")
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
                f"Cleaned up {len(to_destroy)} old sandboxes",
                extra={"count": len(to_destroy)}
            )
        
        return len(to_destroy)
    
    def get_sandbox_info(self, sandbox_id: str) -> Dict:
        """
        Get information about a sandbox.
        
        Args:
            sandbox_id: ID of the sandbox.
        
        Returns:
            Dictionary with sandbox metadata.
        
        Example:
            >>> info = manager.get_sandbox_info(sandbox_id)
            >>> print(info["age"])
        """
        self._validate_sandbox(sandbox_id)
        sandbox = self.active_sandboxes[sandbox_id]
        
        # Reload container stats
        sandbox.container.reload()
        
        return {
            "sandbox_id": sandbox_id,
            "workspace": sandbox.workspace_path,
            "age": time.time() - sandbox.created_at,
            "status": sandbox.container.status,
            "created_at": sandbox.created_at
        }
    
    def _validate_sandbox(self, sandbox_id: str) -> None:
        """
        Validate that sandbox exists and is running.
        
        Raises:
            ValueError: If sandbox doesn't exist or isn't running.
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox not found: {sandbox_id}")
        
        sandbox = self.active_sandboxes[sandbox_id]
        sandbox.container.reload()
        
        if sandbox.container.status != "running":
            raise ValueError(f"Sandbox is not running: {sandbox_id}")
    
    def _ensure_image_exists(self) -> None:
        """Pull sandbox image if it doesn't exist."""
        try:
            self.client.images.get(self.config.image)
        except ImageNotFound:
            self.logger.info(f"Pulling image: {self.config.image}")
            self.client.images.pull(self.config.image)
    
    def _build_container_config(self, workspace_path: str) -> Dict:
        """Build container configuration with security settings."""
        config = {
            "image": self.config.image,
            "detach": True,
            "mem_limit": self.config.memory_limit,
            "cpu_quota": self.config.cpu_quota,
            "cpu_period": 100000,
            "network_mode": "none" if not self.config.network_enabled else "bridge",
            "read_only": True,
            "tmpfs": {
                "/tmp": "rw,noexec,nosuid,size=100m"
            },
            "volumes": {
                workspace_path: {"bind": "/workspace", "mode": "rw"}
            },
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
        }
        
        # Add seccomp profile if available
        if self.config.seccomp_profile and Path(self.config.seccomp_profile).exists():
            with open(self.config.seccomp_profile) as f:
                seccomp_data = json.load(f)
            config["security_opt"].append(f"seccomp={json.dumps(seccomp_data)}")
        
        # Add gVisor runtime if enabled
        if self.config.enable_gvisor:
            config["runtime"] = "runsc"
        
        return config
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all sandboxes."""
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.destroy_sandbox(sandbox_id)
