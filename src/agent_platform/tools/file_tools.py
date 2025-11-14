"""
Secure file operation tools for agents.

This module provides tools for reading, writing, and listing files with
security checks to prevent directory traversal and other vulnerabilities.
"""

import shutil
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from agent_platform.tools.base import BaseTool, ToolResult
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class FileReadInput(BaseModel):
    """Input schema for file reading."""
    
    file_path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class FileReadTool(BaseTool):
    """
    Tool for reading file contents securely.
    
    Prevents directory traversal attacks by ensuring paths stay
    within the designated workspace.
    """
    
    name = "read_file"
    description = "Read contents of a file. Use when you need to examine file contents or understand code."
    input_schema = FileReadInput
    category = "file_operations"
    
    def __init__(self, workspace_path: str):
        """
        Initialize with workspace directory.
        
        Args:
            workspace_path: Root directory for file operations.
        """
        self.workspace = Path(workspace_path).resolve()
        
    def _run(self, file_path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file contents."""
        try:
            # Validate and resolve path
            full_path = self._validate_path(file_path)
            
            # Check file exists
            if not full_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )
            
            # Check file size (limit to 1MB)
            if full_path.stat().st_size > 1024 * 1024:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File too large (>1MB): {file_path}"
                )
            
            # Read file
            content = full_path.read_text(encoding=encoding)
            
            return ToolResult(
                success=True,
                output=content,
                metadata={"file_size": full_path.stat().st_size}
            )
            
        except SecurityError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read file: {str(e)}")
    
    async def _arun(self, file_path: str, encoding: str = "utf-8") -> ToolResult:
        """Async file reading."""
        import aiofiles
        
        try:
            full_path = self._validate_path(file_path)
            
            if not full_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {file_path}")
            
            if full_path.stat().st_size > 1024 * 1024:
                return ToolResult(success=False, output="", error=f"File too large: {file_path}")
            
            async with aiofiles.open(full_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            return ToolResult(
                success=True,
                output=content,
                metadata={"file_size": full_path.stat().st_size}
            )
        except SecurityError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read file: {str(e)}")
    
    def _validate_path(self, file_path: str) -> Path:
        """
        Validate path is within workspace bounds.
        
        Args:
            file_path: Path to validate.
            
        Returns:
            Resolved absolute path.
            
        Raises:
            SecurityError: If path escapes workspace.
        """
        # Resolve path relative to workspace
        full_path = (self.workspace / file_path).resolve()
        
        # Check it's within workspace
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise SecurityError(f"Path escapes workspace: {file_path}")
        
        return full_path


class FileWriteInput(BaseModel):
    """Input schema for file writing."""
    
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    create_dirs: bool = Field(default=True, description="Create parent directories if needed")
    backup: bool = Field(default=True, description="Create backup of existing file")


class FileWriteTool(BaseTool):
    """Tool for writing files securely with atomic operations."""
    
    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist."
    input_schema = FileWriteInput
    category = "file_operations"
    
    def __init__(self, workspace_path: str, enable_backups: bool = True):
        """Initialize with workspace directory."""
        self.workspace = Path(workspace_path).resolve()
        self.enable_backups = enable_backups
        
        # Create backup directory
        self.backup_dir = self.workspace / ".backups"
        if enable_backups:
            self.backup_dir.mkdir(exist_ok=True)
    
    def _run(self, file_path: str, content: str, create_dirs: bool = True, backup: bool = True) -> ToolResult:
        """Write content to file."""
        try:
            full_path = self._validate_path(file_path)
            
            # Create parent directories if needed
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if full_path.exists() and backup and self.enable_backups:
                self._create_backup(full_path)
            
            # Atomic write: write to temp file, then rename
            temp_path = full_path.with_suffix(full_path.suffix + '.tmp')
            temp_path.write_text(content, encoding='utf-8')
            temp_path.replace(full_path)
            
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {file_path}",
                metadata={"bytes_written": len(content)}
            )
            
        except SecurityError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to write file: {str(e)}")
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate path is within workspace."""
        full_path = (self.workspace / file_path).resolve()
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise SecurityError(f"Path escapes workspace: {file_path}")
        return full_path
    
    def _create_backup(self, file_path: Path):
        """Create backup of existing file."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        
        # Keep only last 5 backups
        backups = sorted(self.backup_dir.glob(f"{file_path.name}.*.backup"))
        for old_backup in backups[:-5]:
            old_backup.unlink()


class FileListInput(BaseModel):
    """Input schema for listing files."""
    
    directory_path: str = Field(default=".", description="Directory to list")
    pattern: Optional[str] = Field(default=None, description="Glob pattern to filter files")
    recursive: bool = Field(default=False, description="Recursively list subdirectories")


class FileListTool(BaseTool):
    """Tool for listing files in a directory."""
    
    name = "list_files"
    description = "List files in a directory. Optionally filter by pattern."
    input_schema = FileListInput
    category = "file_operations"
    
    def __init__(self, workspace_path: str):
        """Initialize with workspace directory."""
        self.workspace = Path(workspace_path).resolve()
    
    def _run(self, directory_path: str = ".", pattern: Optional[str] = None, recursive: bool = False) -> ToolResult:
        """List files in directory."""
        try:
            full_path = self._validate_path(directory_path)
            
            if not full_path.exists():
                return ToolResult(success=False, output="", error=f"Directory not found: {directory_path}")
            
            if not full_path.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {directory_path}")
            
            # List files
            if pattern:
                if recursive:
                    files = full_path.rglob(pattern)
                else:
                    files = full_path.glob(pattern)
            else:
                if recursive:
                    files = full_path.rglob("*")
                else:
                    files = full_path.glob("*")
            
            # Format output
            file_list = []
            for f in files:
                if f.is_file():
                    rel_path = f.relative_to(self.workspace)
                    size = f.stat().st_size
                    file_list.append(f"{rel_path} ({size} bytes)")
            
            output = "\n".join(file_list) if file_list else "No files found"
            
            return ToolResult(
                success=True,
                output=output,
                metadata={"file_count": len(file_list)}
            )
            
        except SecurityError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to list files: {str(e)}")
    
    def _validate_path(self, dir_path: str) -> Path:
        """Validate path is within workspace."""
        full_path = (self.workspace / dir_path).resolve()
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise SecurityError(f"Path escapes workspace: {dir_path}")
        return full_path
