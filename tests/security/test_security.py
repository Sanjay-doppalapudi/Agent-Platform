"""
Security validation tests.

Tests for security vulnerabilities, sandbox escapes, and input validation.
"""

import pytest

from agent_platform.tools.file_tools import FileReadTool, FileWriteTool


@pytest.mark.security
class TestSandboxSecurity:
    """Security tests for sandbox isolation."""
    
    def test_path_traversal_prevention(self, temp_workspace):
        """Test that path traversal attacks are blocked."""
        tool = FileReadTool(str(temp_workspace))
        
        # Attempt path traversal
        malicious_paths = [
            "../../../etc/passwd",
            "../../.ssh/id_rsa",
        ]
        
        for malicious_path in malicious_paths:
            result = tool.run(file_path=malicious_path)
            
            # Should fail due to security error
            assert not result.success, f"Path traversal should be blocked: {malicious_path}"
            assert "workspace" in result.error.lower() or "security" in result.error.lower()


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_large_input_handling(self, temp_workspace):
        """Test handling of excessively large inputs."""
        tool = FileWriteTool(str(temp_workspace))
        
        # Attempt to write large content
        large_content = "A" * (10 * 1024 * 1024)  # 10MB
        
        result = tool.run(file_path="large.txt", content=large_content)
        
        # Should either succeed or fail gracefully (no crash)
        assert isinstance(result.success, bool)
