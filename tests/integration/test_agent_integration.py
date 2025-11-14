"""
Integration tests for Agent with real LLM and tools.

These tests use real API calls and file operations to verify
end-to-end functionality. They are skipped if API keys are not available.
"""

import os
from pathlib import Path

import pytest

from agent_platform.agent_core.factory import create_coding_agent, create_agent
from agent_platform.tools.file_tools import FileReadTool, FileWriteTool, FileListTool

# Check for API keys
HAS_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").startswith("sk-")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
@pytest.mark.timeout(60)
class TestRealFileOperations:
    """Integration tests with real file operations."""
    
    def test_create_and_read_file(self, temp_workspace):
        """
        Test agent can create a file and read it back.
        
        This tests the complete flow of:
        1. Agent understanding the task
        2. Calling write_file tool
        3. Calling read_file tool
        4. Synthesizing results
        """
        # Create agent with file tools
        tools = [
            FileWriteTool(str(temp_workspace)),
            FileReadTool(str(temp_workspace)),
        ]
        agent = create_agent("gpt-3.5-turbo", tools=tools, max_iterations=10)
        
        # Ask agent to create and verify file
        result = agent.run(
            "Create a file called 'hello.txt' with the content 'Hello, World!' "
            "and then read it back to confirm it was created correctly."
        )
        
        # Verify file was created
        test_file = temp_workspace / "hello.txt"
        assert test_file.exists(), "File should be created"
        assert test_file.read_text() == "Hello, World!", "Content should match"
        
        # Verify agent confirmed the operation
        assert "Hello, World!" in result or "confirmed" in result.lower()
    
    def test_list_files_in_directory(self, temp_workspace):
        """Test agent can list files in a directory."""
        # Create some test files
        (temp_workspace / "file1.txt").write_text("Content 1")
        (temp_workspace / "file2.txt").write_text("Content 2")
        (temp_workspace / "file3.py").write_text("print('hello')")
        
        tools = [FileListTool(str(temp_workspace))]
        agent = create_agent("gpt-3.5-turbo", tools=tools)
        
        result = agent.run("List all .txt files in the current directory")
        
        # Should mention the txt files
        assert "file1.txt" in result or "file2.txt" in result


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
@pytest.mark.timeout(90)
class TestRealMultiStepTask:
    """Integration tests for complex multi-step tasks."""
    
    def test_create_modify_read_file(self, temp_workspace):
        """
        Test agent can perform multi-step file operations.
        
        Steps:
        1. Create a Python file
        2. Modify it to add more content
        3. Read and verify the final content
        """
        tools = [
            FileWriteTool(str(temp_workspace)),
            FileReadTool(str(temp_workspace)),
        ]
        agent = create_agent("gpt-3.5-turbo", tools=tools, max_iterations=15, verbose=True)
        
        result = agent.run(
            "Create a Python file called 'greet.py' with a function that prints 'Hello'. "
            "Then read the file to verify it was created."
        )
        
        # Verify file exists
        greet_file = temp_workspace / "greet.py"
        assert greet_file.exists()
        
        content = greet_file.read_text()
        # Should contain a function and print/hello
        assert "def" in content or "print" in content.lower()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
@pytest.mark.timeout(60)
class TestRealErrorRecovery:
    """Integration tests for error handling and recovery."""
    
    def test_handle_nonexistent_file(self, temp_workspace):
        """Test agent handles attempting to read non-existent file."""
        tools = [FileReadTool(str(temp_workspace))]
        agent = create_agent("gpt-3.5-turbo", tools=tools)
        
        result = agent.run("Read the file 'nonexistent.txt'")
        
        # Agent should report file not found
        assert "not found" in result.lower() or "does not exist" in result.lower()
    
    def test_continue_after_tool_failure(self, temp_workspace):
        """Test agent can continue after a tool fails."""
        tools = [
            FileReadTool(str(temp_workspace)),
            FileWriteTool(str(temp_workspace)),
        ]
        agent = create_agent("gpt-3.5-turbo", tools=tools, max_iterations=10)
        
        # First part will fail (file doesn't exist), second part should succeed
        result = agent.run(
            "Try to read 'missing.txt', and if it doesn't exist, "
            "create a new file called 'recovery.txt' with content 'Recovered!'"
        )
        
        # Should have created the recovery file
        recovery_file = temp_workspace / "recovery.txt"
        assert recovery_file.exists(), "Agent should have created recovery file"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
@pytest.mark.timeout(120)
class TestRealComplexWorkflow:
    """Integration tests for complex multi-tool workflows."""
    
    def test_analyze_and_summarize_files(self, temp_workspace):
        """Test agent can analyze multiple files and provide summary."""
        # Create test files
        (temp_workspace / "data1.txt").write_text("Temperature: 20C, Humidity: 65%")
        (temp_workspace / "data2.txt").write_text("Temperature: 22C, Humidity: 60%")
        (temp_workspace / "data3.txt").write_text("Temperature: 19C, Humidity: 70%")
        
        tools = [
            FileReadTool(str(temp_workspace)),
            FileListTool(str(temp_workspace)),
            FileWriteTool(str(temp_workspace)),
        ]
        agent = create_agent("gpt-3.5-turbo", tools=tools, max_iterations=20)
        
        result = agent.run(
            "Find all .txt files that start with 'data', read their contents, "
            "and create a summary file called 'summary.txt' with the average temperature."
        )
        
        # Verify summary was created
        summary_file = temp_workspace / "summary.txt"
        assert summary_file.exists(), "Summary file should be created"
        
        # Summary should mention temperature
        summary_content = summary_file.read_text()
        assert "temperature" in summary_content.lower() or "20" in summary_content


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
class TestConversationState:
    """Integration tests for conversation state tracking."""
    
    def test_state_tracks_all_interactions(self, temp_workspace):
        """Test that conversation state properly tracks all interactions."""
        tools = [FileWriteTool(str(temp_workspace))]
        agent = create_agent("gpt-3.5-turbo", tools=tools)
        
        agent.run("Create a file called 'test.txt' with content 'test'")
        
        # Get conversation state
        state = agent.get_state()
        messages = state.get_messages()
        
        # Should have multiple messages: system, user, assistant, tool results
        assert len(messages) >= 3
        assert any(m.role == "system" for m in messages)
        assert any(m.role == "user" for m in messages)
        assert any(m.role == "assistant" for m in messages)
        
        # Should have tool calls tracked
        assert len(state.tool_calls) >= 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
class TestCodingAgent:
    """Integration tests for the coding agent preset."""
    
    def test_coding_agent_creation(self, temp_workspace):
        """Test that coding agent is properly configured."""
        agent = create_coding_agent(str(temp_workspace), model="gpt-3.5-turbo")
        
        # Should have file tools
        assert agent.registry.get_tool_count() >= 3
        
        # Should be able to handle coding tasks
        result = agent.run("Create a Python file 'hello.py' that prints 'Hello, World!'")
        
        hello_file = temp_workspace / "hello.py"
        assert hello_file.exists()
