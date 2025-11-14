"""
Unit tests for the Agent orchestrator.

These tests verify the agent's decision-making loop, tool execution,
and error handling using mocks.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_platform.agent_core.agent import Agent, AgentConfig
from agent_platform.llm.adapter import LLMConfig, UniversalLLMAdapter
from agent_platform.tools.base import BaseTool, ToolInput, ToolResult
from agent_platform.tools.registry import ToolRegistry


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    
    class MockTool(BaseTool):
        name = "mock_tool"
        description = "A mock tool for testing"
        input_schema = ToolInput
        
        def _run(self, **kwargs):
            return ToolResult(success=True, output="Mock result")
    
    return MockTool()


@pytest.fixture
def agent_with_mock_tool(mock_openai_provider, mock_tool):
    """Create an agent with a mocked LLM and a mock tool."""
    registry = ToolRegistry()
    registry.register(mock_tool)
    
    config = AgentConfig(max_iterations=5, verbose=False)
    agent = Agent(mock_openai_provider, registry, config)
    
    return agent


@pytest.mark.unit
class TestAgentInitialization:
    """Test cases for agent initialization."""
    
    def test_agent_initialization_default_config(self, mock_openai_provider):
        """Test agent initialization with default configuration."""
        registry = ToolRegistry()
        agent = Agent(mock_openai_provider, registry)
        
        assert agent.llm == mock_openai_provider
        assert agent.registry == registry
        assert agent.config.max_iterations == 10
        assert agent.config.max_execution_time == 300
        assert agent.state is not None
    
    def test_agent_initialization_custom_config(self, mock_openai_provider):
        """Test agent initialization with custom configuration."""
        registry = ToolRegistry()
        config = AgentConfig(
            max_iterations=20,
            max_execution_time=600,
            verbose=True,
            system_message="Custom system message"
        )
        agent = Agent(mock_openai_provider, registry, config)
        
        assert agent.config.max_iterations == 20
        assert agent.config.max_execution_time == 600
        assert agent.config.verbose is True
    
    def test_agent_initialization_with_system_message(self, mock_openai_provider):
        """Test that custom system message is added to state."""
        registry = ToolRegistry()
        config = AgentConfig(system_message="You are a test assistant")
        agent = Agent(mock_openai_provider, registry, config)
        
        messages = agent.state.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert "test assistant" in messages[0].content.lower()


@pytest.mark.unit
class TestSimpleTaskNoTools:
    """Test cases for simple tasks without tool usage."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    def test_simple_question_direct_answer(self, mock_completion, agent_with_mock_tool):
        """Test agent handles simple question without tools."""
        # Mock LLM to return direct answer
        mock_completion.return_value = {
            "content": "Final answer: Paris is the capital of France.",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }
        
        result = agent_with_mock_tool.run("What is the capital of France?")
        
        assert "Paris" in result
        assert mock_completion.call_count >= 1


@pytest.mark.unit
class TestTaskWithSingleToolCall:
    """Test cases for tasks requiring a single tool call."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    def test_single_tool_call_flow(self, mock_completion, agent_with_mock_tool):
        """Test complete flow with one tool call."""
        # Mock LLM responses
        responses = [
            # First: request tool call
            {
                "content": "I'll use mock_tool with test='value'",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            },
            # Second: final answer after tool result
            {
                "content": "Final answer: The tool returned 'Mock result'",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            }
        ]
        mock_completion.side_effect = responses
        
        result = agent_with_mock_tool.run("Use the mock tool")
        
        assert "Mock result" in result or "answer" in result.lower()
        assert mock_completion.call_count == 2


@pytest.mark.unit
class TestTaskWithMultipleToolCalls:
    """Test cases for tasks requiring multiple tool calls."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    def test_multiple_tool_calls_sequence(self, mock_completion, agent_with_mock_tool):
        """Test agent can execute multiple tool calls in sequence."""
        # Mock LLM responses for multiple tool calls
        responses = [
            {"content": "I'll use mock_tool with arg1='first'", "model": "gpt-4",
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            {"content": "I'll use mock_tool with arg2='second'", "model": "gpt-4",
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            {"content": "Final answer: Completed both tool calls", "model": "gpt-4",
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        ]
        mock_completion.side_effect = responses
        
        result = agent_with_mock_tool.run("Execute mock_tool twice")
        
        assert "Completed" in result or "answer" in result.lower()
        assert mock_completion.call_count == 3


@pytest.mark.unit
class TestInvalidToolCall:
    """Test cases for handling invalid tool calls."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    def test_nonexistent_tool_handling(self, mock_completion, agent_with_mock_tool):
        """Test agent handles request for non-existent tool."""
        responses = [
            {"content": "I'll use nonexistent_tool with arg='value'", "model": "gpt-4",
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            {"content": "Final answer: I apologize, that tool doesn't exist", "model": "gpt-4",
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        ]
        mock_completion.side_effect = responses
        
        result = agent_with_mock_tool.run("Use a tool that doesn't exist")
        
        # Should complete without crashing
        assert isinstance(result, str)


@pytest.mark.unit
class TestToolExecutionFailure:
    """Test cases for handling tool execution failures."""
    
    def test_tool_failure_recovery(self, mock_openai_provider):
        """Test agent handles tool execution failures gracefully."""
        
        class FailingTool(BaseTool):
            name = "failing_tool"
            description = "A tool that fails"
            input_schema = ToolInput
            
            def _run(self, **kwargs):
                return ToolResult(success=False, output="", error="Tool failed!")
        
        registry = ToolRegistry()
        registry.register(FailingTool())
        agent = Agent(mock_openai_provider, registry)
        
        with patch.object(agent.llm, 'completion') as mock_completion:
            mock_completion.side_effect = [
                {"content": "I'll use failing_tool", "model": "gpt-4",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
                {"content": "Final answer: The tool failed but I handled it", "model": "gpt-4",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            ]
            
            result = agent.run("Use the failing tool")
            
            # Should complete despite tool failure
            assert isinstance(result, str)


@pytest.mark.unit
class TestMaxIterationsExceeded:
    """Test cases for iteration limit handling."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    def test_max_iterations_limit(self, mock_completion, agent_with_mock_tool):
        """Test agent stops at max iterations."""
        # Make LLM always request a tool (infinite loop scenario)
        mock_completion.return_value = {
            "content": "I'll use mock_tool again",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }
        
        # Set low iteration limit
        agent_with_mock_tool.config.max_iterations = 3
        
        result = agent_with_mock_tool.run("Keep using tools forever")
        
        assert "incomplete" in result.lower() or "maximum" in result.lower()
        # Should have called LLM exactly max_iterations times
        assert mock_completion.call_count == 3


@pytest.mark.unit
class TestMaxExecutionTimeExceeded:
    """Test cases for execution time limit handling."""
    
    @patch('agent_platform.llm.adapter.UniversalLLMAdapter.completion')
    @patch('time.time')
    def test_execution_timeout(self, mock_time, mock_completion, agent_with_mock_tool):
        """Test agent stops when execution time limit reached."""
        # Simulate time passing
        start_time = 1000.0
        mock_time.side_effect = [
            start_time,  # Initial time
            start_time + 1,  # After first iteration
            start_time + 400,  # Exceeds 300s limit
        ]
        
        mock_completion.return_value = {
            "content": "I'll use mock_tool",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }
        
        agent_with_mock_tool.config.max_execution_time = 300
        
        result = agent_with_mock_tool.run("Task that takes too long")
        
        assert "incomplete" in result.lower() or "maximum" in result.lower()


@pytest.mark.unit
class TestStateManagement:
    """Test cases for conversation state management."""
    
    def test_messages_added_to_state(self, agent_with_mock_tool):
        """Test that messages are properly added to state."""
        with patch.object(agent_with_mock_tool.llm, 'completion') as mock_completion:
            mock_completion.return_value = {
                "content": "Final answer: Done",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            }
            
            agent_with_mock_tool.run("Test task")
            
            messages = agent_with_mock_tool.state.get_messages()
            
            # Should have: system, user, assistant
            assert len(messages) >= 3
            assert messages[0].role == "system"
            assert any(m.role == "user" and "Test task" in m.content for m in messages)
    
    def test_tool_calls_tracked(self, agent_with_mock_tool):
        """Test that tool calls are tracked in state."""
        with patch.object(agent_with_mock_tool.llm, 'completion') as mock_completion:
            mock_completion.side_effect = [
                {"content": "I'll use mock_tool", "model": "gpt-4",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
                {"content": "Final answer: Done", "model": "gpt-4",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            ]
            
            agent_with_mock_tool.run("Use the tool")
            
            # Check tool calls were tracked
            assert len(agent_with_mock_tool.state.tool_calls) >= 1
    
    def test_state_reset(self, agent_with_mock_tool):
        """Test that reset clears state properly."""
        with patch.object(agent_with_mock_tool.llm, 'completion') as mock_completion:
            mock_completion.return_value = {
                "content": "Final answer: Done",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            }
            
            agent_with_mock_tool.run("First task")
            agent_with_mock_tool.reset()
            
            messages = agent_with_mock_tool.state.get_messages()
            
            # Should only have system message after reset
            assert len(messages) == 1
            assert messages[0].role == "system"


@pytest.mark.unit
class TestResponseParsing:
    """Test cases for LLM response parsing."""
    
    def test_parse_tool_call_explicit(self, agent_with_mock_tool):
        """Test parsing explicit tool call format."""
        response = "I'll use mock_tool with arg1='value1', arg2='value2'"
        parsed = agent_with_mock_tool._parse_llm_response(response)
        
        assert parsed["type"] == "tool_call"
        assert parsed["tool_name"] == "mock_tool"
        assert "arg1" in parsed["tool_args"]
    
    def test_parse_final_answer(self, agent_with_mock_tool):
        """Test parsing final answer format."""
        response = "Final answer: This is my response to the user."
        parsed = agent_with_mock_tool._parse_llm_response(response)
        
        assert parsed["type"] == "final_answer"
        assert "response" in parsed["answer"]
    
    def test_parse_ambiguous_response(self, agent_with_mock_tool):
        """Test handling of ambiguous responses."""
        response = "I'm thinking about this..."
        parsed = agent_with_mock_tool._parse_llm_response(response)
        
        # Should be classified as ambiguous or final answer
        assert parsed["type"] in ["ambiguous", "final_answer"]
