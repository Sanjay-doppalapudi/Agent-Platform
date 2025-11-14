"""
Pytest configuration and shared fixtures for the Agent Platform test suite.

This module provides reusable fixtures for testing LLM adapters, tools,
and other components. It includes mocks, sample data, and test utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import SecretStr

from agent_platform.config import Settings
from agent_platform.llm.adapter import LLMConfig, UniversalLLMAdapter

# Configure pytest plugins
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """
    Fixture providing a mock LiteLLM completion response.

    Returns a dictionary that mimics the structure of a real LiteLLM response
    with configurable content, model, and usage statistics.

    Returns:
        Dictionary with mock response structure including choices, model, and usage.

    Example:
        >>> def test_something(mock_llm_response):
        ...     assert mock_llm_response["choices"][0]["message"]["content"] == "Hello, World!"
    """

    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockChoice:
        def __init__(self, message: MockMessage):
            self.message = message
            self.finish_reason = "stop"

    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 10
            self.completion_tokens = 20
            self.total_tokens = 30

    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice(MockMessage("Hello, World!"))]
            self.model = "gpt-4"
            self.usage = MockUsage()

    return MockResponse()


@pytest.fixture
def mock_llm_stream_response() -> Iterator[Any]:
    """
    Fixture providing a mock streaming LiteLLM response.

    Returns an iterator of mock streaming chunks that simulate
    a real streaming completion response.

    Yields:
        Mock streaming chunks with delta content.

    Example:
        >>> def test_streaming(mock_llm_stream_response):
        ...     chunks = list(mock_llm_stream_response)
        ...     assert len(chunks) > 0
    """

    class MockDelta:
        def __init__(self, content: str):
            self.content = content

    class MockChoice:
        def __init__(self, delta: MockDelta):
            self.delta = delta

    class MockChunk:
        def __init__(self, content: str):
            self.choices = [MockChoice(MockDelta(content))]

    # Split response into chunks
    content = "Hello, World!"
    chunks = [content[i : i + 5] for i in range(0, len(content), 5)]

    return iter([MockChunk(chunk) for chunk in chunks])


@pytest.fixture
def mock_openai_provider(monkeypatch) -> UniversalLLMAdapter:
    """
    Fixture providing a mocked UniversalLLMAdapter with no real API calls.

    This fixture patches litellm.completion to return mock responses,
    allowing tests to run without actual API credentials or network calls.

    Args:
        monkeypatch: Pytest monkeypatch fixture for patching.

    Returns:
        UniversalLLMAdapter instance with mocked API calls.

    Example:
        >>> def test_adapter(mock_openai_provider):
        ...     messages = [{"role": "user", "content": "Hi"}]
        ...     response = mock_openai_provider.completion(messages)
        ...     assert response["content"] == "Hello, World!"
    """
    # Mock litellm.completion
    def mock_completion(*args, **kwargs):
        class MockMessage:
            content = "Hello, World!"

        class MockChoice:
            message = MockMessage()
            finish_reason = "stop"

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 20
            total_tokens = 30

        class MockResponse:
            choices = [MockChoice()]
            model = "gpt-4"
            usage = MockUsage()

        return MockResponse()

    # Mock litellm.token_counter
    def mock_token_counter(*args, **kwargs):
        return len(kwargs.get("text", "")) // 4

    monkeypatch.setattr("litellm.completion", mock_completion)
    monkeypatch.setattr("litellm.token_counter", mock_token_counter)

    # Create adapter with test configuration
    config = LLMConfig(
        model="gpt-4",
        api_key=SecretStr("sk-test-key"),
        temperature=0.7,
        max_tokens=1000,
        timeout=30,
    )

    return UniversalLLMAdapter(config)


@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """
    Fixture providing sample message dictionaries for testing.

    Returns a list of messages including system, user, and assistant messages
    that can be used for testing conversation flows.

    Returns:
        List of message dictionaries with role and content.

    Example:
        >>> def test_messages(sample_messages):
        ...     assert len(sample_messages) == 3
        ...     assert sample_messages[0]["role"] == "system"
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specialized in coding tasks.",
        },
        {"role": "user", "content": "What is Python?"},
        {
            "role": "assistant",
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        },
    ]


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Iterator[Path]:
    """
    Fixture providing a temporary workspace directory for file operation tests.

    Creates a temporary directory that is automatically cleaned up after the test.
    Useful for testing file tools and other operations that need a workspace.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture.

    Yields:
        Path to the temporary workspace directory.

    Example:
        >>> def test_file_operations(temp_workspace):
        ...     test_file = temp_workspace / "test.txt"
        ...     test_file.write_text("Hello")
        ...     assert test_file.exists()
    """
    # Create a workspace structure
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Create some sample directories
    (workspace / "src").mkdir(exist_ok=True)
    (workspace / "tests").mkdir(exist_ok=True)
    (workspace / "docs").mkdir(exist_ok=True)

    # Create a sample file
    sample_file = workspace / "README.md"
    sample_file.write_text("# Test Project\n\nThis is a test workspace.")

    yield workspace

    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_settings(monkeypatch) -> Settings:
    """
    Fixture providing a Settings instance with test values.

    Creates a Settings object with test configuration that doesn't
    require real environment variables or .env files.

    Args:
        monkeypatch: Pytest monkeypatch fixture for patching.

    Returns:
        Settings instance configured for testing.

    Example:
        >>> def test_config(mock_settings):
        ...     assert mock_settings.environment == "development"
        ...     assert mock_settings.debug is True
    """
    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "development",
        "LOG_LEVEL": "DEBUG",
        "DEBUG": "true",
        "API_HOST": "localhost",
        "API_PORT": "8001",
        "OPENAI_API_KEY": "sk-test-openai-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-anthropic-key",
        "SANDBOX_IMAGE": "test-sandbox:latest",
        "SANDBOX_TIMEOUT": "10",
        "SANDBOX_MEMORY_LIMIT": "256m",
        "SANDBOX_CPU_LIMIT": "0.5",
        "REDIS_URL": "redis://localhost:6379/1",
        "SECRET_KEY": "test-secret-key-at-least-32-chars-long-for-testing",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    # Clear the lru_cache on get_settings to force reload
    from agent_platform.config import get_settings

    get_settings.cache_clear()

    return Settings()


@pytest.fixture
def mock_cost_tracker():
    """
    Fixture providing a mock CostTracker for testing.

    Returns:
        Mock CostTracker instance with tracking methods.

    Example:
        >>> def test_tracking(mock_cost_tracker):
        ...     mock_cost_tracker.track_usage("gpt-4", 100, 50)
        ...     assert mock_cost_tracker.get_total_cost() > 0
    """
    from agent_platform.llm.cost_tracker import CostTracker

    return CostTracker()


@pytest.fixture
def mock_logger(monkeypatch):
    """
    Fixture providing a mock logger to prevent log output during tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mock logger instance.

    Example:
        >>> def test_with_logging(mock_logger):
        ...     # Logs won't appear in test output
        ...     pass
    """
    mock = MagicMock()
    monkeypatch.setattr("agent_platform.utils.logger.get_logger", lambda name: mock)
    return mock


# Pytest configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Tests that take significant time")
    config.addinivalue_line(
        "markers", "requires_api_key: Tests that require real API keys"
    )
