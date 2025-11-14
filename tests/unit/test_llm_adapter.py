"""
Unit tests for the UniversalLLMAdapter.

This module contains comprehensive unit tests for the LLM adapter,
testing initialization, completion, streaming, retries, and error handling.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from agent_platform.llm.adapter import LLMConfig, UniversalLLMAdapter


@pytest.mark.unit
class TestAdapterInitialization:
    """Test cases for adapter initialization and configuration."""

    def test_adapter_initialization_valid(self):
        """
        Test that adapter initializes successfully with valid configuration.

        Verifies that all configuration parameters are properly stored
        and the adapter is ready for use.
        """
        config = LLMConfig(
            model="gpt-4",
            api_key=SecretStr("sk-test-key"),
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
        )

        adapter = UniversalLLMAdapter(config)

        assert adapter.config.model == "gpt-4", "Model should be set correctly"
        assert adapter.config.temperature == 0.7, "Temperature should be 0.7"
        assert adapter.config.max_tokens == 1000, "Max tokens should be 1000"
        assert adapter._request_count == 0, "Request count should start at 0"
        assert adapter._total_cost == 0.0, "Total cost should start at 0.0"

    def test_adapter_initialization_without_api_key(self):
        """
        Test that adapter can be initialized without explicit API key.

        The adapter should work with environment variables for API keys.
        """
        config = LLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        )

        adapter = UniversalLLMAdapter(config)

        assert adapter.config.api_key is None, "API key should be None when not provided"

    def test_adapter_initialization_invalid_temperature(self):
        """
        Test that adapter rejects invalid temperature values.

        Temperature must be between 0.0 and 2.0.
        """
        with pytest.raises(ValueError, match="Temperature must be between"):
            LLMConfig(
                model="gpt-4",
                temperature=3.0,  # Invalid: too high
                max_tokens=1000,
            )

        with pytest.raises(ValueError, match="Temperature must be between"):
            LLMConfig(
                model="gpt-4",
                temperature=-0.5,  # Invalid: negative
                max_tokens=1000,
            )

    def test_adapter_initialization_invalid_timeout(self):
        """
        Test that adapter rejects invalid timeout values.

        Timeout must be at least 1 second.
        """
        with pytest.raises(ValueError, match="Timeout must be at least"):
            LLMConfig(
                model="gpt-4",
                temperature=0.7,
                max_tokens=1000,
                timeout=0,  # Invalid: zero
            )

    def test_adapter_initialization_invalid_max_tokens(self):
        """
        Test that adapter rejects invalid max_tokens values.

        Max tokens must be between 1 and 32000.
        """
        with pytest.raises(ValueError, match="max_tokens must be between"):
            LLMConfig(
                model="gpt-4",
                temperature=0.7,
                max_tokens=0,  # Invalid: zero
            )

        with pytest.raises(ValueError, match="max_tokens must be between"):
            LLMConfig(
                model="gpt-4",
                temperature=0.7,
                max_tokens=50000,  # Invalid: too high
            )


@pytest.mark.unit
class TestCompletion:
    """Test cases for completion functionality."""

    def test_completion_success(self, mock_openai_provider, sample_messages):
        """
        Test successful completion request.

        Verifies that the adapter correctly processes a completion request
        and returns a normalized response with content, model, and usage.
        """
        response = mock_openai_provider.completion(sample_messages)

        assert "content" in response, "Response should contain content"
        assert "model" in response, "Response should contain model"
        assert "usage" in response, "Response should contain usage stats"
        assert response["content"] == "Hello, World!", "Content should match mock response"
        assert response["model"] == "gpt-4", "Model should be gpt-4"
        assert response["usage"]["total_tokens"] == 30, "Total tokens should be 30"

    def test_completion_validates_messages(self, mock_openai_provider):
        """
        Test that completion validates message format.

        Invalid messages should raise ValueError.
        """
        # Test with empty messages
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            mock_openai_provider.completion([])

        # Test with invalid role
        with pytest.raises(ValueError, match="invalid role"):
            mock_openai_provider.completion(
                [{"role": "invalid_role", "content": "test"}]
            )

        # Test with missing content
        with pytest.raises(ValueError, match="missing 'content' key"):
            mock_openai_provider.completion([{"role": "user"}])

    def test_completion_with_kwargs_override(self, mock_openai_provider, sample_messages):
        """
        Test that kwargs can override default configuration.

        The adapter should allow per-request overrides of temperature,
        max_tokens, and other parameters.
        """
        # This should not raise an error
        response = mock_openai_provider.completion(
            sample_messages, temperature=0.5, max_tokens=500
        )

        assert response is not None, "Response should be returned"

    @patch("litellm.completion")
    def test_completion_with_retry(self, mock_completion, sample_messages):
        """
        Test retry logic on transient failures.

        The adapter should retry failed requests with exponential backoff.
        """
        # Configure mock to fail twice, then succeed
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Rate limit exceeded (429)")
            # Success on third attempt
            return self._create_mock_response()

        mock_completion.side_effect = side_effect

        # Create adapter with retry configuration
        config = LLMConfig(
            model="gpt-4",
            retry_attempts=3,
            retry_delay=0.1,  # Short delay for testing
        )
        adapter = UniversalLLMAdapter(config)

        # Patch token_counter
        with patch("litellm.token_counter", return_value=10):
            start_time = time.time()
            response = adapter.completion(sample_messages)
            elapsed_time = time.time() - start_time

        assert call_count == 3, "Should have made 3 attempts (2 failures + 1 success)"
        assert response is not None, "Should eventually succeed"
        assert elapsed_time >= 0.3, "Should have waited for backoff (0.1 + 0.2)"

    @patch("litellm.completion")
    def test_completion_failure(self, mock_completion, sample_messages):
        """
        Test handling of permanent failures.

        Non-retryable errors should fail immediately without retries.
        """
        # Configure mock to always fail with non-retryable error
        mock_completion.side_effect = ValueError("Invalid API key")

        config = LLMConfig(
            model="gpt-4",
            retry_attempts=3,
        )
        adapter = UniversalLLMAdapter(config)

        with pytest.raises(ValueError, match="Invalid API key"):
            adapter.completion(sample_messages)

    def _create_mock_response(self):
        """Helper to create a mock LiteLLM response."""

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


@pytest.mark.unit
class TestStreamingCompletion:
    """Test cases for streaming completion functionality."""

    @patch("litellm.completion")
    def test_streaming_completion(self, mock_completion, sample_messages):
        """
        Test streaming completion.

        Verifies that the adapter correctly yields chunks from a streaming response.
        """

        class MockDelta:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.delta = MockDelta(content)

        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        # Create streaming response
        chunks = ["Hello", ", ", "World", "!"]
        mock_completion.return_value = iter([MockChunk(c) for c in chunks])

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        # Patch token_counter
        with patch("litellm.token_counter", return_value=10):
            collected_chunks = list(adapter.stream_completion(sample_messages))

        assert len(collected_chunks) == 4, "Should yield 4 chunks"
        assert "".join(collected_chunks) == "Hello, World!", "Chunks should form complete message"

    @patch("litellm.completion")
    def test_streaming_completion_interruption(self, mock_completion, sample_messages):
        """
        Test handling of stream interruptions.

        The adapter should handle exceptions during streaming gracefully.
        """

        def failing_generator():
            yield self._create_chunk("Hello")
            raise Exception("Network error during stream")

        mock_completion.return_value = failing_generator()

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        with pytest.raises(Exception, match="Network error"):
            list(adapter.stream_completion(sample_messages))

    def _create_chunk(self, content):
        """Helper to create a mock streaming chunk."""

        class MockDelta:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.delta = MockDelta(content)

        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        return MockChunk(content)


@pytest.mark.unit
class TestTokenCounting:
    """Test cases for token counting functionality."""

    @patch("litellm.token_counter")
    def test_token_counting(self, mock_counter):
        """
        Test token counting with various text lengths.

        Verifies that the adapter correctly counts tokens.
        """
        mock_counter.return_value = 42

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        count = adapter.count_tokens("This is a test message")

        assert count == 42, "Should return mocked token count"
        mock_counter.assert_called_once()

    @patch("litellm.token_counter")
    def test_token_counting_caching(self, mock_counter):
        """
        Test that token counting results are cached.

        Identical text should not trigger multiple API calls.
        """
        mock_counter.return_value = 42

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        # Count same text twice
        text = "This is a test message"
        count1 = adapter.count_tokens(text)
        count2 = adapter.count_tokens(text)

        assert count1 == count2, "Counts should be identical"
        # Due to caching, should only call once
        assert mock_counter.call_count == 1, "Should only call token_counter once due to cache"

    @patch("litellm.token_counter")
    def test_token_counting_fallback(self, mock_counter):
        """
        Test fallback token counting when API fails.

        Should use character-based estimation if token counting fails.
        """
        mock_counter.side_effect = Exception("Token counting API error")

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        count = adapter.count_tokens("This is a test message with 32 chars")

        # Fallback: ~1 token per 4 characters
        assert count == 8, "Should fallback to character-based estimation (32/4=8)"


@pytest.mark.unit
class TestCostTracking:
    """Test cases for cost tracking functionality."""

    @patch("litellm.completion")
    @patch("litellm.token_counter")
    def test_cost_tracking(self, mock_counter, mock_completion, sample_messages):
        """
        Test that adapter tracks costs across multiple requests.

        Verifies cumulative cost and token tracking.
        """
        mock_counter.return_value = 10

        class MockMessage:
            content = "Response"

        class MockChoice:
            message = MockMessage()
            finish_reason = "stop"

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            choices = [MockChoice()]
            model = "gpt-4"
            usage = MockUsage()

        mock_completion.return_value = MockResponse()

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        # Make multiple completions
        adapter.completion(sample_messages)
        adapter.completion(sample_messages)

        cost_info = adapter.get_cost_info()

        assert cost_info["request_count"] == 2, "Should have 2 requests"
        assert cost_info["total_tokens"] == 300, "Should have 300 total tokens (150*2)"
        assert cost_info["average_cost_per_request"] >= 0, "Average cost should be non-negative"


@pytest.mark.unit
@pytest.mark.asyncio
class TestConcurrentRequests:
    """Test cases for concurrent request handling."""

    @patch("litellm.completion")
    @patch("litellm.token_counter")
    async def test_concurrent_requests(self, mock_counter, mock_completion, sample_messages):
        """
        Test handling of multiple concurrent requests.

        Verifies that the adapter can handle multiple simultaneous requests
        without conflicts or race conditions.
        """
        import asyncio

        mock_counter.return_value = 10

        class MockMessage:
            content = "Response"

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

        mock_completion.return_value = MockResponse()

        config = LLMConfig(model="gpt-4")
        adapter = UniversalLLMAdapter(config)

        # Make concurrent requests (simulated)
        async def make_request():
            # Since completion is sync, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, adapter.completion, sample_messages)

        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5, "All 5 requests should complete"
        assert all(r["content"] == "Response" for r in results), "All responses should be valid"

        cost_info = adapter.get_cost_info()
        assert cost_info["request_count"] == 5, "Should track all 5 requests"
