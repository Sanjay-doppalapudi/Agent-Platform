"""
Integration tests for LLM adapters with real API calls.

These tests make actual API calls to LLM providers and require valid API keys.
They are skipped if the necessary API keys are not available in the environment.

Run these tests with: pytest tests/integration/ -v -s
"""

import os

import pytest

from agent_platform.llm.adapter import LLMConfig, UniversalLLMAdapter
from agent_platform.llm.cost_tracker import calculate_cost
from agent_platform.llm.factory import create_llm

# Check for API keys
HAS_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").startswith("sk-")
HAS_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
@pytest.mark.timeout(30)
class TestOpenAIIntegration:
    """Integration tests for OpenAI models."""

    def test_real_openai_completion(self):
        """
        Test real OpenAI completion request.

        Makes an actual API call to OpenAI and verifies the response structure.
        Uses a simple prompt to keep costs low.
        """
        # Create adapter
        adapter = create_llm("gpt-3.5-turbo", temperature=0.7, max_tokens=50)

        # Simple test prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."},
        ]

        # Make request
        response = adapter.completion(messages)

        # Verify response structure
        assert "content" in response, "Response should contain content"
        assert "model" in response, "Response should contain model"
        assert "usage" in response, "Response should contain usage stats"
        assert isinstance(response["content"], str), "Content should be a string"
        assert len(response["content"]) > 0, "Content should not be empty"

        # Verify the response contains expected text
        content_lower = response["content"].lower()
        assert (
            "hello" in content_lower and "world" in content_lower
        ), "Response should contain 'Hello, World!'"

        # Verify usage stats
        assert response["usage"]["total_tokens"] > 0, "Should have used some tokens"
        assert response["usage"]["prompt_tokens"] > 0, "Should have prompt tokens"
        assert response["usage"]["completion_tokens"] > 0, "Should have completion tokens"

    def test_real_openai_streaming(self):
        """
        Test real OpenAI streaming completion.

        Verifies that streaming works correctly and all chunks form a complete message.
        """
        adapter = create_llm("gpt-3.5-turbo", temperature=0.7, max_tokens=50, streaming=True)

        messages = [
            {"role": "user", "content": "Count from 1 to 5, separated by commas."}
        ]

        # Collect all chunks
        chunks = []
        for chunk in adapter.stream_completion(messages):
            chunks.append(chunk)
            assert isinstance(chunk, str), "Each chunk should be a string"

        # Verify we got chunks
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Verify complete message
        complete_message = "".join(chunks)
        assert len(complete_message) > 0, "Complete message should not be empty"

    def test_real_token_counting_accuracy(self):
        """
        Test token counting accuracy against real API.

        Compares our token counter with actual tokens used by the API.
        """
        adapter = create_llm("gpt-3.5-turbo", max_tokens=50)

        test_text = "This is a test message for token counting accuracy."
        messages = [{"role": "user", "content": test_text}]

        # Get our token count
        our_count = adapter.count_tokens(test_text)

        # Make actual API call to get real count
        response = adapter.completion(messages)
        actual_prompt_tokens = response["usage"]["prompt_tokens"]

        # Allow for small differences due to special tokens, role markers, etc.
        # Real prompt tokens include system tokens, so we just check our count is reasonable
        assert our_count > 0, "Our token count should be positive"
        assert our_count <= actual_prompt_tokens, (
            "Our count should be less than or equal to actual "
            "(actual includes role tokens and special tokens)"
        )

    def test_real_cost_calculation(self):
        """
        Test cost calculation accuracy.

        Makes a real API call and verifies the cost is in the expected range.
        """
        adapter = create_llm("gpt-3.5-turbo", max_tokens=50)

        messages = [{"role": "user", "content": "Say hello."}]

        # Make request
        response = adapter.completion(messages)

        # Calculate cost using our tracker
        cost = calculate_cost(
            "gpt-3.5-turbo",
            response["usage"]["prompt_tokens"],
            response["usage"]["completion_tokens"],
        )

        # Verify cost is reasonable (should be very small for this simple request)
        assert cost > 0, "Cost should be positive"
        assert cost < 0.01, "Cost should be less than 1 cent for this simple request"

        # Verify it's in the ballpark (GPT-3.5-turbo is cheap)
        expected_min = (response["usage"]["prompt_tokens"] / 1000) * 0.0001
        expected_max = (response["usage"]["total_tokens"] / 1000) * 0.01
        assert expected_min <= cost <= expected_max, "Cost should be in expected range"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="Anthropic API key not available")
@pytest.mark.timeout(30)
class TestAnthropicIntegration:
    """Integration tests for Anthropic Claude models."""

    def test_real_anthropic_completion(self):
        """
        Test real Anthropic Claude completion request.

        Makes an actual API call to Anthropic and verifies the response structure.
        """
        # Create adapter (using Haiku for cost efficiency)
        adapter = create_llm("claude-3-haiku-20240307", temperature=0.7, max_tokens=50)

        # Simple test prompt
        messages = [{"role": "user", "content": "Say 'Hello, World!' and nothing else."}]

        # Make request
        response = adapter.completion(messages)

        # Verify response structure
        assert "content" in response, "Response should contain content"
        assert "model" in response, "Response should contain model"
        assert "usage" in response, "Response should contain usage stats"
        assert isinstance(response["content"], str), "Content should be a string"
        assert len(response["content"]) > 0, "Content should not be empty"

        # Verify the response contains expected text
        content_lower = response["content"].lower()
        assert (
            "hello" in content_lower and "world" in content_lower
        ), "Response should contain 'Hello, World!'"

    def test_real_anthropic_streaming(self):
        """
        Test real Anthropic streaming completion.

        Verifies that streaming works correctly with Claude models.
        """
        adapter = create_llm(
            "claude-3-haiku-20240307", temperature=0.7, max_tokens=50, streaming=True
        )

        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        # Collect all chunks
        chunks = list(adapter.stream_completion(messages))

        # Verify we got chunks
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Verify complete message
        complete_message = "".join(chunks)
        assert len(complete_message) > 0, "Complete message should not be empty"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
class TestErrorHandling:
    """Integration tests for error handling with real APIs."""

    def test_error_handling_invalid_api_key(self):
        """
        Test error handling with invalid API key.

        Verifies that the adapter properly handles authentication errors.
        """
        from pydantic import SecretStr

        # Create adapter with invalid API key
        config = LLMConfig(
            model="gpt-3.5-turbo",
            api_key=SecretStr("sk-invalid-key-for-testing"),
            temperature=0.7,
            max_tokens=50,
            retry_attempts=0,  # Don't retry for this test
        )
        adapter = UniversalLLMAdapter(config)

        messages = [{"role": "user", "content": "Hello"}]

        # Should raise an exception
        with pytest.raises(Exception) as exc_info:
            adapter.completion(messages)

        # Verify error message indicates authentication issue
        error_msg = str(exc_info.value).lower()
        # Could be various auth-related errors
        assert (
            "api" in error_msg
            or "auth" in error_msg
            or "invalid" in error_msg
            or "key" in error_msg
        ), "Error should indicate API/authentication issue"

    def test_error_handling_malformed_request(self):
        """
        Test error handling with malformed request.

        Verifies that invalid messages are caught before making API calls.
        """
        adapter = create_llm("gpt-3.5-turbo")

        # Test with empty messages
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            adapter.completion([])

        # Test with invalid role
        with pytest.raises(ValueError, match="invalid role"):
            adapter.completion([{"role": "invalid", "content": "test"}])

    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
    def test_error_handling_rate_limit_retry(self):
        """
        Test that adapter handles rate limits with retries.

        Note: This test doesn't actually trigger a rate limit (expensive/slow),
        but verifies the retry configuration is set up correctly.
        """
        adapter = create_llm("gpt-3.5-turbo", retry_attempts=3, retry_delay=1.0)

        # Verify retry configuration
        assert adapter.config.retry_attempts == 3, "Should have 3 retry attempts"
        assert adapter.config.retry_delay == 1.0, "Should have 1 second retry delay"

        # Make a normal request to verify it works
        messages = [{"role": "user", "content": "Say hi."}]
        response = adapter.completion(messages)

        assert response["content"], "Should get a response"


@pytest.mark.integration
@pytest.mark.skipif(
    not (HAS_OPENAI_KEY or HAS_ANTHROPIC_KEY),
    reason="No API keys available for testing",
)
class TestMultiProvider:
    """Integration tests across multiple providers."""

    def test_adapter_cost_tracking_across_providers(self):
        """
        Test that cost tracking works across different providers.

        Makes requests to different models and verifies costs are tracked separately.
        """
        if HAS_OPENAI_KEY:
            openai_adapter = create_llm("gpt-3.5-turbo", max_tokens=20)
            messages = [{"role": "user", "content": "Hi"}]

            openai_adapter.completion(messages)
            openai_cost_info = openai_adapter.get_cost_info()

            assert openai_cost_info["request_count"] > 0, "Should track OpenAI requests"
            assert openai_cost_info["total_tokens"] > 0, "Should track OpenAI tokens"

        if HAS_ANTHROPIC_KEY:
            anthropic_adapter = create_llm("claude-3-haiku-20240307", max_tokens=20)
            messages = [{"role": "user", "content": "Hi"}]

            anthropic_adapter.completion(messages)
            anthropic_cost_info = anthropic_adapter.get_cost_info()

            assert (
                anthropic_cost_info["request_count"] > 0
            ), "Should track Anthropic requests"
            assert anthropic_cost_info["total_tokens"] > 0, "Should track Anthropic tokens"
