"""
Universal LLM adapter using LiteLLM for multi-provider support.

This module provides a unified interface for interacting with multiple LLM
providers (OpenAI, Anthropic, Google, etc.) through the LiteLLM library.
It handles retries, cost tracking, response normalization, and error handling.
"""

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

import litellm
from litellm import completion, token_counter
from pydantic import SecretStr

from agent_platform.llm.base import BaseLLMProvider, CompletionResponse
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM adapter.

    Attributes:
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet").
        api_key: Optional API key for the provider. If not provided, will use
                environment variables.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random).
        max_tokens: Maximum number of tokens to generate.
        timeout: Request timeout in seconds.
        streaming: Whether to use streaming by default.
        fallback_models: List of models to try if primary model fails.
        retry_attempts: Number of times to retry failed requests.
        retry_delay: Initial delay between retries in seconds (exponential backoff).
    """

    model: str
    api_key: Optional[SecretStr] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    streaming: bool = True
    fallback_models: List[str] = field(default_factory=list)
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if not 1 <= self.max_tokens <= 32000:
            raise ValueError("max_tokens must be between 1 and 32000")
        if self.timeout < 1:
            raise ValueError("Timeout must be at least 1 second")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")


class UniversalLLMAdapter(BaseLLMProvider):
    """
    Universal adapter for multiple LLM providers using LiteLLM.

    This adapter provides a consistent interface for interacting with various
    LLM providers while handling provider-specific differences internally.
    It includes features like automatic retries, cost tracking, and response
    normalization.

    Example:
        >>> config = LLMConfig(
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> adapter = UniversalLLMAdapter(config)
        >>>
        >>> messages = [
        ...     {"role": "user", "content": "What is AI?"}
        ... ]
        >>> response = adapter.completion(messages)
        >>> print(response["content"])
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the adapter with configuration.

        Args:
            config: LLMConfig instance with model and parameters.
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Set API key if provided
        if config.api_key:
            # LiteLLM will use environment variables by default,
            # but we can set them here if provided
            self._set_api_key(config.api_key)

        # Initialize cost tracking
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._request_count: int = 0
        self._usage_by_model: Dict[str, Dict[str, Any]] = {}

        # Token counting cache
        self._token_cache: Dict[str, int] = {}

        self.logger.info(
            f"Initialized UniversalLLMAdapter",
            extra={
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )

    def _set_api_key(self, api_key: SecretStr) -> None:
        """Set the API key for the LLM provider."""
        # LiteLLM uses environment variables, but we can set them programmatically
        import os

        key_value = api_key.get_secret_value()

        # Determine provider from model name and set appropriate env var
        model_lower = self.config.model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            os.environ["OPENAI_API_KEY"] = key_value
        elif "claude" in model_lower or "anthropic" in model_lower:
            os.environ["ANTHROPIC_API_KEY"] = key_value
        elif "gemini" in model_lower or "google" in model_lower:
            os.environ["GEMINI_API_KEY"] = key_value

    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Override configuration parameters (temperature, max_tokens, etc.).

        Returns:
            CompletionResponse with content, model, usage, and finish_reason.

        Raises:
            ValueError: If messages are invalid.
            Exception: If all retry attempts fail.

        Example:
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = adapter.completion(messages, temperature=0.5)
        """
        # Validate messages
        self.validate_messages(messages)

        # Merge config with kwargs
        params = self._build_params(kwargs)

        # Log request
        self.logger.debug(
            f"Sending completion request",
            extra={
                "model": params["model"],
                "message_count": len(messages),
                "temperature": params.get("temperature"),
            },
        )

        # Try with retries
        last_error = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Make the request
                response = completion(messages=messages, **params)

                # Normalize and track
                normalized = self._normalize_response(response)
                self._track_usage(normalized)

                # Log success
                self.logger.debug(
                    f"Completion successful",
                    extra={
                        "model": normalized["model"],
                        "tokens": normalized["usage"]["total_tokens"],
                    },
                )

                return normalized

            except Exception as e:
                last_error = e
                error_info = self.format_error(e)

                self.logger.warning(
                    f"Completion attempt {attempt + 1} failed: {error_info['error_message']}",
                    extra={
                        "attempt": attempt + 1,
                        "is_retryable": error_info["is_retryable"],
                    },
                )

                # Don't retry if not retryable or last attempt
                if not error_info["is_retryable"] or attempt >= self.config.retry_attempts:
                    break

                # Exponential backoff
                delay = self.config.retry_delay * (2**attempt)
                time.sleep(delay)

        # All attempts failed
        self.logger.error(
            f"All completion attempts failed",
            extra={"attempts": self.config.retry_attempts + 1},
        )
        raise last_error

    def stream_completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate a streaming completion for the given messages.

        Args:
            messages: List of message dictionaries.
            **kwargs: Override configuration parameters.

        Yields:
            String chunks of generated content.

        Raises:
            ValueError: If messages are invalid.
            Exception: If the stream fails.

        Example:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in adapter.stream_completion(messages):
            ...     print(chunk, end="", flush=True)
        """
        # Validate messages
        self.validate_messages(messages)

        # Merge config with kwargs and enable streaming
        params = self._build_params(kwargs)
        params["stream"] = True

        self.logger.debug(
            f"Starting streaming completion",
            extra={"model": params["model"], "message_count": len(messages)},
        )

        try:
            # Make streaming request
            response = completion(messages=messages, **params)

            # Track cumulative content and tokens
            full_content = ""
            cumulative_tokens = 0

            # Iterate over chunks
            for chunk in response:
                # Extract content from chunk based on provider format
                content = self._extract_chunk_content(chunk)

                if content:
                    full_content += content
                    yield content

            # Estimate token count for tracking
            cumulative_tokens = self.count_tokens(full_content)

            # Track usage (approximate)
            self._track_usage(
                {
                    "model": params["model"],
                    "usage": {
                        "prompt_tokens": self.count_tokens(
                            " ".join(m["content"] for m in messages)
                        ),
                        "completion_tokens": cumulative_tokens,
                        "total_tokens": cumulative_tokens
                        + self.count_tokens(" ".join(m["content"] for m in messages)),
                    },
                    "content": full_content,
                    "finish_reason": "stop",
                }
            )

            self.logger.debug(
                f"Streaming completion finished",
                extra={"total_tokens": cumulative_tokens},
            )

        except Exception as e:
            error_info = self.format_error(e)
            self.logger.error(
                f"Streaming failed: {error_info['error_message']}",
                extra={"error_type": error_info["error_type"]},
            )
            raise

    def _extract_chunk_content(self, chunk: Any) -> str:
        """
        Extract content from a streaming chunk.

        Different providers format chunks differently. This method handles
        the various formats.

        Args:
            chunk: The chunk from the streaming response.

        Returns:
            The content string from the chunk.
        """
        try:
            # LiteLLM normalizes chunk format
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    return delta.content
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to extract chunk content: {e}")
            return ""

    @lru_cache(maxsize=1000)
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using LiteLLM's token counter.

        Results are cached for performance.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.

        Example:
            >>> count = adapter.count_tokens("Hello, world!")
            >>> print(count)
        """
        try:
            return token_counter(model=self.config.model, text=text)
        except Exception as e:
            self.logger.warning(
                f"Token counting failed, using estimate: {e}",
            )
            # Fallback: rough estimate of 1 token per 4 characters
            return len(text) // 4

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model metadata.

        Example:
            >>> info = adapter.get_model_info()
            >>> print(info["name"])
        """
        return {
            "name": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "streaming_enabled": self.config.streaming,
            "fallback_models": self.config.fallback_models,
        }

    def get_cost_info(self) -> Dict[str, Any]:
        """
        Get cost and usage information.

        Returns:
            Dictionary with total cost, token counts, and per-model statistics.

        Example:
            >>> cost_info = adapter.get_cost_info()
            >>> print(f"Total cost: ${cost_info['total_cost']:.4f}")
        """
        avg_cost_per_request = (
            self._total_cost / self._request_count if self._request_count > 0 else 0.0
        )

        return {
            "total_cost": self._total_cost,
            "total_tokens": self._total_tokens,
            "request_count": self._request_count,
            "average_cost_per_request": avg_cost_per_request,
            "usage_by_model": self._usage_by_model,
        }

    def _build_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build request parameters by merging config with kwargs.

        Args:
            kwargs: Override parameters.

        Returns:
            Merged parameter dictionary.
        """
        params = {
            "model": self.config.model,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "timeout": kwargs.get("timeout", self.config.timeout),
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        return params

    def _normalize_response(self, response: Any) -> CompletionResponse:
        """
        Normalize LiteLLM response to standard format.

        Args:
            response: Raw response from LiteLLM.

        Returns:
            Normalized CompletionResponse.
        """
        # Extract content
        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content or ""

        # Extract model
        model = getattr(response, "model", self.config.model)

        # Extract usage
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage"):
            usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0)
            usage["completion_tokens"] = getattr(response.usage, "completion_tokens", 0)
            usage["total_tokens"] = getattr(response.usage, "total_tokens", 0)

        # Extract finish reason
        finish_reason = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            finish_reason = getattr(response.choices[0], "finish_reason", None)

        return CompletionResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _track_usage(self, response: CompletionResponse) -> None:
        """
        Track token usage and cost.

        Args:
            response: Normalized completion response.
        """
        self._request_count += 1
        tokens = response["usage"]["total_tokens"]
        self._total_tokens += tokens

        # Track per-model usage
        model = response["model"]
        if model not in self._usage_by_model:
            self._usage_by_model[model] = {
                "requests": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        self._usage_by_model[model]["requests"] += 1
        self._usage_by_model[model]["total_tokens"] += response["usage"]["total_tokens"]
        self._usage_by_model[model]["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self._usage_by_model[model]["completion_tokens"] += response["usage"][
            "completion_tokens"
        ]

        # Cost tracking would require pricing data
        # This is handled in cost_tracker.py
