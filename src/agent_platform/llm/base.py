"""
Abstract base class for LLM providers.

This module defines the interface that all LLM provider implementations must follow.
It provides a consistent API for interacting with different language models while
allowing for provider-specific implementations.

Example:
    >>> from agent_platform.llm.adapter import UniversalLLMAdapter
    >>> from agent_platform.llm.base import Message
    >>>
    >>> # Create an adapter instance
    >>> adapter = UniversalLLMAdapter(config)
    >>>
    >>> # Prepare messages
    >>> messages = [
    ...     Message(role="system", content="You are a helpful assistant."),
    ...     Message(role="user", content="What is Python?"),
    ... ]
    >>>
    >>> # Get completion
    >>> response = adapter.completion(messages)
    >>> print(response["content"])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict


class Message(TypedDict):
    """
    Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender ("system", "user", or "assistant").
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResponse(TypedDict):
    """
    Represents a completion response from an LLM.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        usage: Token usage statistics including prompt_tokens, completion_tokens,
               and total_tokens.
        finish_reason: Optional reason why the generation stopped (e.g., "stop",
                      "length", "content_filter").
    """

    content: str
    model: str
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    finish_reason: Optional[str]


class StreamChunk(TypedDict):
    """
    Represents a single chunk in a streaming completion response.

    Attributes:
        content: The text content of this chunk.
        finish_reason: Optional reason why the stream ended. Present only in the
                      final chunk (e.g., "stop", "length").
    """

    content: str
    finish_reason: Optional[str]


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.

    This class defines the interface that all LLM providers must implement.
    It ensures consistency across different provider implementations and
    provides common utility methods for message validation and error handling.

    Concrete implementations must provide:
    - completion(): Non-streaming text generation
    - stream_completion(): Streaming text generation
    - count_tokens(): Token counting for the specific model
    - get_model_info(): Model metadata and capabilities

    Example:
        >>> class MyLLMProvider(BaseLLMProvider):
        ...     def completion(self, messages, **kwargs):
        ...         # Implementation here
        ...         pass
        ...
        ...     def stream_completion(self, messages, **kwargs):
        ...         # Implementation here
        ...         pass
        ...
        ...     def count_tokens(self, text):
        ...         # Implementation here
        ...         pass
        ...
        ...     def get_model_info(self):
        ...         # Implementation here
        ...         pass
    """

    @abstractmethod
    def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages.

        This method sends a list of messages to the LLM and returns a complete
        response. Use this for non-streaming interactions where you want to
        receive the full response at once.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters (e.g., temperature,
                     max_tokens, top_p).

        Returns:
            CompletionResponse containing the generated content, model name,
            token usage statistics, and finish reason.

        Raises:
            ValueError: If messages are invalid or empty.
            Exception: Provider-specific errors (API errors, rate limits, etc.).

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = provider.completion(messages, temperature=0.7)
            >>> print(response["content"])
        """
        pass

    @abstractmethod
    def stream_completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate a streaming completion for the given messages.

        This method sends messages to the LLM and yields content chunks as they
        are generated. Use this for real-time interactions where you want to
        display results progressively.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the generated content as they become available.

        Raises:
            ValueError: If messages are invalid or empty.
            Exception: Provider-specific errors (API errors, stream interruptions).

        Example:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in provider.stream_completion(messages):
            ...     print(chunk, end="", flush=True)
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Token counting is model-specific as different models use different
        tokenizers. This method should use the appropriate tokenizer for
        the provider's model.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.

        Example:
            >>> text = "Hello, how are you?"
            >>> token_count = provider.count_tokens(text)
            >>> print(f"Tokens: {token_count}")
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and its capabilities.

        Returns metadata about the model including its name, context window size,
        pricing, and supported features.

        Returns:
            Dictionary containing model information with keys such as:
            - name: Model name/identifier
            - context_window: Maximum context window size in tokens
            - max_tokens: Maximum tokens that can be generated
            - supports_streaming: Whether streaming is supported
            - supports_functions: Whether function calling is supported
            - pricing: Cost per token (if available)

        Example:
            >>> info = provider.get_model_info()
            >>> print(f"Model: {info['name']}")
            >>> print(f"Context window: {info['context_window']} tokens")
        """
        pass

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate that messages are properly formatted.

        Checks that:
        - Messages list is not empty
        - Each message has 'role' and 'content' keys
        - Roles are valid ("system", "user", or "assistant")
        - Content is a non-empty string

        Args:
            messages: List of message dictionaries to validate.

        Returns:
            True if all messages are valid.

        Raises:
            ValueError: If any message is invalid, with a descriptive error message.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello"}
            ... ]
            >>> provider.validate_messages(messages)  # Returns True
            >>>
            >>> invalid_messages = [{"role": "invalid", "content": ""}]
            >>> provider.validate_messages(invalid_messages)  # Raises ValueError
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant"}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' key")

            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' key")

            if msg["role"] not in valid_roles:
                raise ValueError(
                    f"Message {i} has invalid role '{msg['role']}'. "
                    f"Must be one of: {', '.join(valid_roles)}"
                )

            if not isinstance(msg["content"], str):
                raise ValueError(f"Message {i} content must be a string")

            if not msg["content"].strip():
                raise ValueError(f"Message {i} content cannot be empty")

        return True

    def format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format an exception into a standardized error dictionary.

        This provides a consistent error format across different providers,
        making it easier to handle errors in application code.

        Args:
            error: The exception to format.

        Returns:
            Dictionary containing error information with keys:
            - error_type: The type/name of the exception
            - error_message: The error message
            - is_retryable: Whether the error might succeed on retry

        Example:
            >>> try:
            ...     response = provider.completion(messages)
            ... except Exception as e:
            ...     error_info = provider.format_error(e)
            ...     print(f"Error: {error_info['error_message']}")
            ...     if error_info['is_retryable']:
            ...         print("This error might succeed if retried")
        """
        error_type = type(error).__name__
        error_message = str(error)

        # Determine if error is retryable based on common patterns
        retryable_errors = [
            "timeout",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
            "connection",
            "network",
        ]
        is_retryable = any(
            keyword in error_message.lower() for keyword in retryable_errors
        )

        return {
            "error_type": error_type,
            "error_message": error_message,
            "is_retryable": is_retryable,
            "original_error": error,
        }
