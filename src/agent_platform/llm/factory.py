"""
Factory for creating LLM adapters with pre-configured settings.

This module provides a convenient factory pattern for instantiating LLM adapters
with sensible defaults for popular models. It includes pricing information and
model configurations for major providers.
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import SecretStr

from agent_platform.llm.adapter import LLMConfig, UniversalLLMAdapter
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class Provider(str, Enum):
    """Enumeration of supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# Model configurations with default parameters and pricing
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI Models
    "gpt-4": {
        "provider": Provider.OPENAI,
        "max_tokens": 8192,
        "temperature": 0.7,
        "context_window": 8192,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "gpt-4-turbo": {
        "provider": Provider.OPENAI,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 128000,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "gpt-4o": {
        "provider": Provider.OPENAI,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 128000,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "gpt-3.5-turbo": {
        "provider": Provider.OPENAI,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 16385,
        "cost_per_1k_input": 0.0005,
        "cost_per_1k_output": 0.0015,
        "supports_streaming": True,
        "supports_functions": True,
    },
    # Anthropic Claude Models
    "claude-3-5-sonnet-20241022": {
        "provider": Provider.ANTHROPIC,
        "max_tokens": 8192,
        "temperature": 0.7,
        "context_window": 200000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "claude-3-opus-20240229": {
        "provider": Provider.ANTHROPIC,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 200000,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "claude-3-sonnet-20240229": {
        "provider": Provider.ANTHROPIC,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 200000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "claude-3-haiku-20240307": {
        "provider": Provider.ANTHROPIC,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 200000,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
        "supports_streaming": True,
        "supports_functions": True,
    },
    # Google Gemini Models
    "gemini-pro": {
        "provider": Provider.GOOGLE,
        "max_tokens": 2048,
        "temperature": 0.7,
        "context_window": 32760,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.0005,
        "supports_streaming": True,
        "supports_functions": False,
    },
    "gemini-1.5-pro": {
        "provider": Provider.GOOGLE,
        "max_tokens": 8192,
        "temperature": 0.7,
        "context_window": 1000000,
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.005,
        "supports_streaming": True,
        "supports_functions": True,
    },
    "gemini-1.5-flash": {
        "provider": Provider.GOOGLE,
        "max_tokens": 8192,
        "temperature": 0.7,
        "context_window": 1000000,
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
        "supports_streaming": True,
        "supports_functions": True,
    },
    # Ollama Models (local, free)
    "ollama/llama2": {
        "provider": Provider.OLLAMA,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 4096,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "supports_streaming": True,
        "supports_functions": False,
    },
    "ollama/mistral": {
        "provider": Provider.OLLAMA,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 8192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "supports_streaming": True,
        "supports_functions": False,
    },
    "ollama/codellama": {
        "provider": Provider.OLLAMA,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 16384,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "supports_streaming": True,
        "supports_functions": False,
    },
    "ollama/llama3": {
        "provider": Provider.OLLAMA,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 8192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "supports_streaming": True,
        "supports_functions": False,
    },
    "ollama/mixtral": {
        "provider": Provider.OLLAMA,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window": 32768,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "supports_streaming": True,
        "supports_functions": False,
    },
}


class LLMFactory:
    """
    Factory class for creating LLM adapters with sensible defaults.

    This factory simplifies the creation of LLM adapters by providing
    pre-configured settings for popular models and handling API key
    resolution from environment variables.

    Example:
        >>> # Create adapter with defaults
        >>> adapter = LLMFactory.create_adapter("gpt-4")
        >>>
        >>> # Create with custom settings
        >>> adapter = LLMFactory.create_adapter(
        ...     "claude-3-sonnet",
        ...     temperature=0.5,
        ...     max_tokens=1000
        ... )
        >>>
        >>> # Get model information
        >>> info = LLMFactory.get_model_info("gpt-4")
        >>> print(f"Context window: {info['context_window']}")
    """

    @staticmethod
    def create_adapter(
        model: str, api_key: Optional[str] = None, **kwargs: Any
    ) -> UniversalLLMAdapter:
        """
        Create an LLM adapter for the specified model.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet").
            api_key: Optional API key. If not provided, will try to get from
                    environment variables based on the provider.
            **kwargs: Additional parameters to override default config
                     (temperature, max_tokens, etc.).

        Returns:
            Configured UniversalLLMAdapter instance.

        Raises:
            ValueError: If model is not found in MODEL_CONFIGS.
            RuntimeError: If API key is required but not found.

        Example:
            >>> adapter = LLMFactory.create_adapter("gpt-4", temperature=0.5)
            >>> response = adapter.completion([{"role": "user", "content": "Hi"}])
        """
        # Check if model is configured
        if model not in MODEL_CONFIGS:
            logger.warning(
                f"Model '{model}' not in predefined configs, using default settings"
            )
            model_config = {
                "provider": Provider.OPENAI,
                "max_tokens": 2000,
                "temperature": 0.7,
            }
        else:
            model_config = MODEL_CONFIGS[model].copy()

        # Resolve API key
        if api_key is None:
            api_key = LLMFactory._get_api_key_from_env(model_config.get("provider"))

        # Build configuration
        config_params = {
            "model": model,
            "api_key": SecretStr(api_key) if api_key else None,
            "temperature": kwargs.pop("temperature", model_config.get("temperature", 0.7)),
            "max_tokens": kwargs.pop("max_tokens", model_config.get("max_tokens", 2000)),
            "timeout": kwargs.pop("timeout", 30),
            "streaming": kwargs.pop("streaming", True),
            "fallback_models": kwargs.pop("fallback_models", []),
            "retry_attempts": kwargs.pop("retry_attempts", 3),
            "retry_delay": kwargs.pop("retry_delay", 1.0),
        }

        # Create config
        config = LLMConfig(**config_params)

        # Create and return adapter
        adapter = UniversalLLMAdapter(config)

        logger.info(
            f"Created LLM adapter",
            extra={
                "model": model,
                "provider": model_config.get("provider"),
                "temperature": config.temperature,
            },
        )

        return adapter

    @staticmethod
    def get_model_info(model: str) -> Dict[str, Any]:
        """
        Get configuration and pricing information for a model.

        Args:
            model: Model identifier.

        Returns:
            Dictionary with model configuration and pricing info.

        Raises:
            ValueError: If model is not found in MODEL_CONFIGS.

        Example:
            >>> info = LLMFactory.get_model_info("gpt-4")
            >>> print(f"Cost per 1K input tokens: ${info['cost_per_1k_input']}")
        """
        if model not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model}' not found. "
                f"Available models: {', '.join(LLMFactory.list_available_models())}"
            )

        return MODEL_CONFIGS[model].copy()

    @staticmethod
    def list_available_models(provider: Optional[Provider] = None) -> List[str]:
        """
        List all available models, optionally filtered by provider.

        Args:
            provider: Optional provider to filter by.

        Returns:
            List of model identifiers.

        Example:
            >>> # List all models
            >>> all_models = LLMFactory.list_available_models()
            >>>
            >>> # List only OpenAI models
            >>> openai_models = LLMFactory.list_available_models(Provider.OPENAI)
        """
        if provider is None:
            return sorted(MODEL_CONFIGS.keys())

        return sorted(
            [
                model
                for model, config in MODEL_CONFIGS.items()
                if config.get("provider") == provider
            ]
        )

    @staticmethod
    def _get_api_key_from_env(provider: Optional[Provider]) -> Optional[str]:
        """
        Get API key from environment variables based on provider.

        Args:
            provider: The LLM provider.

        Returns:
            API key from environment, or None if not found.
        """
        if provider == Provider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        elif provider == Provider.ANTHROPIC:
            return os.getenv("ANTHROPIC_API_KEY")
        elif provider == Provider.GOOGLE:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == Provider.OPENROUTER:
            return os.getenv("OPENROUTER_API_KEY")
        elif provider == Provider.OLLAMA:
            # Ollama runs locally, no API key needed
            return None

        return None


def create_llm(model: str = "gpt-4", **kwargs: Any) -> UniversalLLMAdapter:
    """
    Convenient wrapper for creating an LLM adapter.

    This is a shorthand for LLMFactory.create_adapter() for quick usage.

    Args:
        model: Model identifier. Defaults to "gpt-4".
        **kwargs: Additional configuration parameters.

    Returns:
        Configured UniversalLLMAdapter instance.

    Example:
        >>> # Quick creation with defaults
        >>> llm = create_llm()
        >>>
        >>> # Custom model and settings
        >>> llm = create_llm("claude-3-sonnet", temperature=0.8, max_tokens=1500)
        >>>
        >>> # Use the adapter
        >>> response = llm.completion([{"role": "user", "content": "Hello!"}])
        >>> print(response["content"])
    """
    return LLMFactory.create_adapter(model, **kwargs)
