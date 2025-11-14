"""
Configuration management for Agent Platform using Pydantic Settings.

This module provides a comprehensive settings system that loads configuration
from environment variables and .env files with validation and type safety.
"""

import multiprocessing
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM providers and API keys."""

    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    openrouter_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenRouter API key for accessing multiple models",
    )
    default_model: str = Field(
        default="gpt-4",
        description="Default LLM model to use",
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses",
    )
    default_max_tokens: int = Field(
        default=2000,
        ge=1,
        le=32000,
        description="Default maximum tokens for LLM responses",
    )

    @field_validator("openai_api_key", "anthropic_api_key", "openrouter_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr], info) -> Optional[SecretStr]:
        """Validate API key format based on provider."""
        if v is None:
            return v

        key = v.get_secret_value()
        field_name = info.field_name

        if field_name == "openai_api_key" and not key.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        elif field_name == "anthropic_api_key" and not key.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        elif field_name == "openrouter_api_key" and not key.startswith("sk-or-"):
            raise ValueError("OpenRouter API key must start with 'sk-or-'")

        return v


class APIConfig(BaseSettings):
    """Configuration for the FastAPI server."""

    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for the API server",
    )
    reload: bool = Field(
        default=True,
        description="Auto-reload on code changes (development only)",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes",
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v


class SandboxConfig(BaseSettings):
    """Configuration for Docker-based sandbox environments."""

    image: str = Field(
        default="agent-sandbox:latest",
        description="Docker image to use for sandbox environments",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum execution time in seconds",
    )
    memory_limit: str = Field(
        default="512m",
        description="Memory limit for sandbox containers (e.g., 512m, 1g, 2g)",
    )
    cpu_limit: float = Field(
        default=1.0,
        ge=0.1,
        le=float(multiprocessing.cpu_count()),
        description="CPU limit (number of cores)",
    )
    enable_gvisor: bool = Field(
        default=False,
        description="Enable gVisor for enhanced security isolation",
    )
    network_disabled: bool = Field(
        default=True,
        description="Disable network access in sandbox",
    )
    enabled: bool = Field(
        default=True,
        description="Enable sandbox functionality. Set to False to disable Docker-based execution",
    )
    mock_mode: bool = Field(
        default=False,
        description="Use mock sandbox implementation when Docker is not available",
    )
    docker_auto_detect: bool = Field(
        default=True,
        description="Automatically detect Docker availability on startup",
    )
    docker_required: bool = Field(
        default=False,
        description="Require Docker to be available (fail startup if not available)",
    )
    docker_connection_timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Docker connection timeout in seconds",
    )
    docker_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of Docker connection retry attempts",
    )

    @field_validator("memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        """Validate memory limit format (e.g., 512m, 1g, 2g)."""
        pattern = r"^\d+[kmg]$"
        if not re.match(pattern, v.lower()):
            raise ValueError(
                "Memory limit must be in format: <number><unit> "
                "(e.g., 512m, 1g, 2g, where unit is k/m/g)"
            )
        return v

    @field_validator("cpu_limit")
    @classmethod
    def validate_cpu_limit(cls, v: float) -> float:
        """Validate CPU limit is within available cores."""
        max_cpus = multiprocessing.cpu_count()
        if v > max_cpus:
            raise ValueError(
                f"CPU limit ({v}) cannot exceed available CPUs ({max_cpus})"
            )
        return v


class SecurityConfig(BaseSettings):
    """Configuration for security settings."""

    secret_key: SecretStr = Field(
        default=SecretStr("your-secret-key-here-change-in-production"),
        description="Secret key for signing tokens and encrypting sensitive data",
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Allowed hosts for the application",
    )
    token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT token expiration time in minutes",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        """Validate secret key length and warn if using default."""
        key = v.get_secret_value()
        if len(key) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        if "change-in-production" in key.lower():
            import warnings

            warnings.warn(
                "Using default secret key! Generate a secure key for production.",
                UserWarning,
            )
        return v


class Settings(BaseSettings):
    """
    Main settings class for Agent Platform.

    This class aggregates all configuration sections and provides a unified
    interface for accessing application settings. Settings are loaded from
    environment variables and .env files with automatic validation.

    Example:
        >>> settings = get_settings()
        >>> print(settings.llm.default_model)
        'gpt-4'
        >>> print(settings.api.port)
        8000
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application metadata
    environment: str = Field(
        default="development",
        description="Environment: development, staging, or production",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode for detailed error messages",
    )

    # Database settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for caching and session storage",
    )

    # Telemetry settings
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable telemetry collection for monitoring and analytics",
    )

    # Nested configuration sections
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v_upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the expected values."""
        valid_envs = ["development", "staging", "production"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v_lower

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get the cached singleton settings instance.

    This function returns a cached instance of the Settings class, ensuring
    that configuration is loaded only once and reused throughout the application.

    Returns:
        Settings: The singleton settings instance.

    Example:
        >>> settings = get_settings()
        >>> # Subsequent calls return the same instance
        >>> settings2 = get_settings()
        >>> assert settings is settings2
    """
    return Settings()