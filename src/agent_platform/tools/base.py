"""
Abstract base classes for tools following LangChain patterns.

This module provides the foundation for creating tools that can be used
by agents. Tools are actions that agents can take, such as reading files,
executing commands, or making web requests.

Example:
    >>> from pydantic import BaseModel, Field
    >>> from agent_platform.tools.base import BaseTool, ToolResult
    >>>
    >>> # Define input schema
    >>> class CalculatorInput(BaseModel):
    ...     expression: str = Field(description="Math expression to evaluate")
    >>>
    >>> # Implement tool
    >>> class CalculatorTool(BaseTool):
    ...     name = "calculator"
    ...     description = "Evaluate mathematical expressions"
    ...     input_schema = CalculatorInput
    ...
    ...     def _run(self, expression: str) -> ToolResult:
    ...         try:
    ...             result = eval(expression)
    ...             return ToolResult(success=True, output=str(result))
    ...         except Exception as e:
    ...             return ToolResult(success=False, output="", error=str(e))
    >>>
    >>> # Use tool
    >>> calc = CalculatorTool()
    >>> result = calc.run(expression="2 + 2")
    >>> print(result.output)  # "4"
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class ToolInput(BaseModel):
    """
    Base class for tool input schemas.

    All tool-specific input classes should inherit from this.
    Use Pydantic Field() to add descriptions and validation.

    Example:
        >>> from pydantic import Field
        >>>
        >>> class MyToolInput(ToolInput):
        ...     file_path: str = Field(description="Path to the file")
        ...     encoding: str = Field(default="utf-8", description="File encoding")
    """

    class Config:
        extra = "forbid"  # Don't allow extra fields


@dataclass
class ToolResult:
    """
    Result returned by a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The output/result from the tool (empty string if failed).
        error: Error message if execution failed (None if successful).
        metadata: Additional metadata about the execution (timing, etc.).

    Example:
        >>> result = ToolResult(success=True, output="File contents here")
        >>> if result.success:
        ...     print(result.output)
        >>>
        >>> failed_result = ToolResult(
        ...     success=False,
        ...     output="",
        ...     error="File not found"
        ... )
    """

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return f"Success: {self.output[:100]}..." if len(self.output) > 100 else f"Success: {self.output}"
        else:
            return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools are actions that agents can take. Each tool must define:
    - name: Unique identifier for the tool
    - description: Human-readable description of what the tool does
    - input_schema: Pydantic model defining the tool's inputs

    Tools must implement:
    - _run(): Synchronous execution
    - _arun(): Asynchronous execution (can raise NotImplementedError)

    Example:
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...     input_schema = MyToolInput
        ...
        ...     def _run(self, **kwargs) -> ToolResult:
        ...         # Implementation here
        ...         return ToolResult(success=True, output="Done")
        ...
        ...     async def _arun(self, **kwargs) -> ToolResult:
        ...         # Async implementation
        ...         return self._run(**kwargs)
    """

    # Class attributes (must be overridden in subclasses)
    name: str
    description: str
    input_schema: Type[ToolInput]

    # Optional attributes
    return_direct: bool = False  # If True, return tool output directly to user
    category: Optional[str] = None  # Category for organizing tools

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required attributes."""
        super().__init_subclass__(**kwargs)

        # Check that required class attributes are defined
        if not hasattr(cls, "name") or cls.name == "":
            raise TypeError(f"{cls.__name__} must define 'name' attribute")
        if not hasattr(cls, "description") or cls.description == "":
            raise TypeError(f"{cls.__name__} must define 'description' attribute")
        if not hasattr(cls, "input_schema"):
            raise TypeError(f"{cls.__name__} must define 'input_schema' attribute")

    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult:
        """
        Synchronous tool execution logic.

        This method should be implemented by subclasses to define the
        actual tool behavior. It receives validated inputs as kwargs.

        Args:
            **kwargs: Validated input parameters from input_schema.

        Returns:
            ToolResult with success status, output, and optional error.

        Example:
            >>> def _run(self, file_path: str) -> ToolResult:
            ...     try:
            ...         with open(file_path) as f:
            ...             content = f.read()
            ...         return ToolResult(success=True, output=content)
            ...     except Exception as e:
            ...         return ToolResult(success=False, output="", error=str(e))
        """
        pass

    async def _arun(self, **kwargs: Any) -> ToolResult:
        """
        Asynchronous tool execution logic.

        This method can be overridden by subclasses to provide async execution.
        By default, it raises NotImplementedError.

        Args:
            **kwargs: Validated input parameters from input_schema.

        Returns:
            ToolResult with success status, output, and optional error.

        Raises:
            NotImplementedError: If async execution is not supported.

        Example:
            >>> async def _arun(self, url: str) -> ToolResult:
            ...     async with httpx.AsyncClient() as client:
            ...         response = await client.get(url)
            ...         return ToolResult(success=True, output=response.text)
        """
        raise NotImplementedError(
            f"Async execution not implemented for {self.name}. "
            "Override _arun() to support async execution."
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with input validation and error handling.

        This method wraps _run() with:
        - Input validation using input_schema
        - Exception handling
        - Execution timing
        - Logging

        Args:
            **kwargs: Input parameters for the tool.

        Returns:
            ToolResult with execution outcome.

        Example:
            >>> tool = MyTool()
            >>> result = tool.run(file_path="/path/to/file.txt")
            >>> if result.success:
            ...     print(result.output)
        """
        start_time = time.time()

        try:
            # Validate inputs
            validated_input = self.validate_input(**kwargs)

            logger.debug(
                f"Executing tool: {self.name}",
                extra={"tool": self.name, "inputs": kwargs},
            )

            # Execute tool
            result = self._run(**validated_input.dict())

            # Add execution time to metadata
            execution_time = time.time() - start_time
            result.metadata["execution_time"] = execution_time

            logger.debug(
                f"Tool execution completed: {self.name}",
                extra={
                    "tool": self.name,
                    "success": result.success,
                    "execution_time": execution_time,
                },
            )

            return result

        except ValidationError as e:
            # Input validation failed
            error_msg = f"Input validation failed: {str(e)}"
            logger.warning(
                f"Tool validation error: {self.name}",
                extra={"tool": self.name, "error": error_msg},
            )
            return ToolResult(success=False, output="", error=error_msg)

        except Exception as e:
            # Execution failed
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(
                f"Tool execution error: {self.name}",
                extra={"tool": self.name, "error": error_msg},
                exc_info=True,
            )
            return create_tool_error(e)

    async def arun(self, **kwargs: Any) -> ToolResult:
        """
        Asynchronously execute the tool with validation and error handling.

        This method wraps _arun() with the same features as run().

        Args:
            **kwargs: Input parameters for the tool.

        Returns:
            ToolResult with execution outcome.

        Example:
            >>> tool = MyAsyncTool()
            >>> result = await tool.arun(url="https://example.com")
            >>> if result.success:
            ...     print(result.output)
        """
        start_time = time.time()

        try:
            # Validate inputs
            validated_input = self.validate_input(**kwargs)

            logger.debug(
                f"Executing tool (async): {self.name}",
                extra={"tool": self.name, "inputs": kwargs},
            )

            # Execute tool
            result = await self._arun(**validated_input.dict())

            # Add execution time to metadata
            execution_time = time.time() - start_time
            result.metadata["execution_time"] = execution_time

            logger.debug(
                f"Tool execution completed (async): {self.name}",
                extra={
                    "tool": self.name,
                    "success": result.success,
                    "execution_time": execution_time,
                },
            )

            return result

        except ValidationError as e:
            # Input validation failed
            error_msg = f"Input validation failed: {str(e)}"
            logger.warning(
                f"Tool validation error (async): {self.name}",
                extra={"tool": self.name, "error": error_msg},
            )
            return ToolResult(success=False, output="", error=error_msg)

        except NotImplementedError:
            # Async not supported
            error_msg = f"Async execution not supported for {self.name}"
            logger.warning(error_msg)
            return ToolResult(success=False, output="", error=error_msg)

        except Exception as e:
            # Execution failed
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(
                f"Tool execution error (async): {self.name}",
                extra={"tool": self.name, "error": error_msg},
                exc_info=True,
            )
            return create_tool_error(e)

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool.

        Returns a dictionary describing the tool's name, description,
        and input schema in a format suitable for LLM tool use.

        Returns:
            Dictionary with tool metadata and input schema.

        Example:
            >>> tool = MyTool()
            >>> schema = tool.get_schema()
            >>> print(schema["name"])  # "my_tool"
            >>> print(schema["description"])  # "Does something useful"
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.schema(),
            "return_direct": self.return_direct,
            "category": self.category,
        }

    def validate_input(self, **kwargs: Any) -> ToolInput:
        """
        Validate input parameters using the tool's input schema.

        Args:
            **kwargs: Input parameters to validate.

        Returns:
            Validated ToolInput instance.

        Raises:
            ValidationError: If validation fails.

        Example:
            >>> tool = MyTool()
            >>> validated = tool.validate_input(file_path="/path/to/file")
            >>> print(validated.file_path)  # "/path/to/file"
        """
        return self.input_schema(**kwargs)

    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"<{self.__class__.__name__}(name='{self.name}')>"


def create_tool_error(error: Exception) -> ToolResult:
    """
    Create a ToolResult from an exception.

    Helper function to convert exceptions into standardized ToolResult objects.

    Args:
        error: The exception that occurred.

    Returns:
        ToolResult with success=False and error message.

    Example:
        >>> try:
        ...     # Some operation
        ...     raise ValueError("Invalid input")
        ... except Exception as e:
        ...     result = create_tool_error(e)
        >>> print(result.error)  # "Invalid input"
    """
    error_type = type(error).__name__
    error_message = str(error)

    return ToolResult(
        success=False,
        output="",
        error=f"{error_type}: {error_message}",
        metadata={"error_type": error_type},
    )
