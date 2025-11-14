"""
Tool registry for managing and discovering tools.

This module provides a centralized registry for tools, allowing agents to
discover and use available tools dynamically. It includes search functionality
and automatic registration via decorators.

Example:
    >>> from agent_platform.tools.registry import get_registry, register_tool
    >>> from agent_platform.tools.base import BaseTool, ToolInput, ToolResult
    >>>
    >>> # Register a tool using decorator
    >>> @register_tool
    ... class MyTool(BaseTool):
    ...     name = "my_tool"
    ...     description = "Does something useful"
    ...     input_schema = ToolInput
    ...     def _run(self, **kwargs):
    ...         return ToolResult(success=True, output="Done")
    >>>
    >>> # Get tool from registry
    >>> registry = get_registry()
    >>> tool = registry.get_tool("my_tool")
"""

import re
from typing import Dict, List, Optional

from agent_platform.tools.base import BaseTool
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    Central registry for managing tools.

    The registry maintains a collection of tools and provides methods for
    registration, retrieval, search, and schema generation.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(my_tool)
        >>> tool = registry.get_tool("my_tool")
        >>> all_schemas = registry.get_tool_schemas()
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        logger.info("Initialized ToolRegistry")

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If tool name is invalid or already registered.
            TypeError: If tool doesn't inherit from BaseTool.

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(my_tool)
        """
        # Validate tool type
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Tool must inherit from BaseTool, got {type(tool).__name__}"
            )

        # Validate tool name format
        if not self._is_valid_tool_name(tool.name):
            raise ValueError(
                f"Invalid tool name '{tool.name}'. "
                "Names must be lowercase with underscores only."
            )

        # Check if already registered
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use unregister() first if you want to replace it."
            )

        # Register tool
        self._tools[tool.name] = tool

        # Add to category
        if tool.category:
            if tool.category not in self._categories:
                self._categories[tool.category] = []
            self._categories[tool.category].append(tool.name)

        logger.info(
            f"Registered tool: {tool.name}",
            extra={"tool": tool.name, "category": tool.category},
        )

    def unregister(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister.

        Example:
            >>> registry.unregister("my_tool")
        """
        if tool_name not in self._tools:
            logger.warning(f"Attempted to unregister unknown tool: {tool_name}")
            return

        tool = self._tools[tool_name]

        # Remove from category
        if tool.category and tool.category in self._categories:
            self._categories[tool.category].remove(tool_name)
            if not self._categories[tool.category]:
                del self._categories[tool.category]

        # Remove from registry
        del self._tools[tool_name]

        logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool instance, or None if not found.

        Example:
            >>> tool = registry.get_tool("read_file")
            >>> if tool:
            ...     result = tool.run(file_path="example.txt")
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all tool instances.

        Example:
            >>> all_tools = registry.get_all_tools()
            >>> for tool in all_tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of tools in the specified category.

        Example:
            >>> file_tools = registry.get_tools_by_category("file_operations")
            >>> for tool in file_tools:
            ...     print(tool.name)
        """
        if category not in self._categories:
            return []

        tool_names = self._categories[category]
        return [self._tools[name] for name in tool_names]

    def get_tool_schemas(self) -> List[Dict]:
        """
        Get schemas for all registered tools.

        Returns schemas in a format suitable for LLM tool use, including
        name, description, and input schema for each tool.

        Returns:
            List of tool schema dictionaries.

        Example:
            >>> schemas = registry.get_tool_schemas()
            >>> # Pass to LLM for tool selection
            >>> llm.completion(messages, tools=schemas)
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def search_tools(self, query: str) -> List[BaseTool]:
        """
        Search for tools by name or description.

        Performs case-insensitive substring matching on tool names
        and descriptions.

        Args:
            query: Search query string.

        Returns:
            List of matching tools.

        Example:
            >>> file_tools = registry.search_tools("file")
            >>> # Returns tools with "file" in name or description
        """
        query_lower = query.lower()
        matching_tools = []

        for tool in self._tools.values():
            # Check name
            if query_lower in tool.name.lower():
                matching_tools.append(tool)
                continue

            # Check description
            if query_lower in tool.description.lower():
                matching_tools.append(tool)

        logger.debug(
            f"Tool search for '{query}' found {len(matching_tools)} results",
            extra={"query": query, "results": len(matching_tools)},
        )

        return matching_tools

    def get_categories(self) -> List[str]:
        """
        Get all available tool categories.

        Returns:
            List of category names.

        Example:
            >>> categories = registry.get_categories()
            >>> print(categories)  # ['file_operations', 'web', 'git', ...]
        """
        return list(self._categories.keys())

    def get_tool_count(self) -> int:
        """
        Get the total number of registered tools.

        Returns:
            Number of registered tools.

        Example:
            >>> count = registry.get_tool_count()
            >>> print(f"Total tools: {count}")
        """
        return len(self._tools)

    def clear(self) -> None:
        """
        Remove all tools from the registry.

        Example:
            >>> registry.clear()
            >>> assert registry.get_tool_count() == 0
        """
        count = len(self._tools)
        self._tools.clear()
        self._categories.clear()
        logger.info(f"Cleared registry ({count} tools removed)")

    @staticmethod
    def _is_valid_tool_name(name: str) -> bool:
        """
        Validate tool name format.

        Tool names must be lowercase with underscores only.

        Args:
            name: The tool name to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Must match pattern: lowercase letters, numbers, underscores
        # Must not start with a number
        pattern = r"^[a-z][a-z0-9_]*$"
        return bool(re.match(pattern, name))


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry singleton.

    Returns:
        The global ToolRegistry instance.

    Example:
        >>> registry = get_registry()
        >>> tool = registry.get_tool("read_file")
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool_class):
    """
    Decorator for automatic tool registration.

    Use this decorator on tool classes to automatically register
    them in the global registry when they are defined.

    Args:
        tool_class: The tool class to register.

    Returns:
        The unmodified tool class.

    Example:
        >>> @register_tool
        ... class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     description = "Does something"
        ...     input_schema = ToolInput
        ...     def _run(self, **kwargs):
        ...         return ToolResult(success=True, output="Done")
        >>>
        >>> # Tool is now registered and can be retrieved
        >>> registry = get_registry()
        >>> tool = registry.get_tool("my_tool")
    """
    # Validate it's a tool class
    if not issubclass(tool_class, BaseTool):
        raise TypeError(
            f"@register_tool can only be used on BaseTool subclasses, "
            f"got {tool_class.__name__}"
        )

    # Create an instance and register it
    try:
        tool_instance = tool_class()
        registry = get_registry()
        registry.register(tool_instance)
    except Exception as e:
        logger.error(
            f"Failed to register tool {tool_class.__name__}: {e}",
            exc_info=True,
        )
        raise

    # Return the class unmodified
    return tool_class
