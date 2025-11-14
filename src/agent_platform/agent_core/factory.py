"""
Factory and builder patterns for creating agents with preset configurations.

This module provides convenient ways to create agents with different
tool configurations and settings using a fluent builder API.

Example:
    >>> from agent_platform.agent_core.factory import AgentBuilder
    >>>
    >>> agent = (
    ...     AgentBuilder()
    ...     .with_llm("gpt-4")
    ...     .with_tools(FileReadTool(), FileWriteTool())
    ...     .with_system_message("You are a file management assistant.")
    ...     .build()
    ... )
    >>>
    >>> result = agent.run("Read the README.md file")
"""

from pathlib import Path
from typing import List, Optional

from agent_platform.agent_core.agent import Agent, AgentConfig
from agent_platform.llm.factory import create_llm
from agent_platform.tools.base import BaseTool
from agent_platform.tools.file_tools import FileListTool, FileReadTool, FileWriteTool
from agent_platform.tools.registry import ToolRegistry
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class AgentBuilder:
    """
    Fluent builder for creating agents with custom configurations.

    Provides a chainable API for configuring all aspects of an agent
    before creation.

    Example:
        >>> agent = (
        ...     AgentBuilder()
        ...     .with_llm("gpt-4", temperature=0.7)
        ...     .with_tools(read_tool, write_tool)
        ...     .with_config(max_iterations=15)
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize an empty builder."""
        self._llm_model: str = "gpt-4"
        self._llm_kwargs: dict = {}
        self._tools: List[BaseTool] = []
        self._tool_categories: List[str] = []
        self._system_message: Optional[str] = None
        self._config_kwargs: dict = {}
        self._workspace_path: Optional[str] = None

    def with_llm(self, model: str, **kwargs) -> "AgentBuilder":
        """
        Configure the LLM model.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet").
            **kwargs: Additional LLM configuration (temperature, max_tokens, etc.).

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_llm("gpt-4", temperature=0.8, max_tokens=2000)
        """
        self._llm_model = model
        self._llm_kwargs = kwargs
        return self

    def with_tools(self, *tools: BaseTool) -> "AgentBuilder":
        """
        Add specific tools to the agent.

        Args:
            *tools: Tool instances to add.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_tools(FileReadTool(), FileWriteTool())
        """
        self._tools.extend(tools)
        return self

    def with_tool_categories(self, *categories: str) -> "AgentBuilder":
        """
        Add all tools from specified categories.

        Args:
            *categories: Category names to include.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_tool_categories("file_operations", "web")
        """
        self._tool_categories.extend(categories)
        return self

    def with_system_message(self, message: str) -> "AgentBuilder":
        """
        Set a custom system message.

        Args:
            message: The system message to use.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_system_message("You are a Python expert assistant.")
        """
        self._system_message = message
        return self

    def with_config(self, **config_kwargs) -> "AgentBuilder":
        """
        Set agent configuration options.

        Args:
            **config_kwargs: Configuration parameters for AgentConfig.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_config(
            ...     max_iterations=20,
            ...     max_execution_time=600,
            ...     verbose=True
            ... )
        """
        self._config_kwargs.update(config_kwargs)
        return self

    def with_workspace(self, workspace_path: str) -> "AgentBuilder":
        """
        Set workspace path for file operations.

        Args:
            workspace_path: Path to the workspace directory.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_workspace("/path/to/workspace")
        """
        self._workspace_path = workspace_path
        return self

    def build(self) -> Agent:
        """
        Build and return the configured agent.

        Returns:
            Configured Agent instance.

        Raises:
            ValueError: If required configuration is missing.

        Example:
            >>> agent = builder.build()
            >>> result = agent.run("Your task here")
        """
        # Create LLM adapter
        llm = create_llm(self._llm_model, **self._llm_kwargs)

        # Create tool registry
        registry = ToolRegistry()

        # Add individual tools
        for tool in self._tools:
            try:
                registry.register(tool)
            except ValueError as e:
                logger.warning(f"Failed to register tool: {e}")

        # Add tools from categories
        # Note: This requires tools to be pre-registered in a global registry
        # For now, we skip this as we don't have a global tool registry yet

        # Create agent config
        config = AgentConfig(
            model=self._llm_model,
            system_message=self._system_message,
            **self._config_kwargs,
        )

        # Create agent
        agent = Agent(llm, registry, config)

        logger.info(
            f"Created agent with {registry.get_tool_count()} tools",
            extra={"model": self._llm_model, "tools": registry.get_tool_count()},
        )

        return agent


def create_agent(
    model: str = "gpt-4",
    tools: Optional[List[BaseTool]] = None,
    workspace_path: Optional[str] = None,
    **kwargs,
) -> Agent:
    """
    Quick agent creation with sensible defaults.

    Args:
        model: LLM model to use.
        tools: Optional list of tools to include.
        workspace_path: Optional workspace path for file tools.
        **kwargs: Additional configuration options.

    Returns:
        Configured Agent instance.

    Example:
        >>> agent = create_agent("gpt-4", tools=[file_read_tool])
        >>> result = agent.run("Read config.json")
    """
    builder = AgentBuilder().with_llm(model)

    if tools:
        builder.with_tools(*tools)

    if workspace_path:
        builder.with_workspace(workspace_path)

    if kwargs:
        builder.with_config(**kwargs)

    return builder.build()


def create_coding_agent(
    workspace_path: str = ".", model: str = "gpt-4", **kwargs
) -> Agent:
    """
    Create an agent optimized for coding tasks.

    Includes file operations, git tools, and shell command execution.

    Args:
        workspace_path: Path to the code workspace.
        model: LLM model to use.
        **kwargs: Additional configuration options.

    Returns:
        Agent configured for coding tasks.

    Example:
        >>> agent = create_coding_agent("/path/to/project")
        >>> result = agent.run("Create a Python script that prints 'Hello World'")
    """
    workspace = Path(workspace_path).resolve()

    # Create file operation tools
    tools = [
        FileReadTool(str(workspace)),
        FileWriteTool(str(workspace)),
        FileListTool(str(workspace)),
    ]

    # Add git tools if available
    try:
        from agent_platform.tools.git_tools import (
            GitStatusTool,
            GitDiffTool,
            GitLogTool,
        )

        tools.extend(
            [
                GitStatusTool(str(workspace)),
                GitDiffTool(str(workspace)),
                GitLogTool(str(workspace)),
            ]
        )
    except ImportError:
        logger.warning("Git tools not available")

    system_message = """You are an expert coding assistant with access to file operations and git tools.

Your capabilities:
- Read and write files
- List directory contents
- View git status and diffs
- Analyze code and suggest improvements
- Create and modify code files

Best practices:
- Always read files before modifying them
- Use clear, descriptive variable and function names
- Add comments for complex logic
- Follow the project's existing code style
- Test changes when possible
"""

    return (
        AgentBuilder()
        .with_llm(model)
        .with_tools(*tools)
        .with_system_message(system_message)
        .with_workspace(str(workspace))
        .with_config(**kwargs)
        .build()
    )


def create_research_agent(model: str = "gpt-4", **kwargs) -> Agent:
    """
    Create an agent optimized for research and information gathering.

    Includes web search and URL fetching capabilities.

    Args:
        model: LLM model to use.
        **kwargs: Additional configuration options.

    Returns:
        Agent configured for research tasks.

    Example:
        >>> agent = create_research_agent()
        >>> result = agent.run("Research the latest developments in AI")
    """
    tools = []

    # Add web tools if available
    try:
        from agent_platform.tools.web_tools import WebSearchTool, WebFetchTool

        tools.extend([WebSearchTool(), WebFetchTool()])
    except ImportError:
        logger.warning("Web tools not available")

    system_message = """You are a research assistant with web access.

Your capabilities:
- Search the web for information
- Fetch and analyze web pages
- Synthesize information from multiple sources
- Provide well-researched answers with sources

Best practices:
- Verify information from multiple sources
- Cite sources in your answers
- Distinguish facts from opinions
- Provide balanced perspectives
- Summarize complex information clearly
"""

    return (
        AgentBuilder()
        .with_llm(model)
        .with_tools(*tools)
        .with_system_message(system_message)
        .with_config(**kwargs)
        .build()
    )


def create_general_agent(
    workspace_path: str = ".", model: str = "gpt-4", **kwargs
) -> Agent:
    """
    Create a general-purpose agent with a balanced tool set.

    Includes file operations, web access, and command execution.

    Args:
        workspace_path: Path to the workspace.
        model: LLM model to use.
        **kwargs: Additional configuration options.

    Returns:
        Agent configured for general tasks.

    Example:
        >>> agent = create_general_agent()
        >>> result = agent.run("Create a summary of the README file")
    """
    workspace = Path(workspace_path).resolve()

    # File operation tools
    tools = [
        FileReadTool(str(workspace)),
        FileWriteTool(str(workspace)),
        FileListTool(str(workspace)),
    ]

    # Add web tools if available
    try:
        from agent_platform.tools.web_tools import WebSearchTool, WebFetchTool

        tools.extend([WebSearchTool(), WebFetchTool()])
    except ImportError:
        logger.warning("Web tools not available")

    # Add command tools if available
    try:
        from agent_platform.tools.command_tool import SafeShellCommandTool

        tools.append(SafeShellCommandTool(str(workspace)))
    except ImportError:
        logger.warning("Command tools not available")

    system_message = """You are a versatile AI assistant with multiple capabilities.

Your capabilities:
- File operations (read, write, list)
- Web search and research
- Command execution
- Information synthesis
- Problem solving

Best practices:
- Choose the right tool for each task
- Break complex tasks into steps
- Verify important information
- Provide clear, helpful responses
- Ask for clarification when needed
"""

    return (
        AgentBuilder()
        .with_llm(model)
        .with_tools(*tools)
        .with_system_message(system_message)
        .with_workspace(str(workspace))
        .with_config(**kwargs)
        .build()
    )
