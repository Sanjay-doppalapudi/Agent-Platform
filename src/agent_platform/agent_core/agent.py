"""
Core Agent orchestrator with tool execution and decision-making loop.

This module implements the main Agent class that coordinates between LLMs
and tools to accomplish tasks autonomously. The agent follows a think-act-observe
loop until the task is complete or limits are reached.

Example:
    >>> from agent_platform.llm.factory import create_llm
    >>> from agent_platform.tools.registry import get_registry
    >>> from agent_platform.agent_core.agent import Agent, AgentConfig
    >>>
    >>> # Create agent
    >>> llm = create_llm("gpt-4")
    >>> registry = get_registry()
    >>> agent = Agent(llm, registry)
    >>>
    >>> # Run task
    >>> result = agent.run("Create a file called hello.txt with 'Hello, World!'")
    >>> print(result)
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from agent_platform.agent_core.state import ConversationState
from agent_platform.llm.adapter import UniversalLLMAdapter
from agent_platform.tools.base import BaseTool, ToolResult
from agent_platform.tools.registry import ToolRegistry
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration for Agent behavior.

    Attributes:
        model: LLM model to use for decision making.
        temperature: Sampling temperature for LLM responses.
        max_iterations: Maximum number of think-act-observe loops.
        max_execution_time: Maximum total execution time in seconds.
        system_message: Optional custom system message for the agent.
        verbose: Whether to log detailed execution information.
        enable_streaming: Whether to support streaming responses.
    """

    model: str = "gpt-4"
    temperature: float = 0.7
    max_iterations: int = 10
    max_execution_time: int = 300  # seconds
    system_message: Optional[str] = None
    verbose: bool = False
    enable_streaming: bool = False


class Agent:
    """
    Autonomous agent that uses LLMs and tools to accomplish tasks.

    The agent follows a reasoning loop:
    1. Think: Analyze the current state and decide next action
    2. Act: Execute a tool or provide final answer
    3. Observe: See the result and update state
    4. Repeat until task complete or limits reached

    Example:
        >>> agent = Agent(llm_adapter, tool_registry)
        >>> result = agent.run("What files are in the current directory?")
        >>> print(result)
    """

    def __init__(
        self,
        llm_adapter: UniversalLLMAdapter,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_adapter: LLM adapter for decision making.
            tool_registry: Registry of available tools.
            config: Optional configuration (uses defaults if not provided).
        """
        self.llm = llm_adapter
        self.registry = tool_registry
        self.config = config or AgentConfig()

        # Create system message
        system_message = self.config.system_message or self._create_default_system_message()

        # Initialize conversation state
        self.state = ConversationState(system_message=system_message)

        # Initialize logger
        self.logger = get_logger(__name__)

        self.logger.info(
            "Initialized Agent",
            extra={
                "model": self.config.model,
                "max_iterations": self.config.max_iterations,
                "tools_count": self.registry.get_tool_count(),
            },
        )

    def run(self, task: str, **kwargs) -> str:
        """
        Execute a task synchronously.

        This is the main entry point for running agent tasks. The agent will
        autonomously decide which tools to use and when to provide a final answer.

        Args:
            task: The task description or user query.
            **kwargs: Additional parameters for LLM calls.

        Returns:
            The final answer or result from the agent.

        Example:
            >>> result = agent.run("Create a file named test.txt with content 'Hello'")
            >>> print(result)
        """
        start_time = time.time()

        try:
            # Add user task to conversation
            self.state.add_message("user", task)

            if self.config.verbose:
                self.logger.info(f"Starting task: {task}")

            # Main agent loop
            for iteration in range(1, self.config.max_iterations + 1):
                # Check if should continue
                if not self._should_continue(iteration, start_time):
                    warning = (
                        f"Maximum execution limits reached "
                        f"(iterations: {iteration}/{self.config.max_iterations})"
                    )
                    self.logger.warning(warning)
                    return f"Task incomplete: {warning}\n\nPartial result: Check conversation state for details."

                if self.config.verbose:
                    self.logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

                # Get messages for LLM
                messages = self.state.get_messages_for_llm(max_tokens=4000)

                # Add tool schemas to the system understanding
                tool_schemas = self._format_tool_schemas()

                # Call LLM
                self.logger.debug(f"Calling LLM with {len(messages)} messages")
                response = self.llm.completion(
                    messages,
                    temperature=kwargs.get("temperature", self.config.temperature),
                )

                assistant_message = response["content"]
                self.state.add_message("assistant", assistant_message)

                if self.config.verbose:
                    self.logger.info(f"LLM Response: {assistant_message[:200]}...")

                # Parse response to determine action
                parsed = self._parse_llm_response(assistant_message)

                if parsed["type"] == "tool_call":
                    # Execute tool
                    tool_result = self._execute_tool(
                        parsed["tool_name"], parsed["tool_args"]
                    )

                    # Add tool result to conversation
                    result_message = f"Tool '{parsed['tool_name']}' result: {tool_result.output}"
                    if not tool_result.success:
                        result_message = f"Tool '{parsed['tool_name']}' failed: {tool_result.error}"

                    self.state.add_message("tool", result_message)

                    if self.config.verbose:
                        self.logger.info(f"Tool result: {result_message[:200]}...")

                elif parsed["type"] == "final_answer":
                    # Task complete
                    final_answer = parsed["answer"]

                    if self.config.verbose:
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"Task completed in {elapsed:.2f}s after {iteration} iterations"
                        )

                    return final_answer

                else:
                    # Ambiguous response, ask for clarification
                    clarification = "I need you to either use a tool or provide a final answer. Please be explicit about your action."
                    self.state.add_message("user", clarification)

            # Max iterations reached
            return "Task incomplete: Maximum iterations reached. The agent was unable to complete the task in the allowed number of steps."

        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            return f"Error during task execution: {str(e)}"

    async def arun(self, task: str, **kwargs) -> str:
        """
        Execute a task asynchronously.

        Similar to run() but uses async tool execution and LLM calls
        where available for better performance.

        Args:
            task: The task description or user query.
            **kwargs: Additional parameters for LLM calls.

        Returns:
            The final answer or result from the agent.

        Example:
            >>> result = await agent.arun("List all Python files in the project")
        """
        start_time = time.time()

        try:
            self.state.add_message("user", task)

            if self.config.verbose:
                self.logger.info(f"Starting async task: {task}")

            for iteration in range(1, self.config.max_iterations + 1):
                if not self._should_continue(iteration, start_time):
                    return f"Task incomplete: Maximum execution limits reached"

                if self.config.verbose:
                    self.logger.info(f"Async iteration {iteration}/{self.config.max_iterations}")

                messages = self.state.get_messages_for_llm(max_tokens=4000)
                tool_schemas = self._format_tool_schemas()

                # Note: LiteLLM completion is sync, so we use it directly
                # In production, you might want to use httpx async client
                response = self.llm.completion(
                    messages,
                    temperature=kwargs.get("temperature", self.config.temperature),
                )

                assistant_message = response["content"]
                self.state.add_message("assistant", assistant_message)

                parsed = self._parse_llm_response(assistant_message)

                if parsed["type"] == "tool_call":
                    # Execute tool asynchronously
                    tool = self.registry.get_tool(parsed["tool_name"])
                    if tool is None:
                        error_msg = f"Tool '{parsed['tool_name']}' not found"
                        self.state.add_message("tool", f"Error: {error_msg}")
                        continue

                    try:
                        tool_result = await tool.arun(**parsed["tool_args"])
                    except NotImplementedError:
                        # Fall back to sync if async not supported
                        tool_result = tool.run(**parsed["tool_args"])

                    self.state.add_tool_call(
                        parsed["tool_name"],
                        parsed["tool_args"],
                        tool_result,
                        tool_result.metadata.get("execution_time", 0.0),
                    )

                    result_message = f"Tool '{parsed['tool_name']}' result: {tool_result.output}"
                    if not tool_result.success:
                        result_message = f"Tool '{parsed['tool_name']}' failed: {tool_result.error}"

                    self.state.add_message("tool", result_message)

                elif parsed["type"] == "final_answer":
                    return parsed["answer"]

            return "Task incomplete: Maximum iterations reached"

        except Exception as e:
            self.logger.error(f"Async agent execution failed: {str(e)}", exc_info=True)
            return f"Error during task execution: {str(e)}"

    def stream(self, task: str, **kwargs) -> Iterator[str]:
        """
        Execute a task with streaming responses.

        Yields intermediate results and thinking process as they occur,
        providing real-time feedback.

        Args:
            task: The task description or user query.
            **kwargs: Additional parameters for LLM calls.

        Yields:
            Status updates, tool results, and final answer chunks.

        Example:
            >>> for chunk in agent.stream("Analyze the project structure"):
            ...     print(chunk, end="", flush=True)
        """
        start_time = time.time()

        try:
            self.state.add_message("user", task)
            yield f"[AGENT] Starting task: {task}\n\n"

            for iteration in range(1, self.config.max_iterations + 1):
                if not self._should_continue(iteration, start_time):
                    yield "\n[AGENT] Maximum execution limits reached.\n"
                    return

                yield f"[AGENT] Iteration {iteration}...\n"

                messages = self.state.get_messages_for_llm(max_tokens=4000)

                # Stream LLM response
                yield "[THINKING] "
                full_response = ""
                for chunk in self.llm.stream_completion(messages):
                    full_response += chunk
                    yield chunk

                yield "\n\n"

                self.state.add_message("assistant", full_response)

                parsed = self._parse_llm_response(full_response)

                if parsed["type"] == "tool_call":
                    yield f"[TOOL] Executing: {parsed['tool_name']}\n"

                    tool_result = self._execute_tool(
                        parsed["tool_name"], parsed["tool_args"]
                    )

                    if tool_result.success:
                        yield f"[RESULT] {tool_result.output[:500]}\n\n"
                    else:
                        yield f"[ERROR] {tool_result.error}\n\n"

                    result_message = f"Tool result: {tool_result.output if tool_result.success else tool_result.error}"
                    self.state.add_message("tool", result_message)

                elif parsed["type"] == "final_answer":
                    yield f"[ANSWER] {parsed['answer']}\n"
                    return

            yield "\n[AGENT] Task incomplete: Maximum iterations reached.\n"

        except Exception as e:
            self.logger.error(f"Streaming execution failed: {str(e)}", exc_info=True)
            yield f"\n[ERROR] {str(e)}\n"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to determine the next action.

        Analyzes the LLM's response to identify:
        - Tool calls with tool name and arguments
        - Final answers
        - Ambiguous responses

        Args:
            response: The LLM's text response.

        Returns:
            Dictionary with:
            - type: "tool_call", "final_answer", or "ambiguous"
            - tool_name: Name of tool (if tool_call)
            - tool_args: Arguments dict (if tool_call)
            - answer: Final answer text (if final_answer)

        Example:
            >>> parsed = agent._parse_llm_response("I'll use read_file with path='test.txt'")
            >>> print(parsed["type"])  # "tool_call"
        """
        # Look for tool call patterns
        # Pattern 1: Explicit tool call format
        tool_pattern = r"(?:use|call|execute)\s+(\w+)(?:\s+with\s+|\s*\()(.+?)(?:\)|$)"
        match = re.search(tool_pattern, response, re.IGNORECASE)

        if match:
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            tool_args = self._parse_tool_arguments(args_str)

            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_args": tool_args,
            }

        # Pattern 2: JSON format tool call
        json_pattern = r"\{[^}]*\"tool\":\s*\"([^\"]+)\"[^}]*\}"
        match = re.search(json_pattern, response, re.IGNORECASE)

        if match:
            try:
                tool_call = json.loads(match.group(0))
                return {
                    "type": "tool_call",
                    "tool_name": tool_call.get("tool", ""),
                    "tool_args": tool_call.get("args", {}),
                }
            except json.JSONDecodeError:
                pass

        # Pattern 3: Check for final answer indicators
        final_answer_indicators = [
            "final answer:",
            "answer:",
            "result:",
            "conclusion:",
            "therefore",
        ]

        response_lower = response.lower()
        for indicator in final_answer_indicators:
            if indicator in response_lower:
                # Extract answer after indicator
                idx = response_lower.index(indicator)
                answer = response[idx + len(indicator):].strip()
                return {"type": "final_answer", "answer": answer if answer else response}

        # Pattern 4: If response is short and doesn't mention tools, treat as final answer
        if len(response.split()) < 50 and "tool" not in response_lower:
            return {"type": "final_answer", "answer": response}

        # Ambiguous response
        return {"type": "ambiguous", "response": response}

    def _parse_tool_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse tool arguments from string.

        Args:
            args_str: String containing arguments (e.g., "path='file.txt', encoding='utf-8'")

        Returns:
            Dictionary of parsed arguments.
        """
        args = {}

        # Try to parse as JSON first
        try:
            args = json.loads(args_str)
            return args
        except json.JSONDecodeError:
            pass

        # Parse key=value format
        arg_pattern = r"(\w+)\s*=\s*['\"]?([^,'\"]+)['\"]?"
        matches = re.findall(arg_pattern, args_str)

        for key, value in matches:
            # Try to convert to appropriate type
            if value.lower() == "true":
                args[key] = True
            elif value.lower() == "false":
                args[key] = False
            elif value.isdigit():
                args[key] = int(value)
            else:
                args[key] = value.strip("'\"")

        return args

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool and track the result.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments to pass to the tool.

        Returns:
            ToolResult from the tool execution.
        """
        # Get tool from registry
        tool = self.registry.get_tool(tool_name)

        if tool is None:
            error_msg = f"Tool '{tool_name}' not found in registry"
            self.logger.warning(error_msg)
            return ToolResult(success=False, output="", error=error_msg)

        # Execute tool
        try:
            self.logger.info(
                f"Executing tool: {tool_name}",
                extra={"tool": tool_name, "args": tool_args},
            )

            start_time = time.time()
            result = tool.run(**tool_args)
            execution_time = time.time() - start_time

            # Add to conversation state
            self.state.add_tool_call(tool_name, tool_args, result, execution_time)

            self.logger.info(
                f"Tool execution completed: {tool_name}",
                extra={
                    "tool": tool_name,
                    "success": result.success,
                    "execution_time": execution_time,
                },
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Tool execution failed: {tool_name}",
                extra={"tool": tool_name, "error": str(e)},
                exc_info=True,
            )
            return ToolResult(
                success=False, output="", error=f"Tool execution error: {str(e)}"
            )

    def _should_continue(self, iteration: int, start_time: float) -> bool:
        """
        Check if the agent loop should continue.

        Args:
            iteration: Current iteration number.
            start_time: Timestamp when execution started.

        Returns:
            True if should continue, False if limits reached.
        """
        # Check iteration limit
        if iteration > self.config.max_iterations:
            return False

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > self.config.max_execution_time:
            return False

        return True

    def _format_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Format tool schemas for LLM consumption.

        Returns:
            List of tool schema dictionaries.
        """
        return self.registry.get_tool_schemas()

    def _create_default_system_message(self) -> str:
        """
        Create the default system message with tool instructions.

        Returns:
            System message string.
        """
        tools = self.registry.get_all_tools()
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )

        return f"""You are an autonomous AI agent capable of using tools to accomplish tasks.

Available Tools:
{tool_descriptions}

Instructions:
1. Analyze the user's task carefully
2. Break down complex tasks into steps
3. Use tools when needed by stating: "I'll use <tool_name> with <arguments>"
4. Wait for tool results before proceeding
5. Provide a final answer when the task is complete

When using tools, be explicit:
- Example: "I'll use read_file with file_path='example.txt'"
- Example: "I'll use write_file with file_path='output.txt' and content='Hello World'"

When you have completed the task, provide your final answer clearly.
"""

    def get_state(self) -> ConversationState:
        """
        Get the current conversation state.

        Returns:
            The ConversationState object.
        """
        return self.state

    def reset(self) -> None:
        """
        Reset the agent to initial state.

        Clears conversation history while preserving system message.
        """
        self.state.clear()
        self.logger.info("Agent state reset")
