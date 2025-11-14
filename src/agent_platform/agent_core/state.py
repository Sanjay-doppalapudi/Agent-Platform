"""
Conversation state management for agents.

This module provides classes for managing agent conversation state,
including messages, tool calls, and metadata tracking.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from agent_platform.tools.base import ToolResult
from agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        metadata: Additional metadata about the message.
        timestamp: When the message was created.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return Message(**data)


@dataclass
class ToolCall:
    """
    Represents a tool execution in the conversation.
    
    Attributes:
        tool_name: Name of the tool that was called.
        tool_input: Input parameters passed to the tool.
        tool_output: Result returned by the tool.
        execution_time: How long the tool took to execute (seconds).
    """
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: ToolResult
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": {
                "success": self.tool_output.success,
                "output": self.tool_output.output,
                "error": self.tool_output.error,
                "metadata": self.tool_output.metadata,
            },
            "execution_time": self.execution_time,
        }


class ConversationState:
    """
    Manages the state of an agent conversation.
    
    Tracks messages, tool calls, and metadata throughout a conversation.
    Provides methods for adding messages, retrieving history, and exporting state.
    
    Example:
        >>> state = ConversationState(system_message="You are a helpful assistant.")
        >>> state.add_message("user", "Hello!")
        >>> state.add_message("assistant", "Hi there!")
        >>> messages = state.get_messages_for_llm()
    """
    
    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize conversation state.
        
        Args:
            system_message: Optional system message to set context.
        """
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCall] = []
        self.metadata: Dict[str, Any] = {
            "total_tokens": 0,
            "start_time": datetime.now(),
        }
        
        # Add system message if provided
        if system_message:
            self.add_message("system", system_message)
        
        logger.info("Initialized ConversationState")
    
    def add_message(
        self, 
        role: Literal["system", "user", "assistant", "tool"], 
        content: str, 
        **metadata
    ) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: Role of the message sender.
            content: Message content.
            **metadata: Additional metadata to attach to the message.
        
        Example:
            >>> state.add_message("user", "What is Python?", intent="question")
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )
        
        self.messages.append(message)
        
        # Update token count (rough estimate)
        tokens = estimate_tokens(content)
        self.metadata["total_tokens"] += tokens
        
        logger.debug(
            f"Added message",
            extra={"role": role, "tokens": tokens, "total_messages": len(self.messages)}
        )
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: ToolResult,
        execution_time: float,
    ) -> None:
        """
        Record a tool call in the conversation.
        
        Args:
            tool_name: Name of the tool that was called.
            tool_input: Input parameters passed to the tool.
            tool_output: Result returned by the tool.
            execution_time: Execution time in seconds.
        
        Example:
            >>> state.add_tool_call(
            ...     "read_file",
            ...     {"file_path": "example.txt"},
            ...     tool_result,
            ...     0.05
            ... )
        """
        tool_call = ToolCall(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            execution_time=execution_time,
        )
        
        self.tool_calls.append(tool_call)
        
        logger.debug(
            f"Added tool call",
            extra={
                "tool": tool_name,
                "success": tool_output.success,
                "execution_time": execution_time,
            }
        )
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get conversation messages.
        
        Args:
            limit: Optional limit on number of recent messages to return.
        
        Returns:
            List of messages (most recent if limit specified).
        
        Example:
            >>> recent = state.get_messages(limit=10)
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]
    
    def get_messages_for_llm(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API calls.
        
        Converts messages to the format expected by LLM APIs and
        truncates if necessary to fit within token limit.
        
        Args:
            max_tokens: Maximum tokens to include (truncates old messages).
        
        Returns:
            List of message dictionaries with 'role' and 'content'.
        
        Example:
            >>> llm_messages = state.get_messages_for_llm(max_tokens=2000)
            >>> response = llm.completion(llm_messages)
        """
        # Always include system message if present
        system_messages = [m for m in self.messages if m.role == "system"]
        other_messages = [m for m in self.messages if m.role != "system"]
        
        # Calculate tokens for system messages
        system_tokens = sum(estimate_tokens(m.content) for m in system_messages)
        remaining_tokens = max_tokens - system_tokens
        
        # Truncate from oldest to newest to fit token limit
        included_messages = []
        current_tokens = 0
        
        for message in reversed(other_messages):
            msg_tokens = estimate_tokens(message.content)
            if current_tokens + msg_tokens > remaining_tokens:
                break
            included_messages.insert(0, message)
            current_tokens += msg_tokens
        
        # Combine system and included messages
        final_messages = system_messages + included_messages
        
        # Convert to LLM format
        return [
            {"role": msg.role, "content": msg.content}
            for msg in final_messages
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the conversation.
        
        Returns:
            Dictionary with conversation statistics.
        
        Example:
            >>> summary = state.get_summary()
            >>> print(f"Total messages: {summary['total_messages']}")
        """
        duration = datetime.now() - self.metadata["start_time"]
        
        return {
            "total_messages": len(self.messages),
            "total_tool_calls": len(self.tool_calls),
            "total_tokens": self.metadata["total_tokens"],
            "duration_seconds": duration.total_seconds(),
            "start_time": self.metadata["start_time"].isoformat(),
        }
    
    def clear(self) -> None:
        """
        Clear all messages except system message.
        
        Resets the conversation while preserving the system context.
        
        Example:
            >>> state.clear()
            >>> # System message is preserved, all others removed
        """
        # Keep system messages
        system_messages = [m for m in self.messages if m.role == "system"]
        
        self.messages = system_messages
        self.tool_calls.clear()
        self.metadata["total_tokens"] = sum(
            estimate_tokens(m.content) for m in system_messages
        )
        
        logger.info("Cleared conversation state (kept system messages)")
    
    def export_to_json(self) -> str:
        """
        Export conversation state to JSON.
        
        Returns:
            JSON string representation of the conversation.
        
        Example:
            >>> json_data = state.export_to_json()
            >>> with open("conversation.json", "w") as f:
            ...     f.write(json_data)
        """
        data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "metadata": {
                **self.metadata,
                "start_time": self.metadata["start_time"].isoformat(),
            },
            "summary": self.get_summary(),
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_from_json(json_str: str) -> "ConversationState":
        """
        Load conversation state from JSON.
        
        Args:
            json_str: JSON string representation of a conversation.
        
        Returns:
            Reconstructed ConversationState object.
        
        Example:
            >>> with open("conversation.json") as f:
            ...     json_data = f.read()
            >>> state = ConversationState.load_from_json(json_data)
        """
        data = json.loads(json_str)
        
        # Create new state
        state = ConversationState()
        
        # Restore messages
        state.messages = [Message.from_dict(msg) for msg in data["messages"]]
        
        # Restore metadata
        state.metadata = data["metadata"].copy()
        if "start_time" in state.metadata and isinstance(state.metadata["start_time"], str):
            state.metadata["start_time"] = datetime.fromisoformat(state.metadata["start_time"])
        
        # Tool calls are not restored (they're historical)
        
        return state


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses rough approximation of 1 token ≈ 4 characters.
    This is not exact but good enough for truncation purposes.
    
    Args:
        text: The text to estimate tokens for.
    
    Returns:
        Estimated token count.
    
    Example:
        >>> count = estimate_tokens("Hello, world!")
        >>> print(count)  # approximately 3
    """
    return len(text) // 4
