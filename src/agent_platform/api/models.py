"""
Pydantic models for API request/response schemas.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request to execute an agent task."""
    
    task: str = Field(..., min_length=1, max_length=5000, description="Task for the agent to complete")
    model: str = Field(default="gpt-4", description="LLM model to use")
    stream: bool = Field(default=False, description="Enable streaming responses")
    tools: Optional[List[str]] = Field(default=None, description="Specific tools to enable")
    system_message: Optional[str] = Field(default=None, description="Custom system message")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum reasoning iterations")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    
    class Config:
        schema_extra = {
            "example": {
                "task": "Create a Python file that prints 'Hello, World!'",
                "model": "gpt-4",
                "max_iterations": 10,
                "temperature": 0.7
            }
        }


class AgentResponse(BaseModel):
    """Response from agent execution."""
    
    output: str = Field(..., description="Agent's final answer")
    success: bool = Field(default=True, description="Whether task completed successfully")
    execution_time: float = Field(..., description="Total execution time in seconds")
    iterations: int = Field(..., description="Number of reasoning iterations")
    tool_calls: List[Dict] = Field(default_factory=list, description="Tools that were called")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "output": "Created hello.py with the requested content",
                "success": True,
                "execution_time": 3.45,
                "iterations": 2,
                "tool_calls": [{"tool": "write_file", "args": {"file_path": "hello.py"}}]
            }
        }


class ToolExecutionRequest(BaseModel):
    """Request to execute a specific tool."""
    
    tool_name: str = Field(..., description="Name of the tool to execute")
    tool_args: Dict[str, Any] = Field(..., description="Arguments for the tool")
    
    class Config:
        schema_extra = {
            "example": {
                "tool_name": "read_file",
                "tool_args": {"file_path": "example.txt"}
            }
        }


class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    
    success: bool = Field(..., description="Whether tool executed successfully")
    output: str = Field(..., description="Tool output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")
    active_sandboxes: int = Field(default=0, description="Number of active sandbox containers")


class StreamChunk(BaseModel):
    """Chunk in a streaming response."""
    
    type: Literal["start", "token", "tool_call", "tool_result", "end"] = Field(..., description="Chunk type")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
