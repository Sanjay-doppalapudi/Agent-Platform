"""
Example 1: Simple Agent

Demonstrates basic agent creation and usage for simple tasks.
"""

from agent_platform.agent_core.factory import create_agent
from agent_platform.llm.factory import create_llm

def main():
    """Run a simple agent example."""
    
    # Create LLM adapter
    llm = create_llm("gpt-3.5-turbo", temperature=0.7)
    
    # Create empty tool registry (no tools needed for this example)
    from agent_platform.tools.registry import ToolRegistry
    registry = ToolRegistry()
    
    # Create agent
    from agent_platform.agent_core.agent import Agent, AgentConfig
    config = AgentConfig(
        max_iterations=5,
        verbose=True
    )
    agent = Agent(llm, registry, config)
    
    # Ask a simple question (no tools needed)
    result = agent.run("What is the capital of France? Just answer briefly.")
    
    print("\n" + "="*60)
    print("AGENT RESPONSE:")
    print("="*60)
    print(result)
    print("="*60)


if __name__ == "__main__":
    main()
