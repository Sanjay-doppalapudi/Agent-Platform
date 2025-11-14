"""
Example 2: File Operations

Demonstrates agent with file manipulation tools.
"""

from pathlib import Path
from agent_platform.agent_core.factory import create_coding_agent

def main():
    """Run file operations example."""
    
    # Create workspace
    workspace = Path("./workspace_example")
    workspace.mkdir(exist_ok=True)
    
    # Create coding agent (includes file tools)
    agent = create_coding_agent(
        workspace_path=str(workspace),
        model="gpt-3.5-turbo"
    )
    
    # Task: Create and manipulate files
    task = """
    Create a file called 'example.txt' with the following content:
    
    Hello from the Agent Platform!
    This file was created by an autonomous agent.
    
    Then read the file back to confirm it was created correctly.
    """
    
    result = agent.run(task)
    
    print("\n" + "="*60)
    print("AGENT RESULT:")
    print("="*60)
    print(result)
    print("="*60)
    
    # Verify file was created
    example_file = workspace / "example.txt"
    if example_file.exists():
        print(f"\n✓ File created successfully!")
        print(f"Content:\n{example_file.read_text()}")
    
    # Cleanup
    # workspace.rmdir()  # Uncomment to cleanup


if __name__ == "__main__":
    main()
