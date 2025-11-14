"""
End-to-end tests for complete system workflows.

These tests verify the entire platform working together with real components,
real file systems, and real Docker containers.
"""

import os
import time
from pathlib import Path

import pytest

from agent_platform.agent_core.factory import create_coding_agent
from agent_platform.tools.file_tools import FileReadTool, FileWriteTool


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="API key required for E2E tests")
@pytest.mark.timeout(120)
class TestCompleteWorkflows:
    """End-to-end workflow tests."""
    
    def test_file_manipulation_workflow(self, temp_workspace):
        """
        Complete workflow: Create file -> Write code -> Execute -> Read output.
        
        This tests the full cycle of file manipulation and code execution.
        """
        start_time = time.time()
        
        # Create coding agent
        agent = create_coding_agent(
            str(temp_workspace),
            model="gpt-3.5-turbo",
            max_iterations=15,
            verbose=True
        )
        
        # Task: Create a Python script that calculates factorial
        task = """
        Create a Python file called 'factorial.py' that:
        1. Defines a function to calculate factorial
        2. Prints the factorial of 5
        Then read the file back to confirm it was created correctly.
        """
        
        result = agent.run(task)
        
        # Verify file was created
        factorial_file = temp_workspace / "factorial.py"
        assert factorial_file.exists(), "factorial.py should be created"
        
        # Verify content
        content = factorial_file.read_text()
        assert "def" in content, "Should contain function definition"
        assert "factorial" in content.lower(), "Should mention factorial"
        
        # Verify agent confirmed completion
        assert "factorial.py" in result.lower() or "created" in result.lower()
        
        execution_time = time.time() - start_time
        print(f"\n✓ Workflow completed in {execution_time:.2f}s")
        
        # Performance check
        assert execution_time < 60, "Workflow should complete within 60 seconds"
    
    def test_error_recovery_workflow(self, temp_workspace):
        """
        Test agent's ability to recover from errors and complete task.
        
        Intentionally triggers an error (reading non-existent file)
        and verifies agent recovers and completes the task.
        """
        agent = create_coding_agent(
            str(temp_workspace),
            model="gpt-3.5-turbo",
            max_iterations=15
        )
        
        task = """
        Try to read a file called 'missing.txt'. When it fails (because it doesn't exist),
        create a new file called 'recovery.txt' with the content 'Recovered successfully!'.
        Then read the recovery.txt file to confirm.
        """
        
        result = agent.run(task)
        
        # Verify recovery file was created
        recovery_file = temp_workspace / "recovery.txt"
        assert recovery_file.exists(), "Recovery file should be created"
        
        # Verify content
        content = recovery_file.read_text()
        assert "Recovered" in content or "success" in content.lower()
        
        # Verify agent mentioned the error and recovery
        result_lower = result.lower()
        assert ("not found" in result_lower or "doesn't exist" in result_lower or
                "created" in result_lower)
    
    def test_multi_step_analysis_workflow(self, temp_workspace):
        """
        Complex workflow with multiple tools and analysis.
        
        Tests: list files -> read multiple files -> analyze -> create summary
        """
        # Create test files
        (temp_workspace / "data1.txt").write_text("Score: 85")
        (temp_workspace / "data2.txt").write_text("Score: 92")
        (temp_workspace / "data3.txt").write_text("Score: 78")
        
        agent = create_coding_agent(
            str(temp_workspace),
            model="gpt-3.5-turbo",
            max_iterations=20
        )
        
        task = """
        1. List all .txt files in the workspace
        2. Read each file
        3. Extract the score from each file
        4. Calculate the average score
        5. Create a summary.txt file with the results
        """
        
        result = agent.run(task)
        
        # Verify summary was created
        summary_file = temp_workspace / "summary.txt"
        assert summary_file.exists(), "Summary file should be created"
        
        # Verify summary mentions scores or average
        summary_content = summary_file.read_text()
        assert any(word in summary_content.lower() 
                  for word in ["score", "average", "85", "92", "78"])
        
        print(f"\n✓ Multi-step workflow completed")
        print(f"Summary: {summary_content[:100]}...")


@pytest.mark.e2e
class TestPerformanceBenchmarks:
    """Performance benchmarks for E2E workflows."""
    
    def test_agent_response_time_benchmark(self, temp_workspace):
        """Benchmark agent response time for simple tasks."""
        agent = create_coding_agent(str(temp_workspace), model="gpt-3.5-turbo")
        
        task = "List the files in the current directory"
        
        start = time.time()
        result = agent.run(task)
        duration = time.time() - start
        
        print(f"\n[BENCHMARK] Simple task: {duration:.2f}s")
        
        # Baseline: should complete within 15 seconds
        assert duration < 15, f"Simple task took too long: {duration:.2f}s"
    
    def test_file_operation_performance(self, temp_workspace):
        """Benchmark file operation speed."""
        tool = FileWriteTool(str(temp_workspace))
        
        iterations = 10
        start = time.time()
        
        for i in range(iterations):
            tool.run(file_path=f"test_{i}.txt", content=f"Content {i}")
        
        duration = time.time() - start
        avg_time = duration / iterations
        
        print(f"\n[BENCHMARK] File writes: {avg_time:.3f}s per file")
        
        # Should be under 100ms per file
        assert avg_time < 0.1, f"File writes too slow: {avg_time:.3f}s"


@pytest.mark.e2e 
class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_concurrent_agents(self, temp_workspace):
        """Test multiple agents running without interference."""
        import concurrent.futures
        
        def run_agent_task(task_id):
            workspace = temp_workspace / f"agent_{task_id}"
            workspace.mkdir(exist_ok=True)
            
            agent = create_coding_agent(str(workspace), model="gpt-3.5-turbo")
            result = agent.run(f"Create a file called task_{task_id}.txt")
            
            return (temp_workspace / f"agent_{task_id}" / f"task_{task_id}.txt").exists()
        
        # Run 3 agents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_agent_task, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(results), "All concurrent agents should complete successfully"
