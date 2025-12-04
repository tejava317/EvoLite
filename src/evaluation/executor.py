# src/evaluation/executor.py
"""
Sandboxed Python code executor using subprocess with timeout.

Provides safe execution of generated code against test cases.
"""

import subprocess
import tempfile
import os
from typing import List, Optional, Tuple


def execute_code(
    code: str,
    test_cases: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: int = 5
) -> bool:
    """
    Execute Python code with test cases in a sandboxed subprocess.
    
    Args:
        code: The Python code to execute (function definition)
        test_cases: List of test case assertions (e.g., "assert func(1) == 2")
        test_imports: Optional list of import statements needed for tests
        timeout: Maximum execution time in seconds
        
    Returns:
        True if all test cases pass, False otherwise
    """
    if not code or not test_cases:
        return False
    
    # Build the complete test script
    script = _build_test_script(code, test_cases, test_imports)
    
    # Execute in subprocess
    success, output = _run_in_subprocess(script, timeout)
    
    return success


def execute_code_with_output(
    code: str,
    test_cases: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: int = 5
) -> Tuple[bool, str]:
    """
    Execute code and return both success status and output/error message.
    
    Useful for debugging and detailed error reporting.
    """
    if not code or not test_cases:
        return False, "Empty code or test cases"
    
    script = _build_test_script(code, test_cases, test_imports)
    success, output = _run_in_subprocess(script, timeout)
    
    return success, output


def _build_test_script(
    code: str,
    test_cases: List[str],
    test_imports: Optional[List[str]] = None
) -> str:
    """Build a complete Python script with code and test assertions."""
    parts = []
    
    # Add common imports that might be needed
    parts.append("import sys")
    parts.append("import math")
    parts.append("from typing import List, Dict, Tuple, Optional, Any")
    
    # Add test-specific imports
    if test_imports:
        for imp in test_imports:
            parts.append(imp)
    
    parts.append("")  # Empty line
    
    # Add the generated code
    parts.append("# Generated code")
    parts.append(code)
    parts.append("")  # Empty line
    
    # Add test cases
    parts.append("# Test cases")
    for i, test in enumerate(test_cases):
        # Wrap each test in a try-except to get better error info
        parts.append(f"try:")
        parts.append(f"    {test}")
        parts.append(f"    print('Test {i+1} passed')")
        parts.append(f"except AssertionError as e:")
        parts.append(f"    print(f'Test {i+1} failed: {{e}}')")
        parts.append(f"    sys.exit(1)")
        parts.append(f"except Exception as e:")
        parts.append(f"    print(f'Test {i+1} error: {{type(e).__name__}}: {{e}}')")
        parts.append(f"    sys.exit(1)")
        parts.append("")
    
    parts.append("print('All tests passed!')")
    
    return "\n".join(parts)


def _run_in_subprocess(script: str, timeout: int) -> Tuple[bool, str]:
    """
    Run a Python script in a subprocess with timeout.
    
    Returns:
        Tuple of (success, output_or_error)
    """
    # Create a temporary file for the script
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(script)
        script_path = f.name
    
    try:
        # Run the script in a subprocess
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                'PYTHONDONTWRITEBYTECODE': '1',  # Don't create .pyc files
            }
        )
        
        # Check if execution was successful
        if result.returncode == 0:
            return True, result.stdout
        else:
            # Combine stdout and stderr for error info
            error_msg = result.stdout + result.stderr
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(script_path)
        except:
            pass


def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if Python code has valid syntax without executing it.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


if __name__ == "__main__":
    # Test the executor
    print("Testing code executor...")
    
    # Simple test case
    code = """
def add(a, b):
    return a + b
"""
    tests = [
        "assert add(1, 2) == 3",
        "assert add(0, 0) == 0",
        "assert add(-1, 1) == 0",
    ]
    
    print("\nTest 1: Simple addition function")
    success, output = execute_code_with_output(code, tests)
    print(f"Success: {success}")
    print(f"Output: {output}")
    
    # Test with failing code
    bad_code = """
def add(a, b):
    return a - b  # Bug!
"""
    
    print("\nTest 2: Buggy function")
    success, output = execute_code_with_output(bad_code, tests)
    print(f"Success: {success}")
    print(f"Output: {output}")
    
    # Test with timeout
    infinite_code = """
def infinite():
    while True:
        pass
"""
    
    print("\nTest 3: Infinite loop (should timeout)")
    success, output = execute_code_with_output(
        infinite_code, 
        ["infinite()"],
        timeout=2
    )
    print(f"Success: {success}")
    print(f"Output: {output}")


