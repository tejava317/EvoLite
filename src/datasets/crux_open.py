# src/datasets/crux_open.py
"""
CRUX-O (Output Prediction) Dataset Loader

Dataset: cruxeval-org/cruxeval
Task: Given Python code and input, predict the output.
Uses direct output format: assert f(input) == output
"""
from datasets import load_dataset
from src.datasets.base import BaseDataset, Problem
from typing import Optional
import re
import ast


class CRUXOpenDataset(BaseDataset):
    """
    CRUX-O dataset loader for output prediction.

    Dataset structure:
    - code: Python function definition
    - input: Input arguments to the function
    - output: Expected output when running code(input)
    - id: Sample identifier (e.g., sample_0)

    The task is to predict what output the code produces given the input.
    Expected response format: assert f(input) == output
    """

    def __init__(self, split: str = "test"):
        super().__init__(split)
        self.dataset_name = "cruxeval-org/cruxeval"

    def load(self) -> None:
        """Load the CRUX-O dataset from HuggingFace."""
        if self._loaded:
            return

        ds = load_dataset(self.dataset_name)
        split_data = ds[self.split]

        for item in split_data:
            code = item.get("code", "")
            input_val = item.get("input", "")
            output_val = item.get("output", "")
            sample_id = item.get("id", "")

            # Build prompt for direct output prediction
            prompt = self._build_direct_output_prompt(code, input_val)

            problem = Problem(
                id=f"crux_o_{sample_id}",
                prompt=prompt,
                ground_truth={
                    "code": code,
                    "input": input_val,
                    "output": output_val,
                },
                metadata={
                    "sample_id": sample_id,
                },
            )
            self.problems.append(problem)

        self._loaded = True

    def _build_direct_output_prompt(self, code: str, input_val: str) -> str:
        """
        Build a direct output prediction prompt.
        Format matches CRUXEval's make_direct_output_prompt.
        """
        return f"""Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete.

{code}
assert f({input_val}) =="""

    def _extract_direct_output(self, response: str) -> str:
        """
        Extract the predicted output using direct output format.
        Matches CRUXEval's extract_answer_direct_output.
        """
        gen = response.strip()
        
        # If response contains ==, take the part after it
        if "==" in gen:
            gen = gen.split("==")[1]
        
        return gen.strip()

    def _normalize_value(self, value: str) -> str:
        """Normalize a value for comparison."""
        value = value.strip()
        # Try to parse as Python literal for consistent formatting
        try:
            parsed = ast.literal_eval(value)
            return repr(parsed)
        except (ValueError, SyntaxError):
            return value

    def evaluate(self, response: str, problem: Problem) -> bool:
        """
        Evaluate by comparing predicted output with expected output.
        Uses direct output extraction matching CRUXEval format.
        """
        expected_output = problem.ground_truth["output"]

        # Extract predicted output from response
        predicted = self._extract_direct_output(response)

        # Method 1: Direct string comparison (normalized)
        if self._normalize_value(predicted) == self._normalize_value(expected_output):
            return True

        # Method 2: Try parsing both as Python objects and compare
        try:
            pred_obj = ast.literal_eval(predicted)
            exp_obj = ast.literal_eval(expected_output)
            if pred_obj == exp_obj:
                return True
        except (ValueError, SyntaxError):
            pass

        # Method 3: Execute code and compare (fallback verification)
        try:
            code = problem.ground_truth["code"]
            input_val = problem.ground_truth["input"]
            actual_output = self._execute_code(code, input_val)
            if actual_output is not None:
                # Compare predicted with actual execution result
                if self._normalize_value(predicted) == self._normalize_value(actual_output):
                    return True
                # Also try object comparison
                try:
                    pred_obj = ast.literal_eval(predicted)
                    actual_obj = ast.literal_eval(actual_output)
                    if pred_obj == actual_obj:
                        return True
                except (ValueError, SyntaxError):
                    pass
        except Exception:
            pass

        return False

    def _execute_code(self, code: str, input_val: str) -> Optional[str]:
        """Execute the code with the given input and return the output."""
        import subprocess
        import tempfile
        import os

        # Build the execution script
        exec_script = f"""{code}

# Execute with input
result = f({input_val})
print(repr(result))
"""

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(exec_script)
                temp_path = f.name

            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=5,
            )

            os.unlink(temp_path)

            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None
