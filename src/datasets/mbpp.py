# src/datasets/mbpp.py
from datasets import load_dataset
from src.datasets.base import BaseDataset, Problem
from typing import Optional
import re


class MBPPDataset(BaseDataset):
    """
    MBPP (Mostly Basic Python Problems) dataset loader.
    
    Uses the sanitized version from google-research-datasets/mbpp.
    Each problem includes a task description, code solution, and test cases.
    """
    
    def __init__(self, split: str = "test"):
        super().__init__(split)
        self.dataset_name = "google-research-datasets/mbpp"
        self.config_name = "sanitized"
    
    def load(self) -> None:
        """Load the MBPP sanitized dataset from HuggingFace."""
        if self._loaded:
            return
        
        ds = load_dataset(self.dataset_name, self.config_name)
        
        # Map split names - sanitized MBPP has: train, test, validation, prompt
        split_data = ds[self.split]
        
        for idx, item in enumerate(split_data):
            # Build the prompt from the task description
            prompt = self._build_prompt(item)
            
            # Ground truth includes the code and test cases
            ground_truth = {
                "code": item["code"],
                "test_list": item["test_list"],
            }
            
            # Extract entry point (function name) from the code
            entry_point = self._extract_entry_point(item["code"])
            
            problem = Problem(
                id=f"mbpp_{item['task_id']}",
                prompt=prompt,
                ground_truth=ground_truth,
                metadata={
                    "task_id": item["task_id"],
                    "entry_point": entry_point,
                    "test_imports": item.get("test_imports", []),
                }
            )
            self.problems.append(problem)
        
        self._loaded = True
    
    def _build_prompt(self, item: dict) -> str:
        """Build a prompt from the MBPP item."""
        prompt = item["prompt"]
        
        # Add example test case to help the model understand expected format
        if item["test_list"]:
            prompt += f"\n\nExample test case:\n{item['test_list'][0]}"
        
        return prompt
    
    def _extract_entry_point(self, code: str) -> Optional[str]:
        """Extract the main function name from the code."""
        # Match function definitions
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        return None
    
    def evaluate(self, response: str, problem: Problem) -> bool:
        """
        Evaluate if the generated code passes all test cases.
        
        This method extracts code from the response and delegates
        to the code executor for actual execution.
        """
        from src.evaluation.executor import execute_code
        
        # Extract code from response (handle markdown code blocks)
        code = self._extract_code(response)
        if not code:
            return False
        
        # Get test cases
        test_cases = problem.ground_truth["test_list"]
        test_imports = problem.metadata.get("test_imports", [])
        
        # Execute and check if all tests pass
        return execute_code(code, test_cases, test_imports)
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from a response that may contain markdown."""
        # Try to find code in markdown code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Return the last code block (usually the final solution)
            return matches[-1].strip()
        
        # If no code blocks, try to find function definitions directly
        if 'def ' in response:
            # Find from first 'def' to the end or next non-code content
            lines = response.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                if in_function:
                    # Stop at obvious non-code lines
                    if line.strip() and not line.startswith((' ', '\t', 'def ', 'class ', 'import ', 'from ', '#', '@', ')')):
                        if not any(c in line for c in ['=', ':', '(', ')', '[', ']', '{', '}', '+', '-', '*', '/']):
                            break
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines).strip()
        
        return None


if __name__ == "__main__":
    # Test the dataset loader
    print("Loading MBPP dataset...")
    dataset = MBPPDataset(split="test")
    dataset.load()
    
    print(f"Loaded {len(dataset)} problems")
    
    # Show first problem
    problem = dataset[0]
    print(f"\nFirst problem:")
    print(f"ID: {problem.id}")
    print(f"Prompt: {problem.prompt[:200]}...")
    print(f"Entry point: {problem.metadata['entry_point']}")
    print(f"Test cases: {problem.ground_truth['test_list']}")


