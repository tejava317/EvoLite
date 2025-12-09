# src/datasets/math_algebra.py
from datasets import load_dataset
from src.datasets.base import BaseDataset, Problem
from typing import Optional
import re


class MathAlgebraDataset(BaseDataset):
    """
    Hendrycks MATH dataset loader (all categories, level 5 only).
    
    Uses EleutherAI/hendrycks_math with all configurations.
    Each problem includes a question and a solution with a boxed answer.
    Filters to only Level 5 (hardest) problems.
    """
    
    # All available categories in the MATH dataset
    CATEGORIES = [
        "algebra",
        "counting_and_probability", 
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    
    def __init__(self, split: str = "test"):
        super().__init__(split)
        self.dataset_name = "EleutherAI/hendrycks_math"
    
    def load(self) -> None:
        """Load the MATH dataset (all categories, level 5 only) from HuggingFace."""
        if self._loaded:
            return
        
        idx = 0
        for category in self.CATEGORIES:
            ds = load_dataset(self.dataset_name, category)
            split_data = ds[self.split]
            
            for item in split_data:
                # Filter for Level 5 only
                level = item.get("level", "")
                if level != "Level 5":
                    continue
                
                # Build the prompt from the problem
                prompt = self._build_prompt(item)
                
                # Extract the boxed answer from the solution
                answer = self._extract_boxed_answer(item["solution"])
                
                # Ground truth is the extracted answer
                ground_truth = {
                    "answer": answer,
                    "full_solution": item["solution"],
                }
                
                problem = Problem(
                    id=f"math_{category}_{idx}",
                    prompt=prompt,
                    ground_truth=ground_truth,
                    metadata={
                        "level": level,
                        "type": item.get("type", category),
                        "category": category,
                    }
                )
                self.problems.append(problem)
                idx += 1
        
        self._loaded = True
    
    def _build_prompt(self, item: dict) -> str:
        """Build a prompt from the MATH item."""
        prompt = f"""Solve the following math problem. Provide your final answer in a box using \\boxed{{answer}} format.

Problem: {item["problem"]}

Show your work step by step, then provide your final answer in \\boxed{{}} format."""
        return prompt
    
    def _extract_boxed_answer(self, solution: str) -> Optional[str]:
        """Extract the answer from \\boxed{answer} in the solution."""
        # Handle nested braces in boxed answers
        # Pattern to match \boxed{...} handling nested braces
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, solution)
        
        if matches:
            # Return the last boxed answer (usually the final answer)
            return matches[-1].strip()
        
        # Try simpler pattern if nested didn't work
        simple_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(simple_pattern, solution)
        
        if matches:
            return matches[-1].strip()
        
        return None
    
    def evaluate(self, response: str, problem: Problem) -> bool:
        """
        Evaluate if the model's answer matches the ground truth.
        
        Extracts the boxed answer from the response and compares
        with the ground truth answer.
        """
        # Extract answer from response
        predicted_answer = self._extract_boxed_answer(response)
        
        if predicted_answer is None:
            # Try to extract answer in other formats
            predicted_answer = self._extract_answer_fallback(response)
        
        if predicted_answer is None:
            return False
        
        expected_answer = problem.ground_truth["answer"]
        if expected_answer is None:
            return False
        
        # Normalize and compare
        return self._normalize_and_compare(predicted_answer, expected_answer)
    
    def _extract_answer_fallback(self, response: str) -> Optional[str]:
        """Try to extract answer using fallback patterns."""
        # Look for "answer is X" or "answer: X" patterns
        patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)',
            r'(?:final\s+)?answer[:\s]+([^\n.]+)',
            r'=\s*([^\n]+)$',  # Last equation result
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _normalize_and_compare(self, predicted: str, expected: str) -> bool:
        """Normalize answers and compare them."""
        # Clean up both answers
        pred_clean = self._normalize_answer(predicted)
        exp_clean = self._normalize_answer(expected)
        
        # Direct string comparison
        if pred_clean == exp_clean:
            return True
        
        # Try numerical comparison
        try:
            pred_num = self._parse_number(pred_clean)
            exp_num = self._parse_number(exp_clean)
            if pred_num is not None and exp_num is not None:
                # Allow small floating point tolerance
                return abs(pred_num - exp_num) < 1e-6
        except:
            pass
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        # Remove whitespace
        answer = answer.strip()
        
        # Remove dollar signs (LaTeX math mode)
        answer = answer.replace('$', '')
        
        # Remove common LaTeX commands that don't affect value
        answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]*)\}', r'\1', answer)
        
        # Normalize fractions: \frac{a}{b} -> a/b
        answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
        
        # Remove spaces
        answer = answer.replace(' ', '')
        
        # Normalize negative signs
        answer = answer.replace('−', '-')
        
        return answer.lower()
    
    def _parse_number(self, s: str) -> Optional[float]:
        """Try to parse a string as a number."""
        try:
            # Handle fractions
            if '/' in s:
                parts = s.split('/')
                if len(parts) == 2:
                    # Remove parentheses
                    num = parts[0].strip('()')
                    den = parts[1].strip('()')
                    return float(num) / float(den)
            return float(s)
        except:
            return None


if __name__ == "__main__":
    # Test the dataset loader
    print("Loading MATH dataset (all categories, level 5 only)...")
    dataset = MathAlgebraDataset(split="test")
    dataset.load()
    
    print(f"Loaded {len(dataset)} Level 5 problems")
    
    # Count problems per category
    categories = {}
    for p in dataset.problems:
        cat = p.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nProblems per category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Show first problem
    problem = dataset[0]
    print(f"\nFirst problem:")
    print(f"ID: {problem.id}")
    print(f"Category: {problem.metadata['category']}")
    print(f"Prompt: {problem.prompt[:300]}...")
    print(f"Expected answer: {problem.ground_truth['answer']}")
    print(f"Level: {problem.metadata['level']}")


