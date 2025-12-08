# src/evaluation/pass_at_k.py
"""
Pass@k metric calculation for code generation benchmarks.

The pass@k metric measures the probability that at least one of k
generated samples passes all test cases.
"""

from typing import List, Callable
from src.datasets.base import BaseDataset, Problem
from src.agents.workflow import Workflow


def calculate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int = 1
) -> float:
    """
    Calculate pass@k using the unbiased estimator.
    
    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    
    Where:
        n = num_samples (total samples generated)
        c = num_correct (samples that pass all tests)
        k = number of samples considered
    
    Args:
        num_samples: Total number of samples generated (n)
        num_correct: Number of correct samples (c)
        k: Number of samples to consider
        
    Returns:
        The pass@k probability (0.0 to 1.0)
    """
    if num_samples < k:
        # Not enough samples
        return 0.0
    
    if num_correct >= num_samples:
        # All samples are correct
        return 1.0
    
    if num_correct == 0:
        return 0.0
    
    # Calculate using the complement formula to avoid numerical issues
    # pass@k = 1 - prod((n-c-i)/(n-i) for i in range(k))
    result = 1.0
    for i in range(k):
        numerator = num_samples - num_correct - i
        denominator = num_samples - i
        if denominator <= 0:
            return 1.0
        result *= numerator / denominator
    
    return 1.0 - result


def evaluate_pass_at_k(
    workflow: Workflow,
    dataset: BaseDataset,
    num_problems: int = 10,
    samples_per_problem: int = 1,
    k: int = 1,
    seed: int = None
) -> dict:
    """
    Evaluate a workflow on a dataset and compute pass@k.
    
    Args:
        workflow: The workflow to evaluate
        dataset: The benchmark dataset
        num_problems: Number of problems to sample from dataset
        samples_per_problem: Number of samples to generate per problem
        k: The k value for pass@k
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation results:
        - pass_at_k: The pass@k score
        - num_correct: Number of correct solutions
        - num_problems: Number of problems evaluated
        - total_samples: Total samples generated
        - details: Per-problem results
    """
    # Sample problems from dataset
    problems = dataset.sample(num_problems, seed=seed)
    
    total_correct = 0
    total_samples = 0
    details = []
    
    for problem in problems:
        problem_correct = 0
        
        for sample_idx in range(samples_per_problem):
            # Run the workflow
            try:
                result = workflow.run(problem.prompt)
                response = result.get("content", "") if isinstance(result, dict) else str(result)
                
                # Evaluate the response
                is_correct = dataset.evaluate(response, problem)
                
                if is_correct:
                    problem_correct += 1
                    
            except Exception as e:
                # Workflow failed, count as incorrect
                pass
        
        # Calculate pass@k for this problem
        problem_pass_at_k = calculate_pass_at_k(
            samples_per_problem, 
            problem_correct, 
            k
        )
        
        total_correct += problem_correct
        total_samples += samples_per_problem
        
        details.append({
            "problem_id": problem.id,
            "num_correct": problem_correct,
            "num_samples": samples_per_problem,
            "pass_at_k": problem_pass_at_k,
        })
    
    # Calculate overall pass@k as average across problems
    overall_pass_at_k = sum(d["pass_at_k"] for d in details) / len(details) if details else 0.0
    
    return {
        "pass_at_k": overall_pass_at_k,
        "num_correct": total_correct,
        "num_problems": len(problems),
        "total_samples": total_samples,
        "k": k,
        "details": details,
    }


def quick_evaluate(
    workflow: Workflow,
    dataset: BaseDataset,
    num_problems: int = 5,
    seed: int = None
) -> float:
    """
    Quick evaluation for GA fitness - returns just pass@1 score.
    
    Optimized for speed during genetic algorithm evolution.
    
    Args:
        workflow: The workflow to evaluate
        dataset: The benchmark dataset
        num_problems: Number of problems to evaluate
        seed: Random seed for problem sampling
        
    Returns:
        Pass@1 score (0.0 to 1.0)
    """
    problems = dataset.sample(num_problems, seed=seed)
    
    correct = 0
    for problem in problems:
        try:
            result = workflow.run(problem.prompt)
            response = result.get("content", "") if isinstance(result, dict) else str(result)
            
            if dataset.evaluate(response, problem):
                correct += 1
        except Exception:
            pass
    
    return correct / len(problems) if problems else 0.0


if __name__ == "__main__":
    # Test pass@k calculation
    print("Testing pass@k calculation...")
    
    # Test cases
    test_cases = [
        (10, 1, 1),   # 1 correct out of 10, k=1
        (10, 5, 1),   # 5 correct out of 10, k=1
        (10, 10, 1),  # 10 correct out of 10, k=1
        (10, 5, 5),   # 5 correct out of 10, k=5
        (100, 50, 10), # 50 correct out of 100, k=10
    ]
    
    for n, c, k in test_cases:
        result = calculate_pass_at_k(n, c, k)
        print(f"pass@{k} with {c}/{n} correct: {result:.4f}")


