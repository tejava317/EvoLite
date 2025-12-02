#!/usr/bin/env python3
"""
Evaluate a single workflow on MBPP or MATH algebra benchmarks.

Usage:
    python evaluate.py --task MBPP --num-problems 10
    python evaluate.py --task MATH --num-problems 20
    python evaluate.py --task MBPP --roles "Task Parsing Agent,Code Generation Agent,Code Review Agent"
"""

import argparse
from src.agents.agent import Agent
from src.agents.workflow import Workflow
from src.agents.extractors import get_extractor_for_task
from src.datasets import MBPPDataset, MathAlgebraDataset
from src.evaluation.pass_at_k import evaluate_pass_at_k, calculate_pass_at_k
from src.config import get_predefined_prompt


# Default workflow configurations for different tasks
# Default workflows using predefined agent names from base_agents.yaml
DEFAULT_WORKFLOWS = {
    "MBPP": [
        "Task Parsing Agent",
        "Code Generation Agent",
        "Code Reviewer Agent",
    ],
    "MATH": [
        "Task Parsing Agent",
        "Task Refinement Agent", 
        "Code Generation Agent",
    ],
}


def create_workflow(
    task_name: str, 
    roles: list[str], 
    use_extractor: bool = True, 
    verbose: bool = False,
    use_predefined_prompts: bool = True
) -> Workflow:
    """
    Create a workflow from a list of role names.
    
    Args:
        task_name: Name of the task (MBPP, MATH, etc.)
        roles: List of role names for the agents
        use_extractor: Whether to add an answer extractor
        verbose: Whether to print intermediate steps during execution
        use_predefined_prompts: If True, use prompts from base_agents.yaml instead of generating with LLM
        
    Returns:
        Configured Workflow object
    """
    workflow_description = " -> ".join(roles)
    
    agents = []
    for role in roles:
        prompt = None
        if use_predefined_prompts:
            # Try to get predefined prompt from base_agents.yaml
            prompt = get_predefined_prompt(role)
        
        # Create agent with predefined prompt (or None to trigger generation)
        agent = Agent(role=role, prompt=prompt, workflow_description=workflow_description)
        agents.append(agent)
    
    extractor = get_extractor_for_task(task_name) if use_extractor else None
    
    workflow = Workflow(
        task_name=task_name,
        agents=agents,
        extractor=extractor,
        verbose=verbose
    )
    
    return workflow


def get_dataset(task_name: str):
    """Load the appropriate dataset for the task."""
    task_lower = task_name.lower()
    
    if 'mbpp' in task_lower:
        dataset = MBPPDataset(split="test")
    elif 'math' in task_lower or 'algebra' in task_lower:
        dataset = MathAlgebraDataset(split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: MBPP, MATH")
    
    dataset.load()
    return dataset


def evaluate_single_workflow(
    task_name: str,
    roles: list[str] = None,
    num_problems: int = 10,
    samples_per_problem: int = 1,
    use_extractor: bool = True,
    seed: int = None,
    verbose: bool = True,
    show_intermediate: bool = False,
    use_predefined_prompts: bool = True
) -> dict:
    """
    Evaluate a single workflow on a benchmark dataset.
    
    Args:
        task_name: Name of the task (MBPP, MATH)
        roles: List of agent roles (uses default if None)
        num_problems: Number of problems to evaluate
        samples_per_problem: Number of samples per problem
        use_extractor: Whether to use answer extractor
        seed: Random seed for reproducibility
        verbose: Print summary output
        show_intermediate: Print intermediate steps for each agent
        use_predefined_prompts: Use prompts from base_agents.yaml (no LLM generation)
        
    Returns:
        Evaluation results dictionary
    """
    # Use default roles if not specified
    if roles is None:
        roles = DEFAULT_WORKFLOWS.get(task_name.upper(), DEFAULT_WORKFLOWS["MBPP"])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WORKFLOW EVALUATION")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        print(f"Problems: {num_problems}")
        print(f"Samples per problem: {samples_per_problem}")
        print(f"Show intermediate: {show_intermediate}")
        print(f"Using predefined prompts: {use_predefined_prompts}")
        print(f"\nWorkflow:")
        for i, role in enumerate(roles, 1):
            has_predefined = get_predefined_prompt(role) is not None
            marker = "📋" if (use_predefined_prompts and has_predefined) else "🔄"
            print(f"  {i}. {role} {marker}")
        if use_extractor:
            extractor = get_extractor_for_task(task_name)
            print(f"  → {extractor.role}")
        print(f"{'='*60}\n")
    
    # Create workflow with verbose option
    workflow = create_workflow(
        task_name, roles, use_extractor, 
        verbose=show_intermediate,
        use_predefined_prompts=use_predefined_prompts
    )
    
    # Load dataset
    if verbose:
        print("Loading dataset...")
    dataset = get_dataset(task_name)
    if verbose:
        print(f"Loaded {len(dataset)} problems\n")
    
    # Run evaluation with progress printing
    if verbose:
        print("Running evaluation...\n")
    
    # Custom evaluation loop for better progress reporting
    problems = dataset.sample(num_problems, seed=seed)
    
    total_correct = 0
    details = []
    
    for i, problem in enumerate(problems):
        if verbose:
            print(f"{'─'*60}")
            print(f"📝 Problem {i+1}/{num_problems}: {problem.id}")
            print(f"{'─'*60}")
            if not show_intermediate:
                print(f"   Prompt: {problem.prompt[:100]}...")
        
        problem_correct = 0
        
        for sample_idx in range(samples_per_problem):
            try:
                result = workflow.run(problem.prompt)
                response = result.get("content", "") if isinstance(result, dict) else str(result)
                
                is_correct = dataset.evaluate(response, problem)
                
                if is_correct:
                    problem_correct += 1
                
                if verbose and not show_intermediate:
                    status = "✓" if is_correct else "✗"
                    print(f"   {status} Sample {sample_idx + 1}: {'PASS' if is_correct else 'FAIL'}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ✗ Sample {sample_idx + 1}: ERROR - {str(e)[:1000]}")
        
        problem_pass = calculate_pass_at_k(samples_per_problem, problem_correct, 1)
        total_correct += problem_correct
        
        details.append({
            "problem_id": problem.id,
            "num_correct": problem_correct,
            "num_samples": samples_per_problem,
            "pass_at_k": problem_pass,
        })
        
        if verbose:
            print(f"   Result: {problem_correct}/{samples_per_problem} correct\n")
    
    # Calculate overall pass@k
    overall_pass_at_k = sum(d["pass_at_k"] for d in details) / len(details) if details else 0.0
    
    results = {
        "pass_at_k": overall_pass_at_k,
        "num_correct": total_correct,
        "num_problems": len(problems),
        "total_samples": len(problems) * samples_per_problem,
        "k": 1,
        "details": details,
    }
    
    # Print final results
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Pass@1: {results['pass_at_k']:.4f} ({results['pass_at_k']*100:.1f}%)")
        print(f"Correct: {results['num_correct']}/{results['total_samples']}")
        print(f"Problems evaluated: {results['num_problems']}")
        
        # Token usage
        if workflow.total_tokens:
            print(f"\nToken usage:")
            print(f"  Prompt tokens: {workflow.prompt_tokens}")
            print(f"  Response tokens: {workflow.response_tokens}")
            print(f"  Total tokens: {workflow.total_tokens}")
        
        print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a workflow on MBPP or MATH benchmarks"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="MBPP",
        choices=["MBPP", "MATH"],
        help="Benchmark task to evaluate on"
    )
    parser.add_argument(
        "--num-problems", 
        type=int, 
        default=10,
        help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1,
        help="Number of samples per problem"
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated list of agent roles (uses default if not specified)"
    )
    parser.add_argument(
        "--no-extractor",
        action="store_true",
        help="Disable the answer extractor"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible problem sampling"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--show-intermediate",
        action="store_true",
        help="Show intermediate agent inputs/outputs during execution"
    )
    parser.add_argument(
        "--generate-prompts",
        action="store_true",
        help="Generate prompts with LLM instead of using predefined prompts from base_agents.yaml"
    )
    
    args = parser.parse_args()
    
    # Parse roles if provided
    roles = None
    if args.roles:
        roles = [r.strip() for r in args.roles.split(",")]
    
    # Run evaluation
    results = evaluate_single_workflow(
        task_name=args.task,
        roles=roles,
        num_problems=args.num_problems,
        samples_per_problem=args.samples,
        use_extractor=not args.no_extractor,
        seed=args.seed,
        verbose=not args.quiet,
        show_intermediate=args.show_intermediate,
        use_predefined_prompts=not args.generate_prompts
    )
    
    # Return pass@1 as exit code hint (0 = all correct, 1 = some failed)
    return 0 if results['pass_at_k'] == 1.0 else 1


if __name__ == "__main__":
    exit(main())

