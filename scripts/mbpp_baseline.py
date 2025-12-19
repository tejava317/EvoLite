#!/usr/bin/env python3
"""
Simple MBPP Dataset Baseline Solver

A standalone script to test LLM performance on the MBPP benchmark.
Does NOT use the EvoLite framework - serves as a baseline comparison.

Usage:
    python mbpp_baseline.py --num_problems 100 --model "your-model-name"
"""

import asyncio
import argparse
import re
import subprocess
import tempfile
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datasets import load_dataset
from openai import AsyncOpenAI


# === Configuration ===
SYSTEM_PROMPT = """You are an expert Python programmer. Write clean, correct, and efficient code.

Guidelines:
1. Write a complete function that solves the problem
2. Use proper Python syntax and conventions
3. Handle edge cases appropriately
4. Output ONLY the Python code in a markdown code block

Example output format:
```python
def function_name(args):
    # Your implementation
    return result
```"""


@dataclass
class Problem:
    id: str
    task_id: int
    prompt: str
    code: str
    test_list: List[str]
    test_imports: List[str]
    entry_point: Optional[str]


@dataclass
class Result:
    problem_id: str
    passed: bool
    code: Optional[str]
    error: Optional[str]
    response: str
    tokens: int
    time_taken: float


# === Dataset Loading ===
def load_mbpp_dataset(split: str = "test") -> List[Problem]:
    """Load MBPP sanitized dataset from HuggingFace."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    split_data = ds[split]
    
    problems = []
    for item in split_data:
        # Extract entry point from code
        entry_point = None
        match = re.search(r'def\s+(\w+)\s*\(', item["code"])
        if match:
            entry_point = match.group(1)
        
        # Build prompt with example test case
        prompt = item["prompt"]
        if item["test_list"]:
            prompt += f"\n\nExample test case:\n{item['test_list'][0]}"
        
        problems.append(Problem(
            id=f"mbpp_{item['task_id']}",
            task_id=item["task_id"],
            prompt=prompt,
            code=item["code"],
            test_list=item["test_list"],
            test_imports=item.get("test_imports", []),
            entry_point=entry_point,
        ))
    
    return problems


# === Code Extraction ===
def extract_code(response: str) -> Optional[str]:
    """Extract Python code from response (handles markdown code blocks)."""
    
    # First, try to get content after </think> tag (Qwen3 style)
    if '</think>' in response:
        response = response.split('</think>')[-1]
    
    # Pattern 1: Markdown code blocks with python tag
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # Return the last code block (usually the final solution)
        return matches[-1].strip()
    
    # Pattern 2: Look for function definitions directly
    if 'def ' in response:
        lines = response.split('\n')
        code_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Start capturing at 'def'
            if stripped.startswith('def '):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                code_lines = [line[indent_level:] if indent_level else line]
                continue
            
            if in_function:
                # Check if line is part of function
                if not stripped:
                    code_lines.append('')
                elif line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                    # Continuation of function
                    code_lines.append(line[indent_level:] if indent_level else line)
                elif stripped.startswith(('def ', 'class ', 'import ', 'from ', '#', '@')):
                    # New definition or import
                    code_lines.append(line[indent_level:] if indent_level else line)
                else:
                    # End of code block
                    break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    return None


# === Code Execution ===
def execute_code(
    code: str,
    test_cases: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: int = 5
) -> Tuple[bool, str]:
    """Execute code with test cases in subprocess."""
    if not code or not test_cases:
        return False, "Empty code or test cases"
    
    # Build test script
    parts = [
        "import sys",
        "import math",
        "from typing import List, Dict, Tuple, Optional, Any",
    ]
    
    if test_imports:
        parts.extend(test_imports)
    
    parts.append("")
    parts.append("# Generated code")
    parts.append(code)
    parts.append("")
    parts.append("# Test cases")
    
    for i, test in enumerate(test_cases):
        parts.extend([
            f"try:",
            f"    {test}",
            f"except AssertionError as e:",
            f"    print(f'Test {i+1} failed: {{e}}')",
            f"    sys.exit(1)",
            f"except Exception as e:",
            f"    print(f'Test {i+1} error: {{type(e).__name__}}: {{e}}')",
            f"    sys.exit(1)",
            "",
        ])
    
    parts.append("print('All tests passed!')")
    script = "\n".join(parts)
    
    # Run in subprocess
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stdout + result.stderr
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Execution error: {e}"
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


# === LLM Inference ===
async def solve_problem(
    client: AsyncOpenAI,
    problem: Problem,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    no_think: bool = False,
) -> Result:
    """Solve a single problem using the LLM."""
    user_prompt = f"""Write a Python function to solve this problem:

{problem.prompt}

Write ONLY the Python code in a markdown code block. No explanations."""

    start_time = time.time()
    
    try:
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if no_think:
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        response = await client.chat.completions.create(**request_params)
        
        elapsed = time.time() - start_time
        content = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0
        
        # Extract code
        code = extract_code(content)
        
        if not code:
            return Result(
                problem_id=problem.id,
                passed=False,
                code=None,
                error="No code extracted",
                response=content,
                tokens=tokens,
                time_taken=elapsed,
            )
        
        # Execute tests
        passed, output = execute_code(
            code,
            problem.test_list,
            problem.test_imports,
            timeout=5
        )
        
        return Result(
            problem_id=problem.id,
            passed=passed,
            code=code,
            error=None if passed else output[:200],
            response=content,
            tokens=tokens,
            time_taken=elapsed,
        )
    
    except Exception as e:
        elapsed = time.time() - start_time
        return Result(
            problem_id=problem.id,
            passed=False,
            code=None,
            error=str(e)[:200],
            response=f"ERROR: {e}",
            tokens=0,
            time_taken=elapsed,
        )


async def evaluate_batch(
    client: AsyncOpenAI,
    problems: List[Problem],
    model: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    verbose: bool = False,
    no_think: bool = False,
) -> List[Result]:
    """Evaluate a batch of problems with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def solve_with_semaphore(problem: Problem, idx: int) -> Result:
        async with semaphore:
            result = await solve_problem(client, problem, model, temperature, max_tokens, no_think)
            if verbose:
                status = "✓" if result.passed else "✗"
                err = f" | {result.error[:50]}" if result.error else ""
                print(f"  [{idx+1}/{len(problems)}] {status} {problem.id}{err}")
            return result
    
    tasks = [solve_with_semaphore(p, i) for i, p in enumerate(problems)]
    results = await asyncio.gather(*tasks)
    
    return list(results)


# === Main ===
async def main():
    parser = argparse.ArgumentParser(description="MBPP Dataset Baseline Solver")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default="empty", help="API key")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--num_problems", type=int, default=100, help="Number of problems to evaluate")
    parser.add_argument("--split", default="test", help="Dataset split (test, train, validation)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for problem selection")
    parser.add_argument("--verbose", action="store_true", help="Print per-problem results")
    parser.add_argument("--no_think", action="store_true", help="Disable thinking mode (for Qwen3 models)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MBPP Dataset Baseline Solver")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API: {args.base_url}")
    print(f"Split: {args.split}")
    print(f"No-think mode: {args.no_think}")
    print(f"Max tokens: {args.max_tokens}")
    print()
    
    # Load dataset
    print("Loading MBPP dataset...")
    all_problems = load_mbpp_dataset(split=args.split)
    print(f"Loaded {len(all_problems)} problems")
    
    # Sample problems
    import random
    random.seed(args.seed)
    if args.num_problems < len(all_problems):
        problems = random.sample(all_problems, args.num_problems)
    else:
        problems = all_problems
    print(f"Evaluating {len(problems)} problems...")
    print()
    
    # Create client
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Run evaluation
    start_time = time.time()
    results = await evaluate_batch(
        client=client,
        problems=problems,
        model=args.model,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        no_think=args.no_think,
    )
    total_time = time.time() - start_time
    
    # Compute metrics
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    total_tokens = sum(r.tokens for r in results)
    
    # Error breakdown
    no_code = sum(1 for r in results if r.error and "No code" in r.error)
    timeout = sum(1 for r in results if r.error and "Timeout" in r.error)
    failed = total - passed - no_code - timeout
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pass@1: {pass_rate:.1%} ({passed}/{total})")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Avg Tokens/Problem: {total_tokens/total:.1f}")
    print(f"Throughput: {total_tokens/total_time:.1f} tok/s")
    print()
    print("Error Breakdown:")
    print(f"  Passed: {passed}")
    print(f"  Failed tests: {failed}")
    print(f"  No code extracted: {no_code}")
    print(f"  Timeout: {timeout}")
    print()
    
    # Show some failures
    if args.verbose:
        failures = [r for r in results if not r.passed][:3]
        if failures:
            print("Sample failures:")
            for r in failures:
                print(f"  - {r.problem_id}: {r.error}")
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

