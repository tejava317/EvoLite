#!/usr/bin/env python3
"""
Simple MATH Dataset Baseline Solver

A standalone script to test LLM performance on the MATH benchmark.
Does NOT use the EvoLite framework - serves as a baseline comparison.

Usage:
    python math_baseline.py --num_problems 100 --model "your-model-name"
"""

import asyncio
import argparse
import re
import time
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from openai import AsyncOpenAI


# === Configuration ===
MATH_CATEGORIES = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Concise prompt to avoid excessively long responses
SYSTEM_PROMPT = """You are a skilled mathematician. Solve the problem concisely.

Be BRIEF - show key steps only, not every calculation detail.
Put your FINAL answer in \\boxed{answer} format at the end.

Example outputs:
- \\boxed{42}
- \\boxed{\\frac{3}{4}}
- \\boxed{x^2 + 1}"""


@dataclass
class Problem:
    id: str
    question: str
    solution: str
    answer: str
    level: str
    category: str


@dataclass
class Result:
    problem_id: str
    predicted: Optional[str]
    expected: str
    correct: bool
    response: str
    tokens: int
    time_taken: float


# === Dataset Loading ===
def load_math_dataset(
    level_filter: Optional[str] = None,
    categories: Optional[list] = None,
    split: str = "test"
) -> list[Problem]:
    """Load MATH dataset from HuggingFace."""
    problems = []
    cats = categories or MATH_CATEGORIES
    
    for category in cats:
        ds = load_dataset("EleutherAI/hendrycks_math", category)
        split_data = ds[split]
        
        for idx, item in enumerate(split_data):
            level = item.get("level", "")
            
            # Filter by level if specified
            if level_filter and level != level_filter:
                continue
            
            answer = extract_boxed_answer(item["solution"])
            
            problems.append(Problem(
                id=f"math_{category}_{idx}",
                question=item["problem"],
                solution=item["solution"],
                answer=answer or "",
                level=level,
                category=category,
            ))
    
    return problems


# === Answer Extraction & Evaluation ===
def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format with multiple fallbacks."""
    
    # First, try to get content after </think> tag (Qwen3 style)
    if '</think>' in text:
        text = text.split('</think>')[-1]
    
    # Pattern 1: Handle deeply nested braces (up to 3 levels)
    # Match \boxed{...} where ... can contain nested braces
    def find_boxed_content(s: str) -> list[str]:
        results = []
        i = 0
        while i < len(s):
            # Find \boxed{
            idx = s.find('\\boxed{', i)
            if idx == -1:
                break
            
            # Find matching closing brace
            start = idx + 7  # len('\\boxed{')
            depth = 1
            j = start
            while j < len(s) and depth > 0:
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                j += 1
            
            if depth == 0:
                results.append(s[start:j-1])
            i = j
        return results
    
    matches = find_boxed_content(text)
    if matches:
        return matches[-1].strip()
    
    # Pattern 2: Look for "answer is X" or "= X" at the end
    fallback_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([^\n$]+?)\$?\s*$',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?[:\s]*\$?([^\n$]+?)\$?\s*$',
        r'=\s*\$?([^\n$=]+?)\$?\s*$',
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            ans = match.group(1).strip()
            # Clean up common artifacts
            ans = re.sub(r'[.,;:]+$', '', ans)
            if ans and len(ans) < 100:  # Sanity check
                return ans
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()
    
    # Remove dollar signs (LaTeX math mode)
    answer = answer.replace('$', '')
    
    # Normalize \dfrac to \frac (common in model outputs)
    answer = answer.replace('\\dfrac', '\\frac')
    
    # Remove \text{}, \mathrm{}, \mathbf{} wrappers
    answer = re.sub(r'\\(?:text|mathrm|mathbf|textbf)\{([^}]*)\}', r'\1', answer)
    

    
    # Normalize various minus signs
    answer = answer.replace('−', '-')
    answer = answer.replace('–', '-')
    
    # Remove trailing punctuation
    answer = re.sub(r'[.,;:!]+$', '', answer)
    
    # Remove thousands separators (e.g., 85,184 -> 85184)
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
    answer = re.sub(r'\\,', '', answer)  # LaTeX thin space in numbers
    answer = re.sub(r'\\!', '', answer)  # LaTeX negative thin space
    
    return answer.lower()


def parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    try:
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                num = parts[0].strip('()')
                den = parts[1].strip('()')
                return float(num) / float(den)
        return float(s)
    except:
        return None


def answers_match(predicted: str, expected: str) -> bool:
    """Check if two answers match with multiple comparison strategies."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # Direct comparison
    if pred_norm == exp_norm:
        return True
    
    # Try without any LaTeX commands
    pred_plain = re.sub(r'\\[a-zA-Z]+', '', pred_norm)
    exp_plain = re.sub(r'\\[a-zA-Z]+', '', exp_norm)
    pred_plain = re.sub(r'[{}]', '', pred_plain)
    exp_plain = re.sub(r'[{}]', '', exp_plain)
    if pred_plain == exp_plain:
        return True
    
    # Numerical comparison
    try:
        pred_num = parse_number(pred_norm)
        exp_num = parse_number(exp_norm)
        if pred_num is not None and exp_num is not None:
            return abs(pred_num - exp_num) < 1e-6
    except:
        pass
    
    # Try evaluating simple expressions
    try:
        # Handle sqrt
        pred_eval = pred_norm.replace('\\sqrt', 'sqrt').replace('{', '(').replace('}', ')')
        exp_eval = exp_norm.replace('\\sqrt', 'sqrt').replace('{', '(').replace('}', ')')
        
        import math
        pred_val = eval(pred_eval, {"sqrt": math.sqrt, "pi": math.pi})
        exp_val = eval(exp_eval, {"sqrt": math.sqrt, "pi": math.pi})
        if isinstance(pred_val, (int, float)) and isinstance(exp_val, (int, float)):
            return abs(pred_val - exp_val) < 1e-6
    except:
        pass
    
    return False


# === LLM Inference ===
async def solve_problem(
    client: AsyncOpenAI,
    problem: Problem,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    no_think: bool = False,
) -> Result:
    """Solve a single problem using the LLM."""
    user_prompt = f"""Solve this math problem. Be concise, show key steps only.
Put your final answer in \\boxed{{}}.

Problem: {problem.question}"""

    start_time = time.time()
    
    try:
        # Build request params
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add chat_template_kwargs for models that support /nothink (e.g., Qwen3)
        if no_think:
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        response = await client.chat.completions.create(**request_params)
        
        elapsed = time.time() - start_time
        content = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0
        
        predicted = extract_boxed_answer(content)
        correct = False
        
        if predicted and problem.answer:
            correct = answers_match(predicted, problem.answer)
        
        return Result(
            problem_id=problem.id,
            predicted=predicted,
            expected=problem.answer,
            correct=correct,
            response=content,
            tokens=tokens,
            time_taken=elapsed,
        )
    
    except Exception as e:
        elapsed = time.time() - start_time
        return Result(
            problem_id=problem.id,
            predicted=None,
            expected=problem.answer,
            correct=False,
            response=f"ERROR: {e}",
            tokens=0,
            time_taken=elapsed,
        )


async def evaluate_batch(
    client: AsyncOpenAI,
    problems: list[Problem],
    model: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    verbose: bool = False,
    no_think: bool = False,
) -> list[Result]:
    """Evaluate a batch of problems with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def solve_with_semaphore(problem: Problem, idx: int) -> Result:
        async with semaphore:
            result = await solve_problem(client, problem, model, temperature, max_tokens, no_think)
            if verbose:
                status = "✓" if result.correct else "✗"
                print(f"  [{idx+1}/{len(problems)}] {status} {problem.id} | pred={result.predicted} | exp={result.expected}")
            return result
    
    tasks = [solve_with_semaphore(p, i) for i, p in enumerate(problems)]
    results = await asyncio.gather(*tasks)
    
    return list(results)


# === Main ===
async def main():
    parser = argparse.ArgumentParser(description="MATH Dataset Baseline Solver")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default="empty", help="API key")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--num_problems", type=int, default=100, help="Number of problems to evaluate")
    parser.add_argument("--level", default=None, help="Filter by level (e.g., 'Level 3', 'Level 5')")
    parser.add_argument("--category", default=None, help="Filter by category (e.g., 'algebra')")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for problem selection")
    parser.add_argument("--verbose", action="store_true", help="Print per-problem results")
    parser.add_argument("--no_think", action="store_true", help="Disable thinking mode (for Qwen3 models)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MATH Dataset Baseline Solver")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API: {args.base_url}")
    print(f"Level filter: {args.level or 'All'}")
    print(f"Category filter: {args.category or 'All'}")
    print(f"No-think mode: {args.no_think}")
    print(f"Max tokens: {args.max_tokens}")
    print()
    
    # Load dataset
    print("Loading MATH dataset...")
    categories = [args.category] if args.category else None
    all_problems = load_math_dataset(level_filter=args.level, categories=categories)
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
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    total_tokens = sum(r.tokens for r in results)
    
    # Per-category breakdown
    category_stats = {}
    for r, p in zip(results, problems):
        cat = p.category
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if r.correct:
            category_stats[cat]["correct"] += 1
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Avg Tokens/Problem: {total_tokens/total:.1f}")
    print(f"Throughput: {total_tokens/total_time:.1f} tok/s")
    print()
    
    if len(category_stats) > 1:
        print("Per-Category Accuracy:")
        for cat, stats in sorted(category_stats.items()):
            cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {cat}: {cat_acc:.1%} ({stats['correct']}/{stats['total']})")
        print()
    
    # Show some failures
    failures = [r for r in results if not r.correct][:3]
    if failures and args.verbose:
        print("Sample failures:")
        for r in failures:
            print(f"  - {r.problem_id}: pred={r.predicted}, exp={r.expected}")
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

