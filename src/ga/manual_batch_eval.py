#!/usr/bin/env python3
"""
Minimal batch evaluator for explicit workflow lists (MBPP & MATH).

- Uses the evaluation server's async batch endpoint.
- Batches of size 10 by default.
- Saves aggregated metrics (pass@1, tokens, time) to CSV.
- Generates plots (pass@1 vs tokens) per framework and mixed per dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from src.client import EvaluationClient
from src.client.models import roles_to_blocks, EvalResult


# ---------------------------------------------------------------------------
# Workflow definitions from the user request
# ---------------------------------------------------------------------------
WORKFLOWS: Dict[str, Dict[str, List[List[str]]]] = {
    "MBPP": {
        "ga_llm": [
            ["Solution Drafter"],
            ["Data & Arrays Specialist"],
            ["Approach Evaluator"],
            ["Data & Arrays Specialist", "Final Presenter"],
        ],
        "ga": [
            ["Graph & Combinatorics Specialist"],
            ["Strings & Text Specialist"],
            ["Solution Drafter"],
            ["Data & Arrays Specialist"],
            ["Math Specialist"],
            ["Final Presenter", "Data & Arrays Specialist"],
            ["Progress Summarizer", "Strings & Text Specialist"],
            ["Correctness Verifier", "Correctness Verifier"],
            ["Summary Author", "Strings & Text Specialist"],
        ],
        "hdlo": [
            ["Performance Optimizer", "Graph & Combinatorics Specialist"],
            ["Graph & Combinatorics Specialist", "Performance Optimizer"],
            ["Performance Optimizer", "Quality Auditor"],
            ["Performance Optimizer"],
            ["Data Structure Planner"],
        ],
    },
    "MATH": {
        "ga_llm": [
            ["Progress Summarizer", "Solution Drafter"],
            ["Counterexample Generator", "Data & Arrays Specialist"],
            ["Solution Refiner"],
            ["Algorithm Strategist"],
        ],
        "ga": [
            ["Logic Implementer"],
            ["Assumption Checker"],
            ["Requirement Extractor", "Logic Implementer"],
            ["Summary Author"],
            ["Logic Implementer", "Logic Implementer"],
            ["Logic Implementer"],
            ["Logic Implementer", "Answer Extractor", "Logic Implementer"],
        ],
        "hdlo": [
            ["Consistency Checker", "Data & Arrays Specialist"],
            ["Consistency Checker", "Data Structure Planner"],
            ["Algorithm Strategist", "Solution Planner"],
            ["Algorithm Strategist", "Data Structure Planner"],
            ["Performance Optimizer", "Solution Ranker"],
            ["Consistency Checker"],
            ["Test Designer"],
            ["Algorithm Strategist"],
        ],
    },
}

# Defaults for full dataset sizes (can be overridden via CLI).
DEFAULT_NUM_PROBLEMS = {
    "MBPP": 257,
    "MATH": 1131,
}

BATCH_SIZE = 2
OUTPUT_DIR = Path("src/ga/manual_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
async def eval_batch(
    client: EvaluationClient,
    task: str,
    workflows: List[List[str]],
    num_problems: int,
) -> List[Tuple[List[str], EvalResult]]:
    """Evaluate a batch of workflows and return paired results."""
    blocks = [roles_to_blocks(w) for w in workflows]
    resp = await client.evaluate_batch_async(
        workflows=blocks,
        task_name=task,
        num_problems=num_problems,
        use_extractor=False,
        seed=None,
        think=False,
    )
    return list(zip(workflows, resp))


async def eval_task_framework(
    task: str,
    framework: str,
    workflows: List[List[str]],
    num_problems: int,
    server_url: str,
) -> List[dict]:
    """Evaluate all workflows for a (task, framework) with batching."""
    client = EvaluationClient(server_url)
    results: List[dict] = []

    total_batches = (len(workflows) + BATCH_SIZE - 1) // BATCH_SIZE
    for b_idx, start in enumerate(range(0, len(workflows), BATCH_SIZE), 1):
        batch = workflows[start : start + BATCH_SIZE]
        paired = await eval_batch(client, task, batch, num_problems)
        for roles, res in paired:
            results.append(
                {
                    "task": task,
                    "framework": framework,
                    "workflow": " -> ".join(roles),
                    "pass_at_1": res.pass_at_1,
                    "num_problems": res.num_problems,
                    "total_tokens": res.total_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_time": res.total_time,
                    "tokens_per_second": res.tokens_per_second,
                    "error": res.error or "",
                }
            )
        print(
            f"[{task}/{framework}] Batch {b_idx}/{total_batches} "
            f"completed ({len(batch)} workflows).",
            flush=True,
        )

    await client.close()
    return results


async def run_all(tasks: List[str], num_problems_override: Dict[str, int], server_url: str, frameworks_filter: List[str] = None) -> List[dict]:
    all_results: List[dict] = []
    for task in tasks:
        for framework, wfs in WORKFLOWS[task].items():
            # Filter by frameworks if specified
            if frameworks_filter and framework not in frameworks_filter:
                continue
            num_problems = num_problems_override.get(task, DEFAULT_NUM_PROBLEMS[task])
            print(f"== Evaluating {task}/{framework}: {len(wfs)} workflows × {num_problems} problems (batch={BATCH_SIZE}) ==")
            res = await eval_task_framework(task, framework, wfs, num_problems, server_url)
            all_results.extend(res)
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results: List[dict]):
    """Generate per-framework and mixed plots: pass@1 vs tokens."""
    if not results:
        print("No results to plot.")
        return

    # Per task & framework
    for task in WORKFLOWS.keys():
        for framework in WORKFLOWS[task].keys():
            subset = [r for r in results if r["task"] == task and r["framework"] == framework and not r["error"]]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: r["total_tokens"])
            tokens = [r["total_tokens"] for r in subset]
            pass1 = [r["pass_at_1"] for r in subset]
            labels = [r["workflow"] for r in subset]

            plt.figure(figsize=(7, 5))
            plt.plot(tokens, pass1, "-o")
            for x, y, lbl in zip(tokens, pass1, labels):
                plt.annotate(lbl, (x, y), fontsize=8, alpha=0.7)
            plt.xlabel("Tokens")
            plt.ylabel("Pass@1")
            plt.title(f"{task} - {framework}")
            plt.grid(True, alpha=0.3)
            fname = OUTPUT_DIR / f"{task.lower()}_{framework}_pass1_tokens.png"
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")

    # Mixed plots per task (all frameworks)
    colors = {"ga": "tab:blue", "ga_llm": "tab:green", "hdlo": "tab:orange"}
    markers = {"ga": "o", "ga_llm": "s", "hdlo": "^"}
    for task in WORKFLOWS.keys():
        plt.figure(figsize=(7, 5))
        added = False
        for framework in WORKFLOWS[task].keys():
            subset = [r for r in results if r["task"] == task and r["framework"] == framework and not r["error"]]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: r["total_tokens"])
            tokens = [r["total_tokens"] for r in subset]
            pass1 = [r["pass_at_1"] for r in subset]
            plt.plot(
                tokens,
                pass1,
                marker=markers.get(framework, "o"),
                color=colors.get(framework, None),
                label=framework,
            )
            added = True
        if not added:
            plt.close()
            continue
        plt.xlabel("Tokens")
        plt.ylabel("Pass@1")
        plt.title(f"{task} - Mixed Frameworks")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = OUTPUT_DIR / f"{task.lower()}_mixed_pass1_tokens.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Batch-evaluate explicit workflows and plot results.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="MBPP,MATH",
        help="Comma-separated task list (subset of MBPP,MATH).",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=os.getenv("EVAL_SERVER_URL", "http://localhost:8002"),
        help="Evaluation server URL.",
    )
    parser.add_argument(
        "--num-problems-mbpp",
        type=int,
        default=DEFAULT_NUM_PROBLEMS["MBPP"],
        help="Number of MBPP problems to evaluate.",
    )
    parser.add_argument(
        "--num-problems-math",
        type=int,
        default=DEFAULT_NUM_PROBLEMS["MATH"],
        help="Number of MATH problems to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(OUTPUT_DIR / "batch_eval_results.csv"),
        help="Where to store aggregated results.",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default=None,
        help="Comma-separated framework list to evaluate (e.g., 'ga,hdlo'). If not specified, evaluates all frameworks.",
    )
    return parser.parse_args()


def save_results(results: List[dict], csv_path: Path):
    if not results:
        print("No results to save.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "framework",
        "workflow",
        "pass_at_1",
        "num_problems",
        "total_tokens",
        "completion_tokens",
        "total_time",
        "tokens_per_second",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results to {csv_path}")


def main():
    args = parse_args()
    tasks = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    num_override = {
        "MBPP": args.num_problems_mbpp,
        "MATH": args.num_problems_math,
    }
    frameworks_filter = None
    if args.frameworks:
        frameworks_filter = [f.strip() for f in args.frameworks.split(",") if f.strip()]

    results = asyncio.run(run_all(tasks, num_override, args.server_url, frameworks_filter))
    save_results(results, Path(args.output_csv))
    plot_results(results)


if __name__ == "__main__":
    main()

