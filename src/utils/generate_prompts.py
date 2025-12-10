# src/utils/generate_prompts.py
"""
Prompt Configuration for Evolving Workflows

This module is the SINGLE SOURCE OF TRUTH for all prompt-related constants and utilities.

Design Principles:
1. Agent prompts are TASK-AGNOSTIC (same for MBPP, MATH, CRUX-O, etc.)
2. Task description is appended only to the FIRST agent in the workflow
3. Extractors are DATASET-SPECIFIC (code vs math vs output prediction)
4. All agents follow PASS-THROUGH pattern with structured output
5. Prompts are designed for SEQUENTIAL MULTI-AGENT COLLABORATION

Note: With LangChain's with_structured_output(), we no longer need explicit
YAML formatting instructions - the schema handles output structure automatically.
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict


# ============== Configuration ==============

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============== 50 Agent Roles for Evolution ==============

AGENT_ROLES = [
    # === Core Solution Agents ===
    ("Task Decomposer", "Parse the benchmark problem and decompose it into clear sub-tasks and constraints."),
    ("Solution Drafter", "Draft an initial solution (reasoning, math derivation, or code) based on the current understanding."),
    ("Solution Reviewer", "Critically review the drafted solution for correctness, missing pieces, and unclear steps."),
    ("Solution Refiner", "Improve the reviewed solution for clarity, robustness, and simplicity without changing its intent."),
    ("Test Designer", "Design strong but minimal tests (examples, edge cases, sanity checks) to validate the solution."),
    
    # === Analysis Agents ===
    ("Problem Analyzer", "Analyze the underlying structure and difficulty of the problem and identify key subproblems."),
    ("Requirement Extractor", "Extract explicit and implicit requirements and success criteria from the problem and shared state."),
    ("Constraint Analyzer", "Identify and formalize constraints such as ranges, limits, invariants, and feasibility conditions."),
    ("Input/State Inspector", "Inspect and normalize the current input and shared state, flagging gaps or inconsistencies."),
    ("Output Specifier", "Clarify the exact expected output format, units, rounding rules, and benchmark-specific quirks."),
    
    # === Planning Agents ===
    ("Solution Planner", "Plan the overall solution as a sequence of high-level steps, suitable for other agents to follow."),
    ("Algorithm Strategist", "Choose the main algorithmic or mathematical strategy (e.g., DP, greedy, proof by contradiction)."),
    ("Data Structure Planner", "Select appropriate data structures or mathematical representations for the chosen strategy."),
    ("Complexity Estimator", "Estimate time, space, or symbolic complexity and check feasibility under constraints."),
    ("Edge Case Planner", "Enumerate and plan explicit handling for edge, corner, and failure cases."),
    
    # === Implementation Agents ===
    ("Interface Designer", "Design function signatures, APIs, or mathematical symbols that the rest of the solution will use."),
    ("Logic Implementer", "Implement the core logic in code, formulas, or structured step-by-step reasoning."),
    ("Robustness Engineer", "Add validation, checks, and guard rails to make the solution resilient to bad inputs or edge states."),
    ("Performance Optimizer", "Optimize the solution for efficiency while preserving correctness and clarity."),
    ("Style & Hygiene Enforcer", "Polish style, naming, formatting, and remove dead or confusing parts of the solution."),
    
    # === Quality Agents ===
    ("Bug Hunter", "Search for concrete bugs using reasoning, tests, and adversarial thinking."),
    ("Correctness Verifier", "Verify step-by-step correctness of reasoning, math, and code relative to the problem statement."),
    ("Boundary Tester", "Design and mentally execute boundary and extreme test cases to stress the solution."),
    ("Regression Sentinel", "Detect whether new changes break previously working behavior or constraints."),
    ("Quality Auditor", "Audit the overall solution against benchmark requirements and best practices."),
    
    # === Documentation / Explanation Agents ===
    ("Comment & Hint Writer", "Insert concise comments or hints where they improve understanding of key steps or code."),
    ("Docstring Author", "Write docstrings or formal statements describing inputs, outputs, and behavior of key components."),
    ("Example Curator", "Create informative examples or micro-cases to illustrate how the solution works."),
    ("Explanation Author", "Write a clear, step-by-step explanation of the solution suitable for a human reader."),
    ("Summary Author", "Produce a concise summary of the final approach, trade-offs, and caveats."),
    
    # === Reasoning Agents ===
    ("Stepwise Reasoner", "Reason carefully step-by-step, keeping each step aligned with assumptions and constraints."),
    ("Assumption Checker", "Make assumptions explicit and verify them against the problem text and shared state."),
    ("Consistency Checker", "Check consistency across code, math, and explanations, resolving contradictions."),
    ("Proof & Derivation Verifier", "Verify mathematical proofs or derivations in detail, line by line."),
    ("Counterexample Generator", "Generate counterexamples or adversarial scenarios to test solution robustness."),
    
    # === Collaboration / Orchestration Agents ===
    ("Feedback Integrator", "Merge feedback and partial results from multiple agents into a coherent updated solution."),
    ("Conflict Resolver", "Resolve conflicts between competing solution paths or design choices."),
    ("Consensus Builder", "Select a single canonical solution path for downstream agents to follow."),
    ("Quality Gatekeeper", "Decide whether the solution is ready to move forward or needs to loop back for fixes."),
    ("Progress Summarizer", "Track what has been done, what remains, and update the shared state summary."),
    
    # === Specialized Agents ===
    ("Math Specialist", "Handle symbolic manipulation, algebra, calculus, and other math-intensive sub-problems."),
    ("Data & Arrays Specialist", "Handle arrays, lists, matrices, and related data transformations or algorithms."),
    ("Strings & Text Specialist", "Handle text parsing, formatting, and string algorithms."),
    ("Graph & Combinatorics Specialist", "Handle graph theory, combinatorics, and search-based problems."),
    ("Dynamic Programming Specialist", "Identify and implement dynamic programming or memoization-based solutions."),
    
    # === Meta / Finalization Agents ===
    ("Strategy Selector", "Choose which high-level strategy or plan to commit to among alternatives."),
    ("Approach Evaluator", "Evaluate pros and cons of existing approaches and decide what to keep or discard."),
    ("Solution Ranker", "Rank multiple solution candidates or answers and select the best one."),
    ("Answer Extractor", "Extract and normalize the benchmark-required final answer from the solution trace."),
    ("Final Presenter", "Produce the final answer and formatted output ready for benchmark evaluation."),
]

# Roles that should actively create/refresh solutions in answer: field.
CREATOR_ROLES = {
    "solution drafter",
    "logic implementer",
    "math specialist",
    "data & arrays specialist",
    "strings & text specialist",
    "graph & combinatorics specialist",
    "dynamic programming specialist",
}


def is_creator_role(role_name: str) -> bool:
    """Check if a role should create solutions vs review/refine."""
    return role_name.strip().lower() in CREATOR_ROLES


# ============== Task Descriptions ==============
# These are appended ONLY to the FIRST agent in a workflow

TASK_DESCRIPTIONS = {
    "MBPP": """
=== TASK: Python Function Generation (MBPP Benchmark) ===

**Problem Format:**
You receive a natural language description of a function to implement, along with example test cases.

**Example:**
```
Write a python function to remove first and last occurrence of a given character from the string.

assert remove_Occ("hello","l") == "heo"
```

**Your Goal:** Generate a correct Python function that passes ALL test cases.

**Key Rules:**
1. Use the EXACT function name from the test case (e.g., `remove_Occ`)
2. Write complete, executable Python code
3. Include necessary imports at the top (e.g., `import re`, `from collections import ...`)
4. Handle edge cases (empty input, not found, etc.)
5. Return ONLY the function definition - no test code, no print statements

**Answer Format:** Complete Python function code
""",

    "MATH": """
=== TASK: Mathematical Problem Solving (MATH Benchmark - Algebra) ===

**Problem Format:**
You receive a math problem, often with LaTeX notation.

**Example:**
```
How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
```

**Your Goal:** Solve the problem step-by-step and provide the final answer.

**Key Rules:**
1. Show clear mathematical reasoning
2. Use proper mathematical notation
3. Verify your calculations
4. The FINAL answer MUST be in `\\boxed{answer}` format

**Answer Format:** Solution ending with `\\boxed{final_answer}`
Examples: `\\boxed{2}`, `\\boxed{\\frac{3}{4}}`, `\\boxed{x^2 + 1}`
""",

    "CRUX-O": """
=== TASK: Code Output Prediction (CRUX-O Benchmark) ===

**Problem Format:**
You receive Python code and must predict what it outputs.

**Example:**
```python
def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output

assert f([1, 1, 3, 1, 3, 1]) == ??
```

**Your Goal:** Predict the exact output value.

**Key Rules:**
1. Trace through the code step-by-step in your `think` field
2. Track all variable values, loop iterations, function calls
3. The `answer` must be ONLY the output value as a Python literal
4. NO code, NO `assert`, NO explanation in the answer - just the value

**Answer Format:** Python literal only
Examples: `[(4, 1), (4, 1), (2, 3)]`, `{'a': 1}`, `'hello'`, `42`, `True`
""",
}


# ============== Extractor Prompts ==============
# Used to extract and normalize the final answer for evaluation

EXTRACTOR_PROMPTS = {
    "MBPP": """Extract the Python function code from the input.

RULES:
1. Return ONLY valid Python code (imports + function definition)
2. If multiple functions exist, return the last complete function with its imports
3. Remove any markdown fences, explanations, or test code
4. If no valid function is found, return exactly: NO_ANSWER

OUTPUT: Raw Python code only, or NO_ANSWER""",

    "MATH": """Extract the final boxed answer from the input.

RULES:
1. Find the last \\boxed{...} expression in the text
2. Return ONLY the \\boxed{...} expression (nothing else)
3. If multiple \\boxed{} exist, choose the last one (final answer)
4. If no \\boxed{} is found, return exactly: NO_ANSWER

OUTPUT: The \\boxed{...} expression only, or NO_ANSWER""",

    "CRUX-O": """Extract the predicted output value from the input.

RULES:
1. Find the Python literal that represents the predicted output
2. Return ONLY the raw value (list, dict, tuple, string, int, bool, None)
3. Do NOT include 'assert', variable assignments, or explanations
4. If no valid literal is found, return exactly: NO_ANSWER

OUTPUT: Python literal only, or NO_ANSWER""",
}


# ============== Prompt Building ==============

def build_agent_prompt(role_name: str, responsibility: str) -> str:
    """
    Build a task-agnostic prompt for a role in sequential multi-agent collaboration.
    """
    is_creator = is_creator_role(role_name)
    
    prompt = f"""You are the **{role_name}** in a multi-agent problem-solving workflow.

**Your Responsibility:** {responsibility}

---

**WORKFLOW CONTEXT:**
You are part of a sequential pipeline where multiple AI agents collaborate to solve a problem.
Each agent receives the previous agent's output and builds upon it.

**INPUT YOU RECEIVE:**
- `problem`: The original problem statement (passed through unchanged)
- `comment`: Notes from the previous agent (insights, concerns, status)
- `answer`: The previous agent's solution attempt (if any)

**OUTPUT YOU PRODUCE:**
- `think`: Your reasoning process (private - NOT forwarded to next agent)
- `problem`: Copy the original problem EXACTLY (do not modify)
- `comment`: Brief notes for the next agent (1-3 sentences)
- `answer`: Your solution for this stage

"""
    
    if is_creator:
        prompt += """**ANSWER HANDLING (Creator Role):**
You are a CREATOR role - you should actively produce or improve solutions.
- If no prior answer exists: Create a new solution from scratch
- If a prior answer exists: Improve, fix, or rewrite it as needed
- ALWAYS place a complete solution in the `answer` field
- Never leave `answer` empty or move the solution to `think`
"""
    else:
        prompt += """**ANSWER HANDLING (Reviewer Role):**
You are a REVIEWER role - you should analyze and refine, not create from scratch.
- If a prior answer exists: Copy it to `answer` and make targeted improvements only if needed
- If no prior answer exists: Leave `answer` empty (do NOT invent a solution)
- Focus your expertise on analysis, verification, or refinement
- Your value is in catching issues, not generating new solutions
"""
    
    prompt += """
**QUALITY GUIDELINES:**
- Be thorough in `think` but concise in `comment`
- Preserve the original problem exactly - downstream agents depend on it
- Your `comment` should help the next agent understand what you did and what to focus on
- If you find issues, note them clearly in `comment`
"""
    
    return prompt


def generate_all_prompts() -> Dict[str, str]:
    """Generate prompts for all 50 agent roles."""
    prompts = {}
    for role_name, responsibility in AGENT_ROLES:
        prompts[role_name] = build_agent_prompt(role_name, responsibility)
    return prompts


# ============== Helper Functions ==============

def get_task_description(task_name: str) -> str:
    """Get task description for a dataset."""
    task_upper = task_name.upper()
    if task_upper in TASK_DESCRIPTIONS:
        return TASK_DESCRIPTIONS[task_upper]
    
    # Fallback for unknown tasks
    return f"""
=== TASK: {task_name} ===

Goal: Complete the given task according to the problem description.
Input: Problem description
Output: Solution in the required format

Follow instructions carefully and produce a clear, correct solution.
"""


def get_extractor_prompt(task_name: str) -> str:
    """Get the appropriate extractor prompt for a task/benchmark."""
    task_upper = task_name.upper()
    if task_upper in EXTRACTOR_PROMPTS:
        return EXTRACTOR_PROMPTS[task_upper]
    
    # Fallback by keyword
    task_lower = task_name.lower()
    if "math" in task_lower:
        return EXTRACTOR_PROMPTS["MATH"]
    elif "crux" in task_lower:
        return EXTRACTOR_PROMPTS["CRUX-O"]
    else:
        return EXTRACTOR_PROMPTS["MBPP"]


def build_first_agent_prompt(agent_prompt: str, task_name: str) -> str:
    """Build prompt for the first agent in workflow (includes task description)."""
    task_desc = get_task_description(task_name)
    return f"{agent_prompt}\n\n{task_desc}"


# ============== YAML Save/Load Functions ==============

def save_prompts_to_yaml(prompts: dict, output_path: Optional[str] = None):
    """Save all generated prompts to YAML file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "configs" / "generated_prompts.yaml"
    
    output = {
        "agents": prompts,
        "task_descriptions": TASK_DESCRIPTIONS,
        "extractors": EXTRACTOR_PROMPTS,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Prompt Configuration for Evolving Workflows\n")
        f.write("# Generated by generate_prompts.py\n")
        f.write("# Agent prompts are task-agnostic\n")
        f.write("# Task descriptions are appended to first agent only\n\n")
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ Saved prompts to {output_path}")
    return output_path


def load_prompts_from_yaml(path: Optional[str] = None) -> dict:
    """Load prompts from YAML file."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "generated_prompts.yaml"
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============== Main ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate agent prompts")
    parser.add_argument("--output", default=None, help="Output YAML path")
    args = parser.parse_args()
    
    print("🚀 Generating Prompts for Evolving Workflows...")
    print(f"   Roles: {len(AGENT_ROLES)}")
    print(f"   Tasks: {len(TASK_DESCRIPTIONS)}")
    print()
    
    prompts = generate_all_prompts()
    save_prompts_to_yaml(prompts, args.output)
    
    # Print summary
    print(f"\n=== Generated {len(prompts)} Agent Prompts ===")
    creators = [r for r, _ in AGENT_ROLES if is_creator_role(r)]
    reviewers = [r for r, _ in AGENT_ROLES if not is_creator_role(r)]
    print(f"   Creator roles: {len(creators)}")
    print(f"   Reviewer roles: {len(reviewers)}")
    print(f"\nTasks: {', '.join(TASK_DESCRIPTIONS.keys())}")
    
    # Example output
    print("\n=== Example: Solution Drafter Prompt ===")
    print(prompts["Solution Drafter"][:600] + "...")
