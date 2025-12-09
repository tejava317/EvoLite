# src/utils/generate_prompts.py
"""
Prompt Configuration for Evolving Workflows

This module is the SINGLE SOURCE OF TRUTH for all prompt-related constants and utilities.
Both the prompt generator and evaluation server import from here.

Design Principles:
1. Agent prompts are TASK-AGNOSTIC (same for MBPP, MATH, CRUX-O, etc.)
2. Task description is appended only to the FIRST agent in the workflow
3. Extractors are DATASET-SPECIFIC (code vs math vs output prediction)
4. All agents follow PASS-THROUGH pattern (pass relevant info to next agent)
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============== Configuration ==============

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============== Structured Output Format ==============
# All agents MUST output in YAML format. problem:, comment:, answer: are passed to next agent. work: is logged only.
# Dataset-specific instructions guide what goes in the answer: field.

STRUCTURED_OUTPUT_BASE = """
**OUTPUT FORMAT:**

You MUST output EXACTLY ONE yaml code block containing ALL your response. Nothing outside the block.

```yaml
problem: |
  Copy the original problem EXACTLY (including given test cases). Do not modify.

work: |
  Your detailed analysis/reasoning. This is NOT passed forward.
  Do NOT place final answers or code here.

comment: |
  1-3 sentences for the next agent: function name, edge cases, key risks, status of answer.

answer: |
  {answer_description}
```

CRITICAL RULES:
1. Output ONLY ONE ```yaml block. Nothing outside it.
2. NEVER use ``` inside the yaml block. Write code/math directly after "answer: |"
3. MUST indent all content by 2 spaces after each field (problem:|, work:|, comment:|, answer:|)
"""

# Dataset-specific answer instructions
STRUCTURED_OUTPUT_INSTRUCTIONS = {
    "MBPP": STRUCTURED_OUTPUT_BASE.format(
        answer_description="The Python function implementation."
    ) + """
MBPP RULES:
- The function name is in the example test case (e.g., assert remove_Occ(...) means def remove_Occ)
- Write the complete function with correct name from the test case
- Add imports if needed (e.g., import re, from collections import ...)
- Write code directly with 2-space indent, NO ```python fences

CORRECT indentation:
answer: |
  def foo():      <- 2 spaces before 'def'
      return 1    <- 2+4 spaces (standard Python indent)

EXAMPLE:
```yaml
problem: |
  Write a python function to remove first and last occurrence of a given character from the string.

  Example test case:
  assert remove_Occ("hello","l") == "heo"

work: |
  Need to find first occurrence, remove it, then find last and remove.
  Function name from test: remove_Occ

comment: |
  Function remove_Occ, handles string manipulation with two passes.

answer: |
  def remove_Occ(s, ch):
      for i in range(len(s)):
          if s[i] == ch:
              s = s[:i] + s[i+1:]
              break
      for i in range(len(s)-1, -1, -1):
          if s[i] == ch:
              s = s[:i] + s[i+1:]
              break
      return s
```
""",

    "MATH": STRUCTURED_OUTPUT_BASE.format(
        answer_description="The mathematical solution. MUST end with \\boxed{final_answer}."
    ) + """
MATH RULES:
- Problems use LaTeX notation (e.g., $y=\\frac{2}{x^2+x-6}$)
- Show step-by-step reasoning in work: field
- answer: MUST end with \\boxed{...} containing the final answer
- Answers can be: integers (\\boxed{2}), fractions (\\boxed{\\dfrac{9}{7}}), expressions (\\boxed{i})

EXAMPLE:
```yaml
problem: |
  How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?

work: |
  Factor denominator: x^2+x-6 = (x-2)(x+3)
  Vertical asymptotes occur where denominator = 0
  x = 2 and x = -3 are the zeros
  Numerator 2 is never zero, so both are asymptotes

comment: |
  Factored quadratic, found 2 vertical asymptotes at x=2 and x=-3.

answer: |
  The denominator factors as $x^2+x-6=(x-2)(x+3)$.
  Setting denominator to zero: $x=2$ or $x=-3$.
  Since numerator is always nonzero, both give vertical asymptotes.
  \\boxed{2}
```
""",

    "CRUX-O": STRUCTURED_OUTPUT_BASE.format(
        answer_description="The predicted output value ONLY. A Python literal (list, dict, int, str, etc). NO code, NO assert."
    ) + """
CRUX-O RULES:
- You are given Python code and must predict what f(input) returns
- Trace through the code step-by-step in work: field
- answer: MUST contain ONLY the raw Python literal output
- Include quotes for strings: 'hello' not hello
- Common output types: lists, tuples, dicts, strings, ints, bools

CORRECT answer: field examples:
  [(4, 1), (4, 1), (2, 3), (2, 3)]
  {1: None, 2: None}
  'hbtofdeiequ'
  True
  42

WRONG (do NOT do these):
  assert f(x) == [1, 2, 3]   <- NO assert
  The output is [1, 2, 3]    <- NO prose

EXAMPLE:
```yaml
problem: |
  def f(nums):
      output = []
      for n in nums:
          output.append((nums.count(n), n))
      output.sort(reverse=True)
      return output
  assert f([1, 1, 3, 1, 3, 1]) ==

work: |
  nums = [1, 1, 3, 1, 3, 1]
  count(1) = 4, count(3) = 2
  output = [(4,1), (4,1), (2,3), (4,1), (2,3), (4,1)]
  After sort(reverse=True): [(4,1), (4,1), (4,1), (4,1), (2,3), (2,3)]

comment: |
  Counts occurrences, creates tuples, sorts descending.

answer: |
  [(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]
```
""",
}

# Default/fallback for unknown datasets
STRUCTURED_OUTPUT_DEFAULT = STRUCTURED_OUTPUT_INSTRUCTIONS["MBPP"]


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
CREATOR_ROLE_OVERRIDES = {
    "solution drafter",
    "logic implementer",
    "math specialist",
    "data & arrays specialist",
    "strings & text specialist",
    "graph & combinatorics specialist",
    "dynamic programming specialist",
}

# Default behaviour: verification/analysis roles copy or fix existing answers, never invent from scratch.
DEFAULT_ROLE_BEHAVIOR = "revise"


def determine_role_behavior(role_name: str) -> str:
    """
    Decide whether a role should create new solutions ("create") or only revise/carry forward ("revise").
    Defaults to revise to prevent non-solver agents from inventing answers.
    """
    role_lower = role_name.strip().lower()
    if role_lower in CREATOR_ROLE_OVERRIDES:
        return "create"
    return DEFAULT_ROLE_BEHAVIOR


def build_role_prompt(role_name: str, responsibility: str, behavior: str) -> str:
    """
    Deterministic, task-agnostic prompt builder with explicit answer handling.
    Output is in YAML format.
    """
    base_intro = f"You are the {role_name}. {responsibility.strip()}"
    shared_flow = (
        "You receive problem, comment, and answer from the previous agent in YAML format. "
        "Use work: for your reasoning (it is NOT forwarded). "
        "Use comment: to leave 1-3 critical notes for the next agent."
    )
    
    if behavior == "create":
        answer_rule = (
            "Always place the full candidate solution in answer: (code/math/output). "
            "If a prior answer exists, rewrite or improve it—never drop it or move it to work:. "
            "Do not leave answer: empty."
        )
    else:
        answer_rule = (
            "If a prior answer exists, copy it into answer: and adjust only when needed; never discard it or hide it in work:. "
            "If no prior answer exists, leave answer: blank instead of inventing a solution."
        )
    
    return " ".join([base_intro, shared_flow, answer_rule])


# ============== Task Descriptions ==============

TASK_DESCRIPTIONS = {
    "MBPP": """
=== TASK: Python Function Generation (MBPP) ===

You will receive a short problem description like:
  "Write a python function to remove first and last occurrence of a given character from the string."
  
Along with example test cases like:
  assert remove_Occ("hello","l") == "heo"

Your goal: Generate a correct Python function that passes ALL test cases.

RULES:
- Write a complete Python function with the exact function name from the test case
- Handle edge cases (empty strings, not found, etc.)
- The function must be executable and pass the provided assertions
- Return ONLY the function definition, no test code""",

    "MATH": """
=== TASK: Mathematical Problem Solving (MATH - Algebra) ===

You will receive a math problem, often with LaTeX notation, like:
  "How many vertical asymptotes does the graph of y=2/(x²+x-6) have?"

Your goal: Solve the problem step-by-step and provide the final answer.

RULES:
- Show clear step-by-step reasoning
- Use proper mathematical notation
- The FINAL answer MUST be in \\boxed{answer} format
- Example: \\boxed{2} or \\boxed{\\frac{3}{4}}
- Verify your calculations before giving the final answer""",

    "CRUX-O": """
=== TASK: Code Output Prediction (CRUX-O) ===

You will receive Python code and must predict the output.

Example problem:
  def f(nums):
      output = []
      for n in nums:
          output.append((nums.count(n), n))
      output.sort(reverse=True)
      return output
  assert f([1, 1, 3, 1, 3, 1]) == ??

Example answer: [(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]

RULES:
1. Trace through the code step-by-step in work: field
2. In answer: field, write ONLY the predicted output value
3. The answer must be a valid Python literal: list, tuple, dict, string, int, bool, etc.

CRITICAL: answer: must contain the actual predicted value like:
  [2, 2, 3, 2, 3, 3]
  'hello world'
  42
  True
  {1: None, 2: None}

DO NOT leave placeholders. DO NOT write code. Just the output value.""",
}


# ============== Extractor Prompts ==============
# Keyed by benchmark name, designed based on actual data formats

EXTRACTOR_PROMPTS = {
    "MBPP": """Extract the Python solution code.

Rules:
- Return ONLY valid Python code (imports + exactly one function def block). No markdown fences or prose.
- If multiple defs exist, return the last full function block with any required imports above it.
- If the text says NO_ANSWER/None yet or no def is present, return exactly NO_ANSWER.

Examples:
INPUT: answer: |\\n  from math import sqrt\\n  def is_square(n):\\n    r=int(sqrt(n));return r*r==n\\nOUTPUT: from math import sqrt\\ndef is_square(n):\\n    r=int(sqrt(n));return r*r==n

INPUT: def add(a,b):\\n    return a+b\\n\\ndef mul(a,b):\\n    return a*b\\nOUTPUT: def mul(a,b):\\n    return a*b

INPUT: None yet\\nOUTPUT: NO_ANSWER

Now return the extracted code or NO_ANSWER:""",

    "MATH": """Extract the final \\boxed{...} expression.

Rules:
- Choose the last non-empty \\boxed{...} in the text (even if other boxed steps exist).
- Ignore surrounding $$ math fences or other markup; output only the \\boxed{...}.
- Output must be exactly ONE line: either the boxed expression or NO_ANSWER (nothing else, no echo, no preamble).
- If there is no \\boxed{...} or the text says NO_ANSWER/None yet, return exactly NO_ANSWER.

Examples:
INPUT: ... we simplify to \\boxed{\\frac{7}{3}}.\\nOUTPUT: \\boxed{\\frac{7}{3}}

INPUT: First step \\boxed{2}, final \\boxed{5}\\nOUTPUT: \\boxed{5}

INPUT: (no boxed)\\nOUTPUT: NO_ANSWER

Now return the boxed answer or NO_ANSWER:""",

    "CRUX-O": """Extract the predicted output as a Python literal.

Rules:
- Prefer the literal to the right of '==' if present (matches CRUX direct-output prompts: 'assert f(input) ==').
- Otherwise return the last explicit Python literal (list/dict/tuple/str/int/bool/None/etc.) in the text.
- If the text says NO_ANSWER/None yet or no literal is present, return exactly NO_ANSWER.
- Do not add markdown, 'assert', or prose.

Examples:
INPUT: assert f([1,1,3]) == [(2, 1), (1, 3)]\\nOUTPUT: [(2, 1), (1, 3)]

INPUT: The output will be {'a': 1, 'b': None}\\nOUTPUT: {'a': 1, 'b': None}

INPUT: NO_ANSWER\\nOUTPUT: NO_ANSWER

Now return the extracted literal or NO_ANSWER:""",
}


# ============== Helper Functions ==============

def get_task_description(task_name: str) -> str:
    """Get task description for a dataset."""
    if task_name in TASK_DESCRIPTIONS:
        return TASK_DESCRIPTIONS[task_name]
    
    task_lower = task_name.lower()
    for key, desc in TASK_DESCRIPTIONS.items():
        if key.lower() == task_lower:
            return desc
    
    return f"""
=== TASK: {task_name} ===
Goal: Complete the given task according to the problem description.
Input: Problem description
Output: Solution in the required format
Key: Follow instructions carefully."""


def get_extractor_prompt(task_name: str) -> str:
    """Get the appropriate extractor prompt for a task/benchmark."""
    # Try exact match first
    if task_name in EXTRACTOR_PROMPTS:
        return EXTRACTOR_PROMPTS[task_name]
    
    # Try case-insensitive match
    task_upper = task_name.upper()
    for key in EXTRACTOR_PROMPTS:
        if key.upper() == task_upper:
            return EXTRACTOR_PROMPTS[key]
    
    # Fallback by keyword
    task_lower = task_name.lower()
    if "math" in task_lower:
        return EXTRACTOR_PROMPTS["MATH"]
    elif "crux" in task_lower:
        return EXTRACTOR_PROMPTS["CRUX-O"]
    else:
        return EXTRACTOR_PROMPTS["MBPP"]


def get_structured_output_instructions(task_name: str) -> str:
    """Get the appropriate structured output instructions for a task/benchmark."""
    # Try exact match first
    if task_name in STRUCTURED_OUTPUT_INSTRUCTIONS:
        return STRUCTURED_OUTPUT_INSTRUCTIONS[task_name]
    
    # Try case-insensitive match
    task_upper = task_name.upper()
    for key in STRUCTURED_OUTPUT_INSTRUCTIONS:
        if key.upper() == task_upper:
            return STRUCTURED_OUTPUT_INSTRUCTIONS[key]
    
    # Fallback by keyword
    task_lower = task_name.lower()
    if "math" in task_lower:
        return STRUCTURED_OUTPUT_INSTRUCTIONS["MATH"]
    elif "crux" in task_lower:
        return STRUCTURED_OUTPUT_INSTRUCTIONS["CRUX-O"]
    else:
        return STRUCTURED_OUTPUT_DEFAULT


def build_first_agent_prompt(agent_prompt: str, task_name: str) -> str:
    """Build prompt for the first agent in workflow (includes task description)."""
    task_desc = get_task_description(task_name)
    output_instructions = get_structured_output_instructions(task_name)
    return f"{agent_prompt}\n\n{task_desc}\n{output_instructions}"


def build_agent_prompt(agent_prompt: str, task_name: str = "MBPP") -> str:
    """Build prompt for non-first agents (includes structured format instructions)."""
    output_instructions = get_structured_output_instructions(task_name)
    return f"{agent_prompt}\n{output_instructions}"


# ============== GPT-based Prompt Generation ==============

PROMPT_GENERATION_SYSTEM = """You are an expert prompt engineer designing system prompts for AI agents in a multi-agent workflow.

CONTEXT:
- Agents exchange structured YAML messages with fields: problem:, work:, comment:, answer:.
- work: is private reasoning and never forwarded; final solutions must NOT be placed there.
- comment: is 1-3 short notes for the next agent (edges, risks, status of answer).
- answer: is the only solution field forwarded.

answer: RULES (role_mode is provided in the user message — never re-infer):
- IMPLEMENTATION roles: always place the full solution in answer:. If a prior answer exists, rewrite or improve it; never drop it or move it to work:. Do not leave answer: blank.
- VERIFICATION/ANALYSIS roles: if a prior answer exists, copy it into answer: and adjust only if needed. If no prior answer exists, leave answer: blank; do NOT invent a solution.

PROMPT REQUIREMENTS:
- State the agent identity/role in 1-2 sentences.
- Explain how to use problem:, comment:, answer: from the previous agent.
- Restate the required answer: behavior per role_mode above.
- Task-agnostic, concise (format instructions will be appended separately).
- Do NOT add formatting beyond plain text."""

PROMPT_GENERATION_USER_TEMPLATE = """Generate a system prompt for this agent:

ROLE NAME: {role_name}
RESPONSIBILITY: {responsibility}
ROLE MODE: {role_mode}  # implementation=create/refresh solutions, verification=copy/fix existing answers only

Use the provided ROLE MODE (do not re-infer) to describe answer: handling:
- IMPLEMENTATION: always put the full solution in answer:; if a prior answer exists, rewrite/improve it, never drop it.
- VERIFICATION: if a prior answer exists, copy it into answer: and only change it when needed; if none exists, leave answer: blank and focus on analysis.

The prompt should:
1. Define the agent's identity and role clearly (1-2 sentences)
2. Explain what the agent does with problem:, comment:, and answer: from input
3. Restate the required answer: behavior from ROLE MODE
4. Be task-agnostic (works for any benchmark)
5. Be concise - the format instructions will be appended separately

Generate ONLY the system prompt text (without format instructions), nothing else."""


class GPTPromptGenerator:
    """Generate agent prompts using GPT API."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        if OpenAI is None:
            raise ImportError("openai package is required for GPT prompt generation. Install with: pip install openai")
        
        self.model = model
        
        # Try to get API key from various sources
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try config first
            try:
                from src.config import OPENAI_API_KEY
                if OPENAI_API_KEY:
                    self.client = OpenAI(api_key=OPENAI_API_KEY)
                    return
            except ImportError:
                pass
            
            # Try environment variable
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                self.client = OpenAI(api_key=env_key)
            else:
                raise ValueError(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
    
    def generate_single_prompt(self, role_name: str, responsibility: str) -> str:
        """Generate a prompt for a single agent role. GPT decides if it's verification or implementation."""
        behavior = determine_role_behavior(role_name)
        role_mode = "implementation" if behavior == "create" else "verification"
        user_content = PROMPT_GENERATION_USER_TEMPLATE.format(
            role_name=role_name,
            responsibility=responsibility,
            role_mode=role_mode,
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PROMPT_GENERATION_SYSTEM},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    
    def generate_all_prompts(self, max_workers: int = 5) -> Dict[str, str]:
        """Generate prompts for all agents using parallel API calls."""
        prompts = {}
        total = len(AGENT_ROLES)
        
        print(f"Generating {total} prompts using GPT API ({self.model})...")
        print(f"Using {max_workers} parallel workers\n")
        
        def generate_one(args):
            idx, (role_name, responsibility) = args
            prompt = self.generate_single_prompt(role_name, responsibility)
            return idx, role_name, prompt
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_one, (i, role)): i 
                for i, role in enumerate(AGENT_ROLES)
            }
            
            completed = 0
            for future in as_completed(futures):
                try:
                    idx, role_name, prompt = future.result()
                    prompts[role_name] = prompt
                    completed += 1
                    print(f"  [{completed}/{total}] Generated: {role_name}")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        return prompts


def generate_prompts_template() -> Dict[str, str]:
    """
    Generate prompts deterministically using templates and role behaviors.
    Avoids reliance on external APIs and enforces strict answer: field handling.
    """
    prompts: Dict[str, str] = {}
    for role_name, responsibility in AGENT_ROLES:
        behavior = determine_role_behavior(role_name)
        prompts[role_name] = build_role_prompt(role_name, responsibility, behavior)
    return prompts


# ============== YAML Save/Load Functions ==============

def save_prompts_to_yaml(prompts: dict, output_path: Optional[str] = None):
    """Save all generated prompts to YAML file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "configs" / "generated_prompts.yaml"
    
    output = {
        "agents": prompts,
        "task_descriptions": TASK_DESCRIPTIONS,
        "structured_output_instructions": STRUCTURED_OUTPUT_INSTRUCTIONS,
        "extractors": EXTRACTOR_PROMPTS,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Prompt Configuration for Evolving Workflows\n")
        f.write("# Generated by generate_prompts.py\n")
        f.write("# Agent prompts are task-agnostic\n")
        f.write("# Task descriptions are appended to first agent only\n")
        f.write("# Structured output instructions are dataset-specific\n\n")
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n✓ Saved prompts to {output_path}")
    return output_path


def load_prompts_from_yaml(path: Optional[str] = None) -> dict:
    """Load prompts from YAML file."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "generated_prompts.yaml"
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_stats(prompts: dict):
    """Print statistics about generated prompts."""
    print("\n=== Prompt Generation Statistics ===")
    print(f"Total Agent Roles: {len(prompts)}")
    print(f"Task Descriptions: {len(TASK_DESCRIPTIONS)}")
    print(f"Structured Output Formats: {len(STRUCTURED_OUTPUT_INSTRUCTIONS)}")
    print(f"Extractor Types: {len(EXTRACTOR_PROMPTS)}")
    print("\nAgent Roles:")
    for i, role in enumerate(prompts.keys(), 1):
        print(f"  {i:2d}. {role}")
    print("\nTasks:", ", ".join(TASK_DESCRIPTIONS.keys()))
    print("Structured Output Formats:", ", ".join(STRUCTURED_OUTPUT_INSTRUCTIONS.keys()))
    print("Extractors:", ", ".join(EXTRACTOR_PROMPTS.keys()))


# ============== Main ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate agent prompts using GPT API")
    parser.add_argument("--model", default="gpt-4o-mini", help="GPT model to use (for --method gpt)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (gpt mode only)")
    parser.add_argument("--output", default=None, help="Output YAML path")
    parser.add_argument("--method", choices=["template", "gpt"], default="template", help="Prompt generation method")
    args = parser.parse_args()
    
    print("🚀 Generating Prompts for Evolving Workflows...")
    print(f"   Method: {args.method}")
    if args.method == "gpt":
        print(f"   Model: {args.model}")
        print(f"   Workers: {args.workers}")
    print()
    
    if args.method == "template":
        prompts = generate_prompts_template()
    else:
        generator = GPTPromptGenerator(model=args.model)
        prompts = generator.generate_all_prompts(max_workers=args.workers)
    
    # Save
    output_path = save_prompts_to_yaml(prompts, args.output)
    
    # Stats
    print_stats(prompts)
    
    # Example
    print("\n=== Example: First Agent Prompt for MBPP ===")
    first_role = list(prompts.keys())[0]
    example = build_first_agent_prompt(prompts[first_role], "MBPP")
    print(example[:600] + "...")
