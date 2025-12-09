# src/utils/generate_prompts.py
"""
Prompt Generator for Evolving Workflows - Uses GPT API

Design Principles:
1. Agent prompts are TASK-AGNOSTIC (same for MBPP, MATH, CRUX-O, etc.)
2. Task description is appended only to the FIRST agent in the workflow
3. Extractors are DATASET-SPECIFIC (code vs math vs output prediction)
4. All agents follow PASS-THROUGH pattern (pass relevant info to next agent)
5. User prompts are SIMPLE to force collaboration between agents
"""

import yaml
import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv

load_dotenv()

# ============== Configuration ==============

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============== Structured Output Format ==============
# All agents MUST output in this format. [PROBLEM], [COMMENT], [ANSWER] are passed to next agent. [WORK] is logged only.

STRUCTURED_OUTPUT_FORMAT = """
**OUTPUT FORMAT (MUST follow exactly):**

---BEGIN STRUCTURED OUTPUT---

[PROBLEM]
{COPY THE EXACT ORIGINAL PROBLEM INCLUDING TEST CASES - DO NOT MODIFY OR TRUNCATE}

[WORK]
{Your detailed analysis/reasoning - NOT passed to next agent}

[COMMENT]
{Brief notes for next agent: function name, edge cases, key insights (1-3 sentences)}

[ANSWER]
{Your working code/solution - function name MUST match the assert statement}

---END STRUCTURED OUTPUT---
"""

STRUCTURED_OUTPUT_INSTRUCTION = """
**CRITICAL OUTPUT RULES:**

1. [PROBLEM] - COPY VERBATIM. Include ALL text: problem description AND test cases (assert statements).
   - The test case contains the EXACT function name you MUST use.
   - DO NOT summarize, paraphrase, or truncate. Copy character-for-character.
   
2. [WORK] - Your detailed reasoning (logged only, NOT passed to next agent).

3. [COMMENT] - Brief notes for the next agent (1-3 sentences). Passed forward.
   - Use this to highlight: function name, edge cases, key insights, warnings.
   
4. [ANSWER] - Your concrete contribution: working code that passes the test case.
   - Function name MUST match the test case exactly.
   - Code must be complete and executable.

Passed to next agent: [PROBLEM], [COMMENT], [ANSWER]
NOT passed (logged only): [WORK]
"""


# 50 Agent Roles for Evolution
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
1. Trace through the code step-by-step in [WORK]
2. In [ANSWER], write ONLY the predicted output value
3. The answer must be a valid Python literal: list, tuple, dict, string, int, bool, etc.

CRITICAL: [ANSWER] must contain the actual predicted value like:
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
    "MBPP": """Extract the Python function.

OUTPUT: Raw code starting with "def" or "import". No markdown, no explanation.

Example:
def remove_Occ(s, ch):
    s = s.replace(ch, '', 1)
    return s[::-1].replace(ch, '', 1)[::-1]""",

    "MATH": """Extract the final answer.

OUTPUT: \\boxed{answer} only.

Examples: \\boxed{2}, \\boxed{\\frac{3}{4}}, \\boxed{x \\in [-2,7]}""",

    "CRUX-O": """Extract the output value.

OUTPUT: Python literal only. No "assert", no explanation.

Examples: [(4, 1), (2, 3)], 'hello', {1: None}, False""",
}


# ============== GPT-based Prompt Generation ==============

PROMPT_GENERATION_SYSTEM = """You are an expert prompt engineer designing system prompts for AI agents in a multi-agent workflow.

CONTEXT:
- These agents work together to solve benchmark problems (code generation, math problems, output prediction)
- Each agent has a specific role and receives input from the previous agent
- Agents use a STRUCTURED OUTPUT format to preserve the original problem and pass forward cleanly

THE STRUCTURED OUTPUT FORMAT:
All agents MUST output in this exact format:

---BEGIN STRUCTURED OUTPUT---
[PROBLEM]
{The exact original problem - copied verbatim}
[WORK]
{Detailed analysis - NOT passed to next agent}
[COMMENT]
{Brief notes for next agent: key insights, edge cases, warnings (1-3 sentences)}
[ANSWER]
{Concrete output passed forward}
---END STRUCTURED OUTPUT---

CRITICAL REQUIREMENTS:
1. The agent must ALWAYS copy the [PROBLEM] section verbatim from input
2. The agent can do detailed work in [WORK] (logged but not passed forward)
3. The agent should add brief notes in [COMMENT] (passed to next agent)
4. The agent's concrete contribution goes in [ANSWER]
5. The prompt should be task-agnostic (works for code, math, output prediction)
6. Be concise (100-200 words, excluding the format instructions)

Generate a professional, effective system prompt for the agent."""

PROMPT_GENERATION_USER_TEMPLATE = """Generate a system prompt for this agent:

ROLE NAME: {role_name}
RESPONSIBILITY: {responsibility}

The prompt should:
1. Define the agent's identity and role clearly (1-2 sentences)
2. Explain what the agent does with [PROBLEM], [COMMENT], and [ANSWER] from input
3. Explain what the agent contributes to [COMMENT] (brief notes) and [ANSWER] (solution)
4. MUST include the structured output format requirement
5. Be task-agnostic (works for any benchmark)
6. Be concise - the format instructions will be appended separately

Generate ONLY the system prompt text (without format instructions), nothing else."""


class GPTPromptGenerator:
    """Generate agent prompts using GPT API."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
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
            import os
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                self.client = OpenAI(api_key=env_key)
            else:
                raise ValueError(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
    
    def generate_single_prompt(self, role_name: str, responsibility: str) -> str:
        """Generate a prompt for a single agent role."""
        user_content = PROMPT_GENERATION_USER_TEMPLATE.format(
            role_name=role_name,
            responsibility=responsibility
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_GENERATION_SYSTEM},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ✗ Error generating prompt for {role_name}: {e}")
            # Fallback to template
            return self._fallback_prompt(role_name, responsibility)
    
    def _fallback_prompt(self, role_name: str, responsibility: str) -> str:
        """Fallback template if API fails."""
        return f"""You are a **{role_name}** in a multi-agent workflow.

Your responsibility: {responsibility}

You will receive input containing [PROBLEM], [COMMENT], and [ANSWER] sections from the previous agent.

YOUR TASK:
1. Read the [PROBLEM] section - this is the original task (DO NOT modify it)
2. Consider the [COMMENT] and [ANSWER] from previous agents
3. Do your specific work based on your role
4. Add brief notes in [COMMENT] for the next agent
5. Produce your contribution in the [ANSWER] section"""
    
    def generate_all_prompts(self, max_workers: int = 5) -> dict:
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
                idx, role_name, prompt = future.result()
                prompts[role_name] = prompt
                completed += 1
                print(f"  [{completed}/{total}] Generated: {role_name}")
        
        return prompts


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


def build_first_agent_prompt(agent_prompt: str, task_name: str) -> str:
    """Build prompt for the first agent in workflow (includes task description)."""
    task_desc = get_task_description(task_name)
    return f"{task_desc}\n\n{agent_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"


def build_agent_prompt(agent_prompt: str) -> str:
    """Build prompt for non-first agents (includes structured format instructions)."""
    return f"{agent_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"


# ============== Save Functions ==============

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
        f.write("# Generated by generate_prompts.py using GPT API\n")
        f.write("# Agent prompts are task-agnostic\n")
        f.write("# Task descriptions are appended to first agent only\n")
        f.write("# Extractors are dataset-specific\n\n")
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
    print(f"Extractor Types: {len(EXTRACTOR_PROMPTS)}")
    print("\nAgent Roles:")
    for i, role in enumerate(prompts.keys(), 1):
        print(f"  {i:2d}. {role}")
    print("\nTasks:", ", ".join(TASK_DESCRIPTIONS.keys()))
    print("Extractors:", ", ".join(EXTRACTOR_PROMPTS.keys()))


# ============== Main ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate agent prompts using GPT API")
    parser.add_argument("--model", default="gpt-4o-mini", help="GPT model to use")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--output", default=None, help="Output YAML path")
    parser.add_argument("--no-api", action="store_true", help="Use template fallback instead of API")
    args = parser.parse_args()
    
    print("🚀 Generating Prompts for Evolving Workflows...")
    print(f"   Model: {args.model}")
    print(f"   Workers: {args.workers}")
    print()
    
    if args.no_api:
        # Use fallback templates
        print("Using template fallback (--no-api flag)")
        generator = GPTPromptGenerator(model=args.model)
        prompts = {}
        for role_name, responsibility in AGENT_ROLES:
            prompts[role_name] = generator._fallback_prompt(role_name, responsibility)
            print(f"  Generated: {role_name}")
    else:
        # Use GPT API
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
