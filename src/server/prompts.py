# src/server/prompts.py
"""
Prompt system for the evaluation server.

Following LangChain best practices:
- System Message: Role/persona definition ONLY (who you are, how you behave)
- Human Message: Task details, format requirements, the actual problem
"""

from typing import Optional, Dict, Any

from ..utils.generate_prompts import (
    AGENT_ROLES,
    TASK_DESCRIPTIONS,
    get_task_description,
    is_creator_role,
)


# Build role description lookup
ROLE_DESCRIPTIONS: Dict[str, str] = {name.lower(): desc for name, desc in AGENT_ROLES}
ROLE_NAMES: Dict[str, str] = {name.lower(): name for name, _ in AGENT_ROLES}


def get_extractor_system_prompt() -> str:
    """Get system prompt for extractor (role only)."""
    return """You are an Answer Extractor.

Your ONLY job is to extract and return the final answer from a solution.
You do NOT solve problems. You do NOT explain. You extract."""


def get_extractor_user_prompt(task_name: str, answer_text: str) -> str:
    """Get user prompt for extractor with format requirements."""
    task_upper = task_name.upper()
    
    if task_upper == "MBPP":
        format_instruction = """Extract ONLY the Python function code.
- Include imports if present
- Remove explanations, comments outside code, test cases
- Return raw Python code only"""
    elif task_upper == "MATH":
        format_instruction = """Extract ONLY the final answer from \\boxed{...}.
- Find the last \\boxed{} expression
- Return just the \\boxed{...} expression
- Nothing else"""
    elif task_upper == "CRUX-O":
        format_instruction = """Extract ONLY the predicted output value.
- Return the Python literal (list, dict, int, str, etc.)
- NO assert, NO variable names, NO explanation
- Just the raw value"""
    else:
        format_instruction = "Extract the final answer in its cleanest form."
    
    return f"""{format_instruction}

INPUT TO EXTRACT FROM:
{answer_text}"""


def get_agent_system_prompt(role: str) -> str:
    """
    Build system prompt for an agent - ROLE/PERSONA ONLY.
    
    System prompt defines WHO the agent is and HOW they should behave.
    It does NOT contain task details or format requirements.
    
    Args:
        role: Agent role name
    """
    role_lower = role.lower()
    
    # Find the role description
    role_desc = ROLE_DESCRIPTIONS.get(role_lower)
    
    # Try partial match if exact match fails
    if role_desc is None:
        for key, desc in ROLE_DESCRIPTIONS.items():
            if role_lower in key or key in role_lower:
                role_desc = desc
                break
    
    if role_desc is None:
        role_desc = "You analyze and process problems according to your expertise."
    
    # Determine if this is a creator or reviewer role
    is_creator = is_creator_role(role)
    
    # Build role-specific behavior guidance
    if is_creator:
        behavior = """YOUR BEHAVIOR:
- You are a CREATOR. Your job is to PRODUCE solutions.
- If no prior solution exists: Create one from scratch.
- If a prior solution exists: Improve, fix, or rewrite it.
- ALWAYS output a complete solution in the answer field.
- Never leave answer empty."""
    else:
        behavior = """YOUR BEHAVIOR:
- You are a REVIEWER/ANALYST. You CANNOT create solutions from scratch.
- If a prior solution exists: Analyze it, find issues, copy it to answer with improvements.
- If NO prior solution exists: Put exactly this in the answer field:
  "[NO_SOLUTION_TO_REVIEW] - Waiting for a creator agent to provide initial solution"
- NEVER write code if there is no prior solution to review.
- Your value is in ANALYSIS, not GENERATION."""
    
    # System prompt is about identity and behavior
    return f"""You are the {role}.

EXPERTISE: {role_desc}

{behavior}

Be precise. Be concise. Stay in your lane."""


def build_agent_input(
    problem_text: str,
    prev_response: Optional[Any],
    prev_role: str,
    current_role: str,
    is_first: bool,
    task_name: str,
    position_info: str = ""
) -> str:
    """
    Build the user message input for an agent.
    
    User message contains:
    - Task description (what benchmark, what format)
    - Output format requirements
    - The actual problem
    - Previous agent's output (if applicable)
    
    Args:
        problem_text: Original problem text
        prev_response: Structured response from previous agent (None for first agent)
        prev_role: Role of the previous agent
        current_role: Role of the current agent
        is_first: Whether this is the first agent
        task_name: Task/benchmark name
        position_info: Agent position info (e.g., "Agent 2 of 3")
    """
    # Task-specific instructions
    task_desc = get_task_description(task_name)
    
    # Check if this is a creator or reviewer
    is_creator = is_creator_role(current_role)
    
    # Output format instructions (this goes in user message, not system)
    if is_creator:
        format_instructions = """
REQUIRED OUTPUT FORMAT:
1. think: Brief reasoning (under 300 words). Private - not forwarded.
2. problem: Copy the original problem EXACTLY. Do not modify.
3. comment: 1-3 sentences for next agent. Key insights or concerns.
4. answer: Your complete solution code or derivation."""
    else:
        format_instructions = """
REQUIRED OUTPUT FORMAT:
1. think: Brief reasoning (under 300 words). Private - not forwarded.
2. problem: Copy the original problem EXACTLY. Do not modify.
3. comment: 1-3 sentences for next agent. Key insights or concerns.
4. answer: If prior solution exists, copy it with your improvements. 
   If NO prior solution exists, put exactly: [NO_SOLUTION_TO_REVIEW]"""

    # Build the user message
    parts = []
    
    # Position context
    if position_info:
        parts.append(f"[{position_info}]")
    
    # Role reminder - stick to your role!
    if is_creator:
        parts.append("⚡ REMEMBER: You are a CREATOR. Your job is to PRODUCE code/solutions.")
    else:
        parts.append("⚡ REMEMBER: You are a REVIEWER. Do NOT create code from scratch. Only review existing solutions.")
    
    # Task description
    parts.append(f"\nTASK:\n{task_desc}")
    
    # Format requirements
    parts.append(format_instructions)
    
    if is_first:
        # First agent gets the raw problem
        parts.append(f"""
---
PROBLEM TO SOLVE:
{problem_text}
---

You are the FIRST agent. Analyze this problem and begin working on it.""")
    else:
        # Subsequent agents get previous agent's work
        prev_problem = prev_response.problem if prev_response else problem_text
        prev_comment = prev_response.comment if prev_response else "(no comment)"
        prev_answer = prev_response.answer if prev_response else "(no solution yet)"
        
        # Check if previous agent refused (no solution to review)
        no_solution_marker = "[NO_SOLUTION_TO_REVIEW]"
        has_no_solution = no_solution_marker in prev_answer if prev_answer else True
        
        if has_no_solution:
            # Previous agent had no solution - tell current agent there's STILL no solution
            parts.append(f"""
---
ORIGINAL PROBLEM:
{prev_problem}

⚠️ WARNING: NO SOLUTION EXISTS YET ⚠️
The previous agent ({prev_role}) could not provide a solution.
Previous answer was: {prev_answer}

There is STILL NO CODE/SOLUTION to review.
---

If you are a REVIEWER: Put exactly [NO_SOLUTION_TO_REVIEW] in your answer field.
If you are a CREATOR: Create the initial solution.""")
        else:
            # Normal case - there IS a solution to work with
            parts.append(f"""
---
ORIGINAL PROBLEM:
{prev_problem}

PREVIOUS AGENT ({prev_role}) SAID:
{prev_comment}

CURRENT SOLUTION:
{prev_answer}
---

Review the above. Apply your expertise to improve, verify, or refine.""")
    
    return "\n".join(parts)
