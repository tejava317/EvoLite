# src/evaluation_server.py
"""
FastAPI Evaluation Server for EvoLite.

High-throughput evaluation using RunPod's native async API.
Fire-all-at-once pattern for maximum parallelism.

Works with the new BlockWorkflow system.

Run with:
    uvicorn src.evaluation_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import time
import os
import json
import uuid
from typing import Optional, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import re
from src.llm.runpod_client import RunPodAsyncClient, JobResult
from src.datasets import MBPPDataset, MathAlgebraDataset, CRUXOpenDataset
from src.datasets.base import Problem
from src.evaluation.executor import execute_code
from src.config import get_predefined_prompt, BASE_AGENTS, INITIAL_PROMPTS


# ============== Pydantic Models ==============

class BlockConfig(BaseModel):
    """Configuration for a single block."""
    type: str = Field(..., description="Block type: 'agent' or 'composite'")
    role: Optional[str] = Field(default=None, description="Role name for agent blocks")
    divider_role: Optional[str] = Field(default="Divider", description="Divider role for composite blocks")
    synth_role: Optional[str] = Field(default="Synthesizer", description="Synthesizer role for composite blocks")


class WorkflowConfig(BaseModel):
    """Configuration for a BlockWorkflow to evaluate."""
    blocks: List[BlockConfig] = Field(..., description="List of block configurations")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")
    think: bool = Field(default=False, description="Enable thinking mode (/think). Default is /no_think")


class SimpleWorkflowConfig(BaseModel):
    """Simple config with just role names (converts to AgentBlocks)."""
    roles: List[str] = Field(..., description="List of agent role names")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")
    think: bool = Field(default=False, description="Enable thinking mode (/think). Default is /no_think")


class EvaluateRequest(BaseModel):
    """Request to evaluate a single workflow."""
    workflow: WorkflowConfig
    num_problems: int = Field(default=10, ge=1, le=5000)
    seed: Optional[int] = Field(default=None, description="Random seed for problem sampling")


class SimpleEvaluateRequest(BaseModel):
    """Simplified request using just role names."""
    roles: List[str] = Field(..., description="List of agent role names")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")
    think: bool = Field(default=False, description="Enable thinking mode (/think). Default is /no_think")
    num_problems: int = Field(default=10, ge=1, le=5000)
    seed: Optional[int] = Field(default=None, description="Random seed for problem sampling")


class BatchEvaluateRequest(BaseModel):
    """Request to evaluate multiple workflows."""
    workflows: List[WorkflowConfig]
    num_problems: int = Field(default=10, ge=1, le=5000)
    seed: Optional[int] = Field(default=None)


class ProblemResult(BaseModel):
    """Result for a single problem."""
    problem_id: str
    correct: bool
    tokens: int
    time: float
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Response from evaluation."""
    pass_at_1: float
    num_correct: int
    num_problems: int
    total_tokens: int
    total_time: float
    tokens_per_second: float
    problems: List[ProblemResult]


# ============== Global State ==============

@dataclass
class ServerState:
    """Cached server state."""
    datasets: dict = None
    runpod_client: RunPodAsyncClient = None
    extractor_prompts: dict = None


state = ServerState()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup, cleanup at shutdown."""
    print("🚀 Starting Evaluation Server...")
    
    # Initialize RunPod client with monitoring enabled
    state.runpod_client = RunPodAsyncClient(
        default_temperature=0.1,
        default_max_tokens=2000,
        poll_interval=0.3,
        max_poll_time=3600,  # 1 hour timeout window
        enable_monitoring=True,
    )
    print("✓ RunPod client initialized (monitoring enabled)")
    
    # Pre-load datasets
    print("📂 Loading datasets...")
    state.datasets = {}
    
    try:
        mbpp = MBPPDataset(split="test")
        mbpp.load()
        state.datasets["MBPP"] = mbpp
        print(f"  ✓ MBPP: {len(mbpp)} problems")
    except Exception as e:
        print(f"  ✗ MBPP failed: {e}")
    
    try:
        math_ds = MathAlgebraDataset(split="test")
        math_ds.load()
        state.datasets["MATH"] = math_ds
        print(f"  ✓ MATH: {len(math_ds)} problems")
    except Exception as e:
        print(f"  ✗ MATH failed: {e}")

    try:
        crux = CRUXOpenDataset(split="test")
        crux.load()
        state.datasets["CRUX-O"] = crux
        print(f"  ✓ CRUX-O: {len(crux)} problems")
    except Exception as e:
        print(f"  ✗ CRUX-O failed: {e}")
    
    # Cache extractor prompts (keyed by benchmark name)
    state.extractor_prompts = {
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

Examples: [(4, 1), (2, 3)], 'hello', {1: None}, False"""
    }
    
    print("✓ Server ready!")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")
    if state.runpod_client:
        await state.runpod_client.close()


# ============== FastAPI App ==============

app = FastAPI(
    title="EvoLite Evaluation Server",
    description="High-throughput BlockWorkflow evaluation using RunPod async API",
    version="2.0.0",
    lifespan=lifespan
)


# Silence access logs for noisy endpoints (e.g., /stats polling)
class _EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/stats " not in msg


logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())


# ============== Helper Functions ==============

# Load GPT-generated prompts from YAML
GENERATED_PROMPTS = {}
GENERATED_TASK_DESCRIPTIONS = {}
GENERATED_EXTRACTORS = {}

try:
    from src.utils.generate_prompts import load_prompts_from_yaml, get_task_description as gen_task_desc
    _loaded = load_prompts_from_yaml()
    if _loaded:
        GENERATED_PROMPTS = {k.lower(): v for k, v in _loaded.get("agents", {}).items()}
        GENERATED_TASK_DESCRIPTIONS = _loaded.get("task_descriptions", {})
        GENERATED_EXTRACTORS = _loaded.get("extractors", {})
        print(f"  ✓ Loaded {len(GENERATED_PROMPTS)} GPT-generated agent prompts")
    USE_NEW_PROMPTS = bool(GENERATED_PROMPTS)
except Exception as e:
    print(f"  ✗ Could not load generated prompts: {e}")
    USE_NEW_PROMPTS = False


def get_task_description(task_name: str) -> str:
    """Get task description for a dataset."""
    # Try loaded descriptions first
    if task_name in GENERATED_TASK_DESCRIPTIONS:
        return GENERATED_TASK_DESCRIPTIONS[task_name]
    
    # Case-insensitive match
    for key, desc in GENERATED_TASK_DESCRIPTIONS.items():
        if key.lower() == task_name.lower():
            return desc
    
    # Fallback
    return f"""
=== TASK: {task_name} ===
Goal: Complete the given task according to the problem description.
Input: Problem description
Output: Solution in the required format"""


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


def get_agent_prompt(role: str, task_name: str = None, is_first_agent: bool = False) -> str:
    """
    Get prompt for an agent role with STRUCTURED OUTPUT format.
    
    New system:
    - All agents must output in structured format: [PROBLEM], [INSIGHTS], [WORK], [ANSWER]
    - Only [PROBLEM], [INSIGHTS], [ANSWER] are passed to next agent
    - [WORK] is logged but not passed (keeps chain clean)
    - Task description appended only for first agent
    """
    # Try GPT-generated prompts first
    if USE_NEW_PROMPTS:
        role_lower = role.lower()
        if role_lower in GENERATED_PROMPTS:
            base_prompt = GENERATED_PROMPTS[role_lower]
        else:
            # Try partial match
            for key, prompt in GENERATED_PROMPTS.items():
                if role_lower in key or key in role_lower:
                    base_prompt = prompt
                    break
            else:
                # Generate generic prompt for unknown roles
                base_prompt = f"""You are a **{role}** in a multi-agent workflow.

You will receive input containing [PROBLEM] and [ANSWER] sections.

YOUR TASK:
1. Read [PROBLEM] - this is the original task (copy it EXACTLY to your output)
2. Consider [ANSWER] from previous agents
3. Do your work based on your role as {role}
4. Output your contribution in the structured format"""
        
        # Append task description for first agent
        if is_first_agent and task_name:
            task_desc = get_task_description(task_name)
            return f"{task_desc}\n\n{base_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"
        
        # Non-first agents also get the structured format requirement
        return f"{base_prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"
    
    # Fallback to old system with structured format
    prompt = get_predefined_prompt(role, task_name)
    if prompt:
        return f"{prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"
    
    fallback = f"""You are a **{role}** in a multi-agent workflow.

You will receive input containing [PROBLEM] and [ANSWER] sections.

YOUR TASK:
1. Read [PROBLEM] - copy it EXACTLY to your output
2. Consider [ANSWER] from previous agents  
3. Do your work based on your role
4. Output your contribution"""
    
    return f"{fallback}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}{STRUCTURED_OUTPUT_FORMAT}"


def get_extractor_prompt(task_name: str) -> str:
    """Get extractor prompt for task/benchmark."""
    # Try exact match in generated extractors first
    if USE_NEW_PROMPTS and GENERATED_EXTRACTORS:
        if task_name in GENERATED_EXTRACTORS:
            return GENERATED_EXTRACTORS[task_name]
        # Try case variations
        for key in GENERATED_EXTRACTORS:
            if key.upper() == task_name.upper():
                return GENERATED_EXTRACTORS[key]
    
    # Try exact match in cached prompts
    if task_name in state.extractor_prompts:
        return state.extractor_prompts[task_name]
    
    # Try case variations
    for key in state.extractor_prompts:
        if key.upper() == task_name.upper():
            return state.extractor_prompts[key]
    
    # Fallback by keyword
    task_lower = task_name.lower()
    if "math" in task_lower:
        return state.extractor_prompts["MATH"]
    if "crux" in task_lower:
        return state.extractor_prompts["CRUX-O"]
    return state.extractor_prompts["MBPP"]


def expand_blocks_to_roles(blocks: List[BlockConfig], client: RunPodAsyncClient = None) -> List[str]:
    """
    Expand block configuration to a list of agent roles.
    
    For agent blocks: just return the role
    For composite blocks: return [divider, inner_role1, inner_role2, ..., synthesizer]
    
    Note: For async expansion of composite blocks, we'd need an LLM call.
    For simplicity, we use default inner roles for composite blocks.
    """
    roles = []
    
    for block in blocks:
        if block.type == "agent":
            if block.role:
                roles.append(block.role)
        elif block.type == "composite":
            # Composite block expands to: divider -> [inner roles] -> synthesizer
            # Default inner roles (in production, these would be dynamically generated)
            roles.append(block.divider_role or "Divider")
            # Add default inner roles for composite block
            roles.extend(["Business Analyst", "Technical Lead", "Quality Assurance"])
            roles.append(block.synth_role or "Synthesizer")
    
    return roles


# File locks for concurrent log writes
_log_locks: dict[str, asyncio.Lock] = {}
_log_locks_lock = asyncio.Lock()

# In-memory storage for problem logs (grouped by request_id -> problem_id)
_problem_logs: dict[str, dict[str, dict]] = {}


async def _get_log_lock(request_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific request_id."""
    async with _log_locks_lock:
        if request_id not in _log_locks:
            _log_locks[request_id] = asyncio.Lock()
        return _log_locks[request_id]


async def log_agent_output(request_id: str, record: dict):
    """Collect agent output in memory (grouped by problem)."""
    lock = await _get_log_lock(request_id)
    
    async with lock:
        problem_id = record.get("problem_id", "unknown")
        
        # Initialize storage for this request if needed
        if request_id not in _problem_logs:
            _problem_logs[request_id] = {}
        
        # Initialize storage for this problem if needed
        if problem_id not in _problem_logs[request_id]:
            _problem_logs[request_id][problem_id] = {
                "problem_id": problem_id,
                "task": record.get("task", ""),
                "request_timestamp": record.get("request_timestamp", ""),
                "agents": [],
                "extractor": None,
                "expected_answer": None,
                "correct": None,
            }
        
        prob_log = _problem_logs[request_id][problem_id]
        
        # Add agent step
        step = {
            "stage": record.get("stage", 0),
            "role": record.get("role", "unknown"),
            "is_extractor": record.get("is_extractor", False),
            "system_prompt": record.get("system_prompt", ""),
            "input": record.get("user_input", ""),
            "output": record.get("content", ""),
            "tokens": record.get("tokens", 0),
            "elapsed": record.get("elapsed", 0),
            "status": record.get("status", ""),
            "error": record.get("error"),
        }
        
        # Add parsed sections if available (for structured output)
        if record.get("parsed_problem"):
            step["parsed"] = {
                "problem": record.get("parsed_problem", ""),
                "work": record.get("parsed_work", ""),
                "comment": record.get("parsed_comment", ""),
                "answer": record.get("parsed_answer", ""),
            }
        
        if record.get("is_extractor"):
            prob_log["extractor"] = step
        else:
            prob_log["agents"].append(step)


async def finalize_problem_log(request_id: str, problem_id: str, expected_answer: str, correct: bool):
    """Finalize a problem's log with the expected answer and correctness."""
    lock = await _get_log_lock(request_id)
    
    async with lock:
        if request_id in _problem_logs and problem_id in _problem_logs[request_id]:
            _problem_logs[request_id][problem_id]["expected_answer"] = expected_answer
            _problem_logs[request_id][problem_id]["correct"] = correct


async def save_request_logs(request_id: str, request_ts: str):
    """Save all problem logs for a request to a single JSON file (grouped by problem)."""
    lock = await _get_log_lock(request_id)
    
    async with lock:
        if request_id not in _problem_logs:
            return
        
        def _write():
            dir_path = os.path.join("logs", "agent_runs", request_ts)
            path = os.path.join(dir_path, f"{request_id}.json")
            os.makedirs(dir_path, exist_ok=True)
            
            # Convert to list sorted by problem_id
            problems = list(_problem_logs[request_id].values())
            problems.sort(key=lambda x: x["problem_id"])
            
            # Format for readability
            output = {
                "request_id": request_id,
                "timestamp": request_ts,
                "total_problems": len(problems),
                "correct_count": sum(1 for p in problems if p.get("correct")),
                "problems": problems,
            }
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        
        await asyncio.to_thread(_write)
        
        # Clean up memory
        del _problem_logs[request_id]


async def expand_composite_block_async(
    block: BlockConfig,
    client: RunPodAsyncClient,
    context: str = ""
) -> List[str]:
    """
    Dynamically expand a composite block using LLM to determine inner roles.
    """
    divider_prompt = f"""You are a task divider. Given a task, identify 3 specific roles needed to complete it effectively.

Task context: {context if context else "General software development task"}

List exactly 3 roles, one per line. Just the role names, nothing else."""

    result = await client.generate(
        system_prompt="You are a helpful assistant that identifies team roles.",
        user_content=divider_prompt
    )
    
    if result.status != "COMPLETED":
        # Fallback to default roles
        return ["Business Analyst", "Technical Lead", "Quality Assurance"]
    
    # Parse roles from response
    lines = result.content.strip().split("\n")
    inner_roles = []
    for line in lines[:3]:
        # Clean up the line (remove numbers, bullets, etc.)
        role = line.strip().lstrip("0123456789.-) ").strip()
        if role:
            inner_roles.append(role)
    
    if len(inner_roles) < 3:
        inner_roles.extend(["Business Analyst", "Technical Lead", "Quality Assurance"][:3 - len(inner_roles)])
    
    return inner_roles


def get_think_suffix(think: bool) -> str:
    """Get the thinking mode suffix for prompts."""
    return " /think" if think else " /no_think"


# ============== Structured Output Parsing ==============

def parse_structured_output(output: str) -> dict:
    """
    Parse the structured output from an agent.
    
    Returns dict with keys: problem, work, comment, answer, raw
    If parsing fails, returns the raw output in answer field.
    """
    result = {
        "problem": "",
        "work": "",
        "comment": "",
        "answer": "",
        "raw": output,
        "parsed": False,
    }
    
    # Try to parse structured format
    # Look for [SECTION] markers
    sections = {
        "problem": r'\[PROBLEM\]\s*(.*?)(?=\[WORK\]|\[COMMENT\]|\[ANSWER\]|---END|$)',
        "work": r'\[WORK\]\s*(.*?)(?=\[COMMENT\]|\[ANSWER\]|---END|$)',
        "comment": r'\[COMMENT\]\s*(.*?)(?=\[ANSWER\]|---END|$)',
        "answer": r'\[ANSWER\]\s*(.*?)(?=---END|$)',
    }
    
    for key, pattern in sections.items():
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()
            result["parsed"] = True
    
    # If no structured format found, treat entire output as answer
    if not result["parsed"]:
        result["answer"] = output.strip()
    
    return result


def build_passthrough_input(original_problem: str, parsed_output: dict) -> str:
    """
    Build the input for the next agent using PROBLEM, COMMENT, and ANSWER.
    
    [WORK] is intentionally excluded to keep the chain clean.
    """
    # If we have a properly parsed problem, use it
    problem = parsed_output.get("problem", "").strip()
    if not problem:
        problem = original_problem
    
    comment = parsed_output.get("comment", "").strip()
    answer = parsed_output.get("answer", "").strip()
    
    # Build the input for next agent
    parts = ["---BEGIN STRUCTURED OUTPUT---", "", "[PROBLEM]", problem, ""]
    
    if comment:
        parts.extend(["[COMMENT]", comment, ""])
    
    if answer:
        parts.extend(["[ANSWER]", answer, ""])
    
    parts.append("---END STRUCTURED OUTPUT---")
    
    return "\n".join(parts)


def format_first_agent_input(problem_prompt: str) -> str:
    """
    Format the problem prompt as structured input for the first agent.
    """
    return f"""---BEGIN STRUCTURED OUTPUT---

[PROBLEM]
{problem_prompt}

[ANSWER]
<PUT YOUR ANSWER HERE - DO NOT LEAVE THIS PLACEHOLDER>

---END STRUCTURED OUTPUT---"""


async def run_workflow_on_problem(
    problem: Problem,
    blocks: List[BlockConfig],
    task_name: str,
    use_extractor: bool,
    client: RunPodAsyncClient,
    poll_interval: float = 0.01,
    request_id: str = "",
    request_ts: str = "",
    think: bool = False,
) -> tuple[str, int, float, str]:
    """
    Run a BlockWorkflow on a single problem using fire-and-poll with STRUCTURED OUTPUT format.
    
    The structured format ensures:
    - [PROBLEM] is ALWAYS preserved and passed through unchanged
    - [INSIGHTS] accumulates brief insights from each agent
    - [WORK] is logged but NOT passed to next agent (keeps chain clean)
    - [ANSWER] contains the concrete output passed forward
    
    Returns: (final_output, total_tokens, execution_time, error)
    """
    start_time = time.time()
    roles = expand_blocks_to_roles(blocks, client)
    think_suffix = get_think_suffix(think)

    # Per-problem state - preserve original problem for the entire chain
    original_problem = problem.prompt
    current_input = format_first_agent_input(original_problem)  # Structured format for first agent
    total_tokens = 0
    error: Optional[str] = None

    pending = []
    # submit first role (with task description)
    if roles:
        prompt = get_agent_prompt(roles[0], task_name, is_first_agent=True) + think_suffix
        job_id = await client.submit_job(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": current_input},
            ]
        )
        pending.append({
            "job_id": job_id, "stage": 0, "is_extractor": False, "role": roles[0],
            "system_prompt": prompt, "user_input": current_input, "original_problem": original_problem
        })
    else:
        # no roles, go straight to extractor
        if use_extractor:
            extractor_prompt = get_extractor_prompt(task_name) + think_suffix
            job_id = await client.submit_job(
                messages=[
                    {"role": "system", "content": extractor_prompt},
                    {"role": "user", "content": original_problem},  # Extractor gets raw problem
                ]
            )
            pending.append({
                "job_id": job_id, "stage": 0, "is_extractor": True,
                "system_prompt": extractor_prompt, "user_input": original_problem, "original_problem": original_problem
            })
        else:
            return original_problem, total_tokens, time.time() - start_time, None

    # Poll loop
    while pending:
        next_pending = []
        for item in pending:
            res = await client.poll_job_once(item["job_id"], start_time)
            if res is None:
                next_pending.append(item)
                continue
            if res.status != "COMPLETED":
                error = res.error or f"Job failed with status {res.status}"
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": item["stage"],
                    "role": item.get("role"),
                    "is_extractor": item["is_extractor"],
                    "system_prompt": item.get("system_prompt", ""),
                    "user_input": item.get("user_input", ""),
                    "status": res.status,
                    "error": res.error,
                    "content": "",
                    "tokens": 0,
                    "elapsed": res.execution_time,
                    "request_timestamp": request_ts,
                })
                continue

            total_tokens += res.total_tokens
            raw_output = res.content
            orig_problem = item.get("original_problem", original_problem)

            if item["is_extractor"]:
                # Extractor output is final - just extract the answer
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": item["stage"],
                    "role": "extractor",
                    "is_extractor": True,
                    "system_prompt": item.get("system_prompt", ""),
                    "user_input": item.get("user_input", ""),
                    "status": res.status,
                    "error": None,
                    "content": raw_output,
                    "tokens": res.total_tokens,
                    "elapsed": res.execution_time,
                    "request_timestamp": request_ts,
                })
                current_input = raw_output  # Final answer
                continue

            # Regular agent - parse structured output
            parsed = parse_structured_output(raw_output)
            
            # Log the FULL output (including [WORK])
            await log_agent_output(request_id, {
                "problem_id": problem.id,
                "task": task_name,
                "stage": item["stage"],
                "role": item.get("role"),
                "is_extractor": False,
                "system_prompt": item.get("system_prompt", ""),
                "user_input": item.get("user_input", ""),
                "status": res.status,
                "error": None,
                "content": raw_output,  # Full output including [WORK]
                "parsed_problem": parsed.get("problem", ""),
                "parsed_work": parsed.get("work", ""),
                "parsed_comment": parsed.get("comment", ""),
                "parsed_answer": parsed.get("answer", ""),
                "tokens": res.total_tokens,
                "elapsed": res.execution_time,
                "request_timestamp": request_ts,
            })
            
            # Build pass-through input: only [PROBLEM], [INSIGHTS], [ANSWER] - NOT [WORK]
            passthrough_input = build_passthrough_input(orig_problem, parsed)
            current_input = passthrough_input
            
            next_stage = item["stage"] + 1
            if next_stage < len(roles):
                # Subsequent agents don't get task description (is_first_agent=False)
                prompt = get_agent_prompt(roles[next_stage], task_name, is_first_agent=False) + think_suffix
                job_id = await client.submit_job(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": passthrough_input},  # Only PROBLEM, INSIGHTS, ANSWER
                    ]
                )
                next_pending.append({
                    "job_id": job_id, "stage": next_stage, "is_extractor": False, "role": roles[next_stage],
                    "system_prompt": prompt, "user_input": passthrough_input, "original_problem": orig_problem
                })
            else:
                # finished agents, maybe extractor
                if use_extractor:
                    # Extractor gets the [ANSWER] section from last agent
                    extractor_input = parsed.get("answer", raw_output)
                    extractor_prompt = get_extractor_prompt(task_name) + think_suffix
                    job_id = await client.submit_job(
                        messages=[
                            {"role": "system", "content": extractor_prompt},
                            {"role": "user", "content": extractor_input},
                        ]
                    )
                    next_pending.append({
                        "job_id": job_id, "stage": next_stage, "is_extractor": True,
                        "system_prompt": extractor_prompt, "user_input": extractor_input, "original_problem": orig_problem
                    })
                # else done

        pending = next_pending
        if pending:
            await asyncio.sleep(poll_interval)

    return current_input, total_tokens, time.time() - start_time, error


async def evaluate_workflow_parallel(
    workflow: WorkflowConfig,
    problems: List[Problem],
    dataset,
    request_id: str,
    request_ts: str,
) -> EvaluateResponse:
    """
    Evaluate a BlockWorkflow on multiple problems in parallel.
    
    Uses fire-all-at-once pattern for maximum throughput.
    """
    start_time = time.time()
    client = state.runpod_client
    
    # Run all problems in parallel using fire-and-poll
    tasks = []
    for problem in problems:
        tasks.append(
            run_workflow_on_problem(
                problem,
                workflow.blocks,
                workflow.task_name,
                workflow.use_extractor,
                client,
                request_id=request_id,
                request_ts=request_ts,
                think=workflow.think,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    problem_results = []
    num_correct = 0
    total_tokens = 0
    
    for problem, result in zip(problems, results):
        if isinstance(result, Exception):
            problem_results.append(ProblemResult(
                problem_id=problem.id,
                correct=False,
                tokens=0,
                time=0,
                error=str(result)[:200]
            ))
            # Finalize log with error
            await finalize_problem_log(request_id, problem.id, str(problem.ground_truth), False)
        else:
            output, tokens, exec_time, err = result
            total_tokens += tokens
            if err:
                problem_results.append(ProblemResult(
                    problem_id=problem.id,
                    correct=False,
                    tokens=tokens,
                    time=exec_time,
                    error=str(err)[:200]
                ))
                # Finalize log with error
                await finalize_problem_log(request_id, problem.id, str(problem.ground_truth), False)
            else:
                try:
                    is_correct = dataset.evaluate(output, problem)
                except Exception as e:
                    is_correct = False
                    err = str(e)
                if is_correct:
                    num_correct += 1
                problem_results.append(ProblemResult(
                    problem_id=problem.id,
                    correct=is_correct,
                    tokens=tokens,
                    time=exec_time,
                    error=str(err)[:200] if err else None
                ))
                # Finalize log with expected answer and correctness
                # Priority: output (CRUX-O), answer (MATH), code (MBPP)
                expected = problem.ground_truth.get("output") or problem.ground_truth.get("answer") or problem.ground_truth.get("code") or str(problem.ground_truth)
                await finalize_problem_log(request_id, problem.id, expected, is_correct)
    
    # Save all logs for this request
    await save_request_logs(request_id, request_ts)
    
    total_time = time.time() - start_time
    
    return EvaluateResponse(
        pass_at_1=num_correct / len(problems) if problems else 0,
        num_correct=num_correct,
        num_problems=len(problems),
        total_tokens=total_tokens,
        total_time=total_time,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        problems=problem_results
    )


# ============== Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "datasets": list(state.datasets.keys()),
        "runpod_connected": state.runpod_client is not None
    }


@app.get("/stats")
async def get_runpod_stats():
    """Get RunPod client statistics for monitoring throughput."""
    if state.runpod_client:
        stats = state.runpod_client.get_stats()
        return {
            "active_jobs": stats["active_jobs"],
            "total_submitted": stats["total_submitted"],
            "total_completed": stats["total_completed"],
            "total_failed": stats["total_failed"],
            "peak_active": stats["peak_active"],
        }
    return {"error": "RunPod client not initialized"}


@app.get("/datasets")
async def list_datasets():
    """List available datasets and their sizes."""
    return {
        name: {"size": len(ds), "type": type(ds).__name__}
        for name, ds in state.datasets.items()
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_workflow(request: EvaluateRequest):
    """
    Evaluate a single BlockWorkflow on a dataset.
    
    Fire-all-at-once: All problems are evaluated in parallel.
    """
    task_name = request.workflow.task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    request_id = str(uuid.uuid4())
    request_ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return await evaluate_workflow_parallel(request.workflow, problems, dataset, request_id=request_id, request_ts=request_ts)


@app.post("/evaluate/simple", response_model=EvaluateResponse)
async def evaluate_simple(request: SimpleEvaluateRequest):
    """
    Evaluate a workflow with simple role list (converts to AgentBlocks).
    
    Example body:
    {
        "roles": ["Task Parsing Agent", "Code Generation Agent"],
        "task_name": "MBPP",
        "num_problems": 10
    }
    """
    task_name = request.task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
    # Convert roles to blocks
    blocks = [BlockConfig(type="agent", role=role) for role in request.roles]
    
    workflow = WorkflowConfig(
        blocks=blocks,
        task_name=request.task_name,
        use_extractor=request.use_extractor,
        think=request.think
    )
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    request_id = str(uuid.uuid4())
    request_ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return await evaluate_workflow_parallel(workflow, problems, dataset, request_id=request_id, request_ts=request_ts)


@app.post("/evaluate/batch")
async def evaluate_batch(request: BatchEvaluateRequest):
    """
    Evaluate multiple BlockWorkflows on the same problems.
    
    Useful for comparing different workflow configurations.
    """
    if not request.workflows:
        raise HTTPException(400, "No workflows provided")
    
    task_name = request.workflows[0].task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    batch_id = str(uuid.uuid4())
    request_ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
    # Evaluate all workflows in parallel
    tasks = [
        evaluate_workflow_parallel(wf, problems, dataset, request_id=f"{batch_id}-{idx}", request_ts=request_ts)
        for idx, wf in enumerate(request.workflows)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format results
    response = []
    for wf, result in zip(request.workflows, results):
        if isinstance(result, Exception):
            response.append({
                "workflow": wf.model_dump(),
                "error": str(result)
            })
        else:
            response.append({
                "workflow": wf.model_dump(),
                "result": result.model_dump()
            })
    
    return response


@app.post("/evaluate/quick")
async def quick_evaluate(
    roles: str,
    task: str = "MBPP",
    num_problems: int = 5
):
    """
    Quick evaluation endpoint with simple parameters.
    
    Example: /evaluate/quick?roles=Code Generation Agent&task=MBPP&num_problems=5
    """
    # Convert comma-separated roles to blocks
    role_list = [r.strip() for r in roles.split(",")]
    blocks = [BlockConfig(type="agent", role=role) for role in role_list]
    
    workflow = WorkflowConfig(
        blocks=blocks,
        task_name=task,
        use_extractor=True
    )
    
    request = EvaluateRequest(
        workflow=workflow,
        num_problems=num_problems
    )
    
    return await evaluate_workflow(request)


# ============== BlockWorkflow Recreation ==============

def create_block_workflow_from_config(config: WorkflowConfig):
    """
    Recreate a BlockWorkflow object from a WorkflowConfig.
    
    This is useful for running workflows locally after receiving config from server.
    """
    from src.agents.block import AgentBlock, CompositeBlock
    from src.agents.workflow_block import BlockWorkflow
    
    blocks = []
    for block_config in config.blocks:
        if block_config.type == "agent":
            blocks.append(AgentBlock(role=block_config.role))
        elif block_config.type == "composite":
            blocks.append(CompositeBlock(
                divider_role=block_config.divider_role,
                synth_role=block_config.synth_role
            ))
    
    return BlockWorkflow(task_name=config.task_name, blocks=blocks)


def workflow_to_config(workflow) -> WorkflowConfig:
    """
    Convert a BlockWorkflow to WorkflowConfig for API calls.
    """
    from src.agents.block import AgentBlock, CompositeBlock
    
    blocks = []
    for block in workflow.blocks:
        if isinstance(block, AgentBlock):
            blocks.append(BlockConfig(type="agent", role=block.role))
        elif isinstance(block, CompositeBlock):
            blocks.append(BlockConfig(
                type="composite",
                divider_role=block.divider_role,
                synth_role=block.synth_role
            ))
    
    return WorkflowConfig(
        blocks=blocks,
        task_name=workflow.task_name,
        use_extractor=True
    )


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
