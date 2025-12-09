# src/evaluation_server.py
"""
FastAPI Evaluation Server for EvoLite.

High-throughput evaluation using vLLM batch completions API.
Processes all problems per agent step in a single batched request.

Works with the new BlockWorkflow system.

Run with:
    uvicorn src.evaluation_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import time
import os
import json
import uuid
from typing import Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import re
from src.llm.vllm_client import VLLMClient, JobResult
from src.datasets import MBPPDataset, MathAlgebraDataset, CRUXOpenDataset
from src.datasets.base import Problem
from src.evaluation.executor import execute_code


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
    completion_tokens: int  # Generated tokens only (excludes prompts)
    total_time: float
    tokens_per_second: float
    problems: List[ProblemResult]


# ============== Global State ==============

@dataclass
class ServerState:
    """Cached server state."""
    datasets: dict = None
    vllm_client: VLLMClient = None
    extractor_prompts: dict = None


state = ServerState()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup, cleanup at shutdown."""
    print("🚀 Starting Evaluation Server...")
    
    # Initialize vLLM client
    base_url = os.getenv("VLLM_BASE_URL", "http://38.128.232.68:27717/v1")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B")
    
    state.vllm_client = VLLMClient(
        base_url=base_url,
        model=model,
        default_temperature=0.6,
        default_max_tokens=2048,  # Room for prompts within 6000 context
        timeout=600.0,  # 10 minutes for large batches
    )
    print(f"✓ vLLM client initialized")
    print(f"  URL: {base_url}")
    print(f"  Model: {model}")
    
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
        "MBPP": """Extract the Python function code.

INPUT: def add(a, b): return a + b
OUTPUT: def add(a, b): return a + b

Extract function (no markdown):""",
        "MATH": """Extract the \\boxed{} answer.

INPUT: \\boxed{42}
OUTPUT: \\boxed{42}

INPUT: \\boxed{\\frac{3}{4}}
OUTPUT: \\boxed{\\frac{3}{4}}

Extract boxed answer:""",
        "CRUX-O": """Extract the output value.

INPUT: Result is [(4, 1)]
OUTPUT: [(4, 1)]

Extract Python literal:"""
    }
    
    print("✓ Server ready!")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")
    if state.vllm_client:
        await state.vllm_client.close()


# ============== FastAPI App ==============

app = FastAPI(
    title="EvoLite Evaluation Server",
    description="High-throughput BlockWorkflow evaluation using vLLM batch completions",
    version="3.0.0",
    lifespan=lifespan
)


# Silence access logs for noisy endpoints (e.g., /stats polling)
class _EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/stats " not in msg


logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())


# ============== Prompt System (imports from generate_prompts.py) ==============

from src.utils.generate_prompts import (
    STRUCTURED_OUTPUT_INSTRUCTIONS,
    TASK_DESCRIPTIONS,
    EXTRACTOR_PROMPTS,
    AGENT_ROLES,
    get_task_description,
    get_extractor_prompt as _get_extractor_prompt,
    get_structured_output_instructions,
    build_first_agent_prompt,
    build_agent_prompt,
    load_prompts_from_yaml,
)

# Build role description lookup
ROLE_DESCRIPTIONS = {name.lower(): desc for name, desc in AGENT_ROLES}

# Load GPT-generated prompts from YAML
GENERATED_PROMPTS = {}

try:
    _loaded = load_prompts_from_yaml()
    if _loaded:
        GENERATED_PROMPTS = {k.lower(): v for k, v in _loaded.get("agents", {}).items()}
        print(f"  ✓ Loaded {len(GENERATED_PROMPTS)} GPT-generated agent prompts")
except Exception as e:
    print(f"  ✗ Could not load generated prompts: {e}")


def get_agent_prompt(role: str, task_name: str = None, is_first_agent: bool = False) -> str:
    """
    Get prompt for an agent role with STRUCTURED OUTPUT format.
    """
    role_lower = role.lower()
    base_prompt = GENERATED_PROMPTS.get(role_lower)
    
    # Try partial match if exact match not found
    if base_prompt is None:
        for key, prompt in GENERATED_PROMPTS.items():
            if role_lower in key or key in role_lower:
                base_prompt = prompt
                break
    
    if base_prompt is None:
        raise ValueError(f"Unknown agent role: {role}. Regenerate prompts with generate_prompts.py")
    
    if is_first_agent and task_name:
        return build_first_agent_prompt(base_prompt, task_name)
    return build_agent_prompt(base_prompt, task_name or "MBPP")


def get_extractor_prompt(task_name: str) -> str:
    """Get extractor prompt for task/benchmark."""
    prompt = _get_extractor_prompt(task_name)
    if prompt:
        return prompt
    
    if task_name in state.extractor_prompts:
        return state.extractor_prompts[task_name]
    
    for key in state.extractor_prompts:
        if key.upper() == task_name.upper():
            return state.extractor_prompts[key]
    
    return state.extractor_prompts.get("MBPP", "")


def expand_blocks_to_roles(blocks: List[BlockConfig]) -> List[str]:
    """
    Expand block configuration to a list of agent roles.
    """
    roles = []
    
    for block in blocks:
        if block.type == "agent":
            if block.role:
                roles.append(block.role)
        elif block.type == "composite":
            roles.append(block.divider_role or "Divider")
            roles.extend(["Business Analyst", "Technical Lead", "Quality Assurance"])
            roles.append(block.synth_role or "Synthesizer")
    
    return roles


# File locks for concurrent log writes
_log_locks: dict[str, asyncio.Lock] = {}
_log_locks_lock = asyncio.Lock()

# In-memory storage for problem logs
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
        
        if request_id not in _problem_logs:
            _problem_logs[request_id] = {}
        
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
        
        # Always include parsed data if any field was extracted
        parsed_problem = record.get("parsed_problem", "")
        parsed_work = record.get("parsed_work", "")
        parsed_comment = record.get("parsed_comment", "")
        parsed_answer = record.get("parsed_answer", "")
        
        if parsed_problem or parsed_work or parsed_comment or parsed_answer:
            step["parsed"] = {
                "problem": parsed_problem,
                "work": parsed_work,
                "comment": parsed_comment,
                "answer": parsed_answer,
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
    """Save all problem logs for a request to a single JSON file."""
    lock = await _get_log_lock(request_id)
    
    async with lock:
        if request_id not in _problem_logs:
            return
        
        def _write():
            dir_path = os.path.join("logs", "agent_runs", request_ts)
            path = os.path.join(dir_path, f"{request_id}.json")
            os.makedirs(dir_path, exist_ok=True)
            
            problems = list(_problem_logs[request_id].values())
            problems.sort(key=lambda x: x["problem_id"])
            
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
        
        del _problem_logs[request_id]


def get_think_suffix(think: bool) -> str:
    """Get the thinking mode suffix for prompts."""
    return " /think" if think else " /no_think"


# ============== Structured Output Parsing ==============

def parse_structured_output(output: str) -> dict:
    """
    Parse the structured YAML output from an agent.
    
    Strips <think>...</think> blocks before parsing to avoid matching
    content from the model's internal reasoning.
    
    Handles both YAML format and legacy bracket format for backwards compatibility.
    """
    import yaml as yaml_lib
    
    result = {
        "problem": "",
        "work": "",
        "comment": "",
        "answer": "",
        "raw": output,
        "parsed": False,
    }
    
    # Remove <think>...</think> blocks before parsing
    clean_output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    
    # Method 1: Try to extract content from ```yaml ... ``` block
    # Use GREEDY match to get ALL content up to the LAST closing ```
    # This handles cases where model incorrectly nests ```python inside yaml
    yaml_match = re.search(r'```yaml\s*(.*?)```(?![\w])', clean_output, re.DOTALL)
    yaml_content = yaml_match.group(1) if yaml_match else None
    
    # If we got truncated content (answer is empty but there's more ```), try greedy
    if yaml_content and 'answer: |' in yaml_content:
        # Check if answer appears empty (just whitespace after |)
        answer_check = re.search(r'answer:\s*\|\s*\n(\s*)$', yaml_content)
        if answer_check:
            # Try greedy match - get content up to the LAST ``` in the output
            greedy_match = re.search(r'```yaml\s*(.*)\n```', clean_output, re.DOTALL)
            if greedy_match:
                yaml_content = greedy_match.group(1)
    
    # Method 1b: Strip nested code fences from yaml content before parsing
    if yaml_content:
        # Remove ```python, ```javascript, etc. fences inside the yaml
        yaml_content = re.sub(r'```\w*\n?', '', yaml_content)
    
    # Method 2: Look for YAML keys directly in the text (key: | or key: value)
    # This handles cases where model outputs YAML without code fences
    if not yaml_content:
        # Check if text looks like YAML (has our expected keys followed by : or :|)
        if re.search(r'^(problem|work|comment|answer)\s*:\s*[|\n]', clean_output, re.MULTILINE | re.IGNORECASE):
            yaml_content = clean_output
            # Also strip nested fences
            yaml_content = re.sub(r'```\w*\n?', '', yaml_content)
    
    # Try parsing as YAML
    if yaml_content:
        try:
            parsed_yaml = yaml_lib.safe_load(yaml_content)
            if isinstance(parsed_yaml, dict):
                for key in ["problem", "work", "comment", "answer"]:
                    if key in parsed_yaml and parsed_yaml[key]:
                        result[key] = str(parsed_yaml[key]).strip()
                        result["parsed"] = True
        except:
            pass
    
    # Method 3: Try regex extraction for YAML-style keys (more permissive)
    if not result["parsed"] or not result.get("answer"):
        # Match "key: |" or "key:" followed by content
        for key in ["problem", "work", "comment", "answer"]:
            # Skip if already parsed for this key (except answer which we always try to get)
            if key != "answer" and result.get(key):
                continue
                
            # Pattern 1: key: | followed by INDENTED lines (proper YAML)
            pattern = rf'{key}\s*:\s*\|\s*\n((?:[ \t]+.*\n?)*)'
            match = re.search(pattern, clean_output, re.IGNORECASE)
            if match and match.group(1).strip():
                # Remove leading indentation
                content = re.sub(r'^[ \t]+', '', match.group(1), flags=re.MULTILINE)
                result[key] = content.strip()
                result["parsed"] = True
            else:
                # Pattern 2: key: | followed by UNINDENTED content (model didn't indent properly)
                # This handles cases like "answer: |\ndef foo():" 
                pattern = rf'{key}\s*:\s*\|\s*\n((?:(?!(?:problem|work|comment|answer)\s*:).*\n?)*)'
                match = re.search(pattern, clean_output, re.IGNORECASE)
                if match and match.group(1).strip():
                    content = match.group(1).strip()
                    # Remove trailing ``` if present
                    content = re.sub(r'\n?```\s*$', '', content)
                    result[key] = content.strip()
                    result["parsed"] = True
                else:
                    # Pattern 3: Try single-line value: key: value (until next key or end)
                    pattern = rf'{key}\s*:\s*(.+?)(?=\n(?:problem|work|comment|answer)\s*:|$)'
                    match = re.search(pattern, clean_output, re.IGNORECASE | re.DOTALL)
                    if match:
                        result[key] = match.group(1).strip()
                        result["parsed"] = True
    
    # Method 4: Try legacy bracket format [PROBLEM], [WORK], etc.
    if not result["parsed"]:
        sections = {
            "problem": r'\[PROBLEM\]\s*(.*?)(?=\[WORK\]|\[COMMENT\]|\[ANSWER\]|\[PREVIOUS|NOTE:|$)',
            "work": r'\[WORK\]\s*(.*?)(?=\[COMMENT\]|\[ANSWER\]|\[PREVIOUS|NOTE:|$)',
            "comment": r'\[COMMENT\]\s*(.*?)(?=\[ANSWER\]|\[PREVIOUS|NOTE:|$)',
            "answer": r'\[ANSWER\]\s*(.*?)(?=\[PREVIOUS|NOTE:|$)',
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, clean_output, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
                result["parsed"] = True
    
    # If still no structured format found, use cleaned output as answer
    if not result["parsed"]:
        result["answer"] = clean_output.strip()
    
    return result


def build_passthrough_input(
    original_problem: str, 
    parsed_output: dict, 
    prev_role: str = "previous agent",
    current_role: str = "",
    current_role_desc: str = "",
) -> str:
    """
    Build the input for the next agent in YAML format.
    """
    problem = parsed_output.get("problem", "").strip() or original_problem
    comment = parsed_output.get("comment", "").strip()
    answer = parsed_output.get("answer", "").strip()
    
    # Indent multi-line content for YAML
    def indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.split("\n"))
    
    lines = []
    
    # Role reminder first
    if current_role:
        lines.append(f"# YOUR ROLE: {current_role}")
        if current_role_desc:
            lines.append(f"# {current_role_desc}")
        lines.append("")
    
    lines.extend([
        "# Input from previous agent",
        f"from: {prev_role}",
        "",
        "problem: |",
        indent(problem),
    ])
    
    if comment:
        lines.extend(["", "previous_comment: |", indent(comment)])
    
    if answer:
        lines.extend(["", "previous_answer: |", indent(answer)])
    else:
        lines.extend(["", "previous_answer: null  # No answer yet"])
    
    return "\n".join(lines)


def format_first_agent_input(
    problem_prompt: str,
    current_role: str = "",
    current_role_desc: str = "",
) -> str:
    """
    Format the problem prompt for the first agent in YAML format.
    """
    # Indent the problem for YAML
    indented_problem = "\n".join("  " + line for line in problem_prompt.split("\n"))
    
    lines = []
    
    # Role reminder first
    if current_role:
        lines.append(f"# YOUR ROLE: {current_role}")
        if current_role_desc:
            lines.append(f"# {current_role_desc}")
        lines.append("")
    
    lines.extend([
        "# You are the first agent in the workflow",
        "from: system",
        "",
        "problem: |",
        indented_problem,
        "",
        "previous_answer: null  # You are the first agent, no previous answer exists",
    ])
    
    return "\n".join(lines)


# ============== Batched Workflow Execution ==============

async def run_workflow_batched(
    problems: List[Problem],
    blocks: List[BlockConfig],
    task_name: str,
    use_extractor: bool,
    client: VLLMClient,
    request_id: str = "",
    request_ts: str = "",
    think: bool = False,
) -> List[tuple[str, int, int, float, Optional[str]]]:
    """
    Run a workflow on ALL problems using batched requests.
    
    For each agent step: ONE batch request containing all N problems.
    This is much more efficient than N individual requests.
    
    Returns: List of (final_output, total_tokens, completion_tokens, execution_time, error) tuples
    """
    start_time = time.time()
    roles = expand_blocks_to_roles(blocks)
    think_suffix = get_think_suffix(think)
    n_problems = len(problems)
    
    # Initialize per-problem state
    original_problems = [p.prompt for p in problems]
    
    # Build first agent input with role info
    first_role = roles[0] if roles else ""
    first_role_desc = ROLE_DESCRIPTIONS.get(first_role.lower(), "") if first_role else ""
    current_inputs = [
        format_first_agent_input(p.prompt, first_role, first_role_desc) 
        for p in problems
    ]
    
    total_tokens = [0] * n_problems
    completion_tokens = [0] * n_problems  # Track generated tokens separately
    errors: List[Optional[str]] = [None] * n_problems
    parsed_outputs = [None] * n_problems  # Track parsed outputs for extractor
    
    # Handle case with no roles
    if not roles:
        if use_extractor:
            # Extractor never uses thinking - it's just simple extraction
            extractor_prompt = get_extractor_prompt(task_name) + " /no_think"
            
            # Build batch of messages for extraction
            messages_list = [
                [
                    {"role": "system", "content": extractor_prompt},
                    {"role": "user", "content": original_problems[i]},
                ]
                for i in range(n_problems)
            ]
            
            results = await client.batch_complete(messages_list)
            
            outputs = []
            for i, (problem, result) in enumerate(zip(problems, results)):
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": 0,
                    "role": "extractor",
                    "is_extractor": True,
                    "system_prompt": extractor_prompt,
                    "user_input": original_problems[i],
                    "status": result.status,
                    "error": result.error,
                    "content": result.content,
                    "tokens": result.total_tokens,
                    "elapsed": result.execution_time,
                    "request_timestamp": request_ts,
                })
                
                if result.status != "COMPLETED":
                    outputs.append(("", 0, 0, time.time() - start_time, result.error or f"Failed: {result.status}"))
                else:
                    outputs.append((result.content, result.total_tokens, result.completion_tokens, time.time() - start_time, None))
            
            return outputs
        else:
            return [(original_problems[i], 0, 0, time.time() - start_time, None) for i in range(n_problems)]
    
    # Run each agent step as a BATCH
    for stage, role in enumerate(roles):
        is_first = (stage == 0)
        prompt = get_agent_prompt(role, task_name, is_first_agent=is_first) + think_suffix
        
        # Build batch of messages for this agent step
        messages_list = [
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": current_inputs[i]},
            ]
            for i in range(n_problems)
            if errors[i] is None  # Skip problems that already failed
        ]
        
        # Map indices: track which problems are in this batch
        active_indices = [i for i in range(n_problems) if errors[i] is None]
        
        if not messages_list:
            break  # All problems have errors
        
        # Single batch request for all active problems
        results = await client.batch_complete(messages_list)
        
        # Process results and update state
        for batch_idx, result in enumerate(results):
            i = active_indices[batch_idx]  # Original problem index
            problem = problems[i]
            
            if result.status != "COMPLETED":
                errors[i] = result.error or f"Job failed with status {result.status}"
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": stage,
                    "role": role,
                    "is_extractor": False,
                    "system_prompt": prompt,
                    "user_input": current_inputs[i],
                    "status": result.status,
                    "error": result.error,
                    "content": "",
                    "tokens": 0,
                    "elapsed": result.execution_time,
                    "request_timestamp": request_ts,
                })
                continue
            
            total_tokens[i] += result.total_tokens
            completion_tokens[i] += result.completion_tokens
            raw_output = result.content
            
            # Parse structured output
            parsed = parse_structured_output(raw_output)
            parsed_outputs[i] = parsed
            
            # Log the output
            await log_agent_output(request_id, {
                "problem_id": problem.id,
                "task": task_name,
                "stage": stage,
                "role": role,
                "is_extractor": False,
                "system_prompt": prompt,
                "user_input": current_inputs[i],
                "status": result.status,
                "error": None,
                "content": raw_output,
                "parsed_problem": parsed.get("problem", ""),
                "parsed_work": parsed.get("work", ""),
                "parsed_comment": parsed.get("comment", ""),
                "parsed_answer": parsed.get("answer", ""),
                "tokens": result.total_tokens,
                "elapsed": result.execution_time,
                "request_timestamp": request_ts,
            })
            
            # Build pass-through input for next agent
            next_stage = stage + 1
            next_role = roles[next_stage] if next_stage < len(roles) else ""
            next_role_desc = ROLE_DESCRIPTIONS.get(next_role.lower(), "") if next_role else ""
            current_inputs[i] = build_passthrough_input(
                original_problems[i], parsed, 
                prev_role=role,
                current_role=next_role,
                current_role_desc=next_role_desc,
            )
    
    # Run extractor if enabled (as a batch)
    # Extractor never uses thinking - it's just simple extraction
    if use_extractor:
        extractor_prompt = get_extractor_prompt(task_name) + " /no_think"
        
        # Build batch of messages for extraction (only for non-errored problems)
        active_indices = [i for i in range(n_problems) if errors[i] is None]
        
        if active_indices:
            messages_list = []
            for i in active_indices:
                extractor_input = ""
                if parsed_outputs[i]:
                    extractor_input = parsed_outputs[i].get("answer", "") or ""
                messages_list.append([
                    {"role": "system", "content": extractor_prompt},
                    {"role": "user", "content": extractor_input},
                ])
            
            results = await client.batch_complete(messages_list)
            
            for batch_idx, result in enumerate(results):
                i = active_indices[batch_idx]
                problem = problems[i]
                extractor_input = ""
                if parsed_outputs[i]:
                    extractor_input = parsed_outputs[i].get("answer", "") or ""
                
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": len(roles),
                    "role": "extractor",
                    "is_extractor": True,
                    "system_prompt": extractor_prompt,
                    "user_input": extractor_input,
                    "status": result.status,
                    "error": result.error,
                    "content": result.content,
                    "tokens": result.total_tokens,
                    "elapsed": result.execution_time,
                    "request_timestamp": request_ts,
                })
                
                if result.status != "COMPLETED":
                    errors[i] = result.error or f"Extractor failed: {result.status}"
                else:
                    total_tokens[i] += result.total_tokens
                    completion_tokens[i] += result.completion_tokens
                    current_inputs[i] = result.content
    
    # Build final results
    exec_time = time.time() - start_time
    return [
        (current_inputs[i], total_tokens[i], completion_tokens[i], exec_time, errors[i])
        for i in range(n_problems)
    ]


async def evaluate_workflow_batched(
    workflow: WorkflowConfig,
    problems: List[Problem],
    dataset,
    request_id: str,
    request_ts: str,
) -> EvaluateResponse:
    """
    Evaluate a BlockWorkflow using batched requests.
    
    Each agent step processes ALL problems in a single batch request.
    """
    start_time = time.time()
    client = state.vllm_client
    
    # Run workflow on all problems (batched)
    results = await run_workflow_batched(
        problems,
        workflow.blocks,
        workflow.task_name,
        workflow.use_extractor,
        client,
        request_id=request_id,
        request_ts=request_ts,
        think=workflow.think,
    )
    
    # Process results
    problem_results = []
    num_correct = 0
    total_tokens = 0
    total_completion_tokens = 0
    
    for problem, result in zip(problems, results):
        output, tokens, comp_tokens, exec_time, err = result
        total_tokens += tokens
        total_completion_tokens += comp_tokens
        
        if err:
            problem_results.append(ProblemResult(
                problem_id=problem.id,
                correct=False,
                tokens=tokens,
                time=exec_time,
                error=str(err)[:200]
            ))
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
            
            expected = problem.ground_truth.get("output") or problem.ground_truth.get("answer") or problem.ground_truth.get("code") or str(problem.ground_truth)
            await finalize_problem_log(request_id, problem.id, expected, is_correct)
    
    # Save logs
    await save_request_logs(request_id, request_ts)
    
    total_time = time.time() - start_time
    
    return EvaluateResponse(
        pass_at_1=num_correct / len(problems) if problems else 0,
        num_correct=num_correct,
        num_problems=len(problems),
        total_tokens=total_tokens,
        completion_tokens=total_completion_tokens,
        total_time=total_time,
        tokens_per_second=total_completion_tokens / total_time if total_time > 0 else 0,  # Based on generated tokens
        problems=problem_results
    )


# ============== Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "datasets": list(state.datasets.keys()),
        "vllm_connected": state.vllm_client is not None
    }


@app.get("/stats")
async def get_vllm_stats():
    """Get vLLM client statistics for monitoring throughput."""
    if state.vllm_client:
        return state.vllm_client.get_stats()
    return {"error": "vLLM client not initialized"}


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
    
    Uses batched requests: all problems processed together per agent step.
    """
    task_name = request.workflow.task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    request_id = str(uuid.uuid4())
    request_ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
    return await evaluate_workflow_batched(workflow=request.workflow, problems=problems, dataset=dataset, request_id=request_id, request_ts=request_ts)


@app.post("/evaluate/simple", response_model=EvaluateResponse)
async def evaluate_simple(request: SimpleEvaluateRequest):
    """
    Evaluate a workflow with simple role list (converts to AgentBlocks).
    """
    task_name = request.task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
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
    
    return await evaluate_workflow_batched(workflow=workflow, problems=problems, dataset=dataset, request_id=request_id, request_ts=request_ts)


@app.post("/evaluate/batch")
async def evaluate_batch(request: BatchEvaluateRequest):
    """
    Evaluate multiple BlockWorkflows on the same problems.
    
    Each workflow is evaluated concurrently (one batch request per agent step per workflow).
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
    
    # Evaluate all workflows concurrently
    tasks = [
        evaluate_workflow_batched(
            workflow=wf,
            problems=problems,
            dataset=dataset,
            request_id=f"{batch_id}-{idx}",
            request_ts=request_ts
        )
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
