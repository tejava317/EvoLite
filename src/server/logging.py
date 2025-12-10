# src/server/logging.py
"""
Logging utilities for the evaluation server.

Handles logging agent outputs, problem results, and saving logs to files.
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


# File locks for concurrent log writes
_log_locks: Dict[str, asyncio.Lock] = {}
_log_locks_lock = asyncio.Lock()

# In-memory storage for problem logs
_problem_logs: Dict[str, Dict[str, Dict]] = {}


async def _get_log_lock(request_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific request_id."""
    async with _log_locks_lock:
        if request_id not in _log_locks:
            _log_locks[request_id] = asyncio.Lock()
        return _log_locks[request_id]


async def log_agent_output(request_id: str, record: Dict[str, Any]):
    """
    Collect agent output in memory (grouped by problem).
    
    Args:
        request_id: Unique request identifier
        record: Dictionary containing:
            - problem_id: Problem identifier
            - task: Task name
            - stage: Agent stage number
            - role: Agent role name
            - is_extractor: Whether this is an extractor step
            - system_prompt: System prompt used
            - user_input: User input message
            - status: COMPLETED or FAILED
            - error: Error message if failed
            - output: Structured output dict
            - tokens: Token count
            - elapsed: Time elapsed
            - request_timestamp: Request timestamp
    """
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
        
        # Build step record with full details
        step = {
            "stage": record.get("stage", 0),
            "role": record.get("role", "unknown"),
            "is_extractor": record.get("is_extractor", False),
            "timestamp": datetime.utcnow().isoformat(),
            "system_prompt": record.get("system_prompt", ""),
            "input": record.get("user_input", ""),
            "output": record.get("output", {}),
            "tokens": record.get("tokens", 0),
            "elapsed": record.get("elapsed", 0),
            "status": record.get("status", ""),
            "error": record.get("error"),
        }
        
        if record.get("is_extractor"):
            prob_log["extractor"] = step
        else:
            prob_log["agents"].append(step)


async def finalize_problem_log(
    request_id: str,
    problem_id: str,
    expected_answer: str,
    correct: bool
):
    """Finalize a problem's log with the expected answer and correctness."""
    lock = await _get_log_lock(request_id)
    
    async with lock:
        if request_id in _problem_logs and problem_id in _problem_logs[request_id]:
            _problem_logs[request_id][problem_id]["expected_answer"] = expected_answer
            _problem_logs[request_id][problem_id]["correct"] = correct


async def save_request_logs(request_id: str, request_ts: str):
    """
    Save all problem logs for a request to a JSON file.
    
    Logs are saved to: logs/agent_runs/{timestamp}/{request_id}.json
    """
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
            
            # Calculate summary stats
            total_problems = len(problems)
            correct_count = sum(1 for p in problems if p.get("correct"))
            total_agents = sum(len(p.get("agents", [])) for p in problems)
            
            output = {
                "request_id": request_id,
                "timestamp": request_ts,
                "summary": {
                    "total_problems": total_problems,
                    "correct_count": correct_count,
                    "accuracy": correct_count / total_problems if total_problems > 0 else 0,
                    "total_agent_steps": total_agents,
                },
                "problems": problems,
            }
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"  📝 Saved logs to {path}")
        
        await asyncio.to_thread(_write)
        
        # Cleanup memory
        del _problem_logs[request_id]
        
        # Also cleanup lock
        async with _log_locks_lock:
            if request_id in _log_locks:
                del _log_locks[request_id]


def get_problem_logs(request_id: str) -> Optional[Dict[str, Dict]]:
    """Get in-memory problem logs for a request (for debugging)."""
    return _problem_logs.get(request_id)
