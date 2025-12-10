# src/server/workflow.py
"""
Workflow execution logic for the evaluation server.

Handles concurrent execution of agent workflows across problems.
Each problem's agents run sequentially, but multiple problems run in parallel.
"""

import asyncio
import time
import os
from typing import List, Optional, Tuple, Any

from langchain_core.messages import SystemMessage, HumanMessage

from .models import BlockConfig, WorkflowConfig, ProblemResult, EvaluateResponse
from .state import state
from .prompts import get_agent_system_prompt, build_agent_input, get_extractor_system_prompt, get_extractor_user_prompt
from .logging import log_agent_output, finalize_problem_log, save_request_logs
from ..schemas import get_schema_for_task

# Max concurrent problems to process (configurable via env)
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))


def expand_blocks_to_roles(blocks: List[BlockConfig]) -> List[str]:
    """Expand block configuration to a list of agent roles."""
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


async def invoke_single(structured_llm, messages) -> Any:
    """Invoke LLM for a single message (runs in thread pool)."""
    return await asyncio.to_thread(structured_llm.invoke, messages)


async def process_single_problem(
    problem: Any,
    problem_idx: int,
    n_problems: int,
    roles: List[str],
    task_name: str,
    use_extractor: bool,
    structured_llm: Any,
    extractor_llm: Any,
    extractor_system_prompt: str,
    request_id: str,
    request_ts: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, int, int, float, Optional[str]]:
    """
    Process a single problem through all agent stages.
    
    Uses semaphore to limit concurrency.
    """
    async with semaphore:  # Limit concurrent requests
        problem_start = time.time()
        original_problem = problem.prompt
        current_response = None
        problem_tokens = 0
        problem_completion_tokens = 0
        error = None
        
        print(f"  [START] Problem {problem_idx + 1}/{n_problems} (id={problem.id})")
        
        # Run through each agent stage sequentially for this problem
        for stage, role in enumerate(roles):
            if error:
                break
                
            is_first = (stage == 0)
            num_agents = len(roles)
            position_info = f"Agent {stage + 1} of {num_agents}" if num_agents > 1 else ""
            
            # System prompt: ROLE ONLY
            system_prompt = get_agent_system_prompt(role)
            prev_role = roles[stage - 1] if stage > 0 else "system"
            
            # User message: task, format, problem
            user_input = build_agent_input(
                problem_text=original_problem,
                prev_response=current_response,
                prev_role=prev_role,
                current_role=role,
                is_first=is_first,
                task_name=task_name,
                position_info=position_info
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            try:
                call_start = time.time()
                result = await invoke_single(structured_llm, messages)
                call_time = time.time() - call_start
                
                if result is None:
                    error = f"No response from model at stage {stage} ({role})"
                    await log_agent_output(request_id, {
                        "problem_id": problem.id,
                        "task": task_name,
                        "stage": stage,
                        "role": role,
                        "is_extractor": False,
                        "system_prompt": system_prompt,
                        "user_input": user_input,
                        "status": "FAILED",
                        "error": error,
                        "output": {},
                        "tokens": 0,
                        "elapsed": call_time,
                        "request_timestamp": request_ts,
                    })
                else:
                    current_response = result
                    est_tokens = len(str(result.model_dump())) // 4
                    problem_tokens += est_tokens
                    problem_completion_tokens += est_tokens
                    
                    await log_agent_output(request_id, {
                        "problem_id": problem.id,
                        "task": task_name,
                        "stage": stage,
                        "role": role,
                        "is_extractor": False,
                        "system_prompt": system_prompt,
                        "user_input": user_input,
                        "status": "COMPLETED",
                        "error": None,
                        "output": result.model_dump(),
                        "tokens": est_tokens,
                        "elapsed": call_time,
                        "request_timestamp": request_ts,
                    })
                    
            except Exception as e:
                error = f"Failed at stage {stage} ({role}): {str(e)}"
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": stage,
                    "role": role,
                    "is_extractor": False,
                    "system_prompt": system_prompt,
                    "user_input": user_input,
                    "status": "FAILED",
                    "error": error,
                    "output": {},
                    "tokens": 0,
                    "elapsed": time.time() - call_start,
                    "request_timestamp": request_ts,
                })
        
        # Run extractor if enabled and no error
        if use_extractor and error is None and current_response is not None:
            answer_text = current_response.answer
            # System: role only, User: format + content
            extractor_user_prompt = get_extractor_user_prompt(task_name, answer_text)
            messages = [
                SystemMessage(content=extractor_system_prompt),
                HumanMessage(content=extractor_user_prompt)
            ]
            
            try:
                call_start = time.time()
                result = await invoke_single(extractor_llm, messages)
                call_time = time.time() - call_start
                
                if result:
                    current_response.answer = result.answer
                    est_tokens = len(result.answer) // 4
                    problem_tokens += est_tokens
                    problem_completion_tokens += est_tokens
                
                await log_agent_output(request_id, {
                    "problem_id": problem.id,
                    "task": task_name,
                    "stage": len(roles),
                    "role": "extractor",
                    "is_extractor": True,
                    "system_prompt": extractor_system_prompt,
                    "user_input": extractor_user_prompt,
                    "status": "COMPLETED" if result else "FAILED",
                    "error": None if result else "Extraction failed",
                    "output": {"answer": result.answer} if result else {},
                    "tokens": est_tokens if result else 0,
                    "elapsed": call_time,
                    "request_timestamp": request_ts,
                })
                
            except Exception as e:
                error = f"Extractor failed: {str(e)}"
        
        # Build result for this problem
        problem_time = time.time() - problem_start
        
        print(f"  [DONE] Problem {problem_idx + 1}/{n_problems} (id={problem.id}) in {problem_time:.1f}s")
        
        if error:
            return ("", problem_tokens, problem_completion_tokens, problem_time, error)
        elif current_response:
            return (
                current_response.answer,
                problem_tokens,
                problem_completion_tokens,
                problem_time,
                None
            )
        else:
            return (original_problem, problem_tokens, problem_completion_tokens, problem_time, None)


async def run_workflow_concurrent(
    problems: List[Any],  # List[Problem]
    blocks: List[BlockConfig],
    task_name: str,
    use_extractor: bool,
    request_id: str = "",
    request_ts: str = "",
    think_mode: bool = False,
    max_concurrency: Optional[int] = None,
) -> List[Tuple[str, int, int, float, Optional[str]]]:
    """
    Run a workflow on ALL problems CONCURRENTLY.
    
    Each problem runs through all agents sequentially, but multiple problems
    are processed in parallel (up to max_concurrency limit).
    
    Args:
        problems: List of problems to process
        blocks: Workflow block configuration
        task_name: Task/benchmark name
        use_extractor: Whether to run answer extractor
        request_id: Request ID for logging
        request_ts: Request timestamp for logging
        think_mode: Enable extended thinking mode
        max_concurrency: Max concurrent problems (default: MAX_CONCURRENCY env or 10)
    
    Returns:
        List of (final_output, total_tokens, completion_tokens, execution_time, error) tuples
    """
    roles = expand_blocks_to_roles(blocks)
    n_problems = len(problems)
    task_upper = task_name.upper()
    
    # Get the structured LLM for this task
    structured_llm = state.structured_llms.get(task_upper)
    if structured_llm is None:
        schema = get_schema_for_task(task_name)
        structured_llm = state.llm.with_structured_output(schema)
    
    extractor_llm = state.structured_llms.get("EXTRACTOR")
    extractor_system_prompt = get_extractor_system_prompt()
    
    # Create semaphore to limit concurrency
    concurrency = max_concurrency or MAX_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    
    print(f"  Running {n_problems} problems with max {concurrency} concurrent...")
    
    # Create tasks for all problems
    tasks = [
        process_single_problem(
            problem=problem,
            problem_idx=idx,
            n_problems=n_problems,
            roles=roles,
            task_name=task_name,
            use_extractor=use_extractor,
            structured_llm=structured_llm,
            extractor_llm=extractor_llm,
            extractor_system_prompt=extractor_system_prompt,
            request_id=request_id,
            request_ts=request_ts,
            semaphore=semaphore,
        )
        for idx, problem in enumerate(problems)
    ]
    
    # Run all tasks concurrently (respecting semaphore limit)
    results = await asyncio.gather(*tasks)
    
    return list(results)


# Aliases for compatibility
run_workflow_sequential = run_workflow_concurrent
run_workflow_batched = run_workflow_concurrent


async def evaluate_workflow_batched(
    workflow: WorkflowConfig,
    problems: List[Any],  # List[Problem]
    dataset: Any,
    request_id: str,
    request_ts: str,
) -> EvaluateResponse:
    """
    Evaluate a BlockWorkflow using concurrent requests with LangChain.
    
    Multiple problems are processed concurrently (up to MAX_CONCURRENCY).
    Each problem's agents still run sequentially.
    
    Args:
        workflow: Workflow configuration
        problems: List of problems to evaluate
        dataset: Dataset instance for evaluation
        request_id: Request ID for logging
        request_ts: Request timestamp for logging
    
    Returns:
        EvaluateResponse with results and metrics
    """
    start_time = time.time()
    
    print(f"Starting evaluation: {len(problems)} problems, {len(workflow.blocks)} blocks (concurrent mode)")
    
    # Run workflow on all problems concurrently
    results = await run_workflow_concurrent(
        problems,
        workflow.blocks,
        workflow.task_name,
        workflow.use_extractor,
        request_id=request_id,
        request_ts=request_ts,
        think_mode=workflow.think,
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
            
            expected = (
                problem.ground_truth.get("output") or 
                problem.ground_truth.get("answer") or 
                problem.ground_truth.get("code") or 
                str(problem.ground_truth)
            )
            await finalize_problem_log(request_id, problem.id, expected, is_correct)
    
    # Save logs to file
    await save_request_logs(request_id, request_ts)
    
    total_time = time.time() - start_time
    
    print(f"Evaluation complete: {num_correct}/{len(problems)} correct ({num_correct/len(problems)*100:.1f}%)")
    
    return EvaluateResponse(
        pass_at_1=num_correct / len(problems) if problems else 0,
        num_correct=num_correct,
        num_problems=len(problems),
        total_tokens=total_tokens,
        completion_tokens=total_completion_tokens,
        total_time=total_time,
        tokens_per_second=total_completion_tokens / total_time if total_time > 0 else 0,
        problems=problem_results
    )
