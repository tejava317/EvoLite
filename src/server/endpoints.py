# src/server/endpoints.py
"""
FastAPI endpoints for the evaluation server.
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException

from .models import (
    BlockConfig,
    WorkflowConfig,
    EvaluateRequest,
    SimpleEvaluateRequest,
    BatchEvaluateRequest,
    EvaluateResponse,
)
from .state import state
from .workflow import evaluate_workflow_batched


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_mode": state.llm_mode,
        "llm_model": state.llm_model,
        "datasets": list(state.datasets.keys()),
        "llm_initialized": state.llm is not None,
        "structured_outputs": list(state.structured_llms.keys()) if state.structured_llms else []
    }


@router.get("/mode")
async def get_mode():
    """Get current LLM mode and model."""
    return {
        "mode": state.llm_mode,
        "model": state.llm_model,
        "available_modes": ["vllm", "openai"],
        "env_vars": {
            "vllm": ["VLLM_BASE_URL", "VLLM_MODEL"],
            "openai": ["OPENAI_API_KEY", "OPENAI_MODEL"]
        }
    }


@router.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "llm_mode": state.llm_mode,
        "llm_model": state.llm_model,
        "datasets": list(state.datasets.keys()),
        "structured_outputs": list(state.structured_llms.keys()) if state.structured_llms else [],
    }


@router.get("/datasets")
async def list_datasets():
    """List available datasets and their sizes."""
    return {
        name: {"size": len(ds), "type": type(ds).__name__}
        for name, ds in state.datasets.items()
    }


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_workflow(request: EvaluateRequest):
    """Evaluate a single BlockWorkflow on a dataset."""
    task_name = request.workflow.task_name.upper()
    
    if task_name not in state.datasets:
        raise HTTPException(404, f"Dataset {task_name} not loaded")
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    request_id = str(uuid.uuid4())
    request_ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
    return await evaluate_workflow_batched(
        workflow=request.workflow,
        problems=problems,
        dataset=dataset,
        request_id=request_id,
        request_ts=request_ts
    )


@router.post("/evaluate/simple", response_model=EvaluateResponse)
async def evaluate_simple(request: SimpleEvaluateRequest):
    """Evaluate a workflow with simple role list (converts to AgentBlocks)."""
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
    
    return await evaluate_workflow_batched(
        workflow=workflow,
        problems=problems,
        dataset=dataset,
        request_id=request_id,
        request_ts=request_ts
    )


@router.post("/evaluate/batch")
async def evaluate_batch(request: BatchEvaluateRequest):
    """Evaluate multiple BlockWorkflows on the same problems."""
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


@router.post("/evaluate/quick")
async def quick_evaluate(
    roles: str,
    task: str = "MBPP",
    num_problems: int = 5
):
    """Quick evaluation endpoint with simple parameters."""
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
