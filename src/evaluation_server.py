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
from typing import Optional, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm.runpod_client import RunPodAsyncClient, JobResult
from src.datasets import MBPPDataset, MathAlgebraDataset
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


class SimpleWorkflowConfig(BaseModel):
    """Simple config with just role names (converts to AgentBlocks)."""
    roles: List[str] = Field(..., description="List of agent role names")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")


class EvaluateRequest(BaseModel):
    """Request to evaluate a single workflow."""
    workflow: WorkflowConfig
    num_problems: int = Field(default=10, ge=1, le=500)
    seed: Optional[int] = Field(default=None, description="Random seed for problem sampling")


class SimpleEvaluateRequest(BaseModel):
    """Simplified request using just role names."""
    roles: List[str] = Field(..., description="List of agent role names")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")
    num_problems: int = Field(default=10, ge=1, le=500)
    seed: Optional[int] = Field(default=None, description="Random seed for problem sampling")


class BatchEvaluateRequest(BaseModel):
    """Request to evaluate multiple workflows."""
    workflows: List[WorkflowConfig]
    num_problems: int = Field(default=10, ge=1, le=500)
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
    
    # Initialize RunPod client
    state.runpod_client = RunPodAsyncClient(
        default_temperature=0.1,
        default_max_tokens=2000,
        poll_interval=0.3,
        max_poll_time=120,
    )
    print("✓ RunPod client initialized")
    
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
    
    # Cache extractor prompts
    state.extractor_prompts = {
        "code": """You are a code extraction specialist. Extract ONLY the Python code from the given response.
Return only executable Python code wrapped in a markdown code block:
```python
[extracted code here]
```
Do NOT include explanations, comments about the code, or test cases.""",
        "math": """You are a mathematical answer extraction specialist. Extract the FINAL answer from the given solution.
Return ONLY the final answer in LaTeX boxed notation:
\\boxed{[final answer]}
Simplify the answer if possible."""
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


# ============== Helper Functions ==============

def get_agent_prompt(role: str, task_name: str = None) -> str:
    """
    Get prompt for an agent role.
    
    Uses task-specific prompts from initial_prompts.yaml when available.
    """
    # Try task-specific prompt first
    prompt = get_predefined_prompt(role, task_name)
    if prompt:
        return prompt
    
    # Fallback to generic prompt
    return f"You are a {role}. Complete the task given to you."


def get_extractor_prompt(task_name: str) -> str:
    """Get extractor prompt for task type."""
    if "math" in task_name.lower():
        return state.extractor_prompts["math"]
    return state.extractor_prompts["code"]


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


async def run_workflow_on_problem(
    problem: Problem,
    blocks: List[BlockConfig],
    task_name: str,
    use_extractor: bool,
    client: RunPodAsyncClient
) -> tuple[str, int, float]:
    """
    Run a BlockWorkflow on a single problem.
    
    Returns: (final_output, total_tokens, execution_time)
    """
    start_time = time.time()
    current_input = problem.prompt
    total_tokens = 0
    
    # Process each block
    for block in blocks:
        if block.type == "agent":
            # Simple agent block - single LLM call
            # Use task-specific prompt from initial_prompts.yaml if available
            prompt = get_agent_prompt(block.role, task_name)
            
            result = await client.generate(
                system_prompt=prompt,
                user_content=current_input
            )
            
            if result.status != "COMPLETED":
                raise Exception(f"Agent {block.role} failed: {result.error}")
            
            current_input = result.content
            total_tokens += result.total_tokens
            
        elif block.type == "composite":
            # Composite block - divider -> inner agents -> synthesizer
            # First, expand to get inner roles
            inner_roles = await expand_composite_block_async(block, client, current_input)
            
            # Run divider with task-specific prompt if available
            divider_prompt = get_agent_prompt(block.divider_role or "Divider", task_name)
            if not divider_prompt or "You are a" in divider_prompt[:20]:
                # Fallback to generic divider prompt
                divider_prompt = f"""You are a {block.divider_role or 'Divider'}. 
Divide the following task into subtasks for these roles: {', '.join(inner_roles)}

Task: {current_input}

Provide clear subtask assignments for each role."""
            
            result = await client.generate(
                system_prompt=divider_prompt,
                user_content=current_input
            )
            
            if result.status != "COMPLETED":
                raise Exception(f"Divider failed: {result.error}")
            
            divided_output = result.content
            total_tokens += result.total_tokens
            
            # Run inner agents in parallel with task-specific prompts
            inner_tasks = []
            for role in inner_roles:
                inner_prompt = get_agent_prompt(role, task_name)
                inner_tasks.append(
                    client.generate(
                        system_prompt=inner_prompt,
                        user_content=f"Your assigned subtask from the divider:\n{divided_output}\n\nOriginal task:\n{current_input}"
                    )
                )
            
            inner_results = await asyncio.gather(*inner_tasks, return_exceptions=True)
            
            # Collect inner outputs
            inner_outputs = []
            for i, res in enumerate(inner_results):
                if isinstance(res, Exception):
                    inner_outputs.append(f"{inner_roles[i]}: [Error: {res}]")
                else:
                    inner_outputs.append(f"{inner_roles[i]}:\n{res.content}")
                    total_tokens += res.total_tokens
            
            # Run synthesizer with task-specific prompt if available
            synth_prompt = get_agent_prompt(block.synth_role or "Synthesizer", task_name)
            synth_input = f"""Combine the following outputs from multiple specialists into a coherent final result:

{chr(10).join(inner_outputs)}

Original task: {current_input}

Provide the synthesized final output."""
            
            result = await client.generate(
                system_prompt=synth_prompt,
                user_content=synth_input
            )
            
            if result.status != "COMPLETED":
                raise Exception(f"Synthesizer failed: {result.error}")
            
            current_input = result.content
            total_tokens += result.total_tokens
    
    # Run extractor if enabled
    if use_extractor:
        extractor_prompt = get_extractor_prompt(task_name)
        result = await client.generate(
            system_prompt=extractor_prompt,
            user_content=current_input
        )
        
        if result.status == "COMPLETED":
            current_input = result.content
            total_tokens += result.total_tokens
    
    return current_input, total_tokens, time.time() - start_time


async def evaluate_workflow_parallel(
    workflow: WorkflowConfig,
    problems: List[Problem],
    dataset
) -> EvaluateResponse:
    """
    Evaluate a BlockWorkflow on multiple problems in parallel.
    
    Uses fire-all-at-once pattern for maximum throughput.
    """
    start_time = time.time()
    client = state.runpod_client
    
    # Run all problems in parallel
    tasks = [
        run_workflow_on_problem(
            problem,
            workflow.blocks,
            workflow.task_name,
            workflow.use_extractor,
            client
        )
        for problem in problems
    ]
    
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
        else:
            output, tokens, exec_time = result
            total_tokens += tokens
            
            # Evaluate correctness
            try:
                is_correct = dataset.evaluate(output, problem)
            except Exception as e:
                is_correct = False
            
            if is_correct:
                num_correct += 1
            
            problem_results.append(ProblemResult(
                problem_id=problem.id,
                correct=is_correct,
                tokens=tokens,
                time=exec_time
            ))
    
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
    
    return await evaluate_workflow_parallel(request.workflow, problems, dataset)


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
        use_extractor=request.use_extractor
    )
    
    dataset = state.datasets[task_name]
    problems = dataset.sample(request.num_problems, seed=request.seed)
    
    return await evaluate_workflow_parallel(workflow, problems, dataset)


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
    
    # Evaluate all workflows in parallel
    tasks = [
        evaluate_workflow_parallel(wf, problems, dataset)
        for wf in request.workflows
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
