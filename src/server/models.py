# src/server/models.py
"""
Pydantic models for the evaluation server API.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


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
    think: bool = Field(default=False, description="Enable extended thinking mode")


class SimpleWorkflowConfig(BaseModel):
    """Simple config with just role names (converts to AgentBlocks)."""
    roles: List[str] = Field(..., description="List of agent role names")
    task_name: str = Field(default="MBPP", description="Task/benchmark name")
    use_extractor: bool = Field(default=True, description="Whether to use answer extractor")
    think: bool = Field(default=False, description="Enable extended thinking mode")


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
    think: bool = Field(default=False, description="Enable extended thinking mode")
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
    completion_tokens: int
    total_time: float
    tokens_per_second: float
    problems: List[ProblemResult]

