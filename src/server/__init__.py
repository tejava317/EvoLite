# src/server/__init__.py
"""
EvoLite Evaluation Server Package.

High-throughput workflow evaluation using LangChain + vLLM with structured output.
"""

from .app import app, lifespan
from .models import (
    BlockConfig,
    WorkflowConfig,
    SimpleWorkflowConfig,
    EvaluateRequest,
    SimpleEvaluateRequest,
    BatchEvaluateRequest,
    ProblemResult,
    EvaluateResponse,
)
from .state import state, ServerState

__all__ = [
    "app",
    "lifespan",
    "BlockConfig",
    "WorkflowConfig",
    "SimpleWorkflowConfig",
    "EvaluateRequest",
    "SimpleEvaluateRequest",
    "BatchEvaluateRequest",
    "ProblemResult",
    "EvaluateResponse",
    "state",
    "ServerState",
]
