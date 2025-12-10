# src/client/__init__.py
"""
EvoLite Evaluation Client Package.

Provides sync and async interfaces for evaluating BlockWorkflows.
"""

from .models import BlockConfig, EvalResult, roles_to_blocks
from .client import EvaluationClient
from .fitness import (
    evaluate_fitness_simple,
    evaluate_fitness,
    evaluate_block_workflow,
)

__all__ = [
    "BlockConfig",
    "EvalResult",
    "roles_to_blocks",
    "EvaluationClient",
    "evaluate_fitness_simple",
    "evaluate_fitness",
    "evaluate_block_workflow",
]
