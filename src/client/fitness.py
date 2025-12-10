# src/client/fitness.py
"""
Fitness evaluation functions for genetic algorithm optimization.
"""

from typing import List

from .models import BlockConfig
from .client import EvaluationClient


def evaluate_fitness_simple(
    roles: List[str],
    task_name: str = "MBPP",
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate fitness using simple role list.
    
    Fitness = pass@1 - (token_penalty * total_tokens)
    
    Args:
        roles: List of agent role names
        task_name: Dataset/task name
        num_problems: Number of problems to evaluate
        server_url: Evaluation server URL
        token_penalty: Penalty per token used
        
    Returns:
        Fitness score
    """
    client = EvaluationClient(server_url)
    result = client.evaluate_simple(roles, task_name, num_problems)
    
    if result.error:
        return 0.0
    
    return result.pass_at_1 - (token_penalty * result.total_tokens)


def evaluate_fitness(
    blocks: List[BlockConfig],
    task_name: str = "MBPP",
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate fitness using BlockConfig list.
    
    Fitness = pass@1 - (token_penalty * total_tokens)
    
    Args:
        blocks: List of BlockConfig objects
        task_name: Dataset/task name
        num_problems: Number of problems to evaluate
        server_url: Evaluation server URL
        token_penalty: Penalty per token used
        
    Returns:
        Fitness score
    """
    client = EvaluationClient(server_url)
    result = client.evaluate(blocks, task_name, num_problems)
    
    if result.error:
        return 0.0
    
    return result.pass_at_1 - (token_penalty * result.total_tokens)


def evaluate_block_workflow(
    workflow,  # BlockWorkflow from src.agents.workflow_block
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate a BlockWorkflow object via the server.
    
    Converts BlockWorkflow.blocks to BlockConfig list for API.
    
    Args:
        workflow: BlockWorkflow object
        num_problems: Number of problems to evaluate
        server_url: Evaluation server URL
        token_penalty: Penalty per token used
        
    Returns:
        Fitness score
    """
    # Import here to avoid circular imports
    from ..agents.block import AgentBlock, CompositeBlock
    
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
    
    return evaluate_fitness(
        blocks=blocks,
        task_name=workflow.task_name,
        num_problems=num_problems,
        server_url=server_url,
        token_penalty=token_penalty
    )
