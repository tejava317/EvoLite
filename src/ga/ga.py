# src/ga/ga.py
import random
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union
from src.agents.agent import Agent
from src.agents.workflow import Workflow
from src.agents.block import Block, AgentBlock, CompositeBlock
from src.agents.workflow_block import BlockWorkflow
from src.agents.extractors import get_extractor_for_task
from src.config import ROLE_DESCRIPTIONS
from src.datasets import MBPPDataset, MathAlgebraDataset, BaseDataset
from src.evaluation.pass_at_k import quick_evaluate

# =============== CONFIGURATION ===============
POPULATION_SIZE = 100
GENERATIONS = 100
ELITISM_RATE = 0.5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1.0
MAX_WORKFLOW_LENGTH = 5

# Evaluation configuration
NUM_EVAL_PROBLEMS = 5  # Number of problems to evaluate per fitness calculation
EVAL_SEED = None  # Set to an integer for reproducible sampling
TOKEN_PENALTY = 0.0001  # Penalty coefficient for token usage

# Server-based evaluation configuration
EVAL_SERVER_URL = "http://localhost:8000"  # Evaluation server URL
USE_EVAL_SERVER = False  # Set to True to use the evaluation server

# Role name
ROLE_LIST = [role["name"] for role in ROLE_DESCRIPTIONS]

# Task Name (options: "MBPP", "HumanEval", "MATH")
TASK_NAME = "MBPP"

# Workflow type: "workflow" or "block"
WORKFLOW_TYPE = "block"

# =====================================================

# Global dataset cache to avoid reloading
_dataset_cache: Optional[BaseDataset] = None


def get_dataset(task_name: str) -> BaseDataset:
    """
    Get or create the dataset for the given task.
    Caches the dataset to avoid reloading on every fitness evaluation.
    """
    global _dataset_cache
    
    if _dataset_cache is not None:
        return _dataset_cache
    
    task_lower = task_name.lower()
    
    if 'mbpp' in task_lower:
        _dataset_cache = MBPPDataset(split="test")
    elif 'math' in task_lower or 'algebra' in task_lower:
        _dataset_cache = MathAlgebraDataset(split="test")
    elif 'humaneval' in task_lower:
        print(f"Warning: HumanEval not yet implemented, using MBPP")
        _dataset_cache = MBPPDataset(split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    _dataset_cache.load()
    print(f"Loaded {len(_dataset_cache)} problems from {task_name}")
    
    return _dataset_cache


def evaluate_fitness(workflow: Union[Workflow, BlockWorkflow], dataset: Optional[BaseDataset] = None) -> float:
    """
    Evaluate the fitness of a workflow using pass@k on the benchmark dataset.
    
    Fitness = pass@1 - (TOKEN_PENALTY * total_tokens)
    
    Works with both Workflow and BlockWorkflow.
    """
    if dataset is None:
        dataset = get_dataset(workflow.task_name)
    
    try:
        pass_at_1 = quick_evaluate(
            workflow=workflow,
            dataset=dataset,
            num_problems=NUM_EVAL_PROBLEMS,
            seed=EVAL_SEED
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        pass_at_1 = 0.0
    
    token_term = workflow.total_tokens or 0
    fitness = pass_at_1 - (TOKEN_PENALTY * token_term)
    
    return fitness


def evaluate_fitness_fast(workflow: Union[Workflow, BlockWorkflow]) -> float:
    """
    Fast fitness evaluation for development/debugging.
    Uses random values instead of actual LLM calls.
    """
    pass_at_k = random.random()
    token_term = workflow.total_tokens or 0
    return pass_at_k - (TOKEN_PENALTY * token_term)


def evaluate_fitness_server(workflow: Union[Workflow, BlockWorkflow], server_url: str = None) -> float:
    """
    Evaluate fitness using the evaluation server.
    Works with BlockWorkflow (converts blocks to API format).
    """
    from src.evaluation_client import EvaluationClient, BlockConfig, evaluate_block_workflow
    
    url = server_url or EVAL_SERVER_URL
    
    if isinstance(workflow, BlockWorkflow):
        # Use the dedicated BlockWorkflow evaluation
        return evaluate_block_workflow(
            workflow=workflow,
            num_problems=NUM_EVAL_PROBLEMS,
            server_url=url,
            token_penalty=TOKEN_PENALTY
        )
    else:
        # Legacy Workflow support - extract roles
        client = EvaluationClient(url)
        roles = [agent.role for agent in workflow.agents]
        
        try:
            result = client.evaluate_simple(
                roles=roles,
                task_name=workflow.task_name,
                num_problems=NUM_EVAL_PROBLEMS,
                use_extractor=hasattr(workflow, 'extractor') and workflow.extractor is not None
            )
            
            if result.error:
                print(f"Server evaluation error: {result.error}")
                return 0.0
            
            fitness = result.pass_at_1 - (TOKEN_PENALTY * result.total_tokens)
            return fitness
            
        except Exception as e:
            print(f"Server connection error: {e}")
            return 0.0


# From the agent list, select one random agent role.
def random_agent() -> str:
    return random.choice(ROLE_LIST)


def initialize_population(task_name: str, use_extractor: bool = True, workflow_type: str = "block"):
    """
    Initialize the population of workflows.
    
    Args:
        task_name: Name of the task/benchmark
        use_extractor: Whether to add an answer extractor to workflows (only for Workflow type)
        workflow_type: "workflow" for simple Workflow, "block" for BlockWorkflow
        
    Returns:
        List of {"workflow": Workflow|BlockWorkflow, "fitness": float} dictionaries
    """
    population = []
    
    if workflow_type == "workflow":
        dataset = get_dataset(task_name)
        extractor = get_extractor_for_task(task_name) if use_extractor else None
        
        for i in range(POPULATION_SIZE):
            length = random.randint(1, MAX_WORKFLOW_LENGTH)
            workflow_list = [random_agent() for _ in range(length)]
            workflow_description = " -> ".join(workflow_list)
            
            agents = [Agent(role=role, workflow_description=workflow_description) for role in workflow_list]
            
            workflow = Workflow(
                task_name=task_name,
                agents=agents,
                extractor=extractor.copy() if extractor else None
            )
            
            fitness = evaluate_fitness_fast(workflow)
            population.append({"workflow": workflow, "fitness": fitness})
    
    else:  # BlockWorkflow
        for _ in range(POPULATION_SIZE):
            length = random.randint(1, MAX_WORKFLOW_LENGTH)
            
            blocks = []
            for _ in range(length):
                if random.random() < 0.9:
                    role = random_agent()
                    block = AgentBlock(role)
                    blocks.append(block)
                else:
                    block = CompositeBlock()
                    block.expand("")
                    blocks.append(block)
            
            workflow = BlockWorkflow(task_name=task_name, blocks=blocks)
            fitness = evaluate_fitness_fast(workflow)
            population.append({"workflow": workflow, "fitness": fitness})
    
    return population


# Addition operator for BlockWorkflow
def addition(workflow: BlockWorkflow) -> BlockWorkflow:
    new_workflow = workflow.copy()
    
    if len(workflow.blocks) < MAX_WORKFLOW_LENGTH:
        new_agent_role = random_agent()
        idx = random.randint(0, len(new_workflow.blocks))
        
        if random.random() < 0.7:
            new_workflow.insert_block(AgentBlock(new_agent_role), idx)
        else:
            new_workflow.insert_block(CompositeBlock(), idx)
    
    return new_workflow


def deletion(workflow: BlockWorkflow) -> BlockWorkflow:
    new_workflow = workflow.copy()
    
    if len(new_workflow.blocks) > 1:
        idx = random.randint(0, len(new_workflow.blocks) - 1)
        new_workflow.remove_block(idx)
    
    return new_workflow


def crossover(parent1: BlockWorkflow, parent2: BlockWorkflow) -> BlockWorkflow:
    w1 = parent1.copy().blocks
    w2 = parent2.copy().blocks
    
    if len(w1) == 0 or len(w2) == 0:
        return parent1.copy()
    
    cut1 = random.randint(0, len(w1) - 1)
    cut2 = random.randint(0, len(w2) - 1)
    
    new_blocks = w1[:cut1] + w2[cut2:]
    new_blocks = new_blocks[:MAX_WORKFLOW_LENGTH]
    
    child = BlockWorkflow(task_name=parent1.task_name, blocks=new_blocks)
    return child


def mutate(workflow: BlockWorkflow) -> BlockWorkflow:
    if random.random() < 0.5:
        return addition(workflow)
    else:
        return deletion(workflow)


# Tournament selection
def select(population, k=3):
    contenders = random.sample(population, min(len(population), k))
    winners = sorted(contenders, key=lambda entry: entry["fitness"], reverse=True)
    return winners[0]


# =============== GA LOOP ===============
def run_ga(
    task_name: str,
    use_real_evaluation: bool = False,
    use_extractor: bool = True,
    verbose: bool = True,
    use_server: bool = False,
    server_url: str = None,
    workflow_type: str = "block"
):
    """
    Run the genetic algorithm to evolve multi-agent workflows.
    
    Args:
        task_name: Name of the task/benchmark (e.g., "MBPP", "MATH")
        use_real_evaluation: If True, use actual LLM calls for fitness
        use_extractor: If True, add answer extractor to workflows
        verbose: If True, print progress information
        use_server: If True, use evaluation server for fitness
        server_url: Evaluation server URL (defaults to EVAL_SERVER_URL)
        workflow_type: "workflow" for Workflow, "block" for BlockWorkflow
        
    Returns:
        Best workflow found
    """
    server_url = server_url or EVAL_SERVER_URL
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm")
        print(f"Task: {task_name}")
        print(f"Workflow type: {workflow_type}")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Generations: {GENERATIONS}")
        if use_server:
            print(f"Evaluation: Server ({server_url})")
        elif use_real_evaluation:
            print(f"Evaluation: Local (real LLM calls)")
        else:
            print(f"Evaluation: Fast (random)")
        print(f"Using extractor: {use_extractor}")
        print(f"{'='*60}\n")
    
    # Choose evaluation function
    if use_server:
        eval_fn = lambda wf: evaluate_fitness_server(wf, server_url)
    elif use_real_evaluation:
        eval_fn = evaluate_fitness
    else:
        eval_fn = evaluate_fitness_fast
    
    # Initialize population
    if verbose:
        print("Initializing population...")
    population = initialize_population(task_name, use_extractor, workflow_type)
    
    for generation in range(GENERATIONS):
        if verbose:
            best_fitness = max(p["fitness"] for p in population)
            avg_fitness = sum(p["fitness"] for p in population) / len(population)
            print(f"Generation {generation:3d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")
        
        new_population = []
        
        # Apply elitism
        elite_count = max(1, int(POPULATION_SIZE * ELITISM_RATE))
        sorted_pop = sorted(population, key=lambda entry: entry["fitness"], reverse=True)
        elites = sorted_pop[:elite_count]
        new_population.extend(elites)
        
        # Generate new population
        while len(new_population) < POPULATION_SIZE:
            parent1 = select(population)
            parent2 = select(population)
            child = {"workflow": None, "fitness": -float("inf")}
            
            child_workflow = crossover(parent1["workflow"], parent2["workflow"])
            
            if random.random() <= MUTATION_RATE:
                child_workflow = mutate(child_workflow)
            
            child["workflow"] = child_workflow
            child["fitness"] = eval_fn(child_workflow)
            new_population.append(child)
        
        population = new_population
    
    # Pick best solution
    best = max(population, key=lambda entity: entity["fitness"])
    best_workflow = best["workflow"]
    best_fitness = best["fitness"]
    
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL BEST WORKFLOW")
        print(f"{'='*60}")
        print(f"{best_workflow}")
        print(f"\nFitness: {best_fitness:.5f}")
        print(f"{'='*60}\n")
    
    return best


def evaluate_workflow(
    workflow: Union[Workflow, BlockWorkflow],
    task_name: str = None,
    num_problems: int = 50,
    verbose: bool = True
) -> dict:
    """
    Thoroughly evaluate a workflow on a dataset.
    """
    from src.evaluation.pass_at_k import evaluate_pass_at_k
    
    task = task_name or workflow.task_name
    dataset = get_dataset(task)
    
    if verbose:
        print(f"\nEvaluating workflow on {num_problems} problems from {task}...")
    
    results = evaluate_pass_at_k(
        workflow=workflow,
        dataset=dataset,
        num_problems=num_problems,
        samples_per_problem=1,
        k=1
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Pass@1: {results['pass_at_k']:.4f}")
        print(f"  Correct: {results['num_correct']}/{results['total_samples']}")
    
    return results


# Run the algorithm
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GA to evolve multi-agent workflows")
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name (MBPP, MATH)")
    parser.add_argument("--real", action="store_true", help="Use real LLM evaluation (local)")
    parser.add_argument("--server", action="store_true", help="Use evaluation server")
    parser.add_argument("--server-url", type=str, default=EVAL_SERVER_URL, help="Evaluation server URL")
    parser.add_argument("--no-extractor", action="store_true", help="Disable answer extractor")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--type", type=str, default="block", choices=["workflow", "block"],
                        help="Workflow type: 'workflow' or 'block'")
    
    args = parser.parse_args()
    
    best = run_ga(
        task_name=args.task,
        use_real_evaluation=args.real,
        use_extractor=not args.no_extractor,
        verbose=not args.quiet,
        use_server=args.server,
        server_url=args.server_url,
        workflow_type=args.type
    )
    
    # Print final best workflow
    if isinstance(best['workflow'], BlockWorkflow):
        best['workflow']._expand_blocks_to_agents()
        print(f"\nBest workflow roles: {[a.role for a in best['workflow'].agents]}")
    else:
        print(f"\nBest workflow roles: {[a.role for a in best['workflow'].agents]}")
    print(f"Fitness: {best['fitness']:.4f}")
