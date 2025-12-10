# src/ga/ga.py
import random
import yaml
import numpy as np
import time
import asyncio
import functools
import math

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
from src.ga.multi_objective import *
from src.ga.checkpoint import *


# Set flushing as True.
print = functools.partial(print, flush=True)

# =============== CONFIGURATION ===============

# Evaluation configuration
NUM_EVAL_PROBLEMS = 30  # Number of problems to evaluate per fitness calculation
EVAL_SEED = None  # Set to an integer for reproducible sampling
TOKEN_PENALTY = 0.0001  # Penalty coefficient for token usage

# Server-based evaluation configuration
EVAL_SERVER_URL = "http://localhost:8000"  # Evaluation server URL
USE_EVAL_SERVER = False  # Set to True to use the evaluation server

# Role name
ROLE_LIST = ROLE_DESCRIPTIONS

# Task Name (options: "MBPP", "HumanEval", "MATH")
TASK_NAME = "MBPP"

# Workflow type: "workflow" or "block"
WORKFLOW_TYPE = "block"

# =====================================================

# Global dataset cache to avoid reloading
_dataset_cache: Optional[BaseDataset] = None

# Caching the workflow.
workflow_set = set()

# Get a dataset from specified task.
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

# Fast fitness evaluation with the random number.
def evaluate_fitness_fast(workflows, num_problems, server_url: str) -> float:
    
    results = []
    for wf in workflows:
        pass_at_k = random.random()
        token_term = random.random()
        results.append({"pass_at_k": pass_at_k, "token" : token_term})
    return results

# Evaluate fitness using the evaluation server.
# Works with BlockWorkflow.
def evaluate_fitness_server(workflow: Union[Workflow, BlockWorkflow], server_url: str = None) -> float:
    
    # Load for the fast API server.
    from src.evaluation_client import EvaluationClient, BlockConfig, evaluate_block_workflow
    
    url = server_url or EVAL_SERVER_URL
    
    if isinstance(BlockWorkflow):
        # Use the dedicated BlockWorkflow evaluation
        return evaluate_block_workflow(
            workflow=workflow,
            num_problems=NUM_EVAL_PROBLEMS,
            server_url=url,
            token_penalty=TOKEN_PENALTY
        )
    else:
        raise NotImplementedError("Legacy Workflow is NOT supported.")

"""
Asynchronously evaluate the fitness.
To avoid the timeout, we use the static batch size.

    Args:
        workflow: BlockWorkflow class
        server_url: Fast API server url
        batch_size: Size of batch to estimate asynchronously
    Return:
        List[EvaluationClient]: The list of result.

"""

def evaluate_fitness_batch(workflows, num_problems, server_url: str = None, batch_size: int = 15):

    async def _run():
        from src.evaluation_client import EvaluationClient, BlockConfig
        
        url = server_url or EVAL_SERVER_URL

        task_name = args.task
        if len({wf.task_name for wf in workflows}) != 1:
            raise ValueError("All workflows must have the same task_name.")

        client = EvaluationClient(url)
        all_results = []

        # Use a batch
        for start in range(0, len(workflows), batch_size):
            batch = workflows[start:start + batch_size]

            # Convert batch → BlockConfig list
            batch_blocks = []
            for wf in batch:
                blocks = []
                for block in wf.blocks:
                    if isinstance(block, AgentBlock):
                        blocks.append(BlockConfig(type="agent", role=block.role))
                    elif isinstance(block, CompositeBlock):
                        blocks.append(BlockConfig(
                            type="composite",
                            divider_role=block.divider_role,
                            synth_role=block.synth_role
                        ))
                batch_blocks.append(blocks)

            # === async evaluate ===
            respond = await client.evaluate_batch_async(
                workflows=batch_blocks,
                task_name=task_name,
                num_problems=num_problems,
                use_extractor=False,
                seed=43211,
                think=False,
            )

            # respond is already a list of EvalResult
            all_results.extend(respond)

        await client.close()
        return all_results

    return asyncio.run(_run())



# From the agent list, select one random agent role.
def random_agent() -> str:
    return random.choice(ROLE_LIST)


"""
Initialize the population of workflows.
   Args:
        task_name: Name of the task/benchmark
        use_extractor: Whether to add an answer extractor to workflows (only for Workflow type)
        workflow_type: "workflow" for simple Workflow, "block" for BlockWorkflow
        
    Returns:
        List of {"workflow": Workflow|BlockWorkflow, "fitness": float} dictionaries
"""

def initialize_population(task_name: str, server_url: str, use_extractor: bool = True, initial_num_problem: int = 5):

    population = []
    workflows = []
    start_time = time.time()
    
    if args.fast:
        eval_fn = evaluate_fitness_fast
    else:
        eval_fn = evaluate_fitness_batch

    while len(population) < args.population_size:
        
        length = random.randint(1, args.max_workflow)
        blocks = []
        
        # Add the agent block.
        for _ in range(length):
            role = random_agent()
            block = AgentBlock(role)
            blocks.append(block)
        
        workflow = BlockWorkflow(task_name=task_name, blocks=blocks)
        
        wf_string = workflow.workflow_to_string()

        if wf_string in workflow_set:
            continue

        workflow_set.add(wf_string)
        workflows.append(workflow)
        population.append({"workflow": workflow, "fitness": -float("inf")})
    
    result = eval_fn(workflows, initial_num_problem, server_url)
    
    for i in range(len(workflows)):
        if args.fast:
            population[i]["fitness"] = result[i]
        else:    
            population[i]["fitness"] = {"pass_at_k": result[i].pass_at_1, "token" : result[i].total_tokens}
    
    # Record the time.
    end_time = time.time()
    if not args.quiet:
        print(f"Initialize {args.population_size} for {end_time - start_time:.4f}s.")

    return population


# Addition operator for BlockWorkflow
def addition(entry)->BlockWorkflow:

    workflow = entry["workflow"]
    new_workflow = workflow.copy()
    
    if len(workflow.blocks) < args.max_workflow:
        new_agent_role = random_agent()
        idx = random.randint(0, len(new_workflow.blocks))
        
        if random.random() < 1.0:
            new_workflow.insert_block(AgentBlock(new_agent_role), idx)
        else:
            new_workflow.insert_block(CompositeBlock(), idx)
    
    return new_workflow


# Deletion operator for BlockWorkflow
def deletion(entry)->BlockWorkflow:
    
    workflow = entry["workflow"]
    new_workflow = workflow.copy()
    
    if len(new_workflow.blocks) > 1:
        idx = random.randint(0, len(new_workflow.blocks) - 1)
        new_workflow.remove_block(idx)
    
    return new_workflow


# Crossover operator for BlockWorkflow
def crossover(parent1, parent2) -> BlockWorkflow:
    
    w1 = parent1["workflow"].copy().blocks
    w2 = parent2["workflow"].copy().blocks
    
    if len(w1) <= 1 or len(w2) <= 1:
        return parent1["workflow"].copy()
    
    cut1 = random.randint(1, len(w1) - 1)
    cut2 = random.randint(1, len(w2) - 1)
    
    new_blocks = w1[:cut1] + w2[cut2:]
    new_blocks = new_blocks[:args.max_workflow]
    
    child = BlockWorkflow(task_name=parent1["workflow"].task_name, blocks=new_blocks)
    return child


# Mutation operator for BlockWorkflow
# An addition or deletion is selectively chosen.
def mutate(entry) -> BlockWorkflow:
    
    # Workflow length
    w = entry["workflow"].blocks
    w_len = len(w)

    # Flexibly alter the workflow length.
    # Regarding a momentum by storing the evolution direction.
    max_len = args.max_workflow
    deletion_bound = int(max_len * 0.7)
    addition_bound = int(max_len * 0.3)

    # if max_len >= deletion_bound:
    #     return deletion(entry)
    # elif max_len <= deletion_bound:
    #     return addition(entry)
    # else:

    
    x = w_len / max_len
    prob = 1 - x
    
    if random.random() < prob:
        return addition(entry)
    else:
        return deletion(entry)


# Selection
def select(population, k=3):
    contenders = random.sample(population, min(len(population), k))
    return contenders[0]


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

# =============== GA LOOP ===============
def run_ga(
    task_name: str,
    use_extractor: bool = False,
    verbose: bool = True,
    server_url: str = None,
    use_batch: bool = True,
    key: str = "ga"
):

    server_url = server_url or EVAL_SERVER_URL
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm")
        print(f"Task: {task_name}")
        print(f"Population size: {args.population_size}")
        print(f"Generations: {args.generation}")
        print(f"Evaluation: Batch with size 15.")
        print(f"Using extractor: {use_extractor}")
        print(f"Max workflow number: {args.max_workflow}")
        print(f"Fast: {args.fast}")
        print(f"Mutation Rate: {args.mutation_rate}")
        print(f"Phase: {args.num_phase}")
        print(f"{'='*60}\n")
    
    # Choose evaluation function
    # Let's use batch evalution in default.
    if args.fast:
        eval_fn = evaluate_fitness_fast
    else:
        eval_fn = evaluate_fitness_batch
    
    # Initialize population
    if verbose:
        print("Initializing population...")

    # Alter initial problem number mode.num_problem = args.problem_number
    phase_num = args.num_phase
    phase_num_problem = [
        (args.num_problem * (i + 1)) // phase_num
        for i in range(phase_num)
    ]
    assert len(phase_num_problem) > 0, "The phase does NOT work."

    population = initialize_population(task_name, server_url, use_extractor, phase_num_problem[0])
    
    for generation in range(args.generation):
        
        if verbose:
            best_fitness = max(p["fitness"]["pass_at_k"] for p in population)
            avg_fitness = sum(p["fitness"]["pass_at_k"] for p in population) / len(population)
            print(f"Generation {(generation + 1):3d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")
        
        new_population = []
        
        # Apply elitism for multi ga.
        elite_count = max(1, int(args.population_size * args.eliticism_rate))
        elites = ngsa_select(population, elite_count)
        new_population.extend(elites)
        

        # Generate new population
        child_workflows = []
        child_population = []
        num_children = args.population_size - len(new_population)
        
        while len(child_population) < num_children:
            
            parent1 = select(population)
            parent2 = select(population)
            child = {"workflow": None, "fitness": -float("inf")}
            
            child_workflow = crossover(parent1, parent2)
            child["workflow"] = child_workflow
            
            # Mutation, Essential Progress.
            if random.random() < args.mutation_rate:
                child_workflow = mutate(child)
                child["workflow"] = child_workflow
            
            child_wf_string = child_workflow.workflow_to_string()

            if child_wf_string in workflow_set:
                continue

            workflow_set.add(child_wf_string)
            child_workflows.append(child_workflow)
            child_population.append(child)
        
        # Perform fitness evaluation.
        total_generations = args.generation
        num_phases = len(phase_num_problem)
        gens_per_phase = max(1, total_generations // num_phases)
        idx = min(generation // gens_per_phase, num_phases - 1)
        
        result = eval_fn(child_workflows, phase_num_problem[idx], server_url)
        for i in range(len(child_population)):
            if args.fast:
                child_population[i]["fitness"] = result[i]
            else:   
                child_population[i]["fitness"] = {"pass_at_k": result[i].pass_at_1, "token" : result[i].total_tokens}

        population = new_population + child_population

        # Record activity.
        if (generation + 1) % 5 == 0:
            
            # Save with .csv file and plot the pareto front.
            save_checkpoint_csv(population, f"{key}_{generation+1}")
            plot_pareto(population, f"{key}_{generation+1}")

    if verbose:
        print("Done.")

    return population


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
    
    # Evaluation information
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name (MBPP, MATH)")
    parser.add_argument("--server-url", type=str, default=EVAL_SERVER_URL, help="Evaluation server URL")
    parser.add_argument("--no-extractor", action="store_true", help="Disable answer extractor")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--batch", action="store_true", help="Perform batch fitness evaluation")
    parser.add_argument("--fast", action="store_true", help="Random number fast evaluation")
    parser.add_argument("--alter-question", action="store_true", help="Alter question mode")


    # GA information
    parser.add_argument("--population-size", type=int, default=100, help="The size of population")
    parser.add_argument("--generation", type=int, default=100, help="The number of generations")
    parser.add_argument("--eliticism-rate", type=float, default=0.5, help="The rate of eliticism")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="The rate of mutation")
    parser.add_argument("--crossover-rate", type=float, default=1.0, help="The rate of crossover")
    parser.add_argument("--max-workflow", type=int, default=5, help="The length of max workflow")
    parser.add_argument("--num_problem", type=int, default=30, help="The number of problems")
    parser.add_argument("--num_phase", type=int, default=1, help="The number of problem phases.")

    # File name information
    parser.add_argument("--key", type=str, default="ga_exp", help="The key to distinguish the experiment")
    
    args = parser.parse_args()
    
    # Run a single workflow evaluation.
    # role = random_agent()
    # block = AgentBlock(role)
    # workflow = BlockWorkflow(task_name="MBPP", blocks=[block])    # BlockWorkflow
    # result = evaluate_fitness_batch([workflow], args.num_problem, EVAL_SERVER_URL)
    # print(f"Workflow: {workflow.workflow_to_string()}")
    # print(f"Pass@1: {result[0].pass_at_1}")
    # print(f"Total Tokens: {result[0].total_tokens}")


    final_population = run_ga(
        task_name=args.task,
        use_extractor=not args.no_extractor,
        verbose=not args.quiet,
        server_url=args.server_url,
        use_batch=args.batch,
        key=args.key
    )