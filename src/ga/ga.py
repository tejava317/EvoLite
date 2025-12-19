# src/ga/ga.py
# Genetic Algorithm with Buffer (Probation) Strategy and Evaluation Count Limiting
import random
import yaml
import numpy as np
import time
import asyncio
import functools
import math
from datetime import datetime

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
EVAL_SERVER_URL = "http://localhost:8001"  # Evaluation server URL
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
    from src.client import EvaluationClient, BlockConfig, evaluate_block_workflow
    
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
        from src.client import EvaluationClient, BlockConfig
        
        url = server_url or EVAL_SERVER_URL

        task_name = args.task
        if len({wf.task_name for wf in workflows}) != 1:
            raise ValueError("All workflows must have the same task_name.")

        client = EvaluationClient(url)
        all_results = []

        # Use a batch
        total_batches = (len(workflows) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(workflows), batch_size)):
            batch = workflows[start:start + batch_size]
            print(f"[Batch {batch_idx + 1}/{total_batches}] Evaluating {len(batch)} workflows...")

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
            print(f"[Batch {batch_idx + 1}/{total_batches}] Completed.")

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
        List of {"workflow": Workflow|BlockWorkflow, "fitness": float, "eval_count": int, "generation_age": int} dictionaries
"""

def initialize_population(task_name: str, server_url: str, use_extractor: bool = True, initial_num_problem: int = 5):

    population = []
    workflows = []
    start_time = time.time()
    
    if args.fast:
        eval_fn = evaluate_fitness_fast
    else:
        eval_fn = lambda wfs, num, url: evaluate_fitness_batch(wfs, num, url, args.batch_size)

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
        # Initialize with eval_count=0 and generation_age=0
        population.append({
            "workflow": workflow, 
            "fitness": -float("inf"),
            "eval_count": 0,
            "generation_age": 0,
            "total_evaluated_problems": 0,
            "was_in_buffer": False  # Track if this individual was in buffer before
        })
    
    if not args.quiet:
        print(f"Generated {len(workflows)} unique workflows. Starting evaluation...")
    
    result = eval_fn(workflows, initial_num_problem, server_url)
    
    for i in range(len(workflows)):
        if args.fast:
            population[i]["fitness"] = result[i]
        else:    
            population[i]["fitness"] = {"pass_at_k": result[i].pass_at_1, "token" : result[i].total_tokens}
        # After first evaluation, eval_count becomes 1
        population[i]["eval_count"] = 1
        population[i]["total_evaluated_problems"] = initial_num_problem
    
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
Select buffer (probation) candidates from the population.
Buffer candidates are individuals that were survivors in previous generations
but fell out of the top N in the current generation.

Buffer priority:
1. Higher Front level (Front 0 > Front 1 > Front 2 > ...) - 낮은 Front 번호가 우선
2. Higher generation_age (veterans who survived longer)
3. Higher fitness (pass_at_k)

Args:
    candidates: List of individuals that didn't make it to survivors
    buffer_size: Maximum number of buffer candidates to select
    fronts: List of fronts from NSGA-II sorting (fronts[0] = Front 0, fronts[1] = Front 1, ...)
    population: Full population list (to map indices)
    
Returns:
    List of buffer candidates
"""
def select_buffer(candidates, buffer_size: int, fronts, population):
    buffer_list = []
    
    # Build a mapping from individual object to its Front level
    # Front 0 = level 0, Front 1 = level 1, etc.
    individual_to_front_level = {}
    for front_level, front_indices in enumerate(fronts):
        for idx in front_indices:
            # Use object id to map individuals to their front level
            individual_to_front_level[id(population[idx])] = front_level
    
    # Sort candidates by:
    # 1. Front level (lower is better, Front 0 > Front 1 > Front 2)
    #    Use negative because we want lower front numbers first (Front 0 before Front 1)
    # 2. generation_age (higher is better - veterans who survived longer)
    # 3. fitness pass_at_k (higher is better)
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            -individual_to_front_level.get(id(x), 999),  # Negative: Front 0 (0) > Front 1 (1) > Front 2 (2)
            x.get("generation_age", 0),
            x["fitness"]["pass_at_k"]
        ),
        reverse=True
    )
    
    for agent in sorted_candidates:
        if len(buffer_list) >= buffer_size:
            break
        
        # Select individuals that have generation_age > 0 (were survivors before)
        # IMPORTANT: Exclude individuals that were already in buffer before
        # Buffer purpose: give ONE chance to veterans who fell out of survivors
        # If they fail again, they should be removed, not given another buffer chance
        if agent.get("generation_age", 0) > 0 and not agent.get("was_in_buffer", False):
            buffer_list.append(agent)
            agent["was_in_buffer"] = True  # Mark as having been in buffer
    
    return buffer_list


"""
Run the genetic algorithm with Buffer (Probation) Strategy.

Args:
    task_name: Name of the task/benchmark (e.g., "MBPP", "MATH")
    use_extractor: If True, add answer extractor to workflows
    verbose: If True, print progress information
    server_url: Evaluation server URL (defaults to EVAL_SERVER_URL)
    use_batch: If True, use batch evaluation
    key: Key for saving checkpoints
    
Returns:
    Final population
"""

# =============== GA LOOP WITH BUFFER ===============
def run_ga(
    task_name: str,
    use_extractor: bool = False,
    verbose: bool = True,
    server_url: str = None,
    use_batch: bool = True,
    key: str = "ga",
    finalize_valid: int = 100
):

    server_url = server_url or EVAL_SERVER_URL
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm with Buffer Strategy")
        print(f"Task: {task_name}")
        print(f"Population size: {args.population_size}")
        print(f"Generations: {args.generation}")
        print(f"Elitism rate: {args.eliticism_rate}")
        print(f"Buffer size: {args.buffer_size}")
        print(f"Max eval iterations per individual: {args.max_eval_iter}")
        print(f"Evaluation: Batch with size {args.batch_size}.")
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
        eval_fn = lambda wfs, num, url: evaluate_fitness_batch(wfs, num, url, args.batch_size)
    
    # Initialize Pareto front archive to store removed Front 0 individuals
    pareto_front_archive = []
    
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
    
    # For generation 0, also sort and select survivors for consistent plotting
    workflows = [entry["workflow"] for entry in population]
    objs = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in population])
    
    fronts = non_dominated_sort(objs)
    
    sorted_indices = []
    for front in fronts:
        if len(sorted_indices) + len(front) > len(population):
            # Need to use crowding distance for the last front
            dist = crowding_distance(objs, front)
            idx_sorted = np.argsort(-dist)
            for idx in idx_sorted[:len(population) - len(sorted_indices)]:
                sorted_indices.append(front[idx])
            break
        else:
            sorted_indices.extend(front)
    
    sorted_population = [population[i] for i in sorted_indices]
    
    # Extract actual Pareto front (first front only) for plotting
    pareto_front_indices_gen0 = fronts[0] if fronts else []
    pareto_front_individuals_gen0 = [population[i] for i in pareto_front_indices_gen0]
    
    # Select survivors for generation 0 (same logic as later generations)
    survivor_count = max(1, int(args.population_size * args.eliticism_rate))
    survivors_gen0 = sorted_population[:survivor_count]
    buffer_list_gen0 = []  # No buffer for generation 0
    
    # Save initial population (generation 0)
    save_checkpoint_csv(population, f"{key}_0")
    plot_pareto(population, f"{key}_0", survivors=survivors_gen0, buffer_list=buffer_list_gen0, pareto_front=pareto_front_individuals_gen0)
    
    for generation in range(args.generation):
        gen_start_time = datetime.now()
        gen_start_str = gen_start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        if verbose:
            best_fitness = max(p["fitness"]["pass_at_k"] for p in population)
            avg_fitness = sum(p["fitness"]["pass_at_k"] for p in population) / len(population)
            total_eval_count = sum(p.get("eval_count", 0) for p in population)
            avg_eval_count = total_eval_count / len(population) if population else 0
            print(f"Generation {(generation + 1):3d} | Start: {gen_start_str} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Avg Eval Count: {avg_eval_count:.2f}")
        
        # ========== STEP 1: Evaluate all individuals in current pool ==========
        # Filter individuals that haven't exceeded max_eval_iter
        # All individuals are evaluated at the same time if they haven't exceeded the limit
        eval_candidates = []
        eval_indices = []
        
        for idx, individual in enumerate(population):
            eval_count = individual.get("eval_count", 0)
            generation_age = individual.get("generation_age", 0)
            # Only re-evaluate individuals that have lived at least 1 generation.
            # Newborn offspring were evaluated at birth, so skip them here to avoid double evaluation.
            if generation_age > 0 and eval_count < args.max_eval_iter:
                eval_candidates.append(individual)
                eval_indices.append(idx)
        
        if len(eval_candidates) > 0:
            if verbose:
                print(f"Evaluating {len(eval_candidates)} individuals (out of {len(population)} total)")
            
            # Determine which phase we're in
            total_generations = args.generation
            num_phases = len(phase_num_problem)
            gens_per_phase = max(1, total_generations // num_phases)
            phase_idx = min(generation // gens_per_phase, num_phases - 1)
            
            # Extract workflows for evaluation
            eval_workflows = [ind["workflow"] for ind in eval_candidates]
            
            # Perform evaluation
            result = eval_fn(eval_workflows, phase_num_problem[phase_idx], server_url)
            
            # Update fitness and eval_count for evaluated individuals using weighted average
            num_problems_this_eval = phase_num_problem[phase_idx]
            for i, orig_idx in enumerate(eval_indices):
                if args.fast:
                    population[orig_idx]["fitness"] = result[i]
                else:
                    individual = population[orig_idx]
                    old_total_problems = individual.get("total_evaluated_problems", 0)
                    old_pass_at_k = individual["fitness"]["pass_at_k"]
                    old_token = individual["fitness"]["token"]
                    
                    new_pass_at_1 = result[i].pass_at_1
                    new_total_tokens = result[i].total_tokens
                    
                    # Weighted average: (old_total * old_value + new_num * new_value) / (old_total + new_num)
                    if old_total_problems > 0:
                        total_problems = old_total_problems + num_problems_this_eval
                        new_pass_at_k = (old_total_problems * old_pass_at_k + num_problems_this_eval * new_pass_at_1) / total_problems
                        new_token = (old_total_problems * old_token + num_problems_this_eval * new_total_tokens) / total_problems
                    else:
                        # First evaluation
                        new_pass_at_k = new_pass_at_1
                        new_token = new_total_tokens
                    
                    population[orig_idx]["fitness"] = {
                        "pass_at_k": new_pass_at_k,
                        "token": new_token
                    }
                    population[orig_idx]["total_evaluated_problems"] = old_total_problems + num_problems_this_eval
                
                # Increment eval_count
                population[orig_idx]["eval_count"] = population[orig_idx].get("eval_count", 0) + 1
        else:
            if verbose:
                print(f"All individuals have reached max_eval_iter ({args.max_eval_iter}). Skipping evaluation.")
        
        # ========== STEP 2: Sort population using NSGA-II ==========
        # Use NSGA-II to sort the entire population
        workflows = [entry["workflow"] for entry in population]
        objs = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in population])
        
        fronts = non_dominated_sort(objs)
        
        sorted_indices = []
        for front in fronts:
            if len(sorted_indices) + len(front) > len(population):
                # Need to use crowding distance for the last front
                dist = crowding_distance(objs, front)
                idx_sorted = np.argsort(-dist)
                for idx in idx_sorted[:len(population) - len(sorted_indices)]:
                    sorted_indices.append(front[idx])
                break
            else:
                sorted_indices.extend(front)
        
        sorted_population = [population[i] for i in sorted_indices]
        
        # Extract actual Pareto front (first front only) for plotting
        pareto_front_indices = fronts[0] if fronts else []
        pareto_front_individuals = [population[i] for i in pareto_front_indices]
        
        # ========== STEP 3: Select Survivors (Parents) ==========
        # Use elitism_rate to determine how many survive
        survivor_count = max(1, int(args.population_size * args.eliticism_rate))
        
        # Select Front 0 (Pareto front) first. If it fits within survivor_count, keep all;
        # otherwise, keep the top survivor_count from Front 0 using crowding distance.
        front_0_individuals = [population[i] for i in pareto_front_indices]
        
        if len(front_0_individuals) <= survivor_count:
            remaining_slots = survivor_count - len(front_0_individuals)
            
            if remaining_slots > 0:
                other_front_individuals = []
                slots_remaining = remaining_slots
                
                for front_idx in range(1, len(fronts)):  # Skip Front 0
                    if slots_remaining <= 0:
                        break
                    
                    front_indices = [idx for idx in fronts[front_idx] if idx not in pareto_front_indices]
                    if not front_indices:
                        continue
                    
                    front_objs = objs[front_indices]
                    front_dist = crowding_distance(front_objs, list(range(len(front_indices))))
                    front_sorted_by_dist = sorted(range(len(front_indices)), key=lambda i: -front_dist[i])
                    
                    num_from_this_front = min(slots_remaining, len(front_indices))
                    selected_from_front = [front_indices[i] for i in front_sorted_by_dist[:num_from_this_front]]
                    other_front_individuals.extend([population[i] for i in selected_from_front])
                    slots_remaining -= num_from_this_front
                    
                    if slots_remaining <= 0:
                        break
                
                survivors = front_0_individuals + other_front_individuals
            else:
                survivors = front_0_individuals
        else:
            # Front 0 has more individuals than survivor_count
            # Select top survivor_count using crowding distance, and archive the rest
            front_0_objs = objs[pareto_front_indices]
            front_0_dist = crowding_distance(front_0_objs, list(range(len(pareto_front_indices))))
            front_0_sorted_by_dist = sorted(range(len(pareto_front_indices)), key=lambda i: -front_0_dist[i])
            selected_front_0_indices = [pareto_front_indices[i] for i in front_0_sorted_by_dist[:survivor_count]]
            survivors = [population[i] for i in selected_front_0_indices]
            
            # Archive removed Front 0 individuals (they were in Pareto front but removed due to size limit)
            removed_front_0_indices = [pareto_front_indices[i] for i in front_0_sorted_by_dist[survivor_count:]]
            for idx in removed_front_0_indices:
                # Deep copy to preserve state at this generation
                archived_individual = deepcopy(population[idx])
                pareto_front_archive.append(archived_individual)
            
            if verbose and len(removed_front_0_indices) > 0:
                print(f"Archived {len(removed_front_0_indices)} Front 0 individuals (Pareto front size limit)")
        
        # Increment generation_age for survivors
        # IMPORTANT: Reset was_in_buffer flag for survivors - they earned their place back
        for survivor in survivors:
            survivor["generation_age"] = survivor.get("generation_age", 0) + 1
            survivor["was_in_buffer"] = False  # Reset buffer flag - they're survivors now
        
        # ========== STEP 4: Select Buffer (Probation) Candidates ==========
        # Buffer purpose: Give probation to veterans (generation_age > 0) who fell out of survivors
        # Priority: Higher Front level (Front 0 > Front 1 > ...) > Higher generation_age > Higher fitness
        # This protects veterans from being eliminated by potentially lucky new offspring
        
        # First, identify all individuals that were NOT selected as survivors
        survivor_ids = set(id(s) for s in survivors)
        non_survivors = [ind for ind in population if id(ind) not in survivor_ids]
        
        # Select buffer from non-survivors, prioritizing by Front level, generation_age, and fitness
        buffer_list = select_buffer(non_survivors, args.buffer_size, fronts, population)
        
        # Dead individuals are those not selected as survivors or buffer
        # They will be removed and not included in the next generation's evaluation pool
        selected_set = set(id(s) for s in survivors) | set(id(b) for b in buffer_list)
        dead_individuals = [ind for ind in population if id(ind) not in selected_set]
        dead_count = len(dead_individuals)
        
        if verbose:
            print(f"Selected {len(survivors)} survivors and {len(buffer_list)} buffer candidates")
            if dead_count > 0:
                print(f"Removed {dead_count} dead individuals (not selected as survivors or buffer)")
            print(f"Pareto front (first front) has {len(pareto_front_individuals)} individuals")
        
        # ========== STEP 5: Generate Offspring ==========
        # Survivors and buffer can participate in reproduction
        # Generate enough offspring to fill population_size
        child_workflows = []
        child_population = []
        num_children = args.population_size - survivor_count
        
        # Create parent pool: survivors + buffer (both can reproduce)
        parent_pool = survivors + buffer_list
        
        while len(child_population) < num_children:
            
            parent1 = select(parent_pool)
            parent2 = select(parent_pool)
            child = {
                "workflow": None, 
                "fitness": -float("inf"),
                "eval_count": 0,
                "generation_age": 0,
                "total_evaluated_problems": 0,
                "was_in_buffer": False  # New offspring, never in buffer
            }
            
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
        
        # Evaluate offspring immediately after generation
        # They start with eval_count=0, so they will always be evaluated here
        if len(child_workflows) > 0:
            total_generations = args.generation
            num_phases = len(phase_num_problem)
            gens_per_phase = max(1, total_generations // num_phases)
            phase_idx = min(generation // gens_per_phase, num_phases - 1)
            
            result = eval_fn(child_workflows, phase_num_problem[phase_idx], server_url)
            num_problems_this_eval = phase_num_problem[phase_idx]
            for i in range(len(child_population)):
                if args.fast:
                    child_population[i]["fitness"] = result[i]
                else:   
                    child_population[i]["fitness"] = {
                        "pass_at_k": result[i].pass_at_1, 
                        "token": result[i].total_tokens
                    }
                child_population[i]["eval_count"] = 1
                child_population[i]["total_evaluated_problems"] = num_problems_this_eval
        
        # ========== STEP 6: Compose Next Generation Pool ==========
        # Next Gen Pool = Survivors(survivor_count) + Offspring(population_size - survivor_count) + Buffer(K)
        # IMPORTANT: Only survivors and buffer survive to the next generation
        # Dead individuals (not selected as survivors or buffer) are removed immediately
        # This means next generation starts with only survivors + buffer (not including dead individuals)
        # Total = population_size + buffer_size (population size is maintained, buffer is extra)
        # 
        # Remove dead individuals: only keep survivors and buffer for next generation evaluation
        # Dead individuals are those in the current population but not selected as survivors or buffer
        population = survivors + child_population + buffer_list
        
        if verbose:
            print(f"Next generation pool size: {len(population)} (Survivors: {len(survivors)}, Offspring: {len(child_population)}, Buffer: {len(buffer_list)})")
            print(f"  -> Core population: {len(survivors) + len(child_population)} (maintains target size)")

        # Record activity - save every generation
        # Save with .csv file and plot the pareto front.
        save_checkpoint_csv(population, f"{key}_{generation+1}")
        
        # IMPORTANT: Re-compute pareto_front_individuals from the new population
        # because population was reconstructed (survivors + children + buffer)
        # and pareto_front_individuals may reference old population objects
        # Re-calculate Pareto front from the new population
        objs_new = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in population])
        fronts_new = non_dominated_sort(objs_new)
        pareto_front_indices_new = fronts_new[0] if fronts_new else []
        pareto_front_individuals_new = [population[i] for i in pareto_front_indices_new]
        
        # Pass actual Pareto front (first front) for line connection, not all survivors
        plot_pareto(population, f"{key}_{generation+1}", survivors=survivors, buffer_list=buffer_list, pareto_front=pareto_front_individuals_new)
        
        # Log generation end time and elapsed time
        gen_end_time = datetime.now()
        gen_end_str = gen_end_time.strftime("%Y-%m-%d %H:%M:%S")
        gen_elapsed = (gen_end_time - gen_start_time).total_seconds()
        if verbose:
            print(f"Generation {(generation + 1):3d} | End: {gen_end_str} | Elapsed: {gen_elapsed:.2f}s")

    if verbose:
        print("Done.")
    
    # ========== FINALIZATION: Combine archive + final front and validate ==========
    if verbose:
        print(f"\n{'='*60}")
        print("Finalization: Combining Pareto front archive with final generation")
        print(f"{'='*60}")
    
    # Get final Pareto front from last generation
    workflows_final = [entry["workflow"] for entry in population]
    objs_final = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in population])
    fronts_final = non_dominated_sort(objs_final)
    pareto_front_indices_final = fronts_final[0] if fronts_final else []
    pareto_front_final = [population[i] for i in pareto_front_indices_final]
    
    # Combine archive and final front
    combined_candidates = pareto_front_archive + pareto_front_final
    
    if verbose:
        print(f"Archive size: {len(pareto_front_archive)}")
        print(f"Final front size: {len(pareto_front_final)}")
        print(f"Combined candidates: {len(combined_candidates)}")
        print(f"Validating top candidates with {finalize_valid} problems...")
    
    # Evaluate combined candidates with finalize_valid problems
    if len(combined_candidates) > 0:
        eval_workflows = [ind["workflow"] for ind in combined_candidates]
        result = eval_fn(eval_workflows, finalize_valid, server_url)
        
        # Update fitness with final validation results
        for i, ind in enumerate(combined_candidates):
            if args.fast:
                ind["fitness"] = result[i]
            else:
                # Use final validation results directly (not weighted average)
                ind["fitness"] = {
                    "pass_at_k": result[i].pass_at_1,
                    "token": result[i].total_tokens
                }
    
    # Re-sort combined candidates using NSGA-II
    objs_combined = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in combined_candidates])
    fronts_combined = non_dominated_sort(objs_combined)
    pareto_front_indices_combined = fronts_combined[0] if fronts_combined else []
    final_pareto_front = [combined_candidates[i] for i in pareto_front_indices_combined]
    
    if verbose:
        print(f"Final Pareto front size: {len(final_pareto_front)}")
        if len(final_pareto_front) > 0:
            print(f"Final Pareto front pass@k range: {min([ind['fitness']['pass_at_k'] for ind in final_pareto_front]):.4f} - {max([ind['fitness']['pass_at_k'] for ind in final_pareto_front]):.4f}")
            print(f"Final Pareto front token range: {min([ind['fitness']['token'] for ind in final_pareto_front]):.0f} - {max([ind['fitness']['token'] for ind in final_pareto_front]):.0f}")
    
    # Save final Pareto front
    save_checkpoint_csv(final_pareto_front, f"{key}_final")
    plot_pareto(final_pareto_front, f"{key}_final", survivors=final_pareto_front, buffer_list=[], pareto_front=final_pareto_front)
    
    if verbose:
        print(f"Final Pareto front saved: {key}_final")
        print(f"{'='*60}\n")

    return final_pareto_front


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
    

    parser = argparse.ArgumentParser(description="Run GA with Buffer Strategy to evolve multi-agent workflows")
    
    # Evaluation information
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name (MBPP, MATH)")
    parser.add_argument("--server-url", type=str, default=EVAL_SERVER_URL, help="Evaluation server URL")
    parser.add_argument("--no-extractor", action="store_true", help="Disable answer extractor")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--batch", action="store_true", help="Perform batch fitness evaluation")
    parser.add_argument("--fast", action="store_true", help="Random number fast evaluation")
    parser.add_argument("--alter-question", action="store_true", help="Alter question mode")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for evaluation")


    # GA information
    parser.add_argument("--population-size", type=int, default=100, help="The size of population")
    parser.add_argument("--generation", type=int, default=100, help="The number of generations")
    parser.add_argument("--eliticism-rate", type=float, default=0.5, help="The rate of eliticism (survivors = population_size * eliticism_rate)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="The rate of mutation")
    parser.add_argument("--crossover-rate", type=float, default=1.0, help="The rate of crossover")
    parser.add_argument("--max-workflow", type=int, default=5, help="The length of max workflow")
    parser.add_argument("--num_problem", type=int, default=30, help="The number of problems")
    parser.add_argument("--num_phase", type=int, default=1, help="The number of problem phases.")
    
    # Buffer strategy parameters
    parser.add_argument("--buffer-size", type=int, default=10, help="The size of buffer (probation) pool")
    parser.add_argument("--max-eval-iter", type=int, default=4, help="Maximum evaluation iterations per individual")
    
    # Finalization parameters
    parser.add_argument("--finalize-valid", type=int, default=100, help="Number of problems for final validation of Pareto front")

    # File name information
    parser.add_argument("--key", type=str, default="ga", help="The key to distinguish the experiment")
    
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
        key=args.key,
        finalize_valid=args.finalize_valid
    )

