# src/ga/ga.py
import random
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Optional
from src.agents.agent import Agent
from src.agents.workflow import Workflow
from src.agents.extractors import get_extractor_for_task
from src.config import ROLE_DESCRIPTIONS
from src.datasets import MBPPDataset, MathAlgebraDataset, BaseDataset
from src.evaluation.pass_at_k import quick_evaluate

# =============== CONFIGURATION ===============
POPULATION_SIZE = 10
GENERATIONS = 100
ELITISM_RATE = 0.5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1.0
MAX_WORKFLOW_LENGTH = 5

# Evaluation configuration
NUM_EVAL_PROBLEMS = 5  # Number of problems to evaluate per fitness calculation
EVAL_SEED = None  # Set to an integer for reproducible sampling
TOKEN_PENALTY = 0.0001  # Penalty coefficient for token usage

# Role name
ROLE_LIST = [role["name"] for role in ROLE_DESCRIPTIONS]

# Task Name (options: "MBPP", "HumanEval", "MATH")
TASK_NAME = "MBPP"

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
        # HumanEval would be similar to MBPP
        # For now, fall back to MBPP
        print(f"Warning: HumanEval not yet implemented, using MBPP")
        _dataset_cache = MBPPDataset(split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Load the dataset
    _dataset_cache.load()
    print(f"Loaded {len(_dataset_cache)} problems from {task_name}")
    
    return _dataset_cache


def evaluate_fitness(workflow: Workflow, dataset: Optional[BaseDataset] = None) -> float:
    """
    Evaluate the fitness of a workflow using pass@k on the benchmark dataset.
    
    Fitness = pass@1 - (TOKEN_PENALTY * total_tokens)
    
    Higher is better. We want high pass rate and low token usage.
    
    Args:
        workflow: The workflow to evaluate
        dataset: Optional dataset to use (will use cached if not provided)
        
    Returns:
        Fitness score (0.0 to 1.0, potentially negative with token penalty)
    """
    if dataset is None:
        dataset = get_dataset(workflow.task_name)
    
    # Calculate pass@1 score
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
    
    # Apply token penalty
    token_term = workflow.total_tokens or 0
    fitness = pass_at_1 - (TOKEN_PENALTY * token_term)
    
    return fitness


def evaluate_fitness_fast(workflow: Workflow) -> float:
    """
    Fast fitness evaluation for development/debugging.
    Uses random values instead of actual LLM calls.
    """
    # Simulate pass@k with random value
    pass_at_k = random.random()
    token_term = workflow.total_tokens or 0

    return pass_at_k - (TOKEN_PENALTY * token_term)


# From the agent list, select one random agent role.
def random_agent() -> str:
    return random.choice(ROLE_LIST)


def initialize_population(task_name: str, use_extractor: bool = True):
    """
    Initialize the population of workflows.
    
    Args:
        task_name: Name of the task/benchmark
        use_extractor: Whether to add an answer extractor to workflows
        
    Returns:
        List of {"workflow": Workflow, "fitness": float} dictionaries
    """
    population = []
    dataset = get_dataset(task_name)

    # Get extractor for this task
    extractor = get_extractor_for_task(task_name) if use_extractor else None

    for i in range(POPULATION_SIZE):
        length = random.randint(1, MAX_WORKFLOW_LENGTH)

        # Generate a workflow list and description string.
        workflow_list = [random_agent() for _ in range(length)]
        workflow_description = " -> ".join(workflow_list)
        print(f"  Workflow {i+1}: {workflow_description}")
        
        # Generate a list of Agents.
        agents = []
        for role in workflow_list:
            agents.append(Agent(role=role, workflow_description=workflow_description))
        
        # Generate a workflow using agents list, with extractor
        workflow = Workflow(
            task_name=task_name, 
            agents=agents,
            extractor=extractor.copy() if extractor else None
        )
        
        # Evaluate fitness (using fast version during init to save API calls)
        # Switch to evaluate_fitness for actual evaluation
        fitness = evaluate_fitness_fast(workflow)
        population.append({"workflow": workflow, "fitness": fitness})

    return population


# Addition operator.
# The agent is added at the random place of the workflow.
def addition(workflow: Workflow) -> Workflow:
    
    new_workflow = workflow.copy()
    agent_list = workflow.agents

    if len(agent_list) < MAX_WORKFLOW_LENGTH:
        
        # Choose an arbitrary agent role and insert position.
        new_agent_role = random_agent()
        idx = random.randint(0, len(new_workflow.agents))

        # Insert the agent.
        new_workflow.insert_agent(new_agent_role, idx)

    return new_workflow


def deletion(workflow: Workflow) -> Workflow:

    new_workflow = workflow.copy()
    
    if len(new_workflow.agents) > 1:
        
        # Choose a removing position.
        idx = random.randint(0, len(new_workflow.agents)-1)
        
        # Remove the agent.
        new_workflow.remove_agent(idx)

    return new_workflow


def crossover(parent1: Workflow, parent2: Workflow) -> Workflow:

    w1 = parent1.copy().agents
    w2 = parent2.copy().agents

    if len(w1) == 0 or len(w2) == 0:
        return parent1.copy()

    cut1 = random.randint(0, len(w1) - 1)
    cut2 = random.randint(0, len(w2) - 1)

    new_agents = w1[:cut1] + w2[cut2:]

    # enforce max size rule
    new_agents = new_agents[:MAX_WORKFLOW_LENGTH]

    # Preserve extractor from parent1
    child = Workflow(
        task_name=parent1.task_name, 
        agents=new_agents,
        extractor=parent1.extractor.copy() if parent1.extractor else None
    )
    return child


def mutate(workflow: Workflow) -> Workflow:
    """Randomly apply mutation: addition or deletion."""
    if random.random() < 0.5:
        return addition(workflow)
    else:
        return deletion(workflow)


# (tournament) selection
# k: sampled numbers
def select(population, k=3):
    # population: list[{"workflow": Workflow, "fitness": float}]

    contenders = random.sample(population, min(len(population), k))
    winners = sorted(contenders, key=lambda entry: entry["fitness"], reverse=True)
    return winners[0]


# =============== GA LOOP ===============
# Note. Each entry in population has the form of {"workflow": Workflow, "fitness": float} tuple.
# =======================================
def run_ga(
    task_name: str, 
    use_real_evaluation: bool = False,
    use_extractor: bool = True,
    verbose: bool = True
):
    """
    Run the genetic algorithm to evolve multi-agent workflows.
    
    Args:
        task_name: Name of the task/benchmark (e.g., "MBPP", "MATH")
        use_real_evaluation: If True, use actual LLM calls for fitness
        use_extractor: If True, add answer extractor to workflows
        verbose: If True, print progress information
        
    Returns:
        Best workflow found
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm")
        print(f"Task: {task_name}")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Generations: {GENERATIONS}")
        print(f"Real evaluation: {use_real_evaluation}")
        print(f"Using extractor: {use_extractor}")
        print(f"{'='*60}\n")
    
    # Choose evaluation function
    eval_fn = evaluate_fitness if use_real_evaluation else evaluate_fitness_fast
    
    # Set an initial population
    if verbose:
        print("Initializing population...")
    population = initialize_population(task_name, use_extractor)

    for generation in range(GENERATIONS):
        
        if verbose:
            best_fitness = max(p["fitness"] for p in population)
            avg_fitness = sum(p["fitness"] for p in population) / len(population)
            print(f"Generation {generation:3d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

        new_population = []

        # Apply elitism.
        elite_count = max(1, int(POPULATION_SIZE * ELITISM_RATE))
        sorted_pop = sorted(
            population,
            key=lambda entry: entry["fitness"],
            reverse=True
        )
        elites = sorted_pop[:elite_count]
        for e in elites:
            new_population.append(e)

        # Generate a new population.
        while len(new_population) < POPULATION_SIZE:
            
            parent1 = select(population)
            parent2 = select(population)
            child = {"workflow": None, "fitness": -float("inf")}

            # Always do a crossover.
            child_workflow = crossover(parent1["workflow"], parent2["workflow"])

            # Mutation
            if random.random() <= MUTATION_RATE:
                child_workflow = mutate(child_workflow)
            
            # Initialize the child's memory.
            if hasattr(child_workflow, "memory"):
                child_workflow.memory = {}

            child["workflow"] = child_workflow

            # Evaluate the fitness.
            child["fitness"] = eval_fn(child_workflow)
            new_population.append(child)

        population = new_population

    # Pick best solution after evolution
    best = max(population, key=lambda entity: entity["fitness"])
    best_workflow = best["workflow"]
    best_fitness = best["fitness"]

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL BEST WORKFLOW")
        print(f"{'='*60}")
        print(f"{best_workflow}")
        if best_workflow.extractor:
            print(f"\nWith extractor: {best_workflow.extractor.role}")
        print(f"\nFitness: {best_fitness:.5f}")
        print(f"{'='*60}\n")
    
    return best


def evaluate_workflow(
    workflow: Workflow,
    task_name: str = None,
    num_problems: int = 50,
    verbose: bool = True
) -> dict:
    """
    Thoroughly evaluate a workflow on a dataset.
    
    Use this for final evaluation of evolved workflows.
    
    Args:
        workflow: The workflow to evaluate
        task_name: Task name (uses workflow's task_name if not provided)
        num_problems: Number of problems to evaluate
        verbose: Print results
        
    Returns:
        Evaluation results dictionary
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
    # Run with fast evaluation for testing
    # Set use_real_evaluation=True for actual LLM-based evaluation
    best = run_ga(
        task_name=TASK_NAME,
        use_real_evaluation=False,  # Set to True for real evaluation
        use_extractor=True,
        verbose=True
    )
    
    # Optionally evaluate the best workflow more thoroughly
    # evaluate_workflow(best["workflow"], num_problems=50)
