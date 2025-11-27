import random
import yaml
from copy import deepcopy
from pathlib import Path
from src.agents.agent import Agent
from src.agents.workflow import Workflow

# =============== CONFIGURATION ===============
POPULATION_SIZE = 10
GENERATIONS = 20
ELITISM_RATE = 0.3
MUTATION_RATE = 0.3
CROSSOVER_RATE = 1.0
MAX_WORKFLOW_LENGTH = 4   # since N < 5

CONFIG_BASE_AGENTS = Path("configs/base_agents.yaml")
CONFIG_TASK_DESCRIPTIONS = Path("configs/task_descriptions.yaml")
CONFIG_DEFAULT_PROMPTS = Path("configs/default_prompts.yaml")

# Open base agent configurations.
with open(CONFIG_BASE_AGENTS, "r") as f:
    agent_configs = yaml.safe_load(f)

# From the yaml file, generate the list of agents.
def create_agent_from_yaml(agent_name: str) -> Agent:
    if agent_name not in agent_configs:
        raise ValueError(f"{agent_name} not found in YAML config.")
    
    conf = agent_configs[agent_name]
    return Agent(
        role=conf["task"],
        prompt=conf["prompt"]
    )

AGENT_LIST = [create_agent_from_yaml(name) for name in agent_configs.keys()]

# Task Name
TASK_NAME = "HumanEval"

# Input Prompt
INPUT_PROMPT = "Hi, we are an EvoLite."

# Evaluate the fitness function.
def evaluate_fitness(workflow, task_name, input_prompt):
    
    total_score = 0
    output = None

    for agent, order in workflow.agents:

        output = agent.run(input_prompt)
        workflow.memory[(agent, order)] = output
        input_prompt = output

    raise NotImplementedError


# From the agent list, select one random agent.
def random_agent() -> Agent:
    return random.choice(AGENT_LIST)

# Initialize the population.
# In order to lessen the pass@k estimation,
# each entity is a tuple (workflow:Workflow, fitness function: Float)
def initialize_population(task_name: str):
    
    population = []

    for _ in range(POPULATION_SIZE):
        length = random.randint(1, MAX_WORKFLOW_LENGTH)
        agents = [random_agent() for _ in range(length)]
        workflow = Workflow(task_name=task_name, agents=agents)
        fitness = evaluate_fitness(workflow, TASK_NAME, INPUT_PROMPT)
        population.append((workflow, fitness))

    return population

# Addition operator.
# The agent is added at the random place of the workflow.
def addition(workflow: Workflow) -> Workflow:
    
    new_workflow = workflow.copy()
    
    if len(new_workflow.agents) < MAX_WORKFLOW_LENGTH:
        idx = random.randint(0, len(new_workflow.agents))
        new_workflow.agents.insert(idx, random_agent())
        new_workflow._build_graph()

    return new_workflow


def deletion(workflow: Workflow) -> Workflow:
    """Remove a random agent if >= 1 agent exists."""
    new_workflow = workflow.copy()
    
    if len(new_workflow.agents) > 1:
        idx = random.randint(0, len(new_workflow.agents) - 1)
        del new_workflow.agents[idx]
        new_workflow._build_graph()

    return new_workflow


def crossover(parent1: Workflow, parent2: Workflow) -> Workflow:
    """Single-point crossover: mix agents list."""

    w1 = deepcopy(parent1.agents)
    w2 = deepcopy(parent2.agents)

    if len(w1) == 0 or len(w2) == 0:
        return parent1.copy()

    cut1 = random.randint(0, len(w1) - 1)
    cut2 = random.randint(0, len(w2) - 1)

    new_agents = w1[:cut1] + w2[cut2:]

    # enforce max size rule
    new_agents = new_agents[:MAX_WORKFLOW_LENGTH]

    child = Workflow(task_name=parent1.task_name, agents=new_agents)
    return child


def mutate(workflow: Workflow) -> Workflow:
    """Randomly apply mutation: addition or deletion."""
    if random.random() < 0.5:
        return addition(workflow)
    else:
        return deletion(workflow)


# (tournamnet) selection
# k: sampled numbers
def select(population, k=3):
    # population: list[(workflow, fitness)]

    contenders = random.sample(population, k)
    winners = sorted(contenders, key=lambda entry: entry[1], reverse=True)
    return winners[0]


# =============== GA LOOP ===============
# Note. Each entity in the population list is form of (Workflow, fitness) tuple.
# =======================================
def run_ga(task_name: str):

    # Set an initial population 
    population = initialize_population(task_name)

    for generation in range(GENERATIONS):
        
        print(f"===== Generation {generation} =====")

        new_population = []

        # Apply an eliticism.
        elite_count = max(1, int(POPULATION_SIZE * ELITISM_RATE))
        sorted_pop = sorted(
            population,
            key=lambda entry: entry[1],  # entry = (workflow, fitness)
            reverse=True
        )

        elites = sorted_pop[:elite_count]

        for e in elites:
            new_population.append(e)


        while len(new_population) < POPULATION_SIZE:
            
            parent1_workflow = select(population, task_name)[0]
            parent2_workflow = select(population, task_name)[0]
            child_workflow = None

            if random.random() < CROSSOVER_RATE:
                child_workflow = crossover(parent1_workflow, parent2_workflow)

            if random.random() < MUTATION_RATE:
                child_workflow = mutate(child_workflow)

            if child_workflow == None:
                raise ValueError
                # child = parent1.copy
            
            # Initialize the child's memory.
            if hasattr(child_workflow, "memory"):
                child_workflow.memory = {}

            # Evaluate the fitness.
            child_fitness = evaluate_fitness(child_workflow, TASK_NAME, INPUT_PROMPT)
            new_population.append((child_workflow, child_fitness))

        population = new_population

    # pick best solution after evolution
    best = max(population, key=lambda wf: evaluate_fitness(wf, task_name))

    print("\n==== FINAL BEST WORKFLOW ====")
    print(best)
    return best


# Run the algorithm
if __name__ == "__main__":
    best_workflow = run_ga(TASK_NAME)
