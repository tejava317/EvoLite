import random
import yaml
from copy import deepcopy
from pathlib import Path
from src.agents.agent import Agent
from src.agents.workflow import Workflow
from src.config import ROLE_DESCRIPTIONS

# =============== CONFIGURATION ===============
POPULATION_SIZE = 10
GENERATIONS = 100
ELITISM_RATE = 0.5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1.0
MAX_WORKFLOW_LENGTH = 5

# Role name
ROLE_LIST = [role["name"] for role in ROLE_DESCRIPTIONS]

# Task Name
TASK_NAME = "HumanEval"

# =====================================================

# Evaluate the fitness function.
# Single objective function considering token penalty.
def evaluate_fitness(workflow):
    
    total_score = 0
    output = None

    pass_at_k = random.random()
    token_term = workflow.total_tokens or 0
    penalty = 0.01

    return pass_at_k + (penalty * token_term)


# From the agent list, select one random agent.
def random_agent() -> Agent:
    return random.choice(ROLE_LIST)

# Initialize the population.
# In order to lessen the pass@k estimation,
# each entity has a dictionary form. (["workflow": Workflow, "fitness": Int])
def initialize_population(task_name: str):
    
    population = []

    for _ in range(POPULATION_SIZE):
        length = random.randint(1, MAX_WORKFLOW_LENGTH)

        # Generate a workflow list and description string.
        workflow_list = [random_agent() for _ in range(length)]
        workflow_description = " -> ".join(workflow_list)
        print(workflow_description)
        
        # Generate a list of Agents.
        agents = []
        for role in workflow_list:
            agents.append(Agent(role=role, workflow_description=workflow_description))
        
        # Generate a workflow using agents list.
        workflow = Workflow(task_name=task_name, agents=agents)
        
        # Evaluate a fitness function.
        fitness = evaluate_fitness(workflow)
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
        
        # Insert the agent.
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

    contenders = random.sample(population, min(len(population), k))
    winners = sorted(contenders, key=lambda entry: entry["fitness"], reverse=True)
    return winners[0]


# =============== GA LOOP ===============
# Note. Each entry in population has the form of {"workflow": Workflow, "fitness": Int} tuple.
# =======================================
def run_ga(task_name: str):

    # Set an initial population and sort at first.
    population = initialize_population(task_name)

    for generation in range(GENERATIONS):
        
        print(f"===== Generation {generation} =====")

        new_population = []

        # Apply an eliticism.
        elite_count = max(1, int(POPULATION_SIZE * ELITISM_RATE))
        sorted_pop = sorted(
            population,
            key=lambda entry: entry["fitness"],
            reverse=True
        )
        elites = sorted_pop[:elite_count]
        for e in elites:
            new_population.append(e)

        # Generate a new popultion.
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
            child["fitness"] = evaluate_fitness(child_workflow)
            new_population.append(child)

        population = new_population

    # pick best solution after evolution
    best = max(population, key=lambda entity: entity["fitness"])
    best_workflow = best["workflow"]
    best_fitness = best["fitness"]

    print("\n==== FINAL BEST WORKFLOW ====")
    print(f"workflow: {best_workflow}")
    print(f"fitness: {best_fitness:.5f}")
    return best


# Run the algorithm
if __name__ == "__main__":
    best_workflow = run_ga(TASK_NAME)
