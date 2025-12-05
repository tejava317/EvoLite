import random
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
from src.agents.agent import Agent
from src.agents.block import *
from src.agents.workflow_block import BlockWorkflow
from src.config import ROLE_DESCRIPTIONS
from src.ga.multi_objective import *

# =============== CONFIGURATION ===============
POPULATION_SIZE = 30
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


# From the agent list, select one random agent.
def random_agent() -> str:
    return random.choice(ROLE_LIST)

# Initialize the population.
# In order to lessen the pass@k estimation,
# each entity has a dictionary form. (["workflow": Workflow, "fitness": Int])
def initialize_population(task_name: str):

    population = []
    
    length = random.randint(1, MAX_WORKFLOW_LENGTH)
    
    for _ in range(POPULATION_SIZE):

        # Agent version.
        # Generate a workflow list and description string.
        # workflow_list = [random_agent() for _ in range(length)]
        # workflow_description = " -> ".join(workflow_list)
        # print(workflow_description)
        
        # Generate a list of Agents.
        # agents = []
        # for role in workflow_list:
        #     agents.append(Agent(role=role, workflow_description=workflow_description))
        
        # Block version.
        blocks = []
        curr_length = 0
        while(curr_length < length):

            if random.random() < 0.8:
                role = random_agent()
                block = AgentBlock(role)
                blocks.append(block)
                curr_length += block.num_agents
            else:
                block = CompositeBlock()
                block.expand("")
                blocks.append(block)

        # print for debugging
        block_strs = [f"[{str(block)}]" for block in blocks]
        chain = " -> ".join(block_strs)

        print(f"{chain}")

        # Generate a workflow using agents list.
        workflow = BlockWorkflow(task_name=task_name, blocks=blocks)
        
        # Evaluate a fitness function.
        fitness = evaluate_objectives(workflow)
        population.append({"workflow": workflow, "fitness": fitness})

    return population

# Addition operator.
# The agent is added at the random place of the workflow.
def addition(workflow: BlockWorkflow) -> BlockWorkflow:
    
    new_workflow = workflow.copy()
    agent_list = workflow.blocks

    if len(agent_list) < MAX_WORKFLOW_LENGTH:
        
        # Choose an arbitrary agent role and insert position.
        new_agent_role = random_agent()
        idx = random.randint(0, len(new_workflow.blocks))

        # Insert the agent.
        if random.random() < 0.7:
            new_workflow.insert_block(AgentBlock(new_agent_role), idx)
        else:
            new_workflow.insert_block(CompositeBlock(), idx)

    return new_workflow


def deletion(workflow: BlockWorkflow) -> BlockWorkflow:

    new_workflow = workflow.copy()
    
    if len(new_workflow.blocks) > 1:
        
        # Choose a removing position.
        idx = random.randint(0, len(new_workflow.blocks)-1)
        
        # Insert the agent.
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

    # enforce max size rule
    new_blocks = new_blocks[:MAX_WORKFLOW_LENGTH]

    child = BlockWorkflow(task_name=parent1.task_name, blocks=new_blocks)
    return child


def mutate(workflow: BlockWorkflow) -> BlockWorkflow:
    if random.random() < 0.5:
        return addition(workflow)
    else:
        return deletion(workflow)


# k: select k members
def select(population, k=3):
    return random.sample(population, min(len(population), k))


# =============== GA LOOP ===============
# Note. Each entry in population has the form of {"workflow": Workflow, "fitness": Int} tuple.
# =======================================
def run_ga(task_name: str):

    # Set an initial population and sort at first.
    population = initialize_population(task_name)

    for generation in range(GENERATIONS):
        
        print(f"===== Generation {generation} =====")

        new_population = population

        # Generate a new popultion.
        while len(new_population) < POPULATION_SIZE:
            
            parent1 = select(new_population, 1)[0]
            parent2 = select(new_population, 1)[0]
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
            child["fitness"] = evaluate_objectives(child_workflow)
            new_population.append(child)

        # Multi-objective selection.
        population = ngsa_select(new_population, int(len(new_population) * ELITISM_RATE))

        # Plot the pareto front.
        if (generation + 1) % 20 == 0:
            pop_objs = np.array([entity["fitness"] for entity in population])
            plot_pareto(pop_objs, f"generation_{generation+1}")

    # pick best solution after evolution
    best = select(population, 1)[0]

    best_workflow = best["workflow"]
    best_fitness = best["fitness"]

    print("\n==== FINAL BEST WORKFLOW ====")
    print(f"workflow: {best_workflow}")
    print(f"fitness: {best_fitness}")
    return best
    # return 0

# Run the algorithm
if __name__ == "__main__":
    best_workflow = run_ga(TASK_NAME)
