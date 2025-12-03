import random
import numpy as np
import os
import matplotlib.pyplot as plt
from src.config import ROLE_DESCRIPTIONS
from src.agents.agent import Agent
from src.agents.workflow import Workflow

# Role name
ROLE_LIST = [role["name"] for role in ROLE_DESCRIPTIONS]

# From the agent list, select one random agent.
def random_agent() -> Agent:
    return random.choice(ROLE_LIST)

# Penalty value means the wieght of token term.
# At first, it is set to a random number.
# NotImplemented
def evaluate_objectives(workflow):

    pass_at_k = random.randint(1, 100)
    obj1 = -pass_at_k

    token_term = workflow.total_tokens or random.uniform(3000, 5000)
    obj2 = token_term

    return np.array([obj1, obj2])


# dominate function
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# Non-dominated sort.
def non_dominated_sort(pop_objs):
    

    N = len(pop_objs)
    S = [[] for _ in range(N)]
    domination_count = np.zeros(N)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while True:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if len(next_front) == 0:
            break
        fronts.append(next_front)
        i += 1

    return fronts

# Crowding distance
def crowding_distance(pop_objs, front):
    if len(front) == 0:
        return []

    M = pop_objs.shape[1]
    dist = np.zeros(len(front))
    f_objs = pop_objs[front]

    for m in range(M):
        idx = np.argsort(f_objs[:, m])
        dist[idx[0]] = dist[idx[-1]] = np.inf
        m_min = f_objs[idx[0], m]
        m_max = f_objs[idx[-1], m]
        if m_max - m_min == 0:
            continue
        for i in range(1, len(front) - 1):
            dist[idx[i]] += (f_objs[idx[i+1], m] - f_objs[idx[i-1], m]) / (m_max - m_min)

    return dist

# NSGA_select
def ngsa_select(population, num_select):
    
    workflows = [entry["workflow"] for entry in population]
    objs = np.array([entry["fitness"] for entry in population])

    fronts = non_dominated_sort(objs)

    selected_indices = []

    for front in fronts:
        if len(selected_indices) + len(front) > num_select:
            dist = crowding_distance(objs, front)
            idx_sorted = np.argsort(-dist)
            for idx in idx_sorted[: num_select - len(selected_indices)]:
                selected_indices.append(front[idx])
            break
        else:
            selected_indices.extend(front)

    return [population[i] for i in selected_indices]


def plot_pareto(pop_objs, file_name="front"):

    save_dir = "graph"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(pop_objs[:, 0], pop_objs[:, 1], c='red', s=40)
    plt.xlabel("Objective 1 (pass@k, negated)")
    plt.ylabel("Objective 2 (token penalty)")
    plt.title("Pareto Front")
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"[Saved] Pareto front saved to: {save_path}")

    plt.show()
