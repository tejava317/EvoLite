import random
import numpy as np
import os
import matplotlib.pyplot as plt
from src.config import ROLE_DESCRIPTIONS
from src.agents.agent import Agent
from src.agents.workflow import Workflow

# Role name
ROLE_LIST = ROLE_DESCRIPTIONS

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
# The first element is token, and the second element is pass@k value.
# We want to minimize the token value while maximing the pass@k value.
def dominates(a, b): return np.all(a <= b) and np.any(a < b)
# def dominates(a, b): return a[1] < b[1]

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
    #  Note that the pass@k value is negated. 
    objs = np.array([[entry["fitness"]["token"], -1 * entry["fitness"]["pass_at_k"]] for entry in population])

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


def plot_pareto(population, file_name="front", log_x = False, log_y = False, save_dir = None, 
                survivors=None, buffer_list=None, pareto_front=None):

    if save_dir is None:
        save_dir = "src/ga/graph"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    
    # If survivors/buffer_list are provided (even if empty lists), use classification
    if survivors is not None or buffer_list is not None:
        # Convert to sets for fast lookup
        # Use _id if available (for shim objects from ga_llm.py), otherwise use id()
        def get_id(entry):
            return entry.get("_id", id(entry))
        
        survivor_set = set(get_id(s) for s in survivors) if survivors else set()
        buffer_set = set(get_id(b) for b in buffer_list) if buffer_list else set()
        pareto_front_set = set(get_id(p) for p in pareto_front) if pareto_front else set()
        
        # Separate population into groups
        pareto_front_points = []  # Actual Pareto front (first front) - will be connected with line
        survivor_points = []  # Other survivors (not in first front)
        buffer_points = []
        dead_points = []
        
        for entry in population:
            entry_id = get_id(entry)
            token = entry["fitness"]["token"]
            pass_at_k = entry["fitness"]["pass_at_k"]
            
            if entry_id in pareto_front_set:
                # Actual Pareto front (first front)
                pareto_front_points.append((token, pass_at_k))
            elif entry_id in survivor_set:
                # Survivors but not in first front
                survivor_points.append((token, pass_at_k))
            elif entry_id in buffer_set:
                buffer_points.append((token, pass_at_k))
            else:
                dead_points.append((token, pass_at_k))
        
        # Plot dead individuals first (gray, small) - so they appear in background
        if dead_points:
            dead_tokens, dead_pass = zip(*dead_points) if dead_points else ([], [])
            plt.scatter(dead_tokens, dead_pass, c='lightgray', s=20, alpha=0.5, label='Dead', marker='x')
        
        # Plot buffer individuals (orange)
        if buffer_points:
            buffer_tokens, buffer_pass = zip(*buffer_points) if buffer_points else ([], [])
            plt.scatter(buffer_tokens, buffer_pass, c='orange', s=50, alpha=0.7, label='Buffer', marker='^')
        
        # Plot other survivors (not in first front) - light blue
        if survivor_points:
            survivor_tokens, survivor_pass = zip(*survivor_points) if survivor_points else ([], [])
            plt.scatter(survivor_tokens, survivor_pass, c='lightblue', s=60, alpha=0.6, label='Survivors (other fronts)', marker='s')
        
        # Plot actual Pareto front (first front) - points first, then line
        if pareto_front_points:
            front_tokens, front_pass = zip(*pareto_front_points) if pareto_front_points else ([], [])
            # Sort by token for line connection
            sorted_indices = np.argsort(front_tokens)
            sorted_tokens = np.array(front_tokens)[sorted_indices]
            sorted_pass = np.array(front_pass)[sorted_indices]
            
            # Plot front points (blue circles)
            plt.scatter(front_tokens, front_pass, c='blue', s=80, alpha=0.8, label='Pareto Front', marker='o', edgecolors='darkblue', linewidths=1.5, zorder=3)
            # Plot line connecting ONLY Pareto front points
            if len(sorted_tokens) > 1:
                plt.plot(sorted_tokens, sorted_pass, 'b-', linewidth=2, alpha=0.6, zorder=2)
    else:
        # If no survivors/buffer provided, plot all as before
        pass_at_k_array = np.array([entry["fitness"]["pass_at_k"] for entry in population])
        token_array = np.array([entry["fitness"]["token"] for entry in population])
        plt.scatter(token_array, pass_at_k_array, c='red', s=40, label='All')
    
    plt.xlabel("Objective 1 (total_tokens)")
    plt.ylabel("Objective 2 (pass@k)")
    plt.title("Pareto Front")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Apply log scale
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    save_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"[Saved] Pareto front saved to: {save_path}")

    plt.close()