import asyncio
import functools
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from src.agents.block import AgentBlock
from src.agents.workflow_block import BlockWorkflow
from src.config import ROLE_DESCRIPTIONS
from src.datasets import MBPPDataset, BaseDataset
from src.evaluation.pass_at_k import quick_evaluate
from src.ga.multi_objective import crowding_distance, non_dominated_sort
from src.ga.checkpoint import save_checkpoint_csv
from src.llm.client import PromptGenerator

# Flush all prints for long running loops
print = functools.partial(print, flush=True)

# Run-scoped paths
RUN_LOG_DIR = "logs/ga_llm_runs"
CHECKPOINT_DIR = "src/ga/ga_llm_checkpoints"
GRAPH_DIR = "src/ga/ga_llm_graph"

# Run-scoped logger (prints + append-to-file)
run_log_path: Optional[str] = None
last_llm_call_count: int = 0


def log_line(msg: str):
    print(msg)
    if run_log_path:
        os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# =======================
# High-level GA settings
# =======================
POPULATION_SIZE = 100
INIT_SEEDED_POPULATION = 50  # Stratified seeding target
MAX_GENERATIONS = 10
MUTATION_RATE = 0.7  # Increased to promote exploration
CROSSOVER_RATE = 0.3
AGNOSTIC_RATIO = 0.2  # Inside mutation branch

NUM_EVAL_PROBLEMS = 30  # Fixed MBPP subset size
EVAL_SEED = 1337

MIN_COST_LIMIT = 1200
MAX_COST_LIMIT = 12000
SCORE_IMPROVE_THRESHOLD = 0.01
PATIENCE_LIMIT = 3

LLM_CALL_BUDGET = 500  # Hard stop to avoid runaway LLM usage
LLM_MAX_TOKENS = 480

ROLE_LIST = ROLE_DESCRIPTIONS
TASK_NAME = "MBPP"
USE_LLM = True  # toggled via CLI
RUN_LOG_DIR = "logs/ga_llm_runs"
CHECKPOINT_DIR = "src/ga/ga_llm_checkpoints"
GRAPH_DIR = "src/ga/ga_llm_graph"

# =======================
# Prompt templates
# =======================
GLOBAL_SYSTEM_PROMPT = """You are an expert AI Workflow Architect.
Your goal is to optimize the workflow structure (Agent connections) to balance Performance and Cost.

# Output Format: Arrow Syntax
- Format: Source_Agent_ID -> Target_Agent_ID
- Linear: AgentA -> AgentB -> AgentC
- Branching: AgentA -> AgentB, AgentA -> AgentC (use separate lines or comma)
- Loop: AgentA -> AgentB -> AgentA
- Parallel: [AgentA, AgentB] -> AgentC (brackets optional), or list multiple connections

# Constraints
1. DO NOT write Python code or class definitions.
2. Use ONLY the Agent IDs provided in the Agent Pool.
3. Ensure the graph represents a valid flow (start to end).
4. Return ONLY the Arrow Syntax string inside a code block."""

SEED_TEMPLATE_LINEAR = """# Task
Design a Linear Chain workflow using the provided Agent Pool.

# Input
- Agent Pool: {agent_pool_description}

# Instruction
1. Select 2-3 agents from the pool.
2. Connect them in a single line using arrow syntax.
3. Example Output: Planner_01 -> Solver_05 -> Formatter_02

# Output
(Provide only the arrow syntax string)"""

SEED_TEMPLATE_REFLEXION = """# Task
Design a Reflexion (Self-Correction) workflow.

# Input
- Agent Pool: {agent_pool_description}

# Instruction
1. Select an Actor and a Critic.
2. Create a loop where the Critic feeds back to the Actor.
3. Example Output: Solver_01 -> Reviewer_03 -> Solver_01

# Output
(Provide only the arrow syntax string)"""

SEED_TEMPLATE_BRANCHING = """# Task
Design a Branching (Plan-and-Solve) workflow.

# Input
- Agent Pool: {agent_pool_description}

# Instruction
1. Select a Planner, 2 Workers, and a Solver.
2. Connect Planner to both Workers, and both Workers to Solver.
3. Example Output:
Planner_01 -> Coder_A
Planner_01 -> Coder_B
Coder_A -> Summarizer_01
Coder_B -> Summarizer_01

# Output
(Provide only the arrow syntax string)"""

SEED_TEMPLATE_TEST = """# Task
Design a Test-Driven workflow.

# Input
- Agent Pool: {agent_pool_description}

# Instruction
1. Select a Test Generator, a Code Generator, and a Verifier.
2. Connect in order with arrow syntax.
3. Example Output: TestGen_01 -> CodeGen_03 -> Verify_02

# Output
(Provide only the arrow syntax string)"""

MUTATION_PROMPT_EXPAND = """# Context
We need to ADD a node to reinforce the logic of the current workflow.

# Input
- Current Topology:
{current_workflow_string}
- Agent Pool: {agent_pool_description}

# Strategy: "Reinforce the Weakest Link"
Choose ONE strategy and rewrite the topology string:
1. Head Injection: Add a Planner/Clarifier at the start. (NewAgent -> OldStart -> ...)
2. Tail Injection: Add a Reviewer/Formatter at the end. (... -> OldEnd -> NewAgent)
3. Parallel Ensemble: Add a similar agent in parallel to a key node.

# Output
(Provide only the new arrow syntax string)"""

MUTATION_PROMPT_COMPRESS = """# Context
The workflow is too expensive. We need to DELETE a redundant node.

# Input
- Current Topology:
{current_workflow_string}

# Strategy: "Prune and Repair"
1. Identify a redundant agent ID in the string.
2. Remove it.
3. Repair the circuit: connect the deleted agent's predecessor directly to its successor.

# Output
(Provide only the new arrow syntax string)"""

CROSSOVER_PROMPT_EXPAND = """# Context
Improve Parent A (Base) by transplanting a module from Parent B (Reference).

# Input
- Parent A Topology: {topology_string_a}
- Parent B Topology: {topology_string_b}

# Strategy: "Transplant or Mix"
Option 1: [Distillation] Identify a strong agent or sub-sequence in Parent B and insert into Parent A.
Option 2: [Hybrid Mixing] If transplant is hard, combine sequentially or in parallel.

# Instruction
- Generate the Child Topology String.
- Use "# STRATEGY: DISTILLATION" or "# STRATEGY: MIXING" as the first line."""

CROSSOVER_PROMPT_COMPRESS = """# Context
Reduce the cost of Parent A (Base) by using the efficient structure of Parent B (Reference).

# Input
- Parent A Topology: {topology_string_a}
- Parent B Topology: {topology_string_b}

# Strategy: "Substitute or Simplify"
Option 1: [Destructive Distillation] Replace a complex section in A with a simple agent from B.
Option 2: [Efficient Mixing] Use Parent B's simple topology but swap in Parent A's key agents.

# Instruction
- Generate the Child Topology String.
- The result must be simpler/shorter than Parent A.
- Use "# STRATEGY: DISTILLATION" or "# STRATEGY: MIXING" as the first line."""


# =======================
# Data structures
# =======================
@dataclass
class Individual:
    code: str
    lineage: str = "init"
    graph: Dict[str, List[str]] = field(default_factory=dict)
    score: float = 0.0
    cost: float = 0.0
    rank: int = 0
    distance: float = 0.0
    direction: str = "EXPAND"
    patience: int = 0
    prev_score: float = 0.0
    tokens_from_llm: int = 0

    def __post_init__(self):
        self.graph = parse_arrow_syntax(self.code)

    def update_fitness(self, score: float, cost: float):
        self.prev_score = self.score
        self.score = score
        self.cost = cost + self.tokens_from_llm  # penalize heavy LLM edits
        return self


# =======================
# Utility helpers
# =======================
def parse_arrow_syntax(topology: str) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    if not topology:
        return graph

    lines = [l.strip() for l in topology.splitlines() if l.strip() and not l.strip().startswith("#")]
    for line in lines:
        cleaned = line.replace("[", "").replace("]", "")
        parts = [p.strip() for p in cleaned.split("->") if p.strip()]
        if len(parts) < 2:
            continue
        for src, tgt in zip(parts, parts[1:]):
            graph.setdefault(src, []).append(tgt)
    return graph


def extract_code_block(text: str) -> str:
    match = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def canonicalize_topology(topology: str) -> str:
    lines = []
    for raw in topology.splitlines():
        line = raw.strip().strip(",")
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s*->\s*", " -> ", line)
        line = re.sub(r"\s{2,}", " ", line)
        if "->" not in line:
            continue
        lines.append(line)
    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            uniq.append(l)
            seen.add(l)
    return "\n".join(uniq)


def filter_agents_to_pool(topology: str) -> str:
    if not topology:
        return ""
    allowed = set(ROLE_LIST)
    cleaned_lines = []
    for line in topology.splitlines():
        nodes = [n.strip() for n in re.split(r"->|,|\[|\]", line) if n.strip()]
        valid_nodes = [n for n in nodes if n in allowed]
        if len(valid_nodes) < 2:
            continue
        rebuilt = " -> ".join(valid_nodes)
        cleaned_lines.append(rebuilt)
    return canonicalize_topology("\n".join(cleaned_lines))


def topology_to_linear_roles(topology: str) -> List[str]:
    tokens = []
    for line in topology.splitlines():
        line = line.replace("[", "").replace("]", "")
        parts = [p.strip() for p in re.split(r"->|,", line) if p.strip()]
        for p in parts:
            if p not in tokens:
                tokens.append(p)
    return [t for t in tokens if t in ROLE_LIST]


def topology_to_block_workflow(topology: str, task_name: str) -> BlockWorkflow:
    roles = topology_to_linear_roles(topology)
    blocks = [AgentBlock(role) for role in roles] if roles else [AgentBlock(random.choice(ROLE_LIST))]
    return BlockWorkflow(task_name=task_name, blocks=blocks)


def random_chain(min_len: int = 2, max_len: int = 4) -> str:
    length = random.randint(min_len, max_len)
    roles = random.sample(ROLE_LIST, min(length, len(ROLE_LIST)))
    return " -> ".join(roles)


def random_loop() -> str:
    roles = random.sample(ROLE_LIST, 2)
    return f"{roles[0]} -> {roles[1]} -> {roles[0]}"


def random_branching() -> str:
    planner = random.choice(ROLE_LIST)
    workers = random.sample([r for r in ROLE_LIST if r != planner], 2)
    solver = random.choice([r for r in ROLE_LIST if r not in workers + [planner]])
    return "\n".join([
        f"{planner} -> {workers[0]}",
        f"{planner} -> {workers[1]}",
        f"{workers[0]} -> {solver}",
        f"{workers[1]} -> {solver}",
    ])


def random_test_driven() -> str:
    roles = random.sample(ROLE_LIST, 3)
    return " -> ".join(roles)


# =======================
# LLM Controller
# =======================
class LLMController:
    def __init__(self, model: Optional[str], temperature: float = 0.35):
        # Use vLLM client for faster batch processing
        from src.llm.vllm_client import VLLMClient
        import asyncio
        self.vllm_client = VLLMClient(model=model, default_temperature=temperature)
        self.call_count = 0
        self._loop = None
        self._pending_batch = []  # Queue for batch requests
        self._batch_lock = asyncio.Lock()

    def _increment_and_check(self) -> bool:
        self.call_count += 1
        return self.call_count <= LLM_CALL_BUDGET

    def _ensure_loop(self):
        """Ensure event loop exists (create if needed for sync context)."""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def generate_topology(self, user_prompt: str) -> Tuple[Optional[str], int]:
        """Generate topology using vLLM client (single call, can be batched later)."""
        if not self._increment_and_check():
            return None, 0
        
        self._ensure_loop()
        
        async def _generate():
            result = await self.vllm_client.generate(
                system_prompt=GLOBAL_SYSTEM_PROMPT,
                user_content=user_prompt,
                temperature=self.vllm_client.default_temperature,
                max_tokens=LLM_MAX_TOKENS,
            )
            if result.status == "COMPLETED":
                content = extract_code_block(result.content)
                return content, result.total_tokens
            else:
                log_line(f"[LLMController] generate failed: {result.error}")
                return None, 0
        
        try:
            content, tokens = self._loop.run_until_complete(_generate())
            return content, tokens
        except Exception as e:
            log_line(f"[LLMController] ERROR: {e}")
            return None, 0

    async def generate_topology_batch(self, user_prompts: List[str]) -> List[Tuple[Optional[str], int]]:
        """Generate multiple topologies in a single batch request (faster)."""
        if not user_prompts:
            return []
        
        messages_list = [
            [
                {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            for prompt in user_prompts
        ]
        
        results = await self.vllm_client.batch_complete(
            messages_list,
            temperature=self.vllm_client.default_temperature,
            max_tokens=LLM_MAX_TOKENS,
        )
        
        outputs = []
        for result in results:
            if result.status == "COMPLETED":
                content = extract_code_block(result.content)
                outputs.append((content, result.total_tokens))
            else:
                log_line(f"[LLMController] batch item failed: {result.error}")
                outputs.append((None, 0))
        
        self.call_count += len(user_prompts)
        return outputs
    
    async def close(self):
        """Close vLLM client connection."""
        await self.vllm_client.close()


# =======================
# Dataset cache
# =======================
_dataset_cache: Optional[BaseDataset] = None


def get_dataset() -> BaseDataset:
    global _dataset_cache
    if _dataset_cache is None:
        ds = MBPPDataset(split="test")
        ds.load()
        _dataset_cache = ds
        print(f"[Dataset] Loaded {len(ds)} MBPP test problems.")
    return _dataset_cache


# =======================
# GA core helpers
# =======================
def stratified_seed_population(llm: Optional[LLMController]) -> List[Individual]:
    seeds: List[Individual] = []

    # Pure random seeding if LLM disabled/unavailable
    if not USE_LLM or llm is None:
        generators = [
            (random_chain, "seed_chain"),
            (random_loop, "seed_loop"),
            (random_branching, "seed_branch"),
            (random_test_driven, "seed_test"),
        ]
        while len(seeds) < POPULATION_SIZE:
            fn, name = random.choice(generators)
            seeds.append(Individual(code=canonicalize_topology(fn()), lineage=name))
        return seeds

    agent_pool_desc = ", ".join(ROLE_LIST[:50])
    counts = {
        "linear": int(0.30 * INIT_SEEDED_POPULATION),
        "reflex": int(0.30 * INIT_SEEDED_POPULATION),
        "branch": int(0.20 * INIT_SEEDED_POPULATION),
        "test": INIT_SEEDED_POPULATION - int(0.30 * INIT_SEEDED_POPULATION) * 2 - int(0.20 * INIT_SEEDED_POPULATION),
    }

    templates = [
        ("linear", SEED_TEMPLATE_LINEAR, random_chain),
        ("reflex", SEED_TEMPLATE_REFLEXION, random_loop),
        ("branch", SEED_TEMPLATE_BRANCHING, random_branching),
        ("test", SEED_TEMPLATE_TEST, random_test_driven),
    ]

    log_line(f"[Seed] Starting LLM-based seeding with BATCH requests. Target counts: {counts}")
    
    # Collect all prompts first, then batch process
    for key, template, fallback_fn in templates:
        target = counts[key]
        log_line(f"[Seed] Preparing {target} {key} seeds for batch request...")
        
        # Prepare batch of prompts
        batch_prompts = []
        for i in range(target):
            prompt = template.format(agent_pool_description=agent_pool_desc)
            batch_prompts.append(prompt)
        
        # Execute batch request
        log_line(f"[Seed] Sending batch request for {len(batch_prompts)} {key} prompts...")
        try:
            import asyncio
            if llm._loop is None:
                llm._ensure_loop()
            batch_results = llm._loop.run_until_complete(
                llm.generate_topology_batch(batch_prompts)
            )
            log_line(f"[Seed] Batch response received: {len(batch_results)} results")
            
            # Process batch results
            for i, (topo, tokens) in enumerate(batch_results):
                if topo is None:
                    log_line(f"[Seed] {key} batch[{i}]: LLM returned None, using fallback")
                    topo = fallback_fn()
                    tokens = 0
                topo = filter_agents_to_pool(canonicalize_topology(topo))
                if not topo:
                    log_line(f"[Seed] {key} batch[{i}]: filtered topo empty, using fallback")
                    topo = fallback_fn()
                indiv = Individual(code=topo, lineage=key)
                indiv.tokens_from_llm += tokens
                seeds.append(indiv)
                log_line(f"[Seed] {key} batch[{i}]: got topo='{topo[:50]}' tokens={tokens}")
        except Exception as e:
            log_line(f"[Seed] ERROR in batch for {key}: {e}, falling back to sequential")
            # Fallback to sequential if batch fails
            for i in range(target):
                prompt = template.format(agent_pool_description=agent_pool_desc)
                topo, tokens = llm.generate_topology(prompt)
                if topo is None:
                    topo = fallback_fn()
                    tokens = 0
                topo = filter_agents_to_pool(canonicalize_topology(topo))
                if not topo:
                    topo = fallback_fn()
                indiv = Individual(code=topo, lineage=key)
                indiv.tokens_from_llm += tokens
                seeds.append(indiv)
        
        log_line(f"[Seed] {key} completed: {len([s for s in seeds if s.lineage == key])} seeds")
    
    # Fill remaining with random
    while len(seeds) < POPULATION_SIZE:
        topo = random_chain()
        seeds.append(Individual(code=topo, lineage="seed_random"))
    log_line(f"[Seed] Total seeds: {len(seeds)}")
    return seeds


def agnostic_addition(topology: str) -> str:
    roles = topology_to_linear_roles(topology)
    insert_idx = random.randint(0, len(roles)) if roles else 0
    new_agent = random.choice(ROLE_LIST)
    roles.insert(insert_idx, new_agent)
    return canonicalize_topology(" -> ".join(roles))


def agnostic_deletion(topology: str) -> str:
    roles = topology_to_linear_roles(topology)
    if len(roles) <= 1:
        return topology
    del_idx = random.randint(0, len(roles) - 1)
    roles.pop(del_idx)
    return canonicalize_topology(" -> ".join(roles))


def semantic_mutation(ind: Individual, llm: LLMController, mode: str) -> Tuple[str, int]:
    agent_pool_desc = ", ".join(ROLE_LIST[:50])
    if mode == "EXPAND":
        prompt = MUTATION_PROMPT_EXPAND.format(
            current_workflow_string=ind.code,
            agent_pool_description=agent_pool_desc,
        )
    else:
        prompt = MUTATION_PROMPT_COMPRESS.format(
            current_workflow_string=ind.code,
        )
    topo, tokens = llm.generate_topology(prompt)
    if topo is None:
        return "", 0
    return filter_agents_to_pool(canonicalize_topology(topo)), tokens


def _agnostic_mutation_only(ind: Individual) -> Individual:
    if ind.direction == "EXPAND":
        topo = agnostic_addition(ind.code)
    else:
        topo = agnostic_deletion(ind.code)
    return Individual(code=topo, lineage=f"mutation_agnostic_{ind.direction.lower()}")


def mutate(ind: Individual, llm: LLMController, use_llm: bool = True) -> Individual:
    # Semantic vs agnostic split
    if not use_llm:
        return _agnostic_mutation_only(ind)

    use_semantic = random.random() > AGNOSTIC_RATIO
    if use_semantic:
        topo, tokens = semantic_mutation(ind, llm, ind.direction)
        if topo:
            return Individual(code=topo, lineage=f"mutation_semantic_{ind.direction.lower()}", tokens_from_llm=tokens)
    # Fallback to agnostic
    if ind.direction == "EXPAND":
        topo = agnostic_addition(ind.code)
    else:
        topo = agnostic_deletion(ind.code)
    return Individual(code=topo, lineage=f"mutation_agnostic_{ind.direction.lower()}")


def adaptive_crossover(base: Individual, donor: Individual, llm: LLMController, use_llm: bool = True) -> Individual:
    if not use_llm:
        # reuse fallback mixing logic only
        roles_base = topology_to_linear_roles(base.code)
        roles_donor = [r for r in topology_to_linear_roles(donor.code) if r not in roles_base]
        if base.direction == "COMPRESS":
            merged = roles_base[: max(1, len(roles_base) - 1)] + roles_donor[:1]
        else:
            merged = roles_base + roles_donor[:2]
        topo_fallback = canonicalize_topology(" -> ".join(merged))
        return Individual(code=topo_fallback, lineage=f"crossover_mix_{base.direction.lower()}")

    if base.direction == "EXPAND":
        prompt = CROSSOVER_PROMPT_EXPAND
    else:
        prompt = CROSSOVER_PROMPT_COMPRESS
    formatted = prompt.format(
        topology_string_a=base.code,
        topology_string_b=donor.code,
    )
    topo, tokens = llm.generate_topology(formatted)
    if topo:
        topo_clean = filter_agents_to_pool(canonicalize_topology(topo))
        if topo_clean:
            lineage = f"crossover_semantic_{base.direction.lower()}"
            return Individual(code=topo_clean, lineage=lineage, tokens_from_llm=tokens)

    # Fallback mixing
    roles_base = topology_to_linear_roles(base.code)
    roles_donor = [r for r in topology_to_linear_roles(donor.code) if r not in roles_base]
    if base.direction == "COMPRESS":
        # keep it short
        merged = roles_base[: max(1, len(roles_base) - 1)] + roles_donor[:1]
    else:
        merged = roles_base + roles_donor[:2]
    topo_fallback = canonicalize_topology(" -> ".join(merged))
    return Individual(code=topo_fallback, lineage=f"crossover_mix_{base.direction.lower()}")


def update_momentum(ind: Individual):
    new_direction = ind.direction
    if ind.cost > MAX_COST_LIMIT:
        new_direction = "COMPRESS"
    elif ind.cost < MIN_COST_LIMIT:
        new_direction = "EXPAND"
    elif ind.patience >= PATIENCE_LIMIT:
        new_direction = "COMPRESS" if ind.direction == "EXPAND" else "EXPAND"
    elif ind.direction == "EXPAND" and (ind.score - ind.prev_score) < SCORE_IMPROVE_THRESHOLD:
        new_direction = "COMPRESS"

    if new_direction == ind.direction:
        ind.patience += 1
    else:
        ind.patience = 0
        ind.direction = new_direction


def assign_nsga_metrics(pop: List[Individual]):
    if not pop:
        return
    # Use node count instead of token cost for multi-objective optimization
    # Objective 1: minimize node count, Objective 2: maximize pass@k (so negate score)
    objs = np.array([[len(topology_to_linear_roles(ind.code)), -1 * ind.score] for ind in pop])
    fronts = non_dominated_sort(objs)
    for rank, front in enumerate(fronts, start=1):
        dist = crowding_distance(objs, front)
        for idx, ind_idx in enumerate(front):
            pop[ind_idx].rank = rank
            pop[ind_idx].distance = dist[idx] if idx < len(dist) else 0.0


def tournament_select(pop: List[Individual]) -> Individual:
    contenders = random.sample(pop, min(2, len(pop)))
    best = contenders[0]
    for c in contenders[1:]:
        if c.rank < best.rank:
            best = c
        elif c.rank == best.rank and c.distance > best.distance:
            best = c
    return best


def evaluate_population(pop: List[Individual], dataset: BaseDataset, fast: bool = False, server_url: str = "http://localhost:8000") -> None:
    log_line(f"[Eval] Starting evaluation: pop_size={len(pop)} fast={fast} num_problems={NUM_EVAL_PROBLEMS if not fast else 'N/A'}")
    workflows = [topology_to_block_workflow(ind.code, TASK_NAME) for ind in pop]
    if fast:
        log_line("[Eval] Using fast (random) evaluation")
        results = [{"pass_at_k": random.random(), "token": random.uniform(1500, 6000)} for _ in workflows]
    else:
        # Use evaluation server (same as ga.py)
        log_line(f"[Eval] Using evaluation server: {len(workflows)} workflows × {NUM_EVAL_PROBLEMS} problems (batch_size=60)")
        try:
            from src.evaluation_client import EvaluationClient, BlockConfig
            
            async def _evaluate_batch():
                client = EvaluationClient(server_url)
                all_results = []
                batch_size = 60
                
                for start in range(0, len(workflows), batch_size):
                    batch = workflows[start:start + batch_size]
                    log_line(f"[Eval] Processing batch {start//batch_size + 1}/{(len(workflows)-1)//batch_size + 1}: workflows {start+1}-{min(start+batch_size, len(workflows))}")
                    
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
                    
                    # Async evaluate batch
                    respond = await client.evaluate_batch_async(
                        workflows=batch_blocks,
                        task_name=TASK_NAME,
                        num_problems=NUM_EVAL_PROBLEMS,
                        use_extractor=False,
                        seed=EVAL_SEED,
                        think=False,
                    )
                    all_results.extend(respond)
                
                await client.close()
                return all_results
            
            eval_results = asyncio.run(_evaluate_batch())
            results = []
            for eval_result in eval_results:
                results.append({
                    "pass_at_k": eval_result.pass_at_1,
                    "token": eval_result.total_tokens
                })
            log_line(f"[Eval] Evaluation server complete: {len(results)} results")
        except Exception as e:
            log_line(f"[Eval] ERROR: Evaluation server failed: {e}, falling back to direct evaluation")
            import traceback
            log_line(f"[Eval] Traceback:\n{traceback.format_exc()}")
            # Fallback to direct evaluation
            results = []
            for i, wf in enumerate(workflows):
                if (i + 1) % 10 == 0:
                    log_line(f"[Eval] Progress: {i+1}/{len(workflows)} workflows evaluated")
                pass_at_1 = quick_evaluate(workflow=wf, dataset=dataset, num_problems=NUM_EVAL_PROBLEMS, seed=EVAL_SEED)
                token_cost = getattr(wf, "total_tokens", 0) or random.uniform(2000, 7000)
                results.append({"pass_at_k": pass_at_1, "token": token_cost})

    for ind, res in zip(pop, results):
        ind.update_fitness(res["pass_at_k"], res["token"])
        update_momentum(ind)
    log_line(f"[Eval] Fitness updated for all individuals")


def _shim_population_for_logging(population: List[Individual]):
    """
    Convert Individual list to the lightweight structure expected by the
    original GA utilities (checkpoint + pareto plot) without changing core flow.
    """
    shim = []
    for ind in population:
        wf = topology_to_block_workflow(ind.code, TASK_NAME)
        shim.append({
            "workflow": wf,
            "fitness": {
                "pass_at_k": ind.score,
                "token": ind.cost,
            },
        })
    return shim


def _plot_pareto_llm(population: List[Individual], file_name: str, save_dir: str = None):
    """
    Local copy of Pareto plotting to save under ga_llm-specific dir.
    X-axis: number of nodes in workflow (instead of token cost)
    Y-axis: pass@k score
    """
    import matplotlib.pyplot as plt
    save_dir = save_dir or GRAPH_DIR
    os.makedirs(save_dir, exist_ok=True)

    pass_at_k_array = np.array([ind.score for ind in population])
    # Calculate node count for each individual
    node_counts = np.array([len(topology_to_linear_roles(ind.code)) for ind in population])

    plt.figure(figsize=(7, 5))
    plt.scatter(node_counts, pass_at_k_array, c="red", s=40)
    plt.xlabel("Objective 1 (Number of Nodes)")
    plt.ylabel("Objective 2 (pass@k)")
    plt.title("Pareto Front (GA LLM)")
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log_line(f"[Saved] Pareto front saved to: {save_path}")
    plt.close()


def evolve(args, run_id: str = None):
    llm = LLMController(model=args.model, temperature=0.35) if USE_LLM else None
    dataset = get_dataset()
    
    # Setup run-specific directories
    if run_id:
        run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, run_id)
        run_graph_dir = os.path.join(GRAPH_DIR, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        os.makedirs(run_graph_dir, exist_ok=True)
    else:
        run_checkpoint_dir = CHECKPOINT_DIR
        run_graph_dir = GRAPH_DIR

    population = stratified_seed_population(llm)
    evaluate_population(population, dataset, fast=args.fast)
    log_line(f"[Init] seeded={len(population)} fast={args.fast} use_llm={USE_LLM}")

    seen_codes = {ind.code for ind in population}

    for gen in range(MAX_GENERATIONS):
        log_line(f"[Gen {gen+1:02d}] ========== START ==========")
        log_line(f"[Gen {gen+1:02d}] Step 1: Assigning NSGA metrics...")
        assign_nsga_metrics(population)
        
        # Select elites by fixed ratio (20% - reduced for more diversity)
        elites_count = max(1, int(0.2 * POPULATION_SIZE))
        sorted_pop = sorted(population, key=lambda i: (i.rank, -i.distance))
        new_pop: List[Individual] = sorted_pop[:elites_count]
        log_line(f"[Gen {gen+1:02d}] Step 2: Selected {elites_count} elites (20% of population)")

        # Create set of existing codes in current population and new_pop for efficient duplicate checking
        existing_codes = {ind.code for ind in population} | {ind.code for ind in new_pop}
        log_line(f"[Gen {gen+1:02d}] Step 2a: Duplicate check set size: seen_codes={len(seen_codes)}, existing_codes={len(existing_codes)}")

        log_line(f"[Gen {gen+1:02d}] Step 3: Generating children (target={POPULATION_SIZE - elites_count})...")
        children_generated = 0
        children_skipped_duplicate = 0
        while len(new_pop) < POPULATION_SIZE:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)

            # Decide operator
            use_mutation = random.random() < MUTATION_RATE
            if use_mutation:
                log_line(f"[Gen {gen+1:02d}] Step 3.{children_generated+1}: Mutating parent (lineage={parent_a.lineage}, dir={parent_a.direction})...")
                child = mutate(parent_a, llm, use_llm=USE_LLM)
                op_type = "mutation"
            else:
                log_line(f"[Gen {gen+1:02d}] Step 3.{children_generated+1}: Crossover (base={parent_a.lineage}, donor={parent_b.lineage})...")
                child = adaptive_crossover(parent_a, parent_b, llm, use_llm=USE_LLM)
                op_type = "crossover"

            # Check duplicates: both in seen_codes (historical) and existing_codes (current population + new_pop)
            if child.code in seen_codes or child.code in existing_codes:
                children_skipped_duplicate += 1
                log_line(f"[Gen {gen+1:02d}] Step 3.{children_generated+1}: Skipped duplicate topology (in seen_codes or current population)")
                continue
            
            seen_codes.add(child.code)
            existing_codes.add(child.code)  # Also add to current generation's set
            new_pop.append(child)
            children_generated += 1
            log_line(f"[Gen {gen+1:02d}] Step 3.{children_generated}: {op_type} success (lineage={child.lineage}, topo='{child.code[:60]}')")
            
            if children_generated % 10 == 0:
                log_line(f"[Gen {gen+1:02d}] Step 3 progress: {children_generated} children, {children_skipped_duplicate} duplicates skipped")

        log_line(f"[Gen {gen+1:02d}] Step 3 complete: {children_generated} children generated, {children_skipped_duplicate} duplicates skipped")
        log_line(f"[Gen {gen+1:02d}] Step 4: Evaluating {len(new_pop)-elites_count} new children...")
        evaluate_population(new_pop[elites_count:], dataset, fast=args.fast)
        for elite in new_pop[:elites_count]:
            elite.patience += 1
            update_momentum(elite)
        population = new_pop

        best = max(population, key=lambda i: i.score)
        avg_score = sum(i.score for i in population) / len(population)
        llm_calls = llm.call_count if llm else 0
        
        # Log detailed statistics
        log_line(f"[Gen {gen+1:02d}] Step 5: Statistics")
        log_line(f"[Gen {gen+1:02d}]   Best: score={best.score:.4f} cost={best.cost:.1f} rank={best.rank} dir={best.direction} lineage={best.lineage}")
        log_line(f"[Gen {gen+1:02d}]   Best topology: {best.code}")
        log_line(f"[Gen {gen+1:02d}]   Avg score: {avg_score:.4f}")
        log_line(f"[Gen {gen+1:02d}]   LLM calls: {llm_calls}")
        
        # Log top 5 individuals
        top5 = sorted(population, key=lambda i: (i.rank, -i.score))[:5]
        log_line(f"[Gen {gen+1:02d}]   Top 5:")
        for idx, ind in enumerate(top5, 1):
            log_line(f"[Gen {gen+1:02d}]     [{idx}] rank={ind.rank} score={ind.score:.4f} cost={ind.cost:.1f} lineage={ind.lineage}")

        # Save checkpoint and plot every generation (or every 5)
        save_every = 1  # Save every generation
        if (gen + 1) % save_every == 0:
            log_line(f"[Gen {gen+1:02d}] Step 6: Saving checkpoint and plot...")
            shim_pop = _shim_population_for_logging(population)
            checkpoint_name = f"{run_id}_gen{gen+1}" if run_id else f"evolite_gen{gen+1}"
            save_checkpoint_csv(shim_pop, checkpoint_name, save_dir=run_checkpoint_dir)
            log_line(f"[Gen {gen+1:02d}]   Checkpoint saved: {os.path.join(run_checkpoint_dir, f'population_{checkpoint_name}.csv')}")
            try:
                plot_name = f"{run_id}_gen{gen+1}" if run_id else f"evolite_gen{gen+1}"
                _plot_pareto_llm(population, plot_name, save_dir=run_graph_dir)
                log_line(f"[Gen {gen+1:02d}]   Plot saved: {os.path.join(run_graph_dir, f'{plot_name}.png')}")
            except Exception as exc:
                log_line(f"[Gen {gen+1:02d}]   [WARN] Pareto plotting skipped: {exc}")
        
        log_line(f"[Gen {gen+1:02d}] ========== END ==========")

    assign_nsga_metrics(population)
    global last_llm_call_count
    last_llm_call_count = llm.call_count if llm else 0
    
    # Close LLM connection
    if llm:
        try:
            import asyncio
            if llm._loop:
                llm._loop.run_until_complete(llm.close())
        except Exception as e:
            log_line(f"[WARN] LLM close failed: {e}")
    
    population.sort(key=lambda i: (i.rank, -i.score))
    return population


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EvoLite GA (LLM-driven topology evolution)")
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name (default: MBPP)")
    parser.add_argument("--fast", action="store_true", help="Use stochastic fast eval (no server calls)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    parser.add_argument("--model", type=str, required=False, help="LLM model name (optional; if omitted PromptGenerator will auto-select)")
    parser.add_argument("--run-id", type=str, required=False, help="Optional run id (default: UTC timestamp)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls; use random mutation/crossover only")
    parser.add_argument("--smoke-llm", action="store_true", help="Run a single lightweight LLM topology call and exit")
    args = parser.parse_args()

    # Setup run log path
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    global run_log_path
    run_log_path = os.path.join(RUN_LOG_DIR, f"{run_id}.log")
    global USE_LLM
    USE_LLM = not args.no_llm
    log_line(f"[Run] ga_llm start run_id={run_id} use_llm={USE_LLM}")

    # Optional LLM smoke test (single call) before full GA
    if args.smoke_llm:
        if not USE_LLM:
            log_line("[Smoke] LLM disabled (--no-llm); skip smoke test.")
            return
        log_line("[Smoke] Starting vLLM smoke test...")
        try:
            log_line("[Smoke] Step 1: Importing VLLMClient...")
            from src.llm.vllm_client import VLLMClient
            import asyncio
            log_line("[Smoke] Step 2: Creating VLLMClient instance...")
            client = VLLMClient()
            log_line(f"[Smoke] Step 3: VLLMClient created. base_url={client.base_url}, model={client.model}")
            
            async def test_single():
                log_line("[Smoke] Step 4: Testing single generate() call...")
                result = await client.generate(
                    system_prompt=GLOBAL_SYSTEM_PROMPT,
                    user_content="Return the simplest arrow syntax with two roles: Planner -> Solver"
                )
                log_line(f"[Smoke] Step 5: Single response received!")
                log_line(f"[Smoke] status={result.status}")
                log_line(f"[Smoke] content='{result.content[:100] if result.content else result.error}'")
                log_line(f"[Smoke] tokens={result.total_tokens}")
                log_line(f"[Smoke] time={result.execution_time:.2f}s")
                return result
            
            async def test_batch():
                log_line("[Smoke] Step 6: Testing batch_complete() call (3 prompts)...")
                messages_list = [
                    [
                        {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Return arrow syntax: Planner -> Solver (test {i+1})"}
                    ]
                    for i in range(3)
                ]
                results = await client.batch_complete(messages_list)
                log_line(f"[Smoke] Step 7: Batch response received! count={len(results)}")
                for i, r in enumerate(results):
                    log_line(f"[Smoke]   [{i+1}] status={r.status} tokens={r.total_tokens} content='{r.content[:50] if r.content else r.error}'")
                return results
            
            async def run_tests():
                try:
                    single_result = await test_single()
                    if single_result.status == "COMPLETED":
                        log_line("[Smoke] SUCCESS - single generate() works!")
                    else:
                        log_line(f"[Smoke] WARNING - single generate() failed: {single_result.error}")
                    
                    batch_results = await test_batch()
                    success_count = sum(1 for r in batch_results if r.status == "COMPLETED")
                    log_line(f"[Smoke] Batch: {success_count}/{len(batch_results)} succeeded")
                    if success_count > 0:
                        log_line("[Smoke] SUCCESS - batch_complete() works!")
                    
                    try:
                        client.print_stats()
                    except Exception as stats_err:
                        log_line(f"[Smoke] Stats print skipped: {stats_err}")
                finally:
                    await client.close()
            
            log_line("[Smoke] Running async tests...")
            asyncio.run(run_tests())
            log_line("[Smoke] All tests completed.")
        except Exception as e:
            log_line(f"[Smoke] ERROR: vLLM test failed: {e}")
            import traceback
            log_line(f"[Smoke] Traceback:\n{traceback.format_exc()}")
        return

    population = evolve(args, run_id=run_id)
    best = population[0]
    log_line("\n[Result] ========== FINAL RESULTS ==========")
    log_line(f"[Result] Top individual:")
    log_line(f"[Result]   Score={best.score:.4f} Cost={best.cost:.1f} Rank={best.rank} Dir={best.direction}")
    log_line(f"[Result]   Lineage={best.lineage}")
    log_line(f"[Result]   LLM_calls={last_llm_call_count}")
    log_line(f"[Result]   Topology: {best.code}")
    log_line(f"[Result] Checkpoints: {os.path.join(CHECKPOINT_DIR, run_id) if run_id else CHECKPOINT_DIR}")
    log_line(f"[Result] Graphs: {os.path.join(GRAPH_DIR, run_id) if run_id else GRAPH_DIR}")
    log_line(f"[Result] ====================================")


if __name__ == "__main__":
    main()
