import asyncio
import functools
import os
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from src.agents.block import AgentBlock, CompositeBlock
from src.agents.workflow_block import BlockWorkflow
from src.config import ROLE_DESCRIPTIONS
from src.datasets import MBPPDataset, BaseDataset
from src.evaluation.pass_at_k import quick_evaluate
from src.ga.multi_objective import crowding_distance, non_dominated_sort, plot_pareto
from src.ga.checkpoint import save_checkpoint_csv
from src.llm.client import PromptGenerator

# Flush all prints for long running loops
print = functools.partial(print, flush=True)

# Run-scoped paths
RUN_LOG_DIR = "src/ga/result"
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
INIT_SEEDED_POPULATION = 15  # Stratified seeding target (5 per type)
MAX_GENERATIONS = 10
MUTATION_RATE = 0.7  # Increased to promote exploration
CROSSOVER_RATE = 0.3
AGNOSTIC_RATIO = 0.2  # Inside mutation branch

NUM_EVAL_PROBLEMS = 30  # Fixed MBPP subset size
EVAL_SEED = 1337

# Cost limits will be set dynamically based on initial population evaluation
MIN_COST_LIMIT = None  # Will be set dynamically
MAX_COST_LIMIT = None  # Will be set dynamically
SCORE_IMPROVE_THRESHOLD = 0.01
PATIENCE_LIMIT = 3

LLM_CALL_BUDGET = 500  # Hard stop to avoid runaway LLM usage
LLM_MAX_TOKENS = 480

ROLE_LIST = ROLE_DESCRIPTIONS
TASK_NAME = "MBPP"
USE_LLM = True  # toggled via CLI
RUN_LOG_DIR = "src/ga/result"
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
    eval_count: int = 0  # Number of times this individual has been evaluated
    generation_age: int = 0  # Number of generations this individual has survived
    total_evaluated_problems: int = 0  # Total number of problems evaluated across all evaluations
    was_in_buffer: bool = False  # Track if this individual was in buffer before (to prevent re-entry)

    def __post_init__(self):
        self.graph = parse_arrow_syntax(self.code)

    def update_fitness(self, score: float, cost: float, num_problems: int = None):
        """
        Update fitness using weighted average if this is a re-evaluation.
        
        Args:
            score: New pass@1 score from this evaluation
            cost: New token cost from this evaluation
            num_problems: Number of problems evaluated in this round (for weighted average)
        """
        self.prev_score = self.score
        
        # If this is a re-evaluation and we have previous data, use weighted average
        if num_problems is not None and self.total_evaluated_problems > 0:
            # Weighted average: (old_total * old_value + new_num * new_value) / (old_total + new_num)
            total_problems = self.total_evaluated_problems + num_problems
            self.score = (self.total_evaluated_problems * self.score + num_problems * score) / total_problems
            self.cost = (self.total_evaluated_problems * self.cost + num_problems * cost) / total_problems
            self.total_evaluated_problems = total_problems
        else:
            # First evaluation or num_problems not provided
            self.score = score
            self.cost = cost + self.tokens_from_llm  # penalize heavy LLM edits
            if num_problems is not None:
                self.total_evaluated_problems = num_problems
        
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
        
        # Test connection on initialization
        self._test_connection()

    def _test_connection(self):
        """Test LLM connection on initialization."""
        try:
            self._ensure_loop()
            import asyncio
            
            async def _test():
                try:
                    # Quick connection test
                    test_result = await self.vllm_client.generate(
                        system_prompt="Test",
                        user_content="Test",
                        max_tokens=10,
                    )
                    if test_result.status == "COMPLETED":
                        log_line(f"[LLMController] Connection test successful: {self.vllm_client.base_url}")
                    else:
                        log_line(f"[LLMController] Connection test failed: {test_result.error}")
                        log_line(f"[LLMController] Will use fallback mode when LLM calls fail")
                except Exception as e:
                    log_line(f"[LLMController] Connection test error: {e}")
                    log_line(f"[LLMController] Will use fallback mode when LLM calls fail")
            
            # Run test in background (non-blocking)
            try:
                self._loop.run_until_complete(_test())
            except:
                pass  # Ignore test failures, will use fallback
        except Exception as e:
            log_line(f"[LLMController] Connection test setup failed: {e}")
            log_line(f"[LLMController] Will use fallback mode when LLM calls fail")

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
    seen_codes = set()  # Track unique topologies to avoid duplicates

    # Pure random seeding if LLM disabled/unavailable
    if not USE_LLM or llm is None:
        generators = [
            (random_chain, "seed_chain"),
            (random_loop, "seed_loop"),
            (random_branching, "seed_branch"),
            (random_test_driven, "seed_test"),
        ]
        max_attempts = POPULATION_SIZE * 10  # Prevent infinite loop
        attempts = 0
        while len(seeds) < POPULATION_SIZE and attempts < max_attempts:
            attempts += 1
            fn, name = random.choice(generators)
            topo = canonicalize_topology(fn())
            if topo not in seen_codes:
                seen_codes.add(topo)
                seeds.append(Individual(code=topo, lineage=name))
        if len(seeds) < POPULATION_SIZE:
            log_line(f"[Seed] WARNING: Only generated {len(seeds)} unique seeds (target: {POPULATION_SIZE})")
        return seeds

    agent_pool_desc = ", ".join(ROLE_LIST[:50])
    # Each type gets 5 seeds, total 15 (3 types × 5 = 15)
    counts = {
        "linear": 5,
        "reflex": 5,
        "branch": 5,
        "test": 0,  # Skip test type to keep total at 15
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
        if target == 0:
            continue
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
            
            # Process batch results with duplicate filtering
            duplicates_skipped = 0
            for i, (topo, tokens) in enumerate(batch_results):
                if topo is None:
                    log_line(f"[Seed] {key} batch[{i}]: LLM returned None, using fallback")
                    topo = fallback_fn()
                    tokens = 0
                topo = filter_agents_to_pool(canonicalize_topology(topo))
                if not topo:
                    log_line(f"[Seed] {key} batch[{i}]: filtered topo empty, using fallback")
                    topo = fallback_fn()
                
                # Check for duplicates
                if topo in seen_codes:
                    duplicates_skipped += 1
                    log_line(f"[Seed] {key} batch[{i}]: Skipped duplicate topology")
                    # Try fallback to get a unique topology
                    max_fallback_attempts = 10
                    for _ in range(max_fallback_attempts):
                        fallback_topo = canonicalize_topology(fallback_fn())
                        if fallback_topo not in seen_codes:
                            topo = fallback_topo
                            tokens = 0
                            break
                    else:
                        # If all fallbacks are duplicates, skip this seed
                        log_line(f"[Seed] {key} batch[{i}]: All fallbacks were duplicates, skipping")
                        continue
                
                seen_codes.add(topo)
                indiv = Individual(code=topo, lineage=key)
                indiv.tokens_from_llm += tokens
                seeds.append(indiv)
                log_line(f"[Seed] {key} batch[{i}]: got topo='{topo}' tokens={tokens}")
            
            if duplicates_skipped > 0:
                log_line(f"[Seed] {key}: Skipped {duplicates_skipped} duplicate(s) from batch")
        except Exception as e:
            log_line(f"[Seed] ERROR in batch for {key}: {e}, falling back to sequential")
            # Fallback to sequential if batch fails
            for i in range(target):
                max_attempts = 10
                attempts = 0
                while attempts < max_attempts:
                    prompt = template.format(agent_pool_description=agent_pool_desc)
                    topo, tokens = llm.generate_topology(prompt)
                    if topo is None:
                        topo = fallback_fn()
                        tokens = 0
                    topo = filter_agents_to_pool(canonicalize_topology(topo))
                    if not topo:
                        topo = fallback_fn()
                    
                    if topo not in seen_codes:
                        seen_codes.add(topo)
                        indiv = Individual(code=topo, lineage=key)
                        indiv.tokens_from_llm += tokens
                        seeds.append(indiv)
                        break
                    attempts += 1
                else:
                    log_line(f"[Seed] {key} seq[{i}]: Could not generate unique topology after {max_attempts} attempts")
        
        log_line(f"[Seed] {key} completed: {len([s for s in seeds if s.lineage == key])} seeds")
    
    # Fill remaining with random (with duplicate checking)
    max_random_attempts = (POPULATION_SIZE - len(seeds)) * 10
    random_attempts = 0
    while len(seeds) < POPULATION_SIZE and random_attempts < max_random_attempts:
        random_attempts += 1
        topo = canonicalize_topology(random_chain())
        if topo not in seen_codes:
            seen_codes.add(topo)
            seeds.append(Individual(code=topo, lineage="seed_random"))
    
    if len(seeds) < POPULATION_SIZE:
        log_line(f"[Seed] WARNING: Only generated {len(seeds)} unique seeds (target: {POPULATION_SIZE})")
    else:
        log_line(f"[Seed] Successfully generated {len(seeds)} unique seeds")
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
    # Only check cost limits if they are set (not None)
    if MAX_COST_LIMIT is not None and ind.cost > MAX_COST_LIMIT:
        new_direction = "COMPRESS"
    elif MIN_COST_LIMIT is not None and ind.cost < MIN_COST_LIMIT:
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
    # Use token cost for multi-objective optimization
    # Objective 1: minimize token cost, Objective 2: maximize pass@k (so negate score)
    objs = np.array([[ind.cost, -1 * ind.score] for ind in pop])
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


def select_buffer(candidates: List[Individual], buffer_size: int, fronts, population: List[Individual]) -> List[Individual]:
    """
    Select buffer (probation) candidates from the population.
    Buffer candidates are individuals that were survivors in previous generations
    but fell out of the top N in the current generation.
    
    Buffer priority:
    1. Higher Front level (Front 0 > Front 1 > Front 2 > ...) - 낮은 Front 번호가 우선
    2. Higher generation_age (veterans who survived longer)
    3. Higher fitness (score/pass_at_k)
    
    Args:
        candidates: List of individuals that didn't make it to survivors
        buffer_size: Maximum number of buffer candidates to select
        fronts: List of fronts from NSGA-II sorting (fronts[0] = Front 0, fronts[1] = Front 1, ...)
        population: Full population list (to map indices)
    
    Returns:
        List of buffer candidates
    """
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
    # 3. fitness score (higher is better)
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            -individual_to_front_level.get(id(x), 999),  # Negative: Front 0 (0) > Front 1 (1) > Front 2 (2)
            x.generation_age,
            x.score
        ),
        reverse=True
    )
    
    for agent in sorted_candidates:
        if len(buffer_list) >= buffer_size:
            break
        
        # IMPORTANT: Exclude individuals that were already in buffer before
        # Buffer purpose: give ONE chance to veterans who fell out of survivors
        # If they fail again, they should be removed, not given another buffer chance
        # Select individuals that have generation_age > 0 (were survivors before)
        # This ensures only veterans (not newborn offspring) get buffer protection
        # Buffer purpose: protect veterans from being eliminated by potentially lucky new offspring
        if agent.generation_age > 0 and not agent.was_in_buffer:
            buffer_list.append(agent)
            agent.was_in_buffer = True  # Mark as having been in buffer
    
    return buffer_list


def evaluate_population(pop: List[Individual], dataset: BaseDataset, fast: bool = False, server_url: str = "http://localhost:8001", max_eval_iter: int = 4, is_initial: bool = False, evaluate_newborns: bool = False, num_problems: int = None) -> None:
    # Filter individuals that haven't exceeded max_eval_iter
    # IMPORTANT: When evaluate_newborns=True, evaluate all individuals (both newborns and existing) with eval_count < max_eval_iter
    # This allows new children and existing individuals to be evaluated together in the same batch
    # Exception: For initial population, evaluate all individuals regardless of generation_age
    if is_initial or evaluate_newborns:
        # Evaluate all individuals that haven't reached max_eval_iter (both newborns and existing)
        eval_candidates = [ind for ind in pop if ind.eval_count < max_eval_iter]
    else:
        # Legacy mode: Only re-evaluate individuals that have lived at least 1 generation
        eval_candidates = [ind for ind in pop if ind.generation_age > 0 and ind.eval_count < max_eval_iter]
    
    if len(eval_candidates) == 0:
        log_line(f"[Eval] All individuals have reached max_eval_iter ({max_eval_iter}). Skipping evaluation.")
        return
    
    if len(eval_candidates) < len(pop):
        log_line(f"[Eval] Evaluating {len(eval_candidates)} individuals (out of {len(pop)} total, {len(pop) - len(eval_candidates)} skipped due to max_eval_iter)")
    
    # Always use the same number of problems for evaluation (no phase-based increase)
    if num_problems is None:
        num_problems_this_eval = NUM_EVAL_PROBLEMS
    else:
        num_problems_this_eval = num_problems
    
    log_line(f"[Eval] Starting evaluation: pop_size={len(eval_candidates)} fast={fast} num_problems={num_problems_this_eval if not fast else 'N/A'}")
    workflows = [topology_to_block_workflow(ind.code, TASK_NAME) for ind in eval_candidates]
    if fast:
        log_line("[Eval] Using fast (random) evaluation")
        results = [{"pass_at_k": random.random(), "token": random.uniform(1500, 6000)} for _ in workflows]
    else:
        # Use evaluation server (same as ga.py)
        log_line(f"[Eval] Using evaluation server: {len(workflows)} workflows × {NUM_EVAL_PROBLEMS} problems (batch_size=60)")
        try:
            from src.client import EvaluationClient, BlockConfig
            
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
                        num_problems=num_problems_this_eval,
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
                pass_at_1 = quick_evaluate(workflow=wf, dataset=dataset, num_problems=num_problems_this_eval, seed=EVAL_SEED)
                token_cost = getattr(wf, "total_tokens", 0) or random.uniform(2000, 7000)
                results.append({"pass_at_k": pass_at_1, "token": token_cost})

    for ind, res in zip(eval_candidates, results):
        # Update fitness with weighted average (pass num_problems for weighted average calculation)
        ind.update_fitness(res["pass_at_k"], res["token"], num_problems=num_problems_this_eval)
        ind.eval_count += 1  # Increment eval_count
        update_momentum(ind)
    log_line(f"[Eval] Fitness updated for {len(eval_candidates)} individuals (num_problems={num_problems_this_eval}, weighted average applied)")


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
            "_id": ind.code,  # Use code as unique identifier for plot_pareto matching
        })
    return shim




def evolve(args, run_id: str = None):
    llm = LLMController(model=args.model, temperature=0.35) if USE_LLM else None
    dataset = get_dataset()
    
    # Initialize Pareto front archive to store removed Front 0 individuals
    pareto_front_archive: List[Individual] = []
    
    # Setup run-specific directories
    if run_id:
        run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, run_id)
        run_graph_dir = os.path.join(GRAPH_DIR, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        os.makedirs(run_graph_dir, exist_ok=True)
    else:
        run_checkpoint_dir = CHECKPOINT_DIR
        run_graph_dir = GRAPH_DIR

    seed_start = time.time()
    log_line(f"[Init] Starting population seeding...")
    population = stratified_seed_population(llm)
    seed_time = time.time() - seed_start
    log_line(f"[Init] Seeding complete | elapsed={seed_time:.2f}s")
    
    eval_start = time.time()
    log_line(f"[Init] Evaluating initial population ({len(population)} individuals)...")
    evaluate_population(population, dataset, fast=args.fast, server_url=args.server_url, max_eval_iter=getattr(args, 'max_eval_iter', 4), is_initial=True)
    eval_time = time.time() - eval_start
    log_line(f"[Init] Initial evaluation complete | elapsed={eval_time:.2f}s")
    log_line(f"[Init] seeded={len(population)} fast={args.fast} use_llm={USE_LLM} elite_ratio={args.elite_ratio} buffer_size={getattr(args, 'buffer_size', 10)} max_eval_iter={getattr(args, 'max_eval_iter', 4)}")
    
    # Log initial statistics
    assign_nsga_metrics(population)  # Assign NSGA metrics for initial population
    
    # Dynamically set cost limits based on initial population token cost distribution
    # This allows the limits to adapt to the actual problem scale
    global MIN_COST_LIMIT, MAX_COST_LIMIT
    if population:
        costs = [ind.cost for ind in population]
        cost_min = min(costs)
        cost_max = max(costs)
        cost_median = np.median(costs)
        cost_q25 = np.percentile(costs, 25)
        cost_q75 = np.percentile(costs, 75)
        
        # Set MIN_COST_LIMIT to 25th percentile (or 0.8 * min if that's too low)
        # This allows expansion when cost is below the lower quartile
        MIN_COST_LIMIT = max(cost_q25 * 0.8, cost_min * 0.7)
        
        # Set MAX_COST_LIMIT to 75th percentile * 1.5 (or 1.3 * max if that's higher)
        # This allows compression when cost exceeds the upper quartile significantly
        MAX_COST_LIMIT = max(cost_q75 * 1.5, cost_max * 1.3)
        
        log_line(f"[Init] Dynamic cost limits set: MIN={MIN_COST_LIMIT:.0f} MAX={MAX_COST_LIMIT:.0f} "
                 f"(cost range: {cost_min:.0f}-{cost_max:.0f}, median={cost_median:.0f}, "
                 f"Q25={cost_q25:.0f}, Q75={cost_q75:.0f})")
    else:
        # Fallback to defaults if population is empty
        MIN_COST_LIMIT = 1200
        MAX_COST_LIMIT = 12000
        log_line(f"[Init] Using default cost limits: MIN={MIN_COST_LIMIT} MAX={MAX_COST_LIMIT}")
    
    # Calculate actual Pareto front (first front) for plotting
    # Use cost (token) and score (pass@1) for plotting, matching plot_pareto expectations
    objs_plot = np.array([[ind.cost, -1 * ind.score] for ind in population])
    fronts_plot = non_dominated_sort(objs_plot)
    pareto_front_indices_gen0 = fronts_plot[0] if fronts_plot else []
    pareto_front_individuals_gen0 = [population[i] for i in pareto_front_indices_gen0]
    
    # Select survivors for generation 0 (same logic as later generations)
    elites_count_gen0 = max(1, int(args.elite_ratio * POPULATION_SIZE))
    sorted_pop_gen0 = sorted(population, key=lambda i: (i.rank, -i.distance))
    
    # Prioritize Front 0 (rank 1) individuals
    front_0_individuals_gen0 = [ind for ind in sorted_pop_gen0 if ind.rank == 1]
    remaining_slots_gen0 = elites_count_gen0 - len(front_0_individuals_gen0)
    
    if remaining_slots_gen0 > 0:
        # Fill remaining slots from other ranks
        other_rank_individuals_gen0 = []
        slots_remaining = remaining_slots_gen0
        for rank in range(2, max([ind.rank for ind in population], default=1) + 1):
            if slots_remaining <= 0:
                break
            rank_individuals = [ind for ind in sorted_pop_gen0 if ind.rank == rank]
            if not rank_individuals:
                continue
            num_from_this_rank = min(slots_remaining, len(rank_individuals))
            other_rank_individuals_gen0.extend(rank_individuals[:num_from_this_rank])
            slots_remaining -= num_from_this_rank
        survivors_gen0 = front_0_individuals_gen0 + other_rank_individuals_gen0
    else:
        # Front 0 has more individuals than elites_count, use crowding distance to select top ones
        if len(front_0_individuals_gen0) > elites_count_gen0:
            front_0_objs_gen0 = np.array([[ind.cost, -1 * ind.score] for ind in front_0_individuals_gen0])
            front_0_indices_gen0 = list(range(len(front_0_individuals_gen0)))
            front_0_dist_gen0 = crowding_distance(front_0_objs_gen0, front_0_indices_gen0)
            front_0_sorted_by_dist_gen0 = sorted(range(len(front_0_individuals_gen0)), key=lambda i: -front_0_dist_gen0[i])
            survivors_gen0 = [front_0_individuals_gen0[i] for i in front_0_sorted_by_dist_gen0[:elites_count_gen0]]
        else:
            survivors_gen0 = front_0_individuals_gen0
    
    buffer_list_gen0 = []  # No buffer for generation 0
    
    best = max(population, key=lambda i: i.score)
    avg_score = sum(i.score for i in population) / len(population)
    llm_calls = llm.call_count if llm else 0
    
    log_line(f"[Init] Initial Statistics:")
    log_line(f"[Init]   Best: score={best.score:.4f} pass@1={best.score:.4f} cost={best.cost:.1f} rank={best.rank} dir={best.direction} lineage={best.lineage}")
    log_line(f"[Init]   Best topology: {best.code}")
    log_line(f"[Init]   Avg score: {avg_score:.4f}")
    log_line(f"[Init]   LLM calls: {llm_calls}")
    
    # Log top 5 individuals
    top5 = sorted(population, key=lambda i: (i.rank, -i.score))[:5]
    log_line(f"[Init]   Top 5:")
    for idx, ind in enumerate(top5, 1):
        log_line(f"[Init]     [{idx}] rank={ind.rank} score={ind.score:.4f} pass@1={ind.score:.4f} cost={ind.cost:.1f} lineage={ind.lineage}")
    
    # Save initial checkpoint and plot (generation 0)
    log_line(f"[Init] Saving initial checkpoint and plot...")
    shim_pop = _shim_population_for_logging(population)
    survivors_shim_gen0 = _shim_population_for_logging(survivors_gen0)
    buffer_shim_gen0 = _shim_population_for_logging(buffer_list_gen0)
    pareto_front_shim_gen0 = _shim_population_for_logging(pareto_front_individuals_gen0)
    checkpoint_name = f"{run_id}_gen0" if run_id else f"evolite_gen0"
    save_checkpoint_csv(shim_pop, checkpoint_name, save_dir=run_checkpoint_dir)
    log_line(f"[Init]   Checkpoint saved: {os.path.join(run_checkpoint_dir, f'population_{checkpoint_name}.csv')}")
    try:
        plot_name = f"{run_id}_gen0" if run_id else f"evolite_gen0"
        # Use actual survivors, buffer, and Pareto front for plotting (matching ga.py)
        plot_pareto(shim_pop, plot_name, save_dir=run_graph_dir, survivors=survivors_shim_gen0, buffer_list=buffer_shim_gen0, pareto_front=pareto_front_shim_gen0)
        log_line(f"[Init]   Plot saved: {os.path.join(run_graph_dir, f'{plot_name}.png')}")
    except Exception as exc:
        log_line(f"[Init]   [WARN] Pareto plotting skipped: {exc}")

    seen_codes = {ind.code for ind in population}

    for gen in range(MAX_GENERATIONS):
        gen_start_time = datetime.now()
        gen_start_str = gen_start_time.strftime("%Y-%m-%d %H:%M:%S")
        gen_start_ts = time.time()
        log_line(f"[Gen {gen+1:02d}] ========== START ==========")
        log_line(f"[Gen {gen+1:02d}] Start time: {gen_start_str}")
        
        step1_start = time.time()
        log_line(f"[Gen {gen+1:02d}] Step 1: Assigning NSGA metrics...")
        assign_nsga_metrics(population)
        
        # Calculate actual Pareto front (first front) for plotting
        # Use cost (token) and score (pass@1) for plotting, matching plot_pareto expectations
        objs_plot = np.array([[ind.cost, -1 * ind.score] for ind in population])
        fronts_plot = non_dominated_sort(objs_plot)
        pareto_front_indices = fronts_plot[0] if fronts_plot else []
        pareto_front_individuals = [population[i] for i in pareto_front_indices]
        
        step1_time = time.time() - step1_start
        log_line(f"[Gen {gen+1:02d}] Step 1 complete | elapsed={step1_time:.2f}s")
        
        # Select elites by ratio (configurable via CLI)
        step2_start = time.time()
        elites_count = max(1, int(args.elite_ratio * POPULATION_SIZE))
        sorted_pop = sorted(population, key=lambda i: (i.rank, -i.distance))
        
        # IMPORTANT: Ensure all Front 0 (rank=1, Pareto front) individuals are selected first
        # Then fill remaining slots from other ranks
        front_0_individuals = [ind for ind in sorted_pop if ind.rank == 1]
        remaining_slots = elites_count - len(front_0_individuals)
        
        if remaining_slots > 0:
            # Add individuals from other ranks (excluding rank=1)
            # IMPORTANT: Select from ranks in order (rank 2 → rank 3 → rank 4 → ...)
            # Within each rank, use distance (crowding distance) order
            other_rank_individuals = []
            slots_remaining = remaining_slots
            
            # Get max rank to iterate through
            max_rank = max((ind.rank for ind in sorted_pop), default=1)
            
            # Iterate through ranks in order (skip rank 1 which is Front 0)
            for rank in range(2, max_rank + 1):
                if slots_remaining <= 0:
                    break
                
                # Get individuals in this rank, sorted by distance (higher is better)
                rank_individuals = [ind for ind in sorted_pop if ind.rank == rank]
                rank_individuals_sorted = sorted(rank_individuals, key=lambda i: -i.distance)
                
                # Select top individuals from this rank
                num_from_this_rank = min(slots_remaining, len(rank_individuals_sorted))
                other_rank_individuals.extend(rank_individuals_sorted[:num_from_this_rank])
                slots_remaining -= num_from_this_rank
            
            survivors: List[Individual] = front_0_individuals + other_rank_individuals
        else:
            # Front 0 has more individuals than elites_count, use crowding distance to select top ones
            if len(front_0_individuals) > elites_count:
                # Re-sort Front 0 by crowding distance and take top elites_count
                front_0_objs = np.array([[ind.cost, -1 * ind.score] for ind in front_0_individuals])
                front_0_indices = list(range(len(front_0_individuals)))
                front_0_dist = crowding_distance(front_0_objs, front_0_indices)
                front_0_sorted_by_dist = sorted(range(len(front_0_individuals)), key=lambda i: -front_0_dist[i])
                survivors: List[Individual] = [front_0_individuals[i] for i in front_0_sorted_by_dist[:elites_count]]
                
                # Archive removed Front 0 individuals (they were in Pareto front but removed due to size limit)
                removed_front_0 = [front_0_individuals[i] for i in front_0_sorted_by_dist[elites_count:]]
                for ind in removed_front_0:
                    # Deep copy to preserve state at this generation
                    archived_individual = deepcopy(ind)
                    pareto_front_archive.append(archived_individual)
                
                if len(removed_front_0) > 0:
                    log_line(f"[Gen {gen+1:02d}] Archived {len(removed_front_0)} Front 0 individuals (Pareto front size limit)")
            else:
                survivors: List[Individual] = front_0_individuals
        
        # Increment generation_age for survivors
        # IMPORTANT: Reset was_in_buffer flag for survivors - they earned their place back
        for survivor in survivors:
            survivor.generation_age += 1
            survivor.was_in_buffer = False  # Reset buffer flag - they're survivors now
        
        log_line(f"[Gen {gen+1:02d}] Step 2: Selected {elites_count} survivors ({args.elite_ratio*100:.1f}% of population)")
        
        # Select Buffer (Probation) Candidates
        # IMPORTANT: Buffer purpose: Give probation to veterans (generation_age > 0) who fell out of survivors
        # Priority: Higher Front level (Front 0 > Front 1 > ...) > Higher generation_age > Higher fitness
        # This protects veterans from being eliminated by potentially lucky new offspring
        
        # First, identify all individuals that were NOT selected as survivors
        survivor_ids = set(id(s) for s in survivors)
        non_survivors = [ind for ind in population if id(ind) not in survivor_ids]
        
        # Select buffer from non_survivors, prioritizing by Front level, generation_age, and fitness
        buffer_size = getattr(args, 'buffer_size', 10)
        buffer_list = select_buffer(non_survivors, buffer_size, fronts_plot, population)
        log_line(f"[Gen {gen+1:02d}] Step 2b: Selected {len(buffer_list)} buffer candidates")
        
        step2_time = time.time() - step2_start
        log_line(f"[Gen {gen+1:02d}] Step 2 complete | elapsed={step2_time:.2f}s")

        # Create parent pool: survivors + buffer (both can reproduce)
        parent_pool = survivors + buffer_list
        
        # Create set of existing codes in current population for efficient duplicate checking
        existing_codes = {ind.code for ind in population} | {ind.code for ind in survivors} | {ind.code for ind in buffer_list}
        log_line(f"[Gen {gen+1:02d}] Step 2a: Duplicate check set size: seen_codes={len(seen_codes)}, existing_codes={len(existing_codes)}")

        step3_start = time.time()
        new_pop: List[Individual] = survivors.copy()  # Start with survivors
        log_line(f"[Gen {gen+1:02d}] Step 3: Generating children (target={POPULATION_SIZE - elites_count})...")
        children_generated = 0
        children_skipped_duplicate = 0
        while len(new_pop) < POPULATION_SIZE:
            parent_a = tournament_select(parent_pool)
            parent_b = tournament_select(parent_pool)

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

        step3_time = time.time() - step3_start
        log_line(f"[Gen {gen+1:02d}] Step 3 complete: {children_generated} children generated, {children_skipped_duplicate} duplicates skipped | elapsed={step3_time:.2f}s")
        
        step4_start = time.time()
        # Evaluate new children AND existing individuals that haven't reached max_eval_iter
        children = new_pop[elites_count:]
        
        # Collect all individuals that need evaluation:
        # 1. New children (generation_age=0, eval_count=0)
        # 2. Existing individuals (survivors + buffer) with eval_count < max_eval_iter
        max_eval_iter = getattr(args, 'max_eval_iter', 4)
        eval_candidates = []
        
        # Add new children
        for child in children:
            if child.eval_count < max_eval_iter:
                eval_candidates.append(child)
        
        # Add existing individuals (survivors + buffer) that need re-evaluation
        for ind in survivors + buffer_list:
            if ind.eval_count < max_eval_iter:
                eval_candidates.append(ind)
        
        log_line(f"[Gen {gen+1:02d}] Step 4: Evaluating {len(eval_candidates)} individuals ({len(children)} new children + {len(eval_candidates) - len(children)} existing individuals with eval_count < {max_eval_iter})...")
        if len(eval_candidates) > 0:
            evaluate_population(eval_candidates, dataset, fast=args.fast, server_url=args.server_url, max_eval_iter=max_eval_iter, evaluate_newborns=True)
        
        for elite in survivors:
            elite.patience += 1
            update_momentum(elite)
        
        # Compose next generation: Survivors + Offspring + Buffer
        population = survivors + children + buffer_list
        log_line(f"[Gen {gen+1:02d}] Next generation pool: {len(survivors)} survivors + {len(children)} offspring + {len(buffer_list)} buffer = {len(population)} total")
        step4_time = time.time() - step4_start
        log_line(f"[Gen {gen+1:02d}] Step 4 complete | elapsed={step4_time:.2f}s")

        step5_start = time.time()
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
        step5_time = time.time() - step5_start
        log_line(f"[Gen {gen+1:02d}] Step 5 complete | elapsed={step5_time:.2f}s")
        
        # Save checkpoint and plot every generation (or every 5)
        save_every = 1  # Save every generation
        step6_start = time.time()
        if (gen + 1) % save_every == 0:
            log_line(f"[Gen {gen+1:02d}] Step 6: Saving checkpoint and plot...")
            shim_pop = _shim_population_for_logging(population)
            checkpoint_name = f"{run_id}_gen{gen+1}" if run_id else f"evolite_gen{gen+1}"
            save_checkpoint_csv(shim_pop, checkpoint_name, save_dir=run_checkpoint_dir)
            log_line(f"[Gen {gen+1:02d}]   Checkpoint saved: {os.path.join(run_checkpoint_dir, f'population_{checkpoint_name}.csv')}")
            try:
                plot_name = f"{run_id}_gen{gen+1}" if run_id else f"evolite_gen{gen+1}"
                # IMPORTANT: Re-compute pareto_front_individuals from the new population
                # because population was reconstructed (survivors + children + buffer)
                # and pareto_front_individuals may reference old population objects
                # Re-calculate Pareto front from the new population
                objs_plot_new = np.array([[ind.cost, -1 * ind.score] for ind in population])
                fronts_plot_new = non_dominated_sort(objs_plot_new)
                pareto_front_indices_new = fronts_plot_new[0] if fronts_plot_new else []
                pareto_front_individuals_new = [population[i] for i in pareto_front_indices_new]
                
                # Convert survivors, buffer, and pareto front to shim format for plot_pareto
                survivors_shim = _shim_population_for_logging(survivors)
                buffer_shim = _shim_population_for_logging(buffer_list)
                pareto_front_shim = _shim_population_for_logging(pareto_front_individuals_new)
                # Use plot_pareto with actual Pareto front for line connection
                plot_pareto(shim_pop, plot_name, save_dir=run_graph_dir, survivors=survivors_shim, buffer_list=buffer_shim, pareto_front=pareto_front_shim)
                log_line(f"[Gen {gen+1:02d}]   Plot saved: {os.path.join(run_graph_dir, f'{plot_name}.png')}")
            except Exception as exc:
                log_line(f"[Gen {gen+1:02d}]   [WARN] Pareto plotting skipped: {exc}")
        step6_time = time.time() - step6_start
        log_line(f"[Gen {gen+1:02d}] Step 6 complete | elapsed={step6_time:.2f}s")
        
        # Log generation end time and elapsed time
        gen_end_time = datetime.now()
        gen_end_str = gen_end_time.strftime("%Y-%m-%d %H:%M:%S")
        gen_elapsed = time.time() - gen_start_ts
        log_line(f"[Gen {gen+1:02d}] End time: {gen_end_str}")
        log_line(f"[Gen {gen+1:02d}] Total elapsed time: {gen_elapsed:.2f}s (Step1={step1_time:.2f}s Step2={step2_time:.2f}s Step3={step3_time:.2f}s Step4={step4_time:.2f}s Step5={step5_time:.2f}s Step6={step6_time:.2f}s)")
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
    
    # ========== FINALIZATION: Combine archive + final front and validate ==========
    finalize_valid = getattr(args, 'finalize_valid', 100)
    log_line(f"\n{'='*60}")
    log_line("Finalization: Combining Pareto front archive with final generation")
    log_line(f"{'='*60}")
    
    # Get final Pareto front from last generation (rank=1)
    pareto_front_final = [ind for ind in population if ind.rank == 1]
    
    # Combine archive and final front
    combined_candidates = pareto_front_archive + pareto_front_final
    
    log_line(f"Archive size: {len(pareto_front_archive)}")
    log_line(f"Final front size: {len(pareto_front_final)}")
    log_line(f"Combined candidates: {len(combined_candidates)}")
    log_line(f"Validating top candidates with {finalize_valid} problems...")
    
    # Evaluate combined candidates with finalize_valid problems
    if len(combined_candidates) > 0:
        workflows_combined = [topology_to_block_workflow(ind.code, TASK_NAME) for ind in combined_candidates]
        
        if args.fast:
            results = [{"pass_at_k": random.random(), "token": random.uniform(1500, 6000)} for _ in workflows_combined]
        else:
            from src.client import EvaluationClient, BlockConfig
            async def _evaluate_final():
                client = EvaluationClient(args.server_url)
                all_results = []
                batch_size = 60
                
                for start in range(0, len(workflows_combined), batch_size):
                    batch = workflows_combined[start:start + batch_size]
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
                    
                    respond = await client.evaluate_batch_async(
                        workflows=batch_blocks,
                        task_name=TASK_NAME,
                        num_problems=finalize_valid,
                        use_extractor=False,
                        seed=EVAL_SEED,
                        think=False,
                    )
                    all_results.extend(respond)
                
                await client.close()
                return all_results
            
            eval_results = asyncio.run(_evaluate_final())
            results = []
            for eval_result in eval_results:
                results.append({
                    "pass_at_k": eval_result.pass_at_1,
                    "token": eval_result.total_tokens
                })
        
        # Update fitness with final validation results
        for ind, res in zip(combined_candidates, results):
            ind.update_fitness(res["pass_at_k"], res["token"], num_problems=finalize_valid)
    
    # Re-sort combined candidates using NSGA-II
    objs_combined = np.array([[ind.cost, -1 * ind.score] for ind in combined_candidates])
    fronts_combined = non_dominated_sort(objs_combined)
    pareto_front_indices_combined = fronts_combined[0] if fronts_combined else []
    final_pareto_front = [combined_candidates[i] for i in pareto_front_indices_combined]
    
    log_line(f"Final Pareto front size: {len(final_pareto_front)}")
    if len(final_pareto_front) > 0:
        log_line(f"Final Pareto front pass@k range: {min([ind.score for ind in final_pareto_front]):.4f} - {max([ind.score for ind in final_pareto_front]):.4f}")
        log_line(f"Final Pareto front token range: {min([ind.cost for ind in final_pareto_front]):.0f} - {max([ind.cost for ind in final_pareto_front]):.0f}")
    
    # Save final Pareto front
    final_shim = _shim_population_for_logging(final_pareto_front)
    checkpoint_name = f"{run_id}_final" if run_id else "evolite_final"
    save_checkpoint_csv(final_shim, checkpoint_name, save_dir=run_checkpoint_dir)
    log_line(f"Final checkpoint saved: {os.path.join(run_checkpoint_dir, f'population_{checkpoint_name}.csv')}")
    
    try:
        plot_name = f"{run_id}_final" if run_id else "evolite_final"
        final_shim_survivors = _shim_population_for_logging(final_pareto_front)
        plot_pareto(final_shim, plot_name, save_dir=run_graph_dir, survivors=final_shim_survivors, buffer_list=[], pareto_front=final_shim_survivors)
        log_line(f"Final plot saved: {os.path.join(run_graph_dir, f'{plot_name}.png')}")
    except Exception as exc:
        log_line(f"[WARN] Final Pareto plotting skipped: {exc}")
    
    log_line(f"{'='*60}\n")
    
    return final_pareto_front


def main():
    import argparse
    global POPULATION_SIZE, MAX_GENERATIONS, NUM_EVAL_PROBLEMS, USE_LLM, run_log_path

    parser = argparse.ArgumentParser(description="EvoLite GA (LLM-driven topology evolution)")
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name (default: MBPP)")
    parser.add_argument("--fast", action="store_true", help="Use stochastic fast eval (no server calls)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    parser.add_argument("--model", type=str, required=False, help="LLM model name (optional; if omitted PromptGenerator will auto-select)")
    parser.add_argument("--run-id", type=str, required=False, help="Optional run id (default: UTC timestamp)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls; use random mutation/crossover only")
    parser.add_argument("--smoke-llm", action="store_true", help="Run a single lightweight LLM topology call and exit")
    parser.add_argument("--server-url", type=str, default="http://localhost:8001", help="Evaluation server URL")
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE, help="The size of population")
    parser.add_argument("--generation", type=int, default=MAX_GENERATIONS, help="The number of generations")
    parser.add_argument("--num-problem", type=int, default=NUM_EVAL_PROBLEMS, help="The number of problems to evaluate")
    parser.add_argument("--elite-ratio", type=float, default=0.2, help="Elitism ratio (default: 0.2 = 20%%)")
    parser.add_argument("--buffer-size", type=int, default=10, help="The size of buffer (probation) pool")
    parser.add_argument("--max-eval-iter", type=int, default=4, help="Maximum evaluation iterations per individual")
    parser.add_argument("--finalize-valid", type=int, default=100, help="Number of problems for final validation of Pareto front")
    args = parser.parse_args()

    # Setup run log path
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_log_path = os.path.join(RUN_LOG_DIR, f"{run_id}.log")
    USE_LLM = not args.no_llm
    
    # Override constants with command-line arguments
    POPULATION_SIZE = args.population_size
    MAX_GENERATIONS = args.generation
    NUM_EVAL_PROBLEMS = args.num_problem
    
    log_line(f"[Run] ga_llm start run_id={run_id} use_llm={USE_LLM} pop_size={POPULATION_SIZE} generations={MAX_GENERATIONS} num_problems={NUM_EVAL_PROBLEMS}")

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
