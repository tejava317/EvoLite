import argparse
import asyncio
import functools
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.agents.block import AgentBlock, CompositeBlock
from src.agents.workflow_block import BlockWorkflow
from src.config import ROLE_DESCRIPTIONS
from src.datasets import BaseDataset, MBPPDataset, MathAlgebraDataset
from src.ga.checkpoint import save_checkpoint_csv
from src.ga.multi_objective import plot_pareto, non_dominated_sort, crowding_distance
import numpy as np
from src.llm.vllm_client import VLLMClient
from src.client import EvaluationClient, BlockConfig

# Ensure immediate flush for long running loops
print = functools.partial(print, flush=True)

# =======================
# High-level configuration
# =======================
NUM_EVAL_PROBLEMS = 30
EVAL_SEED = 42
TOKEN_PENALTY = 1e-4  # aligns with ga.py default
MAX_FAILURE_SAMPLES = 3
SUCCESS_SAMPLES = 1
DEFAULT_TASK = "MBPP"

# Directories (re-use GA style logging/checkpoints)
RUN_LOG_DIR = "logs/hdlo_runs"
JOURNAL_DIR = "logs/hdlo_journal"
CHECKPOINT_DIR = "src/ga/hdlo_checkpoints"
GRAPH_DIR = "src/ga/hdlo_graph"

# Roles pulled from config (same as GA)
ROLE_LIST = ROLE_DESCRIPTIONS
PROPOSALS_PER_ROUND = 1  # can be increased via CLI


# =======================
# Data containers
# =======================
@dataclass
class CaseCapture:
    problem_id: str
    prompt: str
    output: str
    ground_truth: str
    is_correct: bool
    tokens: float = 0.0
    trace: List[dict] = field(default_factory=list)


@dataclass
class EvaluationResult:
    pass_at_1: float
    token_cost: float
    score: float
    success_cases: List[CaseCapture] = field(default_factory=list)
    failure_cases: List[CaseCapture] = field(default_factory=list)
    eval_count: int = 0  # Number of times this workflow has been evaluated
    generation_age: int = 0  # Number of rounds this workflow has been in the front
    total_evaluated_problems: int = 0  # Total number of problems evaluated across all evaluations
    was_in_buffer: bool = False  # Track if this workflow was in buffer before (to prevent re-entry)


@dataclass
class BiasReport:
    summary: str
    raw_bullets: List[str]


@dataclass
class JournalEntry:
    round_idx: int
    action: str
    score_before: float
    score_after: float
    token_before: float
    token_after: float
    lesson: str
    diagnosis: str

    def delta_score(self) -> float:
        return self.score_after - self.score_before

    def delta_token(self) -> float:
        return self.token_after - self.token_before


# =======================
# Utility helpers
# =======================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_writer(run_id: str) -> Callable[[str], None]:
    ensure_dir(RUN_LOG_DIR)
    log_path = Path(RUN_LOG_DIR) / f"{run_id}.log"

    def _log(msg: str):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


def get_dataset(task_name: str) -> BaseDataset:
    task_lower = task_name.lower()
    if "math" in task_lower or "algebra" in task_lower:
        ds: BaseDataset = MathAlgebraDataset(split="test")
    else:
        ds = MBPPDataset(split="test")
    ds.load()
    return ds


def workflow_from_roles(task_name: str, roles: List[str]) -> BlockWorkflow:
    """Create a BlockWorkflow using the allowed role list."""
    role_chain = roles if roles else [random.choice(ROLE_LIST)]
    blocks = [AgentBlock(r) for r in role_chain]
    return BlockWorkflow(task_name=task_name, blocks=blocks)


def extract_roles(text: str) -> List[str]:
    """
    Convert LLM output (arrow syntax) to an ordered list of valid roles.
    CoT-aware: if a 'Topology:' marker exists, parse only the portion after it.
    Accepts separators: -> , newline, comma, brackets.
    """
    # 1) If CoT format includes "Topology:", use the part after that marker
    match = re.search(r"Topology:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    target_text = match.group(1) if match else text

    # 2) Normalize and split
    cleaned = target_text.replace("[", "").replace("]", "").strip()
    tokens = re.split(r"->|,|\n", cleaned)

    roles = []
    for tok in tokens:
        role = re.sub(r"[^a-zA-Z0-9_ ]", "", tok).strip()
        if role and role in ROLE_LIST:
            roles.append(role)

    # 3) Remove consecutive duplicates while keeping order
    deduped = []
    for r in roles:
        if not deduped or deduped[-1] != r:
            deduped.append(r)
    return deduped


def canonicalize_workflow(workflow: BlockWorkflow) -> str:
    """
    Canonicalize workflow string for duplicate detection.
    Normalizes whitespace and standardizes format.
    """
    wf_str = workflow_to_string(workflow)
    if not wf_str:
        return ""
    
    # Normalize arrow spacing to " -> "
    wf_str = re.sub(r"\s*->\s*", " -> ", wf_str)
    # Normalize multiple spaces to single space
    wf_str = re.sub(r"\s{2,}", " ", wf_str)
    # Strip whitespace
    wf_str = wf_str.strip()
    
    # Split by " -> " and remove duplicates while preserving order
    roles = [role.strip() for role in wf_str.split(" -> ") if role.strip()]
    seen = set()
    uniq_roles = []
    for role in roles:
        if role not in seen:
            uniq_roles.append(role)
            seen.add(role)
    
    return " -> ".join(uniq_roles)


def workflow_to_string(wf: BlockWorkflow) -> str:
    return wf.workflow_to_string() or " -> ".join(ROLE_LIST[:2])


def _trace_from_result(result: dict) -> List[dict]:
    """
    BlockWorkflow currently returns only intermediate_results.
    Try to derive a per-step trace from those strings.
    Expected format per entry: 'Agent i (Role): <snippet...>'
    """
    trace: List[dict] = []
    if not isinstance(result, dict):
        return trace
    if "steps" in result and isinstance(result["steps"], list):
        return result["steps"]
    interm = result.get("intermediate_results", []) or []
    for entry in interm:
        try:
            # Example: "Agent 0 (Planner): do something..."
            before, _, after = entry.partition(":")
            role_part = before.split("(")[-1].split(")")[0]
            trace.append({"role": role_part or "unknown", "output": after.strip()})
        except Exception:
            trace.append({"role": "unknown", "output": str(entry)})
    return trace


# =======================
# VLLM Wrapper for synchronous generate() interface
# =======================
class VLLMWrapper:
    """Synchronous wrapper around VLLMClient for compatibility with PromptGenerator interface."""
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000):
        self.vllm_client = VLLMClient(model=model, default_temperature=temperature, default_max_tokens=max_tokens)
        self.model = self.vllm_client.model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, system_prompt: str, user_content: str, max_tokens: int = None) -> dict:
        """Synchronous generate() method compatible with PromptGenerator interface."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            results = asyncio.run(self.vllm_client.batch_complete(
                [messages],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            ))
            if results and results[0].status == "COMPLETED":
                return {
                    "content": results[0].content or "",
                    "prompt_tokens": results[0].prompt_tokens or 0,
                    "response_tokens": results[0].completion_tokens or 0,
                    "total_tokens": (results[0].prompt_tokens or 0) + (results[0].completion_tokens or 0),
                }
            else:
                # Log failure for debugging
                error_msg = results[0].error if results and results[0].error else "Unknown error"
                print(f"[VLLMWrapper] generate failed: {error_msg}")
                return {"content": "", "prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        except Exception as exc:
            print(f"[VLLMWrapper] generate exception: {exc}")
            return {"content": "", "prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}


# =======================
# Module A. Dataset Bias Analyzer
# =======================
class DatasetBiasAnalyzer:
    def __init__(self, llm: Optional[VLLMWrapper], sample_size: int = NUM_EVAL_PROBLEMS):
        self.llm = llm
        self.sample_size = sample_size

    def analyze(self, dataset: BaseDataset, seed: int, task_name: str = None) -> BiasReport:
        problems = dataset.sample(min(self.sample_size, len(dataset)), seed=seed)
        bullets = []
        for p in problems[:10]:
            bullets.append(f"- {p.id}: prompt_len={len(p.prompt.split())} gt_type={type(p.ground_truth).__name__}")

        # Check task_name first, then fall back to dataset type or split
        is_math = False
        if task_name:
            is_math = "math" in task_name.lower() or "algebra" in task_name.lower()
        elif hasattr(dataset, '__class__'):
            is_math = "MathAlgebra" in dataset.__class__.__name__
        else:
            is_math = "math" in dataset.split.lower() if hasattr(dataset, 'split') else False

        heuristic = (
            "Dataset mix leans to short-form code tasks; treat math text explicitly."
            if is_math
            else "Dataset likely MBPP-style code writing; include execution-ready outputs."
        )

        if not self.llm:
            return BiasReport(summary=f"[Heuristic] {heuristic}", raw_bullets=bullets)

        sample_prompts = "\n".join(bullets)
        user_prompt = f"""You are a dataset bias spotter.
Summarize the validation set difficulty and dominant problem shapes.

Context bullets:
{sample_prompts}

Output:
- Difficulty (easy/medium/hard)
- Dominant pattern (math vs string vs control)
- Any over-represented format risks
Keep it under 5 lines."""
        try:
            resp = self.llm.generate(system_prompt="Return concise dataset bias notes.", user_content=user_prompt)
            summary = resp.get("content", "").strip()
            return BiasReport(summary=summary or heuristic, raw_bullets=bullets)
        except Exception:
            return BiasReport(summary=f"[Fallback] {heuristic}", raw_bullets=bullets)


# =======================
# Module B. Black-box Evaluator
# =======================
class BlackBoxEvaluator:
    def __init__(
        self,
        dataset: BaseDataset,
        seed: int = EVAL_SEED,
        task_name: str = DEFAULT_TASK,
        num_problems: int = NUM_EVAL_PROBLEMS,
        use_server: bool = True,
        server_url: str = "http://localhost:8001",
        batch_size: int = 60,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.dataset = dataset
        self.seed = seed
        self.task_name = task_name
        self.num_problems = num_problems
        self.use_server = use_server
        self.server_url = server_url
        self.batch_size = batch_size
        self.log = log or (lambda x: None)

    def evaluate(self, workflow: BlockWorkflow, num_problems: int = None) -> EvaluationResult:
        num_problems = num_problems or self.num_problems
        
        if self.use_server:
            return self._evaluate_with_server(workflow, num_problems)
        else:
            return self._evaluate_local(workflow, num_problems)

    def evaluate_batch(self, workflows: List[BlockWorkflow], num_problems: int = None) -> List[EvaluationResult]:
        num_problems = num_problems or self.num_problems
        if self.use_server:
            res = self._evaluate_with_server_batch(workflows, num_problems)
            if res:
                return res
        # Fallback: evaluate sequentially
        return [self._evaluate_local(wf, num_problems) for wf in workflows]

    def _evaluate_with_server(self, workflow: BlockWorkflow, num_problems: int) -> EvaluationResult:
        """Evaluate using the evaluation server."""
        async def _run():
            client = EvaluationClient(self.server_url)
            blocks_cfg = []
            for block in workflow.blocks:
                if isinstance(block, AgentBlock):
                    blocks_cfg.append(BlockConfig(type="agent", role=block.role))
                elif isinstance(block, CompositeBlock):
                    blocks_cfg.append(BlockConfig(
                        type="composite",
                        divider_role=block.divider_role,
                        synth_role=block.synth_role
                    ))

            respond = await client.evaluate_batch_async(
                workflows=[blocks_cfg],
                task_name=self.task_name,
                num_problems=num_problems,
                use_extractor=False,
                seed=self.seed,
                think=False,
            )
            await client.close()
            return respond

        try:
            respond = asyncio.run(_run())
            if not respond:
                self.log(f"[Eval] Server returned empty response for workflow: {workflow_to_string(workflow)}")
                return EvaluationResult(
                    pass_at_1=0.0,
                    token_cost=0.0,
                    score=0.0,
                    success_cases=[],
                    failure_cases=[],
                )

            if len(respond) == 0:
                self.log(f"[Eval] Server returned empty list for workflow: {workflow_to_string(workflow)}")
                return EvaluationResult(
                    pass_at_1=0.0,
                    token_cost=0.0,
                    score=0.0,
                    success_cases=[],
                    failure_cases=[],
                )

            eval_result = respond[0]
            # Debug: log the actual response object
            self.log(f"[Eval] Server response type: {type(eval_result)}, dir: {dir(eval_result)[:10] if hasattr(eval_result, '__dict__') else 'N/A'}")
            if hasattr(eval_result, '__dict__'):
                self.log(f"[Eval] Server response attributes: {eval_result.__dict__}")
            
            pass_at_1 = getattr(eval_result, "pass_at_1", 0.0)
            token_cost = float(getattr(eval_result, "total_tokens", 0.0) or 0.0)
            
            self.log(f"[Eval] Server result: pass_at_1={pass_at_1}, tokens={token_cost}")

            # Server evaluation doesn't provide detailed output for diagnosis
            # Only error field is available for execution exceptions
            # For wrong answers, we only know correct=False but not why
            return EvaluationResult(
                pass_at_1=pass_at_1,
                token_cost=token_cost,
                score=pass_at_1 - TOKEN_PENALTY * token_cost,
                success_cases=[],
                failure_cases=[],
            )
        except Exception as exc:
            self.log(f"[Eval] Server evaluation failed: {exc}")
            return self._evaluate_local(workflow, num_problems)

    def _evaluate_with_server_batch(self, workflows: List[BlockWorkflow], num_problems: int) -> Optional[List[EvaluationResult]]:
        if not workflows:
            return []

        async def _run():
            client = EvaluationClient(self.server_url)
            wf_cfgs = []
            for wf in workflows:
                blocks_cfg = []
                for block in wf.blocks:
                    if isinstance(block, AgentBlock):
                        blocks_cfg.append(BlockConfig(type="agent", role=block.role))
                    elif isinstance(block, CompositeBlock):
                        blocks_cfg.append(BlockConfig(
                            type="composite",
                            divider_role=block.divider_role,
                            synth_role=block.synth_role
                        ))
                wf_cfgs.append(blocks_cfg)

            respond = await client.evaluate_batch_async(
                workflows=wf_cfgs,
                task_name=self.task_name,
                num_problems=num_problems,
                use_extractor=False,
                seed=self.seed,
                think=False,
            )
            await client.close()
            return respond

        try:
            respond = asyncio.run(_run())
            results: List[EvaluationResult] = []
            if not respond:
                self.log(f"[Eval] Server batch returned empty response for {len(workflows)} workflows")
                return None
            
            if len(respond) != len(workflows):
                self.log(f"[Eval] Server batch returned {len(respond)} results, expected {len(workflows)}")
            
            for idx, eval_result in enumerate(respond):
                # Debug: log the actual response object for first item
                if idx == 0:
                    self.log(f"[Eval] Batch[0] response type: {type(eval_result)}, dir: {dir(eval_result)[:10] if hasattr(eval_result, '__dict__') else 'N/A'}")
                    if hasattr(eval_result, '__dict__'):
                        self.log(f"[Eval] Batch[0] response attributes: {eval_result.__dict__}")
                
                pass_at_1 = getattr(eval_result, "pass_at_1", 0.0)
                token_cost = float(getattr(eval_result, "total_tokens", 0.0) or 0.0)
                self.log(f"[Eval] Batch[{idx}] result: pass_at_1={pass_at_1}, tokens={token_cost}")
                results.append(
                    EvaluationResult(
                        pass_at_1=pass_at_1,
                        token_cost=token_cost,
                        score=pass_at_1 - TOKEN_PENALTY * token_cost,
                        success_cases=[],
                        failure_cases=[],
                    )
                )
            return results
        except Exception as exc:
            self.log(f"[Eval] Server batch evaluation failed: {exc}")
            return None

    def _evaluate_local(self, workflow: BlockWorkflow, num_problems: int) -> EvaluationResult:
        """Evaluate locally using workflow.run()."""
        problems = self.dataset.sample(num_problems, seed=self.seed)
        total_tokens = 0.0
        success_cases: List[CaseCapture] = []
        failure_cases: List[CaseCapture] = []
        correct = 0

        for p in problems:
            try:
                result = workflow.run(p.prompt)
                output = result.get("content", "") if isinstance(result, dict) else str(result)
                tokens = float(result.get("total_tokens", 0) if isinstance(result, dict) else 0)
                trace = _trace_from_result(result)
                total_tokens += tokens
                is_correct = self.dataset.evaluate(output, p)
            except Exception as exc:
                output = ""
                is_correct = False
                tokens = 0
                trace = []
                failure_cases.append(
                    CaseCapture(
                        problem_id=p.id,
                        prompt=p.prompt,
                        output=str(exc),
                        ground_truth=str(p.ground_truth)[:400],
                        is_correct=False,
                        tokens=0,
                        trace=[],
                    )
                )
                continue

            capture = CaseCapture(
                problem_id=p.id,
                prompt=p.prompt,
                output=output,
                ground_truth=str(p.ground_truth)[:400],
                is_correct=is_correct,
                tokens=tokens,
                trace=trace,
            )
            if is_correct:
                correct += 1
                if len(success_cases) < SUCCESS_SAMPLES:
                    success_cases.append(capture)
            else:
                if len(failure_cases) < MAX_FAILURE_SAMPLES:
                    failure_cases.append(capture)

        pass_at_1 = correct / len(problems) if problems else 0.0
        score = pass_at_1 - TOKEN_PENALTY * total_tokens
        return EvaluationResult(
            pass_at_1=pass_at_1,
            token_cost=total_tokens,
            score=score,
            success_cases=success_cases,
            failure_cases=failure_cases,
        )


# =======================
# Module C. Symptom Diagnoser
# =======================
class SymptomDiagnoser:
    def __init__(self, llm: Optional[VLLMWrapper]):
        self.llm = llm

    def _heuristics(self, failures: List[CaseCapture]) -> str:
        if not failures:
            return "No failures sampled; diagnosis skipped."

        buckets = []
        has_output_info = False
        for c in failures:
            # Check if we have actual output (from local eval) or just error (from server)
            if c.output and not c.output.startswith("Error:"):
                has_output_info = True
                # Local evaluation with full output
                if not c.output.strip():
                    buckets.append("Empty output → likely timeout or model silence.")
                elif "Traceback" in c.output or "Error" in c.output:
                    buckets.append("Runtime error string detected → code may be malformed.")
                elif "assert" in c.prompt.lower():
                    buckets.append("Possible format mismatch vs expected assertion format.")
                else:
                    buckets.append("Wrong answer → reasoning or formatting gap.")
            elif c.output and c.output.startswith("Error:"):
                # Server evaluation: error field indicates execution exception
                error_msg = c.output.split("Error:")[-1].strip()
                if "timeout" in error_msg.lower():
                    buckets.append("Timeout detected → workflow may be too slow or stuck.")
                elif "exception" in error_msg.lower() or "traceback" in error_msg.lower():
                    buckets.append("Runtime error detected → code may be malformed.")
                else:
                    buckets.append(f"Execution error → {error_msg[:50]}")
            else:
                # Server evaluation: correct=False but no error (just wrong answer)
                buckets.append("Wrong answer → server evaluation provides limited diagnosis info.")
        
        # Deduplicate while preserving order
        deduped = []
        for b in buckets:
            if b not in deduped:
                deduped.append(b)
        
        result = "; ".join(deduped[:4])
        if not has_output_info and any("Wrong answer" in b for b in deduped):
            result += " [Note: Server eval doesn't provide output details for wrong answers]"
        
        return result

    def diagnose(self, eval_result: EvaluationResult) -> str:
        failures = eval_result.failure_cases
        if not self.llm:
            return self._heuristics(failures)

        if not failures:
            return "No failures sampled; model perfect on sampled set."

        # Check if we have detailed output info (local eval) or limited info (server eval)
        has_detailed_info = any(c.output and not c.output.startswith("Error:") and c.output.strip() for c in failures)
        
        if not has_detailed_info:
            # Server evaluation: limited info, use heuristics only
            # LLM diagnosis requires actual output to analyze why answers are wrong
            return self._heuristics(failures)

        # Local evaluation: has detailed output, can use LLM for deeper analysis
        condensed = []
        for c in failures:
            trace_hint = ""
            if c.trace:
                culprit = c.trace[-1]
                trace_hint = f" | last_step={culprit.get('role')}->{culprit.get('output','')[:50]}"
            condensed.append(
                f"ID={c.problem_id} | prompt[:60]={c.prompt[:60]} | output[:80]={c.output[:80]} | gt[:40]={c.ground_truth[:40]}{trace_hint}"
            )
        user_prompt = f"""You are a troubleshooting resident.
Infer likely failure causes from black-box outputs.
Cases (truncated):
{chr(10).join(condensed)}

Return 3 bullet symptoms. Identify which step likely caused the failure based on the trace and whether planner vs coder mismatch occurred."""
        try:
            resp = self.llm.generate(system_prompt="Summarize failure symptoms compactly.", user_content=user_prompt)
            return resp.get("content", "").strip() or self._heuristics(failures)
        except Exception:
            return self._heuristics(failures)

    def analyze_success(self, eval_result: EvaluationResult) -> str:
        """
        Analyze why the workflow succeeded based on success_cases.
        Returns a concise 'Winning Trait' summary.
        """
        successes = eval_result.success_cases
        if not successes:
            return "No success cases yet."

        if not self.llm:
            return "Workflow has non-zero pass rate (heuristic)."

        # Sample up to 3 successes for analysis
        condensed = []
        for c in successes[:3]:
            trace_hint = ""
            if c.trace:
                trace_summary = " -> ".join([t.get("role", "Unknown") for t in c.trace])
                trace_hint = f" | flow={trace_summary}"
            condensed.append(
                f"ID={c.problem_id} | prompt[:30]={c.prompt[:30]}...{trace_hint}"
            )
        case_str = "\n".join(condensed)

        user_prompt = f"""You are an AI Optimization Analyst.
Analyze these SUCCESSFUL execution traces of an agent workflow.

Cases:
{case_str}

Identify the ONE structural feature (Winning Trait) that likely contributed to the success.
(e.g., "The Planner at the start correctly decomposed the problem", "The Reviewer loop fixed the edge case")

Output ONE concise sentence starting with "Winning Trait:".
"""
        try:
            resp = self.llm.generate(
                system_prompt="Identify the key structural advantage.",
                user_content=user_prompt,
            )
            content = resp.get("content", "").strip()
            if "Winning Trait:" in content:
                return content
            return f"Winning Trait: {content}"
        except Exception:
            return "Winning Trait: Robust performance on sampled cases."


# =======================
# Module D. Insight Journal Manager
# =======================
class InsightJournal:
    def __init__(self, run_id: str):
        ensure_dir(JOURNAL_DIR)
        self.run_id = run_id
        self.entries: List[JournalEntry] = []
        self.journal_path = Path(JOURNAL_DIR) / f"{run_id}.txt"

    def add_entry(self, entry: JournalEntry):
        self.entries.append(entry)
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(
                f"Round {entry.round_idx}: action={entry.action} "
                f"score {entry.score_before:.4f}->{entry.score_after:.4f} "
                f"tokens {entry.token_before:.1f}->{entry.token_after:.1f} "
                f"lesson={entry.lesson} diagnosis={entry.diagnosis}\n"
            )

    def render_prompt(self, limit: int = 8) -> str:
        tail = self.entries[-limit:] if limit else self.entries
        lines = []
        for e in tail:
            lines.append(
                f"R{e.round_idx}: {e.action} | Δscore={e.delta_score():+.4f} "
                f"Δtoken={e.delta_token():+.1f} | lesson={e.lesson}"
            )
        return "\n".join(lines) if lines else "No history yet."


# =======================
# Architect LLM (vLLM wrapper)
# =======================
class ArchitectLLM:
    def __init__(self, model: Optional[str], temperature: float = 0.35, max_tokens: int = 400):
        self.client = VLLMClient(model=model, default_temperature=temperature, default_max_tokens=max_tokens)
        self.model = self.client.model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_topologies(self, user_prompt: str, n: int, diversity_indices: List[int] = None) -> Tuple[List[str], List[int]]:
        """Generate n topology strings using batch_complete.
        
        Args:
            user_prompt: Base prompt for all proposals
            n: Number of proposals to generate
            diversity_indices: Indices (0-based) of proposals that should use diversity prompt
        """
        diversity_indices = diversity_indices or []
        diversity_system = "Design minimal agent chains using given roles. Return arrow syntax only. IMPORTANT: Think outside the box. Avoid local minima by exploring fundamentally different approaches. Consider unconventional role combinations, alternative problem-solving strategies, or roles that are rarely used together. Break free from incremental improvements."
        standard_system = "Design minimal agent chains using given roles. Return arrow syntax only."
        
        messages_list = []
        for i in range(n):
            system_content = diversity_system if i in diversity_indices else standard_system
            messages_list.append([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ])
        try:
            results = asyncio.run(self.client.batch_complete(
                messages_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ))
            topologies: List[str] = []
            tokens: List[int] = []
            for idx, r in enumerate(results):
                if r.status == "COMPLETED":
                    topologies.append((r.content or "").strip())
                    tokens.append(int(r.total_tokens))
                else:
                    # Log failure for debugging
                    error_msg = r.error or "Unknown error"
                    print(f"[ArchitectLLM] batch[{idx}] failed: {error_msg}")
                    topologies.append("")
                    tokens.append(0)
            return topologies, tokens
        except Exception as exc:
            print(f"[ArchitectLLM] batch_complete exception: {exc}")
            return ["" for _ in range(n)], [0 for _ in range(n)]


# =======================
# Module E. Adaptive Architect Agent
# =======================
class AdaptiveArchitectAgent:
    def __init__(
        self,
        llm: Optional[VLLMWrapper],
        task_name: str,
        architect_llm: Optional["ArchitectLLM"] = None,
        proposals_per_round: int = PROPOSALS_PER_ROUND,
    ):
        self.llm = llm
        self.task_name = task_name
        self.architect_llm = architect_llm
        self.proposals_per_round = max(1, proposals_per_round)

    def _fallback(self, best_roles: List[str], mode: str) -> List[str]:
        roles = best_roles[:] if best_roles else [random.choice(ROLE_LIST)]
        if mode == "EFFICIENCY" and len(roles) > 1:
            roles.pop()
        elif mode == "PERFORMANCE":
            roles.append(random.choice([r for r in ROLE_LIST if r not in roles] or ROLE_LIST))
        return roles

    def _build_prompt(
        self,
        mode: str,
        sampled_workflows: List[Tuple[BlockWorkflow, EvaluationResult]],
        bias: BiasReport,
        journal: str,
        avg_length: float = None,
        target_length: int = None,
    ) -> str:
        agent_pool = ", ".join(ROLE_LIST[:40])
        
        # Build workflow comparison section
        workflows_section = ""
        for idx, (wf, eval_res) in enumerate(sampled_workflows, 1):
            workflows_section += f"\nWorkflow {idx}:\n"
            workflows_section += f"  Topology: {wf.workflow_to_string()}\n"
            workflows_section += f"  Performance: pass@1={eval_res.pass_at_1:.4f}, tokens={eval_res.token_cost:.1f}\n"
            workflows_section += f"  Success cases: {len(eval_res.success_cases)}\n"
            workflows_section += f"  Failure cases: {len(eval_res.failure_cases)}\n"
        
        length_guidance = ""
        if avg_length is not None and target_length is not None:
            length_guidance = f"- Average workflow length in front: {avg_length:.1f} nodes. Target: {target_length} nodes."
            else:
            length_guidance = "Keep length reasonable; avoid unnecessary bloat."
        
        return f"""You are the Adaptive Architect Agent.
Mode: {mode}
Task: {self.task_name}

=== Context Analysis ===
[Dataset Bias]: {bias.summary}
[Previous Rounds History - Learn from Past Results]:
{journal}

IMPORTANT: The history above shows what changes were made in previous rounds and their outcomes (Δscore, Δtoken, lesson). Use this to understand:
- Which types of changes led to improvements (positive Δscore)
- Which changes were ineffective or harmful (negative Δscore)
- Patterns in successful modifications
- Avoid repeating failed strategies

=== Sampled Workflows for Analysis ===
{workflows_section}

=== Instructions ===
1. Analyze: Compare the {len(sampled_workflows)} workflows above AND review the history of previous rounds. Identify:
   - What are the strengths of each workflow? (e.g., specific agent roles that work well)
   - What are the weaknesses? (e.g., missing capabilities, redundant steps)
   - How can we combine their strengths or fix their weaknesses?
   - What can we learn from the history? Which past changes worked well? Which didn't?

2. Plan: Design improved workflows by:
   - Combining the best elements from multiple sampled workflows (e.g., merge strong agent sequences)
   - Adding new nodes at strategic positions to address identified weaknesses
   - Removing redundant or ineffective nodes
   - Reordering nodes for better flow
   - Learning from history: apply successful patterns from previous rounds, avoid repeating failed strategies
   Use ONLY roles from: {agent_pool}

3. Generate: Create {self.proposals_per_round} improved workflow topologies that:
   - Leverage the strengths of the sampled workflows
   - Incorporate lessons learned from previous rounds (see History section)
   - Build upon successful past modifications while avoiding known pitfalls

Rules:
- Keep it 2-6 nodes per workflow.
- Do NOT add more than 4 nodes at once.
- {length_guidance}
- Do NOT output python code.
- You can create workflows that combine elements from multiple sampled workflows.
- Reference the History section to inform your design decisions.

=== Output Format ===
For each of the {self.proposals_per_round} proposals, provide:
Analysis: <Your reasoning for this proposal, including what you learned from history>
Plan: <Brief description of changes>
Topology: Role1 -> Role2 -> Role3

Separate each proposal with "---" (three dashes).
"""

    def propose(
        self,
        sampled_workflows: List[Tuple[BlockWorkflow, EvaluationResult]],
        bias: BiasReport,
        journal: str,
        front_lengths: List[int] = None,
    ) -> List[Tuple[BlockWorkflow, str]]:
        # Determine mode based on average pass@1 of sampled workflows
        avg_pass_at_1 = sum(eval_res.pass_at_1 for _, eval_res in sampled_workflows) / len(sampled_workflows)
        mode = "EFFICIENCY" if avg_pass_at_1 >= 0.999 else "PERFORMANCE"
        
        # Get roles from all sampled workflows for fallback
        all_roles = []
        for wf, _ in sampled_workflows:
            roles = extract_roles(wf.workflow_to_string())
            all_roles.extend(roles)
        best_roles = list(dict.fromkeys(all_roles))  # Remove duplicates while preserving order
        if not best_roles:
            best_roles = [random.choice(ROLE_LIST)]

        # Calculate average length and target length for balanced exploration
        avg_length = None
        target_length = None
        if front_lengths and len(front_lengths) > 0:
            avg_length = sum(front_lengths) / len(front_lengths)
            # Calculate average node count of sampled workflows
            avg_sampled_nodes = sum(len(wf.blocks) if wf.blocks else 0 for wf, _ in sampled_workflows) / len(sampled_workflows)
            if avg_sampled_nodes > avg_length:
                target_length = max(2, int(avg_length))  # Shorten toward average
            elif avg_sampled_nodes < avg_length:
                target_length = min(6, int(avg_length) + 1)  # Expand toward average
            else:
                target_length = int(avg_sampled_nodes)  # Maintain

        prompt = self._build_prompt(
            mode,
            sampled_workflows,
            bias,
            journal,
            avg_length,
            target_length,
        )

        # === vLLM-based multi-proposal ===
        if self.architect_llm:
            # Use diversity prompt for 2 out of 5 proposals to avoid local minima
            # Select indices for diversity (e.g., first 2 or random 2)
            num_diversity = min(2, self.proposals_per_round)
            diversity_indices = list(range(num_diversity))  # First 2 proposals use diversity prompt
            
            topo_list, tokens_list = self.architect_llm.generate_topologies(
                prompt, self.proposals_per_round, diversity_indices=diversity_indices
            )
            proposals: List[Tuple[BlockWorkflow, str]] = []
            for idx, (topo, toks) in enumerate(zip(topo_list, tokens_list)):
                if not topo or not topo.strip():
                    # Empty response, use fallback
                    roles = self._fallback(best_roles, mode)
                    wf = workflow_from_roles(self.task_name, roles)
                    proposals.append((wf, f"[{mode}] vllm_empty_{idx} fallback={roles}"))
                    continue
                
                roles = extract_roles(topo)
                if not roles:
                    # Failed to parse, use fallback but log the raw response
                    roles = self._fallback(best_roles, mode)
                    wf = workflow_from_roles(self.task_name, roles)
                    proposals.append((wf, f"[{mode}] vllm_parse_fail_{idx} raw='{topo[:50]}' fallback={roles}"))
                else:
                    wf = workflow_from_roles(self.task_name, roles)
                    proposals.append((wf, f"[{mode}] vllm roles={roles} toks={toks}"))
            if proposals:
                return proposals

        # === Single proposal via VLLMWrapper ===
        if not self.llm:
            roles = self._fallback(best_roles, mode)
            return [(workflow_from_roles(self.task_name, roles), f"[Fallback-{mode}] heuristic edit")]

        try:
            resp = self.llm.generate(
                system_prompt="You are a system architect. You MUST provide Analysis before Topology for each proposal.",
                user_content=prompt,
                max_tokens=2000,  # Increased for multiple proposals
            )
            content = (resp.get("content") or "").strip()
            
            # Parse multiple proposals separated by "---"
            proposal_sections = content.split("---")
            proposals: List[Tuple[BlockWorkflow, str]] = []
            max_allowed_nodes = 6  # General limit
            
            for idx, section in enumerate(proposal_sections):
                section = section.strip()
                if not section:
                    continue
                
                roles = extract_roles(section)
            if not roles:
                    # Fallback for this proposal
                roles = self._fallback(best_roles, mode)
                    wf = workflow_from_roles(self.task_name, roles)
                    proposals.append((wf, f"[{mode}] llm_parse_fail_{idx} fallback={roles}"))
                    continue
                
            wf = workflow_from_roles(self.task_name, roles)
            node_count = len(wf.blocks) if wf.blocks else 0
                
            # Bloat prevention: reject if exceeds max_allowed_nodes
            if node_count > max_allowed_nodes:
                trimmed_roles = roles[:max_allowed_nodes]
                wf = workflow_from_roles(self.task_name, trimmed_roles)
                    action = f"[{mode}] llm_bloat_trimmed_{idx} from_{node_count}_to_{max_allowed_nodes} roles={trimmed_roles}"
            else:
                    reasoning_snippet = section.split("Topology:")[0].replace("\n", " ")[:100] if "Topology:" in section else ""
                    action = f"[{mode}] llm_proposed_{idx} roles={roles} | analysis='{reasoning_snippet}...'"
                proposals.append((wf, action))
                
                # Limit to requested number of proposals
                if len(proposals) >= self.proposals_per_round:
                    break
            
            # If no proposals were parsed, use fallback
            if not proposals:
            roles = self._fallback(best_roles, mode)
                proposals = [(workflow_from_roles(self.task_name, roles), f"[{mode}] llm_no_proposals_fallback")]
            
            return proposals[:self.proposals_per_round]
        except Exception as e:
            # Fallback on error
            roles = self._fallback(best_roles, mode)
            return [(workflow_from_roles(self.task_name, roles), f"[{mode}] llm_error_fallback: {str(e)[:50]}")]


# =======================
# Module F. Greedy Rollback Controller
# =======================
class ParetoFrontController:
    """
    Maintains a Pareto front of workflows and samples from it for improvement.
    Uses multi-objective optimization: minimize tokens, maximize pass@1.
    """
    def __init__(self, patience: int, max_front_size: int = 20, buffer_size: int = 10):
        self.patience = patience
        self.max_front_size = max_front_size
        self.buffer_size = buffer_size
        self.no_improve_steps = 0
        # Pareto front: List[Tuple[BlockWorkflow, EvaluationResult]]
        self.front: List[Tuple[BlockWorkflow, EvaluationResult]] = []
        # Buffer: List[Tuple[BlockWorkflow, EvaluationResult]] - veterans removed from front
        self.buffer: List[Tuple[BlockWorkflow, EvaluationResult]] = []
        # Archive: List[Tuple[BlockWorkflow, EvaluationResult]] - Front 0 individuals removed due to size limit
        self.archive: List[Tuple[BlockWorkflow, EvaluationResult]] = []
        # Track seen workflows to prevent duplicates
        self.seen_workflows: set = set()
    
    def _is_dominated(self, candidate: EvaluationResult, front: List[EvaluationResult]) -> bool:
        """Check if candidate is dominated by any member of the front."""
        # Objectives: [token_cost, -pass_at_1] (minimize both)
        candidate_obj = np.array([candidate.token_cost, -candidate.pass_at_1])
        for member in front:
            member_obj = np.array([member.token_cost, -member.pass_at_1])
            # member dominates candidate if member_obj <= candidate_obj (all) and < (any)
            if np.all(member_obj <= candidate_obj) and np.any(member_obj < candidate_obj):
                return True
        return False
    
    def _remove_dominated(self, front: List[Tuple[BlockWorkflow, EvaluationResult]]) -> List[Tuple[BlockWorkflow, EvaluationResult]]:
        """Remove dominated solutions from the front."""
        if not front:
            return []
        
        # Convert to objectives for sorting
        objs = np.array([[eval_result.token_cost, -eval_result.pass_at_1] for _, eval_result in front])
        fronts = non_dominated_sort(objs)
        
        # Keep only the first front (non-dominated solutions)
        return [front[idx] for idx in fronts[0]]
    
    def add_candidate(self, workflow: BlockWorkflow, eval_result: EvaluationResult) -> bool:
        """
        Add candidate to Pareto front if it's not dominated and not a duplicate.
        Returns True if added, False otherwise.
        """
        # Check for duplicates using canonicalized workflow string
        canonical_wf = canonicalize_workflow(workflow)
        if canonical_wf in self.seen_workflows:
            return False  # Duplicate workflow, skip
        
        front_evals = [eval_res for _, eval_res in self.front]
        
        # Check if candidate is dominated
        if self._is_dominated(eval_result, front_evals):
            return False
        
        # IMPORTANT: Reset was_in_buffer flag when candidate is added to front
        # They earned their place back, so give them another buffer chance if they fall out again
        eval_result.was_in_buffer = False
        
        # Add to seen set and front
        self.seen_workflows.add(canonical_wf)
        self.front.append((workflow, eval_result))
        
        # Remove any existing solutions that are now dominated by candidate
        self.front = self._remove_dominated(self.front)
        
        # Limit front size using crowding distance
        # IMPORTANT: When removing individuals from front, add veterans (generation_age > 0) to buffer
        if len(self.front) > self.max_front_size:
            objs = np.array([[eval_res.token_cost, -eval_res.pass_at_1] for _, eval_res in self.front])
            front_indices = list(range(len(self.front)))
            dist = crowding_distance(objs, front_indices)
            # Sort by crowding distance (higher is better)
            sorted_indices = sorted(range(len(dist)), key=lambda i: dist[i], reverse=True)
            
            # Separate into survivors and removed
            survivors = [self.front[i] for i in sorted_indices[:self.max_front_size]]
            removed = [self.front[i] for i in sorted_indices[self.max_front_size:]]
            
            # Archive all removed Front 0 individuals (they were in Pareto front)
            from copy import deepcopy
            for wf, eval_res in removed:
                archived = (deepcopy(wf), deepcopy(eval_res))
                self.archive.append(archived)
            
            # Add veterans (generation_age > 0) from removed to buffer
            # IMPORTANT: Only add if not already in buffer (was_in_buffer=False)
            # Buffer purpose: give ONE chance to veterans who fell out of front
            buffer_added_count = 0
            for wf, eval_res in removed:
                if eval_res.generation_age > 0 and not eval_res.was_in_buffer:
                    # Sort buffer candidates by: Front level (they were in front), generation_age, pass_at_1
                    # Since they were in front, they have high priority
                    self.buffer.append((wf, eval_res))
                    eval_res.was_in_buffer = True  # Mark as having been in buffer
                    buffer_added_count += 1
            
            # Limit buffer size, prioritizing by generation_age and pass_at_1
            if len(self.buffer) > self.buffer_size:
                self.buffer.sort(
                    key=lambda x: (x[1].generation_age, x[1].pass_at_1),
                    reverse=True
                )
                removed_from_buffer = self.buffer[self.buffer_size:]
                self.buffer = self.buffer[:self.buffer_size]
            
            self.front = survivors
        
        return True
    
    def sample_workflow(self) -> Tuple[BlockWorkflow, EvaluationResult]:
        """
        Sample a workflow from the Pareto front.
        Prefers solutions with higher crowding distance (more diverse).
        """
        if not self.front:
            raise ValueError("Pareto front is empty")
        
        if len(self.front) == 1:
            return self.front[0]
        
        # For small fronts (2-3 items), use uniform sampling to avoid crowding distance issues
        if len(self.front) <= 3:
            idx = np.random.choice(len(self.front))
            return self.front[idx]
        
        try:
            # Calculate crowding distances
            objs = np.array([[eval_res.token_cost, -eval_res.pass_at_1] for _, eval_res in self.front])
            front_indices = list(range(len(self.front)))
            dist = crowding_distance(objs, front_indices)
            
            # Handle edge cases: NaN, inf, or all zeros
            dist = np.array(dist)
            if len(dist) == 0 or len(dist) != len(self.front):
                # Fallback to uniform sampling
                idx = np.random.choice(len(self.front))
                return self.front[idx]
            
            # Check for invalid values
            if np.all(np.isnan(dist)) or np.all(np.isinf(dist)) or np.all(dist == 0):
                # Fallback to uniform sampling
                idx = np.random.choice(len(self.front))
                return self.front[idx]
            
            # Replace NaN/inf with 0, then add epsilon
            dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
            dist = np.maximum(dist, 0.0)  # Ensure non-negative
            dist = dist + 1e-6  # Add epsilon to avoid zero probabilities
            
            # Normalize probabilities
            dist_sum = dist.sum()
            if dist_sum == 0 or np.isnan(dist_sum) or np.isinf(dist_sum) or dist_sum < 1e-10:
                # Fallback to uniform sampling
                idx = np.random.choice(len(self.front))
                return self.front[idx]
            
            probs = dist / dist_sum
            
            # Final check: ensure no NaN or invalid values in probabilities
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)) or np.any(probs < 0):
                idx = np.random.choice(len(self.front))
                return self.front[idx]
            
            # Ensure probabilities sum to 1 (within tolerance)
            prob_sum = probs.sum()
            if abs(prob_sum - 1.0) > 1e-6:
                # Renormalize
                probs = probs / prob_sum
            
            idx = np.random.choice(len(self.front), p=probs)
            return self.front[idx]
        except Exception as e:
            # Any error in crowding distance calculation -> fallback to uniform
            idx = np.random.choice(len(self.front))
            return self.front[idx]
    
    def get_best_workflow(self) -> Tuple[BlockWorkflow, EvaluationResult]:
        """
        Get the best workflow from the front (highest pass@1).
        """
        if not self.front:
            raise ValueError("Pareto front is empty")
        
        best_idx = max(range(len(self.front)), key=lambda i: self.front[i][1].pass_at_1)
        return self.front[best_idx]
    
    def accept(self, best: EvaluationResult, candidate: EvaluationResult) -> bool:
        """
        Check if candidate should be accepted (not dominated by best).
        This is used for tracking improvement, but actual acceptance is handled by add_candidate.
        """
        # Candidate is better if it's not dominated by best
        candidate_obj = np.array([candidate.token_cost, -candidate.pass_at_1])
        best_obj = np.array([best.token_cost, -best.pass_at_1])
        
        # Candidate is better if it dominates best, or is not dominated by best
        candidate_dominates = np.all(candidate_obj <= best_obj) and np.any(candidate_obj < best_obj)
        not_dominated = not (np.all(best_obj <= candidate_obj) and np.any(best_obj < candidate_obj))
        
        improved = candidate_dominates or (not_dominated and candidate.pass_at_1 > best.pass_at_1)
        
        if improved:
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1
        
        return improved
    
    def should_stop(self) -> bool:
        return self.no_improve_steps >= self.patience
    
    def get_front_size(self) -> int:
        return len(self.front)


# =======================
# Main HDLO loop
# =======================
def run_hdlo(args):
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log = log_writer(run_id)

    # Setup directories
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(GRAPH_DIR)

    llm = None
    architect_llm = None
    if not args.no_llm:
        try:
            llm = VLLMWrapper(model=args.model, temperature=args.temperature)
            log(f"[Init] VLLM wrapper ready model={llm.model}")
        except Exception as exc:
            log(f"[WARN] VLLM wrapper unavailable, switching to heuristic for bias/diag: {exc}")
            llm = None
        try:
            architect_llm = ArchitectLLM(model=args.vllm_model, temperature=args.temperature, max_tokens=args.vllm_max_tokens)
            log(f"[Init] Architect vLLM ready model={architect_llm.model}")
        except Exception as exc:
            log(f"[WARN] Architect vLLM unavailable: {exc}")
            architect_llm = None
    else:
        log("[Init] LLM disabled (--no-llm). Using heuristic edits only.")

    dataset = get_dataset(args.task)
    bias_analyzer = DatasetBiasAnalyzer(llm)
    evaluator = BlackBoxEvaluator(
        dataset=dataset,
        task_name=args.task,
        num_problems=args.num_problems,
        seed=args.eval_seed,
        use_server=not args.no_server,
        server_url=args.server_url,
        batch_size=args.batch_size,
        log=log,
    )
    diagnoser = SymptomDiagnoser(llm)
    architect = AdaptiveArchitectAgent(
        llm=llm,
        task_name=args.task,
        architect_llm=architect_llm,
        proposals_per_round=args.proposals_per_round,
    )
    buffer_size = getattr(args, 'buffer_size', 10)
    max_front_size = getattr(args, 'max_front_size', 15)
    controller = ParetoFrontController(patience=args.patience, max_front_size=max_front_size, buffer_size=buffer_size)
    journal = InsightJournal(run_id)
    # Track all evaluated workflows (including dominated ones) for plotting
    all_evaluated: List[Tuple[BlockWorkflow, EvaluationResult]] = []

    # Initial workflow: simple 2-role chain
    init_roles = random.sample(ROLE_LIST, 2) if len(ROLE_LIST) >= 2 else ROLE_LIST
    current_workflow = workflow_from_roles(args.task, init_roles)
    # Note: seen_workflows will be updated in add_candidate
    log(f"[Init] Starting workflow: {workflow_to_string(current_workflow)}")

    bias_report = bias_analyzer.analyze(dataset, seed=args.eval_seed, task_name=args.task)
    log(f"[Bias] {bias_report.summary}")

    # Evaluate baseline
    import time
    round_start = time.time()
    log(f"[Round 0] Starting evaluation...")
    eval_start = time.time()
    best_eval = evaluator.evaluate(current_workflow)
    best_eval.eval_count = 1  # Initial evaluation
    best_eval.total_evaluated_problems = evaluator.num_problems  # Set initial total evaluated problems
    eval_time = time.time() - eval_start
    round_time = time.time() - round_start
    log(
        f"[Round 0] pass@1={best_eval.pass_at_1:.4f} tokens={best_eval.token_cost:.1f} "
        f"score={best_eval.score:.4f} | eval_time={eval_time:.2f}s total_time={round_time:.2f}s"
    )
    journal.add_entry(
        JournalEntry(
            round_idx=0,
            action="bootstrap",
            score_before=0.0,
            score_after=best_eval.score,
            token_before=0.0,
            token_after=best_eval.token_cost,
            lesson="Baseline measurement",
            diagnosis="N/A",
        )
    )
    # Add initial workflow to Pareto front
    added = controller.add_candidate(current_workflow, best_eval)
    all_evaluated.append((current_workflow, best_eval))
    if added:
    log(f"[Init] Added to Pareto front. Front size: {controller.get_front_size()}")
    else:
        log(f"[Init] WARNING: Initial workflow not added to front (duplicate or dominated). Front size: {controller.get_front_size()}")
    
    # Convert all evaluated workflows to plot format
    all_population = [
        {"workflow": wf, "fitness": {"pass_at_k": eval_res.pass_at_1, "token": eval_res.token_cost}}
        for wf, eval_res in all_evaluated
    ]
    
    # IMPORTANT: Find Pareto front and buffer individuals in all_population by matching workflows
    # because front_population may reference different objects than all_population
    # This ensures plot_pareto can correctly match objects by id()
    front_workflows = {wf for wf, _ in controller.front}
    buffer_workflows = {wf for wf, _ in controller.buffer}
    front_population = []
    buffer_population = []
    for entry in all_population:
        if entry["workflow"] in front_workflows:
            front_population.append(entry)
        elif entry["workflow"] in buffer_workflows:
            buffer_population.append(entry)
    
    save_checkpoint_csv(
        all_population,
        f"{run_id}_r0",
        save_dir=CHECKPOINT_DIR,
    )
    plot_pareto(
        all_population,
        file_name=f"{run_id}_r0",
        save_dir=GRAPH_DIR,
        survivors=front_population,
        buffer_list=buffer_population,
        pareto_front=front_population,  # controller.front is already the actual Pareto front
    )

    for round_idx in range(1, args.max_rounds + 1):
        round_start = time.time()
        log(f"\n[Round {round_idx}] === Start ===")
        
        # IMPORTANT: Increment generation_age for all front members at the start of each round
        # This matches ga.py logic: survivors' generation_age is incremented at generation start
        # Front members are "survivors" - they survived to the next round
        for wf, eval_res in controller.front:
            eval_res.generation_age += 1
        
        # Re-evaluate all workflows in Pareto front AND buffer (if eval_count < max_eval_iter)
        # IMPORTANT: All front members are "survivors" - they survived to the next round
        # Re-evaluate ALL front members (not just generation_age > 0) up to max_eval_iter times
        # This matches ga.py: all individuals in population with generation_age > 0 and eval_count < max_eval_iter
        # In hdlo, front members are the "survivors", so evaluate all of them
        max_eval_iter = getattr(args, 'max_eval_iter', 4)
        
        # Collect workflows from front that need re-evaluation
        # All front members are survivors, so evaluate all of them (not just generation_age > 0)
        front_to_reeval = []
        if controller.front:
            for wf, eval_res in controller.front:
                # Front members are survivors - evaluate all of them up to max_eval_iter
                # generation_age was just incremented, so it will be > 0 for all except newly added ones
                # But we want to evaluate all survivors regardless of generation_age
                if eval_res.eval_count < max_eval_iter:
                    front_to_reeval.append((wf, eval_res))
        
        # Collect workflows from buffer that need re-evaluation
        buffer_to_reeval = []
        if controller.buffer:
            for wf, eval_res in controller.buffer:
                if eval_res.eval_count < max_eval_iter:
                    buffer_to_reeval.append((wf, eval_res))
        
        # Combine all workflows that need re-evaluation
        all_to_reeval = front_to_reeval + buffer_to_reeval
        
        if all_to_reeval:
            log(f"[Round {round_idx}] Re-evaluating {len(all_to_reeval)} workflows ({len(front_to_reeval)} from front, {len(buffer_to_reeval)} from buffer) (max_eval_iter={max_eval_iter})...")
            workflows_to_reeval = [wf for wf, _ in all_to_reeval]
            new_evals = evaluator.evaluate_batch(workflows_to_reeval)
            
            # Update evaluation results using weighted average and increment eval_count
            num_problems_this_eval = evaluator.num_problems
            for (wf, old_eval), new_eval in zip(all_to_reeval, new_evals):
                # Update eval_result in place using weighted average
                if old_eval.total_evaluated_problems > 0:
                    # Weighted average: (old_total * old_value + new_num * new_value) / (old_total + new_num)
                    total_problems = old_eval.total_evaluated_problems + num_problems_this_eval
                    old_eval.pass_at_1 = (old_eval.total_evaluated_problems * old_eval.pass_at_1 + num_problems_this_eval * new_eval.pass_at_1) / total_problems
                    old_eval.token_cost = (old_eval.total_evaluated_problems * old_eval.token_cost + num_problems_this_eval * new_eval.token_cost) / total_problems
                    old_eval.total_evaluated_problems = total_problems
                else:
                    # First evaluation
                    old_eval.pass_at_1 = new_eval.pass_at_1
                    old_eval.token_cost = new_eval.token_cost
                    old_eval.total_evaluated_problems = num_problems_this_eval
                
                # Update score and other fields
                old_eval.score = old_eval.pass_at_1 - TOKEN_PENALTY * old_eval.token_cost
                old_eval.eval_count += 1
                # NOTE: generation_age was already incremented at the start of the round for front members
                # For buffer members, increment here since they weren't incremented at round start
                if (wf, old_eval) in buffer_to_reeval:
                    old_eval.generation_age += 1
                old_eval.success_cases = new_eval.success_cases
                old_eval.failure_cases = new_eval.failure_cases
                
                # Re-sort front after re-evaluation
                controller.front = controller._remove_dominated(controller.front)
                if len(controller.front) > controller.max_front_size:
                    objs = np.array([[eval_res.token_cost, -eval_res.pass_at_1] for _, eval_res in controller.front])
                    front_indices = list(range(len(controller.front)))
                    dist = crowding_distance(objs, front_indices)
                    # Sort by crowding distance (higher is better)
                    sorted_indices = sorted(range(len(dist)), key=lambda i: dist[i], reverse=True)
                    
                    # Separate into survivors and removed
                    survivors = [controller.front[i] for i in sorted_indices[:controller.max_front_size]]
                    removed = [controller.front[i] for i in sorted_indices[controller.max_front_size:]]
                    
                    # Add veterans (generation_age > 0) from removed to buffer
                    # Archive all removed Front 0 individuals (they were in Pareto front)
                    for wf, eval_res in removed:
                        # Archive all removed Front 0 individuals
                        from copy import deepcopy
                        archived = (deepcopy(wf), deepcopy(eval_res))
                        controller.archive.append(archived)
                        
                        # Add veterans to buffer
                        # IMPORTANT: Only add if not already in buffer (was_in_buffer=False)
                        # Buffer purpose: give ONE chance to veterans who fell out of front
                        if eval_res.generation_age > 0 and not eval_res.was_in_buffer:
                            controller.buffer.append((wf, eval_res))
                            eval_res.was_in_buffer = True  # Mark as having been in buffer
                            log(f"[Round {round_idx}] Added to buffer: {workflow_to_string(wf)} | "
                                f"pass@1={eval_res.pass_at_1:.4f} gen_age={eval_res.generation_age}")
                    
                    # Limit buffer size, prioritizing by generation_age and pass_at_1
                    if len(controller.buffer) > controller.buffer_size:
                        controller.buffer.sort(
                            key=lambda x: (x[1].generation_age, x[1].pass_at_1),
                            reverse=True
                        )
                        removed_from_buffer = controller.buffer[controller.buffer_size:]
                        controller.buffer = controller.buffer[:controller.buffer_size]
                        if removed_from_buffer:
                            log(f"[Round {round_idx}] Removed {len(removed_from_buffer)} from buffer (size limit)")
                    
                    controller.front = survivors
                    
                    if len(removed) > 0:
                        log(f"[Round {round_idx}] Archived {len(removed)} Front 0 individuals (Pareto front size limit)")
                        log(f"[Round {round_idx}] Buffer size: {len(controller.buffer)}/{controller.buffer_size}")
        
        # Sample 3 workflows from Pareto front for improvement
        if len(controller.front) == 0:
            log(f"[Round {round_idx}] WARNING: Pareto front is empty. Skipping proposal generation.")
            continue
        
        num_samples = min(3, len(controller.front))
        sampled_workflows = []
        for _ in range(num_samples):
        sampled_wf, sampled_eval = controller.sample_workflow()
            sampled_workflows.append((sampled_wf, sampled_eval))
        
        log(f"[Round {round_idx}] Sampled {len(sampled_workflows)} workflow(s) from front:")
        for idx, (wf, eval_res) in enumerate(sampled_workflows, 1):
            log(f"[Round {round_idx}]   [{idx}] {workflow_to_string(wf)} | "
                f"pass@1={eval_res.pass_at_1:.4f} tokens={eval_res.token_cost:.1f}")
        log(f"[Round {round_idx}] Front size: {controller.get_front_size()}")

        # Calculate average workflow length in front for balanced exploration
        front_lengths = [
            len(wf.blocks) if wf.blocks else 0
            for wf, _ in controller.front
        ]
        avg_length = sum(front_lengths) / len(front_lengths) if front_lengths else 0
        
        log(f"[Round {round_idx}] Front length stats: avg={avg_length:.1f} (front sizes: {front_lengths})")
        
        proposal_start = time.time()
        raw_proposals = architect.propose(
            sampled_workflows=sampled_workflows,
            bias=bias_report,
            journal=journal.render_prompt(limit=6),
            front_lengths=front_lengths,
        )
        proposal_time = time.time() - proposal_start
        
        # Filter out duplicate proposals
        proposals = []
        duplicates_skipped = 0
        for wf, action in raw_proposals:
            canonical_wf = canonicalize_workflow(wf)
            if canonical_wf in controller.seen_workflows:
                duplicates_skipped += 1
                log(f"[Round {round_idx}] Skipped duplicate proposal: {workflow_to_string(wf)}")
                continue
            proposals.append((wf, action))
        
        # Log proposals immediately after generation
        avg_sampled_nodes = sum(len(wf.blocks) if wf.blocks else 0 for wf, _ in sampled_workflows) / len(sampled_workflows) if sampled_workflows else 0
        log(f"[Round {round_idx}] Proposed {len(proposals)} candidate(s) (skipped {duplicates_skipped} duplicate(s)) | proposal_time={proposal_time:.2f}s")
        for idx, (wf, action) in enumerate(proposals, 1):
            node_count = len(wf.blocks) if wf.blocks else 0
            delta = node_count - avg_sampled_nodes
            log(f"[Round {round_idx}] Proposal {idx}: {workflow_to_string(wf)} | nodes={node_count} (Δ{delta:+.1f}) | {action}")

        eval_start = time.time()
        candidate_workflows = [wf for wf, _ in proposals]
        candidate_evals = evaluator.evaluate_batch(candidate_workflows)
        # Set eval_count and total_evaluated_problems for new candidates
        for eval_res in candidate_evals:
            eval_res.eval_count = 1
            eval_res.total_evaluated_problems = evaluator.num_problems  # Set initial total evaluated problems
        eval_time = time.time() - eval_start
        round_time = time.time() - round_start

        # Log all candidates with workflow length statistics
        length_changes = {"increase": 0, "decrease": 0, "same": 0}
        
        for idx, ((wf, action), eval_result) in enumerate(zip(proposals, candidate_evals)):
            node_count = len(wf.blocks) if wf.blocks else 0
            delta = node_count - avg_sampled_nodes
            delta_int = int(round(delta))  # Convert to integer for display
            if delta > 0:
                length_changes["increase"] += 1
            elif delta < 0:
                length_changes["decrease"] += 1
            else:
                length_changes["same"] += 1
            
            log(
                f"[Round {round_idx}] Candidate {idx+1}: {workflow_to_string(wf)} | "
                f"pass@1={eval_result.pass_at_1:.4f} tokens={eval_result.token_cost:.1f} "
                f"score={eval_result.score:.4f} | nodes={node_count} (Δ{delta_int:+d}) | {action}"
            )
        
        log(f"[Round {round_idx}] Length changes: +{length_changes['increase']} -{length_changes['decrease']} ={length_changes['same']} (from avg {avg_sampled_nodes:.1f} nodes)")

        # Add all candidates to Pareto front and track all evaluated
        added_count = 0
        for (wf, action), eval_result in zip(proposals, candidate_evals):
            # Track all evaluated workflows (including dominated ones)
            all_evaluated.append((wf, eval_result))
            if controller.add_candidate(wf, eval_result):
                added_count += 1
                log(f"[Round {round_idx}] Added to front: {workflow_to_string(wf)} | "
                    f"pass@1={eval_result.pass_at_1:.4f} tokens={eval_result.token_cost:.1f}")
        
        log(f"[Round {round_idx}] Added {added_count}/{len(proposals)} candidates to front. "
            f"Front size: {controller.get_front_size()}")

        # Get best workflow from front (highest pass@1)
        current_workflow, best_eval = controller.get_best_workflow()
        
        log(
            f"[Round {round_idx}] Best from front: {workflow_to_string(current_workflow)} | "
            f"pass@1={best_eval.pass_at_1:.4f} tokens={best_eval.token_cost:.1f} score={best_eval.score:.4f} | "
            f"eval_time={eval_time:.2f}s proposal_time={proposal_time:.2f}s total_time={round_time:.2f}s"
        )

        # Track improvement for stopping condition
        # Use average of sampled workflows for comparison
        avg_sampled_score = sum(eval_res.score for _, eval_res in sampled_workflows) / len(sampled_workflows) if sampled_workflows else 0
        avg_sampled_eval = EvaluationResult(
            pass_at_1=sum(eval_res.pass_at_1 for _, eval_res in sampled_workflows) / len(sampled_workflows) if sampled_workflows else 0,
            token_cost=sum(eval_res.token_cost for _, eval_res in sampled_workflows) / len(sampled_workflows) if sampled_workflows else 0,
            score=avg_sampled_score
        )
        improved = controller.accept(avg_sampled_eval, best_eval)
        lesson = "Improved" if improved else "No improvement"

        sampled_workflows_str = ", ".join([workflow_to_string(wf) for wf, _ in sampled_workflows[:2]])
        if len(sampled_workflows) > 2:
            sampled_workflows_str += f" (+{len(sampled_workflows)-2} more)"

        journal.add_entry(
            JournalEntry(
                round_idx=round_idx,
                action=f"Pareto front update (sampled {len(sampled_workflows)} workflows: {sampled_workflows_str})",
                score_before=avg_sampled_eval.score,
                score_after=best_eval.score,
                token_before=avg_sampled_eval.token_cost,
                token_after=best_eval.token_cost,
                lesson=lesson,
                diagnosis="",  # Diagnosis is now per-workflow, not aggregated
            )
        )

        # Save all evaluated workflows (entire population, not just Pareto front)
        all_population = [
            {"workflow": wf, "fitness": {"pass_at_k": eval_res.pass_at_1, "token": eval_res.token_cost}}
            for wf, eval_res in all_evaluated
        ]
        
        # IMPORTANT: Find Pareto front and buffer individuals in all_population by matching workflows
        # because front_population may reference different objects than all_population
        # This ensures plot_pareto can correctly match objects by id()
        front_workflows = {wf for wf, _ in controller.front}
        buffer_workflows = {wf for wf, _ in controller.buffer}
        front_population = []
        buffer_population = []
        for entry in all_population:
            if entry["workflow"] in front_workflows:
                front_population.append(entry)
            elif entry["workflow"] in buffer_workflows:
                buffer_population.append(entry)
        
        save_checkpoint_csv(
            all_population,
            f"{run_id}_r{round_idx}",
            save_dir=CHECKPOINT_DIR,
        )
        plot_pareto(
            all_population,
            file_name=f"{run_id}_r{round_idx}",
            save_dir=GRAPH_DIR,
            survivors=front_population,
            buffer_list=buffer_population,
            pareto_front=front_population,  # controller.front is already the actual Pareto front
        )

        if controller.should_stop():
            log(f"[Round {round_idx}] Early stopping: no improvement for {args.patience} rounds.")
            break

        log(f"[Round {round_idx}] === End ===\n")

    # ========== FINALIZATION: Combine archive + final front and validate ==========
    finalize_valid = getattr(args, 'finalize_valid', 100)
    log(f"\n{'='*60}")
    log("Finalization: Combining Pareto front archive with final generation")
    log(f"{'='*60}")
    
    # Get final Pareto front
    pareto_front_final = controller.front
    
    # Combine archive and final front
    combined_candidates = controller.archive + pareto_front_final
    
    log(f"Archive size: {len(controller.archive)}")
    log(f"Final front size: {len(pareto_front_final)}")
    log(f"Combined candidates: {len(combined_candidates)}")
    log(f"Validating top candidates with {finalize_valid} problems...")
    
    # Evaluate combined candidates with finalize_valid problems
    if len(combined_candidates) > 0:
        workflows_combined = [wf for wf, _ in combined_candidates]
        new_evals = evaluator.evaluate_batch(workflows_combined)
        
        # Update evaluation results with final validation
        for (wf, old_eval), new_eval in zip(combined_candidates, new_evals):
            old_eval.pass_at_1 = new_eval.pass_at_1
            old_eval.token_cost = new_eval.token_cost
            old_eval.score = new_eval.pass_at_1 - TOKEN_PENALTY * new_eval.token_cost
            old_eval.success_cases = new_eval.success_cases
            old_eval.failure_cases = new_eval.failure_cases
    
    # Re-sort combined candidates using NSGA-II
    objs_combined = np.array([[eval_res.token_cost, -eval_res.pass_at_1] for _, eval_res in combined_candidates])
    fronts_combined = non_dominated_sort(objs_combined)
    pareto_front_indices_combined = fronts_combined[0] if fronts_combined else []
    final_pareto_front = [combined_candidates[i] for i in pareto_front_indices_combined]
    
    log(f"Final Pareto front size: {len(final_pareto_front)}")
    if len(final_pareto_front) > 0:
        log(f"Final Pareto front pass@k range: {min([eval_res.pass_at_1 for _, eval_res in final_pareto_front]):.4f} - {max([eval_res.pass_at_1 for _, eval_res in final_pareto_front]):.4f}")
        log(f"Final Pareto front token range: {min([eval_res.token_cost for _, eval_res in final_pareto_front]):.0f} - {max([eval_res.token_cost for _, eval_res in final_pareto_front]):.0f}")
    
    # Save final Pareto front
    final_population = []
    for wf, eval_res in final_pareto_front:
        final_population.append({
            "workflow": wf,
            "fitness": {
                "pass_at_k": eval_res.pass_at_1,
                "token": eval_res.token_cost,
            },
        })
    
    save_checkpoint_csv(final_population, f"{run_id}_final" if run_id else "hdlo_final", save_dir=CHECKPOINT_DIR)
    plot_pareto(
        final_population,
        file_name=f"{run_id}_final" if run_id else "hdlo_final",
        save_dir=GRAPH_DIR,
        survivors=final_population,
        pareto_front=final_population,
    )
    log(f"Final Pareto front saved: {run_id}_final" if run_id else "hdlo_final")
    log(f"{'='*60}\n")
    
    log(
        f"[Result] Best pass@1={best_eval.pass_at_1:.4f} tokens={best_eval.token_cost:.1f} "
        f"score={best_eval.score:.4f}"
    )
    log(f"[Result] Final workflow: {workflow_to_string(current_workflow)}")
    log(f"[Result] Journal saved to {journal.journal_path}")
    log(f"[Result] Checkpoints saved to {CHECKPOINT_DIR}")
    log(f"[Result] Graphs saved to {GRAPH_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HDLO: Single-expert LLM-driven baseline")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name (MBPP or MATH)")
    parser.add_argument("--model", type=str, default=None, help="VLLM model name for bias/diagnosis (defaults to VLLM_MODEL env var)")
    parser.add_argument("--temperature", type=float, default=0.35, help="LLM temperature")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM; use heuristics only")
    parser.add_argument("--vllm-model", type=str, default=None, help="vLLM model for architect proposals")
    parser.add_argument("--vllm-max-tokens", type=int, default=480, help="Max tokens for architect proposals")
    parser.add_argument("--proposals-per-round", type=int, default=PROPOSALS_PER_ROUND, help="Number of candidate workflows to generate each round")
    parser.add_argument("--max-rounds", type=int, default=12, help="Maximum optimization rounds")
    parser.add_argument("--patience", type=int, default=4, help="Patience for greedy rollback controller")
    parser.add_argument("--num-problems", type=int, default=NUM_EVAL_PROBLEMS, help="Problems per round evaluation")
    parser.add_argument("--server-url", type=str, default="http://localhost:8001", help="Evaluation server URL")
    parser.add_argument("--no-server", action="store_true", help="Disable server eval and use local scoring")
    parser.add_argument("--batch-size", type=int, default=60, help="Batch size for server evaluation")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED, help="Seed for dataset sampling")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for logs")
    parser.add_argument("--max-eval-iter", type=int, default=4, help="Maximum evaluation iterations per workflow")
    parser.add_argument("--buffer-size", type=int, default=10, help="The size of buffer (probation) pool")
    parser.add_argument("--max-front-size", type=int, default=15, help="Maximum size of Pareto front")
    parser.add_argument("--finalize-valid", type=int, default=100, help="Number of problems for final validation of Pareto front")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hdlo(args)
