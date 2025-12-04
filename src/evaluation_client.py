# src/evaluation_client.py
"""
Client for the EvoLite Evaluation Server.

Provides sync and async interfaces for evaluating BlockWorkflows.
"""

import asyncio
import httpx
from typing import Optional, Union, List
from dataclasses import dataclass


@dataclass
class BlockConfig:
    """Block configuration for API calls."""
    type: str  # "agent" or "composite"
    role: Optional[str] = None
    divider_role: Optional[str] = "Divider"
    synth_role: Optional[str] = "Synthesizer"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API."""
        d = {"type": self.type}
        if self.type == "agent":
            d["role"] = self.role
        else:
            d["divider_role"] = self.divider_role
            d["synth_role"] = self.synth_role
        return d


@dataclass 
class EvalResult:
    """Result from workflow evaluation."""
    pass_at_1: float
    num_correct: int
    num_problems: int
    total_tokens: int
    total_time: float
    tokens_per_second: float
    error: Optional[str] = None


def roles_to_blocks(roles: List[str]) -> List[BlockConfig]:
    """Convert a list of role names to AgentBlock configs."""
    return [BlockConfig(type="agent", role=role) for role in roles]


class EvaluationClient:
    """
    Client for the evaluation server.
    
    Supports both BlockWorkflow configs and simple role lists.
    
    Usage (simple):
        client = EvaluationClient("http://localhost:8000")
        result = client.evaluate_simple(["Code Generation Agent"], "MBPP", num_problems=10)
        print(f"Pass@1: {result.pass_at_1}")
    
    Usage (with blocks):
        blocks = [
            BlockConfig(type="agent", role="Code Generation Agent"),
            BlockConfig(type="composite", divider_role="Divider", synth_role="Synthesizer")
        ]
        result = client.evaluate(blocks, "MBPP", num_problems=10)
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self._async_client: Optional[httpx.AsyncClient] = None
    
    def _get_sync_client(self) -> httpx.Client:
        """Get a sync HTTP client."""
        return httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0)
            )
        return self._async_client
    
    async def close(self):
        """Close the async client."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None
    
    def health_check(self) -> dict:
        """Check if the server is healthy."""
        with self._get_sync_client() as client:
            response = client.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
    
    def evaluate_simple(
        self,
        roles: List[str],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> EvalResult:
        """
        Evaluate a workflow with simple role list (converts to AgentBlocks).
        
        Args:
            roles: List of agent role names
            task_name: Dataset/task name (MBPP, MATH)
            num_problems: Number of problems to evaluate
            use_extractor: Whether to use answer extractor
            seed: Random seed for reproducibility
            
        Returns:
            EvalResult with pass@1 and metrics
        """
        payload = {
            "roles": roles,
            "task_name": task_name,
            "use_extractor": use_extractor,
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            with self._get_sync_client() as client:
                response = client.post(
                    f"{self.server_url}/evaluate/simple",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                return EvalResult(
                    pass_at_1=data["pass_at_1"],
                    num_correct=data["num_correct"],
                    num_problems=data["num_problems"],
                    total_tokens=data["total_tokens"],
                    total_time=data["total_time"],
                    tokens_per_second=data["tokens_per_second"]
                )
        except Exception as e:
            return EvalResult(
                pass_at_1=0.0,
                num_correct=0,
                num_problems=num_problems,
                total_tokens=0,
                total_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    def evaluate(
        self,
        blocks: List[BlockConfig],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> EvalResult:
        """
        Evaluate a BlockWorkflow.
        
        Args:
            blocks: List of BlockConfig objects
            task_name: Dataset/task name (MBPP, MATH)
            num_problems: Number of problems to evaluate
            use_extractor: Whether to use answer extractor
            seed: Random seed for reproducibility
            
        Returns:
            EvalResult with pass@1 and metrics
        """
        payload = {
            "workflow": {
                "blocks": [b.to_dict() for b in blocks],
                "task_name": task_name,
                "use_extractor": use_extractor
            },
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            with self._get_sync_client() as client:
                response = client.post(
                    f"{self.server_url}/evaluate",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                return EvalResult(
                    pass_at_1=data["pass_at_1"],
                    num_correct=data["num_correct"],
                    num_problems=data["num_problems"],
                    total_tokens=data["total_tokens"],
                    total_time=data["total_time"],
                    tokens_per_second=data["tokens_per_second"]
                )
        except Exception as e:
            return EvalResult(
                pass_at_1=0.0,
                num_correct=0,
                num_problems=num_problems,
                total_tokens=0,
                total_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def evaluate_simple_async(
        self,
        roles: List[str],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> EvalResult:
        """Async version of evaluate_simple."""
        payload = {
            "roles": roles,
            "task_name": task_name,
            "use_extractor": use_extractor,
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.server_url}/evaluate/simple",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return EvalResult(
                pass_at_1=data["pass_at_1"],
                num_correct=data["num_correct"],
                num_problems=data["num_problems"],
                total_tokens=data["total_tokens"],
                total_time=data["total_time"],
                tokens_per_second=data["tokens_per_second"]
            )
        except Exception as e:
            return EvalResult(
                pass_at_1=0.0,
                num_correct=0,
                num_problems=num_problems,
                total_tokens=0,
                total_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    async def evaluate_async(
        self,
        blocks: List[BlockConfig],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> EvalResult:
        """Async version of evaluate."""
        payload = {
            "workflow": {
                "blocks": [b.to_dict() for b in blocks],
                "task_name": task_name,
                "use_extractor": use_extractor
            },
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.server_url}/evaluate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return EvalResult(
                pass_at_1=data["pass_at_1"],
                num_correct=data["num_correct"],
                num_problems=data["num_problems"],
                total_tokens=data["total_tokens"],
                total_time=data["total_time"],
                tokens_per_second=data["tokens_per_second"]
            )
        except Exception as e:
            return EvalResult(
                pass_at_1=0.0,
                num_correct=0,
                num_problems=num_problems,
                total_tokens=0,
                total_time=0,
                tokens_per_second=0,
                error=str(e)
            )
    
    def evaluate_batch_simple(
        self,
        workflows: List[List[str]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> List[EvalResult]:
        """
        Evaluate multiple workflows (simple role lists) on the same problems.
        
        Args:
            workflows: List of role lists, e.g. [["Agent1"], ["Agent1", "Agent2"]]
            task_name: Dataset/task name
            num_problems: Number of problems
            use_extractor: Whether to use extractor
            seed: Random seed
            
        Returns:
            List of EvalResults, one per workflow
        """
        # Convert role lists to block configs
        block_workflows = [
            [BlockConfig(type="agent", role=role) for role in roles]
            for roles in workflows
        ]
        return self.evaluate_batch(block_workflows, task_name, num_problems, use_extractor, seed)
    
    def evaluate_batch(
        self,
        workflows: List[List[BlockConfig]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> List[EvalResult]:
        """
        Evaluate multiple BlockWorkflows on the same problems.
        
        Args:
            workflows: List of block lists
            task_name: Dataset/task name
            num_problems: Number of problems
            use_extractor: Whether to use extractor
            seed: Random seed
            
        Returns:
            List of EvalResults, one per workflow
        """
        payload = {
            "workflows": [
                {
                    "blocks": [b.to_dict() for b in blocks],
                    "task_name": task_name,
                    "use_extractor": use_extractor
                }
                for blocks in workflows
            ],
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            with self._get_sync_client() as client:
                response = client.post(
                    f"{self.server_url}/evaluate/batch",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data:
                    if "error" in item:
                        results.append(EvalResult(
                            pass_at_1=0.0,
                            num_correct=0,
                            num_problems=num_problems,
                            total_tokens=0,
                            total_time=0,
                            tokens_per_second=0,
                            error=item["error"]
                        ))
                    else:
                        r = item["result"]
                        results.append(EvalResult(
                            pass_at_1=r["pass_at_1"],
                            num_correct=r["num_correct"],
                            num_problems=r["num_problems"],
                            total_tokens=r["total_tokens"],
                            total_time=r["total_time"],
                            tokens_per_second=r["tokens_per_second"]
                        ))
                return results
        except Exception as e:
            return [
                EvalResult(
                    pass_at_1=0.0,
                    num_correct=0,
                    num_problems=num_problems,
                    total_tokens=0,
                    total_time=0,
                    tokens_per_second=0,
                    error=str(e)
                )
                for _ in workflows
            ]
    
    async def evaluate_batch_async(
        self,
        workflows: List[List[BlockConfig]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None
    ) -> List[EvalResult]:
        """Async version of evaluate_batch."""
        payload = {
            "workflows": [
                {
                    "blocks": [b.to_dict() for b in blocks],
                    "task_name": task_name,
                    "use_extractor": use_extractor
                }
                for blocks in workflows
            ],
            "num_problems": num_problems,
            "seed": seed
        }
        
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.server_url}/evaluate/batch",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data:
                if "error" in item:
                    results.append(EvalResult(
                        pass_at_1=0.0,
                        num_correct=0,
                        num_problems=num_problems,
                        total_tokens=0,
                        total_time=0,
                        tokens_per_second=0,
                        error=item["error"]
                    ))
                else:
                    r = item["result"]
                    results.append(EvalResult(
                        pass_at_1=r["pass_at_1"],
                        num_correct=r["num_correct"],
                        num_problems=r["num_problems"],
                        total_tokens=r["total_tokens"],
                        total_time=r["total_time"],
                        tokens_per_second=r["tokens_per_second"]
                    ))
            return results
        except Exception as e:
            return [
                EvalResult(
                    pass_at_1=0.0,
                    num_correct=0,
                    num_problems=num_problems,
                    total_tokens=0,
                    total_time=0,
                    tokens_per_second=0,
                    error=str(e)
                )
                for _ in workflows
            ]


# ============== Convenience Functions for GA ==============

def evaluate_fitness_simple(
    roles: List[str],
    task_name: str = "MBPP",
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate fitness using simple role list.
    
    Fitness = pass@1 - (token_penalty * total_tokens)
    """
    client = EvaluationClient(server_url)
    result = client.evaluate_simple(roles, task_name, num_problems)
    
    if result.error:
        return 0.0
    
    return result.pass_at_1 - (token_penalty * result.total_tokens)


def evaluate_fitness(
    blocks: List[BlockConfig],
    task_name: str = "MBPP",
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate fitness using BlockConfig list.
    
    Fitness = pass@1 - (token_penalty * total_tokens)
    """
    client = EvaluationClient(server_url)
    result = client.evaluate(blocks, task_name, num_problems)
    
    if result.error:
        return 0.0
    
    return result.pass_at_1 - (token_penalty * result.total_tokens)


def evaluate_block_workflow(
    workflow,  # BlockWorkflow from src.agents.workflow_block
    num_problems: int = 5,
    server_url: str = "http://localhost:8000",
    token_penalty: float = 0.0001
) -> float:
    """
    Evaluate a BlockWorkflow object via the server.
    
    Converts BlockWorkflow.blocks to BlockConfig list for API.
    """
    from src.agents.block import AgentBlock, CompositeBlock
    
    blocks = []
    for block in workflow.blocks:
        if isinstance(block, AgentBlock):
            blocks.append(BlockConfig(type="agent", role=block.role))
        elif isinstance(block, CompositeBlock):
            blocks.append(BlockConfig(
                type="composite",
                divider_role=block.divider_role,
                synth_role=block.synth_role
            ))
    
    return evaluate_fitness(
        blocks=blocks,
        task_name=workflow.task_name,
        num_problems=num_problems,
        server_url=server_url,
        token_penalty=token_penalty
    )


if __name__ == "__main__":
    # Test the client
    print("Testing Evaluation Client (BlockWorkflow style)...")
    
    client = EvaluationClient()
    
    # Health check
    print("\n1. Health check:")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Datasets: {health['datasets']}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure the server is running: uvicorn src.evaluation_server:app")
        exit(1)
    
    # Simple evaluation (role list)
    print("\n2. Simple workflow evaluation (role list):")
    result = client.evaluate_simple(
        roles=["Code Generation Agent"],
        task_name="MBPP",
        num_problems=3
    )
    print(f"   Pass@1: {result.pass_at_1:.2%}")
    print(f"   Correct: {result.num_correct}/{result.num_problems}")
    print(f"   Tokens: {result.total_tokens}")
    print(f"   Time: {result.total_time:.2f}s")
    print(f"   Tokens/s: {result.tokens_per_second:.1f}")
    
    # Block-based evaluation
    print("\n3. Block-based workflow evaluation:")
    blocks = [
        BlockConfig(type="agent", role="Task Parsing Agent"),
        BlockConfig(type="agent", role="Code Generation Agent"),
    ]
    result = client.evaluate(
        blocks=blocks,
        task_name="MBPP",
        num_problems=3
    )
    print(f"   Pass@1: {result.pass_at_1:.2%}")
    print(f"   Correct: {result.num_correct}/{result.num_problems}")
    print(f"   Tokens: {result.total_tokens}")
    
    # With composite block
    print("\n4. Workflow with CompositeBlock:")
    blocks = [
        BlockConfig(type="agent", role="Task Parsing Agent"),
        BlockConfig(type="composite", divider_role="Divider", synth_role="Synthesizer"),
    ]
    result = client.evaluate(
        blocks=blocks,
        task_name="MBPP",
        num_problems=2
    )
    print(f"   Pass@1: {result.pass_at_1:.2%}")
    print(f"   Tokens: {result.total_tokens}")
    
    # Batch evaluation
    print("\n5. Batch evaluation (comparing workflows):")
    results = client.evaluate_batch_simple(
        workflows=[
            ["Code Generation Agent"],
            ["Task Parsing Agent", "Code Generation Agent"],
        ],
        task_name="MBPP",
        num_problems=3
    )
    
    for i, r in enumerate(results):
        print(f"   Workflow {i+1}: Pass@1={r.pass_at_1:.2%}, Tokens={r.total_tokens}")
