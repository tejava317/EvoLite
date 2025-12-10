# src/client/client.py
"""
Evaluation client for communicating with the evaluation server.
"""

import httpx
from typing import Optional, List

from .models import BlockConfig, EvalResult


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
        return httpx.Client(
            timeout=httpx.Timeout(3600.0, connect=30.0),  # 1 hour timeout
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=200,
                keepalive_expiry=60.0,
            ),
        )
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(3600.0, connect=30.0),  # 1 hour timeout
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=400,
                    keepalive_expiry=60.0,
                ),
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
    
    def _parse_result(self, data: dict, num_problems: int) -> EvalResult:
        """Parse API response into EvalResult."""
        return EvalResult(
            pass_at_1=data["pass_at_1"],
            num_correct=data["num_correct"],
            num_problems=data["num_problems"],
            total_tokens=data["total_tokens"],
            completion_tokens=data.get("completion_tokens", 0),
            total_time=data["total_time"],
            tokens_per_second=data["tokens_per_second"]
        )
    
    def _error_result(self, num_problems: int, error: str) -> EvalResult:
        """Create an error result."""
        return EvalResult(
            pass_at_1=0.0,
            num_correct=0,
            num_problems=num_problems,
            total_tokens=0,
            completion_tokens=0,
            total_time=0,
            tokens_per_second=0,
            error=error
        )
    
    def evaluate_simple(
        self,
        roles: List[str],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> EvalResult:
        """
        Evaluate a workflow with simple role list (converts to AgentBlocks).
        
        Args:
            roles: List of agent role names
            task_name: Dataset/task name (MBPP, MATH)
            num_problems: Number of problems to evaluate
            use_extractor: Whether to use answer extractor
            seed: Random seed for reproducibility
            think: Enable thinking mode
            
        Returns:
            EvalResult with pass@1 and metrics
        """
        payload = {
            "roles": roles,
            "task_name": task_name,
            "use_extractor": use_extractor,
            "think": think,
            "num_problems": num_problems,
            "seed": seed,
        }
        
        try:
            with self._get_sync_client() as client:
                response = client.post(
                    f"{self.server_url}/evaluate/simple",
                    json=payload
                )
                response.raise_for_status()
                return self._parse_result(response.json(), num_problems)
        except Exception as e:
            return self._error_result(num_problems, str(e))
    
    def evaluate(
        self,
        blocks: List[BlockConfig],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> EvalResult:
        """
        Evaluate a BlockWorkflow.
        
        Args:
            blocks: List of BlockConfig objects
            task_name: Dataset/task name (MBPP, MATH)
            num_problems: Number of problems to evaluate
            use_extractor: Whether to use answer extractor
            seed: Random seed for reproducibility
            think: Enable thinking mode
            
        Returns:
            EvalResult with pass@1 and metrics
        """
        payload = {
            "workflow": {
                "blocks": [b.to_dict() for b in blocks],
                "task_name": task_name,
                "use_extractor": use_extractor,
                "think": think
            },
            "num_problems": num_problems,
            "seed": seed,
        }
        
        try:
            with self._get_sync_client() as client:
                response = client.post(
                    f"{self.server_url}/evaluate",
                    json=payload
                )
                response.raise_for_status()
                return self._parse_result(response.json(), num_problems)
        except Exception as e:
            return self._error_result(num_problems, str(e))
    
    async def evaluate_simple_async(
        self,
        roles: List[str],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> EvalResult:
        """Async version of evaluate_simple."""
        payload = {
            "roles": roles,
            "task_name": task_name,
            "use_extractor": use_extractor,
            "think": think,
            "num_problems": num_problems,
            "seed": seed,
        }
        
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.server_url}/evaluate/simple",
                json=payload
            )
            response.raise_for_status()
            return self._parse_result(response.json(), num_problems)
        except Exception as e:
            return self._error_result(num_problems, str(e))
    
    async def evaluate_async(
        self,
        blocks: List[BlockConfig],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> EvalResult:
        """Async version of evaluate."""
        payload = {
            "workflow": {
                "blocks": [b.to_dict() for b in blocks],
                "task_name": task_name,
                "use_extractor": use_extractor,
                "think": think
            },
            "num_problems": num_problems,
            "seed": seed,
        }
        
        try:
            client = await self._get_async_client()
            response = await client.post(
                f"{self.server_url}/evaluate",
                json=payload
            )
            response.raise_for_status()
            return self._parse_result(response.json(), num_problems)
        except Exception as e:
            return self._error_result(num_problems, str(e))
    
    def evaluate_batch_simple(
        self,
        workflows: List[List[str]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> List[EvalResult]:
        """
        Evaluate multiple workflows (simple role lists) on the same problems.
        
        Args:
            workflows: List of role lists, e.g. [["Agent1"], ["Agent1", "Agent2"]]
            task_name: Dataset/task name
            num_problems: Number of problems
            use_extractor: Whether to use extractor
            seed: Random seed
            think: Enable thinking mode
            
        Returns:
            List of EvalResults, one per workflow
        """
        block_workflows = [
            [BlockConfig(type="agent", role=role) for role in roles]
            for roles in workflows
        ]
        return self.evaluate_batch(block_workflows, task_name, num_problems, use_extractor, seed, think)
    
    def evaluate_batch(
        self,
        workflows: List[List[BlockConfig]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> List[EvalResult]:
        """
        Evaluate multiple BlockWorkflows on the same problems.
        
        Args:
            workflows: List of block lists
            task_name: Dataset/task name
            num_problems: Number of problems
            use_extractor: Whether to use extractor
            seed: Random seed
            think: Enable thinking mode
            
        Returns:
            List of EvalResults, one per workflow
        """
        payload = {
            "workflows": [
                {
                    "blocks": [b.to_dict() for b in blocks],
                    "task_name": task_name,
                    "use_extractor": use_extractor,
                    "think": think
                }
                for blocks in workflows
            ],
            "num_problems": num_problems,
            "seed": seed,
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
                        results.append(self._error_result(num_problems, item["error"]))
                    else:
                        results.append(self._parse_result(item["result"], num_problems))
                return results
        except Exception as e:
            return [self._error_result(num_problems, str(e)) for _ in workflows]
    
    async def evaluate_batch_async(
        self,
        workflows: List[List[BlockConfig]],
        task_name: str = "MBPP",
        num_problems: int = 10,
        use_extractor: bool = True,
        seed: Optional[int] = None,
        think: bool = False,
    ) -> List[EvalResult]:
        """
        Async version of evaluate_batch.
        
        Each workflow is evaluated concurrently on the server side.
        """
        payload = {
            "workflows": [
                {
                    "blocks": [b.to_dict() for b in blocks],
                    "task_name": task_name,
                    "use_extractor": use_extractor,
                    "think": think
                }
                for blocks in workflows
            ],
            "num_problems": num_problems,
            "seed": seed,
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
                    results.append(self._error_result(num_problems, item["error"]))
                else:
                    results.append(self._parse_result(item["result"], num_problems))
            return results
        except Exception as e:
            return [self._error_result(num_problems, str(e)) for _ in workflows]
