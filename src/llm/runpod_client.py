# src/llm/runpod_client.py
"""
Async RunPod client using native /run + /status API.

Fire-all-at-once pattern: Submit all jobs immediately, let RunPod queue handle concurrency.
"""

import os
import asyncio
import time
from typing import Optional
from collections import deque
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class JobResult:
    """Result from a completed RunPod job."""
    job_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    execution_time: float
    status: str
    error: Optional[str] = None


class RunPodAsyncClient:
    """
    Async client for RunPod serverless vLLM endpoints.
    
    Uses native RunPod API (/run + /status) for maximum throughput.
    Fire-all-at-once pattern - submit all jobs immediately, RunPod handles queuing.
    """
    
    def __init__(
        self,
        api_key: str = None,
        endpoint_id: str = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 2000,
        poll_interval: float = 0.5,
        max_poll_time: float = 300,
        max_concurrent_run: int = 200,
        run_rate_limit: int = 1000,
        run_rate_window: float = 10.0,
        max_concurrent_status: int = 400,
        status_rate_limit: int = 2000,
        status_rate_window: float = 10.0,
        enable_monitoring: bool = True,
    ):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
        
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID not set")
        
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.max_concurrent_run = max_concurrent_run
        self.run_rate_limit = run_rate_limit
        self.run_rate_window = run_rate_window
        self.max_concurrent_status = max_concurrent_status
        self.status_rate_limit = status_rate_limit
        self.status_rate_window = status_rate_window
        
        # Reusable async client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None
        
        # Monitoring counters
        self.enable_monitoring = enable_monitoring
        self._active_jobs = 0  # Jobs submitted but not yet completed
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._peak_active = 0
        self._lock = asyncio.Lock()
        self._submit_lock = asyncio.Lock()
        self._run_timestamps = deque()  # track recent /run submissions for rate limiting
        self._status_timestamps = deque()  # track recent /status calls
    
    async def _inc_active(self):
        """Increment active job count."""
        async with self._lock:
            self._active_jobs += 1
            self._total_submitted += 1
            if self._active_jobs > self._peak_active:
                self._peak_active = self._active_jobs
    
    async def _dec_active(self, success: bool = True):
        """Decrement active job count."""
        async with self._lock:
            self._active_jobs -= 1
            if success:
                self._total_completed += 1
            else:
                self._total_failed += 1
    
    async def _await_status_slot(self):
        """
        Enforce local limits for /status: concurrency and rate.
        """
        while True:
            async with self._lock:
                now = time.time()
                while self._status_timestamps and now - self._status_timestamps[0] > self.status_rate_window:
                    self._status_timestamps.popleft()
                within_rate = len(self._status_timestamps) < self.status_rate_limit
                # we reuse _active_jobs for run tracking, so only rate matters for status;
                # cap concurrent status to max_concurrent_status by counting recent calls in window
                within_concurrency = len(self._status_timestamps) < self.max_concurrent_status
                if within_rate and within_concurrency:
                    self._status_timestamps.append(now)
                    return
            await asyncio.sleep(0.01)
    
    async def _await_submit_slot(self):
        """
        Enforce local limits: concurrent /run <= max_concurrent_run
        and rate limit run_rate_limit per run_rate_window seconds.
        """
        while True:
            async with self._lock:
                now = time.time()
                # prune old timestamps
                while self._run_timestamps and now - self._run_timestamps[0] > self.run_rate_window:
                    self._run_timestamps.popleft()
                within_rate = len(self._run_timestamps) < self.run_rate_limit
                within_concurrency = self._active_jobs < self.max_concurrent_run
                if within_rate and within_concurrency:
                    self._run_timestamps.append(now)
                    return
            await asyncio.sleep(0.01)
    
    def get_stats(self) -> dict:
        """Get current monitoring stats."""
        return {
            "active_jobs": self._active_jobs,
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "peak_active": self._peak_active,
        }
    
    def print_stats(self):
        """Print current stats."""
        stats = self.get_stats()
        print(f"[RunPod Stats] Active: {stats['active_jobs']}, "
              f"Submitted: {stats['total_submitted']}, "
              f"Completed: {stats['total_completed']}, "
              f"Failed: {stats['total_failed']}, "
              f"Peak: {stats['peak_active']}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            # Increase connection pool and queue to avoid local bottlenecks when firing many jobs.
            # Note: RunPod still queues on their side; this just reduces client-side queuing.
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(
                    max_connections=2000,
                    max_keepalive_connections=400,
                    keepalive_expiry=60.0,
                )
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def poll_job_once(self, job_id: str, start_time: float = None) -> Optional[JobResult]:
        """
        Non-blocking poll: return result if completed/failed/timeout, otherwise None.
        """
        start_time = start_time or time.time()
        elapsed = time.time() - start_time
        if elapsed > self.max_poll_time:
            await self._dec_active(success=False)
            return JobResult(
                job_id=job_id,
                content="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                execution_time=elapsed,
                status="TIMEOUT",
                error=f"Job timed out after {self.max_poll_time}s"
            )

        try:
            status_data = await self.check_job_status(job_id)
            status = status_data.get("status", "UNKNOWN")

            if status == "COMPLETED":
                output = status_data.get("output", {})

                # Extract content from various possible response formats
                content = ""
                usage = {}

                # Handle list output format (common in vLLM)
                if isinstance(output, list) and len(output) > 0:
                    item = output[0]
                    if isinstance(item, dict):
                        if "choices" in item:
                            choices = item["choices"]
                            if choices and len(choices) > 0:
                                choice = choices[0]
                                if "tokens" in choice:
                                    tokens = choice["tokens"]
                                    content = tokens[0] if isinstance(tokens, list) and tokens else ""
                                elif "message" in choice:
                                    content = choice["message"].get("content", "")
                                elif "text" in choice:
                                    content = choice["text"]
                        usage = item.get("usage", {})
                    elif isinstance(item, str):
                        content = item

                elif isinstance(output, dict):
                    if "text" in output:
                        text_data = output["text"]
                        content = text_data[0] if isinstance(text_data, list) else text_data
                    elif "choices" in output:
                        choices = output["choices"]
                        if choices and len(choices) > 0:
                            choice = choices[0]
                            if "tokens" in choice:
                                tokens = choice["tokens"]
                                content = tokens[0] if isinstance(tokens, list) and tokens else ""
                            elif "message" in choice:
                                content = choice["message"].get("content", "")
                            elif "text" in choice:
                                content = choice["text"]
                    elif "content" in output:
                        content = output["content"]
                    usage = output.get("usage", {})

                elif isinstance(output, str):
                    content = output

                prompt_tokens = usage.get("prompt_tokens", usage.get("input", 0))
                completion_tokens = usage.get("completion_tokens", usage.get("output", 0))

                await self._dec_active(success=True)
                return JobResult(
                    job_id=job_id,
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    execution_time=time.time() - start_time,
                    status="COMPLETED"
                )

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                await self._dec_active(success=False)
                return JobResult(
                    job_id=job_id,
                    content="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    execution_time=time.time() - start_time,
                    status="FAILED",
                    error=str(error)
                )

            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                return None

            else:
                return None

        except httpx.HTTPError:
            # If we've already waited beyond max_poll_time, treat as timeout
            if elapsed > self.max_poll_time:
                await self._dec_active(success=False)
                return JobResult(
                    job_id=job_id,
                    content="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    execution_time=elapsed,
                    status="TIMEOUT",
                    error="Polling failed repeatedly",
                )
            return None
    
    async def submit_job(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Submit a job to RunPod /run endpoint.
        
        Returns job_id immediately (non-blocking).
        """
        client = await self._get_client()
        await self._await_submit_slot()
        
        payload = {
            "input": {
                "messages": messages,
                "sampling_params": {
                    "temperature": temperature or self.default_temperature,
                    "max_tokens": max_tokens or self.default_max_tokens,
                }
            }
        }
        
        response = await client.post(f"{self.base_url}/run", json=payload)
        response.raise_for_status()
        
        data = response.json()
        await self._inc_active()
        return data["id"]
    
    async def check_job_status(self, job_id: str) -> dict:
        """Check the status of a job."""
        client = await self._get_client()
        await self._await_status_slot()
        
        response = await client.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        
        return response.json()
    
    async def poll_job(self, job_id: str, start_time: float = None) -> JobResult:
        """
        Poll a job until completion.
        
        Args:
            job_id: The job ID to poll
            start_time: When the job was submitted (for timing)
            
        Returns:
            JobResult with content and metadata
        """
        start_time = start_time or time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_poll_time:
                await self._dec_active(success=False)
                return JobResult(
                    job_id=job_id,
                    content="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    execution_time=elapsed,
                    status="TIMEOUT",
                    error=f"Job timed out after {self.max_poll_time}s"
                )
            
            try:
                status_data = await self.check_job_status(job_id)
                status = status_data.get("status", "UNKNOWN")
                
                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    
                    # Extract content from various possible response formats
                    content = ""
                    usage = {}
                    
                    # Handle list output format (common in vLLM)
                    if isinstance(output, list) and len(output) > 0:
                        item = output[0]
                        if isinstance(item, dict):
                            # vLLM format: [{"choices": [{"tokens": ["..."]}], "usage": {...}}]
                            if "choices" in item:
                                choices = item["choices"]
                                if choices and len(choices) > 0:
                                    choice = choices[0]
                                    # Check for tokens array (vLLM format)
                                    if "tokens" in choice:
                                        tokens = choice["tokens"]
                                        content = tokens[0] if isinstance(tokens, list) and tokens else ""
                                    # Check for message.content (OpenAI format)
                                    elif "message" in choice:
                                        content = choice["message"].get("content", "")
                                    # Check for text
                                    elif "text" in choice:
                                        content = choice["text"]
                            # Extract usage from item
                            usage = item.get("usage", {})
                        elif isinstance(item, str):
                            content = item
                    
                    elif isinstance(output, dict):
                        # Check for text in output
                        if "text" in output:
                            text_data = output["text"]
                            content = text_data[0] if isinstance(text_data, list) else text_data
                        elif "choices" in output:
                            choices = output["choices"]
                            if choices and len(choices) > 0:
                                choice = choices[0]
                                if "tokens" in choice:
                                    tokens = choice["tokens"]
                                    content = tokens[0] if isinstance(tokens, list) and tokens else ""
                                elif "message" in choice:
                                    content = choice["message"].get("content", "")
                                elif "text" in choice:
                                    content = choice["text"]
                        elif "content" in output:
                            content = output["content"]
                        usage = output.get("usage", {})
                    
                    elif isinstance(output, str):
                        content = output
                    
                    # Normalize token usage keys (vLLM uses input/output, OpenAI uses prompt/completion)
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input", 0))
                    completion_tokens = usage.get("completion_tokens", usage.get("output", 0))
                    
                    await self._dec_active(success=True)
                    return JobResult(
                        job_id=job_id,
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        execution_time=time.time() - start_time,
                        status="COMPLETED"
                    )
                
                elif status == "FAILED":
                    error = status_data.get("error", "Unknown error")
                    await self._dec_active(success=False)
                    return JobResult(
                        job_id=job_id,
                        content="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        execution_time=time.time() - start_time,
                        status="FAILED",
                        error=str(error)
                    )
                
                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                else:
                    # Unknown status, keep polling
                    await asyncio.sleep(self.poll_interval)
                    
            except httpx.HTTPError as e:
                # Transient error, retry
                await asyncio.sleep(self.poll_interval * 2)
    
    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> JobResult:
        """
        Generate a single response (submit + poll).
        
        Args:
            system_prompt: System message
            user_content: User message
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            JobResult with response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        start_time = time.time()
        job_id = await self.submit_job(messages, temperature, max_tokens)
        return await self.poll_job(job_id, start_time)
    
    async def generate_batch(
        self,
        requests: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> list[JobResult]:
        """
        Generate responses for multiple requests in parallel.
        
        Fire-all-at-once: Submit all jobs immediately, then poll all.
        RunPod handles the queuing and concurrency.
        
        Args:
            requests: List of {"system_prompt": str, "user_content": str}
            temperature: Sampling temperature (applied to all)
            max_tokens: Max tokens (applied to all)
            
        Returns:
            List of JobResults in same order as requests
        """
        if not requests:
            return []
        
        start_time = time.time()
        
        # Phase 1: Submit ALL jobs at once (fire-and-forget to RunPod queue)
        submit_tasks = []
        for req in requests:
            messages = [
                {"role": "system", "content": req["system_prompt"]},
                {"role": "user", "content": req["user_content"]}
            ]
            submit_tasks.append(self.submit_job(messages, temperature, max_tokens))
        
        # Submit all concurrently
        job_ids = await asyncio.gather(*submit_tasks, return_exceptions=True)
        
        # Handle submission errors
        valid_jobs = []  # (index, job_id)
        results = [None] * len(requests)
        
        for i, job_id in enumerate(job_ids):
            if isinstance(job_id, Exception):
                results[i] = JobResult(
                    job_id="",
                    content="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    execution_time=0,
                    status="SUBMIT_FAILED",
                    error=str(job_id)
                )
            else:
                valid_jobs.append((i, job_id))
        
        # Phase 2: Poll ALL jobs in parallel
        if valid_jobs:
            poll_tasks = [self.poll_job(job_id, start_time) for _, job_id in valid_jobs]
            poll_results = await asyncio.gather(*poll_tasks, return_exceptions=True)
            
            for (idx, _), result in zip(valid_jobs, poll_results):
                if isinstance(result, Exception):
                    results[idx] = JobResult(
                        job_id="",
                        content="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        execution_time=time.time() - start_time,
                        status="POLL_FAILED",
                        error=str(result)
                    )
                else:
                    results[idx] = result
        
        return results


# Convenience function for simple usage
async def generate_response(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> JobResult:
    """Quick helper for single generation."""
    client = RunPodAsyncClient()
    try:
        return await client.generate(system_prompt, user_content, temperature, max_tokens)
    finally:
        await client.close()


if __name__ == "__main__":
    async def test():
        print("Testing RunPod Async Client...")
        
        client = RunPodAsyncClient()
        
        try:
            # Single request test
            print("\n1. Single request test:")
            result = await client.generate(
                system_prompt="You are a helpful assistant.",
                user_content="What is 2 + 2? Reply with just the number."
            )
            print(f"   Status: {result.status}")
            print(f"   Content: {result.content[:100]}")
            print(f"   Tokens: {result.total_tokens}")
            print(f"   Time: {result.execution_time:.2f}s")
            
            # Batch request test
            print("\n2. Batch request test (5 requests):")
            requests = [
                {"system_prompt": "You are a math tutor.", "user_content": f"What is {i} + {i}?"}
                for i in range(1, 6)
            ]
            
            start = time.time()
            results = await client.generate_batch(requests)
            total_time = time.time() - start
            
            for i, r in enumerate(results):
                print(f"   [{i+1}] {r.status}: {r.content[:50]}... ({r.execution_time:.2f}s)")
            
            print(f"\n   Total batch time: {total_time:.2f}s")
            print(f"   Avg per request: {total_time/len(requests):.2f}s")
            
        finally:
            await client.close()
    
    asyncio.run(test())

