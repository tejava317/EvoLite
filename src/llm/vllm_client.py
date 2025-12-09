# src/llm/vllm_client.py
"""
Async vLLM client using /v1/completions API with native batching.

Sends batched prompts in a single request for maximum throughput.
Uses HuggingFace tokenizer for proper chat template formatting.
"""

import os
import asyncio
import time
from typing import Optional, List
from dataclasses import dataclass
from collections import deque

import httpx
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


@dataclass
class JobResult:
    """Result from a completed vLLM job."""
    job_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    execution_time: float
    status: str
    error: Optional[str] = None


class VLLMClient:
    """
    Async client for vLLM server using /v1/completions API.
    
    Uses native batching - sends multiple prompts in a single request.
    This is far more efficient than individual requests.
    
    Uses HuggingFace tokenizer for proper chat template formatting.
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        default_temperature: float = 0.6,
        default_max_tokens: int = 2048,  # Room for prompts within 6000 context
        timeout: float = 600.0,  # 10 minutes for large batches
    ):
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://38.128.232.68:27717/v1")).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        
        # Load tokenizer for chat template
        print(f"  Loading tokenizer for {self.model}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        print(f"  ✓ Tokenizer loaded")
        
        # Reusable async client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Monitoring counters
        self._total_batches = 0      # Number of batch API calls
        self._total_prompts = 0      # Total prompts completed
        self._total_failed = 0       # Prompts that failed
        self._total_prompt_tokens = 0
        self._total_gen_tokens = 0
        self._lock = asyncio.Lock()
        
        # Token throughput tracking (1-minute rolling window)
        self._gen_token_history: deque = deque()
        self._prompt_token_history: deque = deque()
    
    def format_prompt(self, messages: List[dict]) -> str:
        """
        Format messages using the model's chat template.
        
        Args:
            messages: List of message dicts [{"role": "system", ...}, {"role": "user", ...}]
            
        Returns:
            Formatted prompt string with generation prompt added
        """
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    async def _record_tokens(self, prompt_tokens: int, gen_tokens: int):
        """Record tokens for throughput calculation."""
        now = time.time()
        async with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_gen_tokens += gen_tokens
            self._prompt_token_history.append((now, prompt_tokens))
            self._gen_token_history.append((now, gen_tokens))
            # Prune entries older than 60 seconds
            cutoff = now - 60.0
            while self._prompt_token_history and self._prompt_token_history[0][0] < cutoff:
                self._prompt_token_history.popleft()
            while self._gen_token_history and self._gen_token_history[0][0] < cutoff:
                self._gen_token_history.popleft()
    
    def _calc_tokens_per_second(self, history: deque) -> float:
        """Calculate tokens/second from a token history deque."""
        now = time.time()
        cutoff = now - 60.0
        
        # Prune old entries
        while history and history[0][0] < cutoff:
            history.popleft()
        
        if not history:
            return 0.0
        
        total = sum(tokens for _, tokens in history)
        oldest_time = history[0][0]
        window = now - oldest_time
        
        if window < 1.0:
            return float(total)
        
        return total / window
    
    def get_stats(self) -> dict:
        """Get current monitoring stats."""
        return {
            "batches": self._total_batches,
            "prompts": self._total_prompts,
            "failed": self._total_failed,
            "prompt_tok/s": round(self._calc_tokens_per_second(self._prompt_token_history), 1),
            "gen_tok/s": round(self._calc_tokens_per_second(self._gen_token_history), 1),
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_gen_tokens": self._total_gen_tokens,
        }
    
    def print_stats(self):
        """Print current stats."""
        stats = self.get_stats()
        print(f"[vLLM] Requests: {stats['requests']}, "
              f"Completed: {stats['completed']}, "
              f"Prompt: {stats['prompt_tok/s']} tok/s, "
              f"Gen: {stats['gen_tok/s']} tok/s")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=30.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                    keepalive_expiry=120.0,
                )
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def batch_complete(
        self,
        messages_list: List[List[dict]],
        temperature: float = None,
        max_tokens: int = None,
        max_retries: int = 3,
    ) -> List[JobResult]:
        """
        Execute batched completions in a single request.
        
        Args:
            messages_list: List of message lists, each containing [system, user] dicts
                          e.g. [[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], ...]
            temperature: Sampling temperature
            max_tokens: Max tokens to generate per prompt
            max_retries: Number of retries on failure
            
        Returns:
            List of JobResult objects, one per input prompt
        """
        if not messages_list:
            return []
        
        start_time = time.time()
        
        # Convert messages to formatted prompts using model's chat template
        prompts = [self.format_prompt(messages) for messages in messages_list]
        
        payload = {
            "model": self.model,
            "prompt": prompts,  # List of prompts for batching
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
            "top_p": 0.95,
        }
        
        async with self._lock:
            self._total_batches += 1
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                response = await client.post(
                    f"{self.base_url}/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                usage = data.get("usage", {})
                total_prompt_tokens = usage.get("prompt_tokens", 0)
                total_completion_tokens = usage.get("completion_tokens", 0)
                
                # Record tokens
                await self._record_tokens(total_prompt_tokens, total_completion_tokens)
                
                # Parse results - one choice per prompt
                results = []
                choices = data.get("choices", [])
                
                # Distribute tokens evenly (approximate)
                prompt_tokens_each = total_prompt_tokens // len(prompts) if prompts else 0
                completion_tokens_each = total_completion_tokens // len(prompts) if prompts else 0
                
                for i, messages in enumerate(messages_list):
                    if i < len(choices):
                        content = choices[i].get("text", "")
                        results.append(JobResult(
                            job_id=f"batch-{i}",
                            content=content,
                            prompt_tokens=prompt_tokens_each,
                            completion_tokens=completion_tokens_each,
                            total_tokens=prompt_tokens_each + completion_tokens_each,
                            execution_time=time.time() - start_time,
                            status="COMPLETED"
                        ))
                    else:
                        results.append(JobResult(
                            job_id=f"batch-{i}",
                            content="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            execution_time=time.time() - start_time,
                            status="MISSING",
                            error="No response for this prompt in batch"
                        ))
                
                async with self._lock:
                    self._total_prompts += len(results)
                
                return results
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {str(e)}"
                # 4xx errors (except 429) are permanent
                if e.response.status_code < 500 and e.response.status_code != 429:
                    break
                    
            except httpx.HTTPError as e:
                last_error = str(e)
            
            # Exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        # All retries exhausted - return error results for all prompts
        async with self._lock:
            self._total_failed += len(messages_list)
        
        return [
            JobResult(
                job_id=f"batch-{i}",
                content="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                execution_time=time.time() - start_time,
                status="FAILED",
                error=f"Failed after {max_retries} attempts: {last_error}"
            )
            for i in range(len(messages_list))
        ]
    
    async def complete(
        self,
        messages: List[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> JobResult:
        """
        Execute a single completion (convenience wrapper around batch_complete).
        
        Args:
            messages: List of message dicts [{"role": "system", ...}, {"role": "user", ...}]
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            JobResult with response
        """
        results = await self.batch_complete([messages], temperature, max_tokens)
        return results[0] if results else JobResult(
            job_id="",
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            execution_time=0,
            status="ERROR",
            error="Empty batch result"
        )
    
    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> JobResult:
        """
        Generate a single response (convenience method).
        
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
        return await self.complete(messages, temperature, max_tokens)


if __name__ == "__main__":
    async def test():
        print("Testing vLLM Client (batch completions)...")
        
        client = VLLMClient()
        
        try:
            # Single request test
            print("\n1. Single request test:")
            result = await client.generate(
                system_prompt="You are a helpful assistant.",
                user_content="What is 2 + 2? Reply with just the number."
            )
            print(f"   Status: {result.status}")
            print(f"   Content: {result.content[:200] if result.content else '(empty)'}")
            print(f"   Tokens: {result.total_tokens}")
            print(f"   Time: {result.execution_time:.2f}s")
            if result.error:
                print(f"   Error: {result.error}")
            
            # Batch request test
            print("\n2. Batch request test (5 prompts in single request):")
            messages_list = [
                [
                    {"role": "system", "content": "You are a math tutor."},
                    {"role": "user", "content": f"What is {i} + {i}? Reply with just the number."}
                ]
                for i in range(1, 6)
            ]
            
            start = time.time()
            results = await client.batch_complete(messages_list)
            total_time = time.time() - start
            
            for i, r in enumerate(results):
                content_preview = r.content[:50] if r.content else r.error[:50] if r.error else "(empty)"
                print(f"   [{i+1}] {r.status}: {content_preview}...")
            
            print(f"\n   Total batch time: {total_time:.2f}s")
            print(f"   (all prompts processed in single request)")
            
            # Print stats
            print("\n3. Stats:")
            client.print_stats()
            
        finally:
            await client.close()
    
    asyncio.run(test())

