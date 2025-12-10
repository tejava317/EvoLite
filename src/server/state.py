# src/server/state.py
"""
Global server state management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ServerState:
    """Cached server state."""
    datasets: Dict[str, Any] = field(default_factory=dict)
    llm: Optional[Any] = None  # ChatOpenAI, but avoid import at module level
    structured_llms: Dict[str, Any] = field(default_factory=dict)
    llm_mode: str = "vllm"  # "vllm" or "openai"
    llm_model: str = ""  # Model name being used


# Global state instance
state = ServerState()
