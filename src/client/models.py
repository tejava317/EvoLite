# src/client/models.py
"""
Data models for the evaluation client.
"""

from typing import Optional, List
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
    completion_tokens: int  # Generated tokens only (excludes prompts)
    total_time: float
    tokens_per_second: float
    error: Optional[str] = None


def roles_to_blocks(roles: List[str]) -> List[BlockConfig]:
    """Convert a list of role names to AgentBlock configs."""
    return [BlockConfig(type="agent", role=role) for role in roles]

