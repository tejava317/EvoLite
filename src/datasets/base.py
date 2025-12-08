# src/datasets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional


@dataclass
class Problem:
    """Represents a single problem from a benchmark dataset."""
    id: str
    prompt: str
    ground_truth: Any
    metadata: Optional[dict] = None
    
    def __repr__(self):
        return f"Problem(id={self.id}, prompt={self.prompt[:50]}...)"


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets."""
    
    def __init__(self, split: str = "test"):
        self.split = split
        self.problems: List[Problem] = []
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from HuggingFace or local source."""
        pass
    
    @abstractmethod
    def evaluate(self, response: str, problem: Problem) -> bool:
        """
        Evaluate a model response against the ground truth.
        
        Args:
            response: The model's response (code or answer)
            problem: The problem being evaluated
            
        Returns:
            True if the response is correct, False otherwise
        """
        pass
    
    def get_problem(self, index: int) -> Problem:
        """Get a specific problem by index."""
        if not self._loaded:
            self.load()
        return self.problems[index]
    
    def get_problems(self, indices: Optional[List[int]] = None) -> List[Problem]:
        """Get multiple problems by indices, or all if indices is None."""
        if not self._loaded:
            self.load()
        if indices is None:
            return self.problems
        return [self.problems[i] for i in indices]
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[Problem]:
        """
        Sample n problems from the dataset.
        
        - If n <= dataset size: sample without replacement (random.sample)
        - If n  > dataset size: sample with replacement (random.choices) so large
          workloads can exceed the dataset cardinality (useful for load testing).
        """
        import random
        if not self._loaded:
            self.load()
        if seed is not None:
            random.seed(seed)
        if n <= len(self.problems):
            return random.sample(self.problems, n)
        # With replacement when n > dataset size
        return random.choices(self.problems, k=n)
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self.problems)
    
    def __iter__(self):
        if not self._loaded:
            self.load()
        return iter(self.problems)
    
    def __getitem__(self, index: int) -> Problem:
        return self.get_problem(index)


