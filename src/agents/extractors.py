# src/agents/extractors.py
"""
Answer Extractor Agents for standardizing workflow outputs for evaluation.

These agents are appended to workflows to extract clean, evaluable answers
from the potentially verbose outputs of multi-agent workflows.
"""

from src.agents.agent import Agent
from src.config import DEFAULT_PROMPTS


class CodeAnswerExtractor(Agent):
    """
    Extracts clean Python code from workflow responses.
    
    Used for MBPP and similar code generation benchmarks.
    Strips explanations, comments, and test cases to return only executable code.
    """
    
    def __init__(self, agent_client=None):
        # Get the extractor prompt from config
        prompt = DEFAULT_PROMPTS.get('CodeAnswerExtractorPrompt', {}).get('prompt', '')
        
        if not prompt:
            # Fallback prompt
            prompt = """You are a code extraction specialist. Extract ONLY the Python code from the given response.

Return only executable Python code wrapped in a markdown code block:
```python
[extracted code here]
```

Do NOT include explanations, comments about the code, or test cases."""
        
        super().__init__(
            role="Code Answer Extractor",
            prompt=prompt,
            workflow_description=None,
            agent_client=agent_client
        )
    
    def copy(self):
        return CodeAnswerExtractor(self.agent_client)


class MathAnswerExtractor(Agent):
    """
    Extracts the final mathematical answer from workflow responses.
    
    Used for MATH algebra and similar math benchmarks.
    Formats the answer in \\boxed{} notation for evaluation.
    """
    
    def __init__(self, agent_client=None):
        # Get the extractor prompt from config
        prompt = DEFAULT_PROMPTS.get('MathAnswerExtractorPrompt', {}).get('prompt', '')
        
        if not prompt:
            # Fallback prompt
            prompt = """You are a mathematical answer extraction specialist. Extract the FINAL answer from the given solution.

Return ONLY the final answer in LaTeX boxed notation:
\\boxed{[final answer]}

Simplify the answer if possible."""
        
        super().__init__(
            role="Math Answer Extractor",
            prompt=prompt,
            workflow_description=None,
            agent_client=agent_client
        )
    
    def copy(self):
        return MathAnswerExtractor(self.agent_client)


def get_extractor_for_task(task_name: str, agent_client=None) -> Agent:
    """
    Get the appropriate answer extractor for a given task/benchmark.
    
    Args:
        task_name: Name of the task (e.g., "MBPP", "HumanEval", "MATH")
        agent_client: Optional LLM client to use
        
    Returns:
        The appropriate extractor agent
    """
    task_lower = task_name.lower()
    
    # Code generation tasks
    if any(name in task_lower for name in ['mbpp', 'humaneval', 'livecodebench', 'code']):
        return CodeAnswerExtractor(agent_client)
    
    # Math tasks
    if any(name in task_lower for name in ['math', 'algebra', 'gsm', 'arithmetic']):
        return MathAnswerExtractor(agent_client)
    
    # Default to code extractor
    return CodeAnswerExtractor(agent_client)


if __name__ == "__main__":
    # Test the extractors
    print("Testing Answer Extractors...")
    
    # Test code extractor
    print("\n=== Code Answer Extractor ===")
    code_extractor = CodeAnswerExtractor()
    print(f"Role: {code_extractor.role}")
    print(f"Prompt preview: {code_extractor.prompt[:200]}...")
    
    # Test math extractor
    print("\n=== Math Answer Extractor ===")
    math_extractor = MathAnswerExtractor()
    print(f"Role: {math_extractor.role}")
    print(f"Prompt preview: {math_extractor.prompt[:200]}...")
    
    # Test factory function
    print("\n=== Factory Function ===")
    for task in ["MBPP", "HumanEval", "MATH", "algebra"]:
        extractor = get_extractor_for_task(task)
        print(f"Task '{task}' -> {extractor.role}")


