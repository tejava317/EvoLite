# src/schemas.py
"""
Task-specific Pydantic schemas for structured output in multi-agent workflows.

Each schema defines the contract between sequential agents:
- think: Agent's reasoning process (logged but not forwarded to next agent)
- problem: Original problem passed through unchanged
- comment: Brief notes for the next agent in the workflow
- answer: The actual solution/output for the task

These schemas are used with LangChain's with_structured_output() for reliable parsing.
"""

from typing import Optional, Union, Type
from pydantic import BaseModel, Field


class MBPPResponse(BaseModel):
    """Response schema for Python code generation tasks (MBPP benchmark).
    
    The answer field should contain a complete, executable Python function
    that passes the provided test cases.
    """
    think: str = Field(
        description="Brief reasoning about how to solve this problem (keep under 300 words). "
                    "Key points only: requirements, edge cases, approach. "
                    "NOT forwarded to the next agent."
    )
    problem: str = Field(
        description="Copy the original problem statement EXACTLY as received, "
                    "including any test cases. Do not modify or summarize."
    )
    comment: str = Field(
        description="1-3 sentences for the next agent: key insights, edge cases handled, "
                    "function name used, any concerns or suggestions for improvement."
    )
    answer: str = Field(
        description="The complete Python function implementation. "
                    "Use the exact function name from the test case. "
                    "Include necessary imports at the top. "
                    "Must be valid, executable Python code."
    )


class MATHResponse(BaseModel):
    """Response schema for mathematical problem solving (MATH benchmark).
    
    The answer field should contain the step-by-step solution ending with
    the final answer in \\boxed{} format.
    """
    think: str = Field(
        description="Brief mathematical reasoning (keep under 300 words). "
                    "Key steps and calculations only. "
                    "NOT forwarded to the next agent."
    )
    problem: str = Field(
        description="Copy the original math problem EXACTLY as received, "
                    "preserving all LaTeX notation. Do not modify or rephrase."
    )
    comment: str = Field(
        description="1-3 sentences for the next agent: key mathematical insights, "
                    "methods used, potential alternative approaches, confidence level."
    )
    answer: str = Field(
        description="The mathematical solution with clear steps. "
                    "MUST end with \\boxed{final_answer} containing the final answer. "
                    "Examples: \\boxed{42}, \\boxed{\\frac{3}{4}}, \\boxed{x^2 + 1}"
    )


class CRUXResponse(BaseModel):
    """Response schema for code output prediction (CRUX-O benchmark).
    
    The answer field should contain ONLY the predicted output value
    as a valid Python literal.
    """
    think: str = Field(
        description="Brief code trace (keep under 300 words). "
                    "Track key variables and outputs only. "
                    "NOT forwarded to the next agent."
    )
    problem: str = Field(
        description="Copy the original code and assertion EXACTLY as received. "
                    "Do not modify the code in any way."
    )
    comment: str = Field(
        description="1-3 sentences for the next agent: key observations about the code, "
                    "tricky parts identified, confidence in the prediction."
    )
    answer: str = Field(
        description="The predicted output value ONLY as a Python literal. "
                    "Examples: [1, 2, 3], {'a': 1}, 'hello', 42, True, None. "
                    "NO code, NO 'assert', NO explanation - just the raw value."
    )


class ExtractorResponse(BaseModel):
    """Simple schema for the final answer extractor.
    
    Extracts and normalizes the final answer from the workflow output.
    """
    answer: str = Field(
        description="The extracted and normalized final answer ready for evaluation."
    )


# Schema registry for easy lookup by task name
TASK_SCHEMAS: dict[str, Type[BaseModel]] = {
    "MBPP": MBPPResponse,
    "MATH": MATHResponse,
    "CRUX-O": CRUXResponse,
}


def get_schema_for_task(task_name: str) -> Type[BaseModel]:
    """Get the appropriate response schema for a given task/benchmark.
    
    Args:
        task_name: Name of the task (MBPP, MATH, CRUX-O)
        
    Returns:
        The Pydantic model class for that task
        
    Raises:
        ValueError: If task_name is not recognized
    """
    task_upper = task_name.upper()
    
    # Try exact match
    if task_upper in TASK_SCHEMAS:
        return TASK_SCHEMAS[task_upper]
    
    # Try partial match
    for key, schema in TASK_SCHEMAS.items():
        if key in task_upper or task_upper in key:
            return schema
    
    # Default to MBPP schema for unknown tasks
    return MBPPResponse


# Type alias for any valid response
AgentResponse = Union[MBPPResponse, MATHResponse, CRUXResponse]

