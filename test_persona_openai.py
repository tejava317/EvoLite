#!/usr/bin/env python3
"""
Test script to verify if OpenAI GPT follows system message constraints better than Qwen.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Check for API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not set!")
    print("Please set it: export OPENAI_API_KEY='your-key-here'")
    exit(1)


class TestResponse(BaseModel):
    """Test schema matching our agent output format."""
    think: str = Field(description="Brief reasoning")
    problem: str = Field(description="Copy the problem exactly")
    comment: str = Field(description="Notes for next agent")
    answer: str = Field(description="Your solution or analysis")


def test_reviewer_behavior():
    """Test different reviewer prompt strategies with OpenAI."""
    
    print("=" * 60)
    print("Testing Reviewer Role Behavior with OpenAI GPT")
    print("=" * 60)
    print(f"Model: gpt-4o-mini")
    print()
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model="gpt-5-nano",  # Using gpt-4o-mini as gpt-5-nano might not exist
        temperature=0.6,
        max_tokens=2000,
    )
    structured_llm = llm.with_structured_output(TestResponse)
    
    # Test problem
    problem = """Write a Python function to add two numbers.

Example test case:
assert add_numbers(2, 3) == 5"""

    user_message = f"""TASK: Python Function Generation

PROBLEM:
{problem}

You are the FIRST agent. There is NO prior solution."""

    # ============== Test 1: Original (vague) reviewer prompt ==============
    print("=" * 60)
    print("TEST 1: Original Vague Reviewer Prompt")
    print("=" * 60)
    
    vague_prompt = """You are the Bug Hunter.

EXPERTISE: Search for concrete bugs using reasoning, tests, and adversarial thinking.

YOUR BEHAVIOR:
- You are a REVIEWER/ANALYST. You do NOT create solutions from scratch.
- If a prior solution exists: Analyze it, find issues, suggest improvements.
- If no prior solution exists: Leave answer field empty or minimal.
- Your value is in ANALYSIS, not GENERATION.
- Do not invent solutions - that's not your role.

Be precise. Be concise. Stay in your lane."""

    messages = [
        SystemMessage(content=vague_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"System prompt: Vague 'leave empty' instruction")
        print(f"\nResponse:")
        print(f"  think: {response.think[:200]}...")
        print(f"  answer: {response.answer[:300] if response.answer else '(empty)'}...")
        
        has_code = "def " in response.answer or "return" in response.answer
        is_empty = len(response.answer.strip()) < 50
        
        if is_empty or not has_code:
            print(f"\n✅ PASSED: Did not generate code (answer is empty/minimal)")
        else:
            print(f"\n❌ FAILED: Generated code anyway!")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Test 2: Explicit refusal phrase ==============
    print("=" * 60)
    print("TEST 2: Explicit Refusal Phrase")
    print("=" * 60)
    
    explicit_prompt = """You are the Bug Hunter.

EXPERTISE: Search for concrete bugs using reasoning, tests, and adversarial thinking.

YOUR BEHAVIOR:
- You are a REVIEWER/ANALYST. You do NOT create solutions from scratch.
- If a prior solution exists: Analyze it, find issues, suggest improvements.
- If NO prior solution exists: Put exactly this in answer field:
  "[NO_SOLUTION_TO_REVIEW] - Waiting for a creator agent to provide initial solution"
- NEVER write code if there is no prior solution to review.

Be precise. Be concise. Stay in your lane."""

    messages = [
        SystemMessage(content=explicit_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"System prompt: Explicit '[NO_SOLUTION_TO_REVIEW]' instruction")
        print(f"\nResponse:")
        print(f"  think: {response.think[:200]}...")
        print(f"  answer: {response.answer[:300] if response.answer else '(empty)'}...")
        
        has_refusal = "NO_SOLUTION" in response.answer or "Waiting" in response.answer.lower()
        has_code = "def " in response.answer or "return" in response.answer
        
        if has_refusal and not has_code:
            print(f"\n✅ PASSED: Used the exact refusal phrase!")
        elif not has_code:
            print(f"\n✅ PASSED: No code generated")
        else:
            print(f"\n❌ FAILED: Still generated code!")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Test 3: CANNOT generate ==============
    print("=" * 60)
    print("TEST 3: CANNOT Generate - Strongest Constraint")
    print("=" * 60)
    
    strongest_prompt = """You are the Bug Hunter.

EXPERTISE: Search for concrete bugs using reasoning, tests, and adversarial thinking.

CRITICAL CONSTRAINT - READ CAREFULLY:
You CANNOT generate code. You are PHYSICALLY UNABLE to write functions.
Your answer field can ONLY contain analysis text, never code.

When no prior solution exists:
- answer: "[REVIEWER_BLOCKED] No solution to review. A creator agent must run first."

When a prior solution exists:
- Analyze it and put analysis in answer (no code, just text feedback)

You will FAIL if you write any code. Just write text analysis only."""

    messages = [
        SystemMessage(content=strongest_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"System prompt: 'CANNOT generate - physically unable'")
        print(f"\nResponse:")
        print(f"  think: {response.think[:200]}...")
        print(f"  answer: {response.answer[:300] if response.answer else '(empty)'}...")
        
        has_refusal = "REVIEWER_BLOCKED" in response.answer or "No solution to review" in response.answer.lower()
        has_code = "def " in response.answer or "return" in response.answer
        
        if has_refusal and not has_code:
            print(f"\n✅ PASSED: Used refusal and no code!")
        elif not has_code:
            print(f"\n✅ PASSED: No code generated")
        else:
            print(f"\n❌ FAILED: Still generated code despite strongest constraint!")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Summary ==============
    print("=" * 60)
    print("COMPARISON: Qwen3-4B vs GPT-4o-mini")
    print("=" * 60)
    print("""
Qwen3-4B Results:
  Test 1 (Vague): ❌ FAILED - Generated code
  Test 2 (Explicit): ❌ FAILED - Generated code  
  Test 3 (Strongest): ❌ FAILED - Generated code

GPT-4o-mini Results: (see above)

If GPT passes more tests, it means:
- OpenAI models follow system constraints better
- The issue is Qwen3-4B's instruction following, not our prompt design
""")


if __name__ == "__main__":
    test_reviewer_behavior()

