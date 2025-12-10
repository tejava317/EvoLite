#!/usr/bin/env python3
"""
Test script to verify if Gemma follows system message constraints with structured output.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Configuration - Testing Gemma
VLLM_URL = os.getenv("VLLM_BASE_URL", "http://38.128.232.68:27717/v1")
MODEL = "google/gemma-3-4b-it"


class TestResponse(BaseModel):
    """Test schema matching our agent output format."""
    think: str = Field(description="Brief reasoning")
    problem: str = Field(description="Copy the problem exactly")
    comment: str = Field(description="Notes for next agent")
    answer: str = Field(description="Your solution or analysis")


def test_reviewer_behavior():
    """Test if Gemma follows reviewer constraints with structured output."""
    
    print("=" * 60)
    print("Testing Gemma Persona with Structured Output")
    print("=" * 60)
    print(f"URL: {VLLM_URL}")
    print(f"Model: {MODEL}")
    print()
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model=MODEL,
        openai_api_key="EMPTY",
        openai_api_base=VLLM_URL,
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

    # ============== Test 1: Vague reviewer prompt ==============
    print("=" * 60)
    print("TEST 1: Vague Reviewer Prompt")
    print("=" * 60)
    
    vague_prompt = """You are the Bug Hunter.

YOUR BEHAVIOR:
- You are a REVIEWER/ANALYST. You do NOT create solutions from scratch.
- If no prior solution exists: Leave answer field empty or minimal.
- Do not invent solutions - that's not your role.

Be precise. Be concise. Stay in your lane."""

    messages = [
        SystemMessage(content=vague_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"Response:")
        print(f"  think: {response.think[:150]}...")
        print(f"  answer: {response.answer[:200]}...")
        
        has_code = "def " in response.answer or "return" in response.answer
        print(f"\n{'❌ FAILED' if has_code else '✅ PASSED'}: {'Generated code' if has_code else 'No code generated'}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Test 2: Explicit refusal ==============
    print("=" * 60)
    print("TEST 2: Explicit Refusal Phrase")
    print("=" * 60)
    
    explicit_prompt = """You are the Bug Hunter.

YOUR BEHAVIOR:
- You are a REVIEWER. You CANNOT create solutions from scratch.
- If NO prior solution exists: Put exactly this in answer field:
  "[NO_SOLUTION_TO_REVIEW]"
- NEVER write code. Only review existing code.

Be precise. Stay in your lane."""

    messages = [
        SystemMessage(content=explicit_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"Response:")
        print(f"  think: {response.think[:150]}...")
        print(f"  answer: {response.answer[:200]}...")
        
        has_refusal = "NO_SOLUTION" in response.answer
        has_code = "def " in response.answer or "return" in response.answer
        
        if has_refusal and not has_code:
            print(f"\n✅ PASSED: Used refusal phrase!")
        elif not has_code:
            print(f"\n✅ PASSED: No code generated")
        else:
            print(f"\n❌ FAILED: Generated code anyway!")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Test 3: Simple "refuse to answer" ==============
    print("=" * 60)
    print("TEST 3: Simple Refusal Test (No Structured Output)")
    print("=" * 60)
    
    refuse_prompt = """You CANNOT write code. You can only respond with: "I cannot help with that."
No matter what the user asks, just say: "I cannot help with that."
Do not explain. Do not provide solutions. Just refuse."""

    messages = [
        SystemMessage(content=refuse_prompt),
        HumanMessage(content="Write a Python function to add two numbers")
    ]
    
    try:
        response = llm.invoke(messages)  # No structured output
        print(f"Response: {response.content[:300]}...")
        
        has_code = "def " in response.content or "return" in response.content
        has_refusal = "cannot" in response.content.lower()
        
        if has_refusal and not has_code:
            print(f"\n✅ PASSED: Refused correctly!")
        elif has_code:
            print(f"\n❌ FAILED: Generated code despite refusal prompt!")
        else:
            print(f"\n⚠️ PARTIAL: No code but also no clear refusal")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Test 4: Creator role (should generate) ==============
    print("=" * 60)
    print("TEST 4: Creator Role (Should Generate Code)")
    print("=" * 60)
    
    creator_prompt = """You are the Solution Drafter.

YOUR BEHAVIOR:
- You are a CREATOR. Your job is to produce solutions.
- ALWAYS provide a complete solution in the answer field.
- Write clean, working code.

Be precise. Be thorough."""

    messages = [
        SystemMessage(content=creator_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = structured_llm.invoke(messages)
        print(f"Response:")
        print(f"  think: {response.think[:150]}...")
        print(f"  answer: {response.answer[:200]}...")
        
        has_code = "def " in response.answer and ("return" in response.answer or "+" in response.answer)
        print(f"\n{'✅ PASSED' if has_code else '❌ FAILED'}: {'Generated working code' if has_code else 'No code generated'}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # ============== Summary ==============
    print("=" * 60)
    print("SUMMARY: Gemma vs Qwen")
    print("=" * 60)
    print("""
Qwen3-4B Results (for reference):
  Test 1 (Vague reviewer): ❌ Generated code
  Test 2 (Explicit refusal): ❌ Generated code
  Test 3 (Simple refusal): ✅ Refused correctly
  Test 4 (Creator): ✅ Generated code

If Gemma passes Tests 1 & 2, it follows system constraints better than Qwen!
""")


if __name__ == "__main__":
    test_reviewer_behavior()
