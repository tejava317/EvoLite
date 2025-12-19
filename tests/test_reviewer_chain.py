#!/usr/bin/env python3
"""
Test: Compare reviewer-only vs reviewer-creator-reviewer workflows
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import EvaluationClient, BlockConfig

# Helper to create workflow from role names
def wf(*roles):
    return [BlockConfig(type="agent", role=r) for r in roles]


async def main():
    client = EvaluationClient("http://localhost:8001")
    
    # Test 1: 4 Reviewers only (should all refuse)
    reviewer_only = wf(
        "Bug Hunter",           
        "Boundary Tester",      
        "Assumption Checker",   
        "Consistency Checker",  
    )
    
    # Test 2: 2 Reviewers -> Creator -> 2 Reviewers (shorter)
    reviewer_creator_reviewer = wf(
        "Bug Hunter",           # Reviewer - should refuse
        "Boundary Tester",      # Reviewer - should refuse
        "Solution Drafter",     # CREATOR - should generate code!
        "Quality Auditor",      # Reviewer - should review
        "Solution Reviewer",    # Reviewer - should review
    )
    
    print("=" * 70)
    print("TEST 1: 4 Reviewers Only (All should refuse)")
    print("=" * 70)
    print("Agents:", [b.role for b in reviewer_only])
    print()
    
    results1 = await client.evaluate_batch_async(
        workflows=[reviewer_only],
        task_name="MBPP",
        num_problems=5,
        use_extractor=False,
        seed=99999,
        think=True,
    )
    
    for r in results1:
        print(f"Pass@1: {r.pass_at_1:.1%}")
        print(f"Time: {r.total_time:.1f}s")
        print(f"Tokens: {r.completion_tokens}")
        if r.error:
            print(f"Error: {r.error}")
    
    print()
    print("=" * 70)
    print("TEST 2: 2 Reviewers -> Creator -> 2 Reviewers")
    print("=" * 70)
    print("Agents:", [b.role for b in reviewer_creator_reviewer])
    print()
    print("Expected: First 2 refuse, Creator generates, last 2 review/improve")
    print()
    
    results2 = await client.evaluate_batch_async(
        workflows=[reviewer_creator_reviewer],
        task_name="MBPP",
        num_problems=5,
        use_extractor=False,
        seed=99999,
        think=True,
    )
    
    for r in results2:
        print(f"Pass@1: {r.pass_at_1:.1%}")
        print(f"Time: {r.total_time:.1f}s")
        print(f"Tokens: {r.completion_tokens}")
        if r.error:
            print(f"Error: {r.error}")
    
    await client.close()
    
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Reviewer-only:           {results1[0].pass_at_1:.1%} pass")
    print(f"Reviewer-Creator-Review: {results2[0].pass_at_1:.1%} pass")
    print()
    print("If Reviewer-Creator-Review >> Reviewer-only, the workflow design matters!")


if __name__ == "__main__":
    asyncio.run(main())
