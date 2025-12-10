import asyncio
from src.client import EvaluationClient, BlockConfig
from src.utils.generate_prompts import AGENT_ROLES

# Helper to create workflow from role names
def wf(*roles):
    return [BlockConfig(type="agent", role=r) for r in roles]

# All 50 agent roles as single-agent workflows
ALL_SINGLE_AGENTS = [wf(role_name) for role_name, _ in AGENT_ROLES]

async def main():
    client = EvaluationClient("http://localhost:8001")
    
    # Sample workflows to test different strategies
    workflows = [
        # === Single Agent Baselines ===
        wf("Solution Drafter"),                          # 1: Direct solution
        wf("Logic Implementer"),                         # 2: Direct implementation
        
        # === 2-Agent Workflows ===
        wf("Task Decomposer", "Solution Drafter"),       # 3: Parse then solve
        wf("Solution Drafter", "Solution Reviewer"),     # 4: Draft then review
        wf("Problem Analyzer", "Logic Implementer"),     # 5: Analyze then implement
        
        # === 3-Agent Workflows ===
        wf("Task Decomposer", "Solution Drafter", "Solution Refiner"),  # 6: Parse->Draft->Refine
        wf("Problem Analyzer", "Algorithm Strategist", "Logic Implementer"),  # 7: Analysis pipeline
        wf("Solution Drafter", "Bug Hunter", "Solution Refiner"),  # 8: Draft->Debug->Refine
        
        # === 4-Agent Workflows ===
        wf("Task Decomposer", "Solution Planner", "Logic Implementer", "Correctness Verifier"),  # 9
        wf("Requirement Extractor", "Algorithm Strategist", "Solution Drafter", "Quality Auditor"),  # 10
        
        # === Specialized Workflows ===
        wf("Stepwise Reasoner", "Logic Implementer"),    # 11: Reasoning-focused
        wf("Edge Case Planner", "Solution Drafter", "Boundary Tester"),  # 12: Edge-case focused
        wf("Interface Designer", "Logic Implementer", "Style & Hygiene Enforcer"),  # 13: Clean code
        
        # === Review-Heavy Workflows ===
        wf("Solution Drafter", "Solution Reviewer", "Solution Refiner", "Quality Gatekeeper"),  # 14
        wf("Solution Drafter", "Bug Hunter", "Correctness Verifier", "Final Presenter"),  # 15
    ]
    # # Test all 50 single-agent workflows
    workflows = ALL_SINGLE_AGENTS
    # workflows = [
    #     wf("Final Presenter"),
    # ]

    
    
    print(f"Testing {len(workflows)} workflows on MBPP...")
    print("=" * 60)
    
    results = await client.evaluate_batch_async(
        workflows=workflows,
        task_name="MBPP",
        num_problems=100,  # Start with 50 for faster testing
        use_extractor=False,
        seed=43211,
        think=True,
    )
    
    # Print results sorted by pass@1
    print("\n=== Results (sorted by Pass@1) ===\n")
    indexed_results = list(enumerate(results, 1))
    indexed_results.sort(key=lambda x: x[1].pass_at_1, reverse=True)
    
    for rank, (idx, r) in enumerate(indexed_results, 1):
        workflow_desc = " -> ".join([b.role for b in workflows[idx-1]])
        print(f"{rank:2d}. [{idx:2d}] Pass@1={r.pass_at_1:.1%} | GenTok={r.completion_tokens:,} | {r.tokens_per_second:.1f} tok/s | Time={r.total_time:.1f}s")
        print(f"        {workflow_desc}")
        if r.error:
            print(f"        ERROR: {r.error[:50]}")
        print()
    
    await client.close()

asyncio.run(main())
