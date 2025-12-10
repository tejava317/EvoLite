#!/usr/bin/env python3
"""
Deep analysis of workflow evolution from checkpoints
"""
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import re

def load_checkpoint(iteration: int) -> List[Dict]:
    """Load checkpoint CSV file"""
    path = f"src/ga/checkpoints/population_adaptive-v2_{iteration}.csv"
    workflows = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                workflows.append({
                    'workflow': row['workflow_roles'],
                    'pass_at_k': float(row['pass_at_k']),
                    'tokens': float(row['tokens']),
                    'iteration': iteration
                })
    except FileNotFoundError:
        print(f"Warning: {path} not found")
    return workflows

def normalize_workflow(wf: str) -> str:
    """Normalize workflow string for comparison"""
    # Remove extra whitespace and normalize arrow syntax
    wf = re.sub(r'\s*->\s*', ' -> ', wf.strip())
    return wf

def find_best_workflows(workflows: List[Dict], top_k: int = 5) -> List[Dict]:
    """Find top-k workflows by pass_at_k, breaking ties with lower tokens"""
    sorted_wfs = sorted(workflows, key=lambda x: (-x['pass_at_k'], x['tokens']))
    return sorted_wfs[:top_k]

def extract_agents(wf: str) -> List[str]:
    """Extract list of agents from workflow string"""
    if ' -> ' in wf:
        return [a.strip() for a in wf.split(' -> ')]
    return [wf.strip()]

def calculate_similarity(wf1: str, wf2: str) -> float:
    """Calculate Jaccard similarity between two workflows"""
    agents1 = set(extract_agents(wf1))
    agents2 = set(extract_agents(wf2))
    if not agents1 and not agents2:
        return 1.0
    if not agents1 or not agents2:
        return 0.0
    intersection = len(agents1 & agents2)
    union = len(agents1 | agents2)
    return intersection / union if union > 0 else 0.0

def find_ancestors(target_wf: str, prev_workflows: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """Find potential ancestor workflows in previous iteration"""
    ancestors = []
    target_normalized = normalize_workflow(target_wf)
    target_agents = set(extract_agents(target_wf))
    
    for wf in prev_workflows:
        wf_normalized = normalize_workflow(wf['workflow'])
        similarity = calculate_similarity(target_wf, wf_normalized)
        
        if similarity >= threshold:
            ancestors.append({
                **wf,
                'similarity': similarity,
                'shared_agents': list(target_agents & set(extract_agents(wf_normalized))),
                'added_agents': list(target_agents - set(extract_agents(wf_normalized))),
                'removed_agents': list(set(extract_agents(wf_normalized)) - target_agents)
            })
    
    return sorted(ancestors, key=lambda x: -x['similarity'])

def analyze_evolution():
    """Main analysis function"""
    iterations = [5, 10, 15, 20, 25, 30]
    
    # Load all checkpoints
    all_workflows = {}
    for it in iterations:
        all_workflows[it] = load_checkpoint(it)
    
    print("=" * 80)
    print("WORKFLOW EVOLUTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze iteration 25
    print("=" * 80)
    print("ITERATION 25 - TOP WORKFLOWS")
    print("=" * 80)
    best_25 = find_best_workflows(all_workflows[25], top_k=5)
    
    for idx, wf in enumerate(best_25, 1):
        print(f"\n[{idx}] Score: {wf['pass_at_k']:.4f}, Tokens: {wf['tokens']:.0f}")
        print(f"    Workflow: {wf['workflow']}")
        agents = extract_agents(wf['workflow'])
        print(f"    Length: {len(agents)} agents")
        print(f"    Agents: {' -> '.join(agents)}")
    
    # Analyze iteration 30
    print("\n" + "=" * 80)
    print("ITERATION 30 - TOP WORKFLOWS")
    print("=" * 80)
    best_30 = find_best_workflows(all_workflows[30], top_k=5)
    
    for idx, wf in enumerate(best_30, 1):
        print(f"\n[{idx}] Score: {wf['pass_at_k']:.4f}, Tokens: {wf['tokens']:.0f}")
        print(f"    Workflow: {wf['workflow']}")
        agents = extract_agents(wf['workflow'])
        print(f"    Length: {len(agents)} agents")
        print(f"    Agents: {' -> '.join(agents)}")
    
    # Trace evolution for top workflows
    print("\n" + "=" * 80)
    print("EVOLUTION TRACE - ITERATION 25 TOP WORKFLOWS")
    print("=" * 80)
    
    for idx, wf_25 in enumerate(best_25[:3], 1):  # Top 3
        print(f"\n{'='*80}")
        print(f"WORKFLOW #{idx} FROM ITERATION 25")
        print(f"Workflow: {wf_25['workflow']}")
        print(f"Score: {wf_25['pass_at_k']:.4f}, Tokens: {wf_25['tokens']:.0f}")
        print(f"{'='*80}")
        
        # Check if it exists in iteration 30
        wf_25_norm = normalize_workflow(wf_25['workflow'])
        in_30 = [w for w in all_workflows[30] if normalize_workflow(w['workflow']) == wf_25_norm]
        if in_30:
            print(f"✓ Found in iteration 30: Score={in_30[0]['pass_at_k']:.4f}, Tokens={in_30[0]['tokens']:.0f}")
        else:
            print("✗ Not found in iteration 30 (may have been replaced)")
        
        # Trace back through iterations
        current_wf = wf_25['workflow']
        for it in [20, 15, 10, 5]:
            ancestors = find_ancestors(current_wf, all_workflows[it], threshold=0.3)
            if ancestors:
                best_ancestor = ancestors[0]
                print(f"\n  Iteration {it} - Best Ancestor (similarity: {best_ancestor['similarity']:.2f}):")
                print(f"    Workflow: {best_ancestor['workflow']}")
                print(f"    Score: {best_ancestor['pass_at_k']:.4f}, Tokens: {best_ancestor['tokens']:.0f}")
                if best_ancestor['shared_agents']:
                    print(f"    Shared agents: {', '.join(best_ancestor['shared_agents'])}")
                if best_ancestor['added_agents']:
                    print(f"    Added in {wf_25['iteration']}: {', '.join(best_ancestor['added_agents'])}")
                if best_ancestor['removed_agents']:
                    print(f"    Removed from {it}: {', '.join(best_ancestor['removed_agents'])}")
                current_wf = best_ancestor['workflow']  # Continue tracing from this ancestor
            else:
                print(f"\n  Iteration {it} - No clear ancestor found (similarity < 0.3)")
    
    # Trace evolution for iteration 30 top workflows
    print("\n" + "=" * 80)
    print("EVOLUTION TRACE - ITERATION 30 TOP WORKFLOWS")
    print("=" * 80)
    
    for idx, wf_30 in enumerate(best_30[:3], 1):  # Top 3
        print(f"\n{'='*80}")
        print(f"WORKFLOW #{idx} FROM ITERATION 30")
        print(f"Workflow: {wf_30['workflow']}")
        print(f"Score: {wf_30['pass_at_k']:.4f}, Tokens: {wf_30['tokens']:.0f}")
        print(f"{'='*80}")
        
        # Check if it exists in iteration 25
        wf_30_norm = normalize_workflow(wf_30['workflow'])
        in_25 = [w for w in all_workflows[25] if normalize_workflow(w['workflow']) == wf_30_norm]
        if in_25:
            print(f"✓ Found in iteration 25: Score={in_25[0]['pass_at_k']:.4f}, Tokens={in_25[0]['tokens']:.0f}")
        else:
            print("✗ Not found in iteration 25 (newly evolved)")
        
        # Trace back through iterations
        current_wf = wf_30['workflow']
        for it in [25, 20, 15, 10, 5]:
            ancestors = find_ancestors(current_wf, all_workflows[it], threshold=0.3)
            if ancestors:
                best_ancestor = ancestors[0]
                print(f"\n  Iteration {it} - Best Ancestor (similarity: {best_ancestor['similarity']:.2f}):")
                print(f"    Workflow: {best_ancestor['workflow']}")
                print(f"    Score: {best_ancestor['pass_at_k']:.4f}, Tokens: {best_ancestor['tokens']:.0f}")
                if best_ancestor['shared_agents']:
                    print(f"    Shared agents: {', '.join(best_ancestor['shared_agents'])}")
                if best_ancestor['added_agents']:
                    print(f"    Added in {wf_30['iteration']}: {', '.join(best_ancestor['added_agents'])}")
                if best_ancestor['removed_agents']:
                    print(f"    Removed from {it}: {', '.join(best_ancestor['removed_agents'])}")
                current_wf = best_ancestor['workflow']  # Continue tracing from this ancestor
            else:
                print(f"\n  Iteration {it} - No clear ancestor found (similarity < 0.3)")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    for it in iterations:
        if not all_workflows[it]:
            continue
        scores = [w['pass_at_k'] for w in all_workflows[it]]
        tokens = [w['tokens'] for w in all_workflows[it]]
        print(f"\nIteration {it}:")
        print(f"  Population size: {len(all_workflows[it])}")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Avg score: {sum(scores)/len(scores):.4f}")
        print(f"  Avg tokens: {sum(tokens)/len(tokens):.0f}")
        print(f"  Workflows with score >= 0.8: {sum(1 for s in scores if s >= 0.8)}")
        print(f"  Workflows with score >= 0.75: {sum(1 for s in scores if s >= 0.75)}")
    
    # Agent frequency analysis
    print("\n" + "=" * 80)
    print("AGENT FREQUENCY IN TOP WORKFLOWS")
    print("=" * 80)
    
    agent_counts = defaultdict(int)
    for wf in best_25 + best_30:
        for agent in extract_agents(wf['workflow']):
            agent_counts[agent] += 1
    
    print("\nMost common agents in top workflows:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {agent}: {count} occurrences")

if __name__ == "__main__":
    analyze_evolution()
