#!/usr/bin/env python3
"""
Deep evolutionary analysis - looking for partial matches and gradual evolution
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

def extract_agents(wf: str) -> List[str]:
    """Extract list of agents from workflow string"""
    if ' -> ' in wf:
        return [a.strip() for a in wf.split(' -> ')]
    return [wf.strip()]

def find_partial_ancestors(target_wf: str, prev_workflows: List[Dict], min_overlap: int = 2) -> List[Dict]:
    """Find workflows that share at least min_overlap agents with target"""
    target_agents = set(extract_agents(target_wf))
    candidates = []
    
    for wf in prev_workflows:
        wf_agents = set(extract_agents(wf['workflow']))
        overlap = target_agents & wf_agents
        
        if len(overlap) >= min_overlap:
            # Check if it's a subsequence or contains key agents
            target_list = extract_agents(target_wf)
            wf_list = extract_agents(wf['workflow'])
            
            # Calculate sequence similarity
            seq_similarity = calculate_sequence_similarity(target_list, wf_list)
            
            candidates.append({
                **wf,
                'overlap_count': len(overlap),
                'overlap_agents': list(overlap),
                'seq_similarity': seq_similarity,
                'is_subsequence': is_subsequence(wf_list, target_list),
                'is_supersequence': is_subsequence(target_list, wf_list)
            })
    
    # Sort by overlap count, then by sequence similarity
    candidates.sort(key=lambda x: (-x['overlap_count'], -x['seq_similarity'], -x['pass_at_k']))
    return candidates

def calculate_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Calculate how similar two sequences are (order matters)"""
    if not seq1 or not seq2:
        return 0.0
    
    # Find longest common subsequence
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    max_len = max(m, n)
    return lcs_length / max_len if max_len > 0 else 0.0

def is_subsequence(sub: List[str], sup: List[str]) -> bool:
    """Check if sub is a subsequence of sup (order preserved)"""
    if not sub:
        return True
    if not sup:
        return False
    
    i = 0
    for agent in sup:
        if i < len(sub) and agent == sub[i]:
            i += 1
    return i == len(sub)

def analyze_workflow_lineage(target_wf: str, iterations: List[int], all_workflows: Dict[int, List[Dict]]):
    """Trace a workflow's lineage through iterations"""
    print(f"\n{'='*80}")
    print(f"LINEAGE ANALYSIS: {target_wf}")
    print(f"{'='*80}")
    
    target_agents = extract_agents(target_wf)
    print(f"Target agents ({len(target_agents)}): {' -> '.join(target_agents)}")
    
    # Check each iteration
    for it in iterations:
        workflows = all_workflows[it]
        
        # Check for exact match
        exact_matches = [w for w in workflows if w['workflow'] == target_wf]
        if exact_matches:
            wf = exact_matches[0]
            print(f"\n  Iteration {it}: ✓ EXACT MATCH")
            print(f"    Score: {wf['pass_at_k']:.4f}, Tokens: {wf['tokens']:.0f}")
            continue
        
        # Find partial ancestors
        ancestors = find_partial_ancestors(target_wf, workflows, min_overlap=2)
        
        if ancestors:
            # Show top 3 most similar
            top_ancestors = ancestors[:3]
            print(f"\n  Iteration {it}: Found {len(ancestors)} potential ancestors")
            
            for idx, anc in enumerate(top_ancestors, 1):
                anc_agents = extract_agents(anc['workflow'])
                print(f"\n    Ancestor #{idx} (overlap: {anc['overlap_count']}, seq_sim: {anc['seq_similarity']:.2f}):")
                print(f"      Workflow: {anc['workflow']}")
                print(f"      Score: {anc['pass_at_k']:.4f}, Tokens: {anc['tokens']:.0f}")
                print(f"      Shared: {', '.join(anc['overlap_agents'])}")
                
                if anc['is_subsequence']:
                    print(f"      → This is a SUBSEQUENCE of target (simpler version)")
                if anc['is_supersequence']:
                    print(f"      → Target is a SUBSEQUENCE of this (more complex version)")
                
                # Show what was added/removed
                added = set(target_agents) - set(anc_agents)
                removed = set(anc_agents) - set(target_agents)
                if added:
                    print(f"      Added in target: {', '.join(added)}")
                if removed:
                    print(f"      Removed from ancestor: {', '.join(removed)}")
        else:
            print(f"\n  Iteration {it}: No clear ancestors (overlap < 2)")

def analyze_evolution_patterns():
    """Main deep analysis"""
    iterations = [5, 10, 15, 20, 25, 30]
    
    # Load all checkpoints
    all_workflows = {}
    for it in iterations:
        all_workflows[it] = load_checkpoint(it)
    
    # Top workflows from iteration 25 and 30
    top_workflows_25 = sorted(all_workflows[25], key=lambda x: (-x['pass_at_k'], x['tokens']))[:4]
    top_workflows_30 = sorted(all_workflows[30], key=lambda x: (-x['pass_at_k'], x['tokens']))[:4]
    
    print("=" * 80)
    print("DEEP EVOLUTIONARY ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing how top workflows evolved from simpler forms...")
    
    # Analyze each top workflow
    unique_workflows = {}
    for wf in top_workflows_25 + top_workflows_30:
        if wf['workflow'] not in unique_workflows:
            unique_workflows[wf['workflow']] = wf
    
    for wf in unique_workflows.values():
        analyze_workflow_lineage(wf['workflow'], iterations, all_workflows)
    
    # Pattern analysis
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    
    # Analyze agent co-occurrence patterns
    print("\nAgent Co-occurrence in High-Performing Workflows (score >= 0.75):")
    cooccurrence = defaultdict(int)
    agent_scores = defaultdict(list)
    
    for it in iterations:
        for wf in all_workflows[it]:
            if wf['pass_at_k'] >= 0.75:
                agents = extract_agents(wf['workflow'])
                for i, a1 in enumerate(agents):
                    agent_scores[a1].append(wf['pass_at_k'])
                    for a2 in agents[i+1:]:
                        pair = tuple(sorted([a1, a2]))
                        cooccurrence[pair] += 1
    
    print("\nMost common agent pairs in high-performing workflows:")
    for pair, count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pair[0]} + {pair[1]}: {count} occurrences")
    
    print("\nAgent average scores (when appearing in workflows with score >= 0.75):")
    avg_scores = {agent: sum(scores)/len(scores) for agent, scores in agent_scores.items() if scores}
    for agent, avg in sorted(avg_scores.items(), key=lambda x: -x[1])[:15]:
        print(f"  {agent}: {avg:.4f} (appeared {len(agent_scores[agent])} times)")

if __name__ == "__main__":
    analyze_evolution_patterns()
