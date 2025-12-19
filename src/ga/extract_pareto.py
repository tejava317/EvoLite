#!/usr/bin/env python3
"""
Extract Pareto front individuals from a CSV file.
Pareto front: individuals that are not dominated by any other individual.
For this problem: maximize pass_at_k, minimize tokens.
"""

import csv
import sys
from typing import List, Tuple

def is_dominated(point1: Tuple[float, float], point2: Tuple[float, float]) -> bool:
    """
    Check if point1 is dominated by point2.
    point1 is dominated if point2 has:
    - Higher or equal pass_at_k AND lower tokens, OR
    - Higher pass_at_k AND lower or equal tokens
    """
    pass1, token1 = point1
    pass2, token2 = point2
    
    # point1 is dominated by point2 if:
    # (pass2 > pass1 and token2 <= token1) OR (pass2 >= pass1 and token2 < token1)
    if (pass2 > pass1 and token2 <= token1) or (pass2 >= pass1 and token2 < token1):
        return True
    return False

def find_pareto_front(data: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Find Pareto front individuals.
    Returns list of (workflow_roles, pass_at_k, tokens) tuples.
    """
    pareto_front = []
    
    for i, (workflow, pass_at_k, tokens) in enumerate(data):
        is_pareto = True
        
        # Check if this point is dominated by any other point
        for j, (other_workflow, other_pass, other_tokens) in enumerate(data):
            if i == j:
                continue
            
            if is_dominated((pass_at_k, tokens), (other_pass, other_tokens)):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_front.append((workflow, pass_at_k, tokens))
    
    # Sort by pass_at_k descending, then by tokens ascending
    pareto_front.sort(key=lambda x: (-x[1], x[2]))
    
    return pareto_front

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pareto.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    data = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                workflow = row['workflow_roles']
                pass_at_k = float(row['pass_at_k'])
                tokens = float(row['tokens'])
                data.append((workflow, pass_at_k, tokens))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    pareto_front = find_pareto_front(data)
    
    print(f"Total individuals: {len(data)}")
    print(f"Pareto front size: {len(pareto_front)}")
    print("\n" + "="*80)
    print("PARETO FRONT INDIVIDUALS:")
    print("="*80)
    print(f"{'Workflow':<60} {'pass_at_k':>12} {'tokens':>12}")
    print("-"*80)
    
    for workflow, pass_at_k, tokens in pareto_front:
        print(f"{workflow:<60} {pass_at_k:>12.6f} {tokens:>12.2f}")

if __name__ == "__main__":
    main()






