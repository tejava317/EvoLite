#!/usr/bin/env python3
"""
토큰 사용량 분석: 왜 workflow로 합쳤을 때 토큰이 줄어드는지 분석
"""
import csv
import re
from collections import defaultdict

def load_checkpoint(iteration: int) -> list:
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
                })
    except FileNotFoundError:
        print(f"Warning: {path} not found")
    return workflows

def extract_agents(wf: str) -> list:
    """Extract list of agents from workflow string"""
    if ' -> ' in wf:
        return [a.strip() for a in wf.split(' -> ')]
    return [wf.strip()]

def analyze_token_patterns():
    """토큰 사용량 패턴 분석"""
    iterations = [5, 10, 15, 20, 25, 30]
    
    print("=" * 80)
    print("토큰 사용량 분석: Workflow vs 단일 에이전트")
    print("=" * 80)
    
    # 모든 iteration의 데이터 수집
    all_workflows = []
    for it in iterations:
        all_workflows.extend(load_checkpoint(it))
    
    # 에이전트 수별 토큰 분석
    print("\n1. 에이전트 수별 평균 토큰 사용량:")
    print("-" * 80)
    
    agent_count_stats = defaultdict(list)
    for wf in all_workflows:
        num_agents = len(extract_agents(wf['workflow']))
        agent_count_stats[num_agents].append(wf['tokens'])
    
    for num_agents in sorted(agent_count_stats.keys()):
        tokens = agent_count_stats[num_agents]
        avg = sum(tokens) / len(tokens)
        min_tok = min(tokens)
        max_tok = max(tokens)
        print(f"  {num_agents}개 에이전트: 평균 {avg:.0f} 토큰 (범위: {min_tok:.0f} ~ {max_tok:.0f}, 샘플 수: {len(tokens)})")
        print(f"    → 에이전트당 평균: {avg/num_agents:.0f} 토큰")
    
    # 최고 성능 workflow들의 토큰 분석
    print("\n2. 최고 성능 workflow들 (score >= 0.8)의 토큰 분석:")
    print("-" * 80)
    
    top_workflows = [wf for wf in all_workflows if wf['pass_at_k'] >= 0.8]
    print(f"  총 {len(top_workflows)}개의 최고 성능 workflow 발견")
    
    for num_agents in sorted(set(len(extract_agents(wf['workflow'])) for wf in top_workflows)):
        wfs = [wf for wf in top_workflows if len(extract_agents(wf['workflow'])) == num_agents]
        tokens = [wf['tokens'] for wf in wfs]
        avg = sum(tokens) / len(tokens)
        print(f"  {num_agents}개 에이전트: 평균 {avg:.0f} 토큰 (샘플: {len(wfs)}개)")
        print(f"    → 에이전트당 평균: {avg/num_agents:.0f} 토큰")
    
    # 단일 에이전트 workflow 분석
    print("\n3. 단일 에이전트 workflow 분석:")
    print("-" * 80)
    
    single_agent_wfs = [wf for wf in all_workflows if len(extract_agents(wf['workflow'])) == 1]
    if single_agent_wfs:
        single_tokens = [wf['tokens'] for wf in single_agent_wfs]
        avg_single = sum(single_tokens) / len(single_tokens)
        min_single = min(single_tokens)
        max_single = max(single_tokens)
        print(f"  단일 에이전트 평균: {avg_single:.0f} 토큰 (범위: {min_single:.0f} ~ {max_single:.0f})")
        print(f"  샘플 수: {len(single_agent_wfs)}")
    
    # 토큰 증가 패턴 분석
    print("\n4. 에이전트 수 증가에 따른 토큰 증가 패턴:")
    print("-" * 80)
    
    # 각 에이전트 수별로 평균 토큰 계산
    for num_agents in sorted(agent_count_stats.keys()):
        if num_agents == 1:
            continue
        tokens = agent_count_stats[num_agents]
        avg = sum(tokens) / len(tokens)
        
        # 단일 에이전트와 비교
        if 1 in agent_count_stats:
            single_avg = sum(agent_count_stats[1]) / len(agent_count_stats[1])
            expected = single_avg * num_agents
            actual = avg
            ratio = actual / expected if expected > 0 else 0
            print(f"  {num_agents}개 에이전트:")
            print(f"    예상 토큰 (단일 × {num_agents}): {expected:.0f}")
            print(f"    실제 평균 토큰: {actual:.0f}")
            print(f"    비율: {ratio:.2f} ({'절약' if ratio < 1 else '초과'})")
            print(f"    절약량: {expected - actual:.0f} 토큰 ({((expected - actual) / expected * 100):.1f}%)")
    
    # Iteration별 토큰 트렌드
    print("\n5. Iteration별 토큰 사용량 트렌드:")
    print("-" * 80)
    
    for it in iterations:
        wfs = load_checkpoint(it)
        if not wfs:
            continue
        tokens = [wf['tokens'] for wf in wfs]
        avg = sum(tokens) / len(tokens)
        print(f"  Iteration {it}: 평균 {avg:.0f} 토큰 (샘플: {len(wfs)}개)")
    
    # 특정 workflow들의 상세 분석
    print("\n6. 최고 성능 workflow들의 상세 토큰 분석:")
    print("-" * 80)
    
    top_4 = sorted([wf for wf in all_workflows if wf['pass_at_k'] >= 0.83], 
                   key=lambda x: (-x['pass_at_k'], x['tokens']))[:4]
    
    for idx, wf in enumerate(top_4, 1):
        agents = extract_agents(wf['workflow'])
        num_agents = len(agents)
        tokens_per_agent = wf['tokens'] / num_agents if num_agents > 0 else 0
        
        print(f"\n  Workflow #{idx}:")
        print(f"    구조: {' -> '.join(agents)}")
        print(f"    에이전트 수: {num_agents}")
        print(f"    총 토큰: {wf['tokens']:.0f}")
        print(f"    에이전트당 평균: {tokens_per_agent:.0f} 토큰")
        print(f"    성능: {wf['pass_at_k']:.4f}")
        
        # 만약 각 에이전트가 독립적으로 3만 토큰을 사용한다면?
        expected_independent = 30000 * num_agents
        if wf['tokens'] < expected_independent:
            savings = expected_independent - wf['tokens']
            savings_pct = (savings / expected_independent) * 100
            print(f"    예상 (독립 실행): {expected_independent:.0f} 토큰")
            print(f"    실제 절약: {savings:.0f} 토큰 ({savings_pct:.1f}%)")

if __name__ == "__main__":
    analyze_token_patterns()
