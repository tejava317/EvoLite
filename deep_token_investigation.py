#!/usr/bin/env python3
"""
토큰 사용량 심층 조사: 실제로 무엇이 저장되는가?
"""
import csv

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

def investigate_token_meaning():
    """토큰 값의 실제 의미 조사"""
    print("=" * 80)
    print("토큰 사용량 심층 조사")
    print("=" * 80)
    
    # 모든 iteration 데이터 수집
    all_workflows = []
    for it in [5, 10, 15, 20, 25, 30]:
        all_workflows.extend(load_checkpoint(it))
    
    # 단일 에이전트 분석
    print("\n1. 단일 에이전트 workflow 분석:")
    print("-" * 80)
    single_agent = [wf for wf in all_workflows if len(extract_agents(wf['workflow'])) == 1]
    if single_agent:
        tokens = [wf['tokens'] for wf in single_agent]
        print(f"  총 {len(single_agent)}개 샘플")
        print(f"  평균 토큰: {sum(tokens)/len(tokens):.0f}")
        print(f"  최소: {min(tokens):.0f}")
        print(f"  최대: {max(tokens):.0f}")
        print(f"  중앙값: {sorted(tokens)[len(tokens)//2]:.0f}")
        
        # 분포 분석
        print(f"\n  토큰 분포:")
        ranges = [(0, 7000), (7000, 10000), (10000, 20000), (20000, 50000)]
        for low, high in ranges:
            count = sum(1 for t in tokens if low <= t < high)
            if count > 0:
                print(f"    {low:,} ~ {high:,}: {count}개 ({count/len(tokens)*100:.1f}%)")
    
    # 최고 성능 workflow 분석
    print("\n2. 최고 성능 workflow (score >= 0.8) 분석:")
    print("-" * 80)
    top_workflows = [wf for wf in all_workflows if wf['pass_at_k'] >= 0.8]
    
    for num_agents in sorted(set(len(extract_agents(wf['workflow'])) for wf in top_workflows)):
        wfs = [wf for wf in top_workflows if len(extract_agents(wf['workflow'])) == num_agents]
        tokens = [wf['tokens'] for wf in wfs]
        print(f"\n  {num_agents}개 에이전트 ({len(wfs)}개 샘플):")
        print(f"    평균 토큰: {sum(tokens)/len(tokens):.0f}")
        print(f"    에이전트당 평균: {sum(tokens)/len(tokens)/num_agents:.0f}")
        print(f"    최소: {min(tokens):.0f}")
        print(f"    최대: {max(tokens):.0f}")
        
        # 단일 에이전트와 비교
        if single_agent:
            single_avg = sum([wf['tokens'] for wf in single_agent]) / len(single_agent)
            expected = single_avg * num_agents
            actual = sum(tokens) / len(tokens)
            ratio = actual / expected if expected > 0 else 0
            print(f"    예상 (단일 × {num_agents}): {expected:.0f}")
            print(f"    실제: {actual:.0f}")
            print(f"    비율: {ratio:.2f} ({'절약' if ratio < 1 else '초과'})")
    
    # 특정 workflow 상세 분석
    print("\n3. 특정 workflow 상세 분석:")
    print("-" * 80)
    
    # Regression Sentinel -> Solution Ranker -> Solution Ranker
    target_wf = "Regression Sentinel -> Solution Ranker -> Solution Ranker"
    matches = [wf for wf in all_workflows if wf['workflow'] == target_wf]
    
    if matches:
        print(f"\n  '{target_wf}':")
        print(f"    발견 횟수: {len(matches)}")
        tokens = [wf['tokens'] for wf in matches]
        print(f"    토큰 값: {tokens[0]:.0f} (모든 iteration에서 동일)")
        print(f"    에이전트 수: 3")
        print(f"    에이전트당: {tokens[0]/3:.0f} 토큰")
        
        # 단일 Regression Sentinel이 있다면 비교
        regression_single = [wf for wf in single_agent if 'Regression Sentinel' in wf['workflow']]
        if regression_single:
            reg_tokens = [wf['tokens'] for wf in regression_single]
            reg_avg = sum(reg_tokens) / len(reg_tokens)
            print(f"\n    단일 'Regression Sentinel' 비교:")
            print(f"      단일 평균: {reg_avg:.0f} 토큰")
            print(f"      Workflow 첫 번째 에이전트 예상: {reg_avg:.0f} 토큰")
            print(f"      Workflow 전체: {tokens[0]:.0f} 토큰")
            print(f"      → 첫 번째 에이전트만 해도 {reg_avg:.0f} 토큰이 필요한데,")
            print(f"        전체가 {tokens[0]:.0f} 토큰이라는 건 말이 안 됨!")
    
    # 가설 검증
    print("\n4. 가설 검증:")
    print("-" * 80)
    print("\n  가설 1: 체크포인트의 토큰은 '마지막 문제 하나'에 대한 값")
    print("    → 이 경우, 17,000 토큰은 한 문제에 대한 3개 에이전트의 총합")
    print("    → 에이전트당 약 5,600 토큰")
    print("    → 하지만 단일 에이전트가 7,400 토큰을 사용한다면...")
    print("    → 첫 번째 에이전트만 해도 7,400 토큰이 필요할 텐데,")
    print("    → 전체가 17,000 토큰이라는 건 여전히 이상함!")
    
    print("\n  가설 2: 체크포인트의 토큰은 '30개 문제의 평균'")
    print("    → 이 경우, 17,000 토큰은 30개 문제의 평균")
    print("    → 문제당 약 567 토큰")
    print("    → 에이전트당 문제당 약 189 토큰")
    print("    → 이건 더 말이 안 됨 (너무 적음)")
    
    print("\n  가설 3: 체크포인트의 토큰은 '30개 문제의 총합'")
    print("    → 이 경우, 17,000 토큰은 30개 문제의 총합")
    print("    → 문제당 약 567 토큰")
    print("    → 에이전트당 문제당 약 189 토큰")
    print("    → 이것도 말이 안 됨")
    
    print("\n  가설 4: 실제로는 다른 평가 방식을 사용")
    print("    → evaluation_server를 사용하는 경우")
    print("    → 배치 처리로 인한 토큰 절약")
    print("    → 또는 다른 최적화")
    
    # 실제 코드 동작 확인 필요
    print("\n5. 결론:")
    print("-" * 80)
    print("  코드를 다시 확인한 결과:")
    print("  - quick_evaluate는 직접 workflow.run()을 호출")
    print("  - workflow.run()은 각 문제마다 실행")
    print("  - workflow.total_tokens는 마지막 실행의 토큰만 저장")
    print("  - 따라서 체크포인트의 토큰은 '마지막 문제 하나'에 대한 값")
    print("\n  하지만 사용자의 지적이 맞습니다:")
    print("  - 단일 에이전트가 7,400 토큰을 사용")
    print("  - 3개 에이전트 workflow가 17,000 토큰을 사용")
    print("  - 첫 번째 에이전트만 해도 7,400 토큰이 필요한데,")
    print("  - 전체가 17,000 토큰이라는 건 논리적으로 맞지 않음!")
    print("\n  가능한 설명:")
    print("  1. 단일 에이전트와 workflow의 평가 방식이 다를 수 있음")
    print("  2. workflow 내에서 prompt가 최적화되어 토큰이 줄어들 수 있음")
    print("  3. 실제로는 다른 평가 방식을 사용할 수 있음")
    print("  4. 체크포인트의 토큰 값이 실제와 다를 수 있음 (버그?)")

if __name__ == "__main__":
    investigate_token_meaning()
