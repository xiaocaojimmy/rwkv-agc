#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STDP 消融实验

验证各组件的必要性：
1. STDP 噪声（可塑性模拟）
2. 重放缓冲（旧任务样本）
3. 巩固阶段（STDP epochs）
4. 交替学习（vs 同时学习）
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import random

print('='*70)
print('STDP 消融实验')
print('='*70)
print()

# ==================== 配置 ====================
NUM_TASKS = 8
BP_EPOCHS = 150
STDP_EPOCHS = 50
BATCH_SIZE = 4
STDP_WEIGHT = 0.05

print(f'任务数：{NUM_TASKS}')
print(f'BP 学习：{BP_EPOCHS} epochs')
print(f'STDP 巩固：{STDP_EPOCHS} epochs')
print()

# ==================== 任务 ====================
def get_task(tid):
    x = torch.randn(BATCH_SIZE, 32) * 0.5
    if tid % 4 == 0: y = x * 2.0 + 1.0
    elif tid % 4 == 1: y = x * (-1.5) + 0.5
    elif tid % 4 == 2: y = x ** 2
    else: y = torch.abs(x * 0.5)
    return x, y

# ==================== 训练方法 ====================

def train_baseline():
    """基线：仅 BP"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    for tid in range(NUM_TASKS):
        for _ in range(BP_EPOCHS + STDP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_losses.append(loss.item())
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    return {'forgetting': forgetting, 'avg_train': sum(train_losses)/NUM_TASKS}

def train_no_stdp_noise():
    """无 STDP 噪声（只有重放）"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    
    train_losses = []
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_losses.append(loss.item())
        
        # 添加重放
        for _ in range(20):
            x, y = get_task(tid)
            replay_buffer.append((x, y))
        
        # 仅重放（无 STDP 噪声）
        for _ in range(STDP_EPOCHS):
            batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
            for x, y in batch:
                pred = model(x)
                loss = ((pred - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    return {'forgetting': forgetting, 'avg_train': sum(train_losses)/NUM_TASKS}

def train_no_replay():
    """无重放缓冲（只有 STDP 噪声）"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_losses.append(loss.item())
        
        # STDP 巩固（无重放，随机噪声）
        for _ in range(STDP_EPOCHS):
            with torch.no_grad():
                delta = torch.randn_like(model.weight) * 0.001
                model.weight.copy_(
                    (1 - STDP_WEIGHT) * model.weight + 
                    STDP_WEIGHT * (model.weight + delta)
                )
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    return {'forgetting': forgetting, 'avg_train': sum(train_losses)/NUM_TASKS}

def train_simultaneous():
    """同时学习（BP+STDP 同时，非交替）"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    for tid in range(NUM_TASKS):
        for _ in range(BP_EPOCHS + STDP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # 同时应用 STDP
            with torch.no_grad():
                delta = torch.randn_like(model.weight) * 0.001
                model.weight.copy_(
                    (1 - STDP_WEIGHT) * model.weight + 
                    STDP_WEIGHT * (model.weight + delta)
                )
        train_losses.append(loss.item())
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    return {'forgetting': forgetting, 'avg_train': sum(train_losses)/NUM_TASKS}

def train_full():
    """完整 BP+STDP 交替（所有组件）"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    
    train_losses = []
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_losses.append(loss.item())
        
        # 添加重放
        for _ in range(20):
            x, y = get_task(tid)
            replay_buffer.append((x, y, tid))
        
        # STDP 巩固（有重放 + 噪声）
        for _ in range(STDP_EPOCHS):
            batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
            for x, y, _ in batch:
                pred = model(x)
                loss = ((pred - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                with torch.no_grad():
                    delta = torch.randn_like(model.weight) * 0.001
                    model.weight.copy_(
                        (1 - STDP_WEIGHT) * model.weight + 
                        STDP_WEIGHT * (model.weight + delta)
                    )
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    return {'forgetting': forgetting, 'avg_train': sum(train_losses)/NUM_TASKS}

# ==================== 运行实验 ====================

print('运行消融实验...')
print()

results = {}

print('1. 基线（仅 BP）...')
results['baseline'] = train_baseline()
print(f'   遗忘量：{results["baseline"]["forgetting"]:.6f}')

print('2. 无 STDP 噪声（只有重放）...')
results['no_stdp_noise'] = train_no_stdp_noise()
print(f'   遗忘量：{results["no_stdp_noise"]["forgetting"]:.6f}')

print('3. 无重放缓冲（只有 STDP 噪声）...')
results['no_replay'] = train_no_replay()
print(f'   遗忘量：{results["no_replay"]["forgetting"]:.6f}')

print('4. 同时学习（非交替）...')
results['simultaneous'] = train_simultaneous()
print(f'   遗忘量：{results["simultaneous"]["forgetting"]:.6f}')

print('5. 完整 BP+STDP 交替...')
results['full'] = train_full()
print(f'   遗忘量：{results["full"]["forgetting"]:.6f}')

print()

# ==================== 结果分析 ====================

print('='*70)
print('消融实验结果')
print('='*70)
print()

baseline_f = results['baseline']['forgetting']

print(f'{"配置":<25} | {"遗忘量":<15} | {"相对基线改善"}')
print(f'{"-"*25}-+-{"-"*15}-+-{"-"*15}')

for name, r in results.items():
    improvement = (baseline_f - r['forgetting']) / baseline_f * 100 if baseline_f > 0 else 0
    display_name = {
        'baseline': '基线（仅 BP）',
        'no_stdp_noise': '无 STDP 噪声（只有重放）',
        'no_replay': '无重放缓冲（只有 STDP）',
        'simultaneous': '同时学习（非交替）',
        'full': '完整 BP+STDP 交替'
    }.get(name, name)
    
    print(f'{display_name:<25} | {r["forgetting"]:<15.6f} | {improvement:>14.1f}%')

print()

# ==================== 组件贡献分析 ====================

print('【组件贡献分析】')
print()

# STDP 噪声贡献
full_f = results['full']['forgetting']
no_noise_f = results['no_stdp_noise']['forgetting']
noise_contribution = (no_noise_f - full_f) / baseline_f * 100
print(f'STDP 噪声贡献：{noise_contribution:.1f}%')

# 重放缓冲贡献
no_replay_f = results['no_replay']['forgetting']
replay_contribution = (no_replay_f - full_f) / baseline_f * 100
print(f'重放缓冲贡献：{replay_contribution:.1f}%')

# 交替学习贡献
sim_f = results['simultaneous']['forgetting']
alternate_contribution = (sim_f - full_f) / baseline_f * 100
print(f'交替学习贡献：{alternate_contribution:.1f}%')

print()

# 判断最关键组件
components = [
    ('STDP 噪声', noise_contribution),
    ('重放缓冲', replay_contribution),
    ('交替学习', alternate_contribution)
]
most_important = max(components, key=lambda x: x[1])
print(f'🏆 最关键组件：{most_important[0]} (贡献{most_important[1]:.1f}%)')

print()

# ==================== 保存结果 ====================

summary = {
    'experiment': 'STDP 消融实验',
    'config': {
        'tasks': NUM_TASKS,
        'bp_epochs': BP_EPOCHS,
        'stdp_epochs': STDP_EPOCHS,
        'stdp_weight': STDP_WEIGHT
    },
    'results': results,
    'component_contributions': {
        'stdp_noise': noise_contribution,
        'replay': replay_contribution,
        'alternate': alternate_contribution
    },
    'most_important_component': most_important[0],
    'timestamp': datetime.now().isoformat()
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path('results') / f'stdp_ablation_study_{ts}.json'
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print('='*70)
print('结果已保存')
print('='*70)
print(f'文件：{output_file.absolute()}')
print()

import time
print('等待 10 秒...')
for i in range(10, 0, -1):
    print(f'{i}...', end=' ', flush=True)
    time.sleep(1)
print('完成')
