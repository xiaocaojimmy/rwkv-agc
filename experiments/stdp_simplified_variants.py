#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP+STDP 简化变体测试

测试：
1. 纯重放 + 交替（无 STDP 噪声）- 简化版
2. 强化巩固（BP:STDP = 1:1）- 当前是 3:1
3. 完整版 BP+STDP（对照）
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import random

print('='*70)
print('BP+STDP 简化变体测试')
print('='*70)
print()

# ==================== 配置 ====================
NUM_TASKS = 8
BP_EPOCHS_BASE = 150
STDP_EPOCHS_BASE = 50
BATCH_SIZE = 4
STDP_WEIGHT = 0.05

print(f'任务数：{NUM_TASKS}')
print(f'基础配置：BP {BP_EPOCHS_BASE} epochs, STDP {STDP_EPOCHS_BASE} epochs')
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

def train_pure_replay_alternate():
    """纯重放 + 交替（无 STDP 噪声）- 简化版"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    
    train_losses = []
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS_BASE):
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
        
        # 纯重放巩固（无 STDP 噪声）
        for _ in range(STDP_EPOCHS_BASE):
            batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
            for x, y, _ in batch:
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
    return {
        'method': '纯重放 + 交替',
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / NUM_TASKS
    }

def train_reinforced_consolidation():
    """强化巩固（BP:STDP = 1:1）"""
    torch.manual_seed(42)
    random.seed(42)
    
    # BP 和 STDP 各 100 epochs（1:1 比例）
    BP_EPOCHS = 100
    STDP_EPOCHS = 100
    
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
        
        # STDP 巩固（强化）
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
    return {
        'method': '强化巩固 (1:1)',
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / NUM_TASKS,
        'config': f'BP {BP_EPOCHS} + STDP {STDP_EPOCHS}'
    }

def train_full_alternate():
    """完整 BP+STDP 交替（对照）"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    
    train_losses = []
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS_BASE):
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
        
        # STDP 巩固
        for _ in range(STDP_EPOCHS_BASE):
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
    return {
        'method': '完整 BP+STDP (3:1)',
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / NUM_TASKS,
        'config': f'BP {BP_EPOCHS_BASE} + STDP {STDP_EPOCHS_BASE}'
    }

def train_baseline():
    """基线：仅 BP"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    for tid in range(NUM_TASKS):
        for _ in range(BP_EPOCHS_BASE + STDP_EPOCHS_BASE):
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
    return {
        'method': '基线（仅 BP）',
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / NUM_TASKS
    }

# ==================== 运行实验 ====================

print('运行实验...')
print()

results = {}

print('1. 基线（仅 BP）...')
results['baseline'] = train_baseline()
print(f'   遗忘量：{results["baseline"]["forgetting"]:.6f}')

print('2. 纯重放 + 交替（简化版）...')
results['pure_replay'] = train_pure_replay_alternate()
print(f'   遗忘量：{results["pure_replay"]["forgetting"]:.6f}')

print('3. 强化巩固 (1:1)...')
results['reinforced'] = train_reinforced_consolidation()
print(f'   遗忘量：{results["reinforced"]["forgetting"]:.6f}')

print('4. 完整 BP+STDP (3:1)...')
results['full'] = train_full_alternate()
print(f'   遗忘量：{results["full"]["forgetting"]:.6f}')

print()

# ==================== 结果分析 ====================

print('='*70)
print('实验结果')
print('='*70)
print()

baseline_f = results['baseline']['forgetting']

print(f'{"方法":<25} | {"遗忘量":<15} | {"训练 Loss":<12} | {"改善率"}')
print(f'{"-"*25}-+-{"-"*15}-+-{"-"*12}-+-{"-"*10}')

for name, r in results.items():
    improvement = (baseline_f - r['forgetting']) / baseline_f * 100 if baseline_f > 0 else 0
    config = r.get('config', '-')
    print(f'{r["method"]:<25} | {r["forgetting"]:<15.6f} | {r["avg_train"]:<12.6f} | {improvement:>9.1f}%')

print()

# ==================== 关键对比 ====================

print('【关键对比】')
print()

# 简化版 vs 完整版
pure_f = results['pure_replay']['forgetting']
full_f = results['full']['forgetting']
diff_pure_full = (full_f - pure_f) / baseline_f * 100

print('1. 简化版（纯重放）vs 完整版（BP+STDP）:')
print(f'   纯重放：{pure_f:.6f}')
print(f'   完整版：{full_f:.6f}')
if abs(diff_pure_full) < 5:
    print(f'   差异：{diff_pure_full:+.1f}% → ✅ 简化版与完整版相当！')
elif diff_pure_full > 0:
    print(f'   差异：{diff_pure_full:+.1f}% → ⚠️ 简化版略差')
else:
    print(f'   差异：{diff_pure_full:+.1f}% → 🎉 简化版更优！')

print()

# 强化巩固 vs 标准
rein_f = results['reinforced']['forgetting']
diff_rein_full = (full_f - rein_f) / baseline_f * 100

print('2. 强化巩固 (1:1) vs 标准 (3:1):')
print(f'   强化：{rein_f:.6f} (BP:STDP = 1:1)')
print(f'   标准：{full_f:.6f} (BP:STDP = 3:1)')
if diff_rein_full > 5:
    print(f'   差异：{diff_rein_full:+.1f}% → ✅ 强化巩固更优！')
elif diff_rein_full > -5:
    print(f'   差异：{diff_rein_full:+.1f}% → ⚠️ 两者相当')
else:
    print(f'   差异：{diff_rein_full:+.1f}% → ⚠️ 标准更优')

print()

# 最佳方法
best = min(results.items(), key=lambda x: x[1]['forgetting'])
print(f'🏆 最佳方法：{best[1]["method"]} (遗忘量={best[1]["forgetting"]:.6f})')

print()

# ==================== 建议 ====================

print('【建议】')
print()

if abs(diff_pure_full) < 5:
    print('✅ 推荐：纯重放 + 交替（简化版）')
    print('   理由：效果与完整版相当，实现更简单，无需 STDP 噪声')
elif diff_pure_full > 0:
    print('⚠️ 推荐：完整版 BP+STDP')
    print('   理由：STDP 噪声有额外贡献')
else:
    print('🎉 推荐：纯重放 + 交替（简化版）')
    print('   理由：不仅简单，效果还更好！')

print()

if diff_rein_full > 5:
    print('✅ 推荐：强化巩固 (1:1)')
    print('   理由：更多巩固时间带来更好效果')
else:
    print('⚠️ 标准巩固 (3:1) 已足够')
    print('   理由：增加巩固时间收益有限')

print()

# ==================== 保存结果 ====================

summary = {
    'experiment': 'BP+STDP 简化变体测试',
    'config': {
        'tasks': NUM_TASKS,
        'bp_epochs_base': BP_EPOCHS_BASE,
        'stdp_epochs_base': STDP_EPOCHS_BASE,
        'stdp_weight': STDP_WEIGHT
    },
    'results': results,
    'comparisons': {
        'pure_vs_full_diff': diff_pure_full,
        'reinforced_vs_standard_diff': diff_rein_full,
        'best_method': best[1]['method']
    },
    'recommendation': '纯重放 + 交替' if abs(diff_pure_full) < 5 else '完整 BP+STDP',
    'timestamp': datetime.now().isoformat()
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path('results') / f'stdp_simplified_variants_{ts}.json'
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
