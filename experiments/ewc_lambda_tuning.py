#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EWC λ 调优实验

测试不同λ值下 EWC 的表现，找到最佳平衡点
并公平对比 BP+STDP
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import random

print('='*70)
print('EWC λ 调优实验')
print('='*70)
print()

# ==================== 配置 ====================
NUM_TASKS = 16
EPOCHS = 200
LAMBDA_VALUES = [0, 10, 50, 100, 200, 500, 1000]
BP_STDP_WEIGHT = 0.05

print(f'任务数：{NUM_TASKS}')
print(f'EWC λ 测试：{LAMBDA_VALUES}')
print(f'BP+STDP 权重：{BP_STDP_WEIGHT}')
print()

# ==================== 任务 ====================
def get_task_16(tid):
    x = torch.randn(4, 32) * 0.5
    if tid % 8 == 0: y = x * 2.0 + 1.0
    elif tid % 8 == 1: y = x * (-1.5) + 0.5
    elif tid % 8 == 2: y = x ** 2
    elif tid % 8 == 3: y = torch.abs(x * 0.5)
    elif tid % 8 == 4: y = torch.sin(x * 0.5) * 0.5 + 0.5
    elif tid % 8 == 5: y = x * torch.abs(x)
    elif tid % 8 == 6: y = 1.0 / (1.0 + x ** 2)
    elif tid % 8 == 7: y = (x > 0).float()
    return x, y

# ==================== EWC 训练 ====================
def train_ewc(lambda_ewc):
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    fisher = None
    prev_params = None
    train_losses = []
    
    for tid in range(NUM_TASKS):
        epoch_losses = []
        for _ in range(EPOCHS):
            x, y = get_task_16(tid)
            loss = ((model(x) - y) ** 2).mean()
            
            # EWC 惩罚
            ewc_loss = torch.tensor(0.0)
            if fisher is not None and prev_params is not None:
                for name, param in model.named_parameters():
                    if name in fisher and name in prev_params:
                        ewc_loss += (fisher[name] * (param - prev_params[name]) ** 2).sum()
                loss += lambda_ewc * ewc_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        
        train_losses.append(sum(epoch_losses[-10:]) / 10)  # 最后 10 个 epoch 平均
        
        # 更新 Fisher 和参数
        fisher = {}
        prev_params = {}
        for name, param in model.named_parameters():
            prev_params[name] = param.clone().detach()
            fisher[name] = torch.ones_like(param) * 0.01
    
    # 测试
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task_16(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    avg_train = sum(train_losses) / NUM_TASKS
    
    return {
        'lambda': lambda_ewc,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_train': avg_train,
        'forgetting': forgetting
    }

# ==================== BP+STDP 交替 ====================
def train_bp_stdp():
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    
    train_losses = []
    BP_EPOCHS = 150
    STDP_EPOCHS = 50
    
    for tid in range(NUM_TASKS):
        # BP 学习
        for _ in range(BP_EPOCHS):
            x, y = get_task_16(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_losses.append(loss.item())
        
        # 添加到重放缓冲
        for _ in range(20):
            x, y = get_task_16(tid)
            replay_buffer.append((x, y, tid))
        
        # STDP 巩固
        for _ in range(STDP_EPOCHS):
            batch = random.sample(replay_buffer, min(4, len(replay_buffer)))
            for x, y, _ in batch:
                pred = model(x)
                loss = ((pred - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                with torch.no_grad():
                    delta = torch.randn_like(model.weight) * 0.001
                    model.weight.copy_(
                        (1 - BP_STDP_WEIGHT) * model.weight + 
                        BP_STDP_WEIGHT * (model.weight + delta)
                    )
    
    # 测试
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            total = 0
            for _ in range(50):
                x, y = get_task_16(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    avg_train = sum(train_losses) / NUM_TASKS
    
    return {
        'method': 'BP+STDP',
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_train': avg_train,
        'forgetting': forgetting
    }

# ==================== 运行实验 ====================

results = []

print('运行 EWC (不同λ值)...')
for lambda_ewc in LAMBDA_VALUES:
    print(f'  λ = {lambda_ewc:4d}...', end=' ', flush=True)
    result = train_ewc(lambda_ewc)
    results.append(result)
    print(f'遗忘={result["forgetting"]:.6f}, 训练={result["avg_train"]:.6f}')

print()
print('运行 BP+STDP 交替...')
bp_stdp_result = train_bp_stdp()
print(f'  遗忘={bp_stdp_result["forgetting"]:.6f}, 训练={bp_stdp_result["avg_train"]:.6f}')
print()

# ==================== 结果分析 ====================

print('='*70)
print('实验结果')
print('='*70)
print()

print(f'{"方法":<15} | {"λ值":<8} | {"遗忘量":<15} | {"训练 Loss":<12} | {"改善率"}')
print(f'{"-"*15}-+-{"-"*8}-+-{"-"*15}-+-{"-"*12}-+-{"-"*10}')

# 基线（λ=0，即仅 BP）
baseline = results[0]

for r in results:
    improvement = (baseline['forgetting'] - r['forgetting']) / baseline['forgetting'] * 100
    print(f'{"EWC":<15} | {r["lambda"]:<8} | {r["forgetting"]:<15.6f} | {r["avg_train"]:<12.6f} | {improvement:>9.1f}%')

# BP+STDP
improvement_stdp = (baseline['forgetting'] - bp_stdp_result['forgetting']) / baseline['forgetting'] * 100
print(f'{"BP+STDP":<15} | {"-":<8} | {bp_stdp_result["forgetting"]:<15.6f} | {bp_stdp_result["avg_train"]:<12.6f} | {improvement_stdp:>9.1f}%')

print()

# ==================== 找最佳 EWC ====================

# 筛选：平均训练 Loss < 0.3（能正常学习新任务）
valid_ewc = [r for r in results if r['avg_train'] < 0.3]

if valid_ewc:
    best_ewc = min(valid_ewc, key=lambda x: x['forgetting'])
    print(f'✅ 最佳 EWC (训练 Loss < 0.3): λ={best_ewc["lambda"]}, 遗忘={best_ewc["forgetting"]:.6f}')
    
    stdp_better = bp_stdp_result['forgetting'] < best_ewc['forgetting']
    if stdp_better:
        print(f'✅ BP+STDP 优于最佳 EWC!')
    else:
        print(f'⚠️  最佳 EWC 仍优于 BP+STDP')
else:
    print('⚠️  所有 EWC 配置训练 Loss 都过高，λ 需要进一步调低')

print()

# ==================== 学习曲线分析 ====================

print('【学习曲线分析】后期任务 (8-15) 平均训练 Loss:')
print()

for r in results:
    late_train = sum(r['train_losses'][8:]) / 8
    print(f'  EWC λ={r["lambda"]:4d}: {late_train:.6f}')

late_bp_stdp = sum(bp_stdp_result['train_losses'][8:]) / 8
print(f'  BP+STDP:       {late_bp_stdp:.6f}')

print()

# ==================== 保存结果 ====================

all_results = {
    'experiment': 'EWC λ 调优 + BP+STDP 对比',
    'config': {
        'tasks': NUM_TASKS,
        'epochs': EPOCHS,
        'lambda_values': LAMBDA_VALUES,
        'bp_stdp_weight': BP_STDP_WEIGHT
    },
    'ewc_results': results,
    'bp_stdp_result': bp_stdp_result,
    'best_valid_ewc': best_ewc if valid_ewc else None,
    'timestamp': datetime.now().isoformat()
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path('results') / f'ewc_lambda_tuning_{ts}.json'
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

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
