#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STDP 扩展验证 - 多任务数测试

测试任务数：4, 8, 16, 32
验证 BP+STDP 在不同规模下的表现
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import random

print('='*70)
print('STDP 扩展验证 - 多任务数测试')
print('='*70)
print()

# ==================== 配置 ====================
TASK_LIST = [4, 8, 16, 32]
BP_EPOCHS = 150
STDP_EPOCHS = 50
BATCH_SIZE = 4

print(f'测试任务数：{TASK_LIST}')
print(f'BP 学习：{BP_EPOCHS} epochs')
print(f'STDP 巩固：{STDP_EPOCHS} epochs')
print()

# ==================== 任务生成器 ====================
def get_task(tid, input_dim=32):
    """生成多样化任务"""
    x = torch.randn(BATCH_SIZE, input_dim) * 0.5
    
    # 基础任务类型
    task_type = tid % 8
    if task_type == 0: y = x * 2.0 + 1.0
    elif task_type == 1: y = x * (-1.5) + 0.5
    elif task_type == 2: y = x ** 2
    elif task_type == 3: y = torch.abs(x * 0.5)
    elif task_type == 4: y = torch.sin(x * 0.5) * 0.5 + 0.5
    elif task_type == 5: y = x * torch.abs(x)
    elif task_type == 6: y = 1.0 / (1.0 + x ** 2)
    else: y = (x > 0).float()
    
    return x, y

# ==================== 训练函数 ====================
def train_bp_only(num_tasks):
    """仅 BP 训练"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    for tid in range(num_tasks):
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
        for tid in range(num_tasks):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / num_tasks
    return {
        'method': 'BP',
        'tasks': num_tasks,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / num_tasks
    }

def train_bp_stdp(num_tasks):
    """BP+STDP 交替训练"""
    torch.manual_seed(42)
    random.seed(42)
    
    model = nn.Linear(32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    replay_buffer = []
    stdp_weight = 0.05
    
    train_losses = []
    for tid in range(num_tasks):
        # BP 学习
        for _ in range(BP_EPOCHS):
            x, y = get_task(tid)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_losses.append(loss.item())
        
        # 添加到重放缓冲
        for _ in range(20):
            x, y = get_task(tid)
            replay_buffer.append((x, y, tid))
        
        # STDP 巩固
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
                        (1 - stdp_weight) * model.weight + 
                        stdp_weight * (model.weight + delta)
                    )
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(num_tasks):
            total = 0
            for _ in range(50):
                x, y = get_task(tid)
                total += ((model(x) - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / num_tasks
    return {
        'method': 'BP+STDP',
        'tasks': num_tasks,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'forgetting': forgetting,
        'avg_train': sum(train_losses) / num_tasks
    }

# ==================== 运行实验 ====================

all_results = {'bp': [], 'bp_stdp': []}

for num_tasks in TASK_LIST:
    print(f'{"="*70}')
    print(f'测试 {num_tasks} 任务')
    print(f'{"="*70}')
    print()
    
    # 仅 BP
    print(f'  运行仅 BP...')
    result_bp = train_bp_only(num_tasks)
    all_results['bp'].append(result_bp)
    print(f'    遗忘量：{result_bp["forgetting"]:.6f}')
    
    # BP+STDP
    print(f'  运行 BP+STDP...')
    result_stdp = train_bp_stdp(num_tasks)
    all_results['bp_stdp'].append(result_stdp)
    print(f'    遗忘量：{result_stdp["forgetting"]:.6f}')
    
    # 改善率
    improvement = (result_bp['forgetting'] - result_stdp['forgetting']) / result_bp['forgetting'] * 100 if result_bp['forgetting'] > 0 else 0
    print(f'    改善率：{improvement:.1f}%')
    print()

# ==================== 结果汇总 ====================

print('='*70)
print('结果汇总')
print('='*70)
print()

print(f'{"任务数":<10} | {"仅 BP 遗忘":<15} | {"BP+STDP 遗忘":<15} | {"改善率"}')
print(f'{"-"*10}-+-{"-"*15}-+-{"-"*15}-+-{"-"*10}')

for i, num_tasks in enumerate(TASK_LIST):
    bp_f = all_results['bp'][i]['forgetting']
    stdp_f = all_results['bp_stdp'][i]['forgetting']
    improvement = (bp_f - stdp_f) / bp_f * 100 if bp_f > 0 else 0
    print(f'{num_tasks:<10} | {bp_f:<15.6f} | {stdp_f:<15.6f} | {improvement:>9.1f}%')

print()

# ==================== 趋势分析 ====================

print('【趋势分析】')
print()

# 遗忘量随任务数变化
print('遗忘量 vs 任务数:')
print('  仅 BP:')
for i, num_tasks in enumerate(TASK_LIST):
    print(f'    {num_tasks}任务：{all_results["bp"][i]["forgetting"]:.6f}')
print('  BP+STDP:')
for i, num_tasks in enumerate(TASK_LIST):
    print(f'    {num_tasks}任务：{all_results["bp_stdp"][i]["forgetting"]:.6f}')

print()

# 改善率趋势
print('改善率 vs 任务数:')
improvements = []
for i, num_tasks in enumerate(TASK_LIST):
    bp_f = all_results['bp'][i]['forgetting']
    stdp_f = all_results['bp_stdp'][i]['forgetting']
    imp = (bp_f - stdp_f) / bp_f * 100 if bp_f > 0 else 0
    improvements.append(imp)
    print(f'    {num_tasks}任务：{imp:.1f}%')

# 判断趋势
if improvements[0] < improvements[-1]:
    trend = '上升 → 任务数越多，STDP 效果越明显'
elif improvements[0] > improvements[-1]:
    trend = '下降 → 任务数越少，STDP 效果越明显'
else:
    trend = '稳定 → STDP 效果与任务数无关'
print(f'  趋势：{trend}')

print()

# ==================== 保存结果 ====================

summary = {
    'experiment': 'STDP 扩展验证 - 多任务数测试',
    'config': {
        'task_list': TASK_LIST,
        'bp_epochs': BP_EPOCHS,
        'stdp_epochs': STDP_EPOCHS,
        'stdp_weight': 0.05
    },
    'results': all_results,
    'trend': {
        'improvements': improvements,
        'description': trend
    },
    'timestamp': datetime.now().isoformat()
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path('results') / f'stdp_extended_validation_{ts}.json'
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
