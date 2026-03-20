#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STDP 与 BP 交替学习 - 方向 2

方案：BP 学新任务 → STDP 巩固旧记忆
- BP 负责快速学习新任务
- STDP 负责巩固已有知识，减少遗忘
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import random

print('='*70)
print('STDP 与 BP 交替学习实验')
print('='*70)
print()

# ==================== 配置 ====================
NUM_TASKS = 8
BP_EPOCHS = 150       # BP 学习新任务
STDP_EPOCHS = 50      # STDP 巩固旧记忆
BATCH_SIZE = 4

print(f'任务数：{NUM_TASKS}')
print(f'BP 学习：{BP_EPOCHS} epochs/任务')
print(f'STDP 巩固：{STDP_EPOCHS} epochs/任务')
print()

# ==================== 任务 ====================
def get_task(tid):
    x = torch.randn(BATCH_SIZE, 32) * 0.5
    if tid % 4 == 0: y = x * 2.0 + 1.0
    elif tid % 4 == 1: y = x * (-1.5) + 0.5
    elif tid % 4 == 2: y = x ** 2
    else: y = torch.abs(x * 0.5)
    return x, y

# 重放缓冲（用于 STDP 巩固）
replay_buffer = []

def add_to_replay(tid):
    """添加任务样本到重放缓冲"""
    for _ in range(20):
        x, y = get_task(tid)
        replay_buffer.append((x, y, tid))

# ==================== 训练方法 ====================

def train_with_bp(model, task_id, epochs):
    """用 BP 学习新任务"""
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    task_fn = get_task
    
    for _ in range(epochs):
        x, y = task_fn(task_id)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item()

def consolidate_with_stdp(model, epochs, stdp_weight=0.05):
    """用 STDP 巩固记忆（从重放缓冲采样）"""
    if not replay_buffer:
        return
    
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    for _ in range(epochs):
        # 从重放缓冲随机采样
        batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
        
        for x, y, _ in batch:
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # STDP 更新
            with torch.no_grad():
                delta = torch.randn_like(model.weight) * 0.001
                model.weight.copy_(
                    (1 - stdp_weight) * model.weight + 
                    stdp_weight * (model.weight + delta)
                )

def train_bp_only(model, task_id):
    """仅用 BP 训练（对照组）"""
    return train_with_bp(model, task_id, BP_EPOCHS + STDP_EPOCHS)

# ==================== 实验 1: 仅 BP ====================
print('【实验 1】仅 BP（对照组）')
torch.manual_seed(42)
random.seed(42)
replay_buffer.clear()

model_bp = nn.Linear(32, 32)
train_losses_bp = []

for tid in range(NUM_TASKS):
    loss = train_bp_only(model_bp, tid)
    train_losses_bp.append(loss)
    print(f'  任务 {tid}: Loss = {loss:.6f}')

# 测试
model_bp.eval()
test_losses_bp = []
with torch.no_grad():
    for tid in range(NUM_TASKS):
        total = 0
        for _ in range(50):
            x, y = get_task(tid)
            total += ((model_bp(x) - y) ** 2).mean().item()
        test_losses_bp.append(total / 50)

forgetting_bp = sum(t - tr for t, tr in zip(test_losses_bp, train_losses_bp)) / NUM_TASKS
print(f'仅 BP 遗忘量：{forgetting_bp:.6f}')
print()

# ==================== 实验 2: BP + STDP 交替 ====================
print('【实验 2】BP 学习 + STDP 巩固')
torch.manual_seed(42)
random.seed(42)
replay_buffer.clear()

model_alt = nn.Linear(32, 32)
train_losses_alt = []

for tid in range(NUM_TASKS):
    # 1. BP 学习新任务
    loss = train_with_bp(model_alt, tid, BP_EPOCHS)
    train_losses_alt.append(loss)
    
    # 2. 添加当前任务到重放缓冲
    add_to_replay(tid)
    
    # 3. STDP 巩固所有记忆
    consolidate_with_stdp(model_alt, STDP_EPOCHS, stdp_weight=0.05)
    
    print(f'  任务 {tid}: BP Loss = {loss:.6f}, 重放缓冲={len(replay_buffer)}样本')

# 测试
model_alt.eval()
test_losses_alt = []
with torch.no_grad():
    for tid in range(NUM_TASKS):
        total = 0
        for _ in range(50):
            x, y = get_task(tid)
            total += ((model_alt(x) - y) ** 2).mean().item()
        test_losses_alt.append(total / 50)

forgetting_alt = sum(t - tr for t, tr in zip(test_losses_alt, train_losses_alt)) / NUM_TASKS
print(f'BP+STDP 遗忘量：{forgetting_alt:.6f}')
print()

# ==================== 对比 ====================
reduction = (forgetting_bp - forgetting_alt) / forgetting_bp * 100 if forgetting_bp > 0 else 0

print('='*70)
print('实验结果')
print('='*70)
print(f'仅 BP 遗忘量：{forgetting_bp:.6f}')
print(f'BP+STDP 遗忘量：{forgetting_alt:.6f}')
print(f'改善率：{reduction:.1f}%')
print()

if reduction > 20:
    print('✅ BP+STDP 交替显著改善 (>20%)')
elif reduction > 10:
    print('✅ BP+STDP 交替轻微改善 (10-20%)')
elif reduction > 5:
    print('✅ BP+STDP 交替略有改善 (5-10%)')
else:
    print(f'⚠️ BP+STDP 交替改善有限 ({reduction:.1f}%)')
print()

# ==================== 保存结果 ====================
result = {
    'experiment': 'STDP 与 BP 交替学习',
    'config': {
        'tasks': NUM_TASKS,
        'bp_epochs': BP_EPOCHS,
        'stdp_epochs': STDP_EPOCHS,
        'stdp_weight': 0.05
    },
    'results': {
        'bp_only': {'forgetting': forgetting_bp},
        'bp_stdp_alternate': {'forgetting': forgetting_alt},
        'improvement': reduction
    },
    'timestamp': datetime.now().isoformat()
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path('results') / f'stdp_bp_alternate_{ts}.json'
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print('='*70)
print('结果已保存')
print('='*70)
print(f'文件：{output_file.absolute()}')
print()

# 等待
import time
print('等待 10 秒...')
for i in range(10, 0, -1):
    print(f'{i}...', end=' ', flush=True)
    time.sleep(1)
print('完成')
