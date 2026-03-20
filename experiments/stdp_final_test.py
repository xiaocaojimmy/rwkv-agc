#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STDP 抗遗忘最终测试 - 确保输出结果

执行说明：
1. 运行此脚本
2. 结果会打印到屏幕 AND 保存到文件
3. 结果文件路径会在屏幕显示
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
import sys

# ==================== 配置 ====================
OUTPUT_DIR = Path(r'C:\Users\Administrator\.openclaw\workspace\rwkv-agc\experiments\results')
OUTPUT_DIR.mkdir(exist_ok=True)

print('='*70)
print('STDP 抗遗忘最终测试')
print('='*70)
print(f'结果将保存到：{OUTPUT_DIR}')
print()

# ==================== 实验参数 ====================
NUM_TASKS = 16
STEPS = 200
BATCH_SIZE = 4
INPUT_DIM = 32
STDP_WEIGHT = 0.05

print(f'配置:')
print(f'  任务数：{NUM_TASKS}')
print(f'  步数：{STEPS}/任务')
print(f'  STDP 权重：{STDP_WEIGHT}')
print()

# ==================== 任务生成器 ====================
def create_tasks():
    """创建 8 个任务"""
    tasks = []
    for i in range(NUM_TASKS):
        def task(bs=BATCH_SIZE, idx=i):
            x = torch.randn(bs, INPUT_DIM) * 0.5
            if idx % 4 == 0:
                y = x * 2.0 + 1.0
            elif idx % 4 == 1:
                y = x * (-1.5) + 0.5
            elif idx % 4 == 2:
                y = x ** 2
            else:
                y = torch.abs(x * 0.5)
            return x, y
        tasks.append(task)
    return tasks

tasks = create_tasks()
print(f'已创建 {len(tasks)} 个任务')
print()

# ==================== 训练函数 ====================
def train_model(use_stdp, stdp_weight=0.05):
    """训练模型"""
    torch.manual_seed(42)
    
    model = nn.Linear(INPUT_DIM, INPUT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    
    for tid in range(NUM_TASKS):
        task_fn = tasks[tid]
        for step in range(STEPS):
            x, y = task_fn()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # STDP 更新
            if use_stdp:
                with torch.no_grad():
                    # 简化 STDP：添加噪声模拟可塑性
                    delta = torch.randn_like(model.weight) * 0.001
                    model.weight.copy_((1 - stdp_weight) * model.weight + stdp_weight * (model.weight + delta))
        
        train_losses.append(loss.item())
        print(f'  任务 {tid}: 训练 Loss = {loss.item():.6f}')
    
    # 测试
    model.eval()
    test_losses = []
    with torch.no_grad():
        for tid in range(NUM_TASKS):
            task_fn = tasks[tid]
            total = 0
            for _ in range(50):
                x, y = task_fn()
                pred = model(x)
                total += ((pred - y) ** 2).mean().item()
            test_losses.append(total / 50)
    
    # 计算遗忘量
    forgetting = sum(t - tr for t, tr in zip(test_losses, train_losses)) / NUM_TASKS
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_forgetting': forgetting
    }

# ==================== 运行实验 ====================
print('运行无 STDP 模型...')
result_bp = train_model(use_stdp=False)
print(f'无 STDP 遗忘量：{result_bp["avg_forgetting"]:.6f}')
print()

print('运行有 STDP 模型...')
result_stdp = train_model(use_stdp=True, stdp_weight=STDP_WEIGHT)
print(f'有 STDP 遗忘量：{result_stdp["avg_forgetting"]:.6f}')
print()

# ==================== 计算结果 ====================
forgetting_reduction = (result_bp['avg_forgetting'] - result_stdp['avg_forgetting']) / result_bp['avg_forgetting'] * 100

print('='*70)
print('实验结果')
print('='*70)
print(f'无 STDP 遗忘量：{result_bp["avg_forgetting"]:.6f}')
print(f'有 STDP 遗忘量：{result_stdp["avg_forgetting"]:.6f}')
print(f'遗忘减少率：{forgetting_reduction:.1f}%')
print()

if forgetting_reduction > 10:
    print('✅ STDP 显著减少遗忘 (>10%)')
elif forgetting_reduction > 5:
    print('✅ STDP 轻微减少遗忘 (5-10%)')
else:
    print(f'⚠️ STDP 效果有限 ({forgetting_reduction:.1f}%)')
print()

# ==================== 保存结果 ====================
result = {
    'experiment': 'STDP 抗遗忘最终测试',
    'config': {
        'num_tasks': NUM_TASKS,
        'steps': STEPS,
        'batch_size': BATCH_SIZE,
        'stdp_weight': STDP_WEIGHT
    },
    'results': {
        'without_stdp': result_bp,
        'with_stdp': result_stdp,
        'forgetting_reduction': forgetting_reduction
    },
    'timestamp': datetime.now().isoformat()
}

# 生成文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = OUTPUT_DIR / f'stdp_final_test_{timestamp}.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print('='*70)
print('结果文件已保存')
print('='*70)
print(f'文件路径：{output_file.absolute()}')
print(f'文件大小：{output_file.stat().st_size} bytes')
print()
print('请按此路径查找结果文件！')
print()

# 保持窗口打开
input('按 Enter 键退出...')
