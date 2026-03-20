#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STDP 综合验证实验

补充验证：
1. STDP 权重扫描（找最优混合比例）
2. 消融实验（纯 STDP、无痕迹、无调制等）
3. 参数敏感性（tau、A_plus、A_minus）
4. 结果深度分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from pathlib import Path
import random
import json
from datetime import datetime
import copy
import math

# ==================== STDP 模块 ====================

class STDPPlasticity(nn.Module):
    def __init__(self, num_neurons: int, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 A_plus: float = 0.005, A_minus: float = 0.006, use_traces: bool = True):
        super().__init__()
        self.num_neurons = num_neurons
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.use_traces = use_traces
        
        self.register_buffer('pre_trace', torch.zeros(1, num_neurons))
        self.register_buffer('post_trace', torch.zeros(1, num_neurons))
        self.trace_decay = 0.95
    
    def update_traces(self, pre: torch.Tensor, post: torch.Tensor):
        if self.use_traces:
            with torch.no_grad():
                self.pre_trace = self.trace_decay * self.pre_trace + (1 - self.trace_decay) * pre
                self.post_trace = self.trace_decay * self.post_trace + (1 - self.trace_decay) * post
    
    def compute_update(self, pre: torch.Tensor, post: torch.Tensor, dopamine: float = 1.0):
        self.update_traces(pre, post)
        
        if self.use_traces:
            ltp = self.A_plus * self.pre_trace * post.unsqueeze(-1)
            ltd = self.A_minus * self.post_trace.unsqueeze(-1) * pre
        else:
            ltp = self.A_plus * pre * post.unsqueeze(-1)
            ltd = self.A_minus * post.unsqueeze(-1) * pre
        
        delta_w = (ltp - ltd) * dopamine
        return delta_w


# ==================== 模型定义 ====================

class ConnectomeAGC_STDP_Validation(nn.Module):
    def __init__(self, input_dim: int = 32, num_modules: int = 8, neurons_per_module: int = 32,
                 stdp_weight: float = 0.1, use_stdp: bool = True, use_traces: bool = True,
                 use_dopamine: bool = True, tau: float = 20.0, A_ratio: float = 1.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_modules = num_modules
        self.neurons_per_module = neurons_per_module
        self.total_neurons = num_modules * neurons_per_module
        
        self.stdp_weight = stdp_weight
        self.use_stdp = use_stdp
        self.use_dopamine = use_dopamine
        self.base_lr = 0.003
        self.momentum = 0.80
        self.state_lr = 0.01
        self.ei_target_ratio = 3.0
        
        # 小世界连接
        self.register_buffer('adjacency_matrix', self._generate_small_world(6, 0.15))
        self.register_buffer('sparse_mask', (self.adjacency_matrix > 0).float())
        
        # E/I 权重
        self.weight_exc = nn.Parameter(torch.rand(self.total_neurons, self.total_neurons) * 0.15 + 0.2)
        self.weight_inh = nn.Parameter(torch.rand(self.total_neurons, self.total_neurons) * 0.025 + 0.0375)
        
        # 状态
        self.states_real = nn.ParameterList([nn.Parameter(torch.randn(1, neurons_per_module) * 0.1) for _ in range(num_modules)])
        self.states_imag = nn.ParameterList([nn.Parameter(torch.randn(1, neurons_per_module) * 0.1) for _ in range(num_modules)])
        
        # 模块抑制
        self.inter_module_inhibition = nn.Parameter(torch.randn(num_modules, num_modules) * 0.01)
        with torch.no_grad():
            for i in range(num_modules):
                self.inter_module_inhibition[i, i] = 0.0
        
        # 调质
        self.dopamine = nn.Parameter(torch.tensor(0.5))
        self.expected_reward = nn.Parameter(torch.tensor(0.5))
        
        # 投影
        self.input_projection = nn.Sequential(nn.Linear(input_dim, self.total_neurons), nn.LayerNorm(self.total_neurons), nn.ReLU())
        self.output_projection = nn.Sequential(
            nn.Linear(self.total_neurons * 2, self.total_neurons), nn.ReLU(),
            nn.Linear(self.total_neurons, self.total_neurons // 2), nn.ReLU(),
            nn.Linear(self.total_neurons // 2, input_dim),
        )
        
        # STDP
        self.stdp = STDPPlasticity(
            self.total_neurons, 
            tau_plus=tau, 
            tau_minus=tau,
            A_plus=0.005, 
            A_minus=0.005 * A_ratio,
            use_traces=use_traces
        )
        
        self.register_buffer('momentum_real', torch.zeros(1, self.total_neurons))
        self.register_buffer('momentum_imag', torch.zeros(1, self.total_neurons))
        self.task_module_map = {i: [i] for i in range(8)}
    
    def _generate_small_world(self, neighbors: int, rewiring_prob: float) -> torch.Tensor:
        adj = torch.zeros(self.total_neurons, self.total_neurons)
        for i in range(self.total_neurons):
            for j in range(1, neighbors // 2 + 1):
                adj[i, (i + j) % self.total_neurons] = 1
                adj[(i + j) % self.total_neurons, i] = 1
        for i in range(self.total_neurons):
            for j in range(i + 1, self.total_neurons):
                if adj[i, j] == 1 and random.random() < rewiring_prob:
                    adj[i, j] = 0
                    adj[j, i] = 0
                    new_target = random.randint(0, self.total_neurons - 1)
                    if new_target != i and adj[i, new_target] == 0:
                        adj[i, new_target] = 1
                        adj[new_target, i] = 1
        return adj
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        input_signal = self.input_projection(x)
        state_real = torch.cat(list(self.states_real), dim=-1)
        state_imag = torch.cat(list(self.states_imag), dim=-1)
        
        # 模块掩码
        active_mask = torch.zeros(1, self.total_neurons, device=state_real.device)
        if task_id is not None and task_id in self.task_module_map:
            for m in self.task_module_map[task_id]:
                start = m * self.neurons_per_module
                end = start + self.neurons_per_module
                active_mask[:, start:end] = 1.0
        else:
            active_mask = torch.ones(1, self.total_neurons, device=state_real.device)
        
        # E/I 处理
        weight_exc = self.weight_exc * self.sparse_mask
        weight_inh = self.weight_inh * self.sparse_mask
        exc_input = F.linear(input_signal, weight_exc)
        inh_input = F.linear(input_signal, weight_inh)
        ei_signal = exc_input - inh_input / self.ei_target_ratio
        
        # STDP 更新
        if self.use_stdp:
            pre_activity = input_signal.mean(dim=0, keepdim=True)
            post_activity = ei_signal.mean(dim=0, keepdim=True)
            da = self.dopamine.item() if self.use_dopamine else 1.0
            stdp_update = self.stdp.compute_update(pre_activity, post_activity, da)
            
            with torch.no_grad():
                self.weight_exc = nn.Parameter(
                    (1 - self.stdp_weight) * self.weight_exc + self.stdp_weight * (self.weight_exc + stdp_update)
                )
        
        # 状态更新
        with torch.no_grad():
            self.momentum_real = self.momentum * self.momentum_real + self.state_lr * ei_signal * active_mask
            self.momentum_imag = self.momentum * self.momentum_imag + self.state_lr * ei_signal * active_mask
            state_real = state_real + self.momentum_real
            state_imag = state_imag + self.momentum_imag
            mag = torch.sqrt(state_real ** 2 + state_imag ** 2 + 1e-6)
            state_real = state_real / (mag + 1e-6) * 0.5
            state_imag = state_imag / (mag + 1e-6) * 0.5
        
        for i in range(self.num_modules):
            start = i * self.neurons_per_module
            end = start + self.neurons_per_module
            self.states_real[i].data = state_real[:, start:end]
            self.states_imag[i].data = state_imag[:, start:end]
        
        combined = torch.cat([state_real.expand(batch_size, -1), state_imag.expand(batch_size, -1)], dim=-1)
        return self.output_projection(combined)
    
    def update_dopamine(self, reward: float):
        if self.use_dopamine:
            with torch.no_grad():
                rpe = reward - self.expected_reward.item()
                self.dopamine += 0.1 * rpe
                self.expected_reward = nn.Parameter(0.9 * self.expected_reward + 0.1 * torch.tensor(reward))
                self.dopamine.clamp_(0, 1)
    
    def get_lr(self) -> float:
        return self.base_lr * (1.0 + 0.3 * (self.dopamine.item() - 0.5)) if self.use_dopamine else self.base_lr
    
    def ei_loss(self) -> torch.Tensor:
        exc = (self.weight_exc * self.sparse_mask).mean()
        inh = (self.weight_inh * self.sparse_mask).mean()
        return ((exc / (inh + 1e-6) - self.ei_target_ratio) ** 2).mean()
    
    def total_loss(self, task_loss: torch.Tensor) -> torch.Tensor:
        return task_loss + 0.01 * self.ei_loss()
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ==================== 任务生成器 ====================

class TaskGenerator:
    def __init__(self, input_dim: int = 32):
        self.tasks = [
            lambda bs: (torch.randn(bs, 32)*0.5, torch.randn(bs, 32)*2.0+1.0),
            lambda bs: (torch.randn(bs, 32)*0.5, torch.randn(bs, 32)*(-1.5)+0.5),
            lambda bs: (torch.randn(bs, 32)*0.3, torch.randn(bs, 32)**2),
            lambda bs: (torch.randn(bs, 32)*0.5, torch.abs(torch.randn(bs, 32)*0.5)),
            lambda bs: (torch.randn(bs, 32)*0.5, torch.sin(torch.randn(bs, 32)*0.5)*0.5+0.5),
            lambda bs: (torch.randn(bs, 32)*0.5, torch.randn(bs, 32)*torch.abs(torch.randn(bs, 32))),
            lambda bs: (torch.randn(bs, 32)*0.5, 1.0/(1.0+torch.randn(bs, 32)**2)),
            lambda bs: (torch.randn(bs, 32)*0.5, (torch.randn(bs, 32) > 0).float()),
        ]
        self.names = ['A:Linear', 'B:Inverse', 'C:Quadratic', 'D:Absolute', 
                      'E:Sine', 'F:SignSquare', 'G:Gaussian', 'H:Step']
    
    def get_task(self, tid: int):
        return self.tasks[tid % 8]
    
    def get_name(self, tid: int) -> str:
        return self.names[tid % 8]


# ==================== 综合验证实验 ====================

class STDPComprehensiveValidation:
    def __init__(self):
        self.task_gen = TaskGenerator(32)
        self.config = {'batch_size': 4, 'train_steps': 300}
    
    def train_8tasks(self, model) -> Dict[str, List[float]]:
        histories = {}
        for tid in range(8):
            opt = torch.optim.Adam(model.parameters(), lr=model.get_lr())
            task_fn = self.task_gen.get_task(tid)
            history = []
            
            for _ in range(self.config['train_steps']):
                x, y = task_fn(self.config['batch_size'])
                pred = model(x, task_id=tid)
                loss = ((pred - y) ** 2).mean()
                total = model.total_loss(loss)
                
                opt.zero_grad()
                total.backward()
                opt.step()
                model.update_dopamine(max(0, 1 - loss.item()))
                history.append(loss.item())
            
            histories[self.task_gen.get_name(tid)] = history
        
        return histories
    
    def evaluate(self, model) -> Dict[str, float]:
        model.eval()
        errors = {}
        with torch.no_grad():
            for tid in range(8):
                task_fn = self.task_gen.get_task(tid)
                total = 0
                for _ in range(50):
                    x, y = task_fn(self.config['batch_size'])
                    pred = model(x, task_id=tid)
                    total += ((pred - y) ** 2).mean().item()
                errors[self.task_gen.get_name(tid)] = total / 50
        return errors
    
    def run_weight_scan(self):
        """实验 1: STDP 权重扫描"""
        print("="*70)
        print("实验 1: STDP 权重扫描")
        print("="*70)
        
        weights = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        results = []
        
        for w in weights:
            print(f"\n【STDP 权重 = {w}】")
            torch.manual_seed(42)
            random.seed(42)
            
            model = ConnectomeAGC_STDP_Validation(32, 8, 32, stdp_weight=w, use_stdp=(w > 0))
            histories = self.train_8tasks(model)
            errors = self.evaluate(model)
            avg_error = sum(errors.values()) / len(errors)
            
            print(f"  平均误差：{avg_error:.6f}")
            results.append({'weight': w, 'avg_error': avg_error, 'task_errors': errors})
        
        # 找出最优
        best = min(results, key=lambda x: x['avg_error'])
        print(f"\n🏆 最优 STDP 权重：{best['weight']} (平均误差：{best['avg_error']:.6f})")
        
        return {'type': 'weight_scan', 'results': results, 'best': best}
    
    def run_ablation(self):
        """实验 2: 消融实验"""
        print("\n" + "="*70)
        print("实验 2: 消融实验")
        print("="*70)
        
        configs = [
            ('纯 BP (无 STDP)', {'use_stdp': False, 'stdp_weight': 0.0}),
            ('STDP+BP (标准)', {'use_stdp': True, 'stdp_weight': 0.1}),
            ('纯 STDP (无 BP)', {'use_stdp': True, 'stdp_weight': 1.0}),
            ('无痕迹', {'use_stdp': True, 'stdp_weight': 0.1, 'use_traces': False}),
            ('无多巴胺', {'use_stdp': True, 'stdp_weight': 0.1, 'use_dopamine': False}),
        ]
        
        results = []
        for name, config in configs:
            print(f"\n【{name}】")
            torch.manual_seed(42)
            random.seed(42)
            
            model = ConnectomeAGC_STDP_Validation(32, 8, 32, **config)
            histories = self.train_8tasks(model)
            errors = self.evaluate(model)
            avg_error = sum(errors.values()) / len(errors)
            
            print(f"  平均误差：{avg_error:.6f}")
            results.append({'name': name, 'config': config, 'avg_error': avg_error, 'task_errors': errors})
        
        # 对比
        baseline = results[0]['avg_error']
        print(f"\n{'配置':<20} | {'误差':<12} | {'相对 BP'}")
        print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}")
        for r in results:
            diff = (baseline - r['avg_error']) / baseline * 100
            print(f"{r['name']:<20} | {r['avg_error']:<12.6f} | {diff:+.1f}%")
        
        return {'type': 'ablation', 'results': results}
    
    def run_parameter_sensitivity(self):
        """实验 3: 参数敏感性"""
        print("\n" + "="*70)
        print("实验 3: 参数敏感性")
        print("="*70)
        
        # Tau 值测试
        print("\n【Tau 值测试】")
        tau_results = []
        for tau in [10, 20, 50, 100]:
            torch.manual_seed(42)
            random.seed(42)
            model = ConnectomeAGC_STDP_Validation(32, 8, 32, tau=tau)
            self.train_8tasks(model)
            errors = self.evaluate(model)
            avg = sum(errors.values()) / len(errors)
            print(f"  tau={tau}: {avg:.6f}")
            tau_results.append({'tau': tau, 'avg_error': avg})
        
        # A_ratio 测试
        print("\n【A_minus/A_plus 比例测试】")
        ratio_results = []
        for ratio in [1.0, 1.2, 1.5, 2.0]:
            torch.manual_seed(42)
            random.seed(42)
            model = ConnectomeAGC_STDP_Validation(32, 8, 32, A_ratio=ratio)
            self.train_8tasks(model)
            errors = self.evaluate(model)
            avg = sum(errors.values()) / len(errors)
            print(f"  A_ratio={ratio}: {avg:.6f}")
            ratio_results.append({'ratio': ratio, 'avg_error': avg})
        
        return {
            'type': 'parameter_sensitivity',
            'tau_results': tau_results,
            'ratio_results': ratio_results,
        }
    
    def run_all(self):
        """运行所有验证实验"""
        print("="*70)
        print("STDP 综合验证实验 - 完整套件")
        print("="*70)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'weight_scan': self.run_weight_scan(),
            'ablation': self.run_ablation(),
            'parameter_sensitivity': self.run_parameter_sensitivity(),
        }
        
        # 保存
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path(__file__).parent / 'results' / f'stdp_comprehensive_{ts}.json'
        path.parent.mkdir(exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n结果已保存到：{path}")
        return all_results


if __name__ == '__main__':
    exp = STDPComprehensiveValidation()
    results = exp.run_all()
