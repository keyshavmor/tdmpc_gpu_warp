"""Precision-mode test of the vectorised agent: ScaledAdam exactness and a smoke test of fp32/tf32/bf16/fp16.

(1) ScaledAdam (masked Adam + dynamic loss scaling, used for fp16) in fp32 == torch.optim.Adam.
    Two agents with identical weights (one with force_scaled_adam=True) run 12 updates on a fixed batch
    (walker-walk, H = 5, S = 2, min_std=0); reported: max parameter difference.
(2) every precision mode runs (incl. CUDA-graph capture) and reports losses: 8 updates per mode
    (total_loss, grad_norm per seed) and whether planned actions are finite.
No environment is created; the replay buffer is filled with random episodes.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_precision_agent.py
Output: stdout only (JSON). Results: docs/03_validation.md, section 3.5 (row "capturable masked Adam");
the precision modes themselves are evaluated in docs/07_precision.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
from pathlib import Path
from cfg import parse_cfg
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode
sys.argv = [sys.argv[0], 'task=walker-walk', 'horizon=5']
cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
cfg.obs_shape, cfg.action_shape, cfg.action_dim = (24,), (6,), 6
cfg.min_std = 0
S, L = 2, cfg.episode_length
torch.manual_seed(0)
buf = VecReplayBuffer(cfg, S)
for _ in range(3):
	ep = VecEpisode(cfg, S, torch.zeros(S, 24, device='cuda')); ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_()
	ep._idx, ep.done = L, True; buf += ep
batch = tuple(x.clone() for x in buf.sample()); buf.sample = lambda: batch
a = VecTDMPC(cfg, S)
c2 = copy.deepcopy(cfg); c2.force_scaled_adam = True  # fp32 maths, but the fp16 optimiser path
b = VecTDMPC(c2, S); b.model.load_state_dict(a.model.state_dict()); b.model_target.load_state_dict(a.model_target.state_dict())
for i in range(12):
	a.update(buf, 10**6 + i); b.update(buf, 10**6 + i)
torch.cuda.synchronize()
d = max((p1 - p2).abs().max().item() for p1, p2 in zip(a.model.parameters(), b.model.parameters()))
res = {'scaled_adam_vs_torch_adam_param_max_abs_diff_after_12_updates': d}
for prec in ['fp32', 'tf32', 'bf16', 'fp16']:
	c = copy.deepcopy(cfg); c.precision = prec
	ag = VecTDMPC(c, S); ag.model.load_state_dict(a.model.state_dict())
	for i in range(8):
		m = ag.update(buf, 10**6 + i)
	act = ag.plan(torch.randn(S, 24, device='cuda'), step=10**6); act = ag.plan(torch.randn(S, 24, device='cuda'), step=10**6)
	res[prec] = {k: [round(float(x), 5) for x in v] for k, v in m.items() if k in ('total_loss', 'grad_norm')}
	res[prec]['action_finite'] = bool(torch.isfinite(act).all())
print(json.dumps(res, indent=1))
