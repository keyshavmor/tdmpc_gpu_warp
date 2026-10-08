"""Activation checkpointing test: ckpt_unroll=true vs false in the vectorised agent.

Activation checkpointing (ckpt_unroll) == no checkpointing: same parameters after 12 updates
(deterministic policy, fixed batch), and peak memory of one update at large H.
Cases: walker-walk H=5, walker-walk H=30, dog-run H=30; S = 2 seeds, eager agent (no CUDA graphs),
min_std=0, replay buffer filled with random episodes (no environment). Both agents start from identical
weights; the peak is the allocator maximum above the memory resident before the update, in MiB.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_ckpt.py
Writes results/equivalence/ckpt_unroll.json.
Results: docs/03_validation.md, section 3.5 (bitwise-identical parameters); memory effect in
docs/06_optimisations.md (optimisation 7) and docs/04_multi_run_capacity.md.

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
res = {}
for task, OBS, ACT, H in (('walker-walk', 24, 6, 5), ('walker-walk', 24, 6, 30), ('dog-run', 223, 38, 30)):
	sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
	cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
	cfg.obs_shape, cfg.action_shape, cfg.action_dim = (OBS,), (ACT,), ACT
	cfg.min_std = 0
	S, L = 2, cfg.episode_length
	torch.manual_seed(0)
	buf = VecReplayBuffer(cfg, S)
	for _ in range(int(np.ceil(cfg.batch_size / (L - H))) + 1):
		ep = VecEpisode(cfg, S, torch.zeros(S, OBS, device='cuda')); ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_()
		ep._idx, ep.done = L, True; buf += ep
	batch = tuple(x.clone() for x in buf.sample()); buf.sample = lambda: batch  # one fixed batch for every update
	out = {}
	agents = {}
	for ck in (False, True):
		c = copy.deepcopy(cfg); c.ckpt_unroll = ck; c.cuda_graph_agent = False
		ag = VecTDMPC(c, S)
		if ck: ag.model.load_state_dict(agents[False].model.state_dict()); ag.model_target.load_state_dict(agents[False].model_target.state_dict())
		agents[ck] = ag
	for ck in (False, True):  # (no-op)
		pass
	# run interleaved from identical starting weights
	a, b = agents[False], agents[True]
	b.model.load_state_dict(a.model.state_dict()); b.model_target.load_state_dict(a.model_target.state_dict())
	peaks = {}
	for name, ag in (('plain', a), ('ckpt', b)):
		for i in range(12):
			torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
			ag.update(buf, 10**6 + i)
			torch.cuda.synchronize(); peaks[name] = (torch.cuda.max_memory_allocated() - base) / 2**20
	d = max((p1 - p2).abs().max().item() for p1, p2 in zip(a.model.parameters(), b.model.parameters()))
	res[f'{task}_H{H}'] = dict(param_max_abs_diff=d, update_peak_extra_mib_plain=peaks['plain'], update_peak_extra_mib_ckpt=peaks['ckpt'])
	print(f'{task} H={H}: param diff {d:.2e}  update peak (MiB above resident) plain {peaks["plain"]:.0f} ckpt {peaks["ckpt"]:.0f}', flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence', 'ckpt_unroll.json'), 'w'), indent=1)
