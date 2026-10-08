"""Autocast memory study: bytes saved for backward per dtype in one update, fp32 vs bf16 vs fp16.

Why does bf16/fp16 autocast not reduce TD-MPC's VRAM? Count the bytes autograd saves for backward
(saved_tensors_hooks) per dtype during one update, fp32 vs bf16 vs fp16, plus the allocator peak.
Cases: walker-walk and dog-run at H = 30, 1 seed, vectorised agent without CUDA graphs (the hooks need
an eager update); one warm-up update first so optimiser states are already allocated. No environment
is created; the replay buffer is filled with random episodes.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_amp_memory.py
Writes results/equivalence/amp_saved_tensors.json.
Results: docs/07_precision.md, section 7.3.

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
from collections import defaultdict
from pathlib import Path
from cfg import parse_cfg
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode

res = {}
for task, OBS, ACT in (('walker-walk', 24, 6), ('dog-run', 223, 38)):
	H = 30
	sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
	cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
	cfg.obs_shape, cfg.action_shape, cfg.action_dim = (OBS,), (ACT,), ACT
	cfg.cuda_graph_agent = False
	S, L = 1, cfg.episode_length
	buf = VecReplayBuffer(cfg, S)
	for _ in range(int(np.ceil(cfg.batch_size / (L - H))) + 1):
		ep = VecEpisode(cfg, S, torch.zeros(S, OBS, device='cuda')); ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_()
		ep._idx, ep.done = L, True; buf += ep
	for prec in ('fp32', 'bf16', 'fp16'):
		c = copy.deepcopy(cfg); c.precision = prec
		ag = VecTDMPC(c, S)
		ag.update(buf, 10**6)  # warm-up (optimizer states)
		saved = defaultdict(int)

		def pack(t):
			"""saved_tensors_hooks pack hook: tally the bytes of every tensor saved for backward by dtype (identity)."""
			saved[str(t.dtype)] += t.numel() * t.element_size()
			return t
		torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
		with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
			ag.update(buf, 10**6 + 1)
		torch.cuda.synchronize()
		peak = (torch.cuda.max_memory_allocated() - base) / 2**20
		res[f'{task}_H{H}_{prec}'] = dict(saved_mib_by_dtype={k: v / 2**20 for k, v in saved.items()},
										  saved_mib_total=sum(saved.values()) / 2**20, update_peak_mib_above_resident=peak)
		print(task, prec, {k: round(v / 2**20) for k, v in saved.items()}, 'total %.0f MiB' % (sum(saved.values()) / 2**20), 'peak %.0f MiB' % peak, flush=True)
		del ag; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence', 'amp_saved_tensors.json'), 'w'), indent=1)
