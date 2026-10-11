"""Update drift test: 200 deterministic updates, eager baseline vs CUDA-graph and vs batched_update.

Long-run drift check: 200 consecutive deterministic updates (min_std=0, fixed rotating batches)
baseline sequential update vs batched_update, and baseline vs CUDA-graph (graph replay vs eager).
Reports per-update relative difference of the total loss and the final parameter difference.
Variants of the vectorised agent (walker-walk, H = 5 and H = 30, S = 2): eager_sequential (baseline,
cuda_graph_agent=False), graph_sequential (default graph agent), graph_batched (batched_update=True).
All start from the same weights and see the same 20 pre-sampled batches in rotation.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_update_drift.py
Writes results/equivalence/update_drift.json.
Results: docs/06_optimisations.md, section 6.1, column "200-update drift" (optimisations 3 and 8).

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
for task, OBS, ACT, H in (('walker-walk', 24, 6, 5), ('walker-walk', 24, 6, 30)):
	sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
	cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
	cfg.obs_shape, cfg.action_shape, cfg.action_dim = (OBS,), (ACT,), ACT
	cfg.min_std = 0
	S, L = 2, cfg.episode_length
	torch.manual_seed(0)
	buf = VecReplayBuffer(cfg, S)
	for _ in range(int(np.ceil(cfg.batch_size / (L - H))) + 2):
		ep = VecEpisode(cfg, S, torch.zeros(S, OBS, device='cuda')); ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_()
		ep._idx, ep.done = L, True; buf += ep
	batches = [tuple(x.clone() for x in buf.sample()) for _ in range(20)]  # 20 fixed batches, used in rotation
	cur = {'i': 0}
	src = [None]
	# buf.sample() returns whatever src[0] points to (set per agent and update below)
	def fixed():
		"""Replacement for buf.sample(): return the batch src[0] currently points to."""
		return src[0]
	buf.sample = fixed
	variants = {'eager_sequential': dict(cuda_graph_agent=False), 'graph_sequential': {}, 'graph_batched': dict(batched_update=True)}
	agents = {}
	for name, kw in variants.items():
		c = copy.deepcopy(cfg)
		for k, v in kw.items(): setattr(c, k, v)
		agents[name] = VecTDMPC(c, S)
		if name != 'eager_sequential':
			agents[name].model.load_state_dict(agents['eager_sequential'].model.state_dict())
			agents[name].model_target.load_state_dict(agents['eager_sequential'].model_target.state_dict())
	losses = {n: [] for n in agents}
	for i in range(200):
		for n, ag in agents.items():
			for j, x in enumerate(batches[i % 20]):  # same batch for every agent (graph agents read static inputs)
				pass  # (no-op)
			src[0] = batches[i % 20] if not hasattr(ag, '_static_batch') else None
			# graph agents capture the tensors returned by sample(): give them persistent buffers
			if not hasattr(ag, '_static_batch'):
				ag._static_batch = tuple(x.clone() for x in batches[0])
			for dst, s_ in zip(ag._static_batch, batches[i % 20]):
				dst.copy_(s_)
			src[0] = ag._static_batch  # every agent (eager and graph) reads the batch from the same persistent tensors
			m = ag.update(buf, 10**6 + i)
			losses[n].append(m['total_loss'].detach().cpu().numpy().copy())
	torch.cuda.synchronize()
	base = np.array(losses['eager_sequential'])
	out = {}
	for n in ('graph_sequential', 'graph_batched'):
		arr = np.array(losses[n]); rel = np.abs(arr - base) / (np.abs(base) + 1e-8)
		pdiff = max((p1 - p2).abs().max().item() for p1, p2 in zip(agents['eager_sequential'].model.parameters(), agents[n].model.parameters()))
		out[n] = dict(loss_rel_diff_update_1=float(rel[0].max()), loss_rel_diff_update_50=float(rel[49].max()),
					  loss_rel_diff_update_200=float(rel[199].max()), loss_rel_diff_max=float(rel.max()),
					  final_param_max_abs_diff=pdiff, loss_start=float(base[0].mean()), loss_end=float(base[-1].mean()))
		print(task, H, n, out[n], flush=True)
	res[f'{task}_H{H}'] = out
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence', 'update_drift.json'), 'w'), indent=1)
