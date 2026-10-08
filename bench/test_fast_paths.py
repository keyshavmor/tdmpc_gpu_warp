"""Exactness test of two opt-in fast paths: skip_substep_sensors and batched_update.

Exactness of the two extra optimisations:
 (a) skip_substep_sensors: GPU env trajectory with vs without sensor skipping (same initial state & actions)
     - 7 tasks x 100 agent steps of random actions; max obs / reward difference (expected: 0, bitwise)
 (b) batched_update: vectorised agent update batched over the horizon vs sequential (fixed batch, min_std=0)
     - walker-walk H=5, walker-walk H=30, dog-run H=10, S = 2, 12 updates; max parameter difference and
       max relative metric difference (expected: rounding-level only)

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_fast_paths.py
Writes results/equivalence/fast_paths.json.
Results: docs/06_optimisations.md, section 6.1 (optimisations 8 and 9).

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
from env_warp import DMCWarpEnv
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode
res = {'sensor_skip': {}, 'batched_update': {}}
AR = {'cartpole': 8, 'finger': 2, 'walker': 2, 'humanoid': 2, 'dog': 2}  # TD-MPC action_repeat per domain (4 otherwise)
for task in ['cartpole-swingup', 'walker-walk', 'finger-turn-hard', 'quadruped-run', 'hopper-hop', 'cheetah-run', 'cup-catch']:
	ar = AR.get(task.split('-')[0], 4)
	e0 = DMCWarpEnv(task, 0, ar, reset_mode='cpu'); e1 = DMCWarpEnv(task, 0, ar, reset_mode='cpu', skip_substep_sensors=True)
	o0, o1 = e0.reset(), e1.reset(); rng = np.random.RandomState(0); err = float((o0 - o1).abs().max()); rerr = 0.0
	for t in range(100):
		a = torch.tensor(rng.uniform(-1, 1, e0.action_dim), dtype=torch.float32, device='cuda')
		o0, r0, _, _ = e0.step(a); o1, r1, _, _ = e1.step(a)
		err = max(err, float((o0 - o1).abs().max())); rerr = max(rerr, float((r0 - r1).abs()))
	res['sensor_skip'][task] = dict(max_obs_diff=err, max_reward_diff=rerr)
	print('sensor skip', task, res['sensor_skip'][task], flush=True)
	del e0, e1
for task, OBS, ACT, H in (('walker-walk', 24, 6, 5), ('walker-walk', 24, 6, 30), ('dog-run', 223, 38, 10)):
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
	a = VecTDMPC(cfg, S)
	c2 = copy.deepcopy(cfg); c2.batched_update = True
	b = VecTDMPC(c2, S); b.model.load_state_dict(a.model.state_dict()); b.model_target.load_state_dict(a.model_target.state_dict())
	for i in range(12):
		ma = a.update(buf, 10**6 + i); mb = b.update(buf, 10**6 + i)
	torch.cuda.synchronize()
	d = max((p1 - p2).abs().max().item() for p1, p2 in zip(a.model.parameters(), b.model.parameters()))
	md = max(float((ma[k] - mb[k]).abs().max() / (ma[k].abs().max() + 1e-8)) for k in ma)
	res['batched_update'][f'{task}_H{H}'] = dict(param_max_abs_diff=d, metric_max_rel_diff=md)
	print('batched update', task, H, res['batched_update'][f'{task}_H{H}'], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence', 'fast_paths.json'), 'w'), indent=1)
