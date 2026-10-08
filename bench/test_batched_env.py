"""Batched environment test: one S-world DMCWarpEnv vs S independent single-world envs.

Batched env (S worlds) vs S independent single-world envs: identical initial states (dm_control
resets with seeds 0..S-1) and identical actions -> per-world observations/rewards must coincide.
S = 4 worlds, T = 60 agent steps of uniform random actions. The single envs' state (qpos, qvel, act,
ctrl, qacc_warmstart and the task's per-world model fields, e.g. randomised targets) is copied into the
corresponding world of the batched env before stepping. Reports the max relative obs error, the max
reward error and the fraction of bitwise-equal steps per task. Humanoid and dog are not meaningful
here because single GPU runs are not deterministic themselves (docs/03_validation.md, 3.2).

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_batched_env.py task [task ...]
       e.g. python bench/test_batched_env.py cartpole-swingup reacher-easy finger-turn-hard walker-walk
Writes results/validation/batched_env.json. Results: docs/03_validation.md, section 3.4.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
from env_warp import DMCWarpEnv
S, T = 4, 60
AR = {'cartpole': 8, 'finger': 2, 'walker': 2, 'humanoid': 2, 'dog': 2}  # TD-MPC action_repeat per domain (4 otherwise)
out = {}
for task in sys.argv[1:]:
	ar = AR.get(task.split('-')[0], 4)
	singles = [DMCWarpEnv(task, i, ar, reset_mode='cpu') for i in range(S)]
	envb = DMCWarpEnv(task, 123, ar, nworld=S)
	envb.reset()
	for i, e in enumerate(singles):
		e.reset()
		for name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
			getattr(envb.v, name)[i].copy_(getattr(e.v, name)[0])
		for f in envb.task.per_world_model_fields:
			import warp as wp
			wp.to_torch(getattr(envb.m, f))[i].copy_(wp.to_torch(getattr(e.m, f))[0])
	# recompute derived quantities (static geoms, forward pass) for the copied states before comparing
	envb._refresh_static_geoms(); envb._run('forward', envb._forward_noact)
	ob0 = envb.task.observation()
	err0 = max(float((ob0[i] - singles[i].task.observation()[0]).abs().max()) for i in range(S))
	rng = np.random.RandomState(0); eo, er = [], []
	for t in range(T):
		a = torch.tensor(rng.uniform(-1, 1, (S, envb.action_dim)), dtype=torch.float32, device='cuda')
		ob, rb, _, _ = envb.step(a)
		for i, e in enumerate(singles):
			o, r, _, _ = e.step(a[i])
			eo.append(float((ob[i] - o).abs().max() / (o.abs().max() + 1e-6))); er.append(float((rb[i] - r).abs()))
	out[task] = dict(init_obs_err=err0, max_rel_obs_err=max(eo), max_reward_err=max(er), bitwise_equal_frac=float(np.mean(np.array(eo) == 0)))
	print(task, out[task], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'validation', 'batched_env.json'), 'w'), indent=1)
