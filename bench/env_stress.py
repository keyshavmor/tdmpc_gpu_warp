"""Environment-only stress process: step the GPU environment with random actions forever (no agent).

Isolates the physics path (MuJoCo Warp, graph-captured agent step, episode resets) from planning and learning, for
the MPS fault bisection (bench/mps_fault_hunt.py --script bench/env_stress.py). Takes the same key=value options as
the training scripts (task, seed, num_seeds = number of worlds, cuda_graph_env, env_graph_mode, ...); runs episodes
of random uniform actions with an environment reset after each episode, like the seeding phase of training.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/env_stress.py task=walker-walk num_seeds=3 seed=1

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import torch
from pathlib import Path
from cfg import parse_cfg
from env_warp import make_env, set_blocking_sync

set_blocking_sync()
cfg = parse_cfg(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfgs')))
cfg.num_seeds = int(cfg.get('num_seeds', 1))
env = make_env(cfg)
S, A = cfg.num_seeds, cfg.action_dim
pace = torch.cuda.Event(blocking=True)
t0, episodes = time.time(), 0
max_s = float(cfg.get('max_seconds', 1e12))
shape = (S, A) if S > 1 else (A,)
while time.time() - t0 < max_s:
	env.reset()
	done = False
	while not done:
		_, _, done, _ = env.step(torch.rand(shape, device='cuda') * 2 - 1)
		pace.record(); pace.synchronize()
	episodes += 1
	if episodes % 5 == 0:
		print(f'[env_stress] {episodes} episodes, {time.time() - t0:.0f} s', flush=True)
