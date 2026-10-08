"""Fault diagnostic: short deterministic run of the code path active when the GPU faults occurred.

Exercise the code path active when the MPS faults happened (vectorised walker, eval-mode planning at a
new horizon incl. CUDA-graph capture, env resets, graph-captured updates) - for compute-sanitizer memcheck.
Sequence: walker-walk with 5 seeds (worlds); one random-action episode + 2 perturbed copies fill the replay
buffer; eval-mode planning at step 0 and step 20000 (two horizons: eager -> capture -> replay); 8 updates;
a few training-mode plan steps. Progress markers ('buffer ok', 'eval plan ok', 'update ok', 'DONE') show
how far the run got. DBG_GRAPHS=0 disables both the agent and the env CUDA graphs (eager reference).

Diagnostic for the open issue "Rare GPU fault in long multi-world (vectorised) runs" in
docs/08_known_issues.md, section 8.1 (`illegal memory access` / `unspecified launch failure` in train_vec.py
processes, mostly under CUDA MPS; cause unconfirmed).

Environment: tdmpc-warp (a CUDA GPU).
Usage: compute-sanitizer --tool memcheck python bench/debug_oob.py      (or DBG_GRAPHS=0 ...)
Output: stdout only. Finding (docs/08_known_issues.md): compute-sanitizer runs out of memory on the graph
path and is too slow eagerly.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
from pathlib import Path
from cfg import parse_cfg
from env_warp import make_env
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode
use_graphs = os.environ.get('DBG_GRAPHS', '1') == '1'
sys.argv = [sys.argv[0], 'task=walker-walk', 'num_seeds=5'] + ([] if use_graphs else ['cuda_graph_agent=false', 'cuda_graph_env=false'])
cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
env = make_env(cfg); S = 5
agent, buf = VecTDMPC(cfg, S), VecReplayBuffer(cfg, S)
obs = env.reset(); ep = VecEpisode(cfg, S, obs)
for t in range(cfg.episode_length):
	a = torch.empty(S, cfg.action_dim, device='cuda').uniform_(-1, 1)
	obs, r, d, _ = env.step(a); ep += (obs, a, r, d)
buf += ep
for _ in range(2):  # two more episodes with random observations (enough data for a batch)
	ep.obs.normal_(); buf += ep
print('buffer ok', flush=True)
for step in (0, 20000):  # eval-mode planning at two horizons (eager -> capture -> replay)
	for e in range(2):
		obs = env.reset()
		for t in range(6):
			a = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
			obs, r, d, _ = env.step(a)
	print('eval plan ok', step, flush=True)
for i in range(8):
	agent.update(buf, 10000 + i)
print('update ok', flush=True)
obs = env.reset()
for t in range(6):
	obs, r, d, _ = env.step(agent.plan(obs, step=20000, t0=t == 0))
torch.cuda.synchronize(); print('DONE', flush=True)
