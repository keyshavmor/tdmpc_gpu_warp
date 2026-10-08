"""Equivalence test: CUDA-graph agent (algorithm/tdmpc_graph.py) vs original agent (algorithm/tdmpc.py).

Checks that the CUDA-graph agent (`graph_agent=true`) computes the same maths as the original agent:

1. update(): identical initial weights + identical fixed batch -> parameters after N updates
   (3 eager warm-up + captured graph replays) must match the original to float32 rounding.
2. plan(): the elite-trajectory choice uses the GPU RNG instead of numpy, so outputs are compared
   in distribution: N independent plans for fixed observations, per-dimension z-test of the mean
   and ratio of standard deviations (train and eval mode, t0=True and chained t0=False steps).
Observation / action sizes are hard-coded per domain (walker, dog, humanoid, cartpole, quadruped,
cheetah); no environment is created, the replay buffer is filled with random episodes.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_graph_agent.py [task] [horizon]      (defaults: walker-walk 5)
Writes results/equivalence/graph_agent_<task>_H<horizon>.json (also printed).
Results: docs/03_validation.md, section 3.5 (rows "CUDA-graph agent vs original"); approval of
optimisation 3 in docs/06_optimisations.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
from pathlib import Path
from cfg import parse_cfg
from algorithm.tdmpc import TDMPC
from algorithm.helper import ReplayBuffer, Episode
from algorithm.tdmpc_graph import GraphTDMPC, GraphReplayBuffer

task = sys.argv[1] if len(sys.argv) > 1 else 'walker-walk'
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
OBS = {'walker': 24, 'dog': 223, 'humanoid': 67, 'cartpole': 5, 'quadruped': 78, 'cheetah': 17}[task.split('-')[0]]
ACT = {'walker': 6, 'dog': 38, 'humanoid': 21, 'cartpole': 1, 'quadruped': 12, 'cheetah': 6}[task.split('-')[0]]
cfg.obs_shape, cfg.action_shape, cfg.action_dim = (OBS,), (ACT,), ACT
torch.manual_seed(0); np.random.seed(0)
res = {'task': task, 'horizon': H}

# ---------------------------------------------------------------- 1. update equivalence
# the update draws policy noise (pi(z, min_std)); for a deterministic comparison of the maths both
# agents use min_std=0 here (noise-free policy) - the noise itself is identical code in both agents
ucfg = copy.deepcopy(cfg); ucfg.min_std = 0
ref, gra = TDMPC(ucfg), GraphTDMPC(ucfg)
gra.model.load_state_dict(ref.model.state_dict()); gra.model_target.load_state_dict(ref.model_target.state_dict())
buf_r, buf_g = ReplayBuffer(cfg), GraphReplayBuffer(cfg)
L = cfg.episode_length
# enough random full-length episodes for one batch; both buffers receive the same episodes
for _ in range(int(np.ceil(cfg.batch_size / (L - H))) + 1):
	ep = Episode(cfg, np.zeros(cfg.obs_shape, dtype=np.float32))
	ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_(0, 1); ep._idx, ep.done = L, True
	buf_r += ep; buf_g += ep
batch = buf_r.sample()
fixed_r = tuple(x.clone() for x in batch)
fixed_g = tuple(x.clone() for x in batch)
# every update sees the same fixed batch (the graph agent reads it from static buffers)
buf_r.sample = lambda: fixed_r
buf_g.sample = lambda: fixed_g
N = 12
for i in range(N):
	m_r = ref.update(buf_r, 10**6 + i)
	m_g = gra.update(buf_g, 10**6 + i)
torch.cuda.synchronize()
pd = max((p1 - p2).abs().max().item() for p1, p2 in zip(ref.model.parameters(), gra.model.parameters()))
pt = max((p1 - p2).abs().max().item() for p1, p2 in zip(ref.model_target.parameters(), gra.model_target.parameters()))
pscale = max(p.abs().max().item() for p in ref.model.parameters())
res['update_param_max_abs_diff'] = pd
res['update_target_param_max_abs_diff'] = pt
res['param_scale'] = pscale
res['update_metric_rel_diff'] = {k: abs(float(m_r[k]) - float(m_g[k])) / (abs(float(m_r[k])) + 1e-8) for k in m_r}
res['priority_max_abs_diff'] = (buf_r._priorities - buf_g._priorities).abs().max().item()

# ---------------------------------------------------------------- 2. plan distribution
ref, gra = TDMPC(cfg), GraphTDMPC(cfg)
gra.model.load_state_dict(ref.model.state_dict())
gra.std = ref.std; gra._std_t.fill_(float(ref.std))  # same exploration std schedule value
n = 1500
obs = torch.randn(OBS, device='cuda')
for eval_mode in (False, True):
	for chained in (False, True):
		outs = {}
		for name, ag in (('ref', ref), ('graph', gra)):
			acts = []
			for i in range(n):
				if chained:  # t0=True then one t0=False step (uses the shifted previous mean)
					ag.plan(obs, eval_mode=eval_mode, step=10**6, t0=True)
					a = ag.plan(obs * 0.9, eval_mode=eval_mode, step=10**6, t0=False)
				else:
					a = ag.plan(obs, eval_mode=eval_mode, step=10**6, t0=True)
				acts.append(a.detach().cpu().numpy())
			outs[name] = np.stack(acts)
		a, b = outs['ref'], outs['graph']
		# two-sample z-test of the per-dimension means; std ratio graph / original
		se = np.sqrt(a.var(0) / n + b.var(0) / n) + 1e-9
		key = f"plan_{'eval' if eval_mode else 'train'}_{'chained' if chained else 't0'}"
		res[key + '_mean_z_max'] = float((np.abs(a.mean(0) - b.mean(0)) / se).max())
		res[key + '_std_ratio_minmax'] = [float((b.std(0) / (a.std(0) + 1e-9)).min()), float((b.std(0) / (a.std(0) + 1e-9)).max())]
print(json.dumps(res, indent=1))
out = os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence')
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, f'graph_agent_{task}_H{H}.json'), 'w'), indent=1)
