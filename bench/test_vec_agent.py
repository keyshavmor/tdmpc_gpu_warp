"""Equivalence test: vectorised S-seed agent (algorithm/tdmpc_vec.py) vs S original agents.

Checks that the vectorised agent used by src/train_vec.py (S seeds as stacked tensors) computes, for
every seed slot, the same maths as an independent original TDMPC agent:

1. update(): seed slots initialised with the weights of S independent original TDMPC agents; each
   seed gets its own fixed batch; policy noise disabled (min_std=0) for a deterministic comparison.
   After N updates (3 eager + graph replays) every seed's parameters must equal its original agent's.
2. plan(): per-seed action distribution vs the corresponding original agent (z-test, std ratio).
Observation / action sizes are hard-coded per domain; no environment is created, the replay buffer is
filled with random episodes.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_vec_agent.py [task] [horizon] [S]     (defaults: walker-walk 5 3)
Writes results/equivalence/vec_agent_<task>_H<horizon>_S<S>.json (also printed).
Results: docs/03_validation.md, section 3.5 (rows "vectorised S=3 agent vs 3 original agents");
approval of optimisation 5 in docs/06_optimisations.md.

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
from algorithm.helper import ReplayBuffer
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode

task = sys.argv[1] if len(sys.argv) > 1 else 'walker-walk'
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
S = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
OBS = {'walker': 24, 'dog': 223, 'humanoid': 67, 'cartpole': 5, 'quadruped': 78, 'cheetah': 17}[task.split('-')[0]]
ACT = {'walker': 6, 'dog': 38, 'humanoid': 21, 'cartpole': 1, 'quadruped': 12, 'cheetah': 6}[task.split('-')[0]]
cfg.obs_shape, cfg.action_shape, cfg.action_dim = (OBS,), (ACT,), ACT
torch.manual_seed(0); np.random.seed(0)
res = {'task': task, 'horizon': H, 'S': S}
L = cfg.episode_length

# ---------------------------------------------------------------- 1. update equivalence
ucfg = copy.deepcopy(cfg); ucfg.min_std = 0
refs = [TDMPC(ucfg) for _ in range(S)]
vec = VecTDMPC(ucfg, S)
for s, r in enumerate(refs):
	vec.load_seed_from(s, r)
vbuf = VecReplayBuffer(ucfg, S)
for _ in range(int(np.ceil(cfg.batch_size / (L - H))) + 1):
	ep = VecEpisode(ucfg, S, torch.zeros(S, OBS, device='cuda'))
	ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_(0, 1); ep._idx, ep.done = L, True
	vbuf += ep
batch = tuple(x.clone() for x in vbuf.sample())
vbuf.sample = lambda: batch
# per-seed original replay buffers that return seed s's slice of the vectorised batch (and its priorities)
rbufs = []
for s in range(S):
	obs, nobs, act, rew, idxs, w = batch
	fixed = (obs[s].clone(), nobs[:, s].clone(), act[:, s].clone(), rew[:, s].clone(), idxs[s].clone(), w[s].clone())
	rb = ReplayBuffer(ucfg); rb._priorities[:] = vbuf._priorities[s]
	rb.sample = (lambda f: (lambda: f))(fixed)  # bind this seed's batch (avoids late binding in the loop)
	rbufs.append(rb)
N = 12
for i in range(N):
	mv = vec.update(vbuf, 10 ** 6 + i)
	mr = [r.update(rb, 10 ** 6 + i) for r, rb in zip(refs, rbufs)]
torch.cuda.synchronize()
pdiff, tdiff, mdiff, prdiff = [], [], [], []
for s, r in enumerate(refs):
	pdiff.append(max((pv.data[s] - pr.data).abs().max().item() for pv, pr in zip(vec.model.parameters(), r.model.parameters())))
	tdiff.append(max((pv.data[s] - pr.data).abs().max().item() for pv, pr in zip(vec.model_target.parameters(), r.model_target.parameters())))
	mdiff.append(max(abs(float(mv[k][s]) - float(mr[s][k])) / (abs(float(mr[s][k])) + 1e-8) for k in mr[s]))
	prdiff.append((vbuf._priorities[s] - rbufs[s]._priorities).abs().max().item())
res['update_param_max_abs_diff_per_seed'] = pdiff
res['update_target_param_max_abs_diff_per_seed'] = tdiff
res['update_metric_max_rel_diff_per_seed'] = mdiff
res['priority_max_abs_diff_per_seed'] = prdiff

# ---------------------------------------------------------------- 2. plan distribution per seed
refs = [TDMPC(cfg) for _ in range(S)]
vec = VecTDMPC(cfg, S)
for s, r in enumerate(refs):
	vec.load_seed_from(s, r)
n = 1000
obs = torch.randn(S, OBS, device='cuda')
for eval_mode in (False, True):
	va = np.stack([vec.plan(obs, eval_mode=eval_mode, step=10 ** 6, t0=True).cpu().numpy() for _ in range(n)])  # (n, S, A)
	zs, rs = [], []
	for s, r in enumerate(refs):
		ra = np.stack([r.plan(obs[s], eval_mode=eval_mode, step=10 ** 6, t0=True).cpu().numpy() for _ in range(n)])
		# two-sample z-test of the per-dimension means; std ratio vectorised / original
		se = np.sqrt(ra.var(0) / n + va[:, s].var(0) / n) + 1e-9
		zs.append(float((np.abs(ra.mean(0) - va[:, s].mean(0)) / se).max()))
		rs.append([float((va[:, s].std(0) / (ra.std(0) + 1e-9)).min()), float((va[:, s].std(0) / (ra.std(0) + 1e-9)).max())])
	key = 'plan_eval' if eval_mode else 'plan_train'
	res[key + '_mean_z_max_per_seed'] = zs
	res[key + '_std_ratio_minmax_per_seed'] = rs
print(json.dumps(res, indent=1))
out = os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence')
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, f'vec_agent_{task}_H{H}_S{S}.json'), 'w'), indent=1)
