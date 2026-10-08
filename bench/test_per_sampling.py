"""PER sampling equivalence: original ReplayBuffer vs GraphReplayBuffer vs VecReplayBuffer.

Fills the original buffer (algorithm/helper.py, np.random.choice), the CUDA-graph agent's buffer
(algorithm/tdmpc_graph.py) and the vectorised agent's buffer (algorithm/tdmpc_vec.py, S = 1) with the same
episodes and the same random priorities, draws many batches from each and compares
  * the number of distinct indices per batch (detects sampling with vs. without replacement),
  * the empirical index frequencies against the exact PER probabilities priority ** per_alpha
    (total-variation distance, with the original's distance as the sampling-noise reference),
  * the importance weights of the sampled indices against the exact formula.
Both the not-yet-full regime (with replacement, as in almost every TD-MPC run) and the full regime
(without replacement) are tested.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_per_sampling.py [task] [horizon] [n_batches]   (default walker-walk 5 3000)
Writes results/equivalence/per_sampling.json. Results: docs/03_validation.md, section 3.5.

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
from algorithm.helper import ReplayBuffer, Episode
from algorithm.tdmpc_graph import GraphReplayBuffer
from algorithm.tdmpc_vec import VecReplayBuffer, VecEpisode

task = sys.argv[1] if len(sys.argv) > 1 else 'walker-walk'
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
M = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
sys.argv = [sys.argv[0], f'task={task}', f'horizon={H}']  # parse_cfg reads overrides from sys.argv
cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
cfg.obs_shape, cfg.action_shape, cfg.action_dim = (4,), (2,), 2  # tiny dummy spaces: only indices matter
L, B = cfg.episode_length, cfg.batch_size
torch.manual_seed(0); np.random.seed(0)


def run(n_eps, capacity_eps=None):
	"""Fill all three buffers with n_eps identical episodes and compare M sampled batches.

	capacity_eps: if set, the buffer capacity is capacity_eps episodes, so n_eps > capacity_eps makes
	the buffers wrap around ("full" regime: sampling without replacement)."""
	c = copy.deepcopy(cfg)
	if capacity_eps is not None:
		c.train_steps = capacity_eps * L
	bufs = {'original': ReplayBuffer(c), 'graph': GraphReplayBuffer(c), 'vec': VecReplayBuffer(c, 1)}
	for _ in range(n_eps):
		ep = Episode(c, np.zeros(c.obs_shape, dtype=np.float32))
		ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_(0, 1); ep._idx, ep.done = L, True
		vep = VecEpisode(c, 1, ep.obs[0].unsqueeze(0))
		vep.obs[:] = ep.obs.unsqueeze(1); vep.action[:] = ep.action.unsqueeze(1); vep.reward[:] = ep.reward.unsqueeze(1)
		vep._idx, vep.done = L, True
		bufs['original'] += ep; bufs['graph'] += ep; bufs['vec'] += vep
	full = bufs['original']._full
	assert bufs['graph']._full == full and bufs['vec']._full == full
	# identical random priorities (zeros of the last H steps of each episode are kept)
	r = torch.rand(bufs['original']._priorities.shape[0], device='cuda') * 4 + 0.1
	for k in ('original', 'graph'):
		bufs[k]._priorities.mul_(r)
	bufs['vec']._priorities[0].mul_(r)
	pr = bufs['original']._priorities if full else bufs['original']._priorities[:bufs['original'].idx]
	p_exact = (pr ** cfg.per_alpha); p_exact = (p_exact / p_exact.sum()).double().cpu().numpy()
	N = len(p_exact)
	w_exact = (N * p_exact) ** (-cfg.per_beta)
	out = {'n_valid_slots': N, 'nonzero_slots': int((p_exact > 0).sum()), 'full': bool(full)}
	for k, b in bufs.items():
		counts = np.zeros(N); distinct = []; wrel = 0.0
		for _ in range(M):
			s = b.sample()
			idxs, w = s[-2].reshape(-1), s[-1].reshape(-1)
			ii = idxs.cpu().numpy()
			counts += np.bincount(ii, minlength=N)[:N]
			distinct.append(len(np.unique(ii)))
			we = w_exact[ii]; we = we / we.max()
			wrel = max(wrel, float(np.abs(w.double().cpu().numpy() - we).max()))
		freq = counts / counts.sum()
		out[k] = {'mean_distinct_per_batch': float(np.mean(distinct)),
				  'tv_distance_to_exact_probs': float(0.5 * np.abs(freq - p_exact).sum()),
				  'max_abs_weight_err': wrel,
				  'zero_prob_slots_sampled': int(counts[p_exact == 0].sum())}
	return out


res = {'task': task, 'horizon': H, 'batch_size': B, 'n_batches': M}
# not full: 2 episodes (as early in training, when the batch is a large fraction of the buffer)
res['not_full_2_eps'] = run(2)
res['not_full_20_eps'] = run(20)
# full: capacity 4 episodes, 6 episodes added -> wrapped around
res['full'] = run(6, capacity_eps=4)
print(json.dumps(res, indent=1))
ok = True
for regime in ('not_full_2_eps', 'not_full_20_eps', 'full'):
	o = res[regime]
	ref_d, ref_tv = o['original']['mean_distinct_per_batch'], o['original']['tv_distance_to_exact_probs']
	for k in ('graph', 'vec'):
		ok &= abs(o[k]['mean_distinct_per_batch'] - ref_d) < 0.01 * ref_d + 1.0
		ok &= o[k]['tv_distance_to_exact_probs'] < 1.5 * ref_tv + 1e-3
		ok &= o[k]['zero_prob_slots_sampled'] == 0 and o[k]['max_abs_weight_err'] < 1e-4
res['pass'] = bool(ok)
print('PASS' if ok else 'FAIL')
d = os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence'); os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, 'per_sampling.json'), 'w'), indent=1)
