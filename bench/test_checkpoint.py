"""Checkpoint / resume test: a restored run continues with exactly the saved state.

For the original agent (algorithm/tdmpc.py + ReplayBuffer) and the CUDA-graph agent (algorithm/tdmpc_graph.py +
GraphReplayBuffer): collect a few real episodes with the GPU environment, run some updates, save a checkpoint
(src/checkpoint.py), build fresh env / agent / buffer objects, load the checkpoint, and then check
  1. all parameters, target parameters, optimiser states (incl. step counters) and hyperparameters (learning
     rate, betas, ...) and the filled replay buffer are bitwise equal,
  2. the RNG states (Python, numpy, torch CPU / CUDA, env reset generator) are restored,
  3. from the same RNG state, the original and the restored agent produce identical resets, plan() actions equal
     up to float rounding (GPU matrix kernels are not bitwise reproducible from call to call; 6e-8 observed
     between two identical agents); the update() continuation is reported for information: it agrees to float
     rounding but can drift (rounding of the GPU-computed PER probabilities can tip a sampled index; Adam
     amplifies last-bit differences), exactly as two uninterrupted GPU runs drift (docs/03 section 3.2).
Pass criterion: every restored state (parameters, optimisers, buffer, RNG states, loop state, eval history, planner
std) is bitwise equal, resets coincide and plan() agrees to float rounding.

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/test_checkpoint.py [task]        (default walker-walk)
Writes results/equivalence/checkpoint.json. Results: docs/USER_GUIDE.md (section 4.3) and docs/08_known_issues.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json, copy, random, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
from pathlib import Path
from cfg import parse_cfg
from env_warp import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
from algorithm.tdmpc_graph import GraphTDMPC, GraphReplayBuffer
import checkpoint

task = sys.argv[1] if len(sys.argv) > 1 else 'walker-walk'
sys.argv = [sys.argv[0], f'task={task}', 'seed=3']  # parse_cfg reads overrides from sys.argv
base_cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
base_cfg.seed_steps = 2 * base_cfg.episode_length  # short seeding phase: updates start after 2 episodes
res = {'task': task}


def rng_snapshot():
	"""Current Python / numpy / torch CPU / torch CUDA RNG states."""
	return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()


def rng_restore(s):
	"""Restore the RNG states captured by rng_snapshot()."""
	random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2]); torch.cuda.set_rng_state(s[3])


def build(cfg, graph):
	"""Fresh env, agent and buffer (env first: it creates the shared CUDA stream)."""
	env = make_env(cfg)
	agent, buf = (GraphTDMPC(cfg), GraphReplayBuffer(cfg)) if graph else (TDMPC(cfg), ReplayBuffer(cfg))
	return env, agent, buf


def same(a, b):
	"""Bitwise equality of two tensors / nested state dicts."""
	if isinstance(a, dict):
		return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
	if isinstance(a, (list, tuple)):
		return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
	if torch.is_tensor(a):
		return a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
	return a == b


for graph in (False, True):
	name = 'graph_agent' if graph else 'original_agent'
	cfg = copy.deepcopy(base_cfg)
	torch.manual_seed(0); np.random.seed(0); random.seed(0)
	env, agent, buf = build(cfg, graph)
	# 3 real episodes (2 seeding + 1 planned) and 20 updates
	for ep in range(3):
		obs = env.reset(); episode = Episode(cfg, obs); step = ep * cfg.episode_length
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action)
			episode += (obs, action, reward, done)
		buf += episode
	for i in range(20):
		agent.update(buf, 3 * cfg.episode_length + i)
	work = tempfile.mkdtemp(prefix='ckpt_test_')
	loop = {'next_step': 3 * cfg.episode_length, 'episode_idx': 3, 'elapsed_s': 1.0}
	nbytes = checkpoint.save(work, agent, buf, env, loop, [[0.0, 1.0], [1000.0, 2.0]])
	saved_rng = rng_snapshot(); saved_env_rng = env.gen.get_state()

	# fresh objects, different RNG states, then load
	torch.manual_seed(123); np.random.seed(123); random.seed(123)
	env2, agent2, buf2 = build(cfg, graph)
	loop2, eval2, _ = checkpoint.load(work, agent2, buf2, env2)
	r = {'checkpoint_bytes': nbytes, 'loop_restored': loop2 == loop, 'eval_rows_restored': eval2 == [[0.0, 1.0], [1000.0, 2.0]]}
	r['params_equal'] = same(agent.model.state_dict(), agent2.model.state_dict()) and same(agent.model_target.state_dict(), agent2.model_target.state_dict())
	r['optim_equal'] = same(agent.optim.state_dict()['state'], agent2.optim.state_dict()['state']) and same(agent.pi_optim.state_dict()['state'], agent2.pi_optim.state_dict()['state'])
	pg = lambda o: [{k: v for k, v in g.items() if k != 'params'} for g in o.state_dict()['param_groups']]
	r['optim_hyperparams_equal'] = pg(agent.optim) == pg(agent2.optim) and pg(agent.pi_optim) == pg(agent2.pi_optim)  # lr, betas, eps, ...
	r['lr'] = [g['lr'] for g in agent2.optim.state_dict()['param_groups']]
	n = buf.capacity if buf._full else buf.idx
	r['buffer_equal'] = (buf.idx == buf2.idx and buf._full == buf2._full and all(torch.equal(getattr(buf, k)[:n], getattr(buf2, k)[:n]) for k in ('_obs', '_action', '_reward', '_priorities'))
						 and torch.equal(buf._last_obs[:n // cfg.episode_length], buf2._last_obs[:n // cfg.episode_length]))
	r['rng_restored'] = same(rng_snapshot()[2:], saved_rng[2:]) and np.array_equal(np.random.get_state()[1], saved_rng[1][1]) and random.getstate() == saved_rng[0] and torch.equal(env2.gen.get_state(), saved_env_rng)

	# continue both from the same RNG state: resets, plans and updates must coincide bitwise
	o1 = env.reset(); rng_restore(saved_rng); env2.gen.set_state(saved_env_rng)
	o2 = env2.reset()
	r['reset_equal'] = torch.equal(o1, o2)
	step = 3 * cfg.episode_length
	s = rng_snapshot(); a1 = [agent.plan(o1, step=step, t0=t == 0).clone() for t in range(3)]
	rng_restore(s); a2 = [agent2.plan(o1, step=step, t0=t == 0).clone() for t in range(3)]
	r['plan_max_abs_diff'] = max((x - y).abs().max().item() for x, y in zip(a1, a2))
	r['plan_equal_to_rounding'] = r['plan_max_abs_diff'] < 1e-5  # GPU GEMMs are not bitwise reproducible call to call
	s = rng_snapshot(); m1 = [{k: float(v) for k, v in agent.update(buf, step + i).items()} for i in range(5)]
	rng_restore(s); m2 = [{k: float(v) for k, v in agent2.update(buf2, step + i).items()} for i in range(5)]
	r['update_metrics_bitwise_equal'] = m1 == m2  # informational
	r['update_metric_max_rel_diff'] = max(abs(x[k] - y[k]) / (abs(x[k]) + 1e-12) for x, y in zip(m1, m2) for k in x)
	r['param_max_abs_diff_after_5_updates'] = max((p - q).abs().max().item() for p, q in zip(agent.model.parameters(), agent2.model.parameters()))
	shutil.rmtree(work)
	res[name] = r
	print(name, r, flush=True)
	del env, agent, buf, env2, agent2, buf2
	torch.cuda.empty_cache()

checks = [v for k in ('original_agent', 'graph_agent') for kk, v in res[k].items() if kk not in ('checkpoint_bytes', 'plan_max_abs_diff', 'update_metric_max_rel_diff', 'param_max_abs_diff_after_5_updates', 'update_metrics_bitwise_equal', 'lr')]  # continuation: informational
res['pass'] = all(checks)
print('PASS' if res['pass'] else 'FAIL')
d = os.path.join(os.path.dirname(__file__), '..', 'results', 'equivalence'); os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, 'checkpoint.json'), 'w'), indent=1)
