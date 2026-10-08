"""Checkpoint / resume for single-seed training runs (src/train.py).

A checkpoint is written at an episode boundary and contains everything the training loop needs to continue:
TOLD model and target network, both Adam optimisers (incl. their step counters), the planner's scheduled
minimum std (`agent.std`, refreshed by the original only inside update()), the filled part of the replay
buffer (observations, actions, rewards, priorities, write index, full flag), the RNG states (Python, numpy, torch
CPU, torch CUDA default generator, the environment's reset generator), the loop state (next step, episode index,
elapsed time) and the evaluation history of the logger (so eval.log keeps all rows).

Resuming restores this state before any CUDA graph is captured (graphs are re-captured afterwards), so the maths of
the continued run is that of the uninterrupted one. A resumed run is not bitwise identical to an uninterrupted one,
for the same reason two uninterrupted GPU runs are not: GPU physics uses float atomics (docs/03_validation.md).

The file is written atomically (temporary file + rename), so a crash while saving leaves the previous checkpoint
intact. Size: the filled replay buffer dominates (tens of MB for most tasks, ~2 GB for a full dog run).

Usage (through src/train.py):
	python src/train.py task=dog-run seed=1 graph_agent=true pace_host=true checkpoint_minutes=30
	python src/train.py task=dog-run seed=1 graph_agent=true pace_host=true checkpoint_minutes=30 resume=true

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json
import os
import random
import numpy as np
import torch

FILENAME = 'checkpoint.pt'
META = 'checkpoint_meta.json'  # small sidecar: loop state + last logged env step (read without loading the checkpoint)
_BUFFER_FIELDS = ('_obs', '_action', '_reward', '_priorities')


def _filled(buffer):
	"""Number of filled transitions of a replay buffer (capacity once it has wrapped around)."""
	return buffer.capacity if buffer._full else buffer.idx


def save(work_dir, agent, buffer, env, loop_state, eval_rows, wandb_pending=()):
	"""Write <work_dir>/checkpoint.pt atomically.

	Args:
		work_dir: run directory (pathlib.Path).
		agent: TDMPC or GraphTDMPC (model, model_target, optim, pi_optim).
		buffer: ReplayBuffer or GraphReplayBuffer.
		env: DMCWarpEnv (its reset generator `gen` is saved).
		loop_state: dict with the training-loop counters (next_step, episode_idx, elapsed_s).
		eval_rows: the logger's evaluation history (list of [env_step, episode_reward]).
		wandb_pending: metrics not yet sent to wandb (deferred sync; re-sent on resume if the flush was cut short).
	Returns the checkpoint size in bytes.
	"""
	torch.cuda.synchronize()
	n = _filled(buffer)
	L = buffer.cfg.episode_length
	state = {
		'version': 1,
		'agent': {
			'model': agent.model.state_dict(), 'model_target': agent.model_target.state_dict(),
			'optim': agent.optim.state_dict(), 'pi_optim': agent.pi_optim.state_dict(),
			'std': float(agent.std)},  # planner's min std; the original only refreshes it inside update()
		'buffer': {
			'idx': buffer.idx, 'full': buffer._full, 'n': n,
			**{k: getattr(buffer, k)[:n].cpu() for k in _BUFFER_FIELDS},
			'_last_obs': buffer._last_obs[:n // L].cpu()},
		'rng': {
			'python': random.getstate(), 'numpy': np.random.get_state(),
			'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(),
			'env': env.gen.get_state()},
		'loop': dict(loop_state),
		'eval_rows': [list(map(float, r)) for r in eval_rows],
		'wandb_pending': list(wandb_pending),
	}
	path = os.path.join(work_dir, FILENAME)
	tmp = path + '.tmp'
	torch.save(state, tmp)
	os.replace(tmp, path)
	meta = dict(loop_state, last_env_step=int((loop_state['next_step'] - L) * buffer.cfg.action_repeat))
	with open(os.path.join(work_dir, META + '.tmp'), 'w') as f:
		json.dump(meta, f)
	os.replace(os.path.join(work_dir, META + '.tmp'), os.path.join(work_dir, META))
	return os.path.getsize(path)


def peek(work_dir):
	"""Loop state of the checkpoint incl. `last_env_step` (the env step logged last before it), from the sidecar."""
	with open(os.path.join(work_dir, META)) as f:
		return json.load(f)


def exists(work_dir):
	"""True if <work_dir>/checkpoint.pt exists."""
	return os.path.exists(os.path.join(work_dir, FILENAME))


def load(work_dir, agent, buffer, env):
	"""Restore a checkpoint written by save() into freshly constructed objects.

	Must run before the first plan()/update() call, i.e. before any CUDA graph of the agent is captured.
	Buffer tensors are filled in place (the graph-captured agent reads them from fixed addresses).
	Returns (loop_state dict, eval_rows list, wandb_pending list).
	"""
	path = os.path.join(work_dir, FILENAME)
	state = torch.load(path, map_location='cpu', weights_only=False)
	a = state['agent']
	agent.model.load_state_dict(a['model'])
	agent.model_target.load_state_dict(a['model_target'])
	agent.optim.load_state_dict(a['optim'])
	agent.pi_optim.load_state_dict(a['pi_optim'])
	agent.std = a['std']
	if hasattr(agent, '_std_t'):  # GraphTDMPC: device copy of std read by the plan graphs
		agent._std_t.fill_(float(agent.std))
	b = state['buffer']
	n, L = b['n'], buffer.cfg.episode_length
	with torch.no_grad():
		for k in _BUFFER_FIELDS:
			getattr(buffer, k)[:n].copy_(b[k])
		buffer._last_obs[:n // L].copy_(b['_last_obs'])
	buffer.idx, buffer._full = b['idx'], b['full']
	if hasattr(buffer, '_n_valid'):  # GraphReplayBuffer: device-side count of valid transitions
		buffer._n_valid.fill_(float(buffer.capacity if buffer._full else buffer.idx))
	r = state['rng']
	random.setstate(r['python'])
	np.random.set_state(r['numpy'])
	torch.set_rng_state(r['torch_cpu'])
	torch.cuda.set_rng_state(r['torch_cuda'])
	env.gen.set_state(r['env'])
	torch.cuda.synchronize()
	return state['loop'], state['eval_rows'], state.get('wandb_pending', [])
