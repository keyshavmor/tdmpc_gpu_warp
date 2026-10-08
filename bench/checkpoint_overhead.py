"""Checkpoint overhead: time and size of saving / loading a checkpoint with a FULL replay buffer.

Worst case of src/checkpoint.py: the replay buffer holds the complete training budget (train_steps transitions),
as at the end of a run. Builds the GPU env, the CUDA-graph agent and its buffer for the given tasks, marks the
buffer as full (random contents), and times checkpoint.save() (GPU -> CPU copy + write) and checkpoint.load().
The training slowdown is save_time / checkpoint interval (the loop pauses while saving; nothing else changes).

Environment: tdmpc-warp (a CUDA GPU). Needs free disk space of ~1.1x the largest checkpoint in the output dir.
Usage: python bench/checkpoint_overhead.py [--tasks walker-walk humanoid-run dog-run] [--dir /tmp/tdmpc_gpu_warp/ckpt_bench]
Writes results/checkpoint/overhead.json. Results: docs/USER_GUIDE.md (section 4.3).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json, time, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import torch
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', default=['walker-walk', 'humanoid-run', 'dog-run'])
ap.add_argument('--dir', default='/tmp/tdmpc_gpu_warp/ckpt_bench')
a = ap.parse_args()
res = {}
for task in a.tasks:
	sys.argv = [sys.argv[0], f'task={task}']  # parse_cfg reads overrides from sys.argv
	from cfg import parse_cfg
	from env_warp import make_env
	from algorithm.tdmpc_graph import GraphTDMPC, GraphReplayBuffer
	import checkpoint
	cfg = parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs')))
	env = make_env(cfg)
	agent, buf = GraphTDMPC(cfg), GraphReplayBuffer(cfg)
	for k in ('_obs', '_last_obs', '_action', '_reward'):
		getattr(buf, k).normal_()
	buf._priorities.uniform_(0.1, 2.0)
	buf.idx, buf._full = 0, True
	work = os.path.join(a.dir, task); os.makedirs(work, exist_ok=True)
	loop = {'next_step': cfg.train_steps, 'episode_idx': cfg.train_steps // cfg.episode_length, 'elapsed_s': 0.0}
	pending = [(i * 1000, {'train/episode_reward': 1.0, 'train/total_loss': 1.0}) for i in range(60)]
	times = []
	for rep in range(3):
		t = time.time(); nbytes = checkpoint.save(work, agent, buf, env, loop, [[0.0, 0.0]] * 50, pending); times.append(time.time() - t)
	t = time.time(); checkpoint.load(work, agent, buf, env); t_load = time.time() - t
	save_s = min(times)
	res[task] = {'buffer_transitions': buf.capacity, 'checkpoint_mib': nbytes / 2**20, 'save_s': save_s, 'save_s_all': times,
				 'load_s': t_load, 'slowdown_pct_at_30min': 100 * save_s / 1800, 'slowdown_pct_at_10min': 100 * save_s / 600}
	print(task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[task].items()}, flush=True)
	shutil.rmtree(work)
	del env, agent, buf
	torch.cuda.empty_cache()
d = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoint'); os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, 'overhead.json'), 'w'), indent=1)
