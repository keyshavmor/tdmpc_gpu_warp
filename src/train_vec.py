"""Train S independent TD-MPC seeds of one configuration in ONE process (vectorised, all on GPU).

  python src/train_vec.py task=walker-walk num_seeds=8 seed=1 [horizon=5] [max_env_steps=...]

Seeds are seed, seed+1, ..., seed+S-1 (logs in <log_root>/<task>/state/<exp_name>/<seed>/, same
layout and eval.log format as the original). The training loop is the original one executed for
all seeds in lockstep (all DMC episodes have the same length): collect one episode per seed with
the planner, add it to each seed's buffer, then the same number of updates per seed. Every seed
has its own model, optimiser state, replay buffer and simulated world (algorithm/tdmpc_vec.py,
env_warp.py with nworld=S). Host pacing (pace_host, default on) keeps CPU use per process tiny.
One plan() / update() call (one CUDA graph replay) serves all seeds; the S worlds are stepped by one
batched mujoco_warp simulation. Per-seed eval.log has the same format as the original Logger writes;
this script does not use logger.Logger (no wandb, no model saving).
Agent options (ckpt_unroll, precision, buffer_dtype, batched_update, eval_graphs, ...): see
algorithm/tdmpc_vec.py and docs/05_how_to_run.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os
# single-threaded CPU libraries (set before numpy / torch are imported)
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
	os.environ.setdefault(_k, '1')
import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_num_threads(1)
import numpy as np
import time
import random
from pathlib import Path
import pandas as pd
from cfg import parse_cfg
from env_warp import make_env
from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def evaluate(env, agent, num_episodes, step):
	"""Per-seed mean return over num_episodes evaluation episodes (all seeds in parallel).

	Returns a numpy array (S,); the only host sync is the final .cpu().
	"""
	rets = []
	for _ in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
			obs, reward, done, _ = env.step(action)
			ep_reward = ep_reward + reward
			t += 1
		rets.append(ep_reward)
	return torch.stack(rets).mean(0).cpu().numpy()


def train(cfg):
	"""Train cfg.num_seeds seeds (cfg.seed, cfg.seed+1, ...) in lockstep; writes per-seed train.log / eval.log."""
	assert torch.cuda.is_available()
	S = int(cfg.get('num_seeds', 1))
	cfg.num_seeds = S
	cfg.save_video = False
	random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
	root = Path(cfg.get('log_root', Path().cwd() / __LOGS__)) / cfg.task / cfg.modality / cfg.exp_name
	seeds = [int(cfg.seed) + s for s in range(S)]
	dirs = [root / str(sd) for sd in seeds]
	for d in dirs:
		os.makedirs(d, exist_ok=True)
	env = make_env(cfg)  # one S-world env (reads cfg.num_seeds); created first: sets up the shared CUDA stream
	agent, buffer = VecTDMPC(cfg, S), VecReplayBuffer(cfg, S)
	# host pacing (default on here): blocking event wait per env step / update, the CPU sleeps while the GPU works
	pace = torch.cuda.Event(blocking=True) if cfg.get('pace_host', True) else None
	max_env_steps = cfg.get('max_env_steps', None)
	print(f'[train_vec] {cfg.task}: {S} seeds {seeds}, horizon {cfg.horizon}, obs {cfg.obs_shape}, act {cfg.action_dim}')

	tlogs = [open(d / 'train.log', 'w') for d in dirs]
	for f in tlogs:
		f.write('episode,env_step,episode_reward,total_time,episode_time,agent_steps_per_s\n')
	evals = [[] for _ in range(S)]
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
		ep_start = time.time()
		obs = env.reset()
		episode = VecEpisode(cfg, S, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action)
			episode += (obs, action, reward, done)
			if pace is not None:
				pace.record(); pace.synchronize()
		assert len(episode) == cfg.episode_length
		buffer += episode

		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				agent.update(buffer, step + i)  # one update of every seed (metrics are not logged here)
				if pace is not None:
					pace.record(); pace.synchronize()

		episode_idx += 1
		env_step = int(step * cfg.action_repeat)
		ep_rewards = episode.cumulative_reward.cpu().numpy()  # (S,), one host sync per episode
		ep_time = time.time() - ep_start
		total = time.time() - start_time
		for s in range(S):
			tlogs[s].write(f'{episode_idx},{env_step},{ep_rewards[s]},{total:.3f},{ep_time:.3f},{cfg.episode_length/ep_time:.2f}\n')
			tlogs[s].flush()
		print(f'[train_vec] E {episode_idx} S {env_step:,} R mean {ep_rewards.mean():.1f} '
			  f'(min {ep_rewards.min():.1f} max {ep_rewards.max():.1f}) T {total:.0f}s  {S*cfg.episode_length/ep_time:.1f} seed-steps/s', flush=True)

		if env_step % cfg.eval_freq == 0:
			er = evaluate(env, agent, cfg.eval_episodes, step)
			for s in range(S):
				evals[s].append([env_step, float(er[s])])
				pd.DataFrame(np.array(evals[s])).to_csv(dirs[s] / 'eval.log', header=['env_step', 'episode_reward'], index=None)
			print(f'[train_vec] eval S {env_step:,} R per seed {np.round(er, 1).tolist()}', flush=True)

		if max_env_steps is not None and env_step >= max_env_steps:
			break
	for f in tlogs:
		f.close()
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
