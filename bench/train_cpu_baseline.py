"""CPU-MuJoCo baseline training: the original TD-MPC code, with the same timing log as the GPU port.

CPU-MuJoCo baseline: the ORIGINAL tdmpc/src code (dm_control on CPU, agent on GPU), unmodified,
driven by the identical training loop plus early stop and the same per-episode timing CSV as
tdmpc_gpu_warp/src/train.py. Run inside the `tdmpc` conda env (Python 3.8, MuJoCo 2.1, dm_control 0.0.403778684),
NOT in tdmpc-warp: the original code is imported from ../../tdmpc/src (sibling checkout of
https://github.com/nicklashansen/tdmpc) and its config from ../../tdmpc/cfgs.

  python bench/train_cpu_baseline.py task=walker-walk seed=1 max_env_steps=20000 [pin_threads=1]

pin_threads=1 limits torch/OMP/MKL to one CPU thread (default: the original's behaviour).
Any other key=value is an override of the original config (as for tdmpc/src/train.py), plus the extra
keys max_env_steps (early stop) and log_root (default logs_cpu, relative to the working directory).

Output: <log_root>/<task>/<modality>/<exp_name>/<seed>/ with the original logger's files (eval.log ...)
and train.log (CSV: episode, env_step, episode_reward, total_time, episode_time, agent_steps_per_s).
Used by bench/learning_runs.py for the "CPU MuJoCo, original code (this machine)" learning curves in
docs/results_learning.md, summarised in docs/06_optimisations.md, section 6.3.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys
# thread pinning must happen before torch / numpy are imported
_pin = any(a == 'pin_threads=1' for a in sys.argv)
if _pin:
	for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
		os.environ[_k] = '1'
ORIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tdmpc'))
sys.path.insert(0, os.path.join(ORIG, 'src'))
import warnings
warnings.filterwarnings('ignore')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import time
from pathlib import Path
import numpy as np
import torch
if _pin:
	torch.set_num_threads(1)
import gym
gym.logger.set_level(40)
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
import train as orig_train  # original module: set_seed, evaluate
torch.backends.cudnn.benchmark = True


def main():
	"""Train one seed with the original TD-MPC loop (tdmpc/src/train.py) plus early stop and train.log timing."""
	sys.argv = [a for a in sys.argv if a != 'pin_threads=1']  # not a key of the original config
	cfg = parse_cfg(Path(ORIG) / 'cfgs')
	assert not cfg.save_video
	orig_train.set_seed(cfg.seed)
	work_dir = Path(cfg.get('log_root', 'logs_cpu')) / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	max_env_steps = cfg.get('max_env_steps', None)
	L = logger.Logger(work_dir, cfg)
	tlog = open(work_dir / 'train.log', 'w')
	tlog.write('episode,env_step,episode_reward,total_time,episode_time,agent_steps_per_s\n')
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
		# --- collect one episode (identical to the original train.py) ---
		ep_start = time.time()
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode
		# --- updates: seed_steps updates once seeding ends, then episode_length per episode (as the original) ---
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		ep_time = time.time() - ep_start
		common_metrics = {'episode': episode_idx, 'step': step, 'env_step': env_step,
						  'total_time': time.time() - start_time, 'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')
		tlog.write(f'{episode_idx},{env_step},{episode.cumulative_reward},{common_metrics["total_time"]:.3f},'
				   f'{ep_time:.3f},{cfg.episode_length/ep_time:.2f}\n')
		tlog.flush()
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = orig_train.evaluate(env, agent, cfg.eval_episodes, step, env_step, None)
			L.log(common_metrics, category='eval')
		if max_env_steps is not None and env_step >= max_env_steps:
			break
	tlog.close()


if __name__ == '__main__':
	main()
