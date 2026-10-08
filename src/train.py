"""TD-MPC training script for one seed with the GPU (mujoco_warp) DMControl environment.

Drop-in replacement for tdmpc/src/train.py: same training loop (random seed episodes, then per episode:
collect with the planner, add to the replay buffer, run the updates; evaluation every eval_freq env
steps), same cfgs/*.yaml
options and the same logs/<task>/state/<exp_name>/<seed>/eval.log format.
	python src/train.py task=walker-walk seed=1                                   # faithful port
	python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true   # optimised, same maths
For S seeds of one configuration in one process use src/train_vec.py instead.

Original code: TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (ICML 2022),
https://github.com/nicklashansen/tdmpc (MIT License, Copyright (c) 2021 Nicklas Hansen).
Modifications for the GPU port (tdmpc_gpu_warp) by Keyshav Mor:
  * env = env_warp.make_env (physics, observation, reward and resets on the GPU) instead of env.make_env
    (no gym / MUJOCO_GL setup); the env is created first because it sets up the CUDA stream shared by
    torch and warp; observations, actions and rewards stay CUDA tensors (no .cpu().numpy() round trip
    per step),
  * each process uses a single CPU thread (OMP/MKL/OpenBLAS/NumExpr = 1, torch intra-/inter-op threads
    = 1 -> no CPU oversubscription when many runs share the machine),
  * opt-in `graph_agent=true`: CUDA-graph agent and buffer (algorithm/tdmpc_graph.py) instead of the
    original agent; opt-in `pace_host=true`: one blocking CUDA event wait per env step / update,
  * training metrics and rewards may be CUDA tensors and are converted to float once per episode,
  * video recording is hard-disabled (cfg.save_video = False, evaluate() never records),
  * optional `log_root` (log directory) and `max_env_steps` (early stop, for benchmarking),
  * per-episode timing is written to <work_dir>/train.log.
"""
import os
# single-threaded CPU libraries (set before numpy / torch are imported)
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
	os.environ.setdefault(_k, '1')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import numpy as np
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env_warp import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	"""Seed Python, numpy and torch (CPU and CUDA) RNGs, as in the original."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent (video is never recorded in the GPU port).

	Returns the mean episode return over num_episodes episodes; `video` is unused (kept for the
	original signature). Rewards may be 0-d CUDA tensors and are read once per episode.
	"""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			t += 1
		episode_rewards.append(float(ep_reward))
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device.

	Writes eval.log (Logger, original format) and train.log (per-episode timing) to
	<log_root>/<task>/<modality>/<exp_name>/<seed>/.
	"""
	assert torch.cuda.is_available()
	cfg.save_video = False
	set_seed(cfg.seed)
	work_dir = Path(cfg.get('log_root', Path().cwd() / __LOGS__)) / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env = make_env(cfg)  # first: sets up the shared torch/warp CUDA stream
	if cfg.get('graph_agent', False):  # CUDA-graph agent (algorithm/tdmpc_graph.py)
		from algorithm.tdmpc_graph import GraphTDMPC, GraphReplayBuffer
		agent, buffer = GraphTDMPC(cfg), GraphReplayBuffer(cfg)
	else:
		agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
	max_env_steps = cfg.get('max_env_steps', None)

	# optional host pacing: one blocking wait per env step / update, so the host thread sleeps while
	# the GPU works instead of spin-waiting on a full launch queue (~1 CPU core -> ~0.2 per run)
	pace = torch.cuda.Event(blocking=True) if cfg.get('pace_host', False) else None

	# Run training
	L = logger.Logger(work_dir, cfg)
	tlog = open(work_dir / 'train.log', 'w')
	tlog.write('episode,env_step,episode_reward,total_time,episode_time,agent_steps_per_s\n')
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
		ep_start = time.time()

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action)
			episode += (obs, action, reward, done)
			if pace is not None:
				pace.record(); pace.synchronize()
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))
				if pace is not None:
					pace.record(); pace.synchronize()

		# Log training episode (metrics may be 0-d CUDA tensors: read once per episode)
		train_metrics = {k: float(v) for k, v in train_metrics.items()}
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		episode_reward = float(episode.cumulative_reward)
		ep_time = time.time() - ep_start
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')
		tlog.write(f'{episode_idx},{env_step},{episode_reward},{common_metrics["total_time"]:.3f},{ep_time:.3f},'
				   f'{cfg.episode_length/ep_time:.2f}\n')
		tlog.flush()

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, None)
			L.log(common_metrics, category='eval')

		if max_env_steps is not None and env_step >= max_env_steps:
			break


	tlog.close()
	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
