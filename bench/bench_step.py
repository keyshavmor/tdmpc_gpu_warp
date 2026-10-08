"""Steady-state cost and peak VRAM of one TD-MPC configuration (task, horizon H).

Single-process micro-benchmark that every sweep / memory study in bench/ is built on: it is launched as a
subprocess by sweep_h.py (horizon sweep, precision sweep), oom_horizon.py (OOM probes with k=3),
vec_memory.py and packing_study.py (vectorised S-seed probes) and the pipeline_*.sh scripts (multi-seed
precision, results/precision_vec/). Numbers feed docs/02_single_run_performance.md (results_sweep_h.md),
docs/07_precision.md (results_precision.md) and the memory tables of docs/04_multi_run_capacity.md
(results_memory.md).

  python bench/bench_step.py --mode gpu --task walker-walk --horizon 5 [--k 100] [--out f.json]
  python bench/bench_step.py --mode gpu --task walker-walk --horizon 5 --extra graph_agent=true pace_host=true
  python bench/bench_step.py --mode gpu --task walker-walk --horizon 5 --extra vec_agent=true num_seeds=16 pace_host=true
  python bench/bench_step.py --mode cpu ...   (original tdmpc/src: dm_control on CPU; run in either env)

--mode gpu imports tdmpc_gpu_warp/src and needs the `tdmpc-warp` conda env; --mode cpu imports the untouched
original ../tdmpc/src (sweep_h.py runs it in the `tdmpc` env). --extra passes cfg overrides (key=value)
exactly as on the train.py command line; it selects the agent variant (graph_agent / vec_agent /
precision / ckpt_unroll ...). With --out the result is written as JSON (otherwise printed only).

Procedure (identical for both modes; agent code identical):
  1. build env, TD-MPC agent and the replay buffer (pre-allocated at full capacity, as in training)
  2. fill the buffer with enough synthetic episodes to allow batch sampling (values do not affect cost)
  3. warm-up, then time K agent steps of `plan + env.step` (collection) followed by K `update`s,
     at a training step past both schedules, i.e. planning horizon = H and min std reached
  4. synced per-component timings (plan / env / update) on a separate pass
  5. peak memory: torch allocator peak and whole-process VRAM (nvidia-smi, includes CUDA context,
     mujoco_warp buffers and allocator cache); process CPU time / wall time.
Per agent step in training = plan + env + update (one update per env step after seeding).

Output keys (JSON): step_ms = collect_ms_per_step + update_ms (the training-loop cost per agent step),
plan_ms_synced / env_ms_synced, seed_steps_per_s (S seeds x agent steps per second), process_vram_mib,
torch_peak_alloc_mib / torch_peak_reserved_mib, cpu_cores_used, gpu_util_mean, episode_length,
batch_size, train_steps, action_repeat (the last four are used to extrapolate full-run times).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys, threading, time

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['gpu', 'cpu'], required=True)
p.add_argument('--task', required=True)
p.add_argument('--horizon', type=int, required=True)
p.add_argument('--k', type=int, default=100)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--out', default=None)
p.add_argument('--extra', nargs='*', default=[], help='extra cfg overrides key=value')
p.add_argument('--threads', type=int, default=1, help='CPU threads for torch/OMP (gpu mode always 1)')
args = p.parse_args()

# thread pinning must happen before numpy / torch are imported (they read these variables at import time);
# --threads 0 in cpu mode keeps the original code's default threading
if args.mode == 'gpu' or args.threads == 1:
	for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
		os.environ[_k] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
# gpu mode: this repository's src/ + cfgs/; cpu mode: the untouched original ../tdmpc/src + cfgs/
SRC = os.path.join(HERE, '..', 'src') if args.mode == 'gpu' else os.path.join(HERE, '..', '..', 'tdmpc', 'src')
CFGS = os.path.join(HERE, '..', 'cfgs') if args.mode == 'gpu' else os.path.join(HERE, '..', '..', 'tdmpc', 'cfgs')
sys.path.insert(0, os.path.abspath(SRC))
import warnings
warnings.filterwarnings('ignore')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import numpy as np
import torch
if args.mode == 'gpu' or args.threads == 1:
	torch.set_num_threads(1)
from pathlib import Path
from cfg import parse_cfg
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
if args.mode == 'gpu':
	from env_warp import make_env
else:
	import gym
	gym.logger.set_level(40)
	from env import make_env
torch.backends.cudnn.benchmark = True

PID = os.getpid()


def proc_vram_mib():
	"""VRAM of this process as reported by nvidia-smi (MiB; NaN if unavailable).

	This is the whole-process footprint (CUDA context, module images, mujoco_warp buffers, torch allocator
	cache, CUDA-graph pools) - the quantity that decides how many runs fit on the GPU, unlike the torch
	allocator statistics which only cover tensors.
	"""
	try:
		out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
							 capture_output=True, text=True, timeout=10).stdout
		for line in out.strip().splitlines():
			pid, mem = [x.strip() for x in line.split(',')]
			if int(pid) == PID:
				return float(mem)
	except Exception:
		pass
	return float('nan')


class GpuUtilSampler(threading.Thread):
	"""Background thread polling the GPU utilisation (nvidia-smi, every 0.25 s) during the timed phase."""

	def __init__(self):
		super().__init__(daemon=True)
		self.samples, self.stop = [], False

	def run(self):
		while not self.stop:
			try:
				o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
								   capture_output=True, text=True, timeout=5).stdout
				self.samples.append(float(o.strip().splitlines()[0]))
			except Exception:
				pass
			time.sleep(0.25)


def main():
	"""Build env / agent / buffer, run the timed collection + update phases and report one JSON record."""
	t_start = time.time()
	# hand the overrides to the original hydra-style cfg parser, as train.py would receive them
	sys.argv =[sys.argv[0], f'task={args.task}', f'horizon={args.horizon}', f'seed={args.seed}'] + args.extra
	cfg = parse_cfg(Path(CFGS))
	import random
	random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
	env = make_env(cfg)
	S = int(cfg.get('num_seeds', 1))
	EpisodeCls = Episode
	if S > 1 or cfg.get('vec_agent', False):  # vectorised multi-seed agent (algorithm/tdmpc_vec.py)
		from algorithm.tdmpc_vec import VecTDMPC, VecReplayBuffer, VecEpisode
		agent, buffer = VecTDMPC(cfg, S), VecReplayBuffer(cfg, S)
		EpisodeCls = lambda cfg_, init_obs: VecEpisode(cfg_, S, torch.zeros(S, *cfg_.obs_shape, device='cuda'))
	elif cfg.get('graph_agent', False):
		from algorithm.tdmpc_graph import GraphTDMPC, GraphReplayBuffer
		agent, buffer = GraphTDMPC(cfg), GraphReplayBuffer(cfg)
	else:
		agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
	gpu = args.mode == 'gpu'
	to_env = (lambda a: a) if gpu else (lambda a: a.cpu().numpy())
	L = cfg.episode_length
	BIG = 10 ** 6  # past horizon/std schedules (and past seed_steps)

	# synthetic fill: enough valid start indices (L - H per episode) for a batch without replacement
	n_fill = int(np.ceil(cfg.batch_size / max(1, L - cfg.horizon))) + 1
	for _ in range(n_fill):
		ep = EpisodeCls(cfg, np.zeros(cfg.obs_shape, dtype=np.float32))
		ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_(0, 1)
		ep._idx, ep.done = L, True
		buffer += ep
	t_setup = time.time() - t_start

	# host pacing (pace_host=true): after every step the CPU waits on a blocking-sync CUDA event, so it
	# sleeps instead of spinning / queueing far ahead of the GPU (same mechanism as src/train.py)
	pace = torch.cuda.Event(blocking=True) if cfg.get('pace_host', False) else None

	def collect(k):
		obs, t = env.reset(), 0
		for i in range(k):
			a = agent.plan(obs, step=BIG, t0=(t == 0))
			obs, r, done, _ = env.step(to_env(a))
			if pace is not None:
				pace.record(); pace.synchronize()
			t += 1
			if done:
				obs, t = env.reset(), 0

	def updates(k):
		for i in range(k):
			agent.update(buffer, BIG + i)
			if pace is not None:
				pace.record(); pace.synchronize()

	# warm-up (CUDA-graph capture, cudnn autotuning, allocator growth), then reset the torch peak so the
	# reported allocator peak covers the steady state only (nvidia-smi process VRAM includes everything)
	collect(5); updates(5); torch.cuda.synchronize()
	torch.cuda.reset_peak_memory_stats()

	util = GpuUtilSampler(); util.start()
	cpu0, w0 = time.process_time(), time.perf_counter()
	collect(args.k); torch.cuda.synchronize()
	w1 = time.perf_counter()
	updates(args.k); torch.cuda.synchronize()
	w2, cpu1 = time.perf_counter(), time.process_time()
	util.stop = True
	collect_ms = (w1 - w0) / args.k * 1e3
	update_ms = (w2 - w1) / args.k * 1e3

	# synced component breakdown (separate pass; the per-call synchronisation hides overlap, so these
	# numbers are only a breakdown - step_ms above is the training-loop cost)
	def timed(fn, n):
		"""Median wall time (ms) of `fn` over n calls, with a device sync before and after each call."""
		ts = []
		for _ in range(n):
			torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
		return float(np.median(ts) * 1e3)
	obs = env.reset()
	plan_ms = timed(lambda: agent.plan(obs, step=BIG, t0=False), 20)
	a = agent.plan(obs, step=BIG, t0=False)
	env_ms = timed(lambda: env.step(to_env(a)), 20)

	out = dict(mode=args.mode, task=args.task, horizon=args.horizon, k=args.k, num_seeds=S, seed_steps_per_s=S * 1000.0 / (collect_ms + update_ms), action_repeat=cfg.action_repeat,
			   episode_length=L, batch_size=cfg.batch_size, train_steps=cfg.train_steps,
			   torch=torch.__version__, setup_s=t_setup,
			   collect_ms_per_step=collect_ms, update_ms=update_ms, step_ms=collect_ms + update_ms,
			   plan_ms_synced=plan_ms, env_ms_synced=env_ms,
			   cpu_cores_used=(cpu1 - cpu0) / (w2 - w0),
			   gpu_util_mean=float(np.mean(util.samples)) if util.samples else float('nan'),
			   torch_peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
			   torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
			   process_vram_mib=proc_vram_mib(), wall_s=time.time() - t_start)
	print(json.dumps(out), flush=True)
	if args.out:
		os.makedirs(os.path.dirname(args.out), exist_ok=True)
		json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
	main()
