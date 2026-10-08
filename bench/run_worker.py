"""One TD-MPC run in steady state, for multi-run (concurrency / packing) experiments.

Builds env + agent + full-capacity replay buffer (exactly as training), fills the buffer with synthetic
episodes, warms up, writes <sync>/ready.<id>, waits for <sync>/go, then repeats the training-loop
pattern [collect `chunk` steps (plan + env.step), `chunk` updates] for `duration` seconds at a step
past the horizon/std schedules (planning horizon = H). Writes a JSON result.

  python bench/run_worker.py --mode {cpu,gpu} --task T --horizon H --sync DIR --id I --duration S
                             [--extra graph_agent=true ...]
Exit code 3 = CUDA out of memory (reported in the JSON).

Not meant to be started by hand: bench/multi_run.py (timed concurrent runs) and bench/pack_test.py
(packing until OOM) launch one worker per run - with the `tdmpc-warp` interpreter for --mode gpu and with
the `tdmpc` interpreter for --mode cpu (which imports the untouched original ../tdmpc/src).

Synchronisation via files in the --sync directory (a start barrier, so that all runs are already built,
warmed up and holding their memory when the timed phase starts):
  ready.<id>        written by the worker when set-up + warm-up are done (also written on failure)
  go                written by the driver once all workers are ready -> timed phase starts
  stop              packing mode (--duration 0) only: release the memory and exit
  result.<id>.json  the worker's result (steps_per_s = seeds x agent steps / s, CPU cores, torch peak)
With --duration 0 (packing mode) the worker reports its memory right after warm-up and holds it until
`stop` appears. Results end up in docs/04_multi_run_capacity.md via the drivers listed above.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, sys, time

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['gpu', 'cpu'], required=True)
p.add_argument('--task', required=True)
p.add_argument('--horizon', type=int, required=True)
p.add_argument('--sync', required=True)
p.add_argument('--id', required=True)
p.add_argument('--duration', type=float, default=60)
p.add_argument('--chunk', type=int, default=50)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--threads', type=int, default=1, help='0 = do not pin CPU threads (original behaviour)')
p.add_argument('--hold', type=float, default=0, help='after the timed phase keep memory allocated for N s')
p.add_argument('--extra', nargs='*', default=[])
args = p.parse_args()

# pin math libraries to one thread before numpy / torch are imported (--threads 0: original behaviour)
if args.mode == 'gpu' or args.threads == 1:
	for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
		os.environ[_k] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src') if args.mode == 'gpu' else os.path.join(HERE, '..', '..', 'tdmpc', 'src')
CFGS = os.path.join(HERE, '..', 'cfgs') if args.mode == 'gpu' else os.path.join(HERE, '..', '..', 'tdmpc', 'cfgs')
sys.path.insert(0, os.path.abspath(SRC))
import warnings
warnings.filterwarnings('ignore')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
OUT = os.path.join(args.sync, f'result.{args.id}.json')


def write(d):
	"""Write the worker's result record to <sync>/result.<id>.json."""
	json.dump(d, open(OUT, 'w'))


def main():
	"""Build, warm up, wait at the start barrier, then run the timed collect/update loop (see module doc)."""
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
	sys.argv = [sys.argv[0], f'task={args.task}', f'horizon={args.horizon}', f'seed={args.seed}'] + args.extra
	cfg = parse_cfg(Path(CFGS))
	np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
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
	L, BIG = cfg.episode_length, 10 ** 6  # BIG: a step past the horizon / std schedules
	# synthetic episodes: enough valid start indices (L - H per episode) for one batch without replacement
	for _ in range(int(np.ceil(cfg.batch_size / max(1, L - cfg.horizon))) + 1):
		ep = EpisodeCls(cfg, np.zeros(cfg.obs_shape, dtype=np.float32))
		ep.obs.normal_(); ep.action.uniform_(-1, 1); ep.reward.uniform_(0, 1)
		ep._idx, ep.done = L, True
		buffer += ep
	state = {'obs': env.reset(), 't': 0}  # carried across chunks: episodes continue between timed chunks
	# host pacing (pace_host=true): blocking-sync event after every step, as in src/train.py
	pace =torch.cuda.Event(blocking=True) if cfg.get('pace_host', False) else None

	def collect(k):
		for _ in range(k):
			a = agent.plan(state['obs'], step=BIG, t0=(state['t'] == 0))
			state['obs'], r, done, _ = env.step(to_env(a))
			if pace is not None:
				pace.record(); pace.synchronize()
			state['t'] += 1
			if done:
				state['obs'], state['t'] = env.reset(), 0

	def updates(k):
		for i in range(k):
			agent.update(buffer, BIG + i)
			if pace is not None:
				pace.record(); pace.synchronize()

	collect(10); updates(10); torch.cuda.synchronize()
	if args.duration <= 0:  # packing mode: peak memory is allocated (and cached); report and hold it
		write(dict(id=args.id, mode=args.mode, task=args.task, horizon=args.horizon, extra=args.extra, ok=True, num_seeds=S,
				   torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20, pid=os.getpid()))
		open(os.path.join(args.sync, f'ready.{args.id}'), 'w').close()
		while not os.path.exists(os.path.join(args.sync, 'stop')):
			time.sleep(0.2)
		return
	# synchronised start: announce readiness, then poll for the driver's `go` file so that all concurrent
	# runs enter the timed phase at the same moment (set-up / warm-up never overlap with timed work)
	open(os.path.join(args.sync, f'ready.{args.id}'), 'w').close()
	while not os.path.exists(os.path.join(args.sync, 'go')):
		time.sleep(0.05)
	steps, t0, c0 = 0, time.perf_counter(), time.process_time()
	while time.perf_counter() - t0 < args.duration:
		collect(args.chunk); updates(args.chunk)
		torch.cuda.synchronize()
		steps += args.chunk
	wall, cpu = time.perf_counter() - t0, time.process_time() - c0
	res = dict(id=args.id, mode=args.mode, task=args.task, horizon=args.horizon, extra=args.extra, ok=True, num_seeds=S,
			   agent_steps=steps, wall_s=wall, steps_per_s=S * steps / wall, cpu_cores=cpu / wall,
			   torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20, pid=os.getpid())
	write(res)
	if args.hold > 0:
		time.sleep(args.hold)


if __name__ == '__main__':
	try:
		main()
	except Exception as e:  # report OOM (and other failures) instead of dying silently
		msg = f'{type(e).__name__}: {e}'
		oom = 'out of memory' in msg.lower() or 'OutOfMemory' in msg
		write(dict(id=args.id, task=args.task, horizon=args.horizon, extra=args.extra, ok=False, oom=oom, error=msg[:500]))
		# also signal `ready` so the driver's barrier / packing loop does not wait for this run
		open(os.path.join(args.sync, f'ready.{args.id}'), 'w').close()
		sys.exit(3 if oom else 1)
