"""Multi-run throughput / CPU contention / GPU utilisation study (drives bench/multi_run.py).

For each task and N concurrent seeds (all at the same horizon):
  cpu        N processes of the ORIGINAL code (CPU MuJoCo), default threading (what one would normally run)
  cpu_pin    same, but each process pinned to 1 CPU thread (OMP/MKL/torch)
  gpu        N processes of the faithful GPU port (eager original agent)
  gpu_opt    N processes of the optimised GPU port (CUDA-graph agent + host pacing), plain time-slicing
  gpu_mps    same as gpu_opt under CUDA MPS (kernels of different processes may run concurrently)
  vec        ONE process with N seeds (vectorised agent + N-world env)
Results: results/contention/<task>/<mode>_N<N>.json (resumable)

  python bench/contention_study.py [--tasks ...] [--ns 1 2 4 8 16] [--modes cpu gpu_opt gpu_mps vec] [--duration 30]

Run in the `tdmpc-warp` env; the cpu modes launch the original code in the `tdmpc` env (via
bench/multi_run.py -> bench/run_worker.py --mode cpu). cpu_pin and gpu only run for --extra_modes_tasks
(walker-walk, dog-run by default). Each point is a multi_run.py call with a timed phase of --duration s.
Tabulated by bench/analyze_interference.py (docs/results_capacity.md, section B, and
docs/figures/contention_throughput.png); reported in docs/04_multi_run_capacity.md (4.1, 4.7, 4.8) and
used for the "5 seeds parallel" columns of docs/results_runtime.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)

p = argparse.ArgumentParser()
p.add_argument('--tasks', nargs='+', default=['cartpole-swingup', 'walker-walk', 'humanoid-run', 'dog-run'])
p.add_argument('--ns', nargs='+', type=int, default=[1, 2, 4, 8, 16])
p.add_argument('--modes', nargs='+', default=['cpu', 'gpu_opt', 'gpu_mps', 'vec'])
p.add_argument('--extra_modes_tasks', nargs='+', default=['walker-walk', 'dog-run'],
			   help='tasks that additionally get the cpu_pin and gpu modes')
p.add_argument('--horizon', type=int, default=5)
p.add_argument('--duration', type=float, default=30)
a = p.parse_args()

for task in a.tasks:
	modes = list(a.modes) + (['cpu_pin', 'gpu'] if task in a.extra_modes_tasks else [])
	for mode in modes:
		for n in a.ns:
			out = os.path.join(ROOT, 'results', 'contention', task, f'{mode}_N{n}.json')
			if os.path.exists(out):
				continue
			H = a.horizon
			# translate the study mode into multi_run.py specs (N* = N identical processes) and flags
			if mode in ('cpu', 'cpu_pin'):
				spec, flags = [f'{n}*cpu:{task}:{H}'], ['--threads', '1' if mode == 'cpu_pin' else '0']
			elif mode == 'gpu':
				spec, flags = [f'{n}*gpu:{task}:{H}'], []
			elif mode in ('gpu_opt', 'gpu_mps'):
				spec, flags = [f'{n}*gpu:{task}:{H}:graph_agent=true,pace_host=true'], (['--mps'] if mode == 'gpu_mps' else [])
			else:  # vec: a single process holding all N seeds (N-world env, stacked agents)
				spec, flags = [f'1*gpu:{task}:{H}:vec_agent=true,num_seeds={n},pace_host=true'], []
			cmd = [PY, os.path.join(HERE, 'multi_run.py'), '--spec'] + spec + ['--duration', str(a.duration), '--out', out] + flags
			r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
			line = (r.stdout.strip().splitlines() or [''])[-1]  # multi_run prints its summary as the last line
			try:
				d = json.loads(line)
				print(f'{task:18s} {mode:8s} N={n:2d}: total {d["steps_per_s_total"]:7.1f} seed-steps/s, ok {d["n_ok"]}/{d["n_runs"]}, '
					  f'oom {d["n_oom"]}, gpu util {d["gpu_util_mean"]:.0f}%, sys cpu {100*d["system_cpu_util"]:.0f}%, mem {d["gpu_mem_peak_mib"]:.0f} MiB', flush=True)
			except Exception:
				print(f'{task} {mode} N={n}: FAILED {r.stderr[-300:]}', flush=True)
print('contention study done', flush=True)
