"""Horizon sweep: steady-state time per agent step and peak VRAM for every task x H x mode.

Driver script (run it in the `tdmpc-warp` env): every (task, H, mode) point is one fresh subprocess of
bench/bench_step.py, so each measurement starts from an empty GPU context. GPU modes run with the
`tdmpc-warp` interpreter; the cpu mode runs the ORIGINAL code with the interpreter of the `tdmpc` conda env.

Modes:
  cpu      original tdmpc/src (CPU MuJoCo 2.1 / dm_control, agent on GPU), `tdmpc` env, original threads
  gpu      faithful GPU port (mujoco_warp env, original eager agent)
  gpu_opt  GPU port + CUDA-graph agent + host pacing (+ blocking sync, default on)
  vec_*    single-seed vectorised agent in fp32 / tf32 / bf16 / fp16 / bf16 + bf16 replay buffer /
           fp32 + activation checkpointing (precision and memory study, same code path for all variants)
Resumable: existing results/sweep_h/<mode>/<task>_H<H>.json are skipped; a failed point is recorded as
{ok: false, oom: ...} so that OOMs show up in the tables.
  python bench/sweep_h.py [--modes cpu gpu gpu_opt] [--horizons 2 5 10 15 20 25 30] [--tasks ...] [--k 60]
  python bench/sweep_h.py --modes vec_fp32 vec_tf32 vec_bf16 vec_fp16 vec_bf16_buf vec_ckpt --tasks walker-walk

Reported in docs/02_single_run_performance.md (tables: docs/results_sweep_h.md, figures by
bench/analyze_sweep.py: 441 configurations, 0 failures) and, for the vec_* modes, docs/07_precision.md
(docs/results_precision.md, bench/analyze_precision.py). The sweep also provides the per-process VRAM used
for the memory limits in docs/04_multi_run_capacity.md and the OOM-horizon fits (bench/oom_horizon.py).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = {'cpu': os.environ.get('TDMPC_CPU_PYTHON', os.path.expanduser('~/anaconda3/envs/tdmpc/bin/python')),
	  'gpu': os.environ.get('TDMPC_WARP_PYTHON', sys.executable)}
# mode -> (bench_step --mode, extra bench_step flags, cfg overrides passed via --extra)
MODES = {
	'cpu': ('cpu', ['--threads', '0'], []),  # --threads 0: keep the original code's default CPU threading
	'gpu': ('gpu', [], []),
	'gpu_opt': ('gpu', [], ['graph_agent=true', 'pace_host=true']),
	# precision / memory study (vectorised agent with a single seed; same code path for all variants)
	'vec_fp32': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true']),
	'vec_tf32': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'precision=tf32']),
	'vec_bf16': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'precision=bf16']),
	'vec_fp16': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'precision=fp16']),
	'vec_bf16_buf': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'precision=bf16', 'buffer_dtype=bf16']),
	'vec_ckpt': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'ckpt_unroll=true']),
}

p = argparse.ArgumentParser()
p.add_argument('--modes', nargs='+', default=['cpu', 'gpu', 'gpu_opt'])
p.add_argument('--horizons', nargs='+', type=int, default=[2, 5, 10, 15, 20, 25, 30])
p.add_argument('--tasks', nargs='+', default=None)
p.add_argument('--k', type=int, default=60)
p.add_argument('--outdir', default=os.path.join(ROOT, 'results', 'sweep_h'))
a = p.parse_args()
tasks = a.tasks or [l.strip() for l in open(os.path.join(ROOT, 'tasks.txt')) if l.strip()]
todo = [(t, H, m) for t in tasks for H in a.horizons for m in a.modes]
t0 = time.time()
for i, (task, H, mode) in enumerate(todo):
	out = os.path.join(a.outdir, mode, f'{task}_H{H}.json')
	if os.path.exists(out):
		continue
	os.makedirs(os.path.dirname(out), exist_ok=True)
	m, flags, extra = MODES[mode]
	cmd = [PY[m], os.path.join(HERE, 'bench_step.py'), '--mode', m, '--task', task, '--horizon', str(H), '--k', str(a.k),
		   '--out', out] + flags + (['--extra'] + extra if extra else [])
	ts = time.time()
	r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
	if r.returncode != 0 or not os.path.exists(out):
		err = (r.stderr or '')[-600:]
		json.dump(dict(task=task, horizon=H, mode=mode, ok=False, oom='out of memory' in err.lower(), error=err), open(out, 'w'))
		print(f'[{i+1}/{len(todo)}] {mode} {task} H={H} FAILED ({time.time()-ts:.0f}s): {err[-200:]}', flush=True)
	else:
		d = json.load(open(out))
		print(f'[{i+1}/{len(todo)}] {mode:7s} {task:24s} H={H:2d} step {d["step_ms"]:8.2f} ms  vram {d["process_vram_mib"]:6.0f} MiB  '
			  f'({time.time()-ts:.0f}s, total {(time.time()-t0)/60:.0f} min)', flush=True)
print('sweep done', flush=True)
