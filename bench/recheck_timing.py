"""Re-time the optimised GPU port after a code change and compare with the stored horizon sweep.

Runs bench/bench_step.py (same benchmark as bench/sweep_h.py, mode gpu_opt = CUDA-graph agent + host pacing) for
the given tasks and horizons, each point in a fresh process on an otherwise idle GPU, writes the results to
results/sweep_h_recheck/<label>/<task>_H<H>.json and prints the ratio to results/sweep_h/gpu_opt/ (same point).
Used after the PER replacement-flag fix (docs/06_optimisations.md, section 6.4) to confirm that the speed numbers
of the reports still hold.

Environment: tdmpc-warp (a CUDA GPU; run it while nothing else uses the GPU).
Usage: python bench/recheck_timing.py [--label per_fix] [--horizons 5 30] [--tasks ...] [--k 60] [--extra key=value ...]
(--extra times an optimisation variant against the stored baseline, e.g. --label fused --extra fused_adam=true)
Default tasks: the 15 tasks of the baseline scope (all except humanoid-* and dog-*).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TASKS15 = [t.strip() for t in open(os.path.join(ROOT, 'tasks.txt')) if t.strip() and not t.startswith(('humanoid', 'dog'))]

p = argparse.ArgumentParser()
p.add_argument('--label', default='per_fix')
p.add_argument('--horizons', type=int, nargs='+', default=[5, 30])
p.add_argument('--tasks', nargs='+', default=TASKS15)
p.add_argument('--k', type=int, default=60)
p.add_argument('--extra', nargs='*', default=[], help='extra agent options, e.g. fused_adam=true compile_model=true')
a = p.parse_args()
outdir = os.path.join(ROOT, 'results', 'sweep_h_recheck', a.label); os.makedirs(outdir, exist_ok=True)
rows = []
for task in a.tasks:
	for H in a.horizons:
		out = os.path.join(outdir, f'{task}_H{H}.json')
		if not os.path.exists(out):
			cmd = [sys.executable, os.path.join(HERE, 'bench_step.py'), '--mode', 'gpu', '--task', task, '--horizon', str(H),
				   '--k', str(a.k), '--out', out, '--extra', 'graph_agent=true', 'pace_host=true'] + a.extra
			subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
		ref_f = os.path.join(ROOT, 'results', 'sweep_h', 'gpu_opt', f'{task}_H{H}.json')
		if not (os.path.exists(out) and os.path.exists(ref_f)):
			print(f'{task} H={H}: missing result', flush=True); continue
		new, ref = json.load(open(out)), json.load(open(ref_f))
		r = dict(task=task, H=H, step_ms_new=new['step_ms'], step_ms_ref=ref['step_ms'], ratio=new['step_ms'] / ref['step_ms'],
				 update_ms_new=new['update_ms'], update_ms_ref=ref['update_ms'], vram_new=new['process_vram_mib'], vram_ref=ref['process_vram_mib'])
		rows.append(r)
		print(f"{task:24s} H={H:2d} step {r['step_ms_ref']:7.2f} -> {r['step_ms_new']:7.2f} ms ({r['ratio']:.3f}x)  "
			  f"update {r['update_ms_ref']:6.2f} -> {r['update_ms_new']:6.2f} ms  VRAM {r['vram_ref']:.0f} -> {r['vram_new']:.0f} MiB", flush=True)
json.dump(rows, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
if rows:
	import statistics
	print(f"step-time ratio new/ref: mean {statistics.mean(r['ratio'] for r in rows):.3f}, "
		  f"min {min(r['ratio'] for r in rows):.3f}, max {max(r['ratio'] for r in rows):.3f} over {len(rows)} points")
