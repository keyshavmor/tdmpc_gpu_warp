"""Real 5-seed training batches under CUDA MPS: measured wall-clock time, fault statistics, resume in practice.

For each task, runs src/launch_parallel.py with 5 seeds of the optimised GPU port (CUDA-graph agent + host pacing)
under MPS for the full default training budget, with checkpoints and automatic resume after a crash. Records per
batch: wall-clock time, launches, crashes and restarts (from the launcher's event log), and at the end the
kernel's GPU fault records (Xid lines from `journalctl -k`) for the whole period. The learning curves
(eval.log per seed) are compared with the paper's reference curves by bench/analyze_learning.py.

Environment: tdmpc-warp; the GPU should not run other CUDA jobs (they would change the timings).
Usage:
	python bench/mps_batches.py --tasks walker-walk cheetah-run --label baseline_mps [-- use_wandb=true wandb_project=...]
Arguments after `--` are passed to every run. Output: results/mps_batches/<label>/ (logs, event logs, summary.json).
Results: docs/04_multi_run_capacity.md (section 4.9) and docs/08_known_issues.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, csv, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

p = argparse.ArgumentParser()
p.add_argument('--tasks', nargs='+', required=True)
p.add_argument('--label', default='baseline_mps')
p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
p.add_argument('--checkpoint_minutes', type=float, default=10.0)
a, extra = p.parse_known_args()
extra = [x for x in extra if x != '--']
out = os.path.join(ROOT, 'results', 'mps_batches', a.label); os.makedirs(out, exist_ok=True)
summary_f = os.path.join(out, 'summary.json')
summary = json.load(open(summary_f)) if os.path.exists(summary_f) else {'batches': []}
t_start = time.strftime('%Y-%m-%d %H:%M:%S')
for task in a.tasks:
	if any(b['task'] == task and b.get('complete') for b in summary['batches']):
		continue  # resumable driver: skip finished batches
	ev = os.path.join(out, f'events_{task}.csv')
	cmd = [sys.executable, os.path.join(ROOT, 'src', 'launch_parallel.py'), '--task', task, '--seeds', *map(str, a.seeds),
		   '--mps', '--checkpoint_minutes', str(a.checkpoint_minutes), '--log_root', out, '--exp_name', a.label,
		   '--events', ev, '--'] + extra
	t0 = time.time()
	with open(os.path.join(out, f'launcher_{task}.log'), 'a') as lf:
		subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
	wall = time.time() - t0
	events = list(csv.DictReader(open(ev))) if os.path.exists(ev) else []
	count = lambda e: sum(1 for r in events if r['event'] == e)
	b = dict(task=task, wall_h=wall / 3600, launches=count('start') + count('restart'), crashes=count('crash'),
			 restarts=count('restart'), finished=count('finished'), complete=count('finished') == len(a.seeds),
			 crash_times=[r['time'] for r in events if r['event'] == 'crash'])
	summary['batches'] = [x for x in summary['batches'] if x['task'] != task] + [b]
	json.dump(summary, open(summary_f, 'w'), indent=1)
	print(json.dumps(b), flush=True)
# GPU fault records of the kernel driver during the whole period
try:
	x = subprocess.run(['journalctl', '-k', '--since', t_start], capture_output=True, text=True, timeout=120).stdout
	xid = [l for l in x.splitlines() if 'Xid' in l]
except (OSError, subprocess.SubprocessError):
	xid = ['journalctl not available']
summary.setdefault('xid', []).extend(xid)
json.dump(summary, open(summary_f, 'w'), indent=1)
print(f'{len(xid)} Xid lines since {t_start}', flush=True)
