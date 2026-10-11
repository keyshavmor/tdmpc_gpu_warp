"""Measured 5-seed batch times under MPS vs the predicted times and the original CPU code (table for the docs).

For every task with a completed batch of bench/mps_batches.py (results/mps_batches/<label>/summary.json) it reports
the measured wall-clock time of the batch (5 seeds in parallel under MPS, from the first start to the last run's end),
the predicted time of docs/results_runtime.md (column "5 seeds parallel: GPU optimised + MPS"), the measured parallel
factor (GPU optimised 5 seeds back-to-back / measured batch time, i.e. how much faster the batch is than running the 5
seeds one after another) and the speed-up over the original CPU-MuJoCo code with 5 seeds in parallel and back-to-back
(columns of the same table), and the number of MPS fault events (one fault ends all co-running runs). Prints a
markdown table with a total row.

Environment: any Python 3 (tdmpc-warp).
Usage: python bench/runtime_measured_table.py [--label baseline_mps]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, datetime, json, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
p = argparse.ArgumentParser()
p.add_argument('--label', default='baseline_mps')
a = p.parse_args()
batches = json.load(open(os.path.join(ROOT, 'results', 'mps_batches', a.label, 'summary.json')))['batches']


def hours(cell):
	"""Hours of a run-time cell of docs/results_runtime.md ('3.8 h' or '93 h (3.9 d)'), None if not a time."""
	m = re.match(r'([0-9.]+) h', cell.strip())
	return float(m.group(1)) if m else None


def fault_events(b):
	"""Number of MPS fault events of a batch: crashes within 60 s of each other are one fault (it ends all clients)."""
	ts = sorted(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp() for t in b.get('crash_times', []))
	return sum(1 for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > 60)


# predicted table of docs/results_runtime.md: task -> columns (cells after the task name)
pred = {}
for line in open(os.path.join(ROOT, 'docs', 'results_runtime.md')):
	cells = [c.strip() for c in line.strip().strip('|').split('|')]
	if len(cells) == 10 and cells[1].endswith('M') and cells[0] not in pred:
		pred[cells[0]] = cells
print('| task | measured, 5 seeds parallel + MPS | predicted | measured parallel factor (5 seeds back-to-back / batch) '
	  '| vs CPU original, 5 seeds parallel | vs CPU original, back-to-back | MPS faults |')
print('|---|---|---|---|---|---|---|')
tot = {'m': 0.0, 'p': 0.0, 'b2b_gpu': 0.0, 'par_cpu': 0.0, 'b2b_cpu': 0.0, 'crashes': 0}
for b in batches:
	if not b.get('complete'):
		continue
	c, m = pred[b['task']], b['wall_h']
	b2b_cpu, b2b_gpu, par_cpu, par_mps = hours(c[5]), hours(c[6]), hours(c[7]), hours(c[8])
	print(f"| {b['task']} | {m:.2f} h | {par_mps} h | {b2b_gpu / m:.2f}x | {par_cpu / m:.1f}x faster | {b2b_cpu / m:.1f}x faster | {fault_events(b)} |")
	for k, v in (('m', m), ('p', par_mps), ('b2b_gpu', b2b_gpu), ('par_cpu', par_cpu), ('b2b_cpu', b2b_cpu), ('crashes', fault_events(b))):
		tot[k] += v
n = sum(1 for b in batches if b.get('complete'))
print(f"| **{n} tasks** | **{tot['m']:.1f} h** | {tot['p']:.1f} h | {tot['b2b_gpu'] / tot['m']:.2f}x | **{tot['par_cpu'] / tot['m']:.1f}x** "
	  f"| **{tot['b2b_cpu'] / tot['m']:.1f}x** | {tot['crashes']} |")
print(f"\nGPU back-to-back total {tot['b2b_gpu']:.1f} h, CPU original 5 parallel {tot['par_cpu']:.1f} h, back-to-back {tot['b2b_cpu']:.1f} h; "
	  f"process-hours under MPS {5 * tot['m']:.1f}")
