"""Final-results table of the 15-task x 5-seed baseline (5 seeds in parallel under MPS, one task after another).

For every task with a completed 5-seed batch (bench/mps_batches.py, label baseline_mps) it reports the evaluation
return at the end of training of seeds 1-5 (last line of each seed's eval.log: mean over cfg.eval_episodes = 10
episodes at the final env step, 500k for all 15 tasks), the mean and standard deviation across the 5 seeds, and the
run time of each seed (total_time of the last episode in train.log, i.e. wall-clock time of that run from start to
end of training, including evaluations and checkpoints) as mean and standard deviation across the 5 seeds. Standard
deviations are sample standard deviations (n - 1). Writes docs/results_baseline_15tasks.md (and prints it).

Environment: any Python 3 with numpy (tdmpc-warp).
Usage: python bench/final_results_table.py [--label baseline_mps]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, csv, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TASKS15 = [t.strip() for t in open(os.path.join(ROOT, 'tasks.txt')) if t.strip() and not t.startswith(('humanoid', 'dog'))]
p = argparse.ArgumentParser()
p.add_argument('--label', default='baseline_mps')
a = p.parse_args()
base = os.path.join(ROOT, 'results', 'mps_batches', a.label)
summary = {b['task']: b for b in json.load(open(os.path.join(base, 'summary.json')))['batches']} if os.path.exists(os.path.join(base, 'summary.json')) else {}


def last_row(path):
	"""Last data row of a CSV log as a dict (None if missing / empty)."""
	if not os.path.exists(path):
		return None
	rows = list(csv.DictReader(open(path)))
	return rows[-1] if rows else None


def paper_final(task):
	"""Paper reference (original TD-MPC, 5 seeds) at 500k env steps (or its last step): mean, sd (n - 1), step."""
	f = os.path.join(ROOT, '..', 'tdmpc', 'results', f'{task}.csv')
	if not os.path.exists(f):
		return None
	rows = list(csv.DictReader(open(f)))
	steps = {int(r['step']) for r in rows}
	at = 500000 if 500000 in steps else max(steps)  # compare at the end of the 500k-step budget
	v = [float(r['return']) for r in rows if int(r['step']) == at]
	return np.mean(v), np.std(v, ddof=1), at


md = ['# Baseline results: 15 tasks x 5 seeds (GPU port, 5 seeds in parallel under MPS)', '',
	  '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
	  'Setup: optimised GPU port (CUDA-graph agent + host pacing, `src/launch_parallel.py --mps`), original TD-MPC hyperparameters '
	  '(H = 5), full training budget of 500k environment steps per run, 5 seeds (1-5) of a task in parallel under CUDA MPS, tasks one '
	  'after another on one RTX 3090, checkpoints every 15 min, W&B logging (`bench/mps_batches.py`, label `' + a.label + '`).', '',
	  '**Final return** = evaluation return at the end of training (mean over 10 evaluation episodes at 500k env steps). '
	  '**Run time** = wall-clock time of each run from start to the end of training (5 runs share the GPU). Mean and standard '
	  'deviation (sample sd, n - 1) across the 5 seeds. The paper reference is the original TD-MPC (5 seeds) at 500k env steps. '
	  'Learning curves and statistical comparison with the paper: [results_mps_batches.md](results_mps_batches.md).', '',
	  '| task | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std. dev. | paper (mean +- sd) | run time mean (h) | run time std. dev. (h) | batch crashes |',
	  '|---|---|---|---|---|---|---|---|---|---|---|---|']
done = 0
for t in TASKS15:
	d = os.path.join(base, t, 'state', a.label)
	finals, times = [], []
	for s in range(1, 6):
		e = last_row(os.path.join(d, str(s), 'eval.log'))
		tr = last_row(os.path.join(d, str(s), 'train.log'))
		complete = e is not None and tr is not None and int(float(e['env_step'])) >= 500000
		finals.append(float(e['episode_reward']) if complete else None)
		times.append(float(tr['total_time']) / 3600 if complete else None)
	ref = paper_final(t)
	refs = f'{ref[0]:.1f} +- {ref[1]:.1f}' if ref else '-'
	if all(v is not None for v in finals):
		done += 1
		cr = summary.get(t, {}).get('crashes', '-')
		md.append(f"| {t} | " + ' | '.join(f'{v:.1f}' for v in finals) +
				  f" | **{np.mean(finals):.1f}** | {np.std(finals, ddof=1):.1f} | {refs} | {np.mean(times):.2f} | {np.std(times, ddof=1):.2f} | {cr} |")
	else:
		state = 'running' if os.path.isdir(d) else 'pending'
		md.append(f"| {t} | " + ' | '.join(state if v is None else f'{v:.1f}' for v in finals) + f" | - | - | {refs} | - | - | - |")
md += ['', f'{done} of {len(TASKS15)} tasks complete.', '']
open(os.path.join(ROOT, 'docs', 'results_baseline_15tasks.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
