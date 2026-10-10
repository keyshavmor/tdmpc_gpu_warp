"""Analyse the real 5-seed training batches under MPS (bench/mps_batches.py): time, faults, learning vs the paper.

Per task: measured wall-clock time of the batch vs the predicted "5 seeds in parallel, GPU + MPS" time of
docs/results_runtime.md, launches / crashes / restarts (launcher event log), and the learning curves of the 5 seeds
(eval.log) vs the paper's reference curves of the original TD-MPC (tdmpc/results/<task>.csv, 5 seeds): mean +- sd at
every evaluation step with per-step Welch p-values (descriptive: neighbouring steps are correlated, so their count
overstates the evidence), and as the primary test one Welch t-test per task on the per-seed mean return over the
whole run (area under the learning curve) and over the plateau (from 100k steps). Writes docs/results_mps_batches.md
and docs/figures/mps_batches_learning.png.

Environment: tdmpc-warp (pandas, matplotlib). The reference curves are read from ../tdmpc/results (original repository).
Usage: python bench/analyze_mps_batches.py [--label baseline_mps]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, csv, glob, json, os, re
from scipy import stats
from math import erf, sqrt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REF = os.path.join(ROOT, '..', 'tdmpc', 'results')
p = argparse.ArgumentParser()
p.add_argument('--label', default='baseline_mps')
a = p.parse_args()
base = os.path.join(ROOT, 'results', 'mps_batches', a.label)
summary = json.load(open(os.path.join(base, 'summary.json')))


def welch_p(x, y):
	"""Two-sided Welch test p-value with a normal approximation (NaN if fewer than 2 samples)."""
	x, y = np.asarray(x, float), np.asarray(y, float)
	if len(x) < 2 or len(y) < 2:
		return float('nan')
	t = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y) + 1e-12)
	return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def predicted_hours(task):
	"""Predicted 5-seed GPU + MPS time from docs/results_runtime.md (column 'GPU optimised + MPS')."""
	for line in open(os.path.join(ROOT, 'docs', 'results_runtime.md')):
		if line.startswith(f'| {task} |'):
			cells = [c.strip() for c in line.strip().strip('|').split('|')]
			m = re.match(r'([0-9.]+) h', cells[8])
			return float(m.group(1)) if m else None
	return None


md = ['# Real 5-seed training batches under MPS', '',
	  '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
	  f'Label `{a.label}`: 5 single-seed runs per task in parallel under CUDA MPS (`src/launch_parallel.py --mps`, CUDA-graph agent + '
	  'host pacing, default hyperparameters, H = 5, full training budget, checkpoints every 15 min, W&B logging), tasks one after '
	  'another, no other CUDA process on the GPU (docs/USER_GUIDE.md section 4.4). Reference = the paper\'s 5 seeds of the original '
	  'code; p = Welch test per evaluation step.', '',
	  '## Time and stability', '',
	  '| task | measured wall time (h) | predicted (h, docs/results_runtime.md) | launches | crashes | restarts |', '|---|---|---|---|---|---|']
curves = {}
for b in summary['batches']:
	t = b['task']; pred = predicted_hours(t)
	md.append(f"| {t} | {b['wall_h']:.2f} | {pred if pred is not None else '-'} | {b['launches']} | {b['crashes']} | {b['restarts']} |")
	runs = {}
	for f in glob.glob(os.path.join(base, t, 'state', a.label, '*', 'eval.log')):
		d = pd.read_csv(f)
		if len(d):
			runs[os.path.basename(os.path.dirname(f))] = d.set_index('env_step')['episode_reward']
	curves[t] = runs
xid = [x for x in summary.get('xid', []) if 'Xid' in x]
md += ['', f"Kernel GPU fault records (Xid) during the batches: {len(xid)}.", '', '## Learning vs the paper', '']
for t, runs in curves.items():
	ref = pd.read_csv(os.path.join(REF, f'{t}.csv'))
	refg = {s: g['return'].values for s, g in ref.groupby('step')}
	steps = sorted({s for r in runs.values() for s in r.index if s in refg})
	ps = []
	md += [f'### {t}', '', '| env step | reference (paper) | GPU port, 5 seeds under MPS | p |', '|---|---|---|---|']
	for s in steps:
		vals = [r[s] for r in runs.values() if s in r.index]
		pv = welch_p(vals, refg[s]); ps.append(pv)
		if s % 100000 == 0 or s == steps[-1]:
			md.append(f"| {int(s) // 1000}k | {refg[s].mean():.0f} +- {refg[s].std():.0f} | {np.mean(vals):.0f} +- {np.std(vals):.0f} ({len(vals)}) | {pv:.2f} |")
	ps = [x for x in ps if not np.isnan(x)]
	md += ['', f"Per-step (descriptive): {len(ps)} evaluation steps, min p = {min(ps):.3f}, steps with p < 0.05: "
		   f"{sum(1 for x in ps if x < 0.05)} (expected by chance: {0.05 * len(ps):.1f})." if ps else '']
	common = [st for st in steps if all(st in r.index for r in runs.values())]
	for lo, name in ((0, 'whole run'), (100000, 'from 100k steps')):
		st = [x for x in common if x >= lo]
		if len(st) < 2 or len(runs) < 2:
			continue
		gpu = [r.loc[st].mean() for r in runs.values()]
		pap = [g.set_index('step').reindex(st)['return'].mean() for _, g in ref.groupby('seed')]
		tt = stats.ttest_ind(gpu, pap, equal_var=False)
		md.append(f"Primary test, per-seed mean return over the {name}: GPU {np.mean(gpu):.1f} +- {np.std(gpu, ddof=1):.1f} vs "
				  f"paper {np.mean(pap):.1f} +- {np.std(pap, ddof=1):.1f} ({100 * (np.mean(gpu) / np.mean(pap) - 1):+.2f} %), Welch t-test p = {tt.pvalue:.3f}.")
		# optional reference: the original CPU-MuJoCo code run on this machine (bench/train_cpu_baseline.py, 5 seeds)
		cpu_runs = {os.path.basename(os.path.dirname(f)): pd.read_csv(f).set_index('env_step')['episode_reward']
					for f in glob.glob(os.path.join(ROOT, 'results', 'learning_full', t, 'state', 'cpu_original', '*', 'eval.log'))}
		if len(cpu_runs) >= 2 and all(all(x in r.index for x in st) for r in cpu_runs.values()):
			cpu = [r.loc[st].mean() for r in cpu_runs.values()]
			t1 = stats.ttest_ind(gpu, cpu, equal_var=False); t2 = stats.ttest_ind(cpu, pap, equal_var=False)
			md.append(f"  Original code on this machine ({len(cpu)} seeds, {name}): {np.mean(cpu):.1f} +- {np.std(cpu, ddof=1):.1f}; "
					  f"GPU vs original-on-this-machine {100 * (np.mean(gpu) / np.mean(cpu) - 1):+.2f} % (p = {t1.pvalue:.3f}), "
					  f"original-on-this-machine vs paper {100 * (np.mean(cpu) / np.mean(pap) - 1):+.2f} % (p = {t2.pvalue:.3f}).")
	md.append('')
open(os.path.join(ROOT, 'docs', 'results_mps_batches.md'), 'w').write('\n'.join(md) + '\n')
# figure: one panel per task
n = len(curves)
if n:
	fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0), dpi=130, squeeze=False)
	for ax, (t, runs) in zip(axes[0], curves.items()):
		ref = pd.read_csv(os.path.join(REF, f'{t}.csv'))
		rg = ref.groupby('step')['return']
		ax.plot(rg.mean().index, rg.mean().values, color='#52514e', lw=2, label='paper (5 seeds)')
		ax.fill_between(rg.mean().index, rg.mean() - rg.std(), rg.mean() + rg.std(), color='#52514e', alpha=0.15)
		if runs:
			df = pd.DataFrame(runs)
			ax.plot(df.index, df.mean(1), color='#2a78d6', lw=2, label='GPU port, MPS (5 seeds)')
			ax.fill_between(df.index, df.mean(1) - df.std(1), df.mean(1) + df.std(1), color='#2a78d6', alpha=0.2)
		ax.set_title(t, fontsize=9); ax.set_xlabel('env steps', fontsize=8); ax.tick_params(labelsize=7)
	axes[0][0].set_ylabel('evaluation return', fontsize=8); axes[0][0].legend(fontsize=7)
	fig.tight_layout(); fig.savefig(os.path.join(ROOT, 'docs', 'figures', 'mps_batches_learning.png')); plt.close(fig)
print('\n'.join(md))
