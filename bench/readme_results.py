"""Results table and learning-curve figure of the 15-task x 5-seed baseline for the main README.

Reads the 5-seed MPS batches (results/mps_batches/<label>/<task>/state/<label>/<seed>/{eval,train}.log), the paper's
reference curves of the original TD-MPC (../tdmpc/results/<task>.csv, 5 seeds) and, where available, the original
code run on this machine (results/learning_full/<task>/state/cpu_original/<seed>/eval.log). Writes
  * docs/figures/baseline_15tasks.png: evaluation return vs env steps per task, mean +- 1 sample standard deviation
    over the 5 seeds (shaded) for the GPU port, the paper and the original code on this machine;
  * the results table (final return at 500k and run time per run, mean +- 1 sample sd over seeds 1-5, paper at 500k)
    into README.md between the markers <!-- results-table:start --> and <!-- results-table:end -->.
A task can be taken from another batch label with --use task=label (e.g. a rerun after a fix); the replaced run is
then shown as a dashed line in the figure and named in a table footnote.

Environment: tdmpc-warp (numpy, pandas, matplotlib).
Usage: python bench/readme_results.py [--label baseline_mps] [--use hopper-hop=baseline_mps_capfix]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, csv, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TASKS = sorted(t.strip() for t in open(os.path.join(ROOT, 'tasks.txt')) if t.strip() and not t.startswith(('humanoid', 'dog')))
p = argparse.ArgumentParser()
p.add_argument('--label', default='baseline_mps')
p.add_argument('--use', nargs='*', default=[], help='task=label: take this task from another batch label')
a = p.parse_args()
use = dict(x.split('=', 1) for x in a.use)
C_GPU, C_PAPER, C_CPU = '#2a78d6', '#52514e', '#d9822b'


def runs(task, label):
	"""{seed: (eval Series indexed by env_step, run time in h)} of a batch; only seeds that reached 500k."""
	out = {}
	for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'mps_batches', label, task, 'state', label, '*', 'eval.log'))):
		d = os.path.dirname(f)
		ev = pd.read_csv(f).set_index('env_step')['episode_reward']
		tr = list(csv.DictReader(open(os.path.join(d, 'train.log'))))
		if len(ev) and ev.index.max() >= 500000 and tr:
			out[os.path.basename(d)] = (ev, float(tr[-1]['total_time']) / 3600)
	return out


def curves(series):
	"""DataFrame (env_step x seed) of evaluation returns."""
	return pd.DataFrame({k: v for k, v in series.items()})


rows, panels = [], []
for t in TASKS:
	label = use.get(t, a.label)
	r = runs(t, label)
	ref = pd.read_csv(os.path.join(ROOT, '..', 'tdmpc', 'results', f'{t}.csv'))
	cpu = {os.path.basename(os.path.dirname(f)): pd.read_csv(f).set_index('env_step')['episode_reward']
		   for f in glob.glob(os.path.join(ROOT, 'results', 'learning_full', t, 'state', 'cpu_original', '*', 'eval.log'))}
	cpu = {k: v for k, v in cpu.items() if len(v) and v.index.max() >= 500000}
	old = runs(t, a.label) if t in use else {}
	panels.append((t, r, ref, cpu, old))
	if len(r) == 5:
		fin = np.array([v[0].loc[500000] for v in r.values()])
		rt = np.array([v[1] for v in r.values()])
		pf = ref[ref.step == 500000]['return'].values
		rows.append(f"| {t}{' *' if t in use else ''} | **{fin.mean():.1f} ± {fin.std(ddof=1):.1f}** | {pf.mean():.1f} ± {pf.std(ddof=1):.1f} "
					f"| {rt.mean():.2f} ± {rt.std(ddof=1):.2f} |")
	else:
		rows.append(f"| {t} | pending | - | - |")

# figure: 3 x 5 panels, mean +- 1 sd shading
fig, axes = plt.subplots(3, 5, figsize=(17, 9.6), dpi=120)
for ax, (t, r, ref, cpu, old) in zip(axes.flat, panels):
	rg = ref.groupby('step')['return']
	m, s = rg.mean(), rg.std()
	ax.fill_between(m.index, m - s, m + s, color=C_PAPER, alpha=0.15, lw=0)
	ax.plot(m.index, m.values, color=C_PAPER, lw=1.8)
	if len(cpu) >= 2:
		df = curves(cpu)
		ax.fill_between(df.index, df.mean(axis=1) - df.std(axis=1), df.mean(axis=1) + df.std(axis=1), color=C_CPU, alpha=0.15, lw=0)
		ax.plot(df.index, df.mean(axis=1), color=C_CPU, lw=1.8)
	if old:
		df = curves({k: v[0] for k, v in old.items()})
		ax.plot(df.index, df.mean(axis=1), color=C_GPU, lw=1.2, ls='--')
	if r:
		df = curves({k: v[0] for k, v in r.items()})
		ax.fill_between(df.index, df.mean(axis=1) - df.std(axis=1), df.mean(axis=1) + df.std(axis=1), color=C_GPU, alpha=0.2, lw=0)
		ax.plot(df.index, df.mean(axis=1), color=C_GPU, lw=2)
	ax.set_title(t, fontsize=10)
	ax.set_xticks([0, 250000, 500000]); ax.set_xticklabels(['0', '250k', '500k'])
	ax.tick_params(labelsize=8); ax.grid(alpha=0.25, lw=0.5)
	for sp in ('top', 'right'):
		ax.spines[sp].set_visible(False)
for ax in axes[:, 0]:
	ax.set_ylabel('evaluation return', fontsize=9)
for ax in axes[-1]:
	ax.set_xlabel('environment steps', fontsize=9)
handles = [plt.Line2D([], [], color=C_GPU, lw=2, label='GPU port (this repository), 5 seeds'),
		   plt.Line2D([], [], color=C_PAPER, lw=1.8, label='paper: original TD-MPC, 5 seeds'),
		   plt.Line2D([], [], color=C_CPU, lw=1.8, label='original code run on this machine, 5 seeds')]
if use:
	handles.append(plt.Line2D([], [], color=C_GPU, lw=1.2, ls='--', label='GPU port before the fix (mean)'))
fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=9, frameon=False)
fig.text(0.5, 0.005, 'Lines: mean over seeds; shaded: ±1 standard deviation across seeds.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(os.path.join(ROOT, 'docs', 'figures', 'baseline_15tasks.png'))

# table into the README
table = ['| task | final return at 500k (GPU port) | paper at 500k | run time per run (h) |', '|---|---|---|---|'] + rows
if use:
	table += ['', '\\* ' + '; '.join(f'{t}: rerun with label `{l}`' for t, l in use.items()) + ' (see the notes below the table).']
readme = os.path.join(ROOT, 'README.md')
s = open(readme).read()
i, j = s.index('<!-- results-table:start -->'), s.index('<!-- results-table:end -->')
s = s[:i] + '<!-- results-table:start -->\n' + '\n'.join(table) + '\n' + s[j:]
open(readme, 'w').write(s)
print('\n'.join(table))
