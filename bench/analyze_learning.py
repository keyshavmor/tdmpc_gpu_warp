"""Learning-mechanism equivalence: eval returns of every variant vs the paper's reference curves.

Reference: 5 seeds of the original code, ../tdmpc/results/<task>.csv; plus A/B tests between variants.
Writes docs/results_learning.md and docs/figures/learning_*.png

Inputs: results/learning/<task>/state/<variant>/<seed>/eval.log written by the runs of
bench/learning_runs.py (variant = exp_name: cpu_original, gpu_faithful, vec_fp32, vec_fp32_opt, vec_bf16,
vec_fp16, vec_tf32 - variants without logs are skipped). Every eval step is compared with a Welch test
(normal approximation; with 3-5 seeds only large differences are detectable): each variant vs the
reference, and each exact / reduced-precision variant vs vec_fp32 (same code path).

  python bench/analyze_learning.py

Pure post-processing (no GPU needed), `tdmpc-warp` env (needs pandas, matplotlib). Reported in
docs/06_optimisations.md (learning curves of the exact optimisations) and docs/07_precision.md (7.4).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LR = os.path.join(ROOT, 'results', 'learning')
REF = os.path.join(ROOT, '..', 'tdmpc', 'results')
DOCS = os.path.join(ROOT, 'docs'); FIG = os.path.join(DOCS, 'figures')
VARIANTS = [('cpu_original', 'CPU MuJoCo, original code (this machine)', '#2a78d6'),
			('gpu_faithful', 'GPU port, original agent', '#eb6834'),
			('graph_fixed', 'GPU port, CUDA-graph agent, separate processes (recommended)', '#c0392b'),
			('vec_fp32_fixed', 'GPU port, vectorised CUDA-graph agent', '#0b6e99'),
			('vec_fp32', 'vectorised agent, before PER fix (exact maths)', '#1baf7a'),
			('vec_fp32_opt', '+ batched update + sensor skipping, before PER fix', '#4a3aa7'),
			('vec_bf16', 'bf16 autocast, before PER fix', '#eda100'), ('vec_fp16', 'fp16 autocast, before PER fix', '#e87ba4'),
			('vec_tf32', 'TF32', '#008300')]
# "before PER fix": runs made while the CUDA-graph / vectorised replay buffers sampled without replacement before the
# buffer was full (the original samples with replacement); fixed and re-run as graph_fixed / vec_fp32_fixed (docs/06, 6.4)


def load(task, variant):
	"""{seed: Series(env_step -> episode_reward)} from every non-empty eval.log of one variant."""
	runs = {}
	for f in glob.glob(os.path.join(LR, task, 'state', variant, '*', 'eval.log')):
		d = pd.read_csv(f)
		if len(d):
			runs[os.path.basename(os.path.dirname(f))] = d.set_index('env_step')['episode_reward']
	return runs


def welch_p(a, b):
	"""Two-sided p-value of Welch's t statistic with a normal approximation (NaN if < 2 samples)."""
	a, b =np.asarray(a, float), np.asarray(b, float)
	if len(a) < 2 or len(b) < 2:
		return np.nan
	va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
	t = (a.mean() - b.mean()) / np.sqrt(va + vb + 1e-12)
	from math import erf, sqrt
	return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))  # normal approximation


def main():
	"""Per task: return table vs reference, A/B table vs vec_fp32, and a learning-curve figure."""
	tasks = sorted({os.path.basename(p) for p in glob.glob(os.path.join(LR, '*')) if os.path.isdir(os.path.join(p, 'state'))})
	md = ['# Learning-mechanism equivalence (eval return vs env steps)', '', '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
		  'Reference = 5 seeds of the original TD-MPC code from the paper (`tdmpc/results/<task>.csv`). '
		  'Each cell: mean ± sd over seeds (n). p = Welch test (normal approx.) of the variant vs the reference '
		  '(or vs `vec_fp32` for the A/B rows) at that step; p > 0.05 = no detectable difference. '
		  'Columns marked "before PER fix" were run while the CUDA-graph / vectorised replay buffers drew batches without '
		  'replacement before the buffer was full (the original draws with replacement); the first two GPU-agent columns '
		  'are the re-runs after the fix (docs/06_optimisations.md, section 6.4). In the A/B rows, the optimisation and precision '
		  'variants share the sampling of `vec_fp32`; the row of the vectorised re-run shows the effect of the fix.', '']
	for task in tasks:
		ref = pd.read_csv(os.path.join(REF, f'{task}.csv'))
		refg = {s: g['return'].values for s, g in ref.groupby('step')}
		data = {v: load(task, v) for v, _, _ in VARIANTS}
		steps = sorted({s for v in data.values() for r in v.values() for s in r.index})
		steps = [s for s in steps if s in refg]  # only eval steps that also exist in the reference curves
		md += [f'## {task}', '', '| env step | reference (paper) | ' + ' | '.join(lab for v, lab, _ in VARIANTS if data[v]) + ' |',
			   '|' + '---|' * (2 + sum(1 for v, _, _ in VARIANTS if data[v]))]
		for s in steps:
			row = [f'{s // 1000}k', f'{refg[s].mean():.0f} ± {refg[s].std():.0f} (5)']
			for v, _, _ in VARIANTS:
				if not data[v]:
					continue
				vals = [r[s] for r in data[v].values() if s in r.index]
				if not vals:
					row.append('–'); continue
				p = welch_p(vals, refg[s])
				row.append(f'{np.mean(vals):.0f} ± {np.std(vals):.0f} ({len(vals)})' + (f', p={p:.2f}' if not np.isnan(p) else ''))
			md.append('| ' + ' | '.join(row) + ' |')
		# A/B vs vec_fp32
		if data['vec_fp32']:
			md += ['', f'A/B vs `vec_fp32` (same code path, exact maths), p-values per eval step:', '',
				   '| variant | ' + ' | '.join(f'{s // 1000}k' for s in steps) + ' | mean over steps (variant - vec_fp32) |', '|' + '---|' * (len(steps) + 2)]
			for v, lab, _ in VARIANTS:
				if v in ('vec_fp32', 'cpu_original', 'gpu_faithful', 'graph_fixed') or not data[v]:
					continue
				cells, diffs = [], []
				for s in steps:
					a = [r[s] for r in data[v].values() if s in r.index]; b = [r[s] for r in data['vec_fp32'].values() if s in r.index]
					cells.append(f'{welch_p(a, b):.2f}' if a and b else '–')
					if a and b:
						diffs.append(np.mean(a) - np.mean(b))
				md.append(f'| {lab} | ' + ' | '.join(cells) + f' | {np.mean(diffs):+.0f} |' if diffs else f'| {lab} | ' + ' | '.join(cells) + ' | – |')
		md.append('')
		# figure
		fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
		rs = sorted(refg)
		m = np.array([refg[s].mean() for s in rs]); sd = np.array([refg[s].std() for s in rs])
		lim = max(steps) if steps else max(rs)
		sel = [i for i, s in enumerate(rs) if s <= lim]
		ax.fill_between(np.array(rs)[sel], (m - sd)[sel], (m + sd)[sel], color='#c3c2b7', alpha=0.5, lw=0)
		ax.plot(np.array(rs)[sel], m[sel], color='#52514e', lw=2, label='reference: original code, paper (5 seeds)')
		for v, lab, col in VARIANTS:
			if not data[v]:
				continue
			ss = sorted({s for r in data[v].values() for s in r.index})
			mm = [np.mean([r[s] for r in data[v].values() if s in r.index]) for s in ss]
			ax.plot(ss, mm, color=col, lw=2, marker='o', ms=3, label=f'{lab} ({len(data[v])} seed{"s" if len(data[v]) > 1 else ""})')
		ax.set_xlabel('environment steps'); ax.set_ylabel('evaluation return'); ax.set_title(f'{task}: learning curves')
		ax.grid(True, color='#e4e3df', lw=0.8)
		for sp in ('top', 'right'):
			ax.spines[sp].set_visible(False)
		ax.legend(frameon=False, fontsize=7, loc='lower right')
		fig.tight_layout(); fig.savefig(os.path.join(FIG, f'learning_{task}.png')); plt.close(fig)
	open(os.path.join(DOCS, 'results_learning.md'), 'w').write('\n'.join(md))
	print('wrote docs/results_learning.md')


if __name__ == '__main__':
	main()
