"""Tables + figures for the horizon sweep (results/sweep_h/<mode>/<task>_H<H>.json).

Writes docs/results_sweep_h.md and docs/figures/sweep_*.png. Full-run time is extrapolated from the
steady-state measurements with the original training-loop structure:
  T_full = seed_steps * env_ms                      (seeding: random actions, no planning)
         + (train_steps - seed_steps) * collect_ms  (plan + env per agent step)
         + train_steps * update_ms                  (one update per agent step overall)
         + n_eval * eval_episodes * L * collect_ms  (n_eval = train_steps*ar // eval_freq + 1)
(the planning-horizon warm-up schedule over the first 25k steps is ignored -> slight over-estimate).

  python bench/analyze_sweep.py

Pure post-processing (no GPU needed), run in the `tdmpc-warp` env (needs numpy + matplotlib). Inputs come
from bench/sweep_h.py (modes cpu / gpu / gpu_opt). Outputs: docs/results_sweep_h.md (per-task tables),
docs/figures/sweep_step_time_vs_H.png, sweep_vram_vs_H.png and sweep_speedup_H5.png; summarised in
docs/02_single_run_performance.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results', 'sweep_h')
DOCS = os.path.join(ROOT, 'docs'); FIG = os.path.join(DOCS, 'figures')
os.makedirs(FIG, exist_ok=True)
MODES = ['cpu', 'gpu', 'gpu_opt']
LABEL = {'cpu': 'CPU MuJoCo (original)', 'gpu': 'GPU port (faithful)', 'gpu_opt': 'GPU port (optimised)'}
COLOR = {'cpu': '#2a78d6', 'gpu': '#eb6834', 'gpu_opt': '#1baf7a'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEED_STEPS, EVAL_FREQ, EVAL_EPS = 5000, 20000, 10  # defaults of the original cfgs/default.yaml


def load():
	"""All sweep records keyed by (mode, task, H), parsed from the <task>_H<H>.json file names."""
	data = {}
	for m in MODES:
		for f in glob.glob(os.path.join(RES, m, '*.json')):
			d = json.load(open(f))
			task = os.path.basename(f).rsplit('_H', 1)[0]
			H = int(os.path.basename(f).rsplit('_H', 1)[1][:-5])
			data[(m, task, H)] = d
	return data


def full_run_hours(d):
	"""Extrapolated wall time (h) of a complete training run from one bench_step record (module doc).

	train_steps is in agent steps; eval_freq is in env steps (agent steps x action_repeat).
	"""
	L, ar, ts =d['episode_length'], d['action_repeat'], d['train_steps']
	n_eval = ts * ar // EVAL_FREQ + 1
	ms = SEED_STEPS * d['env_ms_synced'] + (ts - SEED_STEPS) * d['collect_ms_per_step'] + ts * d['update_ms'] \
		+ n_eval * EVAL_EPS * L * d['collect_ms_per_step']
	return ms / 1e3 / 3600


def style(ax):
	"""Shared minimal axis style (light grid, no top/right spines, muted ticks)."""
	ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
	for s in ('top', 'right'):
		ax.spines[s].set_visible(False)
	for s in ('left', 'bottom'):
		ax.spines[s].set_color(MUTED)
	ax.tick_params(colors=MUTED, labelsize=7)


def main():
	data = load()
	tasks = [l.strip() for l in open(os.path.join(ROOT, 'tasks.txt')) if l.strip()]
	Hs = sorted({k[2] for k in data})
	# a point is usable if it exists and succeeded (failed points are stored as {ok: false, oom: ...})
	ok = lambda k: k in data and data[k].get('ok', True) and 'step_ms' in data[k]
	md = ['# Horizon sweep: CPU MuJoCo baseline vs GPU port', '', '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
		  'Steady-state cost per agent step (plan + env step + one update) and peak process VRAM, single run alone on the RTX 3090.',
		  'Modes: **cpu** = original code (CPU MuJoCo 2.1 / dm_control, agent on GPU, torch 2.4.1); **gpu** = faithful mujoco_warp port '
		  '(original eager agent, torch 2.11); **gpu_opt** = GPU port + CUDA-graph agent + host pacing.',
		  'Speed-up = cpu time / mode time. VRAM = whole process as reported by nvidia-smi (CUDA context + buffers + allocator cache).', '']
	for task in tasks:
		md += [f'## {task}', '', '| H | cpu ms | gpu ms | gpu_opt ms | speed-up gpu | speed-up gpu_opt | VRAM cpu MiB | VRAM gpu MiB | VRAM gpu_opt MiB | full run cpu h | full run gpu_opt h |',
			   '|---|---|---|---|---|---|---|---|---|---|---|']
		for H in Hs:
			row = [str(H)]
			c = data.get(('cpu', task, H)) if ok(('cpu', task, H)) else None
			for m in MODES:
				row.append(f"{data[(m, task, H)]['step_ms']:.1f}" if ok((m, task, H)) else ('OOM' if (m, task, H) in data and data[(m, task, H)].get('oom') else '–'))
			for m in ('gpu', 'gpu_opt'):
				row.append(f"{c['step_ms'] / data[(m, task, H)]['step_ms']:.2f}x" if c and ok((m, task, H)) else '–')
			for m in MODES:
				row.append(f"{data[(m, task, H)]['process_vram_mib']:.0f}" if ok((m, task, H)) else '–')
			for m in ('cpu', 'gpu_opt'):
				row.append(f"{full_run_hours(data[(m, task, H)]):.1f}" if ok((m, task, H)) else '–')
			md.append('| ' + ' | '.join(row) + ' |')
		md.append('')
	open(os.path.join(DOCS, 'results_sweep_h.md'), 'w').write('\n'.join(md))

	# small multiples: step time vs H, VRAM vs H
	for key, ylabel, fname in (('step_ms', 'ms per agent step', 'sweep_step_time_vs_H.png'),
							   ('process_vram_mib', 'process VRAM (MiB)', 'sweep_vram_vs_H.png')):
		fig, axes = plt.subplots(4, 6, figsize=(16, 10.5), dpi=130)
		for ax, task in zip(axes.flat, tasks):
			for m in MODES:
				xs = [H for H in Hs if ok((m, task, H))]
				ys = [data[(m, task, H)][key] for H in xs]
				if xs:
					ax.plot(xs, ys, color=COLOR[m], lw=2, marker='o', ms=3.5, label=LABEL[m])
			ax.set_title(task, fontsize=8.5, color=INK)
			style(ax)
			ax.set_ylim(bottom=0)
		for ax in axes.flat[len(tasks):]:
			ax.axis('off')
		handles, labels = axes.flat[0].get_legend_handles_labels()
		axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=10)
		fig.supxlabel('planning / unroll horizon H', color=INK); fig.supylabel(ylabel, color=INK)
		fig.suptitle(f'TD-MPC {ylabel} vs horizon (RTX 3090, single run)', color=INK)
		fig.tight_layout()
		fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

	# speed-up bars at default H=5
	H0 = 5 if 5 in Hs else Hs[0]
	rows = [(t, data[('cpu', t, H0)]['step_ms'] / data[('gpu', t, H0)]['step_ms'] if ok(('gpu', t, H0)) and ok(('cpu', t, H0)) else np.nan,
			 data[('cpu', t, H0)]['step_ms'] / data[('gpu_opt', t, H0)]['step_ms'] if ok(('gpu_opt', t, H0)) and ok(('cpu', t, H0)) else np.nan)
			for t in tasks]
	fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
	y = np.arange(len(rows))
	ax.barh(y + 0.2, [r[1] for r in rows], height=0.38, color=COLOR['gpu'], label=LABEL['gpu'])
	ax.barh(y - 0.2, [r[2] for r in rows], height=0.38, color=COLOR['gpu_opt'], label=LABEL['gpu_opt'])
	for yi, r in zip(y, rows):
		if not np.isnan(r[2]):
			ax.text(r[2] + 0.05, yi - 0.2, f'{r[2]:.1f}x', va='center', fontsize=7, color=MUTED)
	ax.axvline(1, color=MUTED, lw=1, ls='--')
	ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
	ax.set_xlabel('speed-up vs CPU MuJoCo baseline (x, higher is better)', color=INK)
	ax.set_title(f'Single-run speed-up at H={H0}', color=INK)
	style(ax); ax.legend(frameon=False, fontsize=8, loc='lower right')
	fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sweep_speedup_H5.png')); plt.close(fig)
	print('wrote', os.path.join(DOCS, 'results_sweep_h.md'))


if __name__ == '__main__':
	main()
