"""Precision / checkpointing study tables + figures (single seed, vectorised agent code path).

Reads results/sweep_h/vec_<variant>/<task>_H<H>.json -> docs/results_precision.md, docs/figures/precision_*.png
Speed-up and VRAM change are relative to vec_fp32 (identical code, fp32 numerics = original maths).
Variants: TF32 matmul, bf16 autocast, fp16 autocast + loss scaling, bf16 autocast + bf16 replay buffer,
fp32 + activation checkpointing (exact); one representative task per domain, H = 2..30.

  python bench/analyze_precision.py

Pure post-processing (no GPU needed), `tdmpc-warp` env. Inputs are produced by
`bench/sweep_h.py --modes vec_fp32 vec_tf32 vec_bf16 vec_fp16 vec_bf16_buf vec_ckpt` (bench/pipeline_phase3.sh).
Reported in docs/07_precision.md (7.1); the multi-seed precision numbers of 7.2 come from
results/precision_vec/ (bench/bench_step.py runs in the pipeline_phase4*/final scripts) and are not
processed here.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results', 'sweep_h')
DOCS = os.path.join(ROOT, 'docs'); FIG = os.path.join(DOCS, 'figures')
HS = [2, 5, 10, 15, 20, 25, 30]
REPS = ['acrobot-swingup', 'cartpole-swingup', 'cheetah-run', 'cup-catch', 'dog-run', 'finger-spin', 'fish-swim',
		'hopper-hop', 'humanoid-run', 'quadruped-run', 'reacher-easy', 'walker-walk']
VARS = ['vec_fp32', 'vec_tf32', 'vec_bf16', 'vec_fp16', 'vec_bf16_buf', 'vec_ckpt']
LAB = {'vec_fp32': 'fp32 (reference)', 'vec_tf32': 'TF32 matmul', 'vec_bf16': 'bf16 autocast', 'vec_fp16': 'fp16 autocast + loss scaling',
	   'vec_bf16_buf': 'bf16 autocast + bf16 replay buffer', 'vec_ckpt': 'fp32 + activation checkpointing (exact)'}
COL = {'vec_fp32': '#2a78d6', 'vec_tf32': '#eb6834', 'vec_bf16': '#1baf7a', 'vec_fp16': '#eda100', 'vec_bf16_buf': '#e87ba4', 'vec_ckpt': '#4a3aa7'}


def J(v, t, H):
	"""Sweep record of variant v, task t, horizon H; None if missing or failed (no step_ms)."""
	p =os.path.join(R, v, f'{t}_H{H}.json')
	if not os.path.exists(p):
		return None
	d = json.load(open(p))
	return d if 'step_ms' in d else None


def main():
	"""Per-domain tables, averages over the 12 domains, and step-time / VRAM-vs-H figures."""
	md = ['# Reduced precision (bf16 / fp16 / TF32) and activation checkpointing: speed and peak VRAM', '', '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
		  'Single seed, vectorised-agent code path for every variant (so fp32 is an exact reference), GPU env, CUDA graphs, '
		  'host pacing. speed-up = fp32 time / variant time; dVRAM = variant process VRAM - fp32 process VRAM (negative = saving). '
		  'Tasks of one domain have identical cost (same networks / buffers), one representative per domain is measured.', '']
	for t in REPS:
		md += [f'## {t}', '', '| H | fp32 ms | ' + ' | '.join(f'{LAB[v]} speed-up' for v in VARS[1:]) + ' | fp32 VRAM | ' +
			   ' | '.join(f'{LAB[v]} dVRAM' for v in VARS[1:]) + ' |', '|' + '---|' * (2 * len(VARS) + 1)]
		for H in HS:
			f = J('vec_fp32', t, H)
			if not f:
				continue
			row = [str(H), f'{f["step_ms"]:.1f}']
			for v in VARS[1:]:
				d = J(v, t, H); row.append(f'{f["step_ms"] / d["step_ms"]:.2f}x' if d else '–')
			row.append(f'{f["process_vram_mib"]:.0f}')
			for v in VARS[1:]:
				d = J(v, t, H); row.append(f'{d["process_vram_mib"] - f["process_vram_mib"]:+.0f}' if d else '–')
			md.append('| ' + ' | '.join(row) + ' |')
		md.append('')
	# averages
	md += ['## Averages over the 12 domains', '', '| variant | mean speed-up H=5 | mean speed-up H=30 | mean dVRAM H=5 (MiB) | mean dVRAM H=30 (MiB) |', '|---|---|---|---|---|']
	for v in VARS[1:]:
		vals = {}
		for H in (5, 30):
			sp = [J('vec_fp32', t, H)['step_ms'] / J(v, t, H)['step_ms'] for t in REPS if J(v, t, H) and J('vec_fp32', t, H)]
			dv = [J(v, t, H)['process_vram_mib'] - J('vec_fp32', t, H)['process_vram_mib'] for t in REPS if J(v, t, H) and J('vec_fp32', t, H)]
			vals[H] = (np.mean(sp) if sp else np.nan, np.mean(dv) if dv else np.nan)
		md.append(f'| {LAB[v]} | {vals[5][0]:.2f}x | {vals[30][0]:.2f}x | {vals[5][1]:+.0f} | {vals[30][1]:+.0f} |')
	md.append('')
	open(os.path.join(DOCS, 'results_precision.md'), 'w').write('\n'.join(md))

	for key, ylab, fname in (('step_ms', 'ms per agent step', 'precision_step_time_vs_H.png'), ('process_vram_mib', 'process VRAM (MiB)', 'precision_vram_vs_H.png')):
		fig, axes = plt.subplots(2, 6, figsize=(17, 6.5), dpi=130)
		for ax, t in zip(axes.flat, REPS):
			for v in VARS:
				xs = [H for H in HS if J(v, t, H)]
				if xs:
					ax.plot(xs, [J(v, t, H)[key] for H in xs], color=COL[v], lw=2, marker='o', ms=3, label=LAB[v])
			ax.set_title(t, fontsize=8.5); ax.set_ylim(bottom=0); ax.grid(True, color='#e4e3df', lw=0.8)
			for s in ('top', 'right'):
				ax.spines[s].set_visible(False)
			ax.tick_params(labelsize=7)
		h, l = axes.flat[0].get_legend_handles_labels()
		fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=8)
		fig.supxlabel('H', y=0.06); fig.supylabel(ylab)
		fig.suptitle(f'{ylab} vs H: fp32 vs reduced precision vs activation checkpointing (single seed)')
		fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
	print('wrote docs/results_precision.md')


if __name__ == '__main__':
	main()
