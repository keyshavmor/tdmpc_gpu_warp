"""Memory tables: OOM horizon, seeds per RTX 3090 vs H, mixed-horizon capacity, packing tests.

Q2 / Q3 tables: OOM horizon of a single run, seeds per RTX 3090 vs H (separate processes and
vectorised single process), mixed-horizon capacity. Writes docs/results_memory.md (+ figure).

Model (validated by the measurements): process VRAM(H) = a + b*H. For runs with horizons H_1..H_N
running together, sum_i (a + b*H_i) <= free  <=>  N <= free / (a + b*mean(H)), i.e. the capacity of a
mixed-horizon set depends only on its mean horizon (and on the number of separate processes).
Vectorised process with S seeds: VRAM = V1(H) + (S - 1) * m(H), V1 and m measured at H = 5 and 30 and
interpolated linearly in H.

VRAM accounting: free VRAM = 24576 MiB minus the memory used by the desktop with no runs active
(results/gpu_baseline_memory.txt, 963 MiB on this machine -> 23613 MiB); per-run VRAM is the
whole-process nvidia-smi figure from bench/bench_step.py.

  python bench/analyze_memory.py

Pure post-processing (no GPU needed), `tdmpc-warp` env. Inputs: results/sweep_h/ (bench/sweep_h.py),
results/oom_horizon/ (bench/oom_horizon.py), results/vec_memory/ (bench/vec_memory.py),
results/packing/ (bench/packing_study.py). Outputs: docs/results_memory.md and
docs/figures/seeds_per_gpu_vs_H.png; the numbers are used in docs/04_multi_run_capacity.md (4.4, 4.5).

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
R = os.path.join(ROOT, 'results')
DOCS = os.path.join(ROOT, 'docs'); FIG = os.path.join(DOCS, 'figures')
HS = [2, 5, 10, 15, 20, 25, 30]
# representative task per domain (tasks of one domain share networks and buffer sizes -> same memory)
DOMAIN_REP ={'acrobot': 'acrobot-swingup', 'cartpole': 'cartpole-swingup', 'cheetah': 'cheetah-run', 'cup': 'cup-catch',
			  'dog': 'dog-run', 'finger': 'finger-spin', 'fish': 'fish-swim', 'hopper': 'hopper-hop', 'humanoid': 'humanoid-run',
			  'quadruped': 'quadruped-run', 'reacher': 'reacher-easy', 'walker': 'walker-walk'}
COLOR = {'cpu': '#2a78d6', 'gpu': '#eb6834', 'gpu_opt': '#1baf7a', 'vec': '#eda100'}


def J(p):
	"""Load a JSON file, or None if it does not exist."""
	return json.load(open(p)) if os.path.exists(p) else None


def free_mib():
	"""VRAM available to runs: 24 GiB minus the idle desktop usage recorded in gpu_baseline_memory.txt."""
	base =float(open(os.path.join(R, 'gpu_baseline_memory.txt')).read().split()[0])
	return 24576 - base


def vec_mem(rep, H, ck, free):
	"""Vectorised memory record; if the S=8 probe went OOM, derive the marginal per-seed VRAM from the
	packing bisection (S_fit fits, S_oom OOMs): m = (free - V1) / (mean(S_fit, S_oom) - 1)."""
	v = J(os.path.join(R, 'vec_memory', f'{rep}_H{H}' + ('_ckpt' if ck else '') + '.json'))
	if v and 'marginal_mib_per_seed' not in v and v.get('S1', {}).get('ok'):
		pk = J(os.path.join(R, 'packing', f'vec_{rep}_H{H}' + ('_ckpt_unroll-true' if ck else '') + '.json'))
		if pk and pk.get('S_min_oom'):
			v = dict(v); v['marginal_mib_per_seed'] = (free - v['S1']['vram']) / ((pk['S_max_fit'] + pk['S_min_oom']) / 2 - 1)
			v['marginal_from_packing'] = True
	return v


def sweep_fit(mode, task):
	"""Least-squares fit VRAM(H) = a + b*H over the sweep points; returns (a, b, {H: VRAM}) or None."""
	pts =[(H, J(os.path.join(R, 'sweep_h', mode, f'{task}_H{H}.json'))) for H in HS]
	pts = [(H, d['process_vram_mib']) for H, d in pts if d and 'process_vram_mib' in d]
	if len(pts) < 3:
		return None
	b, a = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
	return a, b, dict(pts)


def main():
	"""Assemble all memory tables into docs/results_memory.md and draw the seeds-per-GPU figure."""
	free = free_mib()
	tasks = [l.strip() for l in open(os.path.join(ROOT, 'tasks.txt')) if l.strip()]
	md = ['# Memory: OOM horizon, seeds per GPU, mixed horizons (RTX 3090, 24 GiB)', '', '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
		  f'Free VRAM for runs: {free:.0f} MiB (24576 MiB minus {24576 - free:.0f} MiB used by the desktop).', '']

	# ---------------- Q2a: single-run OOM horizon
	md += ['## Single run: OOM horizon', '',
		   '`H limit` = largest horizon the algorithm can run at all: H <= L - 1 (L = episode length in agent steps; each '
		   'episode then still has at least one valid start index, and PER batches are drawn with replacement until the '
		   'buffer is full). Fit = VRAM(H) = a + b*H from H = 2..30; boundaries confirmed with real runs (`fits <=` / '
		   '`OOM >=`). The probes were capped at `largest probed H` = L - ceil(batch / (seed_steps/L + 1)); where the fit '
		   'predicts OOM far beyond L - 1, no OOM is possible at any valid horizon. (The cap came from an earlier, '
		   'incorrect assumption that the first batch is drawn without replacement; see docs/06, section 6.4.)', '',
		   '| domain (tasks) | mode | a (MiB) | b (MiB per H) | H limit (L-1) | largest probed H | predicted OOM H | measured |',
		   '|---|---|---|---|---|---|---|---|']
	for dom, rep in DOMAIN_REP.items():
		names = ', '.join(t for t in tasks if t.split('-')[0] == dom)
		for mode in ('cpu', 'gpu', 'gpu_opt', 'vec1_ckpt'):
			d = J(os.path.join(R, 'oom_horizon', mode, f'{rep}.json'))
			if not d:
				continue
			ref = J(os.path.join(R, 'sweep_h', 'gpu', f'{rep}_H5.json'))
			Hlim = ref['episode_length'] - 1  # every episode keeps >= 1 valid start index
			if d['result'] == 'fits_at_H_max':
				meas = f'fits at {d["H_max_valid"]}; OOM predicted only at {d["H_oom_predicted"]:.0f} > {Hlim} - no OOM possible'
			elif d['result'] == 'bracketed':
				meas = f'fits <= {d["H_fit_max"]}, OOM >= {d["H_oom_min"]}'
			else:
				meas = d['result']
			lab = {'cpu': 'CPU baseline', 'gpu': 'GPU faithful', 'gpu_opt': 'GPU optimised', 'vec1_ckpt': 'GPU + ckpt_unroll'}[mode]
			md.append(f'| {dom} ({names}) | {lab} | {d["intercept_mib"]:.0f} | {d["slope_mib_per_H"]:.1f} | {Hlim} | {d["H_max_valid"]} | '
					  f'{d["H_oom_predicted"]:.0f} | {meas} |')
	md.append('')

	# ---------------- Q2b: seeds per GPU vs H
	md += ['## Seeds per RTX 3090 vs horizon', '',
		   '**Separate processes** (one run per process): N = floor(free / VRAM_process(H)), VRAM measured in the horizon sweep. '
		   '**Vectorised** (all seeds in ONE `train_vec.py` process): S = 1 + floor((free - V1(H)) / m(H)) with V1 = one-seed process '
		   'VRAM and m = marginal VRAM per extra seed (measured with S = 1 and S = 8, linear in H between H = 5 and 30). '
		   'Empirical packing tests (next section) confirm these numbers.', '']
	header = '| task | mode | ' + ' | '.join(f'H={H}' for H in HS) + ' |'
	md += [header, '|' + '---|' * (len(HS) + 2)]
	vec_rows = {}
	for task in tasks:
		dom = task.split('-')[0]
		rep = DOMAIN_REP[dom]
		for mode, lab in (('cpu', 'CPU baseline, processes'), ('gpu', 'GPU faithful, processes'), ('gpu_opt', 'GPU optimised, processes')):
			cells = []
			for H in HS:
				d = J(os.path.join(R, 'sweep_h', mode, f'{task}_H{H}.json'))
				cells.append(str(int(free // d['process_vram_mib'])) if d and 'process_vram_mib' in d else '–')
			md.append(f'| {task} | {lab} | ' + ' | '.join(cells) + ' |')
		for ck, lab in ((False, 'GPU vectorised, 1 process'), (True, 'GPU vectorised + ckpt_unroll')):
			v5, v30 = vec_mem(rep, 5, ck, free), vec_mem(rep, 30, ck, free)
			if not (v5 and v30 and 'marginal_mib_per_seed' in v5 and 'marginal_mib_per_seed' in v30):
				continue
			cells = []
			for H in HS:
				# linear interpolation (extrapolation for H = 2) between the H = 5 and H = 30 measurements
				w = (H - 5) / 25
				V1 = (1 - w) * v5['S1']['vram'] + w * v30['S1']['vram']
				m = (1 - w) * v5['marginal_mib_per_seed'] + w * v30['marginal_mib_per_seed']
				cells.append(str(int(1 + (free - V1) // m)))
			vec_rows[(task, ck)] = cells
			md.append(f'| {task} | {lab} | ' + ' | '.join(cells) + ' |')
	md.append('')

	# ---------------- Q3: mixed horizons
	md += ['## Mixed horizons running together', '',
		   'Separate processes with horizons H_1..H_N fit iff sum_i (a + b*H_i) <= free, i.e. N_max = floor(free / (a + b*mean H)). '
		   'Columns: mean horizon of the mix (e.g. 17.5 = an even mix of H = 5, 10, 15, 20, 25, 30; 16 = H = 2..30 as in the sweep).', '',
		   '| task | mode | a | b | mean H = 5 | 10 | 16 | 17.5 | 30 |', '|---|---|---|---|---|---|---|---|---|']
	for task in tasks:
		for mode, lab in (('cpu', 'CPU baseline'), ('gpu_opt', 'GPU optimised')):
			f = sweep_fit(mode, task)
			if not f:
				continue
			a, b, _ = f
			md.append(f'| {task} | {lab} | {a:.0f} | {b:.1f} | ' + ' | '.join(str(int(free // (a + b * mh))) for mh in (5, 10, 16, 17.5, 30)) + ' |')
	md.append('')

	# ---------------- Q2: task x horizon settings x seeds
	md += ['## Seeds x horizon settings: k seeds at EACH of H = 2, 5, 10, 15, 20, 25, 30 (7 settings, 7k runs)', '',
		   'Largest k such that all 7k runs fit in VRAM at the same time. (a) one process per run; (b) one vectorised process '
		   'per horizon setting holding its k seeds (V1(H) + (k-1) m(H) each; m measured at H = 5 and 30, linear in H); '
		   '(c) as (b) with activation checkpointing. **No-slowdown limit for any of these sets: 1 run in total** (sections above / chapter 4).', '',
		   '| task | (a) processes: k | total runs | (b) vectorised: k | total runs | (c) vectorised + ckpt: k | total runs |', '|---|---|---|---|---|---|---|']
	SET = [2, 5, 10, 15, 20, 25, 30]
	for task in tasks:
		rep = DOMAIN_REP[task.split('-')[0]]
		vp = [J(os.path.join(R, 'sweep_h', 'gpu_opt', f'{task}_H{H}.json')) for H in SET]
		ka = int(free // sum(d['process_vram_mib'] for d in vp)) if all(vp) else None
		cells = [str(ka) if ka is not None else '–', str(7 * ka) if ka is not None else '–']
		for ck in (False, True):
			v5, v30 = vec_mem(rep, 5, ck, free), vec_mem(rep, 30, ck, free)
			if not (v5 and v30 and 'marginal_mib_per_seed' in v5 and 'marginal_mib_per_seed' in v30):
				cells += ['–', '–']; continue
			V1 = lambda H: v5['S1']['vram'] + (H - 5) / 25 * (v30['S1']['vram'] - v5['S1']['vram'])
			m = lambda H: v5['marginal_mib_per_seed'] + (H - 5) / 25 * (v30['marginal_mib_per_seed'] - v5['marginal_mib_per_seed'])
			sV1, sm = sum(V1(H) for H in SET), sum(m(H) for H in SET)
			# 7 processes (one per H) with k seeds each: sum_H V1(H) + (k - 1) * sum_H m(H) <= free
			k =int((free - sV1) // sm) + 1 if free > sV1 else 0
			cells += [str(k), str(7 * k)]
		md.append(f'| {task} | ' + ' | '.join(cells) + ' |')
	md.append('')

	# ---------------- packing (empirical)
	pk = sorted(glob.glob(os.path.join(R, 'packing', '*.json')))
	if pk:
		md += ['## Empirical packing tests (launch runs until CUDA OOM)', '',
			   '| experiment | runs that fit | first failure | GPU memory used when it failed (MiB) | mean process VRAM (MiB) |', '|---|---|---|---|---|']
		for f in pk:
			d = J(f)
			name = os.path.basename(f)[:-5]
			if name.startswith('vec_'):
				md.append(f'| {name} (one process, seeds S) | S <= {d["S_max_fit"]} | OOM at S = {d["S_min_oom"]} | – | – |')
				continue
			fail = d.get('failure') or {}
			# mean process VRAM over the runs that fit (x == x drops NaN entries)
			pv =[r.get('process_vram_mib') for r in d.get('runs', []) if r.get('process_vram_mib') == r.get('process_vram_mib')]
			md.append(f'| {name}{" (MPS)" if d.get("mps") else ""} | {d.get("n_fit")} | {"OOM" if fail.get("oom") else (fail.get("error", "")[:60] or "-")} | '
					  f'{fail.get("gpu_mem_used_mib", float("nan")):.0f} | {np.mean(pv) if pv else float("nan"):.0f} |')
		md.append('')
	open(os.path.join(DOCS, 'results_memory.md'), 'w').write('\n'.join(md))

	# figure: seeds per GPU vs H for 4 representative tasks
	fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=130)
	for ax, task in zip(axes, ['cartpole-swingup', 'walker-walk', 'humanoid-run', 'dog-run']):
		for mode, lab in (('cpu', 'CPU baseline (processes)'), ('gpu_opt', 'GPU optimised (processes)')):
			ys = []
			for H in HS:
				d = J(os.path.join(R, 'sweep_h', mode, f'{task}_H{H}.json'))
				ys.append(free // d['process_vram_mib'] if d else np.nan)
			ax.plot(HS, ys, color=COLOR[mode], lw=2, marker='o', ms=3.5, label=lab)
		if (task, False) in vec_rows:
			ax.plot(HS, [int(c) for c in vec_rows[(task, False)]], color=COLOR['vec'], lw=2, marker='o', ms=3.5, label='GPU vectorised (1 process)')
		if (task, True) in vec_rows:
			ax.plot(HS, [int(c) for c in vec_rows[(task, True)]], color='#4a3aa7', lw=2, marker='o', ms=3.5, label='GPU vectorised + ckpt_unroll')
		ax.set_title(task, fontsize=9); ax.set_yscale('log'); ax.grid(True, color='#e4e3df', lw=0.8)
		for s in ('top', 'right'):
			ax.spines[s].set_visible(False)
		ax.set_xlabel('H')
	axes[0].set_ylabel('seeds that fit on the RTX 3090 (log)')
	axes[-1].legend(frameon=False, fontsize=7, loc='upper right')
	fig.suptitle('How many TD-MPC seeds fit on one RTX 3090')
	fig.tight_layout(); fig.savefig(os.path.join(FIG, 'seeds_per_gpu_vs_H.png')); plt.close(fig)
	print('wrote docs/results_memory.md')


if __name__ == '__main__':
	main()
