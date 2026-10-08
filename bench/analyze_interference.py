"""Chapter 4 data: concurrent-run capacity of the RTX 3090 (interference, partitioning, contention).

How many concurrent TD-MPC runs (different horizons) fit on the RTX 3090
(a) without slowing any run down (interference study), (b) in memory (packing study), and the
throughput / CPU / GPU picture (contention study). Writes docs/results_capacity.md + figures.

Sections of docs/results_capacity.md and their inputs:
  A   no-slowdown capacity, mixed horizons   results/interference/<mode>/<task>[_longH].json  (bench/interference_study.py)
      memory limit column                    results/sweep_h/{gpu_opt,cpu}/ (per-process VRAM of the horizon sweep)
  A2  same-H pairs, memory limit per H       results/interference_sameH/                     (bench/interference_sameH.py)
  C   MPS SM partitioning                    results/partition/                              (bench/partition_study.py)
  B   throughput / CPU / GPU with N seeds    results/contention/<task>/<mode>_N<N>.json      (bench/contention_study.py)
Figures: docs/figures/interference_min_rel_speed.png, docs/figures/contention_throughput.png.

  python bench/analyze_interference.py

Pure post-processing (no GPU needed), `tdmpc-warp` env. Summarised in docs/04_multi_run_capacity.md.

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
DOM_TASKS = {}
for t in [l.strip() for l in open(os.path.join(ROOT, 'tasks.txt')) if l.strip()]:
	DOM_TASKS.setdefault(t.split('-')[0], []).append(t)
COL = {'gpu_mps': '#1baf7a', 'gpu': '#eb6834', 'cpu': '#2a78d6'}
LAB = {'gpu_mps': 'GPU optimised + MPS', 'gpu': 'GPU optimised, no MPS', 'cpu': 'CPU MuJoCo (original)'}


def J(p):
	"""Load a JSON file, or None if it does not exist."""
	return json.load(open(p)) if os.path.exists(p) else None


def free_mib():
	"""VRAM available to runs: 24 GiB minus the idle desktop usage recorded in gpu_baseline_memory.txt."""
	return 24576 - float(open(os.path.join(R, 'gpu_baseline_memory.txt')).read().split()[0])


def mem_capacity_mixed(task, mode, hs):
	"""Memory limit for a mix: sum of measured per-process VRAM (sweep) over the cycled horizons.

	Adds processes in the study's horizon order until the next one would exceed the free VRAM and returns
	that count (None if a sweep point is missing).
	"""
	# both GPU modes run the optimised port -> its sweep VRAM; cpu mode -> the original code's VRAM
	sm ={'gpu_mps': 'gpu_opt', 'gpu': 'gpu_opt', 'cpu': 'cpu'}[mode]
	v = {}
	for H in set(hs):
		d = J(os.path.join(R, 'sweep_h', sm, f'{task}_H{H}.json'))
		if not d:
			return None
		v[H] = d['process_vram_mib']
	free, used, n = free_mib(), 0.0, 0
	while True:
		H = hs[n % len(hs)]
		if used + v[H] > free:
			return n
		used += v[H]; n += 1


def main():
	"""Build every section of docs/results_capacity.md (those whose results exist) and the two figures."""
	md = ['# How many runs fit on one RTX 3090 - without slowing each other down', '', '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '']
	# ---------------- interference
	rows = []
	for f in sorted(glob.glob(os.path.join(R, 'interference', '*', '*.json'))):
		d = J(f); mode = os.path.basename(os.path.dirname(f)); tag = '_longH' if f.endswith('_longH.json') else ''
		rows.append((mode, d['task'], tag, d))
	if rows:
		md += ['## A. No-slowdown capacity (interference study)', '',
			   'Each run is a separate process with its own horizon, taken in order from the mixed list '
			   '**5, 30, 15, 2, 25, 10, 20, 5, 30, ...** (long-horizon variant: 10, 100, 50, 25, 75, ...). Every run executes the '
			   'steady-state training loop (plan + env step + update). A run\'s *relative speed* = its agent steps/s with N runs '
			   'sharing the GPU / its steps/s when running alone at the same horizon. **No-slowdown capacity** = largest N for '
			   'which *every* run keeps >= 95 % (or >= 90 %) of its solo speed (and all smaller N too). Memory limit = how many '
			   'processes of the same horizon mix fit in VRAM (from measured per-process VRAM; validated by the packing tests).', '',
			   '| domain (tasks) | mode | horizons | no-slowdown capacity (<= 5 %) | (<= 10 %) | memory limit (processes) | '
			   'min relative speed at N = 2 / 3 / 4 / 6 / 8 | total steps/s at N = 1 / 2 / 4 / 8 |', '|---|---|---|---|---|---|---|---|']
		for mode, task, tag, d in sorted(rows, key=lambda r: (r[2], r[0] != 'gpu_mps', r[1])):
			dom = task.split('-')[0]
			tr = {t['n']: t for t in d['trials']}
			# slowest run / its solo speed at N = 2, 3, 4, 6, 8 ('OOM' if the trial lost a run, '–' if not run)
			mr = ' / '.join(f"{tr[n]['min_rel']:.2f}" if n in tr and tr[n].get('min_rel') is not None else ('OOM' if n in tr else '–') for n in (2, 3, 4, 6, 8))
			# N = 1 throughput = solo run at the first horizon of the order (JSON keys are strings)
			solo_tot = d['solo_steps_per_s'].get(str(d['order'][0])) or d['solo_steps_per_s'].get(d['order'][0])
			tot = [f'{solo_tot:.0f}' if solo_tot else '–'] + [f"{tr[n]['total_steps_per_s']:.0f}" if n in tr and tr[n].get('total_steps_per_s') else '–' for n in (2, 4, 8)]
			memcap = mem_capacity_mixed(task, mode, d['order'])
			hs = ','.join(str(h) for h in d['order'])
			md.append(f"| {dom} ({', '.join(DOM_TASKS.get(dom, [task]))}) | {LAB[mode]} | {hs} | **{d['max_n_within_5pct']}** | "
					  f"{d['max_n_within_10pct']} | {memcap if memcap is not None else '–'} | {mr} | {' / '.join(tot)} |")
		md.append('')
		# figure: min relative speed vs N
		gm = [r for r in rows if r[0] == 'gpu_mps' and r[2] == '']
		if gm:
			fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
			cmap = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
			for i, (_, task, _, d) in enumerate(sorted(gm, key=lambda r: r[1])):
				ns = [1] + [t['n'] for t in d['trials'] if t.get('min_rel') is not None]
				ys = [1.0] + [t['min_rel'] for t in d['trials'] if t.get('min_rel') is not None]
				ax.plot(ns, ys, lw=2, marker='o', ms=3.5, color=cmap[i % 8], ls='-' if i < 8 else '--', label=task)
			ax.axhline(0.95, color='#52514e', lw=1, ls=':'); ax.axhline(0.90, color='#52514e', lw=1, ls='--')
			from matplotlib.ticker import MaxNLocator
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			ax.text(ax.get_xlim()[1], 0.95, ' 95 %', va='center', fontsize=7, color='#52514e')
			ax.text(ax.get_xlim()[1], 0.90, ' 90 %', va='center', fontsize=7, color='#52514e')
			ax.set_xlabel('number of concurrent runs N (mixed horizons 5, 30, 15, 2, 25, 10, 20, ...)')
			ax.set_ylabel('speed of the slowest run / its solo speed')
			ax.set_title('Interference between concurrent GPU TD-MPC runs (optimised port + MPS, RTX 3090)')
			ax.grid(True, color='#e4e3df', lw=0.8)
			for s in ('top', 'right'):
				ax.spines[s].set_visible(False)
			ax.legend(frameon=False, fontsize=7, ncol=2)
			fig.tight_layout(); fig.savefig(os.path.join(FIG, 'interference_min_rel_speed.png')); plt.close(fig)

	# ---------------- same-H interference + memory limit per H (Q1)
	sh = sorted(glob.glob(os.path.join(R, 'interference_sameH', '*.json')))
	if sh:
		md += ['## A2. Per task and horizon: runs of the SAME H in parallel - memory limit vs no-slowdown limit', '',
			   'Memory limit = floor(free VRAM / measured process VRAM at that H) for the optimised GPU port (horizon sweep; '
			   'validated by packing tests to +-1 run). No-slowdown limit: 2 concurrent runs of the same H were measured at '
			   'H = 2 and H = 30 (+ 50, 100 for walker and dog); "each" = speed of each of the 2 runs / its solo speed. '
			   'Because a single run already saturates the GPU and cost grows monotonically with H, the H = 2 and H = 30 '
			   'measurements bound every H in between.', '']
		HS = [2, 5, 10, 15, 20, 25, 30]
		md += ['| task | ' + ' | '.join(f'H={H} mem' for H in HS) + ' | 2 runs at H=2: each | 2 runs at H=30: each | no-slowdown limit (any H) |',
			   '|' + '---|' * (len(HS) + 4)]
		shd = {}
		for f in sh:
			d = J(f); shd[(d['task'], d['horizon'])] = d
		for task in sorted({k[0] for k in shd}):
			mem = []
			for H in HS:
				v = J(os.path.join(R, 'sweep_h', 'gpu_opt', f'{task}_H{H}.json'))
				mem.append(str(int(free_mib() // v['process_vram_mib'])) if v else '–')
			e2 = shd.get((task, 2)); e30 = shd.get((task, 30))
			fmt = lambda e: f"{min(e['rel_speed']):.2f}" if e and e.get('rel_speed') else '–'
			extra = ''.join(f", H={H}: {min(shd[(task, H)]['rel_speed']):.2f}" for H in (50, 100) if (task, H) in shd)
			cap = min(e['no_slowdown_capacity'] for (t, H), e in shd.items() if t == task)  # worst case over measured H
			md.append(f'| {task} | ' + ' | '.join(mem) + f' | {fmt(e2)} | {fmt(e30)}{extra} | **{cap}** |')
		md.append('')
	# ---------------- partition
	pt = sorted(glob.glob(os.path.join(R, 'partition', '*.json')))
	if pt:
		md += ['## C. MPS SM partitioning (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) - does it isolate runs?', '',
			   '| task | SM share per run | runs | isolation (speed / solo at same share), min | speed / unrestricted solo, mean | total steps/s |', '|---|---|---|---|---|---|']
		for f in pt:
			d = J(f)
			for e in d['shares']:
				md.append(f"| {d['task']} | {e['sm_pct']:.0f} % | {e['n']} (H = {','.join(map(str, e['horizons']))}) | {e['isolation_min']:.2f} | "
						  f"{e['speed_vs_unpartitioned_solo_mean']:.2f} | {e['total_steps_per_s']:.1f} |")
		md.append('')

	# ---------------- contention (same horizon, N runs)
	cont = sorted(glob.glob(os.path.join(R, 'contention', '*', '*.json')))
	if cont:
		md += ['## B. Throughput, CPU use and GPU utilisation with N concurrent seeds (same horizon H = 5)', '',
			   '| task | mode | N | total seed-steps/s | per-run steps/s (mean) | CPU cores per run | system CPU % | GPU util % | peak GPU MiB | OOM |',
			   '|---|---|---|---|---|---|---|---|---|---|']
		data = {}
		for f in cont:
			d = J(f); task = os.path.basename(os.path.dirname(f)); mode, n = os.path.basename(f)[:-5].rsplit('_N', 1)
			data[(task, mode, int(n))] = d
		for (task, mode, n), d in sorted(data.items()):
			each = d.get('steps_per_s_each') or []
			md.append(f"| {task} | {mode} | {n} | {d['steps_per_s_total']:.1f} | {np.mean(each) if each else float('nan'):.1f} | "
					  f"{np.mean(d['cpu_cores_each']) if d.get('cpu_cores_each') else float('nan'):.2f} | {100 * d['system_cpu_util']:.0f} | "
					  f"{d['gpu_util_mean']:.0f} | {d['gpu_mem_peak_mib']:.0f} | {d['n_oom']} |")
		md.append('')
		tasks = sorted({k[0] for k in data})
		fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), dpi=130, squeeze=False)
		cm = {'cpu': '#2a78d6', 'cpu_pin': '#4a3aa7', 'gpu': '#eb6834', 'gpu_opt': '#8a8983', 'gpu_mps': '#1baf7a', 'vec': '#eda100'}
		lab = {'cpu': 'CPU original', 'cpu_pin': 'CPU original, 1 thread', 'gpu': 'GPU faithful', 'gpu_opt': 'GPU optimised (no MPS)',
			   'gpu_mps': 'GPU optimised + MPS', 'vec': 'GPU vectorised (1 process)'}
		for ax, task in zip(axes[0], tasks):
			for mode in cm:
				# only points where every run survived (OOM points are excluded, see the table)
				ns = sorted(n for (t, m, n) in data if t == task and m == mode and data[(t, m, n)]['n_ok'] == data[(t, m, n)]['n_runs'])
				if ns:
					ax.plot(ns, [data[(task, mode, n)]['steps_per_s_total'] for n in ns], color=cm[mode], lw=2, marker='o', ms=3.5, label=lab[mode], ls='--' if mode == 'gpu_opt' else '-')
			ax.set_title(task, fontsize=9); ax.set_xscale('log', base=2); ax.set_xlabel('concurrent seeds N')
			ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(['1', '2', '4', '8', '16']); ax.set_ylim(bottom=0)
			ax.grid(True, color='#e4e3df', lw=0.8)
			for s in ('top', 'right'):
				ax.spines[s].set_visible(False)
		axes[0][0].set_ylabel('total seed-steps / s (all runs)')
		h_, l_ = axes[0][-1].get_legend_handles_labels()
		fig.legend(h_, l_, loc='lower center', ncol=6, frameon=False, fontsize=8)
		fig.suptitle('Aggregate throughput of N concurrent TD-MPC seeds on one RTX 3090 (H = 5; OOM points omitted)')
		fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, 'contention_throughput.png')); plt.close(fig)
	open(os.path.join(DOCS, 'results_capacity.md'), 'w').write('\n'.join(md))
	print('wrote docs/results_capacity.md')


if __name__ == '__main__':
	main()
