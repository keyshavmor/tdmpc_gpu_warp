"""Summarise the MPS fault experiments (results/fault_hunt/*.json from bench/mps_fault_hunt.py) into one table.

Per variant: faults, rounds, exposure (process-hours of all processes on the GPU), fault rate per process-hour with
an exact 95 % Poisson (Garwood) confidence interval, median time to fault within a round, and the number of
process starts. Prints a markdown table and writes docs/results_fault_hunt.md.

Environment: any Python 3 with scipy (tdmpc-warp).
Usage: python bench/analyze_fault_hunt.py

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import glob, json, os, statistics
from scipy.stats import chi2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# description of each variant (what runs next to what); variants not listed are shown with their raw settings
DESC = {
	'baseline_vec': 'MPS only: 6 vectorised training processes (3 seeds each)',
	'env_only_vec': 'MPS only: 6 physics-only stress processes',
	'env_eager_vec': 'MPS only: 6 vectorised training processes, eager physics',
	'churn_single': 'MPS only: 6 single-seed training processes + 1 new MPS process every 30 s (SIGKILL after 60 s)',
	'routed_kill': 'MPS only: 6 training + 2 extra physics processes through MPS, restarted every 120 s by SIGKILL',
	'routed_graceful': 'MPS only: 6 training + 2 extra physics processes through MPS, clean exit every 120 s',
	'routed_vec_kill': 'MPS only: 6 training + 2 extra vectorised training processes through MPS, SIGKILL every 120 s',
	'mixed_vec': 'MIXED: 6 MPS training + 2 non-MPS vectorised training processes (restarted every 120 s)',
	'mixed_kill': 'MIXED: 6 MPS training + 2 non-MPS physics processes (SIGKILL every 120 s)',
	'mixed_graceful': 'MIXED: 6 MPS training + 2 non-MPS physics processes (clean exit every 120 s)',
	'mixed_ping': 'MIXED: 6 MPS training + 2 non-MPS minimal torch processes (20 s each, clean exit)',
	'mixed_vulkan': 'MIXED: 6 MPS training + 1 process creating Vulkan instances every second',
	'mixed_nocond': 'MIXED: 6 MPS training + 2 non-MPS physics processes without CUDA-graph conditional nodes',
	'mixed_eager': 'MIXED: 6 MPS training + 2 non-MPS physics processes with eager (non-graph) physics',
}


# lifetime (s) of the extra processes beyond the first `nproc` starts of a round (restart / churn settings of each
# variant, as run); the steady processes live until the end of the round
EXTRA_LIFE = {'mixed_vec': 120, 'mixed_kill': 120, 'mixed_graceful': 120, 'mixed_ping': 20, 'mixed_vulkan': 20,
			  'churn_single': 60, 'routed_kill': 120, 'routed_graceful': 120, 'routed_vec_kill': 120}


def exposure_hours(d):
	"""Summed lifetime of all processes of a variant in hours (exact if recorded, else from the restart settings)."""
	total = 0.0
	for r in d['rounds']:
		if 'lifetimes_s' in r:
			total += sum(r['lifetimes_s']); continue
		n, life = d.get('nproc', 6), EXTRA_LIFE.get(d['variant'], float('inf'))
		for i, s0 in enumerate(r['starts_s']):
			left = max(0.0, r['duration_s'] - s0)
			total += left if i < n else min(life, left)
	return total / 3600


def garwood(k, t):
	"""Exact 95 % confidence interval of a Poisson rate with k events in exposure t."""
	lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2 / t
	hi = chi2.ppf(0.975, 2 * (k + 1)) / 2 / t
	return lo, hi


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'fault_hunt', '*.json'))):
	d = json.load(open(f))
	if 'rounds' not in d:
		continue
	faults = [r['fault'] for r in d['rounds'] if r.get('fault') and 'note' not in r['fault']]
	ph = exposure_hours(d)
	k = len(faults)
	lo, hi = garwood(k, ph) if ph > 0 else (float('nan'), float('nan'))
	ttf = [x['t_since_round_s'] for x in faults]
	rows.append(dict(variant=d['variant'], desc=DESC.get(d['variant'], f"{d.get('script')} x{d.get('nproc')} {d.get('extra')}"),
					 rounds=len(d['rounds']), faults=k, process_hours=ph, rate=k / ph if ph else float('nan'), lo=lo, hi=hi,
					 median_ttf_s=statistics.median(ttf) if ttf else None, starts=sum(len(r['starts_s']) for r in d['rounds']),
					 minutes=d.get('minutes', 0)))
order = lambda r: (r['desc'].startswith('MIXED'), r['variant'])
rows.sort(key=order)
md = ['# MPS fault experiments', '',
	  '*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*', '',
	  'Each variant runs rounds of up to 20-30 min under a private CUDA MPS daemon (bench/mps_fault_hunt.py); a round ends '
	  'at the first fatal fault (recorded by the MPS server and the kernel as Xid 31). Exposure = summed lifetime of all '
	  'processes on the GPU. Rate = faults per process-hour with an exact 95 % Poisson confidence interval. Raw data: '
	  '`results/fault_hunt/<variant>.json`; analysis: docs/08_known_issues.md.', '',
	  '| variant | setup | wall time (min) | process starts | exposure (process-h) | faults | rate per process-h [95 % CI] | median time to fault |',
	  '|---|---|---|---|---|---|---|---|']
for r in rows:
	ttf = f"{r['median_ttf_s'] / 60:.1f} min" if r['median_ttf_s'] is not None else '-'
	md.append(f"| {r['variant']} | {r['desc']} | {r['minutes']:.0f} | {r['starts']} | {r['process_hours']:.1f} | {r['faults']} | "
			  f"{r['rate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] | {ttf} |")
mps_only = [r for r in rows if not r['desc'].startswith('MIXED')]
k0, t0 = sum(r['faults'] for r in mps_only), sum(r['process_hours'] for r in mps_only)
lo, hi = garwood(k0, t0)
md += ['', f'All MPS-only variants together: {k0} faults in {t0:.1f} process-hours, rate {k0 / t0:.3f} [{lo:.3f}, {hi:.3f}] per process-hour.', '']
open(os.path.join(ROOT, 'docs', 'results_fault_hunt.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
