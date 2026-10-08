"""Interference-free sharing by MPS SM partitioning (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE).

For SM share p in {50, 33, 25, 12.5} %: N = floor(100 / p) runs with mixed horizons (order 5, 30, 15, 2, 25,
10, 20, ...) run concurrently, each limited to p % of the SMs. Isolation = each run's speed with N
partitioned runs / its speed running alone with the same p % share (1.0 = no interference at all).
Also reported: per-run speed relative to an unpartitioned solo run (the price of the partition).
Results: results/partition/<task>.json (resumable)

  python bench/partition_study.py [task ...]      # default: walker-walk dog-run

As implemented, the shares 50 % (N = 2) and 25 % (N = 4) are measured. All runs are the optimised GPU
port under CUDA MPS (bench/multi_run.py --mps --sm_pct p); unpartitioned solo speeds are reused from
results/interference/gpu_mps/<task>.json when available. Run in the `tdmpc-warp` env (started by
bench/pipeline_phase4b.sh / pipeline_final.sh). Tabulated by bench/analyze_interference.py
(docs/results_capacity.md, section C); reported in docs/04_multi_run_capacity.md (4.6: partitioning does
not make runs interference-free and was rejected).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
ORDER = [5, 30, 15, 2, 25, 10, 20]
OPT = 'graph_agent=true,pace_host=true'
OUT = os.path.join(ROOT, 'results', 'partition'); os.makedirs(OUT, exist_ok=True)


def multi(specs, pct, duration=30):
	"""Run specs concurrently under MPS, each client limited to pct % of the SMs (None = no limit).

	Returns ({horizon: steps/s} of the successful runs, full multi_run summary). Runs are keyed by
	horizon, which is unique within each partitioned set used below.
	"""
	f =os.path.join(OUT, '_mr.json')
	cmd = [PY, os.path.join(HERE, 'multi_run.py'), '--spec'] + specs + ['--duration', str(duration), '--out', f, '--mps']
	if pct is not None:
		cmd += ['--sm_pct', str(pct)]
	subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
	d = json.load(open(f)) if os.path.exists(f) else None
	if os.path.exists(f):
		os.remove(f)
	return {r['horizon']: r['steps_per_s'] for r in (d or {}).get('runs', []) if r.get('ok')} if d else {}, d


for task in (sys.argv[1:] or ['walker-walk', 'dog-run']):
	fout = os.path.join(OUT, f'{task}.json')
	if os.path.exists(fout):
		continue
	res = dict(task=task, order=ORDER, unpartitioned_solo={}, shares=[])
	prev = os.path.join(ROOT, 'results', 'interference', 'gpu_mps', f'{task}.json')  # same harness, reuse
	prev = json.load(open(prev))['solo_steps_per_s'] if os.path.exists(prev) else {}
	for H in ORDER[:4]:
		if str(H) in prev:
			res['unpartitioned_solo'][H] = prev[str(H)]
		else:
			s, _ = multi([f'gpu:{task}:{H}:{OPT}'], None)
			res['unpartitioned_solo'][H] = s.get(H)
	for pct, n in ((50, 2), (25, 4)):
		hs = [ORDER[i % len(ORDER)] for i in range(n)]
		solo = {}
		for H in sorted(set(hs)):
			s, _ = multi([f'gpu:{task}:{H}:{OPT}'], pct)
			solo[H] = s.get(H)
		_, d = multi([f'gpu:{task}:{H}:{OPT}' for H in hs], pct)
		runs = [r for r in (d or {}).get('runs', []) if r.get('ok')]
		iso = [r['steps_per_s'] / solo[r['horizon']] for r in runs if solo.get(r['horizon'])]
		vs_full = [r['steps_per_s'] / res['unpartitioned_solo'][r['horizon']] for r in runs if res['unpartitioned_solo'].get(r['horizon'])]
		e = dict(sm_pct=pct, n=n, horizons=hs, n_ok=len(runs), solo_partitioned=solo,
				 isolation_min=min(iso) if iso else None, isolation_mean=sum(iso) / len(iso) if iso else None,
				 speed_vs_unpartitioned_solo_min=min(vs_full) if vs_full else None,
				 speed_vs_unpartitioned_solo_mean=sum(vs_full) / len(vs_full) if vs_full else None,
				 total_steps_per_s=(d or {}).get('steps_per_s_total'))
		res['shares'].append(e)
		print(f'{task} p={pct}% N={n}: isolation min {e["isolation_min"]}, speed vs full solo mean {e["speed_vs_unpartitioned_solo_mean"]}, total {e["total_steps_per_s"]}', flush=True)
	json.dump(res, open(fout, 'w'), indent=1)
print('partition study done', flush=True)
