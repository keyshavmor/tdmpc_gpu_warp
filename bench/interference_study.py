"""How many runs with DIFFERENT horizons can share the RTX 3090 without slowing any of them down?

For each task: horizons are assigned from a fixed mixed order (5, 30, 15, 2, 25, 10, 20, 5, 30, ...),
so N concurrent runs use the first N entries. Each run is a separate process (run_worker.py, steady-state
training loop: plan + env step + update). Interference of run i = its steps/s with N runs / its steps/s
when running ALONE at the same horizon (solo baselines measured with the same harness).
Reported per task and mode: max N with every run >= 95 % and >= 90 % of its solo speed, total throughput.

  modes: gpu_mps  (optimised GPU port, CUDA MPS)     - default, all tasks
         gpu      (optimised GPU port, no MPS)        - comparison, subset
         cpu      (original CPU-MuJoCo code)          - comparison, subset
Results: results/interference/<mode>/<task>.json (resumable per task); the long-horizon variant
(order 10, 100, 50, 25, 75; walker, humanoid, dog under MPS) is stored as <task>_longH.json.

  python bench/interference_study.py [--duration 30]

Run in the `tdmpc-warp` env (the cpu comparison launches the original code in the `tdmpc` env through
bench/multi_run.py). N is increased over 2, 3, 4, 5, 6, 8, 10, 12 and stops once a run fails (OOM) or the
slowest run drops below 0.6 of its solo speed. The solo baselines stored here are reused by
bench/interference_sameH.py and bench/partition_study.py. Tabulated by bench/analyze_interference.py
(docs/results_capacity.md, section A, and docs/figures/interference_min_rel_speed.png); reported in
docs/04_multi_run_capacity.md (4.4: no-slowdown capacity = 1 for all 12 domains under MPS).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
# N concurrent runs use the first N horizons of the (cycled) order: a deliberately mixed set of light and
# heavy runs, so that every N >= 2 contains both short and long horizons
ORDER = [5, 30, 15, 2, 25, 10, 20]
LONG_ORDER = [10, 100, 50, 25, 75]
REPS = ['acrobot-swingup', 'cartpole-swingup', 'cheetah-run', 'cup-catch', 'dog-run', 'finger-spin', 'fish-swim',
		'hopper-hop', 'humanoid-run', 'quadruped-run', 'reacher-easy', 'walker-walk']
OPT = 'graph_agent=true,pace_host=true'


def spec(mode, task, H):
	"""multi_run.py spec of one run: original code for mode 'cpu', otherwise the optimised GPU port."""
	return f'cpu:{task}:{H}' if mode == 'cpu' else f'gpu:{task}:{H}:{OPT}'


def multi(mode, specs, duration, tmpdir):
	"""Run the given specs concurrently with bench/multi_run.py (MPS for 'gpu_mps', original threading for
	'cpu') and return its summary dict (None if it produced no result)."""
	out =os.path.join(tmpdir, 'mr.json')
	cmd = [PY, os.path.join(HERE, 'multi_run.py'), '--spec'] + specs + ['--duration', str(duration), '--out', out]
	cmd += ['--mps'] if mode == 'gpu_mps' else []
	cmd += ['--threads', '0'] if mode == 'cpu' else []
	subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
	d = json.load(open(out)) if os.path.exists(out) else None
	if os.path.exists(out):
		os.remove(out)
	return d


def study(mode, task, order, duration, ns, tag=''):
	"""Solo baselines for every horizon in `order`, then N = ns[0], ns[1], ... concurrent runs.

	Relative speed of a run = steps/s with N runs / solo steps/s at its horizon. The no-slowdown capacity
	at threshold thr is the largest N whose slowest run (and that of every smaller N) keeps >= thr.
	"""
	outdir =os.path.join(ROOT, 'results', 'interference', mode)
	os.makedirs(outdir, exist_ok=True)
	fout = os.path.join(outdir, f'{task}{tag}.json')
	if os.path.exists(fout):
		return
	tmp = os.path.join(outdir, '_tmp'); os.makedirs(tmp, exist_ok=True)
	solo = {}
	for H in sorted(set(order)):
		d = multi(mode, [spec(mode, task, H)], duration, tmp)
		ok = [r for r in (d or {}).get('runs', []) if r.get('ok')]
		solo[H] = ok[0]['steps_per_s'] if ok else None
		print(f'{mode} {task}{tag} solo H={H}: {solo[H]}', flush=True)
	res = dict(task=task, mode=mode, order=order, solo_steps_per_s=solo, trials=[])
	for n in ns:
		hs = [order[i % len(order)] for i in range(n)]
		d = multi(mode, [spec(mode, task, H) for H in hs], duration, tmp)
		if d is None:
			res['trials'].append(dict(n=n, horizons=hs, error='no result')); break
		runs = d.get('runs', [])
		okr = [r for r in runs if r.get('ok')]
		rel = [r['steps_per_s'] / solo[r['horizon']] for r in okr if solo.get(r['horizon'])]
		# min_rel only counts when every run of the trial survived (an OOM trial has no valid minimum)
		t = dict(n=n, horizons=hs, n_ok=len(okr), n_oom=d.get('n_oom'), rel_speed=rel,
				 min_rel=min(rel) if rel and len(okr) == n else None, mean_rel=sum(rel) / len(rel) if rel else None,
				 total_steps_per_s=d.get('steps_per_s_total'), gpu_util=d.get('gpu_util_mean'),
				 cpu_cores_each=d.get('cpu_cores_each'), gpu_mem_peak_mib=d.get('gpu_mem_peak_mib'))
		res['trials'].append(t)
		print(f'{mode} {task}{tag} N={n} H={hs}: min rel {t["min_rel"]}, mean rel {t["mean_rel"]}, total {t["total_steps_per_s"]}, '
			  f'ok {len(okr)}/{n} oom {t["n_oom"]}', flush=True)
		if len(okr) < n or (t['min_rel'] is not None and t['min_rel'] < 0.6):
			break  # past the interference-free region (or out of memory)
	# capacity at threshold thr: largest N with min_rel >= thr for this AND all smaller trials (N = 1 always ok)
	ok_n = lambda thr: max([1] + [t['n'] for t in res['trials'] if t.get('min_rel') is not None and t['min_rel'] >= thr
								 and all(u.get('min_rel') is not None and u['min_rel'] >= thr for u in res['trials'] if u['n'] <= t['n'])])
	res['max_n_within_5pct'] = ok_n(0.95)
	res['max_n_within_10pct'] = ok_n(0.90)
	json.dump(res, open(fout, 'w'), indent=1)
	os.rmdir(tmp)
	print(f'==> {mode} {task}{tag}: no-slowdown capacity {res["max_n_within_5pct"]} runs (<=5%), {res["max_n_within_10pct"]} runs (<=10%)', flush=True)


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--duration', type=float, default=30)
	a = p.parse_args()
	NS = [2, 3, 4, 5, 6, 8, 10, 12]
	for t in REPS:
		study('gpu_mps', t, ORDER, a.duration, NS)
	for t in ('cartpole-swingup', 'walker-walk', 'humanoid-run', 'dog-run'):
		study('gpu', t, ORDER, a.duration, NS)
		study('cpu', t, ORDER, a.duration, NS)
	for t in ('walker-walk', 'humanoid-run', 'dog-run'):
		study('gpu_mps', t, LONG_ORDER, a.duration, [2, 3, 4, 5, 6, 8], tag='_longH')
	print('interference study done', flush=True)
