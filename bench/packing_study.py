"""Empirical "how many runs fit on the RTX 3090" study (drives bench/pack_test.py), resumable.

  same-H   : N processes of one task at H=5 until CUDA OOM (cpu baseline, gpu_opt, gpu_opt+MPS)
  mixed-H  : processes cycling through horizons 5,10,20,30 of one task until OOM (gpu_opt+MPS)
  cross    : processes cycling through different tasks AND horizons until OOM (gpu_opt+MPS)
  vec      : largest number of seeds S that fits in ONE vectorised process (bisection on S)
Results: results/packing/*.json

  python bench/packing_study.py                 # all parts: same mixed cross vec
  python bench/packing_study.py same vec        # selected parts

Run in the `tdmpc-warp` env on an otherwise idle GPU (the same-H cpu baseline launches the original code
through pack_test.py with the `tdmpc` interpreter). The vec part probes with bench/bench_step.py
(k = 3): doubling S from 8 until OOM, then bisection to ~6 %. These empirical limits validate the computed
memory limits of docs/04_multi_run_capacity.md (to +-1 run); tables in docs/results_memory.md
("Empirical packing tests", written by bench/analyze_memory.py, which also uses the vec bisection when
bench/vec_memory.py could not measure S = 8).

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
OUT = os.path.join(ROOT, 'results', 'packing')
os.makedirs(OUT, exist_ok=True)
OPT = 'graph_agent=true,pace_host=true'  # cfg overrides of the optimised GPU port


def pack(name, cycle, mps=False, max_n=200):
	"""Run bench/pack_test.py for one experiment (skipped if results/packing/<name>.json exists)."""
	out =os.path.join(OUT, f'{name}.json')
	if os.path.exists(out):
		return json.load(open(out))
	cmd = [PY, os.path.join(HERE, 'pack_test.py'), '--cycle'] + cycle + ['--max', str(max_n), '--out', out] + (['--mps'] if mps else [])
	r = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
	d = json.load(open(out)) if os.path.exists(out) else dict(error=r.stderr[-500:])
	print(f'{name}: fit {d.get("n_fit")} (failure: {(d.get("failure") or {}).get("oom")})', flush=True)
	return d


def vec_fits(task, H, S, extra=''):
	"""Does one vectorised process with S seeds fit? (bench_step with k=3)"""
	out = os.path.join(OUT, f'_vecprobe.json')
	ex = ['vec_agent=true', f'num_seeds={S}', 'pace_host=true'] + ([extra] if extra else [])
	cmd = [PY, os.path.join(HERE, 'bench_step.py'), '--mode', 'gpu', '--task', task, '--horizon', str(H), '--k', '3', '--out', out, '--extra'] + ex
	r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
	ok = r.returncode == 0 and os.path.exists(out)
	d = json.load(open(out)) if ok else None
	if os.path.exists(out):
		os.remove(out)
	return ok, d


def vec_max(task, H, s_hi=512, extra=''):
	"""Largest S fitting in one vectorised process: double S from 8 until OOM (or s_hi), then bisect to
	within max(1, S/16). Writes results/packing/vec_<task>_H<H>[_<extra>].json (S_max_fit, S_min_oom)."""
	name =f'vec_{task}_H{H}' + ('_' + extra.replace('=', '-') if extra else '')
	out = os.path.join(OUT, f'{name}.json')
	if os.path.exists(out):
		return
	probes = []
	lo, hi, S = 0, None, 8
	while hi is None and S <= s_hi:
		ok, d = vec_fits(task, H, S, extra); probes.append(dict(S=S, fits=ok, vram=d['process_vram_mib'] if d else None))
		if ok: lo, S = S, S * 2
		else: hi = S
	while hi is not None and hi - lo > max(1, lo // 16):
		S = (lo + hi) // 2
		ok, d = vec_fits(task, H, S, extra); probes.append(dict(S=S, fits=ok, vram=d['process_vram_mib'] if d else None))
		if ok: lo = S
		else: hi = S
	res = dict(task=task, horizon=H, extra=extra, S_max_fit=lo, S_min_oom=hi, probes=probes)
	json.dump(res, open(out, 'w'), indent=1)
	print(f'{name}: S fits up to {lo}, OOM at {hi}', flush=True)


if __name__ == '__main__':
	what = sys.argv[1:] or ['same', 'mixed', 'cross', 'vec']
	tasks = ['cartpole-swingup', 'walker-walk', 'humanoid-run', 'dog-run']
	if 'same' in what:
		for t in tasks:
			pack(f'same_gpu_opt_mps_{t}_H5', [f'gpu:{t}:5:{OPT}'], mps=True)
		for t in ('walker-walk', 'dog-run'):
			pack(f'same_gpu_opt_{t}_H5', [f'gpu:{t}:5:{OPT}'], mps=False)
			pack(f'same_cpu_{t}_H5', [f'cpu:{t}:5'], mps=False)
	if 'mixed' in what:
		for t in ('walker-walk', 'dog-run', 'humanoid-run'):
			pack(f'mixed_gpu_opt_mps_{t}_H5-10-20-30', [f'gpu:{t}:{H}:{OPT}' for H in (5, 10, 20, 30)], mps=True)
	if 'cross' in what:
		pack('cross_gpu_opt_mps', [f'gpu:dog-run:5:{OPT}', f'gpu:humanoid-run:10:{OPT}', f'gpu:walker-walk:20:{OPT}',
								   f'gpu:cartpole-swingup:30:{OPT}', f'gpu:quadruped-run:15:{OPT}', f'gpu:cheetah-run:25:{OPT}'], mps=True)
	if 'vec' in what:
		for t in tasks:
			for H in (5, 30):
				vec_max(t, H)
		vec_max('dog-run', 30, extra='ckpt_unroll=true')
		vec_max('walker-walk', 30, extra='ckpt_unroll=true')
	print('packing study done', flush=True)
