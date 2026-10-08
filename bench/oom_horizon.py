"""Single-run OOM horizon per task and mode, confirmed with real runs.

1. linear fit of process VRAM vs H from the horizon sweep (H = 2..30)
2. predicted OOM horizon H* = (free VRAM - intercept) / slope, free = total - other processes (desktop)
3. valid horizons: H <= L - ceil(batch / (seed_steps/L + 1)) (the original algorithm samples the first
   batch without replacement from (L-H) valid start indices per episode; larger H crashes the original)
   - if H* > H_max: run at H_max and confirm it fits ("no OOM in the valid range")
   - else run below/above H* and bisect the fit/OOM boundary to within ~2 %
Each probe = bench_step.py with k=3 (3 planning steps + 3 updates at horizon H; peak is reached in update).
Results: results/oom_horizon/<mode>/<task>.json
  python bench/oom_horizon.py [--tasks ...] [--modes gpu gpu_opt cpu vec1 vec1_ckpt]

Run in the `tdmpc-warp` env (default modes: gpu, gpu_opt, cpu, vec1_ckpt; one representative task per
domain). The cpu mode probes the original code with the `tdmpc` env interpreter. The vec modes have no
sweep data and fit their line from their own probes at H = 5, 15, 30. Needs the horizon sweep
(results/sweep_h/, bench/sweep_h.py) first, and the GPU must otherwise be idle: the free VRAM is read from
nvidia-smi before each prediction. Tabulated by bench/analyze_memory.py in docs/results_memory.md
("Single run: OOM horizon"); summarised in docs/04_multi_run_capacity.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = {'cpu': os.environ.get('TDMPC_CPU_PYTHON', os.path.expanduser('~/anaconda3/envs/tdmpc/bin/python')),
	  'gpu': os.environ.get('TDMPC_WARP_PYTHON', sys.executable)}
# mode -> (bench_step --mode, bench_step flags, cfg overrides, sweep_h mode whose VRAM(H) points are fitted)
MODES = {
	'cpu': ('cpu', ['--threads', '0'], [], 'cpu'),
	'gpu': ('gpu', [], [], 'gpu'),
	'gpu_opt': ('gpu', [], ['graph_agent=true', 'pace_host=true'], 'gpu_opt'),
	'vec1': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true'], 'gpu_opt'),
	'vec1_ckpt': ('gpu', [], ['vec_agent=true', 'num_seeds=1', 'pace_host=true', 'ckpt_unroll=true'], 'gpu_opt'),
}
SEED_STEPS = 5000  # seed_steps of the default config (random-action episodes before the first update)
# one representative task per domain (tasks of a domain share networks / buffer sizes, hence memory)
REPS = ['acrobot-swingup', 'cartpole-swingup', 'cheetah-run', 'cup-catch', 'dog-run', 'finger-spin', 'fish-swim',
		'hopper-hop', 'humanoid-run', 'quadruped-run', 'reacher-easy', 'walker-walk']


def smi_used_total():
	"""(used, total) GPU memory in MiB from nvidia-smi."""
	o =subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
					   capture_output=True, text=True).stdout.split(',')
	return float(o[0]), float(o[1])


def probe(mode, task, H, outdir):
	"""Run one short bench_step.py (k=3) at horizon H and classify the outcome.

	Returns {H, fits: True, vram, torch_reserved} on success, otherwise {H, fits: False, oom} where `oom`
	is detected from torch / warp / CUDA error strings in stderr (any other failure keeps its error text).
	"""
	m, flags, extra, _ = MODES[mode]
	out = os.path.join(outdir, f'probe_{task}_H{H}.json')
	cmd = [PY[m], os.path.join(HERE, 'bench_step.py'), '--mode', m, '--task', task, '--horizon', str(H), '--k', '3',
		   '--out', out] + flags + (['--extra'] + extra if extra else [])
	t = time.time()
	r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
	if r.returncode == 0 and os.path.exists(out):
		d = json.load(open(out))
		return dict(H=H, fits=True, vram=d['process_vram_mib'], torch_reserved=d['torch_peak_reserved_mib'], s=time.time() - t)
	err = r.stderr[-3000:]
	oom = any(k in err.lower() for k in ('out of memory', 'outofmemory', 'cudaerrormemoryallocation', 'failed to allocate'))
	return dict(H=H, fits=False, oom=oom, error=None if oom else err[-400:], s=time.time() - t)


def main():
	"""For every mode x task: fit VRAM(H), predict the OOM horizon, then confirm / bracket it with probes."""
	p = argparse.ArgumentParser()
	p.add_argument('--tasks', nargs='+', default=REPS)
	p.add_argument('--modes', nargs='+', default=['gpu', 'gpu_opt', 'cpu', 'vec1_ckpt'])
	a = p.parse_args()
	for mode in a.modes:
		for task in a.tasks:
			outdir = os.path.join(ROOT, 'results', 'oom_horizon', mode)
			os.makedirs(outdir, exist_ok=True)
			fout = os.path.join(outdir, f'{task}.json')
			if os.path.exists(fout):
				continue
			# fit from sweep (vec modes use their own small-H probes)
			sweep_mode = MODES[mode][3]
			pts = []
			for H in (2, 5, 10, 15, 20, 25, 30):
				f = os.path.join(ROOT, 'results', 'sweep_h', sweep_mode, f'{task}_H{H}.json')
				if mode.startswith('vec') or not os.path.exists(f):
					continue
				d = json.load(open(f))
				if d.get('ok', True) and 'process_vram_mib' in d:
					pts.append((H, d['process_vram_mib']))
			probes = []
			if len(pts) < 3:
				for H in (5, 15, 30):
					pr = probe(mode, task, H, outdir); probes.append(pr)
					if pr['fits']:
						pts.append((H, pr['vram']))
			Hs, Vs = np.array([x[0] for x in pts], float), np.array([x[1] for x in pts], float)
			slope, icpt = np.polyfit(Hs, Vs, 1)
			used, total = smi_used_total()
			free = total - used  # the GPU is otherwise idle during this study (desktop processes only)
			ref = json.load(open(os.path.join(ROOT, 'results', 'sweep_h', 'gpu', f'{task}_H5.json')))
			L, B = ref['episode_length'], ref['batch_size']
			# largest horizon the ORIGINAL algorithm can run: at the first update the buffer holds
			# seed_steps/L + 1 episodes with L - H valid (non-zero priority) start indices each, and the
			# batch is drawn without replacement -> need (L - H) * n_eps >= batch_size
			n_eps = SEED_STEPS // L + 1
			Hmax = L - int(np.ceil(B / n_eps))
			Hpred = (free - icpt) / slope  # VRAM(H) = icpt + slope * H reaches the free VRAM
			res = dict(task=task, mode=mode, slope_mib_per_H=slope, intercept_mib=icpt, free_mib=free, total_mib=total,
					   H_max_valid=Hmax, H_oom_predicted=Hpred, probes=probes)
			print(f'{mode} {task}: slope {slope:.1f} MiB/H, intercept {icpt:.0f} MiB, free {free:.0f} -> H* {Hpred:.0f} (H_max {Hmax})', flush=True)
			if Hpred > Hmax:
				pr = probe(mode, task, Hmax, outdir); probes.append(pr)
				res['result'] = 'fits_at_H_max' if pr['fits'] else ('oom_at_H_max' if pr.get('oom') else 'error_at_H_max')
				if not pr['fits'] and pr.get('oom'):  # prediction too optimistic: bisect below H_max
					Hpred = Hmax
			if res.get('result') != 'fits_at_H_max':
				# bracket the boundary: lo = largest H that fits, hi = smallest H that OOMs;
				# start at +-10 % around the prediction, widen if needed, then bisect to ~2 %
				lo, hi = None, None
				for H in (int(0.9 * Hpred), int(np.ceil(1.1 * Hpred))):
					H = int(min(max(H, 2), Hmax))
					pr = probe(mode, task, H, outdir); probes.append(pr)
					if pr['fits']:
						lo = H if lo is None else max(lo, H)
					elif pr.get('oom'):
						hi = H if hi is None else min(hi, H)
				# widen if needed
				while lo is None and probes[-1].get('oom') and probes[-1]['H'] > 2:
					H = max(2, int(probes[-1]['H'] * 0.7)); pr = probe(mode, task, H, outdir); probes.append(pr)
					if pr['fits']: lo = H
				while hi is None and lo is not None and lo < Hmax:
					H = min(Hmax, int(lo * 1.3) + 1); pr = probe(mode, task, H, outdir); probes.append(pr)
					if pr['fits']: lo = H
					elif pr.get('oom'): hi = H
					else: break
				while lo is not None and hi is not None and hi - lo > max(1, int(0.02 * lo)):
					H = (lo + hi) // 2; pr = probe(mode, task, H, outdir); probes.append(pr)
					if pr['fits']: lo = H
					elif pr.get('oom'): hi = H
					else: break
				res['H_fit_max'], res['H_oom_min'] = lo, hi
				res['result'] = 'bracketed' if (lo is not None and hi is not None) else res.get('result', 'incomplete')
			res['probes'] = probes
			json.dump(res, open(fout, 'w'), indent=1)
			for f in os.listdir(outdir):
				if f.startswith('probe_'):
					os.remove(os.path.join(outdir, f))
			print(f'   -> {res["result"]}: fit<= {res.get("H_fit_max")}, oom>= {res.get("H_oom_min")}', flush=True)


if __name__ == '__main__':
	main()
