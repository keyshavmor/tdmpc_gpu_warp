"""Vectorised-mode memory per seed (S = 1 vs S = 8 seeds in one train_vec-style process).

Process VRAM and step time with S = 1 and S = 8 seeds in one process, per domain representative, at
H = 5 and H = 30 (with and without activation checkpointing, ckpt_unroll=true). Each point is one
bench/bench_step.py subprocess (--extra vec_agent=true num_seeds=S pace_host=true, k = 20).
marginal VRAM per seed = (V8 - V1) / 7 ;  predicted max seeds per process = 1 + (free - V1) / marginal.
If the S = 8 probe itself runs out of memory (dog-run at H = 30), no marginal is stored here and bench/analyze_memory.py
derives it from the vectorised packing bisection of bench/packing_study.py instead.
Results: results/vec_memory/<task>_H<H>[_ckpt].json (resumable)

  python bench/vec_memory.py                       # all 12 domain representatives
  python bench/vec_memory.py walker-walk dog-run   # selected tasks

Run in the `tdmpc-warp` env. Tabulated by bench/analyze_memory.py (docs/results_memory.md, "Seeds per
RTX 3090 vs horizon" and "Seeds x horizon settings"); reported in docs/04_multi_run_capacity.md (4.5).

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
REPS = ['acrobot-swingup', 'cartpole-swingup', 'cheetah-run', 'cup-catch', 'dog-run', 'finger-spin', 'fish-swim',
		'hopper-hop', 'humanoid-run', 'quadruped-run', 'reacher-easy', 'walker-walk']
OUT = os.path.join(ROOT, 'results', 'vec_memory')
os.makedirs(OUT, exist_ok=True)


def run(task, H, S, ckpt):
	"""One bench_step.py probe with S seeds in one vectorised process; returns VRAM / speed or the error."""
	f =os.path.join(OUT, '_probe.json')
	ex = ['vec_agent=true', f'num_seeds={S}', 'pace_host=true'] + (['ckpt_unroll=true'] if ckpt else [])
	r = subprocess.run([PY, os.path.join(HERE, 'bench_step.py'), '--mode', 'gpu', '--task', task, '--horizon', str(H),
						'--k', '20', '--out', f, '--extra'] + ex, capture_output=True, text=True, timeout=7200)
	if r.returncode != 0:
		return dict(ok=False, error=r.stderr[-300:])
	d = json.load(open(f)); os.remove(f)
	return dict(ok=True, vram=d['process_vram_mib'], step_ms=d['step_ms'], seed_steps_per_s=d['seed_steps_per_s'],
				torch_reserved=d['torch_peak_reserved_mib'])


for task in (sys.argv[1:] or REPS):
	for H in (5, 30):
		for ckpt in (False, True):
			fout = os.path.join(OUT, f'{task}_H{H}' + ('_ckpt' if ckpt else '') + '.json')
			if os.path.exists(fout):
				continue
			r1, r8 = run(task, H, 1, ckpt), run(task, H, 8, ckpt)
			res = dict(task=task, horizon=H, ckpt=ckpt, S1=r1, S8=r8)
			if r1['ok'] and r8['ok']:
				res['marginal_mib_per_seed'] = (r8['vram'] - r1['vram']) / 7
			json.dump(res, open(fout, 'w'), indent=1)
			print(task, H, 'ckpt' if ckpt else '', r1.get('vram'), r8.get('vram'), res.get('marginal_mib_per_seed'),
				  'step S1 %.1f S8 %.1f ms' % (r1.get('step_ms', -1), r8.get('step_ms', -1)), flush=True)
print('vec memory done', flush=True)
