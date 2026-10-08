"""No-slowdown capacity per task PER HORIZON (same H for all runs).

2 concurrent runs of the same H, at the lightest (H=2) and heaviest (H=30) horizon of the sweep, plus
H=50/100 for walker and dog. If 2 runs already slow each other down at H=2 and H=30, they do for every H in
between (cost is linear in H). Solo baselines are reused from results/interference/gpu_mps/<task>.json
(same harness), so bench/interference_study.py must have run first; missing solo horizons (50, 100) are
measured here. All runs: optimised GPU port under CUDA MPS (bench/multi_run.py), 30 s timed phase.
Results: results/interference_sameH/<task>_H<H>.json (resumable)

  python bench/interference_sameH.py

Run in the `tdmpc-warp` env (started by bench/pipeline_final.sh). Tabulated by
bench/analyze_interference.py (docs/results_capacity.md, section A2); reported in the per-task / per-H
table of docs/04_multi_run_capacity.md (4.4).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
from interference_study import multi, spec, REPS
OUT = os.path.join(ROOT, 'results', 'interference_sameH'); os.makedirs(OUT, exist_ok=True)
tmp = os.path.join(OUT, '_tmp'); os.makedirs(tmp, exist_ok=True)
jobs = [(t, H) for t in REPS for H in (2, 30)] + [(t, H) for t in ('walker-walk', 'dog-run') for H in (50, 100)]
for task, H in jobs:
	f = os.path.join(OUT, f'{task}_H{H}.json')
	if os.path.exists(f):
		continue
	prev = json.load(open(os.path.join(ROOT, 'results', 'interference', 'gpu_mps', f'{task}.json')))
	solo = prev['solo_steps_per_s'].get(str(H))
	if solo is None:
		d = multi('gpu_mps', [spec('gpu_mps', task, H)], 30, tmp)
		solo = [r for r in d['runs'] if r.get('ok')][0]['steps_per_s']
	d = multi('gpu_mps', [spec('gpu_mps', task, H)] * 2, 30, tmp)  # two identical runs side by side
	rel =[r['steps_per_s'] / solo for r in d['runs'] if r.get('ok')]
	res = dict(task=task, horizon=H, solo_steps_per_s=solo, n=2, rel_speed=rel, min_rel=min(rel) if len(rel) == 2 else None,
			   total_steps_per_s=d.get('steps_per_s_total'), no_slowdown_capacity=2 if rel and min(rel) >= 0.95 else 1)
	json.dump(res, open(f, 'w'), indent=1)
	print(f'{task} H={H}: solo {solo:.1f}, 2 runs -> each {[round(x, 2) for x in rel]} of solo -> capacity {res["no_slowdown_capacity"]}', flush=True)
os.rmdir(tmp)
print('same-H interference done', flush=True)
