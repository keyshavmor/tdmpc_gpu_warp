"""Divergence study: where do the remaining GPU-vs-CPU trajectory differences come from?

Why do contact-rich tasks decorrelate? Divergence from CPU MuJoCo (float64, tol 1e-8) of:
 gpu_default : mujoco_warp (float32, solver tolerance clamped to 1e-6 by mujoco_warp)
 gpu_tol1e-8 : mujoco_warp with the model's original tolerance 1e-8
 cpu_f32round: CPU MuJoCo, state rounded to float32 after every control step
 cpu_tol1e-6 : CPU MuJoCo with tolerance 1e-6
Same initial state, same open-loop random actions. Metric: max |obs - obs_ref| / max |obs_ref|,
evaluated after 1/5/10/20/50 agent steps, median over 3 episodes. If the GPU diverges no faster than
cpu_f32round, the difference is explained by float32 precision + chaos; the tolerance variants show
whether the solver tolerance matters.

Environment: tdmpc-warp (mujoco_warp + dm_control; a CUDA GPU).

Usage: python bench/divergence_study.py task:action_repeat [task:action_repeat ...]
       e.g. python bench/divergence_study.py dog-run:2 walker-walk:2 quadruped-run:4 humanoid-walk:2 hopper-hop:4
Writes results/validation/divergence_study.json (per task: variant -> {check step: median rel. error}).
Results: docs/03_validation.md, section 3.2 (table also in docs/_divergence_table.md).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch, mujoco
from dm_control import suite
from env_warp import DMCWarpEnv
CHK = [1, 5, 10, 20, 50]
def flat(o): return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in o.values()])
res = {}
for task, ar in [(a, int(b)) for a, b in (x.split(':') for x in sys.argv[1:])]:
    # task name -> dm_control (domain, task); the 'cup' domain is called 'ball_in_cup' in dm_control
    dom, tk = task.replace('-', '_').split('_', 1); dom = dict(cup='ball_in_cup').get(dom, dom)
    curves = {k: {c: [] for c in CHK} for k in ['gpu_default', 'gpu_tol1e-8', 'cpu_f32round', 'cpu_tol1e-6']}
    for ep in range(3):
        seed = 10 + ep
        rng = np.random.RandomState(ep)
        acts = rng.uniform(-1, 1, (max(CHK), 64)).astype(np.float32)  # shared open-loop actions (64 >= any action_dim; sliced per task)
        def cpu_run(mod=None):
            """CPU MuJoCo rollout of the shared actions; mod=None reference, 'tol' tolerance 1e-6, 'f32' float32-rounded state."""
            e = suite.load(dom, tk, task_kwargs={'random': seed})
            if mod == 'tol': e.physics.model.opt.tolerance = 1e-6
            e.reset(); lo, hi = e.action_spec().minimum, e.action_spec().maximum
            out = []
            for t in range(max(CHK)):
                a = acts[t, :len(lo)]
                for _ in range(ar):
                    ts = e.step(lo + (hi - lo) * (a + 1) / 2)
                    if mod == 'f32':
                        d = e.physics.data
                        d.qpos[:] = d.qpos.astype(np.float32); d.qvel[:] = d.qvel.astype(np.float32)
                        if d.act.size: d.act[:] = d.act.astype(np.float32)
                out.append(flat(ts.observation))
            return np.stack(out)
        def gpu_run(tol=None):
            """mujoco_warp rollout from the same dm_control initial state (reset_mode='cpu'); tol overrides the solver tolerance."""
            e = DMCWarpEnv(task, seed, ar, reset_mode='cpu')
            if tol is not None: e.m.opt.tolerance.fill_(tol)
            e.reset(); out = []
            for t in range(max(CHK)):
                o, _, _, _ = e.step(torch.from_numpy(acts[t, :e.action_dim]).cuda())
                out.append(o.cpu().numpy().astype(np.float64))
            return np.stack(out)
        ref = cpu_run()
        runs = {'gpu_default': gpu_run(), 'gpu_tol1e-8': gpu_run(1e-8), 'cpu_f32round': cpu_run('f32'), 'cpu_tol1e-6': cpu_run('tol')}
        for k, r in runs.items():
            for c in CHK:
                curves[k][c].append(float(np.abs(r[c-1] - ref[c-1]).max() / (np.abs(ref[c-1]).max() + 1e-9)))
    res[task] = {k: {c: float(np.median(v)) for c, v in cv.items()} for k, cv in curves.items()}
    print(task, json.dumps(res[task]), flush=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results', 'validation'), exist_ok=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'validation', 'divergence_study.json'), 'w'), indent=1)
