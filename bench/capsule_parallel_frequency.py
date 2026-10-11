"""How often do touching capsules come close to parallel? (exposure of each task to the capsule-collision fix)

The float32 segment-segment solve fixed by patches/mujoco_warp/0002-capsule-collision-float32-precision.patch goes
wrong only when two touching capsules are nearly parallel (relative determinant error ~ 6e-8 / angle^2 > 1 % below
about 2.4e-3 rad). This script counts, per task, the fraction of states with a capsule-capsule contact whose axes are
within 3e-3 rad of parallel, evaluated in C MuJoCo:
  * on the replay buffers of the trained GPU runs (hopper-hop, cheetah-run: their observations contain the full state
    except the x position, which does not affect collisions);
  * on random-policy rollouts (all affected domains), as a policy-independent reference.

Environment: tdmpc-warp (CPU only).
Usage: python bench/capsule_parallel_frequency.py
Writes results/validation/capsule_parallel_frequency.json. Results: docs/03_validation.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os
import numpy as np
import torch
import mujoco
from dm_control import suite

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CAP = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
THRESH = 3e-3


def near_parallel(m, d):
	"""True if any capsule-capsule contact of the current state has axes within THRESH rad of parallel."""
	for c in d.contact[:d.ncon]:
		if m.geom_type[c.geom1] == CAP and m.geom_type[c.geom2] == CAP:
			a1 = d.geom_xmat[c.geom1].reshape(3, 3)[:, 2]; a2 = d.geom_xmat[c.geom2].reshape(3, 3)[:, 2]
			if np.arccos(min(1.0, abs(float(a1 @ a2)))) < THRESH:
				return True
	return False


res = {'threshold_rad': THRESH, 'buffers': {}, 'random_policy': {}}
rng = np.random.default_rng(0)
# trained-agent states from the replay buffers (obs = qpos[1:], qvel, ...)
for task, dom, nq in (('hopper-hop', 'hopper', 7), ('cheetah-run', 'cheetah', 9)):
	env = suite.load(dom, task.split('-')[1]); m, d = env.physics.model.ptr, env.physics.data.ptr
	fr = []
	for s in range(1, 6):
		f = os.path.join(ROOT, 'results', 'mps_batches', 'baseline_mps', task, 'state', 'baseline_mps', str(s), 'checkpoint.pt')
		b = torch.load(f, map_location='cpu', weights_only=False)['buffer']
		o = b['_obs'][:b['n']].numpy()
		idx = rng.choice(len(o), min(20000, len(o)), replace=False)
		hit = 0
		for i in idx:
			d.qpos[:] = np.r_[0.0, o[i, :nq - 1]]
			mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
			hit += near_parallel(m, d)
		fr.append(hit / len(idx))
	res['buffers'][task] = {'per_seed': fr, 'mean': float(np.mean(fr))}
	print(task, 'trained-agent states:', res['buffers'][task])
# random-policy rollouts
for dom, task in (('hopper', 'hop'), ('cheetah', 'run'), ('quadruped', 'walk'), ('finger', 'turn_hard'), ('cartpole', 'swingup')):
	env = suite.load(dom, task, task_kwargs={'random': 0}); m, d = env.physics.model.ptr, env.physics.data.ptr
	spec = env.action_spec(); hit = n = 0
	for ep in range(5):
		env.reset()
		for t in range(1000):
			env.step(rng.uniform(spec.minimum, spec.maximum))
			hit += near_parallel(m, d); n += 1
	res['random_policy'][f'{dom}-{task}'] = hit / n
	print(f'{dom}-{task}', 'random policy:', hit / n)
json.dump(res, open(os.path.join(ROOT, 'results', 'validation', 'capsule_parallel_frequency.json'), 'w'), indent=1)
