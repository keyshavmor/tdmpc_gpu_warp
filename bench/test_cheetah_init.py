"""Cheetah episode initialisation: GPU (dmc_tasks.Cheetah.init_episode) vs dm_control on C MuJoCo.

dm_control initialises cheetah inside Physics.reset_context(): Physics.reset() (mj_resetData, actuation-disabled
forward at qpos0), the limited joints drawn uniformly in their ranges, then physics.step(nstep=200) with the legacy
step (mj_step2, 199 mj_step, mj_step1) and time = 0. The test feeds the same joint draws to both implementations and
compares the settled state (qpos, qvel) and the observation after the after-reset forward.

	python bench/test_cheetah_init.py [--samples 5]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (ICML 2022); original code:
https://github.com/nicklashansen/tdmpc (MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dm_control import suite  # noqa: E402

from env_warp import DMCWarpEnv  # noqa: E402


def cpu_settled(u):
	"""dm_control's cheetah initialisation with joint fractions u in [0, 1): settled (qpos, qvel, observation)."""
	env = suite.load('cheetah', 'run', task_kwargs={'random': 0})
	physics = env.physics
	lim = np.nonzero(physics.model.jnt_limited == 1)[0]
	lo, hi = physics.model.jnt_range[lim, 0], physics.model.jnt_range[lim, 1]
	with physics.reset_context():
		physics.data.qpos[lim] = lo + (hi - lo) * u      # Cheetah.initialize_episode's joint draw
		physics.step(nstep=200)                            # its settling, legacy step
		physics.data.time = 0
	obs = np.concatenate([np.asarray(v, float).ravel() for v in env.task.get_observation(physics).values()])
	return physics.data.qpos.copy(), physics.data.qvel.copy(), obs


def main():
	"""Compare GPU and CPU settled states for --samples joint draws; exit 1 if the deviation is not at float32 level."""
	p = argparse.ArgumentParser()
	p.add_argument('--samples', type=int, default=5)
	a = p.parse_args()
	env = DMCWarpEnv('cheetah-run', 0, 4, use_graph=True)
	rng = np.random.default_rng(0)
	worst = 0.0
	for i in range(a.samples):
		u = rng.uniform(size=len(env.task.i_lim))
		env.rand = lambda *shape: torch.tensor(u, dtype=torch.float32, device='cuda').reshape(shape)
		obs = env.reset().double().cpu().numpy()
		q, v, o = cpu_settled(u)
		gq, gv = env.v.qpos[0].double().cpu().numpy(), env.v.qvel[0].double().cpu().numpy()
		err = max(np.abs(gq - q).max(), np.abs(gv - v).max(), np.abs(obs - o).max())
		worst = max(worst, err)
		print(f'sample {i}: max |GPU - CPU| qpos {np.abs(gq - q).max():.2e} qvel {np.abs(gv - v).max():.2e} '
			  f'obs {np.abs(obs - o).max():.2e}', flush=True)
	ok = worst < 1e-3
	print('PASS' if ok else 'FAIL', f'(worst {worst:.2e})')
	sys.exit(0 if ok else 1)


if __name__ == '__main__':
	main()
