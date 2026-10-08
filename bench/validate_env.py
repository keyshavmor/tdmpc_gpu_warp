"""Environment validation: mujoco_warp DMControl port vs dm_control (CPU MuJoCo), all TD-MPC tasks.

Validates the GPU environment (src/env_warp.py) against dm_control for every task in tasks.txt:

  A  function check : GPU obs/reward vs dm_control obs/reward evaluated on the *same* state
                      (GPU state copied into CPU physics; isolates the obs/reward port from physics)
  B  reset check    : distribution of initial observations, 100 GPU resets vs 100 dm_control resets
                      (max |z| of the per-dimension mean difference, min/max std ratio)
  C  trajectory     : GPU vs CPU open-loop rollouts from the identical initial state & actions
                      (reset_mode='cpu'), relative obs error after 1/10/50 agent steps, next to a
                      CPU-vs-CPU chaos control (1e-6 perturbation of the initial velocity), plus
                      random-policy returns of 3 episodes on both simulators
  D  capacity       : max contacts / constraint rows used vs the allocated GPU capacity
                      (nconmax / njmax)
  E  timing         : env step time (agent step = action_repeat control steps) and reset time

Environment: tdmpc-warp (needs both mujoco_warp and dm_control; a CUDA GPU).

Usage: python bench/validate_env.py [task ...]   (no args = all tasks in tasks.txt)
Writes results/validation/<task>.json (one line of JSON per task is also printed);
bench/summarize_validation.py prints a one-line summary per task.

Results: docs/03_validation.md, section 3.1 (A-D, table also in docs/_validation_table.md);
timing E in docs/_env_timing_table.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
import mujoco
from omegaconf import OmegaConf
from dm_control import suite
from env_warp import DMCWarpEnv

ROOT = os.path.join(os.path.dirname(__file__), '..')


def action_repeat_of(task):
	"""Return the task's action_repeat from cfgs/tasks/<domain>.yaml (falls back to default.yaml)."""
	dom = task.split('-')[0]
	p = os.path.join(ROOT, 'cfgs', 'tasks', f'{dom}.yaml')
	if not os.path.exists(p):
		p = os.path.join(ROOT, 'cfgs', 'tasks', 'default.yaml')
	return int(OmegaConf.load(p).action_repeat)


def flat(obs):
	"""Flatten a dm_control observation dict into one float64 vector (dict order = TD-MPC's order)."""
	return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in obs.values()])


def cpu_eval_on_gpu_state(env, physics, task):
	"""Copy GPU state (world 0) into CPU physics and evaluate dm_control's observation & reward.

	Used by check A: both sides see exactly the same state, so any difference comes from the
	obs/reward port, not from the physics. Returns (flat observation, reward).
	"""
	d, v = physics.data, env.v
	d.qpos[:] = v.qpos[0].cpu().numpy(); d.qvel[:] = v.qvel[0].cpu().numpy()
	if d.act.size: d.act[:] = v.act[0].cpu().numpy()
	d.ctrl[:] = v.ctrl[0].cpu().numpy()
	mujoco.mj_forward(physics.model.ptr, d.ptr)
	sd = v.sensordata[0].cpu().numpy()
	acc = env._acc_idx.numpy()
	d.sensordata[acc] = sd[acc]  # acc-stage sensors come from the last sub-step (as in dm_control)
	return flat(task.get_observation(physics)), float(task.get_reward(physics))


def main(tasks):
	"""Run checks A-E for every task and write results/validation/<task>.json."""
	os.makedirs(os.path.join(ROOT, 'results', 'validation'), exist_ok=True)
	for task in tasks:
		t0 = time.time()
		ar = action_repeat_of(task)
		out = {'task': task, 'action_repeat': ar}
		rng = np.random.RandomState(0)

		# ---- A: function check (action_repeat=1 so each reward is one control step) ----------
		env = DMCWarpEnv(task, seed=1, action_repeat=1, reset_mode='cpu')
		physics, dtask = env.cpu_env.physics, env.cpu_env.task
		obs_err, rew_err, obs_scale = [], [], []
		max_con = max_efc = 0
		for ep in range(2):
			env.reset()
			for t in range(300):
				a = torch.tensor(rng.uniform(-1, 1, env.action_dim), dtype=torch.float32, device='cuda')
				o, r, done, _ = env.step(a)
				if t % 3 == 0:
					oc, rc = cpu_eval_on_gpu_state(env, physics, dtask)
					og = o.cpu().numpy().astype(np.float64)
					obs_err.append(np.abs(og - oc).max()); obs_scale.append(np.abs(oc).max())
					rew_err.append(abs(float(r) - rc))
					nc, ne, _, _ = env.overflow(); max_con = max(max_con, nc); max_efc = max(max_efc, ne)
		_, _, nconmax, njmax = env.overflow()
		out['A_obs_max_abs_err'] = float(np.max(obs_err))
		out['A_obs_median_abs_err'] = float(np.median(obs_err))
		out['A_reward_max_abs_err'] = float(np.max(rew_err))
		out['D_max_contacts'], out['D_nconmax'] = max_con, nconmax
		out['D_max_efc_rows'], out['D_njmax'] = max_efc, njmax
		del env

		# ---- B: reset distribution (GPU sampler vs dm_control) ------------------------------
		n_reset = 100
		envg = DMCWarpEnv(task, seed=2, action_repeat=ar, reset_mode='gpu')
		g = np.stack([envg.reset().cpu().numpy() for _ in range(n_reset)]).astype(np.float64)
		cenv = suite.load(*envg.cpu_env_key if hasattr(envg, 'cpu_env_key') else (envg.domain, envg.task_name),
						  task_kwargs={'random': 3})
		c = np.stack([flat(cenv.reset().observation) for _ in range(n_reset)])
		pooled = np.sqrt(0.5 * (g.var(0) + c.var(0)))
		live = pooled > 1e-3 * (np.abs(c).max() + 1e-6)  # ignore (near-)constant dimensions
		z_mean = np.abs(g.mean(0) - c.mean(0))[live] / (pooled[live] / np.sqrt(n_reset / 2))
		std_ratio = g.std(0)[live] / c.std(0)[live]
		out['B_reset_n_live_dims'] = int(live.sum())
		out['B_reset_mean_z_max'] = float(z_mean.max())       # |mean diff| in standard errors
		out['B_reset_mean_z_median'] = float(np.median(z_mean))
		out['B_reset_std_ratio_minmax'] = [float(std_ratio.min()), float(std_ratio.max())]

		# ---- C: open-loop trajectory agreement from identical initial state -----------------
		envc = DMCWarpEnv(task, seed=4, action_repeat=ar, reset_mode='cpu')
		ref = suite.load(envc.domain, envc.task_name, task_kwargs={'random': 4})
		lo, hi = ref.action_spec().minimum, ref.action_spec().maximum
		rets_g, rets_c, err_at = [], [], {1: [], 10: [], 50: []}
		# chaos control: CPU vs CPU with a float32-sized perturbation of the initial velocity
		ref2 = suite.load(envc.domain, envc.task_name, task_kwargs={'random': 4})
		err_self = {1: [], 10: [], 50: []}
		for ep in range(3):
			og = envc.reset().cpu().numpy(); oc = flat(ref.reset().observation)
			ref2.reset()
			with ref2.physics.reset_context():
				ref2.physics.data.qpos[:] = ref.physics.data.qpos; ref2.physics.data.act[:] = ref.physics.data.act
				ref2.physics.data.qvel[:] = ref.physics.data.qvel + 1e-6 * rng.randn(ref.physics.model.nv)
			out.setdefault('C_initial_obs_max_abs_err', 0.0)
			out['C_initial_obs_max_abs_err'] = max(out['C_initial_obs_max_abs_err'], float(np.abs(og - oc).max()))
			Rg = Rc = 0.0
			for t in range(envc.ep_len):
				a = rng.uniform(-1, 1, envc.action_dim).astype(np.float32)
				og, rg, done, _ = envc.step(torch.from_numpy(a).cuda())
				rc = 0.0
				# CPU side: repeat the action ar times, rescaled from [-1, 1] to the actuator range
				for _ in range(ar):
					ts = ref.step(lo + (hi - lo) * (a + 1) / 2)
					ts2 = ref2.step(lo + (hi - lo) * (a + 1) / 2)
					rc += ts.reward or 0.0
				Rg += float(rg); Rc += rc
				if t + 1 in err_at:
					oc = flat(ts.observation)
					err_at[t + 1].append(float(np.abs(og.cpu().numpy() - oc).max() / (np.abs(oc).max() + 1e-6)))
					err_self[t + 1].append(float(np.abs(flat(ts2.observation) - oc).max() / (np.abs(oc).max() + 1e-6)))
			rets_g.append(Rg); rets_c.append(Rc)
		out['C_rel_obs_err_after_steps'] = {k: float(np.max(v)) for k, v in err_at.items()}
		out['C_cpu_chaos_rel_err_after_steps'] = {k: float(np.max(v)) for k, v in err_self.items()}
		out['C_random_policy_return_gpu'] = rets_g
		out['C_random_policy_return_cpu'] = rets_c

		# ---- E: timing ------------------------------------------------------------------------
		# zero actions, 20 warm-up steps, then the mean of 300 agent steps (GPU) vs 300 x ar control steps (CPU)
		envg.reset(); a = torch.zeros(envg.action_dim, device='cuda')
		for _ in range(20): envg.step(a)
		torch.cuda.synchronize(); t1 = time.perf_counter(); n = 300
		for _ in range(n): o, r, d_, _ = envg.step(a)
		torch.cuda.synchronize(); out['E_gpu_env_step_ms'] = (time.perf_counter() - t1) / n * 1e3
		torch.cuda.synchronize(); t1 = time.perf_counter()
		for _ in range(20): envg.reset()
		torch.cuda.synchronize(); out['E_gpu_reset_ms'] = (time.perf_counter() - t1) / 20 * 1e3
		cenv.reset(); z = np.zeros(cenv.action_spec().shape)
		t1 = time.perf_counter()
		for _ in range(n):
			for _ in range(ar): cenv.step(z)
		out['E_cpu_env_step_ms'] = (time.perf_counter() - t1) / n * 1e3
		out['wall_s'] = time.time() - t0
		json.dump(out, open(os.path.join(ROOT, 'results', 'validation', f'{task}.json'), 'w'), indent=1)
		print(json.dumps(out), flush=True)
		del envg, envc


if __name__ == '__main__':
	tasks = sys.argv[1:] or [l.strip() for l in open(os.path.join(ROOT, 'tasks.txt')) if l.strip()]
	main(tasks)
