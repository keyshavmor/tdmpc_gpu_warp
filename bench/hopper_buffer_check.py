"""hopper-hop: GPU env vs dm_control on states the trained agent actually visited (touch sensors, hop reward).

bench/validate_env.py checks the observation / reward port on random-policy states. For hopper-hop those never
earn reward (the random policy does not hop) and the touch sensors are acceleration-stage sensors that check A
copies from the GPU, so neither the hop reward nor the touch observation is tested there. This check closes that
gap on the states of a trained run: hopper's observation (qpos[1:], qvel, log1p(touch)) is its full physical state
except the x position, which does not affect the dynamics (slide joint on a flat floor), so every replay-buffer
transition can be replayed exactly. For each selected transition (the highest-reward ones and a random sample) both
simulators are set to the stored state, the stored action is applied for one agent step (action_repeat control
steps) and the next observation and the summed reward are compared:
  * GPU env (DMCWarpEnv, CUDA graph) vs dm_control (CPU MuJoCo, legacy Physics.step semantics);
  * chaos control: dm_control vs dm_control with a 1e-6 perturbation of the velocity (how much a float32-sized
    difference grows within one agent step on these contact states);
  * sanity: GPU vs the next observation / reward stored in the buffer (state reconstruction).
Writes results/validation/hopper_buffer_check.json. Results: docs/03_validation.md.

Environment: tdmpc-warp (CUDA GPU; do not run it next to MPS clients - docs/USER_GUIDE.md section 4.4).
Usage: python bench/hopper_buffer_check.py [checkpoint.pt] [--n 400]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, sys
import numpy as np
import torch
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from env_warp import DMCWarpEnv

p = argparse.ArgumentParser()
p.add_argument('checkpoint', nargs='?', default=os.path.join(
	ROOT, 'results', 'mps_batches', 'baseline_mps', 'hopper-hop', 'state', 'baseline_mps', '4', 'checkpoint.pt'))
p.add_argument('--n', type=int, default=400, help='transitions per group (top reward, random)')
a = p.parse_args()

AR, L = 4, 250  # hopper: action_repeat 4 (cfgs/tasks/default.yaml), 1000 // 4 agent steps per episode
b = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['buffer']
n = b['n']
obs, act, rew = b['_obs'][:n].numpy(), b['_action'][:n].numpy(), b['_reward'][:n].numpy()
inner = np.array([i for i in range(n) if i % L != L - 1])  # next observation stored in _obs (not at the episode end)
rng = np.random.RandomState(0)
top = inner[np.argsort(-rew[inner])[:a.n]]
rand = rng.choice(inner, a.n, replace=False)

env = DMCWarpEnv('hopper-hop', seed=1, action_repeat=AR, reset_mode='cpu')
env.reset()
cenv = env.cpu_env
physics, task = cenv.physics, cenv.task
nsub = cenv._n_sub_steps


def flat(o):
	"""dm_control observation dict -> flat float64 vector (TD-MPC's order)."""
	return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in o.values()])


def set_cpu(qpos, qvel):
	"""Put the state into the CPU physics (zero warm start, as on the GPU side) and recompute derived quantities."""
	d = physics.data
	d.qpos[:] = qpos; d.qvel[:] = qvel; d.qacc_warmstart[:] = 0; d.ctrl[:] = 0; d.time = 0
	mujoco.mj_forward(physics.model.ptr, d.ptr)


def cpu_step(action):
	"""One agent step on the CPU: action_repeat x (set control, legacy Physics.step(n_sub_steps)); summed reward."""
	r = 0.0
	for _ in range(AR):
		task.before_step(action.astype(np.float64), physics)
		physics.step(nsub)
		r += task.get_reward(physics) or 0.0
	return flat(task.get_observation(physics)), r


def gpu_step(qpos, qvel, action):
	"""One agent step on the GPU env from the given state (zero warm start)."""
	v = env.v
	v.qpos[0].copy_(torch.as_tensor(qpos, dtype=torch.float32)); v.qvel[0].copy_(torch.as_tensor(qvel, dtype=torch.float32))
	v.qacc_warmstart.zero_(); v.ctrl.zero_(); v.time.zero_()
	env._run('forward', env._forward_noact)
	o, r, _, _ = env.step(torch.as_tensor(action, device='cuda'))
	return o.cpu().numpy().astype(np.float64), float(r)


SL = {'position': slice(0, 6), 'velocity': slice(6, 13), 'touch (log1p)': slice(13, 15)}
res = {'checkpoint': os.path.relpath(a.checkpoint, ROOT), 'buffer_transitions': int(n), 'groups': {}}
for name, idx in (('top_reward', top), ('random', rand)):
	rows = []
	for i in idx:
		qpos, qvel = np.r_[0.0, obs[i, :6]].astype(np.float64), obs[i, 6:13].astype(np.float64)
		og, rg = gpu_step(qpos, qvel, act[i])
		set_cpu(qpos, qvel); oc, rc = cpu_step(act[i])
		set_cpu(qpos, qvel + 1e-6 * rng.randn(7)); oc2, rc2 = cpu_step(act[i])
		on = obs[i + 1].astype(np.float64)
		rows.append(dict(
			**{f'gpu_cpu {k}': float(np.abs(og[s] - oc[s]).max()) for k, s in SL.items()},
			**{f'chaos {k}': float(np.abs(oc2[s] - oc[s]).max()) for k, s in SL.items()},
			gpu_cpu_reward=abs(rg - rc), chaos_reward=abs(rc2 - rc), gpu_buffer_obs=float(np.abs(og - on).max()),
			gpu_buffer_reward=abs(rg - float(rew[i])), reward_cpu=rc, touch_cpu=float(np.abs(oc[13:15]).max())))
	g = {k: {'median': float(np.median([r[k] for r in rows])), 'p95': float(np.percentile([r[k] for r in rows], 95)),
			 'max': float(np.max([r[k] for r in rows]))} for k in rows[0] if k not in ('reward_cpu', 'touch_cpu')}
	g['mean_reward_per_agent_step_cpu'] = float(np.mean([r['reward_cpu'] for r in rows]))
	g['fraction_with_touch'] = float(np.mean([r['touch_cpu'] > 0 for r in rows]))
	g['n'] = len(rows)
	res['groups'][name] = g
	print(name, json.dumps(g, indent=1))
os.makedirs(os.path.join(ROOT, 'results', 'validation'), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, 'results', 'validation', 'hopper_buffer_check.json'), 'w'), indent=1)
