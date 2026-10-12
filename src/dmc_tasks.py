"""GPU (torch) ports of the DMControl tasks used by TD-MPC, batched over S simulated worlds.

Re-implements the observation, reward and episode-initialisation functions of the 12 DMC domains
used by TD-MPC, batched over S simulated worlds (one world per seed). Used by env_warp.DMCWarpEnv.

Each task mirrors dm_control/suite/<domain>.py line by line (same quantities, same order of the
observation OrderedDict, same `rewards.tolerance` shaping, same initial-state distribution) but reads
the simulator state from zero-copy torch views of mujoco_warp's device arrays, so no data leaves the
GPU. All views carry a leading world dimension S: e.g. v.qpos is (S, nq), v.xmat is (S, nbody, 3, 3);
observation() returns (S, obs_dim) and reward() returns (S,). With S == 1 this is exactly the
single-environment port.

Conventions: dm_control `named.data.xmat[body, 'ab']` -> xmat[..., body, a, b] with a,b in {x,y,z};
`v.dot(frame)` with a row-major 3x3 frame -> `v @ frame`.

Graph capture: observation() and reward() are captured in the env's CUDA graph, so they must not sync
with the host or allocate tensors that the graph later reads; every index tensor is created once in
__init__ and kept as an attribute (it must stay alive as long as the env's graphs).

The observation / reward / initialize_episode functions are ports of the DeepMind Control Suite task
code (dm_control, https://github.com/google-deepmind/dm_control, Apache License 2.0,
Copyright DeepMind Technologies).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import math
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

_INF = float('inf')
_DEFAULT_VALUE_AT_MARGIN = 0.1
_AX = {'x': 0, 'y': 1, 'z': 2}


def _idx(lst):
	"""List of ints -> CUDA long tensor (index tensor for gathers inside captured graphs)."""
	return torch.tensor(lst, dtype=torch.long, device='cuda')


# --------------------------------------------------------------------------------------------
# dm_control.utils.rewards.tolerance (torch port, elementwise)
# --------------------------------------------------------------------------------------------
def _sigmoids(x, value_at_1, sigmoid):
	"""Port of dm_control.utils.rewards._sigmoids: maps distance x >= 0 to (0, 1], equal to value_at_1 at x = 1.

	`value_at_1` and `sigmoid` are Python scalars/strings, so the scale is computed on the host once per
	call (no device sync); x is a torch tensor.
	"""
	if sigmoid == 'gaussian':
		scale = math.sqrt(-2 * math.log(value_at_1))
		return torch.exp(-0.5 * (x * scale) ** 2)
	elif sigmoid == 'hyperbolic':
		scale = math.acosh(1 / value_at_1)
		return 1 / torch.cosh(x * scale)
	elif sigmoid == 'long_tail':
		scale = math.sqrt(1 / value_at_1 - 1)
		return 1 / ((x * scale) ** 2 + 1)
	elif sigmoid == 'reciprocal':
		scale = 1 / value_at_1 - 1
		return 1 / (x.abs() * scale + 1)
	elif sigmoid == 'cosine':
		scale = math.acos(2 * value_at_1 - 1) / math.pi
		scaled_x = x * scale
		return torch.where(scaled_x.abs() < 1, (1 + torch.cos(math.pi * scaled_x)) / 2, torch.zeros_like(x))
	elif sigmoid == 'linear':
		scale = 1 - value_at_1
		scaled_x = x * scale
		return torch.where(scaled_x.abs() < 1, 1 - scaled_x, torch.zeros_like(x))
	elif sigmoid == 'quadratic':
		scale = math.sqrt(1 - value_at_1)
		scaled_x = x * scale
		return torch.where(scaled_x.abs() < 1, 1 - scaled_x ** 2, torch.zeros_like(x))
	elif sigmoid == 'tanh_squared':
		scale = math.atanh(math.sqrt(1 - value_at_1))
		return 1 - torch.tanh(x * scale) ** 2
	raise ValueError(f'Unknown sigmoid type {sigmoid!r}.')


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
	"""Same semantics as dm_control.utils.rewards.tolerance; x is a torch tensor (elementwise).

	Returns 1 inside [lower, upper]; outside, the sigmoid of the distance to the nearest bound divided
	by `margin` (value_at_margin at distance == margin). margin == 0 gives a 0/1 indicator.
	"""
	lower, upper = bounds
	in_bounds = (lower <= x) & (x <= upper)
	if margin == 0:
		return in_bounds.to(x.dtype)
	d = torch.where(x < lower, lower - x, x - upper) / margin  # only used where x is out of bounds
	return torch.where(in_bounds, torch.ones_like(x), _sigmoids(d, value_at_margin, sigmoid))


# --------------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------------
class Task:
	"""Observation/reward/initialisation on GPU for one dm_control task, batched over S worlds.

	Subclasses resolve all names to indices in __init__ (on the host, from the compiled MjModel) and
	implement observation() -> (S, obs_dim), reward() -> (S,) and init_episode(env). `v` is the env's
	_Views object (zero-copy torch views of the mujoco_warp state).
	"""
	# model fields that dm_control's initialize_episode modifies per episode (used by reset_mode='cpu')
	model_sync = ()
	# Model fields that differ per world (per-seed episode targets) -> put_model(batch_sizes=...)
	per_world_model_fields = ()
	# world-attached geoms whose geom_pos is edited per episode: mujoco_warp computes the pose of such
	# static geoms only once in make_data, so the env refreshes geom_xpos for them after each reset
	moved_static_geoms = ()
	# tasks whose initialisation uses rejection sampling ("resample until no contact")
	uses_rejection = False

	def __init__(self, task_name, physics, v):
		self.task_name = task_name
		self.mjm = physics.model.ptr
		self.v = v

	# name helpers ------------------------------------------------------------------------------
	def _id(self, obj, name):
		"""MuJoCo id of the object `name` of type `obj` ('body', 'geom', ...); KeyError if missing."""
		i = mujoco.mj_name2id(self.mjm, getattr(mujoco.mjtObj, 'mjOBJ_' + obj.upper()), name)
		if i < 0:
			raise KeyError(f'{obj} {name!r} not found')
		return i

	# name -> id helpers (like dm_control's named indexing)
	def body(self, name):
		"""MuJoCo id of the body `name`."""
		return self._id('body', name)

	def geom(self, name):
		"""MuJoCo id of the geom `name`."""
		return self._id('geom', name)

	def site(self, name):
		"""MuJoCo id of the site `name`."""
		return self._id('site', name)

	def sensor_slice(self, name):
		"""Slice of sensor `name` in sensordata."""
		i = self._id('sensor', name)
		adr, dim = int(self.mjm.sensor_adr[i]), int(self.mjm.sensor_dim[i])
		return slice(adr, adr + dim)

	def sensor_idx(self, names):
		"""Concatenated sensordata indices of the given sensors (in order) as a CUDA index tensor."""
		idx = []
		for n in names:
			s = self.sensor_slice(n)
			idx.extend(range(s.start, s.stop))
		return _idx(idx)

	def jqpos(self, name):
		"""qpos address of joint `name`."""
		return int(self.mjm.jnt_qposadr[self._id('joint', name)])

	def joint_qpos_idx(self, names):
		"""qpos addresses of the given (scalar) joints as a CUDA index tensor."""
		return _idx([self.jqpos(n) for n in names])

	@staticmethod
	def mat(xmat, ab):
		"""Element 'ab' (e.g. 'zz') of a (..., 3, 3) rotation matrix, like dm_control's xmat[..., 'ab']."""
		return xmat[..., _AX[ab[0]], _AX[ab[1]]]

	def observation(self):
		"""Flat observation (S, obs_dim): the task's OrderedDict concatenated in dm_control's order."""
		raise NotImplementedError

	def reward(self):
		"""Per-world reward (S,) of the current state (one control step; the env sums over action repeat)."""
		raise NotImplementedError

	# GPU episode initialisation (port of `initialize_episode`) -----------------------------------
	# Called by DMCWarpEnv.reset() for all S worlds after the state was reset to mj_resetData
	# defaults (qpos0, zero velocities/activations/controls). `env` provides the seeded CUDA
	# generator helpers env.rand/randn/uniform(*shape), env.randomize_joints(S, K) -> (S, K, nq)
	# candidates and env.reject(cands (S, K, nq)) -> picks the first non-penetrating candidate.
	def init_episode(self, env):
		"""Write the randomised initial state of all worlds into the GPU state (port of initialize_episode)."""
		raise NotImplementedError

	def resample(self, env, K):
		"""Fresh rejection-sampling candidates (S, K, nq) (used only if a whole batch was penetrating)."""
		return env.randomize_joints(env.nworld, K)


# --------------------------------------------------------------------------------------------
# Domains
# --------------------------------------------------------------------------------------------
class Acrobot(Task):
	"""dm_control suite/acrobot.py (swingup, swingup_sparse)."""
	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.sparse = task_name == 'swingup_sparse'
		self.arms = _idx([self.body('upper_arm'), self.body('lower_arm')])
		self.s_target, self.s_tip = self.site('target'), self.site('tip')
		self.target_radius = float(self.mjm.site_size[self.s_target, 0])
		self.i_q = _idx([self.jqpos('shoulder'), self.jqpos('elbow')])

	def init_episode(self, env):
		"""Shoulder and elbow angles uniform in [-pi, pi)."""
		self.v.qpos[:, self.i_q] = env.uniform(-math.pi, math.pi, env.nworld, 2)

	def observation(self):
		"""Port of dm_control.suite.acrobot.Balance.get_observation -> (S, 6): orientations = xmat 'xz' and 'zz' of both arms, then velocity = qvel."""
		v = self.v
		xm = v.xmat[:, self.arms]
		return torch.cat([self.mat(xm, 'xz'), self.mat(xm, 'zz'), v.qvel], -1)  # orientations (horizontal, vertical), velocity

	def reward(self):
		"""Port of dm_control.suite.acrobot.Balance._get_reward -> (S,): tolerance of the tip-target distance, bounds (0, target radius), margin 1 (0 for swingup_sparse)."""
		v = self.v
		dist = torch.linalg.norm(v.site_xpos[:, self.s_target] - v.site_xpos[:, self.s_tip], dim=-1)
		return tolerance(dist, bounds=(0, self.target_radius), margin=0 if self.sparse else 1)


class Cartpole(Task):
	"""dm_control suite/cartpole.py (swingup, swingup_sparse; generic in the number of poles)."""
	_CART_RANGE = (-.25, .25)
	_ANGLE_COSINE_RANGE = (.995, 1)

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.sparse = task_name.endswith('sparse')
		self.slider = self.jqpos('slider')
		self.hinge = self.jqpos('hinge_1')

	def init_episode(self, env):  # swing_up=True for both swingup tasks
		"""Port of dm_control.suite.cartpole.Balance.initialize_episode with swing_up=True.

		Per world: slider ~ 0.01 N(0, 1), hinge_1 ~ pi + 0.01 N(0, 1) (pole pointing down), qpos[2:] ~ 0.1 N(0, 1)
		(only with more than one pole), all velocities ~ 0.01 N(0, 1).
		"""
		v, nv, S = self.v, self.mjm.nv, env.nworld
		v.qpos[:, self.slider] = .01 * env.randn(S)
		v.qpos[:, self.hinge] = math.pi + .01 * env.randn(S)
		if nv > 2:
			v.qpos[:, 2:] = .1 * env.randn(S, nv - 2)
		v.qvel.copy_(0.01 * env.randn(S, nv))

	def observation(self):
		"""Port of dm_control.suite.cartpole.Balance.get_observation -> (S, 1 + 2 * n_poles + nv).

		Layout: position = [cart x, then per pole (xmat 'zz', 'xz') = (cos, sin) of the pole angle]
		(Physics.bounded_position), followed by velocity = qvel.
		"""
		v = self.v
		xm = v.xmat[:, 2:]  # pole bodies (body 0 = world, body 1 = cart)
		pos = torch.cat([v.qpos[:, self.slider:self.slider + 1],
						 torch.stack([self.mat(xm, 'zz'), self.mat(xm, 'xz')], -1).flatten(1)], -1)
		return torch.cat([pos, v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.cartpole.Balance._get_reward -> (S,).

		Sparse: cart x within (-0.25, 0.25) times all pole cosines within (0.995, 1) (0/1 indicators).
		Smooth: upright = mean over poles of (cos + 1) / 2, times centered = (1 + tolerance(x, margin=2)) / 2,
		times small_control = (4 + quadratic tolerance of the control) / 5, times
		small_velocity = (1 + min over pole angular velocities of tolerance(margin=5)) / 2.
		"""
		v = self.v
		cart_position = v.qpos[:, self.slider]
		pole_angle_cosine = self.mat(v.xmat[:, 2:], 'zz')
		if self.sparse:
			cart_in_bounds = tolerance(cart_position, self._CART_RANGE)
			angle_in_bounds = tolerance(pole_angle_cosine, self._ANGLE_COSINE_RANGE).prod(-1)
			return cart_in_bounds * angle_in_bounds
		upright = (pole_angle_cosine + 1) / 2
		centered = tolerance(cart_position, margin=2)
		centered = (1 + centered) / 2
		small_control = tolerance(v.ctrl, margin=1, value_at_margin=0, sigmoid='quadratic')[:, 0]
		small_control = (4 + small_control) / 5
		small_velocity = tolerance(v.qvel[:, 1:], margin=5).min(-1).values
		small_velocity = (1 + small_velocity) / 2
		return upright.mean(-1) * small_control * small_velocity * centered


class Cheetah(Task):
	"""dm_control suite/cheetah.py (run)."""
	_RUN_SPEED = 10

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		# torso_subtreelinvel sensor, x component = forward speed
		self.s_speed = self.sensor_slice('torso_subtreelinvel').start
		m = self.mjm
		assert m.nq == m.njnt
		lim = np.nonzero(m.jnt_limited == 1)[0]
		self.i_lim = _idx(lim.tolist())
		self.lim_lo = torch.tensor(m.jnt_range[lim, 0], dtype=torch.float32, device='cuda')
		self.lim_hi = torch.tensor(m.jnt_range[lim, 1], dtype=torch.float32, device='cuda')

	def init_episode(self, env):
		"""Port of dm_control.suite.cheetah.Cheetah.initialize_episode.

		Exactly dm_control's sequence: Physics.reset() (mj_resetData, then an actuation-disabled forward at qpos0),
		limited joints uniform in their joint range (per world), then physics.step(nstep=200) = mj_step2 (it
		integrates the accelerations of the qpos0 forward with the new joint positions), 199 mj_step and mj_step1,
		and the simulation time is reset to 0 (validated against C MuJoCo: bench/test_cheetah_init.py).
		"""
		# Physics.reset(): forward at qpos0 with actuation disabled (the state is at the mj_resetData defaults here)
		env._run('forward', env._forward_noact)
		# limited joints uniform in their range (cheetah's own loop over jnt_limited joints)
		self.v.qpos[:, self.i_lim] = self.lim_lo + (self.lim_hi - self.lim_lo) * env.rand(env.nworld, len(self.i_lim))
		# physics.step(nstep=200), dm_control's legacy step: mj_step2 (not a no-op: it integrates the qpos0
		# accelerations, e.g. the free fall of rootz), then 199 regular steps and mj_step1 (env.settle)
		mjw.step2(env.m, env.d)
		env.settle(199)
		self.v.time.zero_()  # dm_control resets physics.data.time to 0 after settling

	def observation(self):
		"""Port of dm_control.suite.cheetah.Cheetah.get_observation -> (S, nq - 1 + nv): position = qpos[1:] (without root x), velocity = qvel."""
		v = self.v
		return torch.cat([v.qpos[:, 1:], v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.cheetah.Cheetah.get_reward -> (S,): linear tolerance of the forward speed, bounds (10, inf), margin 10, value 0 at the margin."""
		return tolerance(self.v.sensordata[:, self.s_speed], bounds=(self._RUN_SPEED, _INF), margin=self._RUN_SPEED,
						 value_at_margin=0, sigmoid='linear')


class BallInCup(Task):
	"""dm_control suite/ball_in_cup.py (catch); ball position rejection-sampled until no contact."""
	uses_rejection = True

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.s_target, self.b_ball, self.g_ball = self.site('target'), self.body('ball'), self.geom('ball')
		self.xz = _idx([0, 2])  # tensors used in captured graphs must stay alive with the env
		self.target_size = torch.tensor(self.mjm.site_size[self.s_target, [0, 2]], dtype=torch.float32, device='cuda')
		self.ball_size = float(self.mjm.geom_size[self.g_ball, 0])
		self.q_x, self.q_z = self.jqpos('ball_x'), self.jqpos('ball_z')

	def resample(self, env, K):
		"""K candidates per world: ball x ~ U(-.2, .2), z ~ U(.2, .5), rest at qpos0 (cup at rest)."""
		cands = env.qpos0.repeat(env.nworld, K, 1)
		cands[..., self.q_x] = env.uniform(-.2, .2, env.nworld, K)
		cands[..., self.q_z] = env.uniform(.2, .5, env.nworld, K)
		return cands

	def init_episode(self, env):
		"""GPU port of the task's dm_control initialize_episode for all S worlds (see Task.init_episode)."""
		env.reject(self.resample(env, env.n_candidates))

	def observation(self):
		"""Flat observation (S, obs_dim) in dm_control's order (see Task.observation)."""
		return torch.cat([self.v.qpos, self.v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.ball_in_cup.BallInCup.get_reward (Physics.in_target) -> (S,).

		Sparse 0/1: 1 if |target - ball| < target half-size - ball radius in both x and z.
		"""
		v = self.v
		# in_target: ball inside the target box in x and z (sparse 0/1 reward)
		ball_to_target = (v.site_xpos[:, self.s_target][:, self.xz] - v.xpos[:, self.b_ball][:, self.xz]).abs()
		return (ball_to_target < self.target_size - self.ball_size).all(-1).to(torch.float32)


class Finger(Task):
	"""dm_control suite/finger.py (spin, turn_easy, turn_hard); joints rejection-sampled until no contact.

	The task-specific model edits that are constant across episodes (spin: hinge damping; turn: target
	size) are already compiled into the GPU model by the env's first CPU reset; only the turn target's
	position changes per episode (per world).
	"""
	_SPIN_VELOCITY = 15.0
	model_sync = ('site_pos', 'site_size', 'dof_damping')
	per_world_model_fields = ('site_pos',)  # finger-turn target site (unused but harmless for spin)
	uses_rejection = True

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.turn = task_name.startswith('turn')
		sl = self.sensor_slice
		self.i_touch = self.sensor_idx(['touchtop', 'touchbottom'])
		self.i_hinge_vel = sl('hinge_velocity').start
		self.i_tip = sl('tip').start
		self.i_spinner = sl('spinner').start
		self.i_target = sl('target').start
		self.i_bounded = self.sensor_idx(['proximal', 'distal'])
		self.i_vel = self.sensor_idx(['proximal_velocity', 'distal_velocity', 'hinge_velocity'])
		self.s_target = self.site('target')
		m, d = physics.model.ptr, physics.data.ptr
		# hinge anchor (x, z) and the spinner cap radius: the turn target is placed on this circle
		self.hinge_xz = (float(d.xanchor[self._id('joint', 'hinge'), 0]), float(d.xanchor[self._id('joint', 'hinge'), 2]))
		self.cap_radius = float(m.geom_size[self.geom('cap1')].sum())

	def init_episode(self, env):
		"""Port of dm_control.suite.finger Spin/Turn.initialize_episode.

		Turn tasks: the target site is placed at a uniform random angle on the circle of radius cap_radius around
		the hinge anchor (per-world site_pos, written in place). All tasks: joint angles from
		randomize_limited_and_rotational_joints, rejection-sampled until contact-free (_set_random_joint_angles).
		"""
		if self.turn:
			target_angle = env.uniform(-math.pi, math.pi, env.nworld)
			site_pos = self.v.m_site_pos[:, self.s_target]  # per-world model field (view, written in place)
			site_pos[:, 0] = self.hinge_xz[0] + self.cap_radius * torch.sin(target_angle)
			site_pos[:, 2] = self.hinge_xz[1] + self.cap_radius * torch.cos(target_angle)
		env.reject(env.randomize_joints(env.nworld, env.n_candidates))

	def _xz(self, start):
		"""(x, z) components of a 3D position sensor starting at sensordata index `start` -> (S, 2)."""
		sd = self.v.sensordata
		return torch.stack([sd[:, start], sd[:, start + 2]], -1)

	def tip_position(self):
		"""Finger tip position relative to the spinner, (S, 2)."""
		return self._xz(self.i_tip) - self._xz(self.i_spinner)

	def target_position(self):
		"""Target position relative to the spinner, (S, 2)."""
		return self._xz(self.i_target) - self._xz(self.i_spinner)

	def dist_to_target(self):
		"""Signed distance from the tip to the target surface, (S,)."""
		return torch.linalg.norm(self.target_position() - self.tip_position(), dim=-1) - self.v.m_site_size[self.s_target, 0]

	def observation(self):
		"""Port of dm_control.suite.finger Spin/Turn.get_observation -> (S, obs_dim).

		Layout: position = [proximal, distal sensors, tip (x, z) relative to the spinner], velocity =
		[proximal_velocity, distal_velocity, hinge_velocity], touch = log1p([touchtop, touchbottom]); turn tasks
		append target_position (x, z) relative to the spinner and dist_to_target (signed, negative inside).
		"""
		sd = self.v.sensordata
		obs = [sd[:, self.i_bounded], self.tip_position(), sd[:, self.i_vel], torch.log1p(sd[:, self.i_touch])]
		if self.turn:
			obs += [self.target_position(), self.dist_to_target().unsqueeze(-1)]
		return torch.cat(obs, -1)

	def reward(self):
		"""Port of dm_control.suite.finger Spin/Turn.get_reward -> (S,): turn: 1 if the tip is inside the target (dist_to_target <= 0); spin: 1 if hinge velocity <= -15."""
		if self.turn:
			return (self.dist_to_target() <= 0).to(torch.float32)
		return (self.v.sensordata[:, self.i_hinge_vel] <= -self._SPIN_VELOCITY).to(torch.float32)


class Fish(Task):
	"""dm_control suite/fish.py (swim); the target geom is moved per episode (per-world geom_pos)."""
	_JOINTS = ['tail1', 'tail_twist', 'tail2', 'finright_roll', 'finright_pitch', 'finleft_roll', 'finleft_pitch']
	model_sync = ('geom_pos',)
	per_world_model_fields = ('geom_pos',)

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		assert task_name == 'swim'
		self.i_joints = self.joint_qpos_idx(self._JOINTS)
		self.b_torso = self.body('torso')
		self.g_target, self.g_mouth = self.geom('target'), self.geom('mouth')
		self.moved_static_geoms = (self.g_target,)  # world-attached target: env refreshes geom_xpos at reset
		self.radii = float(self.mjm.geom_size[[self.g_mouth, self.g_target], 0].sum())
		self.root_qadr = self.jqpos('root')

	def init_episode(self, env):
		"""Port of dm_control.suite.fish.Swim.initialize_episode.

		Per world: root orientation = normalised N(0, I4) quaternion, the 7 tail/fin joints ~ U(-0.2, 0.2), and
		the target geom position (per-world geom_pos) x, y ~ U(-0.4, 0.4), z ~ U(0.1, 0.3).
		"""
		v, S = self.v, env.nworld
		quat = env.randn(S, 4)  # uniform random root orientation
		v.qpos[:, self.root_qadr + 3:self.root_qadr + 7] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
		v.qpos[:, self.i_joints] = env.uniform(-.2, .2, S, len(self._JOINTS))
		gp = v.m_geom_pos[:, self.g_target]
		gp[:, 0] = env.uniform(-.4, .4, S)
		gp[:, 1] = env.uniform(-.4, .4, S)
		gp[:, 2] = env.uniform(.1, .3, S)

	def mouth_to_target(self):
		"""Target position in the mouth frame, (S, 3)."""
		v = self.v
		return ((v.geom_xpos[:, self.g_target] - v.geom_xpos[:, self.g_mouth]).unsqueeze(-2) @ v.geom_xmat[:, self.g_mouth]).squeeze(-2)

	def upright(self):
		"""Projection of the torso z-axis on the world z-axis, (S,)."""
		return self.mat(self.v.xmat[:, self.b_torso], 'zz')

	def observation(self):
		"""Port of dm_control.suite.fish.Swim.get_observation -> (S, obs_dim): joint_angles (7), upright, target in the mouth frame (3), velocity = qvel."""
		v = self.v
		return torch.cat([v.qpos[:, self.i_joints], self.upright().unsqueeze(-1), self.mouth_to_target(), v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.fish.Swim.get_reward -> (S,): (7 * in_target + is_upright) / 8, in_target = tolerance of the mouth-target distance (bounds (0, radii), margin 2 * radii)."""
		in_target = tolerance(torch.linalg.norm(self.mouth_to_target(), dim=-1), bounds=(0, self.radii), margin=2 * self.radii)
		is_upright = 0.5 * (self.upright() + 1)
		return (7 * in_target + is_upright) / 8


class Hopper(Task):
	"""dm_control suite/hopper.py (stand, hop)."""
	_STAND_HEIGHT = 0.6
	_HOP_SPEED = 2

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.hopping = task_name == 'hop'
		self.b_torso, self.b_foot = self.body('torso'), self.body('foot')
		self.i_speed = self.sensor_slice('torso_subtreelinvel').start
		self.i_touch = self.sensor_idx(['touch_toe', 'touch_heel'])

	def init_episode(self, env):
		"""GPU port of the task's dm_control initialize_episode for all S worlds (see Task.init_episode)."""
		self.v.qpos.copy_(env.randomize_joints(env.nworld, 1)[:, 0])

	def observation(self):
		"""Port of dm_control.suite.hopper.Hopper.get_observation -> (S, obs_dim): position = qpos[1:] (without root x), velocity = qvel, touch = log1p(toe, heel touch sensors)."""
		v = self.v
		return torch.cat([v.qpos[:, 1:], v.qvel, torch.log1p(v.sensordata[:, self.i_touch])], -1)  # position w/o x, velocity, touch

	def reward(self):
		"""Port of dm_control.suite.hopper.Hopper.get_reward -> (S,).

		standing = tolerance(torso z - foot z (xipos), (0.6, 2)). hop: standing * linear tolerance of the torso
		subtree forward speed (bounds (2, inf), margin 1, value 0.5 at the margin). stand: standing *
		small_control, with small_control = (mean quadratic control tolerance + 4) / 5.
		"""
		v = self.v
		height = v.xipos[:, self.b_torso, 2] - v.xipos[:, self.b_foot, 2]
		standing = tolerance(height, (self._STAND_HEIGHT, 2))
		if self.hopping:
			hopping = tolerance(v.sensordata[:, self.i_speed], bounds=(self._HOP_SPEED, _INF), margin=self._HOP_SPEED / 2,
								value_at_margin=0.5, sigmoid='linear')
			return standing * hopping
		small_control = tolerance(v.ctrl, margin=1, value_at_margin=0, sigmoid='quadratic').mean(-1)
		small_control = (small_control + 4) / 5
		return standing * small_control


class Humanoid(Task):
	"""dm_control suite/humanoid.py (stand, walk, run); joints rejection-sampled until no contact."""
	_STAND_HEIGHT = 1.4
	_SPEEDS = {'stand': 0, 'walk': 1, 'run': 10}
	uses_rejection = True

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.move_speed = self._SPEEDS[task_name]
		self.b_torso, self.b_head = self.body('torso'), self.body('head')
		self.b_ext = _idx([self.body(s + l) for s in ('left_', 'right_') for l in ('hand', 'foot')])
		self.i_comvel = self.sensor_idx(['torso_subtreelinvel'])

	def init_episode(self, env):
		"""GPU port of the task's dm_control initialize_episode for all S worlds (see Task.init_episode)."""
		env.reject(env.randomize_joints(env.nworld, env.n_candidates))

	def observation(self):
		"""Port of dm_control.suite.humanoid.Humanoid.get_observation (pure_state=False) -> (S, obs_dim).

		Layout: joint_angles = qpos[7:] (without the free root), head_height, extremities = hands and feet
		relative to the torso in the torso frame (4 x 3), torso_vertical = torso xmat row z ('zx', 'zy', 'zz'),
		com_velocity = torso_subtreelinvel sensor, velocity = qvel.
		"""
		v = self.v
		frame = v.xmat[:, self.b_torso]                                  # (S, 3, 3)
		extremities = ((v.xpos[:, self.b_ext] - v.xpos[:, self.b_torso].unsqueeze(1)) @ frame).flatten(1)
		torso_vertical = frame[:, 2]                                      # ['zx', 'zy', 'zz']
		# joint_angles (qpos without the free root), head_height, extremities (in the torso frame),
		# torso_vertical, com_velocity, velocity
		return torch.cat([v.qpos[:, 7:], v.xpos[:, self.b_head, 2:3], extremities, torso_vertical,
						  v.sensordata[:, self.i_comvel], v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.humanoid.Humanoid.get_reward -> (S,).

		stand_reward = standing (head height >= 1.4, margin 0.35) * upright (torso 'zz' >= 0.9, linear, margin
		1.9); small_control = (4 + mean quadratic control tolerance) / 5. stand: also times dont_move = mean
		tolerance of the horizontal CoM velocity (margin 2). walk/run: also times move = (5 * linear tolerance of
		the horizontal CoM speed (bounds (move_speed, inf), margin move_speed) + 1) / 6.
		"""
		v = self.v
		standing = tolerance(v.xpos[:, self.b_head, 2], bounds=(self._STAND_HEIGHT, _INF), margin=self._STAND_HEIGHT / 4)
		upright = tolerance(self.mat(v.xmat[:, self.b_torso], 'zz'), bounds=(0.9, _INF), sigmoid='linear', margin=1.9,
							value_at_margin=0)
		stand_reward = standing * upright
		small_control = tolerance(v.ctrl, margin=1, value_at_margin=0, sigmoid='quadratic').mean(-1)
		small_control = (4 + small_control) / 5
		com_vel = v.sensordata[:, self.i_comvel]
		if self.move_speed == 0:
			dont_move = tolerance(com_vel[:, :2], margin=2).mean(-1)
			return small_control * stand_reward * dont_move
		com_velocity = torch.linalg.norm(com_vel[:, :2], dim=-1)
		move = tolerance(com_velocity, bounds=(self.move_speed, _INF), margin=self.move_speed, value_at_margin=0,
						 sigmoid='linear')
		move = (5 * move + 1) / 6
		return small_control * stand_reward * move


class Quadruped(Task):
	"""dm_control suite/quadruped.py (walk, run); initial height found by a batched contact search."""
	_SPEEDS = {'walk': 0.5, 'run': 5}
	uses_rejection = True

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		m = self.mjm
		self.desired_speed = self._SPEEDS[task_name]
		self.b_torso = self.body('torso')
		hinge_ids = np.nonzero(m.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))[0]
		self.i_hinge_qpos = _idx(m.jnt_qposadr[hinge_ids].tolist())
		self.i_hinge_qvel = _idx(m.jnt_dofadr[hinge_ids].tolist())
		self.i_velocimeter = self.sensor_idx(['velocimeter'])

		def by_type(*types):
			"""sensordata indices of all sensors of the given mjtSensor types (in sensor order)."""
			ids = np.where(np.isin(m.sensor_type, [int(x) for x in types]).ravel())[0]
			return self.sensor_idx([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in ids])
		S = mujoco.mjtSensor
		self.i_imu = by_type(S.mjSENS_GYRO, S.mjSENS_ACCELEROMETER)
		self.i_ft = by_type(S.mjSENS_FORCE, S.mjSENS_TORQUE)
		self.root_qadr = self.jqpos('root')

	def init_episode(self, env):
		"""Port of dm_control.suite.quadruped.Move.initialize_episode (with _find_non_contacting_height).

		Per world: random unit-quaternion root orientation, then the lowest root height z = 0, 0.01, 0.02, ...
		without contacts. Heights are tested in passes of env.n_height_candidates via env.reject(retry=False);
		the loop ends on a host sync (bool(done.all())), so this runs in the non-captured reset path.
		"""
		# _find_non_contacting_height: z = 0, .01, .02, ... until no contact. Evaluated as one batch of
		# consecutive heights per world; the first non-contacting one is exactly the sequential result.
		S = env.nworld
		orientation = env.randn(S, 4)
		orientation = orientation / torch.linalg.norm(orientation, dim=-1, keepdim=True)
		K, a = env.n_height_candidates, self.root_qadr
		z0 = torch.zeros(S, device='cuda')
		done = torch.zeros(S, dtype=torch.bool, device='cuda')
		final = env.qpos0.repeat(S, 1)
		# heights z0 .. z0 + 0.01 * (K - 1) per pass; rarely needed further passes continue the grid
		while True:
			cands = env.qpos0.repeat(S, K, 1)
			cands[..., a + 2] = z0.unsqueeze(1) + 0.01 * torch.arange(K, device='cuda', dtype=torch.float32)
			cands[..., a + 3:a + 7] = orientation.unsqueeze(1)
			found, pick = env.reject(cands, retry=False)
			newly = found & ~done
			final[newly] = pick[newly]
			done |= found
			if bool(done.all()):
				break
			z0 += 0.01 * K
		self.v.qpos.copy_(final)

	def torso_upright(self):
		"""Projection of the torso z-axis on the world z-axis, (S,)."""
		return self.mat(self.v.xmat[:, self.b_torso], 'zz')

	def observation(self):
		"""Port of dm_control.suite.quadruped._common_observations (Move.get_observation) -> (S, obs_dim).

		Layout: egocentric_state = [hinge qpos, hinge qvel, act], torso_velocity = velocimeter, torso_upright,
		imu = gyro + accelerometer sensors, force_torque = asinh(toe force/torque sensors).
		"""
		v = self.v
		# egocentric_state, torso_velocity, torso_upright, imu, force_torque (asinh-squashed, as in dm_control)
		ego = torch.cat([v.qpos[:, self.i_hinge_qpos], v.qvel[:, self.i_hinge_qvel], v.act], -1)
		return torch.cat([ego, v.sensordata[:, self.i_velocimeter], self.torso_upright().unsqueeze(-1),
						  v.sensordata[:, self.i_imu], torch.asinh(v.sensordata[:, self.i_ft])], -1)

	def reward(self):
		"""Port of dm_control.suite.quadruped.Move.get_reward -> (S,).

		_upright_reward (torso 'zz' with bounds (1, inf), linear, margin 2) times move_reward = linear tolerance
		of the velocimeter x component (bounds (desired_speed, inf), margin desired_speed, value 0.5 at the margin).
		"""
		v = self.v
		upright = tolerance(self.torso_upright(), bounds=(1.0, _INF), sigmoid='linear', margin=2.0, value_at_margin=0)
		move_reward = tolerance(v.sensordata[:, self.i_velocimeter][:, 0], bounds=(self.desired_speed, _INF),
								margin=self.desired_speed, value_at_margin=0.5, sigmoid='linear')
		return upright * move_reward


class Reacher(Task):
	"""dm_control suite/reacher.py (easy, hard); the target geom is moved per episode (per-world geom_pos).

	The target size (easy vs hard) is constant across episodes and compiled into the GPU model.
	"""
	model_sync = ('geom_pos', 'geom_size')
	per_world_model_fields = ('geom_pos',)

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.g_target, self.g_finger = self.geom('target'), self.geom('finger')
		self.moved_static_geoms = (self.g_target,)

	def init_episode(self, env):
		"""Port of dm_control.suite.reacher.Reacher.initialize_episode.

		Per world: joints from randomize_limited_and_rotational_joints (no rejection), and the target geom
		(per-world geom_pos) at angle ~ U(0, 2 pi), radius ~ U(0.05, 0.20) in the xy plane.
		"""
		v, S = self.v, env.nworld
		v.qpos.copy_(env.randomize_joints(S, 1)[:, 0])
		angle = env.uniform(0, 2 * math.pi, S)
		radius = env.uniform(.05, .20, S)
		gp = v.m_geom_pos[:, self.g_target]
		gp[:, 0] = radius * torch.sin(angle)
		gp[:, 1] = radius * torch.cos(angle)

	def finger_to_target(self):
		"""Target minus finger position in the xy plane, (S, 2)."""
		v = self.v
		return v.geom_xpos[:, self.g_target, :2] - v.geom_xpos[:, self.g_finger, :2]

	def observation(self):
		"""Port of dm_control.suite.reacher.Reacher.get_observation -> (S, nq + 2 + nv): position = qpos, to_target = finger_to_target (xy), velocity = qvel."""
		v = self.v
		return torch.cat([v.qpos, self.finger_to_target(), v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.reacher.Reacher.get_reward -> (S,): 1 if the finger-target distance is within the sum of both geom radii, else 0."""
		radii = self.v.m_geom_size[0, self.g_target, 0] + self.v.m_geom_size[0, self.g_finger, 0]
		dist = torch.linalg.norm(self.finger_to_target(), dim=-1)
		return ((0 <= dist) & (dist <= radii)).to(torch.float32)  # tolerance(dist, (0, radii)), margin 0


class Walker(Task):
	"""dm_control suite/walker.py (stand, walk, run)."""
	_STAND_HEIGHT = 1.2
	_SPEEDS = {'stand': 0, 'walk': 1, 'run': 8}

	def __init__(self, task_name, physics, v):
		super().__init__(task_name, physics, v)
		self.move_speed = self._SPEEDS[task_name]
		self.b_torso = self.body('torso')
		self.i_speed = self.sensor_slice('torso_subtreelinvel').start

	def init_episode(self, env):
		"""GPU port of the task's dm_control initialize_episode for all S worlds (see Task.init_episode)."""
		self.v.qpos.copy_(env.randomize_joints(env.nworld, 1)[:, 0])

	def observation(self):
		"""Port of dm_control.suite.walker.PlanarWalker.get_observation -> (S, 2 * (nbody - 1) + 1 + nv).

		Layout: orientations = xmat ('xx', 'xz') of every body except the world (interleaved per body),
		height = torso z, velocity = qvel.
		"""
		v = self.v
		xm = v.xmat[:, 1:]  # all bodies except the world
		orient = torch.stack([self.mat(xm, 'xx'), self.mat(xm, 'xz')], -1).flatten(1)
		return torch.cat([orient, v.xpos[:, self.b_torso, 2:3], v.qvel], -1)

	def reward(self):
		"""Port of dm_control.suite.walker.PlanarWalker.get_reward -> (S,).

		stand_reward = (3 * standing + upright) / 4, with standing = tolerance(torso z, (1.2, inf), margin 0.6)
		and upright = (1 + torso 'zz') / 2. stand: stand_reward. walk/run: stand_reward * (5 * move_reward + 1) / 6,
		with move_reward = linear tolerance of the torso subtree forward speed (bounds (move_speed, inf),
		margin move_speed / 2, value 0.5 at the margin).
		"""
		v = self.v
		standing = tolerance(v.xpos[:, self.b_torso, 2], bounds=(self._STAND_HEIGHT, _INF), margin=self._STAND_HEIGHT / 2)
		upright = (1 + self.mat(v.xmat[:, self.b_torso], 'zz')) / 2
		stand_reward = (3 * standing + upright) / 4
		if self.move_speed == 0:
			return stand_reward
		move_reward = tolerance(v.sensordata[:, self.i_speed], bounds=(self.move_speed, _INF), margin=self.move_speed / 2,
								value_at_margin=0.5, sigmoid='linear')
		return stand_reward * (5 * move_reward + 1) / 6


class Dog(Task):
	"""dm_control suite/dog.py (walk, trot, run; the Move task).

	Needs dm_control's task object for constants computed in its initialize_episode (stand heights,
	body weight). The model uses MuJoCo's noslip solver (GPU port in mujoco_warp/_src/noslip.py).
	"""
	_MIN_UPRIGHT_COSINE = float(np.cos(np.deg2rad(30)))
	_SPEEDS = {'walk': 1, 'trot': 3, 'run': 9}

	def __init__(self, task_name, physics, v, task_obj=None):
		super().__init__(task_name, physics, v)
		m = self.mjm
		self.move_speed = self._SPEEDS[task_name]
		hinge = m.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)
		self.i_hinge_qpos = _idx(m.jnt_qposadr[hinge].tolist())
		self.i_hinge_qvel = _idx(m.jnt_dofadr[hinge].tolist())
		self.b_tp = _idx([self.body('torso'), self.body('pelvis')])
		self.b_z = _idx([self.body('skull'), self.body('torso'), self.body('pelvis')])
		self.b_torso = self.body('torso')
		self.i_torso_linvel = self.sensor_idx(['torso_linvel'])
		self.i_inertial = self.sensor_idx(['accelerometer', 'velocimeter', 'gyro'])
		self.i_foot = self.sensor_idx(['foot_L', 'foot_R', 'hand_L', 'hand_R'])
		self.i_touch = self.sensor_idx(['palm_L', 'palm_R', 'sole_L', 'sole_R'])
		# constants computed by dm_control's Stand.initialize_episode (identical every episode)
		self.stand_height = [float(x) for x in task_obj._stand_height]
		self.body_weight = float(task_obj._body_weight)
		self.root_qadr = self.jqpos('root')
		assert m.nu == m.na
		self.ctrl_lo = torch.tensor(m.actuator_ctrlrange[:, 0], dtype=torch.float32, device='cuda')
		self.ctrl_hi = torch.tensor(m.actuator_ctrlrange[:, 1], dtype=torch.float32, device='cuda')

	def init_episode(self, env):
		"""Port of dm_control.suite.dog.Stand.initialize_episode (used by Move).

		Per world: root orientation = rotation by azimuth ~ U(0, 2 pi) about world z; root qvel x, y and yaw
		rate ~ 2 N(0, 1); actuator activations uniform in the control range. The stand heights and body weight
		computed there are constant and taken from the dm_control task object in __init__.
		"""
		v, S = self.v, env.nworld
		azimuth = env.uniform(0, 2 * math.pi, S)
		a = self.root_qadr
		# root orientation: rotation by a random azimuth about the world z-axis
		v.qpos[:, a + 3] = torch.cos(azimuth / 2)
		v.qpos[:, a + 4] = 0.
		v.qpos[:, a + 5] = 0.
		v.qpos[:, a + 6] = torch.sin(azimuth / 2)
		r = 2 * env.randn(S, 3)  # random planar root velocities: vx, vy and yaw rate
		v.qvel[:, 0], v.qvel[:, 1], v.qvel[:, 5] = r[:, 0], r[:, 1], r[:, 2]
		# random actuator activations uniform in the control range
		v.act.copy_(self.ctrl_lo + (self.ctrl_hi - self.ctrl_lo) * env.rand(S, self.mjm.nu))

	def torso_com_velocity(self):
		"""Torso linear velocity sensor expressed in the torso frame, (S, 3)."""
		v = self.v
		return (v.sensordata[:, self.i_torso_linvel].unsqueeze(-2) @ v.xmat[:, self.b_torso]).squeeze(-2)

	def observation(self):
		"""Port of dm_control.suite.dog.Stand.get_observation_components (no reward factors) -> (S, obs_dim).

		Layout: joint_angles (hinge qpos), joint_velocities (hinge qvel), torso_pelvis_height (2), z_projection
		(xmat row z of skull, torso, pelvis; 9), torso_com_velocity (torso frame), inertial_sensors
		(accelerometer, velocimeter, gyro), foot_forces, touch_sensors, actuator_state = act.
		"""
		v = self.v
		z_projection = v.xmat[:, self.b_z][:, :, 2, :].flatten(1)  # rows 'zx','zy','zz'
		return torch.cat([v.qpos[:, self.i_hinge_qpos], v.qvel[:, self.i_hinge_qvel], v.xpos[:, self.b_tp, 2], z_projection,
						  self.torso_com_velocity(), v.sensordata[:, self.i_inertial], v.sensordata[:, self.i_foot],
						  v.sensordata[:, self.i_touch], v.act], -1)

	def reward(self):
		"""Port of dm_control.suite.dog.Move.get_reward (product of Move.get_reward_factors) -> (S,).

		Factors: torso and pelvis height >= stand_height (margin stand_height), upright cosines of skull, torso and
		pelvis >= cos(30 deg) (linear), touch-sensor sum >= body weight (linear, value 0.9 at the margin) and
		forward = (4 * linear tolerance of the torso-frame CoM x velocity in (move_speed, 2 * move_speed),
		margin max(1, move_speed)) + 1) / 5.
		"""
		v = self.v
		tp = v.xpos[:, self.b_tp, 2]
		torso = tolerance(tp[:, 0], bounds=(self.stand_height[0], _INF), margin=self.stand_height[0])
		pelvis = tolerance(tp[:, 1], bounds=(self.stand_height[1], _INF), margin=self.stand_height[1])
		upright = tolerance(v.xmat[:, self.b_z][:, :, 2, 2], bounds=(self._MIN_UPRIGHT_COSINE, _INF), sigmoid='linear',
							margin=self._MIN_UPRIGHT_COSINE + 1, value_at_margin=0)
		touch = tolerance(v.sensordata[:, self.i_touch].sum(-1), bounds=(self.body_weight, _INF), margin=self.body_weight,
						  sigmoid='linear', value_at_margin=0.9)
		speed_margin = max(1.0, self.move_speed)
		forward = tolerance(self.torso_com_velocity()[:, 0], bounds=(self.move_speed, 2 * self.move_speed),
							margin=speed_margin, value_at_margin=0, sigmoid='linear')
		forward = (4 * forward + 1) / 5
		return torso * pelvis * upright.prod(-1) * touch * forward


# dm_control domain name -> Task subclass (used by env_warp.DMCWarpEnv)
DOMAINS = {
	'acrobot': Acrobot, 'cartpole': Cartpole, 'cheetah': Cheetah, 'ball_in_cup': BallInCup, 'finger': Finger,
	'fish': Fish, 'hopper': Hopper, 'humanoid': Humanoid, 'quadruped': Quadruped, 'reacher': Reacher,
	'walker': Walker, 'dog': Dog,
}

# per-world contact / constraint-row capacities (>= 4x the maxima observed in random-action rollouts
# with CPU MuJoCo; usage is checked by bench/validate_env.py)
CAPACITY = {
	'acrobot': (8, 16), 'cartpole': (8, 16), 'reacher': (8, 16), 'fish': (8, 16),
	'cheetah': (16, 64), 'ball_in_cup': (16, 64), 'finger': (16, 64),
	'hopper': (32, 128), 'walker': (32, 128), 'quadruped': (32, 128),
	'humanoid': (64, 256), 'dog': (128, 640),
}
