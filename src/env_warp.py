"""GPU DMControl environment for TD-MPC on MuJoCo Warp (state modality), S worlds at once.

Replaces tdmpc/src/env.py: physics runs in MuJoCo Warp (https://github.com/google-deepmind/mujoco_warp,
Apache License 2.0), observations / rewards / episode resets are computed on the GPU by dmc_tasks.py,
and the stepping semantics reproduce dm_control exactly (see below). `make_env(cfg)` is the entry
point used by train.py / train_vec.py; `DMCWarpEnv.step` takes and returns CUDA tensors.

Faithfulness to the original CPU pipeline (tdmpc/src/env.py + dm_control):
  * Same MJCF model: the dm_control suite env is built with `task_kwargs={'random': seed}` and its
    compiled MjModel is put on the GPU with mujoco_warp.
  * Same episode initialisation distribution, computed on the GPU (reset_mode='gpu', default):
    every domain's `initialize_episode` is ported in dmc_tasks.py (seeded CUDA generator instead
    of numpy's RandomState). Rejection sampling ("resample until no contact") is evaluated as one
    batched mujoco_warp collision pass over candidate worlds, taking the first non-penetrating
    candidate (= the sequential rejection sampler's output distribution; exact for quadruped's
    deterministic height search). Then `forward` is recomputed with actuation disabled, exactly
    like dm_control's `after_reset`.
    reset_mode='cpu' instead runs dm_control's own Environment.reset() on the CPU and uploads the
    state (bit-identical RNG stream to the original code; single world; used for validation only).
  * Same stepping semantics as dm_control's legacy `Physics.step(n_sub_steps)`:
      Euler/implicit: mj_step2, (n-1) x mj_step, mj_step1;   RK4: n x mj_step, mj_step1
    where mj_step1 keeps the acceleration-stage sensors (touch, force, torque, accelerometer) from
    the last sub-step, as C MuJoCo does (mujoco_warp's step1 zeroes them; the env saves/restores them).
  * Same wrappers: action dtype float32 -> ActionRepeat(action_repeat, reward summed) ->
    action_scale([-1,1] -> actuator range) -> episode of 1000 // action_repeat agent steps.
  * The CPU dm_control env is only used once at construction (model compilation, constants).
The whole agent step (action scaling, all control sub-steps, rewards, observation) is captured in
one CUDA graph. With nworld = S > 1 the S worlds are independent copies (one per seed); all worlds
reset together (all DMC episodes have the same length).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import dataclasses
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src.types import DisableBit
from dm_control import suite

import dmc_tasks

wp.config.quiet = True

_STREAM = None  # process-wide CUDA stream shared by torch and warp (created lazily by cuda_stream())


def set_blocking_sync(device=0):
	"""Make host threads *sleep* while waiting for the GPU instead of spin-waiting (CUDA default).

	Sets CU_CTX_SCHED_BLOCKING_SYNC on the primary context shared by torch and warp. Must run before
	the context is created; no effect on results. Disable with TDMPC_BLOCKING_SYNC=0.
	"""
	import ctypes
	if os.environ.get('TDMPC_BLOCKING_SYNC', '1') != '1':
		return False
	try:
		cu = ctypes.CDLL('libcuda.so.1')
		cu.cuInit(0)
		dev = ctypes.c_int()
		cu.cuDeviceGet(ctypes.byref(dev), device)
		return cu.cuDevicePrimaryCtxSetFlags(dev, 0x4) == 0  # CU_CTX_SCHED_BLOCKING_SYNC
	except OSError:
		return False


def cuda_stream():
	"""One non-default CUDA stream shared by torch and warp (graph capture needs a non-legacy stream).

	On first call also sets blocking sync (before the CUDA context exists), initialises warp and makes
	the stream current for both libraries, so that torch ops and warp kernels are ordered on one stream
	and can be captured into the same CUDA graph. Returns the torch.cuda.Stream.
	"""
	global _STREAM
	if _STREAM is None:
		set_blocking_sync()
		wp.init()
		_STREAM = torch.cuda.Stream()
		torch.cuda.set_stream(_STREAM)
		wp.set_stream(wp.stream_from_torch(_STREAM))
	return _STREAM


@wp.kernel
def _gather(src: wp.array2d[float], idx: wp.array[int], dst: wp.array2d[float]):
	"""dst[w, i] = src[w, idx[i]] for every world w (saves selected sensordata entries)."""
	w, i = wp.tid()
	dst[w, i] = src[w, idx[i]]


@wp.kernel
def _scatter(src: wp.array2d[float], idx: wp.array[int], dst: wp.array2d[float]):
	"""dst[w, idx[i]] = src[w, i] for every world w (restores selected sensordata entries)."""
	w, i = wp.tid()
	dst[w, idx[i]] = src[w, i]


class _Views:
	"""Zero-copy torch views of the mujoco_warp arrays used by observations/rewards (all worlds).

	Data fields have a leading world dimension S (e.g. qpos (S, nq), xmat (S, nbody, 3, 3)). Writing
	into a view writes the GPU simulator state directly; the views alias fixed device buffers, so they
	stay valid inside captured CUDA graphs. Model fields are prefixed `m_`.
	"""
	def __init__(self, m, d):
		t = wp.to_torch
		self.qpos, self.qvel, self.act, self.ctrl = t(d.qpos), t(d.qvel), t(d.act), t(d.ctrl)
		self.xpos, self.xmat, self.xipos = t(d.xpos), t(d.xmat), t(d.xipos)
		self.geom_xpos, self.geom_xmat = t(d.geom_xpos), t(d.geom_xmat)
		self.site_xpos = t(d.site_xpos)
		self.sensordata = t(d.sensordata)
		self.qacc_warmstart, self.time = t(d.qacc_warmstart), t(d.time)
		self.m_geom_size = t(m.geom_size)   # (1, ngeom, 3)
		self.m_site_size = t(m.site_size)   # (nsite, 3)
		self.m_geom_pos = t(m.geom_pos)     # (1 or S, ngeom, 3)
		self.m_site_pos = t(m.site_pos)     # (1 or S, nsite, 3)


class DMCWarpEnv:
	"""TD-MPC DMControl environment simulated by mujoco_warp, batched over `nworld` independent worlds.

	Args:
		task: TD-MPC task name, e.g. 'walker-walk' or 'cup-catch'.
		seed: seeds the CPU dm_control env (model construction) and the env's CUDA generator (resets).
		action_repeat: control steps per agent step (rewards summed); episode = 1000 // action_repeat steps.
		use_graph: capture the agent step and the reset helpers in CUDA graphs (bitwise identical to eager).
		reset_mode: 'gpu' (ported initialize_episode on the GPU, default) or 'cpu' (dm_control's own
			reset on the CPU, state uploaded; exact original RNG stream; nworld must be 1).
		nworld: number of independent worlds S (one per seed in the vectorised trainer).
		batched: True -> obs (S, obs_dim), reward (S,); False -> single-env API obs (obs_dim,), reward ().
			Defaults to nworld > 1.
		skip_substep_sensors: skip sensor computation in intermediate physics sub-steps (exact; see
			docs/06_optimisations.md, item 9).
		graph_mode: 'fused' (whole agent step in one torch graph with warp captured inside, default) or
			'split' (warp-owned physics graph + torch-only reward / observation graphs).

	Invariants: tensors/arrays read by a captured graph (views, index tensors, `_act_in`, `_acc_buf`,
	...) are allocated once and must stay alive for the lifetime of the env; `step` writes the action into
	`_act_in` and replays the graph, which writes into the fixed outputs `_obs_out` / `_rew_out`.
	"""
	n_candidates = 32  # rejection-sampler candidates per world (acceptance >= 0.75 for all tasks)
	n_height_candidates = 160  # quadruped height grid: 0 .. 1.59 m in 0.01 m steps

	def __init__(self, task, seed, action_repeat, use_graph=True, reset_mode='gpu', nworld=1, batched=None,
				 skip_substep_sensors=False, graph_mode='fused'):
		cuda_stream()  # torch and warp must share one non-default stream before anything is allocated/captured
		domain, task_name = task.replace('-', '_').split('_', 1)
		domain = dict(cup='ball_in_cup').get(domain, domain)  # TD-MPC's 'cup-catch' alias
		assert (domain, task_name) in suite.ALL_TASKS, (domain, task_name)
		self.domain, self.task_name = domain, task_name
		self.action_repeat = action_repeat
		self.ep_len = 1000 // action_repeat
		self.use_graph = use_graph
		self.reset_mode = reset_mode
		self.nworld = S = int(nworld)
		self.batched = (S > 1) if batched is None else batched  # False: single-env API (obs (D,), reward ())
		assert self.batched or S == 1
		assert reset_mode == 'gpu' or S == 1
		self.gen = torch.Generator(device='cuda')
		self.gen.manual_seed(int(seed))

		# CPU dm_control env: source of the model (and of episode initial states for reset_mode='cpu')
		self.cpu_env = suite.load(domain, task_name, task_kwargs={'random': seed}, visualize_reward=False)
		self.n_sub = int(self.cpu_env._n_sub_steps)  # physics steps per control step (control_timestep / timestep)
		physics = self.cpu_env.physics
		self.mjm, self.mjd = physics.model.ptr, physics.data.ptr
		# first reset before the model is put on the GPU, so that model edits that are constant across
		# episodes (e.g. finger-spin hinge damping, reacher target size) are compiled into the GPU model;
		# the task RNG state is restored so that reset_mode='cpu' reproduces the original sequence
		rng_state = self.cpu_env.task.random.get_state()
		self.cpu_env.reset()
		self.cpu_env.task.random.set_state(rng_state)

		task_cls = dmc_tasks.DOMAINS[domain]
		nconmax, njmax = dmc_tasks.CAPACITY[domain]
		# model fields that are edited per episode per world (e.g. target positions) get a world dimension
		self.m = mjw.put_model(self.mjm, batch_sizes={f: S for f in task_cls.per_world_model_fields} if S > 1 else None)
		self.d = mjw.put_data(self.mjm, self.mjd, nworld=S, nconmax=nconmax, njmax=njmax)
		# actuation-disabled model view for dm_control's after_reset() forward
		self.m_noact = dataclasses.replace(
			self.m, opt=dataclasses.replace(self.m.opt, disableflags=self.m.opt.disableflags | DisableBit.ACTUATION))
		# sensors are outputs only (no feedback into the dynamics): intermediate physics sub-steps (and
		# reset settling steps) may skip them; the last sub-step and step1 compute every sensor as before,
		# so observations, rewards and the physical trajectory are unchanged (exact optimisation)
		self.m_sub = dataclasses.replace(
			self.m, opt=dataclasses.replace(self.m.opt, disableflags=self.m.opt.disableflags | DisableBit.SENSOR)) \
			if skip_substep_sensors else self.m
		self.v = _Views(self.m, self.d)
		kw = dict(task_obj=self.cpu_env.task) if domain == 'dog' else {}
		self.task = task_cls(task_name, physics, self.v, **kw)  # dog reads constants from dm_control's task object

		# acceleration-stage sensors survive mj_step1 in C MuJoCo but are zeroed by mjw.step1
		acc = [i for i in range(self.mjm.nsensor) if self.mjm.sensor_needstage[i] == mujoco.mjtStage.mjSTAGE_ACC]
		acc_idx = [a for i in acc for a in range(self.mjm.sensor_adr[i], self.mjm.sensor_adr[i] + self.mjm.sensor_dim[i])]
		# sensordata indices to save before / restore after mjw.step1 (see _step1_keep_acc); the buffer has
		# at least one column so that it is a valid array even for models without such sensors
		self._acc_idx = wp.array(np.array(acc_idx, dtype=np.int32), dtype=int)
		self._acc_buf = wp.zeros((S, max(1, len(acc_idx))), dtype=float)

		# action scaling (dm_control action_scale.Wrapper to [-1, 1])
		lo = torch.tensor(self.cpu_env.action_spec().minimum, dtype=torch.float32, device='cuda')
		hi = torch.tensor(self.cpu_env.action_spec().maximum, dtype=torch.float32, device='cuda')
		self._scale, self._offset = (hi - lo) / 2.0, lo + (hi - lo) / 2.0

		# (name, CPU MjModel array, GPU tensor) of model fields that dm_control's reset edits; reset_mode='cpu'
		# copies them to the GPU after every CPU reset
		self._model_sync = [(f, getattr(self.mjm, f), wp.to_torch(getattr(self.m, f))) for f in self.task.model_sync]
		self.qpos0 = torch.tensor(self.mjm.qpos0, dtype=torch.float32, device='cuda')
		self._build_joint_randomizer()
		if self.task.uses_rejection:
			K = self.n_height_candidates if domain == 'quadruped' else self.n_candidates
			self.K = K
			# separate S*K-world data: candidate world k of seed s is world s*K + k; only its collision
			# detection is run (fwd_position), the main simulation state self.d is untouched
			self.dK = mjw.put_data(self.mjm, self.mjd, nworld=S * K, nconmax=nconmax, njmax=njmax)
			self.dK_qpos = wp.to_torch(self.dK.qpos)
			self.dK_worldid = wp.to_torch(self.dK.contact.worldid)
			self.dK_nacon = wp.to_torch(self.dK.nacon)
			self._arange_con = torch.arange(self.dK_worldid.shape[0], device='cuda')

		self.obs_dim = int(sum(int(np.prod(s.shape)) for s in self.cpu_env.observation_spec().values()))
		self.action_dim = int(self.cpu_env.action_spec().shape[0])
		self.t = 0

		# warm-up (compiles kernels), then capture CUDA graphs for the after-reset forward, a plain
		# mj_step (cheetah settling) and the rejection sampler's collision pass
		self._reset_defaults()
		self._forward_noact()
		self._graphs = {}
		fns = {'forward': self._forward_noact, 'step': lambda: mjw.step(self.m_sub, self.d)}
		if self.task.uses_rejection:
			fns['collide'] = lambda: mjw.fwd_position(self.m_noact, self.dK)
		for name, fn in fns.items():
			fn()
			if use_graph:
				with wp.ScopedCapture() as cap:
					fn()
				self._graphs[name] = cap.graph

		# graph_mode='split': physics control step = warp-owned graph, reward / observation =
		#   torch-only graphs; warp and torch never share a graph (each library manages its own graph memory).
		# graph_mode='fused' (default; used for all measurements): whole agent step in ONE torch graph with warp kernels captured inside
		#   (external capture; fewer launches, but warp's capture-time allocations then live inside a graph
		#   instantiated by torch - associated with rare illegal-memory-access faults in long multi-graph runs).
		self.graph_mode = graph_mode
		self._act_in = torch.zeros(S, self.action_dim, dtype=torch.float32, device='cuda')
		self._step_graph = None
		self._reset_defaults(); self._forward_noact()
		self._agent_step_body()  # warm-up
		if use_graph and graph_mode == 'split':
			self._control_step()
			with wp.ScopedCapture() as cap:
				self._control_step()
			self._graphs['control'] = cap.graph
			self._rew_acc = torch.zeros(S, dtype=torch.float32, device='cuda')
			pool = torch.cuda.graph_pool_handle()
			torch.cuda.synchronize()
			self._g_ctrl = torch.cuda.CUDAGraph()
			with torch.cuda.graph(self._g_ctrl, stream=cuda_stream(), pool=pool):
				self.v.ctrl.copy_(self._scale * self._act_in + self._offset)
				self._rew_acc.zero_()
			self._g_rew = torch.cuda.CUDAGraph()
			with torch.cuda.graph(self._g_rew, stream=cuda_stream(), pool=pool):
				self._rew_acc.add_(self.task.reward())
			self._g_obs = torch.cuda.CUDAGraph()
			with torch.cuda.graph(self._g_obs, stream=cuda_stream(), pool=pool):
				self._obs_out = self.task.observation()
			self._rew_out = self._rew_acc
		elif use_graph:
			torch.cuda.synchronize()
			self._step_graph = torch.cuda.CUDAGraph()
			wps = wp.get_stream()
			with torch.cuda.graph(self._step_graph, stream=cuda_stream()):
				wp.capture_begin(stream=wps, external=True)
				try:
					self._obs_out, self._rew_out = self._agent_step_body()
				finally:
					# For an external capture warp returns a Graph wrapping torch's *in-progress* cudaGraph_t.
					# It must neither be garbage-collected immediately nor destroy that graph: warp.Graph.__del__
					# would call cudaGraphDestroy on a graph owned by torch (double destroy / destroy during
					# capture -> undefined behaviour, observed as rare illegal-memory-access faults under MPS).
					# Keep the object alive and drop the foreign handle; warp's capture-time allocations stay
					# owned by the instantiated torch graph through warp's graph user object.
					self._wp_external_graph = wp.capture_end(stream=wps)
					self._wp_external_graph.graph = None
		self._reset_defaults(); self._forward_noact()

	# ------------------------------------------------------------------------------------------
	def _agent_step_body(self):
		"""One agent step from `_act_in`: action scaling, action_repeat control steps, summed reward.

		Graph-capturable (no host syncs, fixed buffers). Returns (obs (S, obs_dim), reward (S,)).
		"""
		# dm_control action_scale: [-1, 1] -> actuator range; ctrlrange clamping is done inside MuJoCo
		self.v.ctrl.copy_(self._scale * self._act_in + self._offset)
		reward = torch.zeros(self.nworld, dtype=torch.float32, device='cuda')
		for _ in range(self.action_repeat):
			self._control_step()
			reward = reward + self.task.reward()
		return self.task.observation(), reward

	def _step1_keep_acc(self):
		"""mjw.step1 that keeps the acceleration-stage sensors, like C MuJoCo's mj_step1.

		C mj_step1 only recomputes position/velocity sensors and leaves touch / force / torque /
		accelerometer readings of the previous mj_step2 in sensordata; mjw.step1 zeroes them first. They
		are gathered before and scattered back after, so observations see the same (one-sub-step-old)
		values as dm_control.
		"""
		n = self._acc_idx.shape[0]
		if n:
			wp.launch(_gather, dim=(self.nworld, n), inputs=[self.d.sensordata, self._acc_idx], outputs=[self._acc_buf])
		mjw.step1(self.m, self.d)
		if n:
			wp.launch(_scatter, dim=(self.nworld, n), inputs=[self._acc_buf, self._acc_idx], outputs=[self.d.sensordata])

	def _control_step(self):
		"""dm_control legacy Physics.step(n_sub_steps) for all worlds (one control step, graph-capturable).

		Euler / implicit: mj_step2, (n-1) x mj_step, mj_step1 - the state is assumed to have been through
		mj_step1 already (after reset: forward), so the step is finished with mj_step2 and the
		position/velocity quantities of the new state are recomputed with mj_step1.
		RK4: n x mj_step, mj_step1.
		"""
		n = self.n_sub
		mk = lambda i: self.m if i == n - 1 else self.m_sub  # last sub-step computes the sensors
		if self.mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
			for i in range(n):
				mjw.step(mk(i), self.d)
		else:
			mjw.step2(mk(0), self.d)
			for i in range(1, n):
				mjw.step(mk(i), self.d)
		self._step1_keep_acc()

	def _forward_noact(self):
		"""mj_forward with actuation disabled (dm_control's after_reset), all worlds of self.d."""
		mjw.forward(self.m_noact, self.d)

	def _run(self, name, fallback):
		"""Replay the warp graph captured under `name` if available, else call `fallback()` eagerly."""
		g = self._graphs.get(name)
		if g is not None:
			wp.capture_launch(g)
		else:
			fallback()

	# random helpers on the env's own seeded CUDA generator ---------------------------------------
	def rand(self, *shape):
		"""U[0, 1) samples of the given shape on the env's CUDA generator."""
		return torch.rand(*shape, generator=self.gen, device='cuda')

	def randn(self, *shape):
		"""Standard normal samples of the given shape on the env's CUDA generator."""
		return torch.randn(*shape, generator=self.gen, device='cuda')

	def uniform(self, lo, hi, *shape):
		"""U[lo, hi) samples of the given shape (like numpy RandomState.uniform)."""
		return lo + (hi - lo) * self.rand(*shape)

	def _build_joint_randomizer(self):
		"""Precompute the qpos addresses / ranges used by randomize_joints (once, on the host).

		Follows dm_control's randomize_limited_and_rotational_joints: limited hinge/slide -> uniform in
		range, unlimited hinge -> uniform in [-pi, pi], limited ball -> random quaternion within the
		angle limit, unlimited ball -> uniform random quaternion, free -> random orientation only
		(position untouched); unlimited slide joints are left at qpos0.
		"""
		m = self.mjm
		# plain ints: numpy ints do not compare reliably against MuJoCo's pybind enums with `in`
		HINGE, SLIDE, BALL, FREE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE),
									int(mujoco.mjtJoint.mjJNT_BALL), int(mujoco.mjtJoint.mjJNT_FREE))
		hs, hs_lo, hs_hi, ball_lim, ball_free, free = [], [], [], [], [], []
		for j in range(m.njnt):
			t, adr, lim = int(m.jnt_type[j]), int(m.jnt_qposadr[j]), bool(m.jnt_limited[j])
			lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
			if lim:
				if t == HINGE or t == SLIDE:
					hs.append(adr); hs_lo.append(lo); hs_hi.append(hi)
				elif t == BALL:
					ball_lim.append((adr, hi))
			else:
				if t == HINGE:
					hs.append(adr); hs_lo.append(-np.pi); hs_hi.append(np.pi)
				elif t == BALL:
					ball_free.append(adr)
				elif t == FREE:
					free.append(adr)
		f = lambda x: torch.tensor(np.array(x, dtype=np.float64), dtype=torch.float32, device='cuda')
		self._rj = dict(hs=torch.tensor(hs, dtype=torch.long, device='cuda'), lo=f(hs_lo), hi=f(hs_hi),
						ball_lim=ball_lim, ball_free=ball_free, free=free)

	def randomize_joints(self, S, K):
		"""Port of dm_control randomizers.randomize_limited_and_rotational_joints -> (S, K, nq).

		Returns K candidate configurations per world, starting from qpos0 (all non-randomised entries).
		"""
		r = self._rj
		q = self.qpos0.repeat(S, K, 1)
		if len(r['hs']):
			q[..., r['hs']] = r['lo'] + (r['hi'] - r['lo']) * self.rand(S, K, len(r['hs']))
		for adr, limit in r['ball_lim']:  # random_limited_quaternion
			axis = self.randn(S, K, 3)
			axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
			angle = self.rand(S, K, 1) * limit
			q[..., adr:adr + 4] = torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], -1)
		for adr in r['ball_free']:
			quat = self.randn(S, K, 4)
			q[..., adr:adr + 4] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
		for adr in r['free']:
			quat = self.rand(S, K, 4)  # uniform in [0, 1)^4 then normalised, as in dm_control
			q[..., adr + 3:adr + 7] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
		return q

	def _first_free(self, cands):
		"""cands (S, K, nq) -> (found (S,), first non-penetrating candidate (S, nq)).

		Runs one collision pass (fwd_position with actuation disabled) over all S*K candidate worlds and
		counts the active contacts per candidate world. Taking the first contact-free candidate per
		world gives the same distribution as dm_control's sequential "sample until no contact" loop.
		If no candidate of a world is free, `found` is False and the returned pick is candidate 0.
		"""
		S, K = self.nworld, self.K
		assert cands.shape[1] == K
		self.dK_qpos.copy_(cands.reshape(S * K, -1))
		self._run('collide', lambda: mjw.fwd_position(self.m_noact, self.dK))
		# contacts are stored in one flat buffer for all worlds: only the first nacon entries are valid,
		# their worldid says which candidate they belong to (stale entries beyond nacon are masked out)
		valid = self._arange_con < self.dK_nacon[0]
		ncon = torch.zeros(S * K, device='cuda').scatter_add_(0, self.dK_worldid.clamp(0, S * K - 1).long(), valid.float())
		free = (ncon == 0).view(S, K)
		idx = torch.argmax(free.to(torch.int32), dim=1)  # first True per world (0 if none)
		return free.any(1), cands[torch.arange(S, device='cuda'), idx]

	def reject(self, cands, retry=True):
		"""Per world: first candidate whose after-reset configuration has no contacts.

		Equivalent to dm_control's `while penetrating: sample; after_reset(); penetrating = ncon > 0`.
		retry=True: writes qpos (resampling worlds whose whole batch penetrated).
		retry=False: returns (found, pick) without writing.
		"""
		found, pick = self._first_free(cands)
		if not retry:
			return found, pick
		while not bool(found.all()):  # one scalar host sync per episode
			f2, p2 = self._first_free(self.task.resample(self, self.K))
			take = ~found & f2  # only worlds still without a valid candidate take the new pick
			pick[take] = p2[take]
			found |= f2
		self.v.qpos.copy_(pick)
		return found, pick

	def settle(self, nstep):
		"""`nstep` plain mj_step calls followed by mj_step1 (used for cheetah's settling steps at reset)."""
		for _ in range(nstep):
			self._run('step', lambda: mjw.step(self.m_sub, self.d))
		self._step1_keep_acc()

	def _refresh_static_geoms(self):
		"""World-attached geoms moved by the episode initialisation: recompute their world position
		(mujoco_warp only computes static geom poses in make_data; C MuJoCo does it in mj_kinematics)."""
		v = self.v
		for g in self.task.moved_static_geoms:
			b = int(self.mjm.geom_bodyid[g])
			assert self.mjm.body_weldid[b] == 0  # geom must be welded to the world body (static)
			# geom_xpos = body_xmat @ geom_pos + body_xpos, with the per-world geom_pos edited by init_episode
			v.geom_xpos[:, g] = (v.xmat[:, b] @ v.m_geom_pos[:, g].unsqueeze(-1)).squeeze(-1) + v.xpos[:, b]

	def _reset_defaults(self):
		"""mj_resetData equivalent for the state variables (all worlds): qpos0, zero vel/act/ctrl/time/forces."""
		v = self.v
		v.qpos.copy_(self.qpos0.expand_as(v.qpos))
		for name in ('qvel', 'act', 'ctrl', 'qacc_warmstart', 'time'):
			getattr(v, name).zero_()
		self.d.qfrc_applied.zero_()
		self.d.xfrc_applied.zero_()

	def _reset_from_cpu(self):
		"""Upload the CPU dm_control state after its reset to the GPU (single world, validation)."""
		d, v, mjd = self.d, self.v, self.mjd
		for name, src in (('qpos', mjd.qpos), ('qvel', mjd.qvel), ('act', mjd.act), ('ctrl', mjd.ctrl),
						  ('qacc_warmstart', mjd.qacc_warmstart)):
			if src.size:
				getattr(v, name)[0].copy_(torch.from_numpy(np.asarray(src, dtype=np.float32)))
		v.time.fill_(float(mjd.time))
		d.qfrc_applied.zero_()
		d.xfrc_applied.zero_()
		for _, src, dst in self._model_sync:
			dst.copy_(torch.from_numpy(np.asarray(src, dtype=np.float32)).reshape(dst.shape))

	# ------------------------------------------------------------------------------------------
	def reset(self):
		"""Start a new episode in all worlds and return the initial observation.

		Not graph-captured as a whole (rejection sampling may loop with a host sync), but uses the captured
		forward / step / collide graphs. Returns obs (S, obs_dim), or (obs_dim,) for the single-env API.
		"""
		if self.reset_mode == 'cpu':
			self.cpu_env.reset()
			self._reset_from_cpu()
		else:
			self._reset_defaults()
			self.task.init_episode(self)
		self._refresh_static_geoms()
		# dm_control after_reset(): forward with actuation disabled (fills derived quantities for the obs)
		self._run('forward', self._forward_noact)
		self.t = 0
		obs = self.task.observation()
		return obs if self.batched else obs[0]

	def step(self, action):
		"""Advance all worlds by one agent step (action_repeat control steps).

		Args:
			action: CUDA tensor (S, action_dim) [or (action_dim,) for the single-env API] in [-1, 1].
		Returns:
			(obs, reward, done, info): obs (S, obs_dim), reward (S,) summed over the action repeat (without
			the leading S for the single-env API), done is True after ep_len steps, info is {}.
			obs / reward are clones of the graph's output buffers, so they are not overwritten by the next step.
		"""
		self._act_in.copy_(torch.as_tensor(action, dtype=torch.float32, device='cuda').reshape(self._act_in.shape))
		if self.use_graph and self.graph_mode == 'split':
			self._g_ctrl.replay()
			for _ in range(self.action_repeat):
				wp.capture_launch(self._graphs['control'])
				self._g_rew.replay()
			self._g_obs.replay()
			obs, reward = self._obs_out.clone(), self._rew_out.clone()
		elif self._step_graph is not None:
			self._step_graph.replay()
			obs, reward = self._obs_out.clone(), self._rew_out.clone()
		else:
			obs, reward = self._agent_step_body()
		self.t += 1
		if not self.batched:
			obs, reward = obs[0], reward[0]
		return obs, reward, self.t == self.ep_len, {}

	def overflow(self):
		"""Returns (max contacts per world, max efc rows per world, nconmax, njmax) (host sync).

		Diagnostic for the per-world capacities in dmc_tasks.CAPACITY (used by bench/validate_env.py).
		"""
		wid = self.d.contact.worldid.numpy()[:int(self.d.nacon.numpy()[0])]
		nc = int(np.bincount(wid, minlength=self.nworld).max()) if wid.size else 0
		return nc, int(self.d.nefc.numpy().max()), self.d.naconmax // self.nworld, self.d.njmax


def make_env(cfg):
	"""GPU counterpart of tdmpc/src/env.py:make_env (state modality only).

	Builds a DMCWarpEnv from the config (num_seeds worlds; options cuda_graph_env, reset_mode,
	skip_substep_sensors, env_graph_mode) and fills in cfg.obs_shape / action_shape / action_dim.
	"""
	if cfg.modality != 'state':
		raise NotImplementedError('mujoco_warp port supports the state modality only')
	n = int(cfg.get('num_seeds', 1))
	env = DMCWarpEnv(cfg.task, cfg.seed, cfg.action_repeat, use_graph=cfg.get('cuda_graph_env', True),
					 reset_mode=cfg.get('reset_mode', 'gpu'), nworld=n, batched=n > 1,
					 skip_substep_sensors=cfg.get('skip_substep_sensors', False),
					 graph_mode=cfg.get('env_graph_mode', 'fused'))
	cfg.obs_shape = (env.obs_dim,)
	cfg.action_shape = (env.action_dim,)
	cfg.action_dim = env.action_dim
	return env
