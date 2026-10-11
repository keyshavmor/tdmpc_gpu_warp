"""Vectorised TD-MPC: S independent seeds (agents) of the same configuration in one process.

Used by src/train_vec.py, e.g.
	python src/train_vec.py task=walker-walk num_seeds=16 seed=1 horizon=5 [ckpt_unroll=true] [precision=bf16]

Every seed has its own TOLD model, target network, Adam states, prioritised replay buffer, planner
state and environment world; seeds never exchange information. The S agents are simply stored as
stacked tensors so that one kernel serves all seeds:
  * BatchedLinear: weight (S, out, in), y = baddbmm(b, x, W^T) == S independent nn.Linear layers,
  * BatchedLayerNorm: per-seed affine parameters (nn.LayerNorm maths, eps=1e-5),
  * Adam over stacked parameters == S independent Adams (Adam is element-wise),
  * gradient clipping per seed: ||g_s|| over all of seed s's parameters, exactly clip_grad_norm_,
  * losses: sum over seeds of each seed's loss (seed s's parameters only receive seed s's gradient),
  * PER sampling, CEM/MPPI top-k, score normalisation and elite sampling along each seed's own axis.
Initialisation: each seed's layers are orthogonally initialised independently (same distribution as
nn.init.orthogonal_ in the original), reward/Q heads zero-initialised as in the original.
Same GPU-side changes as tdmpc_graph.py (CUDA graphs for plan/update, torch.multinomial instead of
np.random.choice, Adam(capturable=True), metrics as tensors).

Optional reduced precision (precision study only; default fp32 = original numerics):
  cfg.precision = 'fp32' | 'tf32' (TF32 tensor-core matmuls) | 'bf16' | 'fp16' (torch.autocast for all
  forward passes of plan() and update(); fp16 additionally uses dynamic loss scaling with GradScaler's
  default schedule, implemented capturably per seed: overflowed seeds skip the optimiser step)
  cfg.buffer_dtype = 'fp32' | 'bf16' | 'fp16' (storage dtype of observations in the replay buffer)

Optional exact-maths speed optimisation:
  cfg.batched_update = true: only the latent dynamics rollout of update() is sequential in t; the
  Q-heads on (z_t, a_t), the target encoder / TD targets and the policy loss over all H(+1) latents are
  evaluated in ONE batched call per network (same per-step terms, same sums; only the floating-point
  summation order and the random stream of the policy noise differ) -> ~H x fewer kernels, larger GEMMs.

Optional exact-maths memory optimisation:
  cfg.ckpt_unroll = true: activation checkpointing of every unroll step of update() (Q, dynamics,
  reward) and of the policy update; activations are recomputed in the backward pass instead of being
  stored (the recomputed forward is deterministic -> identical gradients), memory ~flat in H.

Other switches: cfg.cuda_graph_agent = false runs plan()/update() eagerly (debugging);
cfg.eval_graphs = false plans eagerly during evaluation only.

Equivalence (docs/01, docs/06): with the weights of an original agent loaded into a seed slot
(load_seed_from), per-seed parameters match algorithm/tdmpc.py after 12 updates up to float32 rounding
(3e-6..1e-4 incl. Adam amplification) and plan() gives the same action distributions; ckpt_unroll is
bitwise identical to the non-checkpointed agent. PER sampling uses torch.multinomial with
`replacement=not full`, the same flag as the original np.random.choice(replace=not full).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import contextlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import algorithm.helper as h


# ---------------------------------------------------------------------------------------- layers
class BatchedLinear(nn.Module):
	"""S independent nn.Linear(in_dim, out_dim) layers stored as one stacked weight (S, out, in).

	forward: x (S, B, in) -> (S, B, out), one baddbmm for all seeds (row s only sees weight[s]).
	"""
	def __init__(self, S, in_dim, out_dim):
		super().__init__()
		self.weight = nn.Parameter(torch.empty(S, out_dim, in_dim))
		self.bias = nn.Parameter(torch.zeros(S, out_dim))
		for s in range(S):  # == h.orthogonal_init on S independent nn.Linear layers
			nn.init.orthogonal_(self.weight.data[s])

	def forward(self, x):  # x: (S, B, in)
		"""x (S, B, in) -> (S, B, out): per-seed affine map bias[s] + x[s] @ weight[s]^T."""
		# y[s] = bias[s] + x[s] @ weight[s]^T  ==  nn.Linear of seed s
		return torch.baddbmm(self.bias.unsqueeze(1), x, self.weight.transpose(1, 2))


class BatchedLayerNorm(nn.Module):
	"""S independent nn.LayerNorm(dim) layers: shared normalisation, per-seed affine (S, dim).

	forward: x (S, B, dim) -> (S, B, dim).
	"""
	def __init__(self, S, dim, eps=1e-5):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(S, dim))
		self.bias = nn.Parameter(torch.zeros(S, dim))
		self.dim, self.eps = dim, eps

	def forward(self, x):
		"""x (S, B, dim) -> layer norm over dim with the per-seed affine parameters of seed s."""
		return F.layer_norm(x, (self.dim,), eps=self.eps) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)


def b_mlp(S, in_dim, mlp_dim, out_dim):
	"""S-seed counterpart of algorithm.helper.mlp (Linear-ELU-Linear-ELU-Linear)."""
	return nn.Sequential(BatchedLinear(S, in_dim, mlp_dim), nn.ELU(),
						 BatchedLinear(S, mlp_dim, mlp_dim), nn.ELU(),
						 BatchedLinear(S, mlp_dim, out_dim))


def b_q(S, cfg):
	"""S-seed counterpart of algorithm.helper.q (Linear-LayerNorm-Tanh-Linear-ELU-Linear)."""
	return nn.Sequential(BatchedLinear(S, cfg.latent_dim + cfg.action_dim, cfg.mlp_dim), BatchedLayerNorm(S, cfg.mlp_dim), nn.Tanh(),
						 BatchedLinear(S, cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 BatchedLinear(S, cfg.mlp_dim, 1))


def b_enc(S, cfg):
	"""S-seed counterpart of algorithm.helper.enc for the state modality (the only one supported)."""
	assert cfg.modality == 'state'
	return nn.Sequential(BatchedLinear(S, cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
						 BatchedLinear(S, cfg.enc_dim, cfg.latent_dim))


def truncated_normal_sample(mu, std, clip=0.3, low=-1.0, high=1.0, eps=1e-6):
	"""h.TruncatedNormal(mu, std).sample(clip) (same maths, incl. straight-through clamp)."""
	e = torch.randn_like(mu) * std
	e = torch.clamp(e, -clip, clip)
	x = mu + e
	clamped = torch.clamp(x, low + eps, high - eps)
	return x - x.detach() + clamped.detach()


class VecTOLD(nn.Module):
	"""S independent TOLD models (algorithm.tdmpc.TOLD) with stacked per-seed weights.

	All methods take and return tensors with a leading seed axis, e.g. h: obs (S, B, obs_dim) ->
	z (S, B, latent_dim); next/Q: (S, B, latent), (S, B, A) -> (S, B, .).
	"""
	def __init__(self, cfg, S):
		super().__init__()
		self.cfg, self.S = cfg, S
		self._encoder = b_enc(S, cfg)
		self._dynamics = b_mlp(S, cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = b_mlp(S, cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = b_mlp(S, cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = b_q(S, cfg), b_q(S, cfg)
		# zero-initialised output layers of reward and Q heads, as in the original TOLD
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Enable/disable gradient tracking of the Q-networks (all seeds)."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encode observations (S, B, obs_dim) into latents (S, B, latent_dim)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predict next latent (S, B, latent_dim) and reward (S, B, 1)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Policy action (S, B, A): tanh mean, plus clipped truncated-normal noise if std > 0."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			return truncated_normal_sample(mu, std)
		return mu

	def Q(self, z, a):
		"""Both Q-values (S, B, 1) each."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)


def _tb(x):
	"""(T, S, B, ...) -> (S, T*B, ...), t-major blocks."""
	T, S, B = x.shape[:3]
	return x.transpose(0, 1).reshape(S, T * B, *x.shape[3:])


def _untb(x, T):
	"""(S, T*B, ...) -> (T, S, B, ...)."""
	S, TB = x.shape[:2]
	return x.reshape(S, T, TB // T, *x.shape[2:]).transpose(0, 1)


def clip_grad_norm_per_seed(params, max_norm):
	"""torch.nn.utils.clip_grad_norm_ applied to every seed separately; returns (S,) total norms."""
	grads = [p.grad for p in params if p.grad is not None]  # None grads are skipped, as in clip_grad_norm_
	sq = torch.stack([g.pow(2).flatten(1).sum(1) for g in grads]).sum(0)  # (S,)
	total = sq.sqrt()
	coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)  # same formula as clip_grad_norm_, per seed
	for g in grads:
		g.mul_(coef.view(-1, *([1] * (g.dim() - 1))))
	return total


class ScaledAdam:
	"""torch.optim.Adam maths (betas, eps, bias correction; no weight decay / amsgrad) for stacked
	per-seed parameters, with per-seed dynamic loss scaling (torch.cuda.amp.GradScaler defaults:
	init 2**16, backoff 0.5, growth 2 every 2000 finite steps). A seed whose gradients overflow skips
	the step entirely (parameters, moments and step count unchanged), as GradScaler does.
	Fully on-device (no host sync) so that it can be captured in a CUDA graph."""
	def __init__(self, params, S, lr, betas=(0.9, 0.999), eps=1e-8, init_scale=2.0 ** 16, growth_interval=2000):
		self.params = list(params)
		self.S, self.lr, (self.b1, self.b2), self.eps = S, lr, betas, eps
		dev = self.params[0].device
		self.m = [torch.zeros_like(p) for p in self.params]
		self.v = [torch.zeros_like(p) for p in self.params]
		self.step_t = torch.zeros(S, device=dev)
		self.scale = torch.full((S,), init_scale, device=dev)
		self.growth = torch.zeros(S, device=dev)
		self.growth_interval = growth_interval

	def zero_grad(self, set_to_none=True):
		"""Set all gradients to None (the argument exists for API compatibility only)."""
		for p in self.params:
			p.grad = None

	def scale_loss(self, per_seed_loss):
		"""Scalar to backpropagate: sum_s scale[s] * loss[s]; per_seed_loss has shape (S,)."""
		return (per_seed_loss * self.scale).sum()

	def unscale_(self):
		"""Divide grads by the scale; returns per-seed `finite` mask (S,)."""
		inv = 1.0 / self.scale
		finite = torch.ones(self.S, dtype=torch.bool, device=self.scale.device)
		for p in self.params:
			if p.grad is None:
				continue
			bshape = (-1,) + (1,) * (p.grad.dim() - 1)
			p.grad.mul_(inv.view(bshape))
			finite &= torch.isfinite(p.grad).flatten(1).all(1)
		self._finite = finite
		return finite

	@torch.no_grad()
	def step(self):
		"""Masked Adam step (call after unscale_()); seeds with non-finite gradients are left unchanged.

		Also advances the per-seed loss-scale schedule. All updates are in place on persistent tensors
		and use torch.where instead of Python branching, so the step can be captured in a CUDA graph.
		"""
		ok = self._finite.float()
		self.step_t += ok  # per-seed step count: only seeds that actually take a step advance
		# clamp(min=1) avoids a 0 bias correction (division by 0) for seeds that never stepped; their
		# results are discarded by the mask below anyway
		bc1 = 1 - self.b1 ** self.step_t.clamp(min=1)
		bc2 = 1 - self.b2 ** self.step_t.clamp(min=1)
		for p, m, v in zip(self.params, self.m, self.v):
			if p.grad is None:
				continue
			bshape = (-1,) + (1,) * (p.dim() - 1)
			okb = ok.view(bshape).bool()
			# zero the gradient of overflowed seeds so inf/nan never enters the moment updates
			g = torch.where(okb, p.grad, torch.zeros_like(p.grad))
			m_new = self.b1 * m + (1 - self.b1) * g
			v_new = self.b2 * v + (1 - self.b2) * g * g
			# torch.optim.Adam (non-capturable form): p -= lr / bc1 * m / (sqrt(v) / sqrt(bc2) + eps)
			denom = v_new.sqrt() / bc2.sqrt().view(bshape) + self.eps
			p_new = p - (self.lr / bc1).view(bshape) * m_new / denom
			p.copy_(torch.where(okb, p_new, p)); m.copy_(torch.where(okb, m_new, m)); v.copy_(torch.where(okb, v_new, v))
		# scale schedule (in-place updates: the persistent tensors must change across CUDA-graph replays)
		# GradScaler semantics per seed: overflow -> scale * 0.5 and reset the growth counter;
		# growth_interval consecutive finite steps -> scale * 2 and reset the counter
		self.growth.copy_(torch.where(self._finite, self.growth + 1, torch.zeros_like(self.growth)))
		grow = self.growth >= self.growth_interval
		self.scale.copy_(torch.where(~self._finite, self.scale * 0.5, torch.where(grow, self.scale * 2.0, self.scale)))
		self.growth.copy_(torch.where(grow, torch.zeros_like(self.growth), self.growth))


# ---------------------------------------------------------------------------------------- buffer
class VecEpisode:
	"""Storage for one episode of all S seeds (obs: (L+1, S, obs_dim)).

	S-seed counterpart of algorithm.helper.Episode: action (L, S, A), reward (L, S), cumulative_reward
	(S,); all seeds step in lockstep, so one `done` flag (from the env, a host bool) serves all seeds.
	"""
	def __init__(self, cfg, S, init_obs):
		L = cfg.episode_length
		self.obs = torch.empty((L + 1, S, *cfg.obs_shape), dtype=torch.float32, device='cuda')
		self.obs[0] = init_obs
		self.action = torch.empty((L, S, cfg.action_dim), dtype=torch.float32, device='cuda')
		self.reward = torch.empty((L, S), dtype=torch.float32, device='cuda')
		self.cumulative_reward = torch.zeros(S, device='cuda')
		self.done, self._idx, self.L = False, 0, L

	def __len__(self):
		return self._idx

	@property
	def first(self):
		"""True if no transition has been added to the episode yet."""
		return self._idx == 0

	def __add__(self, transition):
		"""Append (obs (S, obs_dim), action (S, A), reward (S,), done) for all seeds."""
		obs, action, reward, done = transition
		self.obs[self._idx + 1] = obs
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		self.done = done
		self._idx += 1
		return self


class VecReplayBuffer:
	"""S independent copies of algorithm.helper.ReplayBuffer (state modality), GPU-resident.

	Every storage tensor has a leading seed axis (S, capacity, ...). Since all seeds add episodes in
	lockstep, the write index / fill state are shared; priorities and sampling are per seed. sample()
	is graph-capturable (no host sync, static shapes), like GraphReplayBuffer.sample().
	"""
	def __init__(self, cfg, S):
		self.cfg, self.S = cfg, S
		self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
		dev = torch.device('cuda')
		odt = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[cfg.get('buffer_dtype', 'fp32')]
		self._obs = torch.empty((S, self.capacity + 1, *cfg.obs_shape), dtype=odt, device=dev)
		self._last_obs = torch.empty((S, self.capacity // cfg.episode_length, *cfg.obs_shape), dtype=odt, device=dev)
		self._action = torch.empty((S, self.capacity, cfg.action_dim), dtype=torch.float32, device=dev)
		self._reward = torch.empty((S, self.capacity), dtype=torch.float32, device=dev)
		self._priorities = torch.ones((S, self.capacity), dtype=torch.float32, device=dev)
		self._eps = 1e-6
		self._full = False
		self.idx = 0
		self._n_valid = torch.zeros((), dtype=torch.float32, device=dev)  # filled slots (device scalar)
		self._arange = torch.arange(self.capacity, device=dev)
		self._seed_ar = torch.arange(S, device=dev).unsqueeze(1)  # (S, 1): x[_seed_ar, idxs] gathers per seed
		self._n_last = self._last_obs.shape[1]

	def __add__(self, episode):
		self.add(episode)
		return self

	def add(self, episode):
		"""Store one VecEpisode for all seeds (original ReplayBuffer.add per seed, without host syncs).

		New transitions get each seed's current max priority; the last `horizon` steps of the episode
		get priority 0 (no full H-step continuation), as in the original.
		"""
		cfg, L = self.cfg, self.cfg.episode_length
		self._obs[:, self.idx:self.idx + L] = episode.obs[:-1].transpose(0, 1)
		self._last_obs[:, self.idx // L] = episode.obs[-1]
		self._action[:, self.idx:self.idx + L] = episode.action.transpose(0, 1)
		self._reward[:, self.idx:self.idx + L] = episode.reward.transpose(0, 1)
		if self._full:
			max_priority = self._priorities.max(1).values
		elif self.idx == 0:
			max_priority = torch.ones(self.S, device='cuda')
		else:
			max_priority = self._priorities[:, :self.idx].max(1).values
		new_priorities = max_priority.unsqueeze(1).expand(self.S, L).clone()
		new_priorities[:, L - cfg.horizon:] = 0
		self._priorities[:, self.idx:self.idx + L] = new_priorities
		self.idx = (self.idx + L) % self.capacity
		self._full = self._full or self.idx == 0
		self._n_valid.fill_(float(self.capacity if self._full else self.idx))

	def update_priorities(self, idxs, priorities):
		"""Set the PER priorities of the sampled indices idxs (S, B) to priorities (S, B, 1) + eps."""
		self._priorities[self._seed_ar, idxs] = priorities.squeeze(-1) + self._eps

	def sample(self):
		"""Draw one PER batch per seed (same per-seed maths as GraphReplayBuffer.sample()).

		Returns obs (S, B, obs_dim), next_obs (H+1, S, B, obs_dim), action (H+1, S, B, A),
		reward (H+1, S, B, 1), idxs (S, B), weights (S, B); observations are returned in float32
		whatever the storage dtype.
		"""
		cfg, S, B = self.cfg, self.S, self.cfg.batch_size
		# (S, capacity) probabilities, unfilled slots masked to 0, normalised per seed (row)
		probs = torch.where(self._arange < self._n_valid, self._priorities ** cfg.per_alpha, torch.zeros_like(self._priorities))
		probs = probs / probs.sum(1, keepdim=True)
		idxs = torch.multinomial(probs, B, replacement=not self._full)      # (S, B); == replace=not full in the original
		weights = (self._n_valid * probs.gather(1, idxs)) ** (-cfg.per_beta)
		weights = weights / weights.max(1, keepdim=True).values
		sa = self._seed_ar
		obs = self._obs[sa, idxs].float()                                   # (S, B, obs)
		next_obs = torch.empty((cfg.horizon + 1, S, B, *cfg.obs_shape), dtype=torch.float32, device=obs.device)
		action = torch.empty((cfg.horizon + 1, S, B, cfg.action_dim), dtype=torch.float32, device=obs.device)
		reward = torch.empty((cfg.horizon + 1, S, B), dtype=torch.float32, device=obs.device)
		for t in range(cfg.horizon + 1):
			_idxs = idxs + t
			next_obs[t] = self._obs[sa, _idxs + 1]
			action[t] = self._action[sa, _idxs]
			reward[t] = self._reward[sa, _idxs]
		# windows ending at an episode boundary take the stored final observation (torch.where instead
		# of boolean-mask assignment; clamped index keeps the unused gathers in range)
		mask = (_idxs + 1) % cfg.episode_length == 0
		last = self._last_obs[sa,(_idxs // cfg.episode_length).clamp(max=self._n_last - 1)].float()
		next_obs[-1] = torch.where(mask.unsqueeze(-1), last, next_obs[-1])
		return obs, next_obs, action, reward.unsqueeze(-1), idxs, weights


# ---------------------------------------------------------------------------------------- agent
class VecTDMPC:
	"""S independent TD-MPC agents (algorithm.tdmpc.TDMPC maths) in stacked form.

	plan(): obs (S, obs_dim) -> actions (S, A); update(): one update of every seed, per-seed metrics
	(S,). CUDA-graph handling as in GraphTDMPC (plan graph per (horizon, eval_mode), update graph after
	3 eager warm-up updates). Precision / batching / checkpointing options: see the module docstring.
	"""
	def __init__(self, cfg, S):
		self.cfg, self.S = cfg, S
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = VecTOLD(cfg, S).cuda()
		self.model_target = VecTOLD(cfg, S).cuda()
		self.model_target.load_state_dict(self.model.state_dict())
		self.precision = cfg.get('precision', 'fp32')
		assert self.precision in ('fp32', 'tf32', 'bf16', 'fp16'), self.precision
		# process-wide switches: TF32 only for precision=tf32 (bf16/fp16 use autocast, see _amp())
		torch.backends.cuda.matmul.allow_tf32 = self.precision == 'tf32'
		torch.backends.cudnn.allow_tf32 = self.precision == 'tf32'
		# fp16 needs loss scaling -> masked per-seed Adam (ScaledAdam); otherwise capturable torch Adam
		self._scaled = self.precision == 'fp16' or cfg.get('force_scaled_adam', False)  # (flag: test only)
		if self._scaled:
			self.optim = ScaledAdam(self.model.parameters(), S, cfg.lr)
			self.pi_optim = ScaledAdam(self.model._pi.parameters(), S, cfg.lr)
		else:
			self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, capturable=True)
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=cfg.lr, capturable=True)
		self.model.eval(); self.model_target.eval()
		A = cfg.action_dim
		# persistent graph inputs (written in place before replay), as in GraphTDMPC
		self._std_t = torch.full((), float(self.std), device=self.device)
		self._prev_mean_buf = torch.zeros(cfg.horizon, S, A, device=self.device)
		self._obs_in = torch.zeros(S, *cfg.obs_shape, device=self.device)
		self._t0_in = torch.ones((), dtype=torch.bool, device=self.device)
		self._plan_graphs, self._plan_out, self._plan_seen = {}, {}, set()
		self._pool = torch.cuda.graph_pool_handle()
		self._upd_graph, self._upd_out, self._upd_calls, self._upd_full = None, None, 0, None
		self._ema_src = list(self.model.parameters())
		self._ema_dst = list(self.model_target.parameters())
		self.use_graphs = cfg.get('cuda_graph_agent', True)
		# rho^t weights for the batched update (created once: no host->device copies inside CUDA graphs)
		self._rho_upd = torch.tensor([cfg.rho ** t for t in range(cfg.horizon)], device=self.device).view(-1, 1, 1, 1)
		self._rho_pi = torch.tensor([cfg.rho ** t for t in range(cfg.horizon + 1)], device=self.device)

	def _amp(self):
		"""Autocast context for bf16/fp16 forward passes; a no-op context for fp32/tf32."""
		if self.precision in ('bf16', 'fp16'):
			return torch.autocast('cuda', dtype=torch.bfloat16 if self.precision == 'bf16' else torch.float16)
		return contextlib.nullcontext()

	# ----------------------------------------------------------------------------- planning
	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""TDMPC.estimate_value for all seeds: z (S, N, latent), actions (H, S, N, A) -> G (S, N, 1)."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward.float()  # .float(): returns accumulate in fp32 under autocast (no-op in fp32)
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std))).float()
		return G

	@torch.no_grad()
	def _plan_body(self, horizon, eval_mode):
		"""Graph-capturable planning step for all seeds (under autocast if enabled); returns (S, A) fp32."""
		with self._amp():
			return self._plan_body_impl(horizon, eval_mode).float()

	def _plan_body_impl(self, horizon, eval_mode):
		"""Same CEM/MPPI iterations as GraphTDMPC._plan_body, with an extra seed axis.

		Every reduction (top-k elites, score normalisation, mean/std, elite sampling) runs along each
		seed's own sample axis, so the seeds' planners are independent.
		"""
		cfg, S, A = self.cfg, self.S, self.cfg.action_dim
		obs = self._obs_in.unsqueeze(1)                                    # (S, 1, obs)
		num_pi_trajs = int(cfg.mixture_coef * cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, S, num_pi_trajs, A, device=self.device)
			z = self.model.h(obs).repeat(1, num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])
		N = cfg.num_samples + num_pi_trajs
		z = self.model.h(obs).repeat(1, N, 1)                             # (S, N, latent)
		mean = torch.zeros(horizon, S, A, device=self.device)
		std = 2 * torch.ones(horizon, S, A, device=self.device)
		# warm start from the previous step's mean unless t0 (original: `if not t0 and hasattr(...)`)
		mean[:-1] = torch.where(self._t0_in, mean[:-1], self._prev_mean_buf[1:horizon])
		for i in range(cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(2) + std.unsqueeze(2) *
								  torch.randn(horizon, S, cfg.num_samples, A, device=self.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=2)         # (H, S, N, A)
			value = self.estimate_value(z, actions, horizon).float().nan_to_num_(0)  # (S, N, 1)
			elite_idxs = torch.topk(value.squeeze(-1), cfg.num_elites, dim=1).indices  # (S, E)
			elite_value = value.gather(1, elite_idxs.unsqueeze(-1))       # (S, E, 1)
			elite_actions = actions.gather(2, elite_idxs.view(1, S, -1, 1).expand(horizon, S, cfg.num_elites, A))
			max_value = elite_value.max(1, keepdim=True).values
			score = torch.exp(cfg.temperature * (elite_value - max_value))
			score /= score.sum(1, keepdim=True)                           # (S, E, 1)
			ssum = score.sum(1).unsqueeze(0)                              # (1, S, 1)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=2) / (ssum + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(2)) ** 2, dim=2) / (ssum + 1e-9))
			_std = torch.maximum(_std, self._std_t).clamp_(max=2)  # == _std.clamp_(self.std, 2)
			mean, std = cfg.momentum * mean + (1 - cfg.momentum) * _mean, _std
		# one elite trajectory per seed ~ Categorical(score)  (np.random.choice(p=score) in the original)
		choice = torch.multinomial(score.squeeze(-1), 1)                  # (S, 1)
		actions = elite_actions.gather(2, choice.view(1, S, 1, 1).expand(horizon, S, 1, A)).squeeze(2)  # (H, S, A)
		self._prev_mean_buf[:horizon] = mean
		a = actions[0]
		if not eval_mode:
			a = a + _std[0] * torch.randn(S, A, device=self.device)
		return a

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""obs: (S, obs_dim) -> actions (S, action_dim).

		Same arguments and semantics as TDMPC.plan(); `step` and `t0` are shared by all seeds (lockstep).
		Graph handling as in GraphTDMPC.plan(): eager warm-up, capture on the second call per
		(horizon, eval_mode), then replay; the returned tensor is a fresh copy.
		"""
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.S, self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		self._obs_in.copy_(obs)
		self._t0_in.fill_(bool(t0))
		key = (horizon, bool(eval_mode))
		if not self.use_graphs or (eval_mode and not self.cfg.get('eval_graphs', True)):
			return self._plan_body(horizon, eval_mode)
		g = self._plan_graphs.get(key)
		if g is None:
			if key not in self._plan_seen:
				self._plan_seen.add(key)
				return self._plan_body(horizon, eval_mode).clone()
			g = torch.cuda.CUDAGraph()
			with torch.cuda.graph(g, pool=self._pool):
				self._plan_out[key] = self._plan_body(horizon, eval_mode)
			self._plan_graphs[key] = g
		g.replay()
		return self._plan_out[key].clone()

	# ----------------------------------------------------------------------------- learning
	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""TDMPC._td_target per seed: reward + discount * min(Q1, Q2)_target(h(next_obs), pi(.))."""
		next_z = self.model.h(next_obs)
		return reward + self.cfg.discount * torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))

	def update_pi(self, zs):
		"""Policy update of all seeds from the H+1 detached latents zs (each (S, B, latent)).

		pi_loss[s] = sum_t rho^t * -mean(min Q(z_t, pi(z_t))) as in TDMPC.update_pi; the backward pass
		uses the sum over seeds (disjoint parameter slices, so each seed gets exactly its own gradient).
		Returns the per-seed loss (S,).
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		pi_loss = 0
		ck = self.cfg.get('ckpt_unroll', False)
		if self.cfg.get('batched_update', False) and not ck:
			T = len(zs)
			rho = self._rho_pi[:T]
			with self._amp():
				Z = _tb(torch.stack(zs))                                       # (S, T*B, latent)
				a = self.model.pi(Z, self.cfg.min_std)
				Q = torch.min(*self.model.Q(Z, a)).float()                     # (S, T*B, 1)
				Qm = _untb(Q, T).mean(dim=(2, 3))                              # (T, S)
				pi_loss = -(Qm * rho.unsqueeze(1)).sum(0)                      # (S,)
		else:
			with self._amp():
				for t, z in enumerate(zs):
					if ck:  # the policy noise is drawn outside the checkpointed part (identical recompute)
						eps = torch.randn(*z.shape[:-1], self.cfg.action_dim, device=z.device)
						Q = checkpoint(self._pi_q, z, eps, use_reentrant=False)
					else:
						a = self.model.pi(z, self.cfg.min_std)
						Q = torch.min(*self.model.Q(z, a)).float()
					pi_loss += -Q.mean(dim=(1, 2)) * (self.cfg.rho ** t)          # (S,)
		if self._scaled:  # fp16: backprop the scaled loss, then unscale and detect overflow per seed
			self.pi_optim.scale_loss(pi_loss).backward()
			self.pi_optim.unscale_()
		else:
			pi_loss.sum().backward()
		clip_grad_norm_per_seed(list(self.model._pi.parameters()), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.detach()

	def _pi_q(self, z, eps):
		"""min(Q1, Q2)(z, pi(z, min_std)) with given standard-normal noise (== model.pi sampling)."""
		mu = torch.tanh(self.model._pi(z))
		std = self.cfg.min_std
		if std > 0:
			e = torch.clamp(eps * std, -0.3, 0.3)
			x = mu + e
			a = x - x.detach() + torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6).detach()
		else:
			a = mu
		return torch.min(*self.model.Q(z, a)).float()

	def _unroll_step(self, z, a):
		"""One latent unroll step (Q1, Q2, next z, reward prediction); the unit of activation checkpointing."""
		Q1, Q2 = self.model.Q(z, a)
		z_next, reward_pred = self.model.next(z, a)
		return Q1, Q2, z_next, reward_pred

	def _update_body(self, replay_buffer):
		"""One TOLD + policy update of all seeds (captured into the update graph).

		Per seed the same losses as TDMPC.update() (consistency, reward, value; PER-weighted, 1/H
		gradient hook, clipping, Adam, priority update, update_pi); three code paths compute the
		unroll: sequential (original order), sequential with activation checkpointing (ckpt_unroll)
		or batched over t (batched_update). The state modality has no augmentation (aug is the
		identity in the original), so it is omitted. Returns a dict of (S,) tensors.
		"""
		cfg = self.cfg
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		# autocast context entered/exited manually around the forward part only (no-op in fp32/tf32)
		amp = self._amp()
		amp.__enter__()
		z = self.model.h(obs)
		zs = [z.detach()]
		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		ck = cfg.get('ckpt_unroll', False)
		if cfg.get('batched_update', False) and not ck:
			H = cfg.horizon
			rho = self._rho_upd
			zs_g, rps = [z], []
			for t in range(H):  # the only sequential part: latent dynamics rollout
				z, reward_pred = self.model.next(z, action[t])
				zs_g.append(z); rps.append(reward_pred)
			Q1, Q2 = self.model.Q(_tb(torch.stack(zs_g[:-1])), _tb(action[:H]))
			Q1, Q2 = _untb(Q1, H), _untb(Q2, H)                                # (H, S, B, 1)
			with torch.no_grad():
				NO = _tb(next_obses[:H])                                      # (S, H*B, obs)
				next_z = _untb(self.model_target.h(NO), H)                    # (H, S, B, latent)
				td_target = _untb(self._td_target(NO, _tb(reward[:H])), H)    # (H, S, B, 1)
			Zn, RP = torch.stack(zs_g[1:]), torch.stack(rps)
			zs = [x.detach() for x in zs_g]
			# rho (H, 1, 1, 1) weights each step; the sum over t gives the same per-step terms as the loop
			consistency_loss = (rho * torch.mean(h.mse(Zn, next_z), dim=-1, keepdim=True)).sum(0)
			reward_loss = (rho * h.mse(RP, reward[:H])).sum(0)
			value_loss = (rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))).sum(0)
			priority_loss = (rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))).sum(0)
		# sequential unroll (original order); skipped when the batched path above was taken
		for t in (range(cfg.horizon) if not (cfg.get('batched_update', False) and not ck) else []):
			if ck:  # keep only the step inputs; activations are recomputed during backward
				Q1, Q2, z, reward_pred = checkpoint(self._unroll_step, z, action[t], use_reentrant=False)
			else:
				Q1, Q2 = self.model.Q(z, action[t])
				z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = next_obses[t]
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())
			rho = (cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=-1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))
		amp.__exit__(None, None, None)
		# losses and their combination in fp32 (no-ops in fp32 mode)
		consistency_loss, reward_loss = consistency_loss.float(), reward_loss.float()
		value_loss, priority_loss = value_loss.float(), priority_loss.float()
		total_loss = cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 cfg.value_coef * value_loss.clamp(max=1e4)              # (S, B, 1)
		weighted_loss = (total_loss.squeeze(-1) * weights).mean(dim=1)     # (S,)
		weighted_loss.register_hook(lambda grad: grad * (1 / cfg.horizon))
		if self._scaled:
			self.optim.scale_loss(weighted_loss).backward()
			self.optim.unscale_()
		else:
			# sum over seeds: seed s's parameters only depend on seed s's loss -> independent gradients
			weighted_loss.sum().backward()
		grad_norm = clip_grad_norm_per_seed(list(self.model.parameters()), cfg.grad_clip_norm)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())
		pi_loss = self.update_pi(zs)
		return {'consistency_loss': consistency_loss.mean(dim=(1, 2)).detach(),
				'reward_loss': reward_loss.mean(dim=(1, 2)).detach(),
				'value_loss': value_loss.mean(dim=(1, 2)).detach(),
				'pi_loss': pi_loss,
				'total_loss': total_loss.mean(dim=(1, 2)).detach(),
				'weighted_loss': weighted_loss.detach(),
				'grad_norm': grad_norm.detach()}

	def update(self, replay_buffer, step):
		"""One update of every seed; returns a dict of (S,) CUDA tensors (per-seed metrics).

		With CUDA graphs (default): 3 eager warm-up updates, capture on the 4th call, replay afterwards;
		re-captured once when the buffer becomes full (the multinomial replacement flag is baked into
		the graph). The target-network EMA runs outside the graph every `update_freq` steps.
		"""
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self._std_t.fill_(float(self.std))
		self.model.train()
		if not self.use_graphs:
			out = self._update_body(replay_buffer)
		else:
			if self._upd_graph is not None and self._upd_full != replay_buffer._full:
				self._upd_graph, self._upd_calls = None, 0
			if self._upd_graph is None:
				self._upd_calls += 1
				if self._upd_calls <= 3:
					out = self._update_body(replay_buffer)
				else:
					self._upd_graph = torch.cuda.CUDAGraph()
					self._upd_full = replay_buffer._full
					with torch.cuda.graph(self._upd_graph):
						self._upd_out = self._update_body(replay_buffer)
					self._upd_graph.replay()  # capture only records the work: run it once now
					out = self._upd_out
			else:
				self._upd_graph.replay()
				out = self._upd_out
		if step % self.cfg.update_freq == 0:
			with torch.no_grad():
				torch._foreach_lerp_(self._ema_dst, self._ema_src, self.cfg.tau)
		self.model.eval()
		return dict(out)

	# ----------------------------------------------------------------------------- helpers
	def load_seed_from(self, s, agent):
		"""Copy the weights of an original algorithm.tdmpc.TDMPC agent into seed slot s (for tests)."""
		def copy(dst_mod, src_mod):
			"""Copy every parameter of src_mod into slot s of the matching stacked parameter of dst_mod (same parameter order)."""
			for (n1, p1), (n2, p2) in zip(dst_mod.named_parameters(), src_mod.named_parameters()):
				p1.data[s].copy_(p2.data)
		copy(self.model, agent.model)
		copy(self.model_target, agent.model_target)
