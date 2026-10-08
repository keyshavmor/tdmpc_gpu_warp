"""CUDA-graph version of TD-MPC: single-seed agent with captured plan() and update().

Opt-in in the single-seed training script (`graph_agent=true`), usually together with host pacing:
	python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true

Same TOLD model, losses, CEM/MPPI planner, PER buffer and optimiser maths as algorithm/tdmpc.py.
Changes are purely about *how* the GPU work is issued:
  * `plan()` is captured once per (planning horizon, eval_mode) as a CUDA graph and replayed
    -> one launch instead of ~500 small kernel launches per environment step,
  * `update()` (buffer sampling, TOLD forward/backward, grad clipping, Adam, priority update,
    policy update) is captured as one CUDA graph -> one launch instead of ~1000,
  * no host<->device synchronisation inside plan/update:
      - elite trajectory selection `np.random.choice(p=score)` -> `torch.multinomial(score, 1)`,
      - PER sampling `np.random.choice(total, B, p, replace=not full)` ->
        `torch.multinomial(probs, B, replacement=not full)` (GPU RNG; same flag as the original: with
        replacement until the buffer first wraps around, without replacement afterwards; same
        probabilities priority ** per_alpha and importance weights),
      - boolean-mask indexing -> torch.where, metrics returned as tensors (read once per episode),
  * Adam(capturable=True) (identical update rule, step counter kept on the device),
  * EMA via torch._foreach_lerp_ (identical maths, one launch).
The RNG *streams* differ from the original (GPU RNG instead of numpy); all deterministic computations
are the same. Checked against algorithm/tdmpc.py (docs/01, docs/06): parameters identical up to float32
rounding after 12 updates (2e-6..7e-6), bitwise identical to the eager agent over 200 updates, identical
plan() action distributions.

CUDA-graph rules followed below: tensors read by a captured graph are persistent buffers updated in
place (`_obs_in`, `_t0_in`, `_std_t`, `_prev_mean_buf`, the replay-buffer storage, `_n_valid`), shapes
are static (masks instead of slicing by data-dependent sizes), no `.item()` / `.cpu()` inside captured
code, and every graph is preceded by eager warm-up calls (allocator, cuBLAS workspaces, optimiser state).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import numpy as np
import torch
import algorithm.helper as h
from algorithm.tdmpc import TDMPC, TOLD
from algorithm.helper import ReplayBuffer


class GraphReplayBuffer(ReplayBuffer):
	"""PER buffer with a graph-capturable sample(); storage layout and add() identical to the original.

	State modality only. The storage tensors of the parent class are allocated once and only written
	in place, so a captured update() graph always reads the current buffer contents.
	"""
	def __init__(self, cfg):
		super().__init__(cfg)
		# number of filled slots as a device scalar (== `total = len(probs)` of the original sample())
		self._n_valid = torch.zeros((), dtype=torch.float32, device=self.device)
		self._arange = torch.arange(self.capacity, device=self.device)
		self._n_last = self._last_obs.shape[0]

	def add(self, episode):
		"""Store one episode (original add(), runs eagerly once per episode) and refresh `_n_valid`."""
		super().add(episode)
		self._n_valid.fill_(float(self.capacity if self._full else self.idx))

	def sample(self):
		"""Draw a PER batch without host synchronisation (safe inside CUDA-graph capture).

		Returns (obs (B, obs_dim), next_obs (H+1, B, obs_dim), action (H+1, B, A), reward (H+1, B, 1),
		idxs (B,), weights (B,)), the same tuple as ReplayBuffer.sample().
		"""
		cfg = self.cfg
		# fixed-size (capacity,) probabilities with unfilled slots masked to zero instead of slicing
		# priorities[:idx] (a data-dependent shape cannot be captured)
		probs = torch.where(self._arange < self._n_valid, self._priorities ** cfg.per_alpha, torch.zeros_like(self._priorities))
		probs = probs / probs.sum()
		# GPU replacement for np.random.choice (see the module docstring for the replacement flag)
		idxs = torch.multinomial(probs, cfg.batch_size, replacement=not self._full)  # == replace=not full in the original
		weights = (self._n_valid * probs[idxs]) ** (-cfg.per_beta)
		weights = weights / weights.max()

		obs = self._obs[idxs]
		next_obs = torch.empty((cfg.horizon+1, cfg.batch_size, *self._last_obs.shape[1:]), dtype=obs.dtype, device=obs.device)
		action = torch.empty((cfg.horizon+1, cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((cfg.horizon+1, cfg.batch_size), dtype=torch.float32, device=self.device)
		for t in range(cfg.horizon+1):
			_idxs = idxs + t
			next_obs[t] = self._obs[_idxs+1]
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]
		# original: next_obs[-1, mask] = last_obs[...] (boolean indexing -> data-dependent shape).
		# Here the last-observation gather is done for every row (index clamped to stay in range for
		# rows where it is not needed) and selected with torch.where.
		mask = (_idxs+1) % cfg.episode_length == 0
		last = self._last_obs[(_idxs // cfg.episode_length).clamp(max=self._n_last - 1)].float()
		next_obs[-1] = torch.where(mask.unsqueeze(1), last, next_obs[-1])
		return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


class GraphTDMPC(TDMPC):
	"""TD-MPC agent (original TOLD model and maths) whose plan() and update() are CUDA graphs.

	Drop-in replacement for algorithm.tdmpc.TDMPC with the same plan()/update() signatures, used
	together with GraphReplayBuffer. update() returns 0-d CUDA tensors instead of Python floats.
	"""
	def __init__(self, cfg):
		super().__init__(cfg)
		# capturable Adam: same update rule, step counter on the device (required for graph replay)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, capturable=True)
		A = cfg.action_dim
		dev = self.device
		# persistent graph inputs, written in place before each replay
		self._std_t = torch.full((), float(self.std), device=dev)          # scheduled CEM min std (self.std)
		self._prev_mean_buf = torch.zeros(cfg.horizon, A, device=dev)    # self._prev_mean of the original
		self._obs_in = torch.zeros(1, *cfg.obs_shape, device=dev)        # current observation
		self._t0_in = torch.ones((), dtype=torch.bool, device=dev)       # first step of an episode?
		# one plan graph per (horizon, eval_mode); all plan graphs share one memory pool (they are never
		# replayed concurrently and their outputs are cloned right after replay)
		self._plan_graphs, self._plan_out, self._plan_seen = {}, {}, set()
		self._pool = torch.cuda.graph_pool_handle()
		self._upd_graph, self._upd_out, self._upd_calls, self._upd_full = None, None, 0, None
		self._ema_src = list(self.model.parameters())
		self._ema_dst = list(self.model_target.parameters())

	# ------------------------------------------------------------------------------- planning
	@torch.no_grad()
	def _plan_body(self, horizon, eval_mode):
		"""Graph-capturable body of TDMPC.plan() for a fixed horizon; reads `_obs_in` / `_t0_in`.

		Identical CEM/MPPI iterations to the original; returns the action (action_dim,).
		"""
		cfg = self.cfg
		obs = self._obs_in
		num_pi_trajs = int(cfg.mixture_coef * cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		z = self.model.h(obs).repeat(cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, cfg.action_dim, device=self.device)
		# `if not t0 and hasattr(self, '_prev_mean'): mean[:-1] = self._prev_mean[1:]`
		mean[:-1] = torch.where(self._t0_in, mean[:-1], self._prev_mean_buf[1:horizon])

		for i in range(cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, cfg.num_samples, cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			max_value = elite_value.max(0)[0]
			score = torch.exp(cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = torch.maximum(_std, self._std_t).clamp_(max=2)  # == _std.clamp_(self.std, 2)
			mean, std = cfg.momentum * mean + (1 - cfg.momentum) * _mean, _std

		# elite trajectory ~ Categorical(score)  (np.random.choice(p=score) in the original)
		choice = torch.multinomial(score.squeeze(1), 1)
		actions = elite_actions.index_select(1, choice).squeeze(1)
		self._prev_mean_buf[:horizon] = mean
		a = actions[0]
		if not eval_mode:
			a = a + _std[0] * torch.randn(cfg.action_dim, device=std.device)
		return a

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""Plan the next action (same arguments and semantics as TDMPC.plan()).

		obs: observation (obs_dim,) CUDA tensor or array. Returns a fresh (action_dim,) CUDA tensor.
		The first call for a (horizon, eval_mode) key runs eagerly (warm-up), the second captures the
		graph, later calls only copy the inputs and replay it.
		"""
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		self._obs_in.copy_(torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(self._obs_in.shape))
		self._t0_in.fill_(bool(t0))
		key = (horizon, bool(eval_mode))
		if eval_mode and not self.cfg.get('eval_graphs', True):  # eager planning during evaluation
			return self._plan_body(horizon, eval_mode).clone()
		g = self._plan_graphs.get(key)
		if g is None:
			if key not in self._plan_seen:  # first call: run eagerly (warm-up), capture on the next call
				self._plan_seen.add(key)
				return self._plan_body(horizon, eval_mode).clone()
			g = torch.cuda.CUDAGraph()
			with torch.cuda.graph(g, pool=self._pool):
				self._plan_out[key] = self._plan_body(horizon, eval_mode)
			self._plan_graphs[key] = g
		g.replay()
		return self._plan_out[key].clone()  # clone: the static output is overwritten by the next replay

	# ------------------------------------------------------------------------------- learning
	def update_pi(self, zs):
		"""Original policy update; returns the loss as a 0-d tensor instead of `.item()` (no sync)."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		pi_loss = 0
		for t, z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.detach()

	def _update_body(self, replay_buffer):
		"""One TOLD + policy update (sample, losses, backward, clip, Adam, PER update, update_pi).

		Line-by-line the maths of TDMPC.update() without the EMA step and without host syncs; this is
		the part that is captured into the update graph. Returns a dict of 0-d tensors.
		"""
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]
		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())
		pi_loss = self.update_pi(zs)
		return {'consistency_loss': consistency_loss.mean().detach(),
				'reward_loss': reward_loss.mean().detach(),
				'value_loss': value_loss.mean().detach(),
				'pi_loss': pi_loss,
				'total_loss': total_loss.mean().detach(),
				'weighted_loss': weighted_loss.detach(),
				'grad_norm': grad_norm.detach()}

	def update(self, replay_buffer, step):
		"""One update step (same semantics as TDMPC.update()).

		Calls 1-3 run eagerly as genuine updates (warm-up: allocates Adam states and workspaces), call 4
		captures `_update_body` (capture does not execute the work, hence the replay right after) and
		all later calls replay the graph. The EMA of the target network is applied outside the graph
		every `update_freq` steps, as in the original. Returns a dict of 0-d CUDA tensors (call float()
		at logging time).
		"""
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self._std_t.fill_(float(self.std))  # device copy of the schedule value read by the plan graphs
		self.model.train()
		if self._upd_graph is not None and self._upd_full != replay_buffer._full:
			self._upd_graph = None  # sampling switches from with to without replacement once full
			self._upd_calls = 0
		if self._upd_graph is None:
			self._upd_calls += 1
			if self._upd_calls <= 3:  # genuine updates run eagerly as warm-up before capture
				out = self._update_body(replay_buffer)
			else:
				self._upd_graph = torch.cuda.CUDAGraph()
				self._upd_full = replay_buffer._full
				with torch.cuda.graph(self._upd_graph):
					self._upd_out = self._update_body(replay_buffer)
				self._upd_graph.replay()
				out = self._upd_out
		else:
			self._upd_graph.replay()
			out = self._upd_out
		if step % self.cfg.update_freq == 0:
			with torch.no_grad():
				torch._foreach_lerp_(self._ema_dst, self._ema_src, self.cfg.tau)
		self.model.eval()
		return dict(out)
