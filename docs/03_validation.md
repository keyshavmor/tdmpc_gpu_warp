# 3. Validation of the port and equivalence tests

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

All numbers below come from the final code (`results/validation/*.json`, `results/equivalence/*.json`).

## 3.1 Environment port vs dm_control (all 21 tasks) - `bench/validate_env.py`

* **A - function check**: GPU obs/reward vs dm_control's `get_observation` / `get_reward` evaluated on
  the *same* state (GPU state copied into CPU physics). Isolates the obs/reward port from physics.
* **B - reset distribution**: 100 GPU resets vs 100 dm_control resets: max |z| of the per-dimension mean
  difference (dims with negligible spread excluded) and min/max std ratio. With up to 223 dims a max
  |z| of ~3 is expected by chance.
* **C - physics**: open-loop rollouts with identical initial state (`reset_mode=cpu`) and identical random
  actions, max relative observation error after 1/10/50 agent steps; next to it the **chaos control**:
  the same rollout on CPU MuJoCo twice, the second copy with a 1e-6 perturbation of the initial velocity.
  Plus returns of the random policy for 3 episodes.
* **D - capacity**: maximum contacts and constraint rows used per world vs the allocated capacity.

| task | obs fn err (max abs) | reward fn err | reset mean \|z\| max | reset std ratio | open-loop rel. obs err @1 / @10 / @50 agent steps | CPU-vs-CPU chaos control (1e-6 perturbation) @10 / @50 | random-policy returns GPU vs CPU (3 eps) | contacts max/cap | efc rows max/cap |
|---|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | 2.4e-07 | 3.9e-07 | 1.08 | 0.96–1.04 | 2e-07 / 3e-07 / 7e-06 | 6e-07 / 1e-05 | [0, 0, 6] vs [0, 0, 6] | 0/8 | 0/16 |
| cartpole-swingup-sparse | 2.3e-07 | 0.0e+00 | 1.01 | 0.84–1.28 | 4e-07 / 5e-07 / 8e-06 | 1e-06 / 3e-06 | [0, 31, 0] vs [0, 31, 0] | 0/8 | 0/16 |
| cartpole-swingup | 2.3e-07 | 1.0e-07 | 1.01 | 0.84–1.28 | 4e-07 / 5e-07 / 8e-06 | 1e-06 / 3e-06 | [156, 314, 116] vs [156, 314, 116] | 0/8 | 0/16 |
| cheetah-run | 0.0e+00 | 6.9e-08 | 1.54 | 0.50–1.27 | 1e-06 / 4e-06 / 1e-05 | 1e-06 / 2e-06 | [5, 7, 18] vs [5, 7, 18] | 2/16 | 9/64 |
| cup-catch | 0.0e+00 | 0.0e+00 | 1.37 | 0.99–1.03 | 2e-07 / 2e-06 / 5e-06 | 1e-06 / 2e-07 | [0, 0, 0] vs [0, 0, 0] | 1/16 | 5/64 |
| dog-run | 5.7e-06 | 4.4e-08 | 2.93 | 0.86–1.17 | 5e-06 / 2e-01 / 3e+00 | 5e-06 / 2e-01 | [5, 7, 6] vs [5, 6, 4] | 24/128 | 92/640 |
| dog-trot | 6.8e-06 | 3.3e-08 | 2.93 | 0.86–1.17 | 5e-06 / 2e-01 / 2e+00 | 5e-06 / 2e-01 | [7, 10, 4] vs [7, 9, 5] | 16/128 | 58/640 |
| dog-walk | 5.1e-06 | 1.3e-07 | 2.93 | 0.86–1.17 | 5e-06 / 2e-01 / 1e+00 | 5e-06 / 2e-01 | [13, 15, 5] vs [13, 14, 6] | 16/128 | 65/640 |
| finger-spin | 2.9e-08 | 0.0e+00 | 0.97 | 0.96–1.07 | 2e-07 / 8e-07 / 1e-07 | 6e-08 / 4e-09 | [0, 0, 0] vs [0, 0, 0] | 0/16 | 2/64 |
| finger-turn-hard | 3.1e-08 | 0.0e+00 | 1.73 | 0.91–1.15 | 1e-07 / 8e-08 / 3e-07 | 4e-09 / 3e-09 | [0, 0, 0] vs [0, 0, 0] | 1/16 | 4/64 |
| fish-swim | 2.3e-07 | 3.4e-08 | 1.83 | 0.91–1.05 | 4e-07 / 4e-06 / 2e-06 | 1e-06 / 7e-07 | [14, 91, 38] vs [14, 91, 38] | 0/8 | 0/16 |
| hopper-hop | 2.3e-07 | 0.0e+00 | 1.01 | 0.96–∞ (1 rare event) | 7e-07 / 6e-05 / 1e+00 | 2e-05 / 1e+00 | [0, 0, 0] vs [0, 0, 0] | 4/32 | 17/128 |
| humanoid-run | 6.9e-07 | 4.7e-08 | 1.81 | 0.84–1.20 | 1e-06 / 8e-06 / 9e-01 | 7e-06 / 1e+00 | [1, 1, 1] vs [1, 1, 1] | 8/64 | 30/256 |
| humanoid-stand | 6.9e-07 | 2.8e-07 | 1.81 | 0.84–1.20 | 1e-06 / 8e-06 / 9e-01 | 7e-06 / 1e+00 | [4, 8, 8] vs [4, 8, 8] | 7/64 | 30/256 |
| humanoid-walk | 6.9e-07 | 8.5e-08 | 1.81 | 0.84–1.20 | 1e-06 / 8e-06 / 9e-01 | 7e-06 / 1e+00 | [1, 2, 2] vs [1, 2, 2] | 8/64 | 30/256 |
| quadruped-run | 9.5e-07 | 1.0e-07 | 0.98 | 1.03–1.03 | 7e-06 / 1e-03 / 1e+00 | 5e-04 / 8e-01 | [5, 7, 5] vs [5, 6, 5] | 4/32 | 21/128 |
| quadruped-walk | 9.5e-07 | 1.4e-07 | 0.98 | 1.03–1.03 | 7e-06 / 1e-03 / 1e+00 | 5e-04 / 8e-01 | [5, 6, 5] vs [5, 6, 6] | 4/32 | 21/128 |
| reacher-easy | 5.0e-08 | 0.0e+00 | 0.49 | 0.90–1.12 | 1e-07 / 3e-07 / 4e-07 | 2e-07 / 2e-07 | [30, 0, 48] vs [30, 0, 48] | 0/8 | 1/16 |
| reacher-hard | 5.0e-08 | 0.0e+00 | 0.49 | 0.90–1.12 | 1e-07 / 3e-07 / 4e-07 | 2e-07 / 2e-07 | [0, 0, 2] vs [0, 0, 2] | 0/8 | 1/16 |
| walker-run | 2.9e-07 | 5.3e-08 | 2.40 | 0.94–1.08 | 4e-07 / 2e-05 / 1e+00 | 1e-05 / 1e-01 | [25, 23, 23] vs [27, 22, 23] | 5/32 | 21/128 |
| walker-walk | 2.9e-07 | 1.5e-07 | 2.40 | 0.94–1.08 | 4e-07 / 2e-05 / 1e+00 | 1e-05 / 1e-01 | [53, 44, 28] vs [56, 43, 28] | 5/32 | 21/128 |
**Reading the table.** Observations and rewards agree to float32 precision everywhere (worst: dog 7e-6,
from its contact-force sensors). Initial-state distributions agree (|z| <= 2.9; hopper's one "inf" std
ratio is a single reset out of 200 in which the randomised leg touches the floor so the touch sensor
is non-zero - 1/200 vs 0/200 at a ~0.5 % event rate). Smooth or lightly-constrained tasks (acrobot,
cartpole, cheetah, cup, finger, fish, reacher) track CPU MuJoCo to <= 1e-5 for 50 agent steps and give
*identical* random-policy returns. Contact-rich locomotion (walker, hopper, humanoid, quadruped, dog)
decorrelates after 10-50 agent steps - so does CPU MuJoCo against itself with a 1e-6 perturbation;
random-policy returns stay statistically the same. Capacities are never close to being exhausted
(<= 24 % of nconmax, <= 23 % of njmax).

## 3.2 Where do the remaining trajectory differences come from? - `bench/divergence_study.py`

Relative obs error vs CPU MuJoCo (float64, tolerance 1e-8), median over 3 episodes:

| task | variant | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|---|
| dog-run | gpu_default | 3.3e-06 | 6.5e-04 | 6.0e-02 | 2.0e-01 | 1.1e+00 |
| dog-run | gpu_tol1e-8 | 1.0e-06 | 6.5e-04 | 6.1e-02 | 1.4e-01 | 8.4e-01 |
| dog-run | cpu_f32round | 8.3e-08 | 1.9e-07 | 4.4e-06 | 1.3e-05 | 2.7e-03 |
| dog-run | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.5e-13 |
| walker-walk | gpu_default | 5.4e-07 | 1.1e-06 | 1.6e-05 | 4.6e-05 | 3.4e-01 |
| walker-walk | gpu_tol1e-8 | 5.4e-07 | 7.5e-07 | 1.6e-05 | 5.2e-05 | 3.4e-01 |
| walker-walk | cpu_f32round | 1.4e-08 | 3.0e-07 | 9.8e-06 | 1.2e-05 | 1.9e+00 |
| walker-walk | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 2.7e-06 |
| quadruped-run | gpu_default | 1.3e-05 | 1.2e-05 | 3.7e-05 | 9.3e-04 | 2.0e-01 |
| quadruped-run | gpu_tol1e-8 | 1.3e-05 | 1.1e-05 | 7.0e-05 | 8.8e-04 | 4.9e-02 |
| quadruped-run | cpu_f32round | 4.0e-06 | 5.1e-06 | 1.5e-05 | 2.1e-05 | 3.6e-02 |
| quadruped-run | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| humanoid-walk | gpu_default | 4.3e-07 | 3.8e-05 | 1.4e-05 | 7.6e-02 | 4.3e-01 |
| humanoid-walk | gpu_tol1e-8 | 4.3e-07 | 4.1e-05 | 1.5e-05 | 7.6e-02 | 5.8e-01 |
| humanoid-walk | cpu_f32round | 2.1e-08 | 7.1e-06 | 2.9e-06 | 2.5e-04 | 2.8e-01 |
| humanoid-walk | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| hopper-hop | gpu_default | 1.8e-07 | 1.0e-06 | 2.9e-06 | 3.0e-05 | 1.2e+00 |
| hopper-hop | gpu_tol1e-8 | 1.8e-07 | 1.0e-06 | 2.9e-06 | 2.9e-05 | 1.2e+00 |
| hopper-hop | cpu_f32round | 2.8e-08 | 2.8e-07 | 4.3e-06 | 1.7e-05 | 6.3e-02 |
| hopper-hop | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
* The solver tolerance is irrelevant (CPU at 1e-6 == CPU at 1e-8; GPU at 1e-8 ~ GPU default).
* For walker, humanoid and hopper the GPU diverges no faster than a CPU run whose state is merely
  rounded to float32 after every control step -> float32 precision + chaos.
* Dog (and to a lesser extent quadruped) diverges faster. On identical states, 99.1 % of dog's contacting
  geom pairs produce identical contact sets on GPU and CPU (distance error <= 7e-5); the remaining 0.9 %
  are **box-box** pairs for which mujoco_warp's box-box routine generates a different number of contact
  points than C MuJoCo (quadruped and walker: 100 % identical). This is a mujoco_warp implementation
  difference, not a porting error.
* mujoco_warp is **not bitwise deterministic** run-to-run (contact/constraint rows are assembled with
  atomics): two identical GPU humanoid runs first differ at step 30, dog at step 0.

## 3.3 noslip solver (dog) - `bench/test_noslip.py`

184 contact states from a random-action dog rollout, qacc after `forward`:

| quantity | median | max |
|---|---|---|
| effect of noslip on qacc, CPU MuJoCo (rel.) | 7e-15 | 3.011e-2 |
| effect of noslip on qacc, GPU (my port) (rel.) | 5e-6 | 3.011e-2 |
| GPU(noslip) vs CPU(noslip) qacc rel. error | 6e-6 | 0.21 (same as without noslip: contact solver) |

The GPU noslip reproduces the CPU correction (identical maximum effect on the same states). Cost: 1.5 ms
per physics step with a naive single-thread Cholesky -> **0.136 ms** with the tile-Cholesky version.

## 3.4 Batched environment (S worlds) vs S independent environments - `bench/test_batched_env.py`

Identical initial states and actions: cartpole, reacher, finger-turn, walker, cup, cheetah are **bitwise
identical** per world over 60 agent steps; fish differs by <= 2e-8 (float rounding of the target refresh).
(Humanoid and dog are excluded here because single GPU runs are not deterministic themselves, 3.2.)

## 3.5 Agent equivalence

| test | result |
|---|---|
| CUDA-graph agent vs original, update (fixed batch, deterministic policy), 12 updates | max param diff 2e-6 - 7e-6 (walker, cartpole, humanoid, dog), metrics <= 8e-5 rel. |
| CUDA-graph agent vs original, plan() action distribution (1500 plans, train/eval, t0 / chained) | max \|z\| 1.8-2.9, std ratio 0.95-1.07 |
| vectorised S=3 agent vs 3 original agents, update | per-seed max param diff 3e-6, 1e-4, 5e-6 (Adam amplifies ulp differences of near-zero gradients), metrics <= 2e-5 rel. |
| vectorised agent vs original, plan() per seed | max \|z\| <= 2.3, std ratio 0.94-1.04 |
| capturable masked Adam (used for fp16) vs torch.optim.Adam (fp32) | max param diff 2e-6 after 12 updates |
| activation checkpointing (`ckpt_unroll`) vs none | **bitwise identical** parameters after 12 updates |
| PER sampling of the CUDA-graph / vectorised replay buffers vs the original `ReplayBuffer` (`bench/test_per_sampling.py`; same episodes and priorities, 3000 batches; buffer not full with 2 and 20 episodes, and full) | distinct indices per batch 388.2 / 388.2 vs 388.0 (2 episodes), 497.3 / 497.1 vs 497.4 (20), 512 vs 512 (full); index-frequency TV distance to the exact PER probabilities equal to the original's sampling noise; importance weights equal to 2e-7; after the replacement-flag fix of section 6.4 (before it: 512 vs 388 distinct indices - sampling without replacement) |
