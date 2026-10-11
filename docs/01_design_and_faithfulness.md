# 1. Design of the GPU port and what is (and is not) identical to the original

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

## 1.1 Goal

The original TD-MPC (https://github.com/nicklashansen/tdmpc) simulates DMControl with CPU MuJoCo (dm_control, MuJoCo 2.1)
while the TOLD model, planner and replay buffer live on the GPU. Every agent step therefore bounces
observations/actions between host and device, and every run needs a CPU core for physics **and**
for Python kernel launches. With several seeds on one machine the runs contend for CPU threads, the GPU
is starved of work and GPU utilisation stays low. The port moves the *entire* run onto the GPU:
physics (mujoco_warp), observations, rewards, episode resets, replay buffer, planning and learning.

## 1.2 Components (all under `tdmpc_gpu_warp/`)

| file | role |
|---|---|
| `src/env_warp.py` | GPU DMControl environment: S worlds (S = 1 for a normal run, S = #seeds for the vectorised trainer), dm_control-exact stepping semantics, GPU episode resets, whole agent step captured in one CUDA graph |
| `src/dmc_tasks.py` | torch ports of the observation / reward / `initialize_episode` code of all 12 DMC domains used by TD-MPC (21 tasks) |
| `src/algorithm/tdmpc.py`, `helper.py` | the original agent, unchanged except `torch.tensor(obs)` -> `torch.as_tensor(obs)` (3 lines) |
| `src/algorithm/tdmpc_graph.py` | optimised single-seed agent: same maths, CUDA-graph captured `plan()` / `update()`, no host syncs |
| `src/algorithm/tdmpc_vec.py` | vectorised agent: S independent seeds in one process (stacked weights, per-seed Adam / clipping / PER / planning), optional precision modes and activation checkpointing |
| `src/train.py` | training script for one seed (faithful by default; `graph_agent=true pace_host=true` = optimised) |
| `src/train_vec.py` | training script for S seeds of one configuration in one process |
| `../mujoco_warp/mujoco_warp/_src/noslip.py` | **new**: GPU port of MuJoCo's noslip solver (needed by dog; mujoco_warp did not implement it) |
| `bench/*` | validation, equivalence tests, benchmarks, sweeps, packing / contention studies, analysis |

Software: conda env `tdmpc-warp` (Python 3.11, torch 2.11 + CUDA 12.8, warp-lang 1.18, mujoco 3.15,
mujoco_warp 3.15 from the local checkout, dm_control 1.0.48). The original baseline runs unchanged in
the existing `tdmpc` env (Python 3.8, torch 2.4.1, dm_control 0.0.403778684, MuJoCo 2.1).
The DMC task definitions (`suite/*.py`, `*.xml`) are identical between the two dm_control versions
(only numpy API renames; checked with `diff`).

## 1.3 What is reproduced exactly

* **Model**: the MjModel compiled by dm_control (`suite.load(domain, task, task_kwargs={'random': seed})`)
  is put on the GPU with `mujoco_warp.put_model`. Same XML, same options (timestep, integrator, solver,
  cone, noslip iterations, fluid density/viscosity), same actuators/sensors.
* **Stepping semantics** (`env_warp._control_step`): dm_control's legacy `Physics.step(n_sub_steps)` is
  `mj_step2, (n-1) x mj_step, mj_step1` (Euler) or `n x mj_step, mj_step1` (RK4). The port issues
  exactly these calls. C MuJoCo's `mj_step1` keeps the acceleration-stage sensors (touch, force, torque,
  accelerometer) of the last sub-step while mujoco_warp's `step1` zeroes them; the port saves/restores
  them so observations contain the same (one-sub-step-old) values as dm_control.
* **Wrappers**: action float32 -> action repeat (rewards summed) -> action scaling [-1,1] -> actuator
  range -> 1000 // action_repeat agent steps per episode. Same `ctrlrange` clamping (inside MuJoCo).
* **Observations and rewards**: `dmc_tasks.py` mirrors every `get_observation` / `get_reward` /
  `rewards.tolerance` (all sigmoid types) line by line, same OrderedDict concatenation order.
* **Episode initialisation**: every `initialize_episode` is ported (`init_episode`): random joint
  configurations (`randomize_limited_and_rotational_joints` incl. ball/free joints), target placements
  (reacher, fish, finger-turn), dog's random azimuth / velocities / activations, cheetah's 200 settling
  steps (the first of which is a no-op in dm_control because `mj_resetData` zeroes the derived
  quantities - reproduced), and the after-reset `forward()` with **actuation disabled** (dm_control's
  `after_reset`). Rejection sampling ("resample until no contact": humanoid, finger, ball-in-cup) is
  evaluated on the GPU as one collision pass over 32 candidate worlds per seed, taking the first
  non-penetrating candidate - the same output distribution as the sequential loop.
  Quadruped's `_find_non_contacting_height` (z = 0, 0.01, 0.02, ... until no contact) is evaluated as
  160 consecutive heights in parallel; the first contact-free height is exactly the sequential result.
* **Agent**: unchanged maths in all three agents. The CUDA-graph and vectorised agents were verified
  against the original agent: identical parameters after 12 updates up to float32 rounding (max |dw|
  7e-6 / 3e-6..1e-4 incl. Adam amplification) and identical action distributions from `plan()`.

## 1.4 What differs (and why it does not change the science)

| difference | effect | evidence |
|---|---|---|
| float32 GPU physics (mujoco_warp) instead of float64 CPU MuJoCo 2.1 | per-step agreement ~1e-6 relative; chaotic contact tasks decorrelate after 10-50 agent steps exactly like a float32-rounded CPU run does | `results/validation/*.json`, divergence study |
| mujoco_warp clamps the solver tolerance to >= 1e-6 for float32 | none measurable (CPU at 1e-6 == CPU at 1e-8; GPU at 1e-8 == GPU default) | divergence study |
| mujoco_warp's box-box collision produces a different number of contact points than C MuJoCo for 0.9 % of dog's contacting geom pairs | slightly faster trajectory decorrelation for dog | contact-set comparison |
| GPU physics is not bitwise deterministic run-to-run (atomics in contact/constraint assembly) | two identical GPU runs drift apart like two chaotic CPU runs with 1e-6 perturbations | determinism check |
| episode-initialisation RNG: CUDA generator instead of numpy `RandomState` (the distributions are identical; `reset_mode=cpu` reproduces the original stream exactly) | same initial-state distribution | reset-distribution test (`B_*` columns) |
| optimised agents: elite selection / PER sampling with `torch.multinomial` (GPU RNG) instead of `np.random.choice` | same distributions (incl. the original's replacement rule: with replacement until the buffer is full), different random stream | plan-distribution, update-equivalence and PER-sampling tests (`bench/test_per_sampling.py`; replacement flag fixed, docs/06 section 6.4) |
| noslip (dog only): new GPU implementation of `mj_solNoSlip` (pyramidal cones), residual updated incrementally instead of recomputed | same maths; GPU noslip changes qacc exactly like CPU noslip (max effect 3.011e-2 on both) | `bench/test_noslip*.py` output in docs |

## 1.5 Gotchas found while porting (useful if you extend this)

1. mujoco_warp computes the pose of world-attached ("static") geoms only in `make_data`; tasks that move
   a world geom per episode (reacher, fish targets) must refresh `geom_xpos` (done in `_refresh_static_geoms`).
2. `mjw.step1` zeroes `sensordata` before recomputing position/velocity sensors (C `mj_step1` does not).
3. numpy integers compared with MuJoCo's pybind enums via `in (...)` silently evaluate to False
   (this initially broke the joint randomiser) - always cast to `int`.
4. Tensors referenced by a captured CUDA graph must stay alive as long as the graph (a module-level
   index tensor overwritten by a second env caused a device-side assert).
5. CUDA graphs cannot be captured on the legacy default stream: torch and warp share one explicit stream.
6. The CUDA MPS pipe directory must be a short path (unix-socket length limit).
7. **External CUDA-graph capture with warp (found late, fixed):** when warp joins a capture started by torch
   (`wp.capture_begin(external=True)`), `wp.capture_end()` returns a `warp.Graph` wrapping torch's in-progress
   `cudaGraph_t`. Discarding that object makes `Graph.__del__` call `cudaGraphDestroy` on a graph warp does not
   own (destroy during capture / double destroy after torch's own destroy) - undefined behaviour that was
   harmless in every single-process test (all results bit-identical before/after the fix) but surfaced as rare
   *illegal memory access* faults when ~10 processes shared the GPU under MPS. Fix in `env_warp.py`: keep the
   returned object alive and set its `graph` handle to `None`.
8. **float32 cancellation in MuJoCo Warp's capsule collision (found late, fixed by patch 0002):** the segment-segment
   solve computes det = ma * mc - mb * mb, which loses all precision in float32 when two capsules are nearly parallel.
   Random-policy validation never reached such states. hopper's trained agents do: the thigh folds back parallel to
   the torso. It took a check on the states of a trained agent to find it (docs/03 sections 3.1.1 and 3.6). Lesson:
   validate on states the learned policy actually visits, not only on random-policy states.
