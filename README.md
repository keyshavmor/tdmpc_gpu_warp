# tdmpc_gpu_warp: TD-MPC on GPU MuJoCo (MuJoCo Warp)

**Author of the GPU port:** Keyshav Mor

**Based on:** *Temporal Difference Learning for Model Predictive Control* (TD-MPC), Nicklas Hansen, Xiaolong Wang and
Hao Su, ICML 2022 ([paper](https://arxiv.org/abs/2203.04955), [project page](https://nicklashansen.github.io/td-mpc)).
Original source code (the baseline of this port): **https://github.com/nicklashansen/tdmpc** (MIT License, Copyright (c)
2021 Nicklas Hansen). All credit for the TD-MPC method, the TOLD model, the planner, the hyperparameters and the
original implementation goes to the original authors; this repository only moves the method onto the GPU.

---

The original TD-MPC simulates the DeepMind Control Suite with CPU MuJoCo (dm_control, MuJoCo 2.1) and trains the
agent on the GPU. Each run therefore needs CPU threads for physics and for Python kernel launches. When several runs
share one machine they contend for CPU threads, the GPU is starved and its utilisation stays low.

**tdmpc_gpu_warp runs the complete training loop on the GPU**: physics (MuJoCo Warp), observations, rewards, episode
resets, replay buffer, planning and learning. It supports all 21 state-based TD-MPC tasks and keeps the original
algorithm, network architecture and hyperparameters unchanged. A GPU run needs 0.02-0.17 CPU cores instead of a full
core, so many runs can share one GPU without CPU contention.

| | |
|---|---|
| Tasks | all 21 state-based DMControl tasks of TD-MPC (`tasks.txt`), incl. humanoid and dog |
| Speed-up per run vs. the original | **2.4-4.8x** (mean 3.7x) at H = 5, 3.1-3.8x at H = 30; dog 2.0x / 1.7x |
| 21 tasks x 5 seeds (H = 5, default budgets) | **~17 days** (5 seeds in parallel, MPS) vs. ~77 days (original, back-to-back) |
| Learning | statistically indistinguishable from the paper's reference curves (cartpole-swingup, walker-walk, 5 seeds) |
| CPU use | 16 concurrent GPU runs use 17 % of a 16-core CPU (16 original runs: 99 %) |
| Hardware used for all measurements | NVIDIA RTX 3090 (24 GiB), 16-core CPU |

> **Scope: humanoid and dog tasks.** humanoid-run/-stand/-walk and dog-run/-trot/-walk are ported to the GPU,
> and their environments, speed and memory were measured in short benchmarks (those numbers appear in the tables).
> They were **not** trained, and they are excluded from the learning-curve tests and the remaining baseline work
> (MPS stability tests, re-timings, further optimisation). Their training budgets (3M / 5M environment steps, about
> 14-33 h per seed even on the optimised GPU port, 1.4-3.9 days per 5-seed batch in parallel) make them too expensive to pursue for
> this project. The optimised baseline is therefore established and verified on the other 15 tasks.

## Contents
1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Total run time per task: 1 seed vs. 5 seeds](#total-run-time-per-task-1-seed-vs-5-seeds)
4. [Results summary](#results-summary)
5. [Is it the same algorithm?](#is-it-the-same-algorithm)
6. [Repository layout](#repository-layout)
7. [Documentation](#documentation)
8. [Known issue](#known-issue)
9. [Acknowledgements, citation and licenses](#acknowledgements-citation-and-licenses)

## Installation

Directory layout. MuJoCo Warp is used from a checkout next to this repository, at a tested upstream commit plus
a small patch, `patches/mujoco_warp/`. The patch adds the GPU noslip solver that the dog tasks need; upstream
MuJoCo Warp does not implement noslip.

```
<workspace>/
├── tdmpc_gpu_warp/   this repository (GPU port)
├── mujoco_warp/      MuJoCo Warp + GPU noslip solver, created by scripts/setup_mujoco_warp.sh
└── tdmpc/            optional: original TD-MPC (https://github.com/nicklashansen/tdmpc), only needed to
                      reproduce the CPU-MuJoCo comparisons in bench/
```

**1. Machine.** Linux with an NVIDIA GPU and **driver >= 580**. The CUDA toolkit is not needed. For several runs in
parallel the CUDA MPS utilities are needed, which on Ubuntu come with the driver package. Step-by-step set-up, the
per-optimisation requirements and the optional profiling tools: [user guide, section 1.1](docs/USER_GUIDE.md#11-machine-set-up-linux--nvidia).

```bash
sudo apt install nvidia-driver-580-open && sudo reboot      # Ubuntu; brings nvidia-cuda-mps-control along
```

**2. Python environment.** Choose one option:

```bash
git clone https://github.com/keyshavmor/tdmpc_gpu_warp.git && cd tdmpc_gpu_warp

# option A: conda (exact pinned package set)
conda env create -f environment.yaml && conda activate tdmpc-warp
./scripts/setup_mujoco_warp.sh              # clones MuJoCo Warp to ../mujoco_warp, applies the patch, pip install -e

# option B: uv (pyproject.toml + uv.lock; runs scripts/setup_mujoco_warp.sh itself)
./setup_uv.sh && source .venv/bin/activate
```

**3. Check.**

```bash
python check_system.py                      # driver, GPU, MPS, libraries -> OK / WARN / FAIL
```

**Notes**
- **Versions:** Python 3.11, torch 2.11 (CUDA 12.8 wheels), warp-lang 1.18, mujoco 3.15, dm_control 1.0.48 (tested:
  RTX 3090, driver 580).
- **No rendering:** no display or rendering is needed, and video recording is disabled in the GPU port.
- **CPU baseline:** `../tdmpc` uses its own environment (`tdmpc`, Python 3.8, MuJoCo 2.1), as described in the
  original repository.

## Quick start

```bash
conda activate tdmpc-warp
cd tdmpc_gpu_warp

# one seed, optimised GPU port (same maths as the original; recommended)
python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true

# one seed, faithful GPU port (the original agent code unchanged, only the environment on the GPU)
python src/train.py task=walker-walk seed=1

# several seeds in parallel: start CUDA MPS, then one process per seed
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_$USER/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY && nvidia-cuda-mps-control -d
for s in 1 2 3 4 5; do python src/train.py task=walker-walk seed=$s graph_agent=true pace_host=true & sleep 20; done; wait
echo quit | nvidia-cuda-mps-control
```

Results are written to `logs/<task>/state/<exp_name>/<seed>/eval.log` (same format as the original `eval.log`) and
`train.log`. All original options work unchanged (`horizon=`, `exp_name=`, `eval_freq=`, ...).
**See the [user guide](docs/USER_GUIDE.md)** for all options, multi-seed training, MPS, logs and troubleshooting.

## Total run time per task: 1 seed vs. 5 seeds

The table covers state observations, H = 5 and each task's default TD-MPC training budget on one RTX 3090.

**What the columns mean**
- **CPU MuJoCo + GPU learning** is the original TD-MPC code.
- **GPU faithful** is the GPU environment with the original agent.
- **GPU optimised** is the GPU environment with the CUDA-graph agent and host pacing. It computes the same maths.
- **Parallel** columns run the 5 seeds concurrently: as separate processes (CUDA MPS for the GPU port), or as one
  vectorised process.

**How the numbers were obtained**
- Single-run times are extrapolated from the measured steady-state cost per step. They include the seeding phase and
  10 evaluation episodes every 20k environment steps.
- Parallel times use the total-throughput gain measured with concurrent runs.
- Full derivation: [docs/results_runtime.md](docs/results_runtime.md).

| task | env steps | 1 seed: CPU MuJoCo + GPU learning | 1 seed: GPU faithful | 1 seed: GPU optimised | 5 seeds back-to-back: CPU + GPU learning | 5 seeds back-to-back: GPU optimised | 5 seeds parallel: CPU + GPU learning | 5 seeds parallel: GPU optimised + MPS | 5 seeds parallel: GPU vectorised (1 process) |
|---|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | 0.5M | 2.8 h | 2.2 h | 0.7 h | 13.9 h | 3.7 h | 4.1 h | 1.6 h | 1.9 h |
| cartpole-swingup | 0.5M | 1.4 h | 1.1 h | 0.5 h | 7.1 h | 2.3 h | 2.1 h | 1.0 h | 1.2 h |
| cartpole-swingup-sparse | 0.5M | 1.4 h | 1.1 h | 0.5 h | 7.0 h | 2.3 h | 2.1 h | 1.0 h | 1.2 h |
| cheetah-run | 0.5M | 2.7 h | 2.0 h | 0.7 h | 13.3 h | 3.4 h | 3.9 h | 1.5 h | 1.7 h |
| cup-catch | 0.5M | 2.8 h | 2.2 h | 1.0 h | 13.9 h | 5.2 h | 4.1 h | 2.3 h | 2.7 h |
| dog-run | 5M | 65 h (2.7 d) | 49 h (2.0 d) | 33.4 h | 324 h (13.5 d) | 167 h (6.9 d) | 196 h (8.2 d) | 93 h (3.9 d) | 118 h (4.9 d) |
| dog-trot | 5M | 63 h (2.6 d) | 47.0 h | 33.2 h | 315 h (13.1 d) | 166 h (6.9 d) | 191 h (8.0 d) | 93 h (3.9 d) | 118 h (4.9 d) |
| dog-walk | 5M | 66 h (2.7 d) | 50 h (2.1 d) | 33.5 h | 330 h (13.7 d) | 168 h (7.0 d) | 200 h (8.3 d) | 94 h (3.9 d) | 119 h (5.0 d) |
| finger-spin | 0.5M | 5.3 h | 4.4 h | 1.2 h | 26.5 h | 6.2 h | 7.9 h | 2.7 h | 3.2 h |
| finger-turn-hard | 0.5M | 5.7 h | 4.4 h | 1.3 h | 28.3 h | 6.4 h | 8.4 h | 2.8 h | 3.3 h |
| fish-swim | 0.5M | 2.2 h | 2.2 h | 1.1 h | 11.2 h | 5.3 h | 3.3 h | 2.3 h | 2.7 h |
| hopper-hop | 0.5M | 2.5 h | 2.1 h | 0.8 h | 12.6 h | 4.1 h | 3.7 h | 1.8 h | 2.1 h |
| humanoid-run | 3M | 43.7 h | 33.7 h | 13.7 h | 218 h (9.1 d) | 69 h (2.9 d) | 74 h (3.1 d) | 34.5 h | 40.2 h |
| humanoid-stand | 3M | 43.1 h | 32.6 h | 13.6 h | 215 h (9.0 d) | 68 h (2.8 d) | 73 h (3.0 d) | 34.1 h | 39.7 h |
| humanoid-walk | 3M | 42.4 h | 34.0 h | 13.6 h | 212 h (8.8 d) | 68 h (2.8 d) | 72 h (3.0 d) | 34.2 h | 39.8 h |
| quadruped-run | 0.5M | 2.7 h | 2.3 h | 0.9 h | 13.5 h | 4.6 h | 4.0 h | 2.0 h | 2.3 h |
| quadruped-walk | 0.5M | 2.8 h | 2.2 h | 0.9 h | 14.2 h | 4.6 h | 4.2 h | 2.0 h | 2.3 h |
| reacher-easy | 0.5M | 2.6 h | 2.1 h | 0.6 h | 12.8 h | 3.0 h | 3.8 h | 1.3 h | 1.6 h |
| reacher-hard | 0.5M | 2.7 h | 2.1 h | 0.6 h | 13.5 h | 3.1 h | 4.0 h | 1.4 h | 1.6 h |
| walker-run | 0.5M | 5.4 h | 4.4 h | 1.7 h | 27.2 h | 8.7 h | 8.1 h | 3.8 h | 4.4 h |
| walker-walk | 0.5M | 5.6 h | 4.4 h | 1.7 h | 28.1 h | 8.7 h | 8.3 h | 3.8 h | 4.5 h |
| **all 21 tasks** | | **372 h (15.5 d)** | **285 h (11.9 d)** | **155 h (6.5 d)** | **1858 h (77.4 d)** | **776 h (32.4 d)** | **878 h (36.6 d)** | **414 h (17.3 d)** | **511 h (21.3 d)** |

**How to read the table**
- **1 seed:** the optimised GPU port is 2.4x faster than the original over all 21 tasks (15.5 d -> 6.5 d).
- **5 seeds:** running them in parallel under MPS finishes 4.5x earlier than the original back-to-back (77 d -> 17 d).
  It is also ~2x earlier than the GPU port run back-to-back.
- **Why parallel wins:** a single run cannot saturate the GPU, so concurrent runs raise the total throughput even
  though each individual run slows down.
- **All-on-CPU** (physics and learning on the CPU) was not measured.

For the 15 tasks without humanoid and dog (the scope of the baseline):

| 15 tasks | 1 seed per task | 5 seeds per task |
|---|---|---|
| CPU MuJoCo + GPU learning, back-to-back | 48.6 h | 243 h (10.1 d) |
| CPU MuJoCo + GPU learning, 5 seeds in parallel | - | 72 h (3.0 d) |
| GPU optimised, back-to-back | 14.2 h | 71.6 h (3.0 d) |
| GPU optimised, 5 seeds in parallel + MPS | - | 31.3 h (1.3 d) |

## Results summary

All numbers were measured on this machine. Raw data is in `results/`, and the analysis is in `docs/`.

**1. GPU port: complete and validated** ([docs/03](docs/03_validation.md))
- All 21 tasks run on MuJoCo Warp.
- Observation and reward functions match dm_control to <= 8e-6 on identical states, and reset distributions match.
- Physics tracks CPU MuJoCo to <= 1e-5 on smooth tasks. Contact-rich tasks decorrelate after 10-50 agent steps, as a
  float32-rounded CPU run does (chaos); random-policy returns still agree.
- dog needs MuJoCo's noslip solver, which MuJoCo Warp lacked. It was implemented on the GPU
  ([`patches/mujoco_warp/`](patches/mujoco_warp/)) and reproduces C MuJoCo's correction.

**2. Speed vs. the CPU-MuJoCo original, single run, H = 2..30** ([docs/02](docs/02_single_run_performance.md))
- **Optimised port:** 2.4-4.8x at H = 5 and 3.1-3.8x at H = 30 (dog 2.0x / 1.7x).
- **Faithful port:** 1.2-1.4x.
- **Cause:** the original agent is bound by Python kernel launches, which leaves the GPU 73-86 % idle. CUDA graphs
  remove that overhead.

**3. Peak VRAM and OOM horizon** ([docs/04](docs/04_multi_run_capacity.md), [results_memory](docs/results_memory.md))
- **VRAM per run:** about 0.45-0.6 GiB + 16.5 MiB per unit of H (dog about 1.5 GiB + 71 MiB per unit of H).
- **OOM:** only dog can go OOM at horizons the algorithm can run at all, from H ~ 292-309 depending on the variant.
  Activation checkpointing (bit-identical results) removes that limit.

**4. Runs per GPU** ([docs/04](docs/04_multi_run_capacity.md))
- **In memory**, separate processes at H = 5: cartpole 35, walker 33, humanoid 22, dog 11.
- **In one vectorised process:** walker 136 seeds at H = 5.
- **Without slowing each other down: one run** for every task and horizon. A single optimised run already keeps the GPU
  86-97 % busy. A second run slows each run to 58-80 % of its solo speed. The total throughput still rises up to
  ~2.7x at 8 runs under MPS.

**5. CPU thread contention is eliminated** ([docs/04](docs/04_multi_run_capacity.md))
- A GPU run needs 0.02-0.17 CPU cores.
- Without CUDA MPS, GPU processes are time-sliced and running them in parallel gains nothing.

**6. Optimisations, all mathematically equivalent and validated** ([docs/06](docs/06_optimisations.md))
- **GPU environment:** each environment step runs as one CUDA graph; GPU noslip with a tile Cholesky (11x faster).
- **CUDA-graph agent:** bitwise identical to the original agent over 200 updates.
- **Host pacing:** cuts CPU use per run.
- **Optional, exact:**
  - vectorised seeds;
  - activation checkpointing (bit-identical);
  - batched update (equal up to rounding, 1.03-1.27x);
  - sensor skipping in physics sub-steps (bit-identical).
- **Rejected after evaluation:** Triton GEMM fusion, SM partitioning, more processes without MPS.

**7. bf16 / fp16 / TF32** ([docs/07](docs/07_precision.md))
- **Speed:** 1.06-1.24x for a single run; 1.5-1.7x (bf16/fp16) with many seeds in one process.
- **VRAM:** only 3-10 % less.
- **Learning:** like fp32 on walker-walk.
- **Status:** not mathematically equivalent, so these are opt-in only.

## Is it the same algorithm?

Yes in the mathematics, and statistically yes in behaviour.

**What is identical**
- TOLD model, CEM/MPPI planning, prioritized replay, losses, optimiser and hyperparameters, all in float32.
- The faithful port runs the original agent code; only 3 lines differ (`torch.tensor` -> `torch.as_tensor`).
- The optimised agent is bitwise identical to it in deterministic tests.
- Replay sampling follows the original rule: with replacement until the buffer is full. The optimised and
  vectorised agents had this flag inverted. That was found and fixed while documenting the code; the sampling is
  now verified against the original buffer ([docs/06 section 6.4](docs/06_optimisations.md)).

**What is not bitwise identical**
- **Physics:** GPU float32 (MuJoCo Warp) vs. CPU float64 (MuJoCo 2.1). On contact-rich tasks, chaos amplifies the
  rounding differences, as it does between two CPU runs.
- **Random streams:** the GPU random generator replaces numpy's. The distributions are the same, but a given seed does
  not reproduce the exact same run.

**Learning curves**
- cartpole-swingup and walker-walk, 5 seeds, match the paper's reference (p >= 0.14 at every evaluation but one
  marginal point).
- The other 19 tasks are validated at the environment level.

Details: [docs/01](docs/01_design_and_faithfulness.md), [docs/03](docs/03_validation.md),
[docs/results_learning](docs/results_learning.md).

## Repository layout

| path | content |
|---|---|
| `src/env_warp.py` | GPU DMControl environment: batched MuJoCo Warp worlds, dm_control-exact stepping, GPU resets, CUDA-graph step |
| `src/dmc_tasks.py` | batched torch ports of the observation / reward / episode-initialisation code of the 12 DMC domains |
| `src/algorithm/tdmpc.py`, `helper.py` | original TD-MPC agent and helpers (3-line change) |
| `src/algorithm/tdmpc_graph.py` | optimised single-seed agent (CUDA-graph captured planning and update) |
| `src/algorithm/tdmpc_vec.py` | vectorised agent: many independent seeds in one process; checkpointing / precision options |
| `src/train.py` | training script, one seed |
| `src/train_vec.py` | training script, many seeds of one configuration in one process |
| `src/env.py`, `logger.py`, `cfg.py` | original dm_control environment wrapper (used by benchmarks for CPU comparisons), logger, config parser (unchanged) |
| `cfgs/` | original TD-MPC configuration files (unchanged values) |
| `bench/` | validation, equivalence tests, benchmarks, multi-run studies, analysis scripts |
| `docs/` | report chapters, result tables, figures, user guide |
| `results/` | raw measurement data |
| `environment.yaml`, `pyproject.toml` + `uv.lock`, `setup_uv.sh` | conda / uv environment definitions |
| `check_system.py` | checks driver, GPU, MPS and Python packages |
| `patches/mujoco_warp/`, `scripts/setup_mujoco_warp.sh` | GPU noslip solver for MuJoCo Warp (new `noslip.py` + small hooks in `io.py`, `solver.py`, `types.py`) and the script that applies it to the tested upstream commit |

## Documentation

| document | content |
|---|---|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | **how to use the repository**: installation, training, options, multi-seed, MPS, logs, troubleshooting |
| [docs/01_design_and_faithfulness.md](docs/01_design_and_faithfulness.md) | architecture, what is identical, what differs, porting gotchas |
| [docs/02_single_run_performance.md](docs/02_single_run_performance.md) | H = 2..30 speed / VRAM sweep, CPU baseline vs. GPU, full-run times |
| [docs/03_validation.md](docs/03_validation.md) | environment validation (21 tasks), divergence study, noslip, agent equivalence |
| [docs/04_multi_run_capacity.md](docs/04_multi_run_capacity.md) | runs per GPU (memory vs. no slowdown), throughput, CPU use, seed organisation, run times |
| [docs/05_how_to_run.md](docs/05_how_to_run.md) | reproducing every measurement |
| [docs/06_optimisations.md](docs/06_optimisations.md) | optimisations, gains, equivalence evidence, rejected ideas |
| [docs/07_precision.md](docs/07_precision.md) | bf16 / fp16 / TF32: speed, VRAM, accuracy |
| [docs/08_known_issues.md](docs/08_known_issues.md) | open fault, limitations |
| [docs/results_runtime.md](docs/results_runtime.md) | total run time per task, 1 seed vs. 5 seeds, per hardware configuration |
| `docs/results_*.md`, `docs/figures/` | full result tables and comparison figures |

## Known issue

- **Symptom:** a rare `illegal memory access` (driver: Xid 31 MMU fault), root cause not yet found.
- **Where it occurs:**
  - mostly under **CUDA MPS**: 7 times in vectorised `train_vec.py` processes, once in a single-seed `train.py`
    process;
  - without MPS: once in ~6 vectorised process-hours, never in single-seed processes.
- **How it ends:** under MPS one fault ends **all** co-running runs. No effect on completed results was observed, and
  no resume is implemented.
- **Trade-off:**
  - 5-8 single-seed runs under MPS finish ~2x earlier but may occasionally all crash together;
  - single-seed runs without MPS have shown no fault, but get no parallel speed-up.
- Details: [docs/08](docs/08_known_issues.md).

## Acknowledgements, citation and licenses

This port builds on:
- **TD-MPC**, Nicklas Hansen, Xiaolong Wang, Hao Su: https://github.com/nicklashansen/tdmpc (MIT License).
  - Kept from it: the agent, helpers, configuration files, logger and training loop structure.
  - Licence: the original copyright notice is kept in [LICENSE_TDMPC](LICENSE_TDMPC). This repository is MIT-licensed
    as well ([LICENSE](LICENSE), Copyright (c) 2026 Keyshav Mor).
- **MuJoCo Warp**, https://github.com/google-deepmind/mujoco_warp (Apache License 2.0).
  - The GPU noslip solver patch in `patches/mujoco_warp/` is distributed under the same license.
- **MuJoCo** and the **DeepMind Control Suite**, https://github.com/google-deepmind/mujoco and
  https://github.com/google-deepmind/dm_control (Apache License 2.0).
  - The task observation, reward and initialisation code in `src/dmc_tasks.py` is ported from dm_control.

If you use this code, please cite the original TD-MPC paper:

```bibtex
@inproceedings{Hansen2022tdmpc,
  title     = {Temporal Difference Learning for Model Predictive Control},
  author    = {Nicklas Hansen and Xiaolong Wang and Hao Su},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2022}
}
```

GPU port: Keyshav Mor, *tdmpc_gpu_warp: TD-MPC on GPU MuJoCo (MuJoCo Warp)*, 2026.
