# User guide: tdmpc_gpu_warp (TD-MPC on GPU MuJoCo)

*GPU port by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (ICML 2022); original code:
https://github.com/nicklashansen/tdmpc (MIT License).*

This guide explains how to install the port, train agents, run several seeds or configurations on one GPU, read the
logs, and reproduce the measurements.

> **Scope: humanoid and dog tasks.** humanoid-run/-stand/-walk and dog-run/-trot/-walk are ported to the GPU,
> and their environments, speed and memory were measured in short benchmarks (those numbers appear in the tables).
> They were **not** trained, and they are excluded from the learning-curve tests and the remaining baseline work
> (MPS stability tests, re-timings, further optimisation). Their training budgets (3M / 5M environment steps, about
> 14-33 h per seed even on the optimised GPU port, 1.4-3.9 days per 5-seed batch in parallel) make them too expensive to pursue for
> this project. The optimised baseline is therefore established and verified on the other 15 tasks.

**Background:**
- [docs/01](01_design_and_faithfulness.md): what the port changes and what it keeps.
- [docs/04](04_multi_run_capacity.md): how many runs fit on a GPU.

---

## 1. Installation

### 1.1 Machine set-up (Linux + NVIDIA)

**What is needed:** only an NVIDIA GPU and its driver. The CUDA toolkit (`nvcc`) is **not** needed, for two reasons:
- **CUDA libraries:** torch and warp-lang ship them inside their wheels.
- **Kernels:** warp compiles its kernels at run time and caches them in `~/.cache/warp`, which takes about 0.3 GB.

**Which component each feature needs**

| feature / optimisation | needs | how to check |
|---|---|---|
| GPU physics, CUDA-graph environment and agent (all modes) | NVIDIA GPU + **driver >= 580** (warp-lang 1.18 is built with CUDA 13; the torch wheels use CUDA 12.8) | `nvidia-smi` (driver version) |
| graph-captured constraint solver (conditional CUDA graph nodes) | covered by driver >= 580 | - |
| host pacing + blocking sync (low CPU use per run) | `libcuda.so.1` from the driver (Ubuntu: `libnvidia-compute-580`) | `python check_system.py` |
| **several runs in parallel (CUDA MPS)** | `nvidia-cuda-mps-control` (Ubuntu: `nvidia-compute-utils-580`, installed with the driver metapackage); GPU compute mode `Default` (or `Exclusive_Process`) | `which nvidia-cuda-mps-control`, `nvidia-smi -q \| grep "Compute Mode"` |
| GPU noslip solver (dog), tile Cholesky | included in the warp-lang wheel and `../mujoco_warp` | `python check_system.py` |
| vectorised seeds, activation checkpointing, batched update, sensor skipping | nothing extra | - |
| bf16 / TF32 precision options | GPU with compute capability >= 8.0 (Ampere or newer, e.g. RTX 30xx/40xx, A100); fp16 runs on any GPU | `python check_system.py` |
| profiling (`nsys`), memory checking (`compute-sanitizer`) - optional | CUDA toolkit / Nsight Systems | `which nsys` |

**Install the driver (Ubuntu; the reference machine runs Ubuntu 26.04 with `nvidia-driver-580-open`)**

```bash
sudo apt update
ubuntu-drivers list                          # lists the available driver packages
sudo apt install nvidia-driver-580-open      # open kernel modules (Turing / RTX 20xx and newer);
                                             # use nvidia-driver-580 for older GPUs, *-server variants on servers
sudo reboot
nvidia-smi                                   # must show driver >= 580 and the GPU
which nvidia-cuda-mps-control                # MPS control binary (package nvidia-compute-utils-580)
```

**Driver notes**
- **Secure Boot:** with Secure Boot enabled, the driver installation asks for a MOK password, and the key must be
  enrolled at the next boot.
- **Driver dependencies:** the driver metapackage pulls in `nvidia-compute-utils-580` (MPS) and
  `libnvidia-compute-580` (libcuda).
- **Other distributions:** install the NVIDIA driver >= 580 with the distribution's packages or NVIDIA's runfile,
  including its CUDA MPS utilities.

**Optional: profiling and debugging tools** (not needed for training). Install them from NVIDIA's CUDA apt repository
for your Ubuntu release. Install only the toolkit, **not** the `cuda` / `cuda-drivers` metapackages, which would
replace the driver:

```bash
# example for Ubuntu 24.04 (replace ubuntu2404 by your release)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install cuda-toolkit-12-9           # includes compute-sanitizer and Nsight Systems (nsys)
```

**GPU settings (optional, need root)**

| setting | command | effect |
|---|---|---|
| compute mode | keep `Default` | All measurements used `Default`, which lets MPS and ordinary processes share the GPU without root. `sudo nvidia-smi -c EXCLUSIVE_PROCESS` is NVIDIA's recommendation for dedicated MPS servers, but it blocks every non-MPS CUDA process, including desktop applications. |
| persistence mode | `sudo nvidia-smi -pm 1` | keeps the driver initialised on headless machines (faster process start-up); unnecessary on a desktop with a display |
| MPS SM partitioning | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | does **not** isolate runs from each other ([docs/04](04_multi_run_capacity.md), section 4.6); not recommended |

**CPU side:** nothing special is needed. A GPU run uses 0.02-0.17 CPU cores with host pacing, and each process
limits torch/OpenMP to one thread by itself.

### 1.2 Directory layout

```
<workspace>/
├── tdmpc_gpu_warp/   this repository (GPU port)
├── mujoco_warp/      MuJoCo Warp + GPU noslip solver, created by scripts/setup_mujoco_warp.sh
└── tdmpc/            optional: original TD-MPC (https://github.com/nicklashansen/tdmpc), only needed to
                      reproduce the CPU-MuJoCo comparisons in bench/
```

**Setting up MuJoCo Warp:** `./scripts/setup_mujoco_warp.sh [target_dir]` clones
https://github.com/google-deepmind/mujoco_warp into `../mujoco_warp` (default). It then checks out the tested commit
(`2c09ab9`), applies `patches/mujoco_warp/0001-gpu-noslip-solver.patch` and, inside an active Python environment, installs
it with `pip install --no-deps -e`. The patch adds `mujoco_warp/_src/noslip.py` plus small hooks in `solver.py`,
`types.py` and `io.py` (Apache 2.0, like MuJoCo Warp).

### 1.3 Python environment - option A: conda

`environment.yaml` pins the exact package set every result was measured with (89 packages; torch from the CUDA 12.8
wheel index).

```bash
cd tdmpc_gpu_warp
conda env create -f environment.yaml         # ~3 min
conda activate tdmpc-warp
./scripts/setup_mujoco_warp.sh               # MuJoCo Warp + GPU noslip solver in ../mujoco_warp (pip install -e)
```

### 1.4 Python environment - option B: uv

`pyproject.toml` + `uv.lock` define the same package versions. `setup_uv.sh` performs these steps:
1. installs Python 3.11;
2. creates `.venv/` from the lock file;
3. sets up `../mujoco_warp` with `scripts/setup_mujoco_warp.sh` and installs it editable;
4. runs a short import check.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed yet
cd tdmpc_gpu_warp
./setup_uv.sh                                      # ~1 min with a warm cache; .venv is ~7 GB
source .venv/bin/activate                          # or prefix commands with `uv run`
```

**Why mujoco_warp is installed separately:** `../mujoco_warp` is installed with `--no-deps`, because its own
`pyproject.toml` points `mujoco` at a nightly package index. Its dependencies are already in `pyproject.toml`.

**Caution:** a plain `uv sync` removes packages that are not in the lock file, including the editable mujoco_warp.
Use `uv sync --inexact`, or rerun `./setup_uv.sh`, after changing dependencies.

**Verified versions (both options):** Python 3.11, torch 2.11.0+cu128, warp-lang 1.18.0, mujoco 3.15.0,
dm_control 1.0.48, gym 0.26.2, numpy 2.4.6; NVIDIA driver 580.178, RTX 3090. Both environments were created from
scratch from these files and trained walker-walk successfully.

### 1.5 Check the installation

```bash
python check_system.py        # driver, GPU, MPS, libcuda, tools, torch/warp/mujoco_warp - prints OK / WARN / FAIL
python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true max_env_steps=14000 log_root=/tmp/tdmpc_check
```

The short run passes the seeding phase (10k environment steps) and performs the first updates, which takes about
2 minutes.

**Notes**
- MuJoCo Warp must be the patched checkout from `scripts/setup_mujoco_warp.sh`. Upstream MuJoCo Warp raises
  `NotImplementedError` for models that use the noslip solver, which affects the three dog tasks.
- No rendering backend is needed. Video recording is disabled in the port: `save_video` is forced to false, and no
  image files are written.
- The CPU baseline (`../tdmpc`) is only needed to reproduce the CPU-vs-GPU comparisons. It uses its own environment
  (Python 3.8, MuJoCo 2.1, dm_control as in the original repository).

## 2. Training one seed

**Before you start:** run every command from the repository root (`tdmpc_gpu_warp/`). The config directory `cfgs/` and the
log directory `logs/` are resolved relative to the working directory, as in the original code.

```bash
conda activate tdmpc-warp
cd tdmpc_gpu_warp

# Recommended: optimised GPU port (identical maths to the original agent)
python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true

# Faithful GPU port: the original agent code, environment on the GPU
python src/train.py task=walker-walk seed=1
```

**Which one to use**

| mode | agent | speed vs. original (H = 5) | when to use |
|---|---|---|---|
| optimised (`graph_agent=true pace_host=true`) | CUDA-graph captured planning and update (`algorithm/tdmpc_graph.py`), bitwise identical to the original agent in deterministic tests | 2.4-4.8x | default choice |
| faithful (no extra flags) | original `algorithm/tdmpc.py` (3-line change) | 1.2-1.4x | reference / debugging |

**Task names** are those in `tasks.txt`, for example `cartpole-swingup`, `walker-run`, `humanoid-walk` and `dog-run`.
The modality is always `state`; pixel observations are not supported.

**All original TD-MPC options work unchanged** and are set as `key=value` on the command line. Examples:
`horizon=10`, `exp_name=myexp`, `eval_freq=20000`, `eval_episodes=10`, `train_steps=...`, `save_model=true`.

**Per-task hyperparameters** come from `cfgs/default.yaml` and `cfgs/tasks/<domain>.yaml`, exactly as in the original
code.

### 2.1 Options added by the port (`src/train.py`)

| option | default | meaning |
|---|---|---|
| `graph_agent` | `false` | use the CUDA-graph agent (same maths, removes Python launch overhead) |
| `pace_host` | `false` | after each step the CPU thread *sleeps* until the GPU is done (cuts CPU use to ~0.02-0.17 cores; no effect on results) |
| `max_env_steps` | none | stop early after this many environment steps (benchmarking) |
| `log_root` | `./logs` | root directory of the logs |
| `reset_mode` | `gpu` | `gpu`: episode resets sampled on the GPU (same distribution); `cpu`: reproduce the original numpy reset stream via dm_control (slower) |
| `cuda_graph_env` | `true` | capture the environment step in a CUDA graph |
| `env_graph_mode` | `fused` | `fused` (one graph per agent step) or `split` (separate physics / reward / observation graphs; bit-identical) |
| `skip_substep_sensors` | `false` | skip sensor evaluation in intermediate physics sub-steps (bit-identical observations; negligible gain) |

**Environment variable:** `TDMPC_BLOCKING_SYNC=0` disables the blocking-sync CUDA context flag. That flag makes waiting
host threads sleep instead of spin, and is on by default.

## 3. Training many seeds of one configuration in one process (vectorised)

```bash
python src/train_vec.py task=walker-walk num_seeds=8 seed=1 horizon=5
```

- **Seeds:** the run trains seeds `seed, seed+1, ..., seed+num_seeds-1`, each as a fully independent TD-MPC agent.
  Each seed has its own TOLD model, optimiser, replay buffer, planner and environment instance (one MuJoCo Warp
  world).
- **How it runs:** all seeds advance in lockstep, and one kernel serves all of them.
- **Logs:** written per seed, exactly as for `train.py`.

**Memory:** this is the most memory-efficient way to run many seeds. One CUDA context is shared, so walker-walk fits
136 seeds at H = 5 (separate processes: 33).

| option | default | meaning |
|---|---|---|
| `num_seeds` | 1 | number of independent seeds in the process |
| `pace_host` | `true` | as above |
| `cuda_graph_agent` | `true` | CUDA-graph captured planning and update |
| `ckpt_unroll` | `false` | activation checkpointing of the H-step model unroll: **bitwise identical** results, much less activation memory, some extra compute (removes dog's OOM limit) |
| `batched_update` | `false` | batch the horizon-independent parts of the update (equal up to float rounding; 1.03-1.27x) |
| `eval_graphs` | `true` | CUDA-graph captured planning during evaluation |
| `precision` | `fp32` | `tf32`, `bf16`, `fp16`: reduced precision (**not** mathematically equivalent, see section 6) |
| `buffer_dtype` | `fp32` | `bf16` / `fp16` replay-buffer observations (not equivalent) |
| `max_env_steps`, `log_root`, `reset_mode`, `cuda_graph_env`, `env_graph_mode`, `skip_substep_sensors` | | as for `train.py` |

> **Known issue:** runs can abort with a rare `illegal memory access` ([docs/08](08_known_issues.md)).
> Vectorised processes are affected most, especially under MPS. No resume is implemented, so an aborted run must be
> restarted. For long experiments prefer separate single-seed processes (section 4).

## 4. Several runs at the same time (recommended for experiments)

**Why MPS is needed:** without CUDA MPS, processes on one GPU are time-sliced, so running them in parallel gains
nothing. **Start the MPS daemon first**; then the processes' kernels run concurrently.

> **Risk under MPS:** a rare GPU fault (open issue, [docs/08](08_known_issues.md)) in one MPS client ends **all**
> co-running runs. It was seen once with single-seed processes and more often with vectorised ones. Check the runs
> regularly. If that risk is unacceptable, run the same commands without starting MPS: no fault has been observed
> without MPS, but the runs then take as long as running them one after another.

```bash
# start MPS (user-level, no root needed). Keep the pipe path short: unix sockets are limited to ~108 characters.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_$USER/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

# 5 seeds of one task, one process each
for seed in 1 2 3 4 5; do
  python src/train.py task=walker-walk seed=$seed graph_agent=true pace_host=true > walker_s$seed.out 2>&1 &
  sleep 20                                   # stagger the start-up (CUDA-graph capture)
done
wait

echo quit | nvidia-cuda-mps-control         # stop MPS when all runs are done
```

The training processes must inherit the same `CUDA_MPS_PIPE_DIRECTORY` as the daemon, so start them from the same
shell.

### 4.1 How many runs, and how to organise seeds

| N concurrent runs (MPS) | total throughput vs. 1 run (non-dog) | dog | speed of each run |
|---|---|---|---|
| 1 | 1.0x | 1.0x | 100 % |
| 2 | 1.4-1.5x | 1.4x | ~70-77 % |
| 4 | 1.9-2.2x | 1.7x | ~50 % |
| 8 | 2.3-2.7x | 1.9x | ~30 % |

**What the table means**
- One optimised run already keeps the GPU 86-97 % busy. Any concurrent run therefore slows the others: the number of
  runs that do *not* slow each other down is one.
- The **total** throughput still rises with concurrent runs, so for 5 seeds per task **running 5-8 runs in parallel
  finishes about 2x earlier** than running them back-to-back.

**Advice**
- Start heavy tasks (dog, humanoid) first.
- Don't go beyond ~8 runs: the gain flattens and memory gets tight for dog.

**Memory limits** (separate processes, H = 5): cartpole 35, walker 33, humanoid 22, dog 11 runs. For other horizons
and mixed-horizon combinations see [docs/04](04_multi_run_capacity.md).

### 4.2 Expected wall-clock time

Per-task times for 1 seed and 5 seeds (back-to-back and in parallel) are in the
[README](../README.md#total-run-time-per-task-1-seed-vs-5-seeds) and in [results_runtime.md](results_runtime.md).

**Examples at H = 5**

| task | 1 seed | 5 seeds in parallel (MPS) |
|---|---|---|
| walker-walk | 1.7 h | 3.8 h |
| humanoid-run | 13.7 h | 34.5 h |
| dog-run | 33.4 h | 93 h |

## 5. Outputs

**Run directory:** `logs/<task>/state/<exp_name>/<seed>/`

| file | content |
|---|---|
| `eval.log` | CSV `env_step,episode_reward` (mean evaluation return), same format as the original TD-MPC |
| `train.log` | CSV per training episode: `episode,env_step,episode_reward,total_time,episode_time,agent_steps_per_s` |
| `models/model.pt` | final model if `save_model=true` (`train.py`) |

**Console output:** the original logger's lines, plus a progress line per episode.

**Plotting:** `bench/analyze_learning.py` plots several runs' `eval.log` files against each other.

## 6. Precision options (bf16 / fp16 / TF32)

**Default:** float32, identical to the original.

**Reduced precision** is available in `train_vec.py` (`precision=bf16|fp16|tf32`) but is **not mathematically
equivalent**:

| | speed-up | VRAM saving | learning (walker-walk) |
|---|---|---|---|
| single seed | 1.06-1.24x | 3-10 % | |
| many seeds in one process | bf16 1.6x, fp16 1.5x, TF32 1.25x | 3-10 % | bf16 / fp16 like fp32 (3 seeds, 40k steps) |

**Recommendation:** use these modes only after checking the learning curves on your task. To save memory, use
`ckpt_unroll=true` instead: it is exact. Details: [docs/07](07_precision.md).

## 7. Reproducing the measurements

[docs/05](05_how_to_run.md) lists the script behind every table and figure. Each script in `bench/` has a module
docstring with its purpose, command line, output location and the document that reports the result.

**Script families**

| prefix | purpose |
|---|---|
| `validate_env.py`, `divergence_study.py`, `test_*.py` | environment validation and equivalence tests |
| `bench_step.py`, `sweep_h.py`, `oom_horizon.py`, `vec_memory.py` | single-run speed and memory |
| `multi_run.py`, `pack_test.py`, `packing_study.py`, `contention_study.py`, `interference_*.py`, `partition_study.py` | multi-run capacity, contention and interference |
| `learning_runs.py`, `train_cpu_baseline.py` | learning-curve experiments |
| `analyze_*.py` | tables and figures in `docs/` |
| `fault_*.py`, `mps_fault_repro.py`, `debug_oob.py`, `train_vec_warpdebug.py` | diagnostics for the open issue in docs/08 |

**Which environment:** scripts that time the original CPU code run it in the `tdmpc` environment. The GPU scripts use
`tdmpc-warp`.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `NotImplementedError: noslip solver not implemented` | upstream MuJoCo Warp is installed; run `./scripts/setup_mujoco_warp.sh` (patched checkout with the GPU noslip solver) |
| several processes run but the total speed does not increase | MPS daemon not running, or the processes do not see the same `CUDA_MPS_PIPE_DIRECTORY` |
| MPS daemon fails to start | pipe directory path too long (unix-socket limit); use a short path under `/tmp` |
| all runs abort together under MPS | a fatal GPU fault in one MPS client ends all clients (rare; open issue, [docs/08](08_known_issues.md)); restart the runs, prefer single-seed `train.py` processes, or run without MPS (no fault observed, no parallel speed-up) |
| `CUDA out of memory` | check the capacity tables in [docs/04](04_multi_run_capacity.md); for dog at very long horizons use `train_vec.py ckpt_unroll=true` |
| results of one seed are not bit-reproducible | expected: GPU physics uses float atomics (not bitwise deterministic) and the GPU random generator differs from numpy's; statistics are unaffected ([docs/03](03_validation.md)) |
| first episodes are slower | CUDA graphs are captured during the first calls (planning per horizon, update after a few eager updates) |
