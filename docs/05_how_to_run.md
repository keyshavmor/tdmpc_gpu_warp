# 5. How to run (RTX 3090, everything on the GPU)

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Machine set-up (NVIDIA driver, MPS, optional tools), the conda / uv environments and all training options are
described in the [user guide](USER_GUIDE.md); this chapter is the short reference and lists the script behind every
measurement.

```bash
conda activate tdmpc-warp
cd tdmpc_gpu_warp
```

## 5.1 One seed (drop-in replacement for `tdmpc/src/train.py`)

```bash
# faithful port: original agent, GPU physics/obs/reward/resets
python src/train.py task=walker-walk seed=1
# optimised (same maths): CUDA-graph agent + host pacing  (~3.5x faster than the CPU original)
python src/train.py task=walker-walk seed=1 graph_agent=true pace_host=true
```
Logs: `logs/<task>/state/<exp_name>/<seed>/{eval.log,train.log}` (same `eval.log` format as the original).
All options of `cfgs/*.yaml` work as before (`horizon=`, `exp_name=`, ...). Video is hard-disabled.

## 5.2 Many seeds of one configuration - one process (most efficient; see the known issue in chapter 8)

```bash
python src/train_vec.py task=walker-walk num_seeds=16 seed=1 horizon=5
```
Runs seeds 1..16 in lockstep: one N-world mujoco_warp simulation, 16 independent TOLD models / optimisers /
replay buffers / planners stored as stacked tensors (one kernel serves all seeds). Per-seed logs in
`logs/<task>/state/<exp_name>/<seed>/`. Uses ~0.02-0.1 CPU core in total and shares one CUDA context,
so it needs far less VRAM per seed than separate processes (chapter 4).

Extra switches (vectorised agent):
* `ckpt_unroll=true` - activation checkpointing of the H-step unroll: bitwise-identical results,
  horizon-driven activation memory ~16-22x smaller, some extra compute (chapter 4).
* `precision=bf16|fp16|tf32`, `buffer_dtype=bf16|fp16` - reduced precision (chapter 7; changes numerics).
* `batched_update=true` - batched horizon-independent parts of update() (exact up to rounding, 1.03-1.27x).
* `skip_substep_sensors=true` - skip sensors in intermediate physics sub-steps (bit-identical, ~0 % end-to-end).
* `eval_graphs=false` - plan eagerly during evaluation; `env_graph_mode=split|fused` - env CUDA-graph layout.

## 5.3 Several runs / configurations at the same time (recommended for long runs: separate single-seed processes)

Run one process per configuration (each can itself be multi-seed via `train_vec.py`) and **start the CUDA MPS
daemon first**, otherwise the processes are time-sliced by the driver and do not run concurrently:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_$USER/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d                     # start MPS (user-level, no root needed)

for seed in 1 2 3 4 5; do          # 5 seeds of one task, one process each
  python src/train.py task=walker-walk seed=$seed graph_agent=true pace_host=true &   # see docs/08 for the rare MPS fault
  sleep 20                          # stagger starts
done
wait
echo quit | nvidia-cuda-mps-control            # stop MPS
```
(Use a short pipe directory path - unix sockets are limited to ~108 characters.)

## 5.4 Reproducing the measurements

| what | command |
|---|---|
| environment validation (21 tasks) | `python bench/validate_env.py` -> `results/validation/` |
| agent equivalence | `bench/test_graph_agent.py`, `bench/test_vec_agent.py`, `bench/test_ckpt.py`, `bench/test_precision_agent.py`, `bench/test_per_sampling.py` (replay sampling) |
| noslip check (dog) | `python bench/test_noslip.py` |
| horizon sweep CPU vs GPU | `python bench/sweep_h.py` (CPU mode runs the original code in the `tdmpc` env) |
| OOM horizon | `python bench/oom_horizon.py` |
| vectorised memory | `python bench/vec_memory.py` |
| packing (runs until OOM) | `python bench/packing_study.py` |
| multi-run contention | `python bench/contention_study.py` |
| learning runs | `python bench/learning_runs.py start|status|stop` |
| tables/figures | `bench/analyze_*.py` |
