# 8. Known issues and limitations

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

## 8.1 Rare GPU fault (illegal memory access), mainly under CUDA MPS - open

**Symptom:** `CUDA error: an illegal memory access was encountered` / `unspecified launch failure`. Under CUDA MPS a
fatal fault in one client tears down *every* client on the GPU, so all co-running runs end at once.

**Observations (8 faults):**
* **Vectorised processes (7 faults):** all in vectorised training processes (`train_vec.py`, `nworld = num_seeds > 1`)
  whose physics step is captured in a CUDA graph. This covers both capture modes: the default `fused` mode, used for all
  measurements, and the alternative `split` mode (bit-identical results). Faults also occurred with the agent's CUDA
  graphs disabled.
* **Single-world process (1 fault, 2026-10-08):** in a **single-world** process (`train.py graph_agent=true
  pace_host=true`, walker-walk). It happened about 2 minutes after start, during the random-action seeding phase, so
  only the environment's physics graph was running. 10 MPS clients were active, and all of them were torn down. Before
  this, hours of MPS throughput and interference studies with single-world processes had run without a fault, so
  single-world processes are **not** immune under MPS.
* **Frequency:**
  - under MPS with ~6-10 concurrent processes: from ~20 s to > 15 min;
  - without MPS: once in ~6 process-hours (vectorised);
  - single-world processes without MPS: never observed.
* **Kernel driver log:** every fault since 2026-10-08 15:33 (`journalctl -k | grep Xid`) appears as **Xid 31 - MMU fault, `FAULT_PDE`,
  `ACCESS_TYPE_VIRT_READ`**. The faulting unit is the GPC constant/instruction cache (`GPCCLIENT_GCC`) or L1
  (`GPCCLIENT_T1`), at similar virtual addresses. Two earlier events are Xid 13 `SKEDCHECK22_INVALIDATE_ACTIVE_QMD`.
  This means a read from an address whose page is not mapped at all: a kernel or graph node references memory that
  is no longer (or never was) allocated. Logs of the 2026-10-08 fault are in
  `results/learning/logs/mps_fault_2026-10-08/`.
* **No effect on results was observed:** a fault aborts the process. All completed runs match the reference learning
  curves, and all validation results are bit-identical before and after the fixes below.

**Found and fixed on the way (necessary but not sufficient):**
- warp destroying the torch-owned graph of an external capture (chapter 1, gotcha 7);
- the fp16 loss scaler's growth counter (chapter 7).

**Investigated and ruled out:**
- **Code of a reloaded warp module referenced by a graph.** MuJoCo Warp's `forward` module is rebuilt (new hash) right
  after the first graph capture. Instrumenting warp over a full single-seed training cycle (seeding, 5000+ updates,
  2 evaluations, forced garbage collection; `results/fault_hunt/module_unload_instrumentation.json`) showed
  37 module loads, 1 reload and **0 unloads**: warp retains every module used in a capture (also inside the
  solver's conditional while-loop bodies) for the lifetime of the graph.

**Candidate causes (unconfirmed):**
- the conditional while-loop of MuJoCo Warp's graph-captured constraint solver (the eager path runs a plain loop);
- graph-owned memory being freed or reused while still referenced;
- a read of uninitialised memory that becomes an out-of-bounds index once memory is reused.

The much higher rate under MPS fits the last two: MPS clients share one server address space, and memory is not
scrubbed between clients.

**Diagnostics that did not catch it:**
- A GPU core dump (`bench/fault_coredump.py`, `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`) did not capture a fault within its
  15-minute window.
- `compute-sanitizer` runs out of memory on the graph path and is too slow eagerly.

**Mitigation (until fixed):**
* **The trade-off:** running several runs in parallel needs MPS (without it the GPU time-slices the processes and the
  total throughput does not increase). Under MPS, though, a rare fault ends **all** co-running runs; checkpoint /
  resume (below) limits the loss.
* **Most robust:** separate single-seed processes **without** MPS. No fault has been observed in this setting, but
  there is no parallel speed-up: one run at a time is as fast.
* **Fastest:** 5-8 single-seed processes under MPS, ~2x total throughput, **with checkpoints**. Use
  `src/launch_parallel.py --mps` (USER_GUIDE section 4.3): it writes checkpoints and restarts crashed runs from
  their last checkpoint, so a fault costs at most one checkpoint interval (e.g. 15 min) instead of the run. W&B
  curves stay exact.
* `train_vec.py` (vectorised): suitable for shorter runs. It has the highest fault rate under MPS, so run it without
  MPS.

## 8.2 Other limitations
* State modality only (no pixel observations).
* humanoid-* and dog-* were not trained and are excluded from the learning-curve tests and the remaining
  baseline work (MPS stability tests, re-timings, further optimisation): their 3M / 5M-step budgets (14-33 h per
  seed even on the optimised GPU port) make them too expensive for this project. Their port, validation, speed and
  memory measurements (short benchmarks) are reported in docs/02-04.
* mujoco_warp is float32 and not bitwise deterministic run-to-run (atomics); trajectories of chaotic contact tasks
  decorrelate from CPU MuJoCo after 10-50 agent steps (as a float32-rounded CPU run does).
* mujoco_warp's box-box collision generates different contact counts than C MuJoCo for 0.9 % of dog's contacting
  geom pairs.
* The noslip port supports pyramidal cones only (sufficient for all DMC tasks).
* Episode-initialisation RNG stream differs from numpy's (distribution identical; `reset_mode=cpu` reproduces the
  original stream exactly).
