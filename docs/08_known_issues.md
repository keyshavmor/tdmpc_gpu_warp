# 8. Known issues and limitations

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

## 8.1 GPU fault under CUDA MPS: trigger found, mitigated

**Symptom:** `CUDA error: an illegal memory access was encountered` / `unspecified launch failure` in a training
process running under CUDA MPS. The kernel log shows Xid 31: an MMU fault (`FAULT_PDE`, `ACCESS_TYPE_VIRT_READ`) by
the constant/instruction cache unit (`GPCCLIENT_GCC`). A fatal fault in one MPS client tears down every client on the
GPU, so all co-running runs end at once.

**Trigger (reproduced and confirmed):** an ordinary, non-MPS CUDA process doing GPU work on the same GPU at the same
time as MPS clients.

- **Which processes trigger it:** a process running warp / MuJoCo Warp physics or our training. Its start, its
  running and its end can all trigger the fault, typically within 2-3 minutes. The faulting process is always one of
  the MPS clients, never the ordinary process.
- **No trigger needed beyond that:** neither a SIGKILL nor anything else; a process ending cleanly also triggers it.
- **What did not trigger it:**
  - a minimal torch process (about 130 starts and clean exits in 45 min);
  - processes that create and destroy Vulkan instances.

**Evidence** (`bench/mps_fault_hunt.py`, summary in [results_fault_hunt.md](results_fault_hunt.md)): rounds of 6 MPS
training processes plus the stated extra processes. A round ends at the first fault.

| setup | exposure (process-h) | faults | rate per process-h [95 % CI] |
|---|---|---|---|
| MPS clients only: 7 configurations, incl. eager physics, physics-only, ~120 MPS clients started and SIGKILLed, and exact controls of the mixed setups below with the extra processes routed through MPS | 41.7 | **0** | 0 [0, 0.088] |
| + 2 **non-MPS** training processes (restarted every 120 s) | 6.2 | **13** | 2.1 [1.1, 3.6] |
| + 2 **non-MPS** physics processes, ended by SIGKILL every 120 s | 4.6 | **10** | 2.2 [1.0, 4.0] |
| + 2 **non-MPS** physics processes, clean exit every 120 s | 4.7 | **9** | 1.9 [0.9, 3.7] |
| + 2 non-MPS minimal torch processes (20 s each) | 5.5 | 0 | 0 [0, 0.67] |
| + 1 process creating Vulkan instances every second | 5.0 | 0 | 0 [0, 0.74] |

**Long-run validation under the rule:** the six real 5-seed training batches (docs/04 section 4.9) ran 13.6 h with
5 MPS clients at all times. That is 68 process-hours with 0 crashes, 0 restarts and 0 Xid records. MPS-only
operation in total (stress tests + batches): **0 faults in about 110 process-hours** (exact 95 % upper bound 0.034 per
process-hour).

**How strong the evidence is:**
- Each mixed setup has an exact MPS-only control: same processes, same restart pattern, but the extra processes go
  through MPS. All three controls had 0 faults in 45 min each.
- If MPS-only and mixed operation had the same fault rate, all 32 faults would land in the mixed runs with
  probability about 10^-18.
- **Earlier faults fit the pattern:** every earlier fault coincided with non-MPS GPU processes running next to MPS
  clients (the timeline is in the session records):
  - learning runs started without MPS while MPS experiments ran;
  - the 2026-10-08 20:23 fault, 55 s after two non-MPS test processes were started.
- The one "fault without MPS" (a non-MPS learning run) happened while an MPS experiment ran on the same GPU.

**Root cause:** in the NVIDIA driver / MPS (driver 580.178, RTX 3090), not in this repository's code.
- MPS-only operation of the same code is clean.
- The faulting process is an MPS client hit by another, non-MPS context's activity.
- The kernel-level fault is in the MPS server's address space.
- **Which feature of the triggering process matters: CUDA-graph execution.**

  | the non-MPS process runs the physics ... | faults |
  |---|---|
  | in CUDA graphs (default) | 9-13 per variant (table above) |
  | in CUDA graphs without conditional nodes (`mixed_nocond`) | 6 in 7 rounds (5.1 process-h) |
  | eagerly, without CUDA graphs (`mixed_eager`) | 0 in 45 min (5.6 process-h) |

  A minimal torch process without graphs never triggered it either. So the trigger is a non-MPS context *executing
  CUDA graphs*. warp's captured graphs contain graph memory nodes, whose launches map and unmap GPU memory, which
  would fit the page-directory faults (`FAULT_PDE`) in the MPS clients. That is a hypothesis for a report to NVIDIA,
  not something verified here. It does not change the mitigation: every training process of this repository uses
  CUDA graphs.

**Ruled out along the way:**
- warp modules being unloaded under a live graph (instrumented: 0 unloads in a full training cycle);
- the earlier external-capture bug: fixed, but faults continued until the trigger was removed;
- eager physics, physics-only load and process start-up churn of MPS clients: all 0 faults under MPS-only operation.

**Mitigation: operating rule, plus safeguards in the code**
1. **While MPS runs are active on a GPU, every CUDA process on that GPU must be an MPS client.** Start every other GPU
   job from a shell with the same `CUDA_MPS_PIPE_DIRECTORY`, or wait until the MPS batch has finished. This applies
   to evaluation scripts, notebooks and other experiments, and also to a second launcher with a different MPS
   directory, whose server would be a non-MPS process to the first one. The controls above show that MPS clients
   may start, stop and even be killed freely without faults.
2. **Guard in `src/launch_parallel.py --mps`:**
   - before starting, it waits for (or aborts on) ordinary CUDA processes on the GPU (`--foreign wait|warn|abort`),
     using `nvidia-smi` process types: `C` = ordinary, `M+C` = MPS client, `C+G` / `G` = desktop graphics;
   - during the run, it logs any that appear (`foreign_process` events), so a crash can be attributed.
3. **Graceful stop:** `train.py` handles SIGTERM / SIGINT. It finishes the current step, synchronises the GPU and
   exits. NVIDIA documents that terminating an MPS client with outstanding GPU work can corrupt the server and its
   other clients. That was not reproduced here, but there is no reason to risk it.
4. **Checkpoint / resume:** if a crash happens anyway, it costs at most one checkpoint interval
   (USER_GUIDE section 4.3).
5. **Stronger, needs root, not tested here:** `sudo nvidia-smi -c EXCLUSIVE_PROCESS`, NVIDIA's recommended
   compute mode for MPS.
   - With it, only the MPS server holds a CUDA context. Other CUDA programs fail to start on that GPU instead of
     crashing the MPS clients.
   - Desktop graphics are not affected by the compute mode.
   - Revert with `sudo nvidia-smi -c DEFAULT`.

**What remains unknown:**
- whether desktop programs that open and close GPU compute contexts (`C+G` processes, e.g. editor or file-manager
  windows) can trigger the fault. Their long-lived presence during all experiments did not. Creating Vulkan
  instances did not either (a weak test). Opening and closing windows was not tested.
- a fix inside the driver, which is NVIDIA's to make.

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
