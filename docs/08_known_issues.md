# 8. Known issues and limitations

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

## 8.1 GPU fault under CUDA MPS: trigger found, mitigated

**Symptom:** `CUDA error: an illegal memory access was encountered` / `unspecified launch failure` in a training
process running under CUDA MPS. The kernel log shows Xid 31: an MMU fault (`FAULT_PDE`, `ACCESS_TYPE_VIRT_READ`) by
the constant/instruction cache unit (`GPCCLIENT_GCC`). A rarer form, seen both with and without the trigger below, is Xid 13
`SKEDCHECK22_INVALIDATE_ACTIVE_QMD`. A fatal fault in one MPS client tears down every client on the
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

**Long-run validation under the rule:** the 15 real 5-seed training batches (docs/04 section 4.9) ran 34.0 h with
5 MPS clients at all times, about 170 process-hours.
- 14 batches (about 147 process-hours, including all batches of the previous evening and night) had 0 faults.
- On 2026-10-11 MPS-only operation faulted repeatedly: walker-run 6 times in a 1.8 h window, and a later hopper-hop
  rerun 3 times in 26 min. See "Residual faults in MPS-only operation" below.
- MPS-only operation in total (stress tests + batches + the rerun): 9 faults in about 214 process-hours, rate 0.042
  per process-hour [95 % CI 0.019, 0.080], all on 2026-10-11. Mixed operation: about 2 per process-hour.

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

**Residual faults in MPS-only operation (2026-10-11).** Nine faults, each ending all 5 co-running runs:
- walker-run at 08:17, 08:41, 09:04, 09:10, 09:51 and 10:02; then 2.9 h without a fault;
- a hopper-hop rerun at 16:24, 16:33 and 16:40, 8-9 min apart.

What was checked:
- **No external trigger.** The launcher's guard saw no non-MPS CUDA process, and none showed in `nvidia-smi`.
  Every process with the NVIDIA device files open was a long-running desktop program, the MPS daemon and its
  clients, or W&B's NVML-based system monitor. No package, driver or system change happened. The GPU was healthy:
  71 C, no thermal slowdown, no PCIe replays.
- **Signature:** eight were Xid 13, `Graphics Exception: SKEDCHECK22_INVALIDATE_ACTIVE_QMD failed` (a scheduler sanity
  check on a kernel launch descriptor). One, at 09:51, was an Xid 31 MMU fault (`GPCCLIENT_T1_0`, address 0). The same
  Xid 13 signature appeared twice before, during the `mixed_nocond` experiment.
- **Faulting client:** the MPS server named a different seed at different times (seed 5 four times, then seeds 1 and
  3). The replay buffers hold no NaN or extreme states. No single run or state is the trigger.
- **It needs MPS.** The same hopper-hop runs, continued without MPS as 5 ordinary processes sharing the GPU by
  time-slicing, ran 3.9 h (19.6 process-hours) with no fault and no Xid record.
- **The rate changed abruptly.** reacher-hard finished without a fault at 08:15, and walker-run faulted 2.5 min after
  starting at 08:15, with the same code. The most likely explanation is a change of state in the GPU / driver,
  which a reboot (driver reload) would test. That was not done here because it would close the desktop session.
- **Same faults outside this project:** NVIDIA's forums have reports with the same Xid 13 signature in unrelated
  settings (an inference server after 15 days of uptime; bursty small-kernel launches on an RTX 5090), without a
  diagnosis from NVIDIA.

**Conclusion:** a fault in the driver / MPS layer, rare over most of this project's MPS-only operation, but
frequent within one day. Its cause is not known. The safeguards handled every fault: the launcher resumed all runs
from their checkpoints, with exact optimiser, RNG and buffer state and an exact W&B history. walker-run lost about
0.8 h of a 4.67 h batch.

**If MPS faults recur:**
1. Shorten the checkpoint interval (`--checkpoint_minutes 2`; a checkpoint takes well under a second for these tasks)
   and raise `--max_restarts`, so that progress is kept between faults.
2. If they come every few minutes, run without MPS: 5 seeds then share the GPU by time-slicing. That is about 2x
   slower than under MPS, but no fault was observed.
3. Reboot the machine to reset the GPU and driver state.

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
- the cause of the MPS-only faults above (Xid 13 `SKEDCHECK22_INVALIDATE_ACTIVE_QMD`), and whether a reboot clears them;
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
