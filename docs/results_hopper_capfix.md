# Real 5-seed training batches under MPS

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Label `baseline_mps_capfix`: 5 single-seed runs per task in parallel under CUDA MPS (`src/launch_parallel.py --mps`, CUDA-graph agent + host pacing, default hyperparameters, H = 5, full training budget, checkpoints every 15 min, W&B logging), tasks one after another, no other CUDA process on the GPU (docs/USER_GUIDE.md section 4.4). Reference = the paper's 5 seeds of the original code; p = Welch test per evaluation step.

## Time and stability

| task | measured wall time (h) | predicted (h, docs/results_runtime.md) | launches | crashes | restarts |
|---|---|---|---|---|---|
| hopper-hop | 3.93 | 1.8 | 25 | 15 | 10 |

**Notes on individual batches:**

- hopper-hop: rerun with MuJoCo Warp patch 0002 (capsule collision). Under MPS 16:16-16:42 (3 MPS faults, all runs resumed); from 16:42:38 to 20:38:08 without MPS (5 time-sliced processes, no fault). wall_h = the run without MPS; not comparable with the 5-seed MPS batch times.

Kernel GPU fault records (Xid) during the batches: 9.

## Learning vs the paper

### hopper-hop

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 0 +- 0 | 0 +- 0 (5) | 0.25 |
| 100k | 143 +- 78 | 75 +- 78 (5) | 0.22 |
| 200k | 242 +- 144 | 118 +- 125 (5) | 0.19 |
| 300k | 296 +- 168 | 158 +- 129 (5) | 0.19 |
| 400k | 313 +- 160 | 200 +- 156 (5) | 0.31 |
| 500k | 325 +- 153 | 235 +- 152 (5) | 0.40 |

Per-step (descriptive): 26 evaluation steps, min p = 0.116, steps with p < 0.05: 0 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 136.1 +- 117.8 vs paper 221.5 +- 132.4 (-38.58 %), Welch t-test p = 0.313.
  Original code on this machine (5 seeds, whole run): 184.8 +- 108.3; GPU vs original-on-this-machine -26.38 % (p = 0.515), original-on-this-machine vs paper -16.58 % (p = 0.644).
Primary test, per-seed mean return over the from 100k steps: GPU 164.5 +- 140.4 vs paper 267.7 +- 159.8 (-38.53 %), Welch t-test p = 0.310.
  Original code on this machine (5 seeds, from 100k steps): 221.9 +- 131.6; GPU vs original-on-this-machine -25.84 % (p = 0.524), original-on-this-machine vs paper -17.11 % (p = 0.635).

