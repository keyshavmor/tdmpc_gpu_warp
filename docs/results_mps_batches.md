# Real 5-seed training batches under MPS

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Label `baseline_mps`: 5 single-seed runs per task in parallel under CUDA MPS (`src/launch_parallel.py --mps`, CUDA-graph agent + host pacing, default hyperparameters, H = 5, full training budget, checkpoints every 15 min, W&B logging), tasks one after another, no other CUDA process on the GPU (docs/USER_GUIDE.md section 4.4). Reference = the paper's 5 seeds of the original code; p = Welch test per evaluation step.

## Time and stability

| task | measured wall time (h) | predicted (h, docs/results_runtime.md) | launches | crashes | restarts |
|---|---|---|---|---|---|
| walker-walk | 3.90 | 3.8 | 5 | 0 | 0 |
| cheetah-run | 1.65 | 1.5 | 5 | 0 | 0 |
| finger-spin | 3.31 | 2.7 | 5 | 0 | 0 |
| cartpole-swingup | 0.97 | 1.0 | 5 | 0 | 0 |
| reacher-easy | 1.63 | 1.3 | 5 | 0 | 0 |
| cup-catch | 2.17 | 2.3 | 5 | 0 | 0 |
| acrobot-swingup | 1.76 | 1.6 | 5 | 0 | 0 |
| cartpole-swingup-sparse | 0.96 | 1.0 | 5 | 0 | 0 |

Kernel GPU fault records (Xid) during the batches: 0.

## Learning vs the paper

### walker-walk

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 35 +- 1 | 36 +- 1 (5) | 0.17 |
| 100k | 963 +- 5 | 967 +- 4 (5) | 0.24 |
| 200k | 969 +- 7 | 971 +- 5 (5) | 0.64 |
| 300k | 974 +- 4 | 977 +- 3 (5) | 0.31 |
| 400k | 973 +- 9 | 980 +- 4 (5) | 0.17 |
| 500k | 977 +- 3 | 978 +- 5 (5) | 0.72 |

Per-step (descriptive): 26 evaluation steps, min p = 0.013, steps with p < 0.05: 5 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 918.4 +- 15.9 vs paper 914.5 +- 9.0 (+0.43 %), Welch t-test p = 0.646.
Primary test, per-seed mean return over the from 100k steps: GPU 974.5 +- 1.5 vs paper 972.3 +- 2.3 (+0.23 %), Welch t-test p = 0.111.

### cheetah-run

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 11 +- 1 | 12 +- 1 (5) | 0.69 |
| 100k | 603 +- 53 | 567 +- 59 (5) | 0.37 |
| 200k | 561 +- 98 | 653 +- 183 (5) | 0.38 |
| 300k | 769 +- 65 | 840 +- 28 (5) | 0.04 |
| 400k | 777 +- 68 | 851 +- 49 (5) | 0.08 |
| 500k | 814 +- 31 | 846 +- 25 (5) | 0.11 |

Per-step (descriptive): 26 evaluation steps, min p = 0.000, steps with p < 0.05: 5 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 682.3 +- 24.0 vs paper 643.5 +- 36.0 (+6.03 %), Welch t-test p = 0.085.
  Original code on this machine (5 seeds, whole run): 678.7 +- 27.8; GPU vs original-on-this-machine +0.54 % (p = 0.830), original-on-this-machine vs paper +5.46 % (p = 0.125).
Primary test, per-seed mean return over the from 100k steps: GPU 771.7 +- 23.3 vs paper 728.0 +- 38.8 (+6.00 %), Welch t-test p = 0.070.
  Original code on this machine (5 seeds, from 100k steps): 763.0 +- 36.4; GPU vs original-on-this-machine +1.14 % (p = 0.667), original-on-this-machine vs paper +4.81 % (p = 0.179).

### finger-spin

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 0 +- 0 | 1 +- 1 (5) | 0.27 |
| 100k | 949 +- 36 | 949 +- 39 (5) | 0.99 |
| 200k | 979 +- 4 | 978 +- 8 (5) | 0.84 |
| 300k | 983 +- 6 | 962 +- 39 (5) | 0.27 |
| 400k | 964 +- 41 | 983 +- 4 (5) | 0.35 |
| 500k | 981 +- 7 | 971 +- 23 (5) | 0.44 |

Per-step (descriptive): 26 evaluation steps, min p = 0.006, steps with p < 0.05: 2 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 900.1 +- 21.2 vs paper 907.0 +- 8.1 (-0.76 %), Welch t-test p = 0.526.
Primary test, per-seed mean return over the from 100k steps: GPU 973.8 +- 7.4 vs paper 973.8 +- 8.9 (+0.00 %), Welch t-test p = 0.996.

### cartpole-swingup

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 94 +- 8 | 114 +- 14 (5) | 0.01 |
| 100k | 859 +- 6 | 858 +- 7 (5) | 0.85 |
| 200k | 873 +- 8 | 875 +- 2 (5) | 0.60 |
| 300k | 877 +- 6 | 878 +- 3 (5) | 0.72 |
| 400k | 877 +- 7 | 879 +- 3 (5) | 0.61 |
| 500k | 879 +- 6 | 879 +- 3 (5) | 0.99 |

Per-step (descriptive): 26 evaluation steps, min p = 0.011, steps with p < 0.05: 1 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 801.8 +- 4.3 vs paper 799.3 +- 3.0 (+0.31 %), Welch t-test p = 0.323.
Primary test, per-seed mean return over the from 100k steps: GPU 873.5 +- 3.3 vs paper 873.2 +- 6.7 (+0.04 %), Welch t-test p = 0.929.

### reacher-easy

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 58 +- 16 | 52 +- 19 (5) | 0.62 |
| 100k | 893 +- 94 | 956 +- 41 (5) | 0.22 |
| 200k | 974 +- 16 | 956 +- 39 (5) | 0.40 |
| 300k | 944 +- 48 | 953 +- 38 (5) | 0.75 |
| 400k | 978 +- 6 | 961 +- 40 (5) | 0.41 |
| 500k | 976 +- 13 | 984 +- 1 (5) | 0.28 |

Per-step (descriptive): 26 evaluation steps, min p = 0.003, steps with p < 0.05: 1 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 884.5 +- 19.3 vs paper 895.1 +- 10.2 (-1.18 %), Welch t-test p = 0.319.
Primary test, per-seed mean return over the from 100k steps: GPU 961.2 +- 11.9 vs paper 960.6 +- 7.6 (+0.06 %), Welch t-test p = 0.929.

### cup-catch

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 37 +- 34 | 43 +- 37 (5) | 0.81 |
| 100k | 977 +- 5 | 972 +- 4 (5) | 0.12 |
| 200k | 976 +- 3 | 975 +- 10 (5) | 0.79 |
| 300k | 975 +- 3 | 974 +- 11 (5) | 0.90 |
| 400k | 977 +- 4 | 976 +- 5 (5) | 0.76 |
| 500k | 972 +- 10 | 977 +- 8 (5) | 0.42 |

Per-step (descriptive): 26 evaluation steps, min p = 0.021, steps with p < 0.05: 2 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 915.1 +- 13.6 vs paper 912.1 +- 14.4 (+0.34 %), Welch t-test p = 0.738.
Primary test, per-seed mean return over the from 100k steps: GPU 976.0 +- 1.5 vs paper 973.6 +- 4.7 (+0.24 %), Welch t-test p = 0.334.

### acrobot-swingup

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 3 +- 2 | 1 +- 1 (5) | 0.06 |
| 100k | 51 +- 42 | 58 +- 40 (5) | 0.82 |
| 200k | 130 +- 21 | 130 +- 48 (5) | 1.00 |
| 300k | 231 +- 77 | 268 +- 77 (5) | 0.50 |
| 400k | 285 +- 44 | 393 +- 83 (5) | 0.02 |
| 500k | 342 +- 108 | 365 +- 50 (5) | 0.70 |

Per-step (descriptive): 26 evaluation steps, min p = 0.022, steps with p < 0.05: 1 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 218.1 +- 40.3 vs paper 191.9 +- 34.3 (+13.63 %), Welch t-test p = 0.302.
Primary test, per-seed mean return over the from 100k steps: GPU 259.8 +- 46.2 vs paper 233.2 +- 41.5 (+11.43 %), Welch t-test p = 0.366.

### cartpole-swingup-sparse

| env step | reference (paper) | GPU port, 5 seeds under MPS | p |
|---|---|---|---|
| 0k | 0 +- 0 | 0 +- 0 (5) | 0.02 |
| 100k | 772 +- 14 | 595 +- 230 (5) | 0.13 |
| 200k | 813 +- 14 | 833 +- 13 (5) | 0.03 |
| 300k | 821 +- 12 | 821 +- 13 (5) | 0.96 |
| 400k | 840 +- 7 | 840 +- 7 (5) | 0.97 |
| 500k | 846 +- 1 | 749 +- 187 (5) | 0.30 |

Per-step (descriptive): 26 evaluation steps, min p = 0.004, steps with p < 0.05: 6 (expected by chance: 1.3).
Primary test, per-seed mean return over the whole run: GPU 683.8 +- 26.1 vs paper 714.0 +- 11.2 (-4.23 %), Welch t-test p = 0.060.
Primary test, per-seed mean return over the from 100k steps: GPU 817.3 +- 15.7 vs paper 823.9 +- 7.5 (-0.80 %), Welch t-test p = 0.432.

