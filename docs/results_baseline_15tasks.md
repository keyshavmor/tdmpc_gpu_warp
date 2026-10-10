# Baseline results: 15 tasks x 5 seeds (GPU port, 5 seeds in parallel under MPS)

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Setup: optimised GPU port (CUDA-graph agent + host pacing, `src/launch_parallel.py --mps`), original TD-MPC hyperparameters (H = 5), full training budget of 500k environment steps per run, 5 seeds (1-5) of a task in parallel under CUDA MPS, tasks one after another on one RTX 3090, checkpoints every 15 min, W&B logging (`bench/mps_batches.py`, label `baseline_mps`).

**Final return** = evaluation return at the end of training (mean over 10 evaluation episodes at 500k env steps). **Run time** = wall-clock time of each run from start to the end of training (5 runs share the GPU). Mean and standard deviation (sample sd, n - 1) across the 5 seeds. The paper reference is the original TD-MPC (5 seeds) at 500k env steps. Learning curves and statistical comparison with the paper: [results_mps_batches.md](results_mps_batches.md).

| task | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std. dev. | paper (mean +- sd) | run time mean (h) | run time std. dev. (h) | batch crashes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | 343.4 | 457.0 | 358.3 | 359.4 | 304.5 | **364.5** | 56.3 | 341.8 +- 120.9 | 1.72 | 0.01 | 0 |
| cartpole-swingup | 879.3 | 881.6 | 873.8 | 879.9 | 880.6 | **879.0** | 3.0 | 879.0 +- 6.7 | 0.93 | 0.01 | 0 |
| cartpole-swingup-sparse | 844.7 | 847.5 | 376.2 | 845.8 | 832.1 | **749.3** | 208.6 | 846.4 +- 1.5 | 0.92 | 0.01 | 0 |
| cheetah-run | 827.3 | 874.8 | 874.3 | 841.7 | 811.7 | **846.0** | 28.2 | 813.8 +- 34.8 | 1.61 | 0.01 | 0 |
| cup-catch | 983.0 | 969.9 | 968.9 | 974.6 | 988.7 | **977.0** | 8.6 | 972.0 +- 11.1 | 2.12 | 0.01 | 0 |
| finger-spin | 985.0 | 983.6 | 976.8 | 925.9 | 985.5 | **971.4** | 25.7 | 980.6 +- 7.4 | 3.26 | 0.01 | 0 |
| finger-turn-hard | running | running | running | running | running | - | - | 573.6 +- 198.8 | - | - | - |
| fish-swim | pending | pending | pending | pending | pending | - | - | 745.4 +- 17.5 | - | - | - |
| hopper-hop | pending | pending | pending | pending | pending | - | - | 324.6 +- 170.7 | - | - | - |
| quadruped-run | pending | pending | pending | pending | pending | - | - | 796.6 +- 74.2 | - | - | - |
| quadruped-walk | pending | pending | pending | pending | pending | - | - | 931.0 +- 26.6 | - | - | - |
| reacher-easy | 984.6 | 983.5 | 981.4 | 983.5 | 985.0 | **983.6** | 1.4 | 976.4 +- 14.8 | 1.60 | 0.01 | 0 |
| reacher-hard | pending | pending | pending | pending | pending | - | - | 966.6 +- 11.7 | - | - | - |
| walker-run | pending | pending | pending | pending | pending | - | - | 819.8 +- 7.5 | - | - | - |
| walker-walk | 973.1 | 973.6 | 986.3 | 976.1 | 980.0 | **977.8** | 5.5 | 976.8 +- 3.1 | 3.83 | 0.02 | 0 |

8 of 15 tasks complete.

