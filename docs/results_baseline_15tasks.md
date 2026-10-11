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
| finger-turn-hard | 396.2 | 668.8 | 867.9 | 579.2 | 598.7 | **622.2** | 170.3 | 573.6 +- 198.8 | 3.26 | 0.01 | 0 |
| fish-swim | 788.8 | 740.3 | 772.7 | 742.2 | 660.7 | **740.9** | 49.3 | 745.4 +- 17.5 | 2.10 | 0.01 | 0 |
| hopper-hop | 71.7 | 50.3 | 4.9 | 363.6 | 142.0 | **126.5** | 141.5 | 324.6 +- 170.7 | 1.81 | 0.01 | 0 |
| quadruped-run | 787.9 | 863.7 | 896.4 | 804.4 | 833.0 | **837.1** | 44.0 | 796.6 +- 74.2 | 1.94 | 0.01 | 0 |
| quadruped-walk | 933.4 | 950.3 | 957.5 | 952.4 | 959.7 | **950.7** | 10.3 | 931.0 +- 26.6 | 1.95 | 0.01 | 0 |
| reacher-easy | 984.6 | 983.5 | 981.4 | 983.5 | 985.0 | **983.6** | 1.4 | 976.4 +- 14.8 | 1.60 | 0.01 | 0 |
| reacher-hard | running | running | running | running | running | - | - | 966.6 +- 11.7 | - | - | - |
| walker-run | pending | pending | pending | pending | pending | - | - | 819.8 +- 7.5 | - | - | - |
| walker-walk | 973.1 | 973.6 | 986.3 | 976.1 | 980.0 | **977.8** | 5.5 | 976.8 +- 3.1 | 3.83 | 0.02 | 0 |

13 of 15 tasks complete.

**Notes on single low final evaluations:**

- cartpole-swingup-sparse, seed 3: final evaluation 376.2 vs 752.1 averaged over its last 5 evaluations - a one-off low final evaluation (mean of 10 episodes), not a failed run; see the learning curve in results_mps_batches.md.
- hopper-hop, seed 2: final evaluation 50.3 vs 71.1 averaged over its last 5 evaluations - a one-off low final evaluation (mean of 10 episodes), not a failed run; see the learning curve in results_mps_batches.md.

## Robust end-of-training value: mean of the last 5 evaluations (420k-500k env steps)

Same runs; per seed the mean evaluation return over its last 5 evaluations (each the mean of 10 episodes), which is less sensitive to a single noisy final evaluation. Mean and sample std. dev. across the 5 seeds.

| task | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std. dev. |
|---|---|---|---|---|---|---|---|
| acrobot-swingup | 392.1 | 484.6 | 344.4 | 344.3 | 303.4 | **373.8** | 69.5 |
| cartpole-swingup | 877.9 | 877.0 | 874.6 | 879.5 | 879.7 | **877.7** | 2.1 |
| cartpole-swingup-sparse | 844.9 | 844.3 | 752.1 | 830.1 | 836.8 | **821.6** | 39.4 |
| cheetah-run | 841.5 | 866.6 | 846.7 | 854.3 | 781.6 | **838.2** | 33.0 |
| cup-catch | 978.4 | 975.8 | 978.6 | 974.4 | 980.4 | **977.5** | 2.4 |
| finger-spin | 967.9 | 981.0 | 978.8 | 941.3 | 979.7 | **969.7** | 16.7 |
| finger-turn-hard | 448.4 | 529.7 | 697.0 | 463.4 | 567.2 | **541.1** | 99.7 |
| fish-swim | 761.5 | 737.2 | 771.8 | 707.0 | 701.1 | **735.7** | 31.6 |
| hopper-hop | 65.7 | 71.1 | 3.6 | 336.8 | 90.9 | **113.6** | 129.0 |
| quadruped-run | 791.7 | 848.7 | 883.9 | 791.0 | 804.7 | **824.0** | 40.9 |
| quadruped-walk | 928.8 | 927.0 | 946.0 | 934.1 | 948.4 | **936.8** | 9.8 |
| reacher-easy | 981.5 | 965.1 | 984.4 | 930.4 | 984.9 | **969.3** | 23.2 |
| walker-walk | 972.2 | 978.7 | 978.8 | 979.5 | 979.3 | **977.7** | 3.1 |

