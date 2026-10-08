# How many runs fit on one RTX 3090 - without slowing each other down

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

## A. No-slowdown capacity (interference study)

Each run is a separate process with its own horizon, taken in order from the mixed list **5, 30, 15, 2, 25, 10, 20, 5, 30, ...** (long-horizon variant: 10, 100, 50, 25, 75, ...). Every run executes the steady-state training loop (plan + env step + update). A run's *relative speed* = its agent steps/s with N runs sharing the GPU / its steps/s when running alone at the same horizon. **No-slowdown capacity** = largest N for which *every* run keeps >= 95 % (or >= 90 %) of its solo speed (and all smaller N too). Memory limit = how many processes of the same horizon mix fit in VRAM (from measured per-process VRAM; validated by the packing tests).

| domain (tasks) | mode | horizons | no-slowdown capacity (<= 5 %) | (<= 10 %) | memory limit (processes) | min relative speed at N = 2 / 3 / 4 / 6 / 8 | total steps/s at N = 1 / 2 / 4 / 8 |
|---|---|---|---|---|---|---|---|
| acrobot (acrobot-swingup) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 28 | 0.70 / 0.54 / – / – / – | 56 / 49 / – / – |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 27 | 0.71 / 0.57 / – / – / – | 46 / 43 / – / – |
| cheetah (cheetah-run) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 27 | 0.68 / 0.53 / – / – / – | 62 / 53 / – / – |
| cup (cup-catch) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 26 | 0.71 / 0.57 / – / – / – | 42 / 40 / – / – |
| dog (dog-run, dog-trot, dog-walk) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 8 | 0.63 / 0.46 / – / – / – | 24 / 19 / – / – |
| finger (finger-spin, finger-turn-hard) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 27 | 0.67 / 0.52 / – / – / – | 64 / 53 / – / – |
| fish (fish-swim) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 26 | 0.72 / 0.58 / – / – / – | 40 / 39 / – / – |
| hopper (hopper-hop) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 27 | 0.71 / 0.56 / – / – / – | 50 / 46 / – / – |
| humanoid (humanoid-run, humanoid-stand, humanoid-walk) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 19 | 0.68 / 0.52 / – / – / – | 38 / 33 / – / – |
| quadruped (quadruped-run, quadruped-walk) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 25 | 0.70 / 0.55 / – / – / – | 46 / 43 / – / – |
| reacher (reacher-easy, reacher-hard) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 28 | 0.68 / 0.53 / – / – / – | 65 / 54 / – / – |
| walker (walker-run, walker-walk) | GPU optimised + MPS | 5,30,15,2,25,10,20 | **1** | 1 | 26 | 0.71 / 0.55 / – / – / – | 48 / 44 / – / – |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | CPU MuJoCo (original) | 5,30,15,2,25,10,20 | **2** | 2 | 38 | 0.98 / 0.84 / 0.52 / – / – | 19 / 24 / 42 / – |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | GPU optimised, no MPS | 5,30,15,2,25,10,20 | **1** | 1 | 27 | 0.48 / – / – / – / – | 49 / 31 / – / – |
| walker (walker-run, walker-walk) | CPU MuJoCo (original) | 5,30,15,2,25,10,20 | **2** | 2 | 35 | 1.01 / 0.83 / 0.43 / – / – | 19 / 26 / 41 / – |
| walker (walker-run, walker-walk) | GPU optimised, no MPS | 5,30,15,2,25,10,20 | **1** | 1 | 26 | 0.51 / – / – / – / – | 49 / 33 / – / – |

## A2. Per task and horizon: runs of the SAME H in parallel - memory limit vs no-slowdown limit

Memory limit = floor(free VRAM / measured process VRAM at that H) for the optimised GPU port (horizon sweep; validated by packing tests to +-1 run). No-slowdown limit: 2 concurrent runs of the same H were measured at H = 2 and H = 30 (+ 50, 100 for walker and dog); "each" = speed of each of the 2 runs / its solo speed. Because a single run already saturates the GPU and cost grows monotonically with H, the H = 2 and H = 30 measurements bound every H in between.

| task | H=2 mem | H=5 mem | H=10 mem | H=15 mem | H=20 mem | H=25 mem | H=30 mem | 2 runs at H=2: each | 2 runs at H=30: each | no-slowdown limit (any H) |
|---|---|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | 40 | 37 | 32 | 28 | 25 | 22 | 20 | 0.76 | 0.64 | **1** |
| cartpole-swingup | 40 | 36 | 31 | 28 | 24 | 22 | 20 | 0.77 | 0.66 | **1** |
| cheetah-run | 39 | 36 | 31 | 27 | 24 | 22 | 20 | 0.74 | 0.63 | **1** |
| cup-catch | 38 | 35 | 30 | 27 | 24 | 22 | 20 | 0.79 | 0.67 | **1** |
| dog-run | 13 | 11 | 9 | 8 | 7 | 6 | 6 | 0.71 | 0.58, H=50: 0.60, H=100: 0.59 | **1** |
| finger-spin | 40 | 36 | 31 | 28 | 24 | 22 | 20 | 0.74 | 0.64 | **1** |
| fish-swim | 38 | 34 | 30 | 26 | 24 | 21 | 20 | 0.79 | 0.69 | **1** |
| hopper-hop | 38 | 35 | 30 | 27 | 24 | 22 | 20 | 0.77 | 0.66 | **1** |
| humanoid-run | 24 | 23 | 21 | 19 | 17 | 16 | 15 | 0.74 | 0.63 | **1** |
| quadruped-run | 36 | 33 | 29 | 26 | 23 | 21 | 19 | 0.77 | 0.65 | **1** |
| reacher-easy | 41 | 37 | 32 | 28 | 25 | 23 | 21 | 0.74 | 0.63 | **1** |
| walker-walk | 37 | 34 | 29 | 26 | 23 | 21 | 19 | 0.78 | 0.69, H=50: 0.67, H=100: 0.66 | **1** |

## C. MPS SM partitioning (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) - does it isolate runs?

| task | SM share per run | runs | isolation (speed / solo at same share), min | speed / unrestricted solo, mean | total steps/s |
|---|---|---|---|---|---|
| dog-run | 50 % | 2 (H = 5,30) | 0.78 | 0.68 | 20.3 |
| dog-run | 25 % | 4 (H = 5,30,15,2) | 0.66 | 0.41 | 33.7 |
| walker-walk | 50 % | 2 (H = 5,30) | 0.82 | 0.70 | 43.6 |
| walker-walk | 25 % | 4 (H = 5,30,15,2) | 0.70 | 0.48 | 76.7 |

## B. Throughput, CPU use and GPU utilisation with N concurrent seeds (same horizon H = 5)

| task | mode | N | total seed-steps/s | per-run steps/s (mean) | CPU cores per run | system CPU % | GPU util % | peak GPU MiB | OOM |
|---|---|---|---|---|---|---|---|---|---|
| cartpole-swingup | cpu | 1 | 18.3 | 18.3 | 1.00 | 7 | 27 | 1417 | 0 |
| cartpole-swingup | cpu | 2 | 40.2 | 20.1 | 1.00 | 14 | 87 | 1865 | 0 |
| cartpole-swingup | cpu | 4 | 60.8 | 15.2 | 1.00 | 26 | 94 | 2771 | 0 |
| cartpole-swingup | cpu | 8 | 64.1 | 8.0 | 1.00 | 51 | 93 | 4585 | 0 |
| cartpole-swingup | cpu | 16 | 64.1 | 4.0 | 0.98 | 99 | 93 | 8213 | 0 |
| cartpole-swingup | gpu_mps | 1 | 46.0 | 46.0 | 0.16 | 3 | 86 | 1641 | 0 |
| cartpole-swingup | gpu_mps | 2 | 70.4 | 35.2 | 0.13 | 3 | 93 | 2290 | 0 |
| cartpole-swingup | gpu_mps | 4 | 99.2 | 24.8 | 0.09 | 5 | 91 | 3587 | 0 |
| cartpole-swingup | gpu_mps | 8 | 120.4 | 15.1 | 0.07 | 8 | 91 | 6182 | 0 |
| cartpole-swingup | gpu_mps | 16 | 131.9 | 8.2 | 0.06 | 19 | 86 | 11374 | 0 |
| cartpole-swingup | gpu_opt | 1 | 49.4 | 49.4 | 0.17 | 2 | 85 | 1611 | 0 |
| cartpole-swingup | gpu_opt | 2 | 49.7 | 24.8 | 0.09 | 3 | 93 | 2265 | 0 |
| cartpole-swingup | gpu_opt | 4 | 49.3 | 12.3 | 0.05 | 4 | 92 | 3571 | 0 |
| cartpole-swingup | gpu_opt | 8 | 49.3 | 6.2 | 0.03 | 6 | 90 | 6185 | 0 |
| cartpole-swingup | gpu_opt | 16 | 49.5 | 3.1 | 0.02 | 14 | 87 | 11413 | 0 |
| cartpole-swingup | vec | 1 | 48.7 | 48.7 | 0.16 | 2 | 88 | 1615 | 0 |
| cartpole-swingup | vec | 2 | 55.6 | 55.6 | 0.09 | 2 | 89 | 1771 | 0 |
| cartpole-swingup | vec | 4 | 93.4 | 93.4 | 0.07 | 2 | 89 | 2063 | 0 |
| cartpole-swingup | vec | 8 | 113.8 | 113.8 | 0.05 | 2 | 91 | 2801 | 0 |
| cartpole-swingup | vec | 16 | 130.0 | 130.0 | 0.03 | 2 | 91 | 3643 | 0 |
| dog-run | cpu | 1 | 18.4 | 18.4 | 1.00 | 8 | 50 | 2719 | 0 |
| dog-run | cpu | 2 | 28.2 | 14.1 | 1.00 | 14 | 88 | 4481 | 0 |
| dog-run | cpu | 4 | 30.1 | 7.5 | 1.00 | 26 | 92 | 8003 | 0 |
| dog-run | cpu | 8 | 31.0 | 3.9 | 1.00 | 51 | 94 | 15049 | 0 |
| dog-run | cpu | 16 | 30.9 | 5.1 | 1.00 | 39 | 96 | 24059 | 9 |
| dog-run | cpu_pin | 1 | 16.2 | 16.2 | 1.00 | 7 | 43 | 2719 | 0 |
| dog-run | cpu_pin | 2 | 28.5 | 14.3 | 1.00 | 14 | 89 | 4481 | 0 |
| dog-run | cpu_pin | 4 | 30.2 | 7.5 | 1.00 | 26 | 93 | 8003 | 0 |
| dog-run | cpu_pin | 8 | 31.0 | 3.9 | 1.00 | 51 | 95 | 15049 | 0 |
| dog-run | cpu_pin | 16 | 16.3 | 16.3 | 1.00 | 8 | 46 | 24055 | 13 |
| dog-run | gpu | 1 | 16.6 | 16.6 | 1.00 | 7 | 60 | 2847 | 0 |
| dog-run | gpu | 2 | 23.0 | 11.5 | 1.00 | 14 | 92 | 4737 | 0 |
| dog-run | gpu | 4 | 22.9 | 5.7 | 1.00 | 26 | 91 | 8515 | 0 |
| dog-run | gpu | 8 | 23.1 | 2.9 | 1.00 | 51 | 92 | 16075 | 0 |
| dog-run | gpu | 16 | 23.1 | 3.8 | 1.00 | 40 | 93 | 24057 | 10 |
| dog-run | gpu_mps | 1 | 24.4 | 24.4 | 0.08 | 2 | 91 | 2989 | 0 |
| dog-run | gpu_mps | 2 | 33.5 | 16.8 | 0.06 | 2 | 94 | 4986 | 0 |
| dog-run | gpu_mps | 4 | 42.3 | 10.6 | 0.04 | 4 | 93 | 8979 | 0 |
| dog-run | gpu_mps | 8 | 47.4 | 5.9 | 0.03 | 6 | 90 | 16966 | 0 |
| dog-run | gpu_mps | 16 | 24.3 | 24.3 | 0.08 | 11 | 88 | 24056 | 15 |
| dog-run | gpu_opt | 1 | 25.2 | 25.2 | 0.08 | 2 | 91 | 2959 | 0 |
| dog-run | gpu_opt | 2 | 24.5 | 12.2 | 0.04 | 3 | 94 | 4961 | 0 |
| dog-run | gpu_opt | 4 | 24.4 | 6.1 | 0.02 | 3 | 92 | 8963 | 0 |
| dog-run | gpu_opt | 8 | 24.5 | 3.1 | 0.01 | 6 | 90 | 16971 | 0 |
| dog-run | gpu_opt | 16 | 24.5 | 12.2 | 0.04 | 9 | 92 | 24063 | 14 |
| dog-run | vec | 1 | 26.1 | 26.1 | 0.07 | 2 | 90 | 2999 | 0 |
| dog-run | vec | 2 | 26.1 | 26.1 | 0.03 | 2 | 90 | 4629 | 0 |
| dog-run | vec | 4 | 35.3 | 35.3 | 0.02 | 2 | 93 | 7501 | 0 |
| dog-run | vec | 8 | 41.1 | 41.1 | 0.01 | 2 | 92 | 13405 | 0 |
| dog-run | vec | 16 | 0.0 | nan | nan | 6 | 25 | 24041 | 1 |
| humanoid-run | cpu | 1 | 14.6 | 14.6 | 1.00 | 8 | 29 | 1777 | 0 |
| humanoid-run | cpu | 2 | 31.6 | 15.8 | 1.00 | 13 | 87 | 2597 | 0 |
| humanoid-run | cpu | 4 | 42.9 | 10.7 | 1.00 | 26 | 92 | 4235 | 0 |
| humanoid-run | cpu | 8 | 43.9 | 5.5 | 1.00 | 51 | 96 | 7513 | 0 |
| humanoid-run | cpu | 16 | 44.0 | 2.7 | 0.98 | 98 | 92 | 14069 | 0 |
| humanoid-run | gpu_mps | 1 | 38.3 | 38.3 | 0.12 | 2 | 89 | 2005 | 0 |
| humanoid-run | gpu_mps | 2 | 56.0 | 28.0 | 0.09 | 3 | 92 | 3018 | 0 |
| humanoid-run | gpu_mps | 4 | 72.7 | 18.2 | 0.06 | 4 | 92 | 5043 | 0 |
| humanoid-run | gpu_mps | 8 | 86.7 | 10.8 | 0.04 | 7 | 91 | 9094 | 0 |
| humanoid-run | gpu_mps | 16 | 91.9 | 5.7 | 0.03 | 15 | 88 | 17198 | 0 |
| humanoid-run | gpu_opt | 1 | 40.7 | 40.7 | 0.13 | 2 | 88 | 1975 | 0 |
| humanoid-run | gpu_opt | 2 | 40.3 | 20.2 | 0.07 | 3 | 94 | 2993 | 0 |
| humanoid-run | gpu_opt | 4 | 40.4 | 10.1 | 0.04 | 3 | 93 | 5027 | 0 |
| humanoid-run | gpu_opt | 8 | 40.1 | 5.0 | 0.02 | 5 | 93 | 9099 | 0 |
| humanoid-run | gpu_opt | 16 | 40.1 | 2.5 | 0.01 | 12 | 89 | 17237 | 0 |
| humanoid-run | vec | 1 | 40.9 | 40.9 | 0.10 | 2 | 87 | 1987 | 0 |
| humanoid-run | vec | 2 | 41.4 | 41.4 | 0.05 | 2 | 89 | 2483 | 0 |
| humanoid-run | vec | 4 | 67.0 | 67.0 | 0.04 | 2 | 92 | 3517 | 0 |
| humanoid-run | vec | 8 | 78.9 | 78.9 | 0.02 | 2 | 92 | 5631 | 0 |
| humanoid-run | vec | 16 | 88.9 | 88.9 | 0.01 | 2 | 93 | 9443 | 0 |
| walker-walk | cpu | 1 | 18.2 | 18.2 | 1.00 | 8 | 27 | 1457 | 0 |
| walker-walk | cpu | 2 | 40.2 | 20.1 | 1.00 | 14 | 91 | 1957 | 0 |
| walker-walk | cpu | 4 | 61.0 | 15.2 | 1.00 | 26 | 95 | 2955 | 0 |
| walker-walk | cpu | 8 | 63.4 | 7.9 | 1.00 | 51 | 94 | 4953 | 0 |
| walker-walk | cpu | 16 | 63.7 | 4.0 | 0.98 | 99 | 94 | 8949 | 0 |
| walker-walk | cpu_pin | 1 | 18.4 | 18.4 | 1.00 | 7 | 27 | 1457 | 0 |
| walker-walk | cpu_pin | 2 | 40.2 | 20.1 | 1.00 | 14 | 88 | 1957 | 0 |
| walker-walk | cpu_pin | 4 | 60.9 | 15.2 | 1.00 | 26 | 93 | 2955 | 0 |
| walker-walk | cpu_pin | 8 | 63.3 | 7.9 | 1.00 | 51 | 93 | 4953 | 0 |
| walker-walk | cpu_pin | 16 | 63.5 | 4.0 | 0.98 | 98 | 92 | 8949 | 0 |
| walker-walk | gpu | 1 | 19.2 | 19.2 | 1.00 | 7 | 35 | 1571 | 0 |
| walker-walk | gpu | 2 | 38.6 | 19.3 | 1.00 | 13 | 91 | 2185 | 0 |
| walker-walk | gpu | 4 | 46.0 | 11.5 | 1.00 | 26 | 91 | 3411 | 0 |
| walker-walk | gpu | 8 | 46.4 | 5.8 | 1.00 | 51 | 90 | 5865 | 0 |
| walker-walk | gpu | 16 | 46.7 | 2.9 | 0.99 | 99 | 88 | 10773 | 0 |
| walker-walk | gpu_mps | 1 | 48.2 | 48.2 | 0.14 | 2 | 87 | 1683 | 0 |
| walker-walk | gpu_mps | 2 | 74.1 | 37.1 | 0.12 | 3 | 94 | 2374 | 0 |
| walker-walk | gpu_mps | 4 | 103.3 | 25.8 | 0.08 | 4 | 91 | 3757 | 0 |
| walker-walk | gpu_mps | 8 | 128.5 | 16.1 | 0.06 | 8 | 90 | 6518 | 0 |
| walker-walk | gpu_mps | 16 | 142.1 | 8.9 | 0.05 | 17 | 88 | 12046 | 0 |
| walker-walk | gpu_opt | 1 | 51.2 | 51.2 | 0.15 | 2 | 85 | 1651 | 0 |
| walker-walk | gpu_opt | 2 | 52.0 | 26.0 | 0.08 | 3 | 91 | 2345 | 0 |
| walker-walk | gpu_opt | 4 | 52.4 | 13.1 | 0.05 | 4 | 92 | 3731 | 0 |
| walker-walk | gpu_opt | 8 | 53.4 | 6.7 | 0.03 | 6 | 92 | 6505 | 0 |
| walker-walk | gpu_opt | 16 | 53.8 | 3.4 | 0.02 | 14 | 90 | 12053 | 0 |
| walker-walk | vec | 1 | 50.7 | 50.7 | 0.14 | 2 | 87 | 1663 | 0 |
| walker-walk | vec | 2 | 54.3 | 54.3 | 0.07 | 2 | 90 | 1825 | 0 |
| walker-walk | vec | 4 | 90.6 | 90.6 | 0.06 | 2 | 90 | 2199 | 0 |
| walker-walk | vec | 8 | 111.0 | 111.0 | 0.04 | 2 | 93 | 3073 | 0 |
| walker-walk | vec | 16 | 127.6 | 127.6 | 0.02 | 2 | 91 | 4125 | 0 |
