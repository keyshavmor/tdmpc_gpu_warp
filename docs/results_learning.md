# Learning-mechanism equivalence (eval return vs env steps)

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Reference = 5 seeds of the original TD-MPC code from the paper (`tdmpc/results/<task>.csv`). Each cell: mean ± sd over seeds (n). p = Welch test (normal approx.) of the variant vs the reference (or vs `vec_fp32` for the A/B rows) at that step; p > 0.05 = no detectable difference.

## cartpole-swingup

| env step | reference (paper) | CPU MuJoCo, original code (this machine) | GPU port, original agent | GPU port, vectorised CUDA-graph agent (exact) | + batched update + sensor skipping (exact) |
|---|---|---|---|---|---|
| 0.0k | 94 ± 8 (5) | 127 ± 0 (1) | 119 ± 0 (1) | 124 ± 17 (5), p=0.00 | 124 ± 17 (5), p=0.00 |
| 20.0k | 98 ± 15 (5) | 136 ± 0 (1) | 127 ± 0 (1) | 110 ± 7 (5), p=0.14 | 110 ± 7 (5), p=0.14 |
| 40.0k | 574 ± 110 (5) | 653 ± 0 (1) | 472 ± 0 (1) | 553 ± 73 (5), p=0.76 | 630 ± 139 (5), p=0.53 |
| 60.0k | 825 ± 28 (5) | 849 ± 0 (1) | 847 ± 0 (1) | 821 ± 15 (5), p=0.83 | 825 ± 33 (5), p=0.99 |
| 80.0k | 855 ± 4 (5) | 847 ± 0 (1) | 857 ± 0 (1) | 832 ± 19 (5), p=0.02 | 844 ± 8 (5), p=0.01 |
| 100.0k | 859 ± 6 (5) | 867 ± 0 (1) | – | – | – |

A/B vs `vec_fp32` (same code path, exact maths), p-values per eval step:

| variant | 0.0k | 20.0k | 40.0k | 60.0k | 80.0k | 100.0k | mean over steps (variant - vec_fp32) |
|---|---|---|---|---|---|---|---|
| + batched update + sensor skipping (exact) | 1.00 | 1.00 | 0.33 | 0.84 | 0.24 | – | +18 |

## walker-walk

| env step | reference (paper) | CPU MuJoCo, original code (this machine) | GPU port, vectorised CUDA-graph agent (exact) | + batched update + sensor skipping (exact) | bf16 autocast | fp16 autocast |
|---|---|---|---|---|---|---|
| 0.0k | 35 ± 1 (5) | 33 ± 0 (1) | 36 ± 2 (5), p=0.39 | 36 ± 2 (5), p=0.29 | 36 ± 2 (3), p=0.68 | 36 ± 2 (3), p=0.68 |
| 20.0k | 447 ± 212 (5) | 758 ± 0 (1) | 554 ± 222 (5), p=0.49 | 463 ± 115 (5), p=0.89 | 489 ± 155 (3), p=0.78 | 564 ± 96 (3), p=0.35 |
| 40.0k | 954 ± 6 (5) | 957 ± 0 (1) | 910 ± 67 (5), p=0.20 | 936 ± 30 (5), p=0.27 | 944 ± 8 (3), p=0.14 | 936 ± 28 (3), p=0.37 |
| 60.0k | 964 ± 9 (5) | 955 ± 0 (1) | – | – | – | – |
| 80.0k | 959 ± 11 (5) | 953 ± 0 (1) | – | – | – | – |
| 100.0k | 963 ± 5 (5) | 976 ± 0 (1) | – | – | – | – |

A/B vs `vec_fp32` (same code path, exact maths), p-values per eval step:

| variant | 0.0k | 20.0k | 40.0k | 60.0k | 80.0k | 100.0k | mean over steps (variant - vec_fp32) |
|---|---|---|---|---|---|---|---|
| + batched update + sensor skipping (exact) | 0.86 | 0.47 | 0.48 | – | – | – | -22 |
| bf16 autocast | 0.92 | 0.68 | 0.33 | – | – | – | -11 |
| fp16 autocast | 0.92 | 0.94 | 0.51 | – | – | – | +12 |
