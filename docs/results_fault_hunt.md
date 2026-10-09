# MPS fault experiments

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Each variant runs rounds of up to 20-30 min under a private CUDA MPS daemon (bench/mps_fault_hunt.py); a round ends at the first fatal fault (recorded by the MPS server and the kernel as Xid 31). Exposure = summed lifetime of all processes on the GPU. Rate = faults per process-hour with an exact 95 % Poisson confidence interval. Raw data: `results/fault_hunt/<variant>.json`; analysis: docs/08_known_issues.md.

| variant | setup | wall time (min) | process starts | exposure (process-h) | faults | rate per process-h [95 % CI] | median time to fault |
|---|---|---|---|---|---|---|---|
| baseline_vec | MPS only: 6 vectorised training processes (3 seeds each) | 60 | 18 | 5.8 | 0 | 0.00 [0.00, 0.64] | - |
| churn_single | MPS only: 6 single-seed training processes + 1 new MPS process every 30 s (SIGKILL after 60 s) | 60 | 124 | 7.7 | 0 | 0.00 [0.00, 0.48] | - |
| env_eager_vec | MPS only: 6 vectorised training processes, eager physics | 60 | 18 | 5.8 | 0 | 0.00 [0.00, 0.64] | - |
| env_only_vec | MPS only: 6 physics-only stress processes | 60 | 18 | 5.8 | 0 | 0.00 [0.00, 0.64] | - |
| routed_graceful | MPS only: 6 training + 2 extra physics processes through MPS, clean exit every 120 s | 45 | 57 | 5.6 | 0 | 0.00 [0.00, 0.66] | - |
| routed_kill | MPS only: 6 training + 2 extra physics processes through MPS, restarted every 120 s by SIGKILL | 45 | 57 | 5.5 | 0 | 0.00 [0.00, 0.67] | - |
| routed_vec_kill | MPS only: 6 training + 2 extra vectorised training processes through MPS, SIGKILL every 120 s | 45 | 57 | 5.5 | 0 | 0.00 [0.00, 0.67] | - |
| mixed_eager | MIXED: 6 MPS training + 2 non-MPS physics processes with eager (non-graph) physics | 45 | 24 | 5.6 | 0 | 0.00 [0.00, 0.66] | - |
| mixed_graceful | MIXED: 6 MPS training + 2 non-MPS physics processes (clean exit every 120 s) | 45 | 99 | 4.7 | 9 | 1.93 [0.88, 3.66] | 2.3 min |
| mixed_kill | MIXED: 6 MPS training + 2 non-MPS physics processes (SIGKILL every 120 s) | 45 | 106 | 4.6 | 10 | 2.17 [1.04, 4.00] | 2.4 min |
| mixed_nocond | MIXED: 6 MPS training + 2 non-MPS physics processes without CUDA-graph conditional nodes | 45 | 56 | 5.1 | 6 | 1.17 [0.43, 2.55] | 2.8 min |
| mixed_ping | MIXED: 6 MPS training + 2 non-MPS minimal torch processes (20 s each, clean exit) | 45 | 236 | 5.5 | 0 | 0.00 [0.00, 0.67] | - |
| mixed_vec | MIXED: 6 MPS training + 2 non-MPS vectorised training processes (restarted every 120 s) | 60 | 135 | 6.2 | 13 | 2.09 [1.11, 3.58] | 3.0 min |
| mixed_vulkan | MIXED: 6 MPS training + 1 process creating Vulkan instances every second | 45 | 132 | 5.0 | 0 | 0.00 [0.00, 0.74] | - |

All MPS-only variants together: 0 faults in 41.7 process-hours, rate 0.000 [0.000, 0.088] per process-hour.

