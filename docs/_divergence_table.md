*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

| task | variant | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|---|
| dog-run | gpu_default | 3.3e-06 | 6.5e-04 | 6.0e-02 | 2.0e-01 | 1.1e+00 |
| dog-run | gpu_tol1e-8 | 1.0e-06 | 6.5e-04 | 6.1e-02 | 1.4e-01 | 8.4e-01 |
| dog-run | cpu_f32round | 8.3e-08 | 1.9e-07 | 4.4e-06 | 1.3e-05 | 2.7e-03 |
| dog-run | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.5e-13 |
| walker-walk | gpu_default | 5.4e-07 | 1.1e-06 | 1.6e-05 | 4.6e-05 | 3.4e-01 |
| walker-walk | gpu_tol1e-8 | 5.4e-07 | 7.5e-07 | 1.6e-05 | 5.2e-05 | 3.4e-01 |
| walker-walk | cpu_f32round | 1.4e-08 | 3.0e-07 | 9.8e-06 | 1.2e-05 | 1.9e+00 |
| walker-walk | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 2.7e-06 |
| quadruped-run | gpu_default | 1.3e-05 | 1.2e-05 | 3.7e-05 | 9.3e-04 | 2.0e-01 |
| quadruped-run | gpu_tol1e-8 | 1.3e-05 | 1.1e-05 | 7.0e-05 | 8.8e-04 | 4.9e-02 |
| quadruped-run | cpu_f32round | 4.0e-06 | 5.1e-06 | 1.5e-05 | 2.1e-05 | 3.6e-02 |
| quadruped-run | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| humanoid-walk | gpu_default | 4.3e-07 | 3.8e-05 | 1.4e-05 | 7.6e-02 | 4.3e-01 |
| humanoid-walk | gpu_tol1e-8 | 4.3e-07 | 4.1e-05 | 1.5e-05 | 7.6e-02 | 5.8e-01 |
| humanoid-walk | cpu_f32round | 2.1e-08 | 7.1e-06 | 2.9e-06 | 2.5e-04 | 2.8e-01 |
| humanoid-walk | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| hopper-hop | gpu_default | 1.8e-07 | 1.0e-06 | 2.9e-06 | 3.0e-05 | 1.2e+00 |
| hopper-hop | gpu_tol1e-8 | 1.8e-07 | 1.0e-06 | 2.9e-06 | 2.9e-05 | 1.2e+00 |
| hopper-hop | cpu_f32round | 2.8e-08 | 2.8e-07 | 4.3e-06 | 1.7e-05 | 6.3e-02 |
| hopper-hop | cpu_tol1e-6 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |