# Total run time per task (state observations, H = 5, default training budgets)

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

CPU MuJoCo + GPU learning = original tdmpc code (physics on CPU, agent on GPU). GPU faithful = mujoco_warp env + original agent. GPU optimised = mujoco_warp env + CUDA-graph agent + host pacing (same maths). Single-run times are extrapolated from the measured steady-state cost per agent step (incl. seeding phase and 10 evaluation episodes every 20k env steps). "5 seeds parallel" divides 5x the single-run time by the measured total-throughput factor at 5 concurrent seeds (interpolated between the measured N = 4 and N = 8; CPU original: 3.37x small tasks / 2.96x humanoid / 1.65x dog; GPU + MPS: 2.27x / 1.99x / 1.79x; GPU vectorised: 1.96x / 1.71x / 1.41x). Tasks other than cartpole/walker/humanoid/dog use the cartpole/walker factors (same network sizes). All-on-CPU (physics and learning on CPU) was not measured.

> **Scope: humanoid and dog tasks.** humanoid-run/-stand/-walk and dog-run/-trot/-walk are ported to the GPU,
> and their environments, speed and memory were measured in short benchmarks (those numbers appear in the tables).
> They were **not** trained, and they are excluded from the learning-curve tests and the remaining baseline work
> (MPS stability tests, re-timings, further optimisation). Their training budgets (3M / 5M environment steps, about
> 14-33 h per seed even on the optimised GPU port, 1.4-3.9 days per 5-seed batch in parallel) make them too expensive to pursue for
> this project. The optimised baseline is therefore established and verified on the other 15 tasks.

| task | env steps | 1 seed: CPU MuJoCo + GPU learning | 1 seed: GPU faithful | 1 seed: GPU optimised | 5 seeds back-to-back: CPU+GPU learning | 5 seeds back-to-back: GPU optimised | 5 seeds parallel: CPU+GPU learning | 5 seeds parallel: GPU optimised + MPS | 5 seeds parallel: GPU vectorised (1 process) |
|---|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | 0.5M | 2.8 h | 2.2 h | 0.7 h | 13.9 h | 3.7 h | 4.1 h | 1.6 h | 1.9 h |
| cartpole-swingup | 0.5M | 1.4 h | 1.1 h | 0.5 h | 7.1 h | 2.3 h | 2.1 h | 1.0 h | 1.2 h |
| cartpole-swingup-sparse | 0.5M | 1.4 h | 1.1 h | 0.5 h | 7.0 h | 2.3 h | 2.1 h | 1.0 h | 1.2 h |
| cheetah-run | 0.5M | 2.7 h | 2.0 h | 0.7 h | 13.3 h | 3.4 h | 3.9 h | 1.5 h | 1.7 h |
| cup-catch | 0.5M | 2.8 h | 2.2 h | 1.0 h | 13.9 h | 5.2 h | 4.1 h | 2.3 h | 2.7 h |
| dog-run | 5M | 65 h (2.7 d) | 49 h (2.0 d) | 33.4 h | 324 h (13.5 d) | 167 h (6.9 d) | 196 h (8.2 d) | 93 h (3.9 d) | 118 h (4.9 d) |
| dog-trot | 5M | 63 h (2.6 d) | 47.0 h | 33.2 h | 315 h (13.1 d) | 166 h (6.9 d) | 191 h (8.0 d) | 93 h (3.9 d) | 118 h (4.9 d) |
| dog-walk | 5M | 66 h (2.7 d) | 50 h (2.1 d) | 33.5 h | 330 h (13.7 d) | 168 h (7.0 d) | 200 h (8.3 d) | 94 h (3.9 d) | 119 h (5.0 d) |
| finger-spin | 0.5M | 5.3 h | 4.4 h | 1.2 h | 26.5 h | 6.2 h | 7.9 h | 2.7 h | 3.2 h |
| finger-turn-hard | 0.5M | 5.7 h | 4.4 h | 1.3 h | 28.3 h | 6.4 h | 8.4 h | 2.8 h | 3.3 h |
| fish-swim | 0.5M | 2.2 h | 2.2 h | 1.1 h | 11.2 h | 5.3 h | 3.3 h | 2.3 h | 2.7 h |
| hopper-hop | 0.5M | 2.5 h | 2.1 h | 0.8 h | 12.6 h | 4.1 h | 3.7 h | 1.8 h | 2.1 h |
| humanoid-run | 3M | 43.7 h | 33.7 h | 13.7 h | 218 h (9.1 d) | 69 h (2.9 d) | 74 h (3.1 d) | 34.5 h | 40.2 h |
| humanoid-stand | 3M | 43.1 h | 32.6 h | 13.6 h | 215 h (9.0 d) | 68 h (2.8 d) | 73 h (3.0 d) | 34.1 h | 39.7 h |
| humanoid-walk | 3M | 42.4 h | 34.0 h | 13.6 h | 212 h (8.8 d) | 68 h (2.8 d) | 72 h (3.0 d) | 34.2 h | 39.8 h |
| quadruped-run | 0.5M | 2.7 h | 2.3 h | 0.9 h | 13.5 h | 4.6 h | 4.0 h | 2.0 h | 2.3 h |
| quadruped-walk | 0.5M | 2.8 h | 2.2 h | 0.9 h | 14.2 h | 4.6 h | 4.2 h | 2.0 h | 2.3 h |
| reacher-easy | 0.5M | 2.6 h | 2.1 h | 0.6 h | 12.8 h | 3.0 h | 3.8 h | 1.3 h | 1.6 h |
| reacher-hard | 0.5M | 2.7 h | 2.1 h | 0.6 h | 13.5 h | 3.1 h | 4.0 h | 1.4 h | 1.6 h |
| walker-run | 0.5M | 5.4 h | 4.4 h | 1.7 h | 27.2 h | 8.7 h | 8.1 h | 3.8 h | 4.4 h |
| walker-walk | 0.5M | 5.6 h | 4.4 h | 1.7 h | 28.1 h | 8.7 h | 8.3 h | 3.8 h | 4.5 h |
| **all 21 tasks** | | **372 h (15.5 d)** | **285 h (11.9 d)** | **155 h (6.5 d)** | **1858 h (77.4 d)** | **776 h (32.4 d)** | **878 h (36.6 d)** | **414 h (17.3 d)** | **511 h (21.3 d)** |
