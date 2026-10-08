# Memory: OOM horizon, seeds per GPU, mixed horizons (RTX 3090, 24 GiB)

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Free VRAM for runs: 23613 MiB (24576 MiB minus 963 MiB used by the desktop).

## Single run: OOM horizon

`H limit` = largest horizon the algorithm can run at all: H <= L - 1 (L = episode length in agent steps; each episode then still has at least one valid start index, and PER batches are drawn with replacement until the buffer is full). Fit = VRAM(H) = a + b*H from H = 2..30; boundaries confirmed with real runs (`fits <=` / `OOM >=`). The probes were capped at `largest probed H` = L - ceil(batch / (seed_steps/L + 1)); where the fit predicts OOM far beyond L - 1, no OOM is possible at any valid horizon. (The cap came from an earlier, incorrect assumption that the first batch is drawn without replacement; see docs/06, section 6.4.)

| domain (tasks) | mode | a (MiB) | b (MiB per H) | H limit (L-1) | largest probed H | predicted OOM H | measured |
|---|---|---|---|---|---|---|---|
| acrobot (acrobot-swingup) | CPU baseline | 386 | 16.5 | 249 | 225 | 1411 | fits at 225; OOM predicted only at 1411 > 249 - no OOM possible |
| acrobot (acrobot-swingup) | GPU faithful | 489 | 16.3 | 249 | 225 | 1415 | fits at 225; OOM predicted only at 1415 > 249 - no OOM possible |
| acrobot (acrobot-swingup) | GPU optimised | 537 | 19.8 | 249 | 225 | 1163 | fits at 225; OOM predicted only at 1163 > 249 - no OOM possible |
| acrobot (acrobot-swingup) | GPU + ckpt_unroll | 559 | 4.2 | 249 | 225 | 5436 | fits at 225; OOM predicted only at 5436 > 249 - no OOM possible |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | CPU baseline | 367 | 16.4 | 124 | 112 | 1420 | fits at 112; OOM predicted only at 1420 > 124 - no OOM possible |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | GPU faithful | 483 | 16.4 | 124 | 112 | 1413 | fits at 112; OOM predicted only at 1413 > 124 - no OOM possible |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | GPU optimised | 547 | 19.9 | 124 | 112 | 1157 | fits at 112; OOM predicted only at 1157 > 124 - no OOM possible |
| cartpole (cartpole-swingup, cartpole-swingup-sparse) | GPU + ckpt_unroll | 569 | 4.2 | 124 | 112 | 5543 | fits at 112; OOM predicted only at 5543 > 124 - no OOM possible |
| cheetah (cheetah-run) | CPU baseline | 386 | 16.5 | 249 | 225 | 1406 | fits at 225; OOM predicted only at 1406 > 249 - no OOM possible |
| cheetah (cheetah-run) | GPU faithful | 489 | 16.5 | 249 | 225 | 1402 | fits at 225; OOM predicted only at 1402 > 249 - no OOM possible |
| cheetah (cheetah-run) | GPU optimised | 554 | 20.0 | 249 | 225 | 1155 | fits at 225; OOM predicted only at 1155 > 249 - no OOM possible |
| cheetah (cheetah-run) | GPU + ckpt_unroll | 573 | 4.2 | 249 | 225 | 5446 | fits at 225; OOM predicted only at 5446 > 249 - no OOM possible |
| cup (cup-catch) | CPU baseline | 388 | 16.4 | 249 | 225 | 1415 | fits at 225; OOM predicted only at 1415 > 249 - no OOM possible |
| cup (cup-catch) | GPU faithful | 509 | 16.4 | 249 | 225 | 1407 | fits at 225; OOM predicted only at 1407 > 249 - no OOM possible |
| cup (cup-catch) | GPU optimised | 574 | 19.9 | 249 | 225 | 1157 | fits at 225; OOM predicted only at 1157 > 249 - no OOM possible |
| cup (cup-catch) | GPU + ckpt_unroll | 594 | 4.3 | 249 | 225 | 5322 | fits at 225; OOM predicted only at 5322 > 249 - no OOM possible |
| dog (dog-run, dog-trot, dog-walk) | CPU baseline | 1409 | 70.9 | 499 | 313 | 313 | fits <= 305, OOM >= 309 |
| dog (dog-run, dog-trot, dog-walk) | GPU faithful | 1539 | 70.7 | 499 | 313 | 312 | fits <= 301, OOM >= 305 |
| dog (dog-run, dog-trot, dog-walk) | GPU optimised | 1628 | 74.7 | 499 | 313 | 294 | fits <= 289, OOM >= 292 |
| dog (dog-run, dog-trot, dog-walk) | GPU + ckpt_unroll | 1711 | 7.8 | 499 | 313 | 2805 | fits at 313; OOM predicted only at 2805 > 499 - no OOM possible |
| finger (finger-spin, finger-turn-hard) | CPU baseline | 388 | 16.4 | 499 | 453 | 1415 | fits at 453; OOM predicted only at 1415 > 499 - no OOM possible |
| finger (finger-spin, finger-turn-hard) | GPU faithful | 483 | 16.4 | 499 | 453 | 1409 | fits at 453; OOM predicted only at 1409 > 499 - no OOM possible |
| finger (finger-spin, finger-turn-hard) | GPU optimised | 547 | 19.9 | 499 | 453 | 1159 | fits at 453; OOM predicted only at 1159 > 499 - no OOM possible |
| finger (finger-spin, finger-turn-hard) | GPU + ckpt_unroll | 569 | 4.2 | 499 | 453 | 5447 | fits at 453; OOM predicted only at 5447 > 499 - no OOM possible |
| fish (fish-swim) | CPU baseline | 399 | 16.5 | 249 | 225 | 1406 | fits at 225; OOM predicted only at 1406 > 249 - no OOM possible |
| fish (fish-swim) | GPU faithful | 513 | 16.4 | 249 | 225 | 1405 | fits at 225; OOM predicted only at 1405 > 249 - no OOM possible |
| fish (fish-swim) | GPU optimised | 579 | 19.9 | 249 | 225 | 1157 | fits at 225; OOM predicted only at 1157 > 249 - no OOM possible |
| fish (fish-swim) | GPU + ckpt_unroll | 598 | 4.3 | 249 | 225 | 5321 | fits at 225; OOM predicted only at 5321 > 249 - no OOM possible |
| hopper (hopper-hop) | CPU baseline | 386 | 16.5 | 249 | 225 | 1406 | fits at 225; OOM predicted only at 1406 > 249 - no OOM possible |
| hopper (hopper-hop) | GPU faithful | 498 | 16.5 | 249 | 225 | 1400 | fits at 225; OOM predicted only at 1400 > 249 - no OOM possible |
| hopper (hopper-hop) | GPU optimised | 567 | 19.8 | 249 | 225 | 1162 | fits at 225; OOM predicted only at 1162 > 249 - no OOM possible |
| hopper (hopper-hop) | GPU + ckpt_unroll | 586 | 4.3 | 249 | 225 | 5324 | fits at 225; OOM predicted only at 5324 > 249 - no OOM possible |
| humanoid (humanoid-run, humanoid-stand, humanoid-walk) | CPU baseline | 730 | 16.9 | 499 | 453 | 1352 | fits at 453; OOM predicted only at 1352 > 499 - no OOM possible |
| humanoid (humanoid-run, humanoid-stand, humanoid-walk) | GPU faithful | 838 | 16.9 | 499 | 453 | 1349 | fits at 453; OOM predicted only at 1349 > 499 - no OOM possible |
| humanoid (humanoid-run, humanoid-stand, humanoid-walk) | GPU optimised | 910 | 21.1 | 499 | 453 | 1078 | fits at 453; OOM predicted only at 1078 > 499 - no OOM possible |
| humanoid (humanoid-run, humanoid-stand, humanoid-walk) | GPU + ckpt_unroll | 926 | 5.5 | 499 | 453 | 4106 | fits at 453; OOM predicted only at 4106 > 499 - no OOM possible |
| quadruped (quadruped-run, quadruped-walk) | CPU baseline | 426 | 16.5 | 249 | 225 | 1405 | fits at 225; OOM predicted only at 1405 > 249 - no OOM possible |
| quadruped (quadruped-run, quadruped-walk) | GPU faithful | 542 | 16.5 | 249 | 225 | 1403 | fits at 225; OOM predicted only at 1403 > 249 - no OOM possible |
| quadruped (quadruped-run, quadruped-walk) | GPU optimised | 609 | 20.1 | 249 | 225 | 1144 | fits at 225; OOM predicted only at 1144 > 249 - no OOM possible |
| quadruped (quadruped-run, quadruped-walk) | GPU + ckpt_unroll | 628 | 4.4 | 249 | 225 | 5200 | fits at 225; OOM predicted only at 5200 > 249 - no OOM possible |
| reacher (reacher-easy, reacher-hard) | CPU baseline | 388 | 16.4 | 249 | 225 | 1415 | fits at 225; OOM predicted only at 1415 > 249 - no OOM possible |
| reacher (reacher-easy, reacher-hard) | GPU faithful | 483 | 16.4 | 249 | 225 | 1408 | fits at 225; OOM predicted only at 1408 > 249 - no OOM possible |
| reacher (reacher-easy, reacher-hard) | GPU optimised | 531 | 19.7 | 249 | 225 | 1171 | fits at 225; OOM predicted only at 1171 > 249 - no OOM possible |
| reacher (reacher-easy, reacher-hard) | GPU + ckpt_unroll | 551 | 4.3 | 249 | 225 | 5345 | fits at 225; OOM predicted only at 5345 > 249 - no OOM possible |
| walker (walker-run, walker-walk) | CPU baseline | 412 | 16.5 | 499 | 453 | 1409 | fits at 453; OOM predicted only at 1409 > 499 - no OOM possible |
| walker (walker-run, walker-walk) | GPU faithful | 526 | 16.5 | 499 | 453 | 1402 | fits at 453; OOM predicted only at 1402 > 499 - no OOM possible |
| walker (walker-run, walker-walk) | GPU optimised | 590 | 20.0 | 499 | 453 | 1153 | fits at 453; OOM predicted only at 1153 > 499 - no OOM possible |
| walker (walker-run, walker-walk) | GPU + ckpt_unroll | 612 | 4.3 | 499 | 453 | 5410 | fits at 453; OOM predicted only at 5410 > 499 - no OOM possible |

## Seeds per RTX 3090 vs horizon

**Separate processes** (one run per process): N = floor(free / VRAM_process(H)), VRAM measured in the horizon sweep. **Vectorised** (all seeds in ONE `train_vec.py` process): S = 1 + floor((free - V1(H)) / m(H)) with V1 = one-seed process VRAM and m = marginal VRAM per extra seed (measured with S = 1 and S = 8, linear in H between H = 5 and 30). Empirical packing tests (next section) confirm these numbers.

| task | mode | H=2 | H=5 | H=10 | H=15 | H=20 | H=25 | H=30 |
|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| acrobot-swingup | GPU faithful, processes | 45 | 41 | 36 | 32 | 28 | 26 | 24 |
| acrobot-swingup | GPU optimised, processes | 40 | 37 | 32 | 28 | 25 | 22 | 20 |
| acrobot-swingup | GPU vectorised, 1 process | 229 | 134 | 79 | 56 | 43 | 35 | 30 |
| acrobot-swingup | GPU vectorised + ckpt_unroll | 281 | 278 | 273 | 268 | 263 | 259 | 254 |
| cartpole-swingup | CPU baseline, processes | 59 | 52 | 44 | 38 | 34 | 30 | 27 |
| cartpole-swingup | GPU faithful, processes | 45 | 41 | 36 | 32 | 29 | 26 | 24 |
| cartpole-swingup | GPU optimised, processes | 40 | 36 | 31 | 28 | 24 | 22 | 20 |
| cartpole-swingup | GPU vectorised, 1 process | 235 | 136 | 80 | 56 | 44 | 35 | 30 |
| cartpole-swingup | GPU vectorised + ckpt_unroll | 304 | 300 | 294 | 288 | 283 | 278 | 272 |
| cartpole-swingup-sparse | CPU baseline, processes | 59 | 52 | 44 | 38 | 34 | 30 | 27 |
| cartpole-swingup-sparse | GPU faithful, processes | 45 | 42 | 36 | 32 | 29 | 26 | 24 |
| cartpole-swingup-sparse | GPU optimised, processes | 40 | 36 | 31 | 28 | 25 | 22 | 20 |
| cartpole-swingup-sparse | GPU vectorised, 1 process | 235 | 136 | 80 | 56 | 44 | 35 | 30 |
| cartpole-swingup-sparse | GPU vectorised + ckpt_unroll | 304 | 300 | 294 | 288 | 283 | 278 | 272 |
| cheetah-run | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| cheetah-run | GPU faithful, processes | 45 | 41 | 36 | 32 | 28 | 26 | 23 |
| cheetah-run | GPU optimised, processes | 39 | 36 | 31 | 27 | 24 | 22 | 20 |
| cheetah-run | GPU vectorised, 1 process | 209 | 127 | 76 | 55 | 42 | 35 | 29 |
| cheetah-run | GPU vectorised + ckpt_unroll | 256 | 254 | 250 | 246 | 242 | 238 | 234 |
| cup-catch | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| cup-catch | GPU faithful, processes | 43 | 40 | 35 | 31 | 28 | 25 | 23 |
| cup-catch | GPU optimised, processes | 38 | 35 | 30 | 27 | 24 | 22 | 20 |
| cup-catch | GPU vectorised, 1 process | 231 | 135 | 79 | 56 | 43 | 35 | 30 |
| cup-catch | GPU vectorised + ckpt_unroll | 296 | 291 | 283 | 275 | 267 | 260 | 253 |
| dog-run | CPU baseline, processes | 15 | 13 | 11 | 9 | 8 | 7 | 6 |
| dog-run | GPU faithful, processes | 13 | 12 | 10 | 9 | 7 | 7 | 6 |
| dog-run | GPU optimised, processes | 13 | 11 | 9 | 8 | 7 | 6 | 6 |
| dog-run | GPU vectorised, 1 process | 18 | 15 | 12 | 9 | 8 | 7 | 6 |
| dog-run | GPU vectorised + ckpt_unroll | 19 | 19 | 18 | 18 | 18 | 17 | 17 |
| dog-trot | CPU baseline, processes | 15 | 13 | 11 | 9 | 8 | 7 | 6 |
| dog-trot | GPU faithful, processes | 13 | 12 | 10 | 9 | 7 | 7 | 6 |
| dog-trot | GPU optimised, processes | 13 | 11 | 9 | 8 | 7 | 6 | 6 |
| dog-trot | GPU vectorised, 1 process | 18 | 15 | 12 | 9 | 8 | 7 | 6 |
| dog-trot | GPU vectorised + ckpt_unroll | 19 | 19 | 18 | 18 | 18 | 17 | 17 |
| dog-walk | CPU baseline, processes | 15 | 13 | 11 | 9 | 8 | 7 | 6 |
| dog-walk | GPU faithful, processes | 13 | 12 | 10 | 9 | 7 | 7 | 6 |
| dog-walk | GPU optimised, processes | 13 | 11 | 9 | 8 | 7 | 6 | 6 |
| dog-walk | GPU vectorised, 1 process | 18 | 15 | 12 | 9 | 8 | 7 | 6 |
| dog-walk | GPU vectorised + ckpt_unroll | 19 | 19 | 18 | 18 | 18 | 17 | 17 |
| finger-spin | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| finger-spin | GPU faithful, processes | 45 | 41 | 36 | 32 | 29 | 26 | 24 |
| finger-spin | GPU optimised, processes | 40 | 36 | 31 | 28 | 24 | 22 | 20 |
| finger-spin | GPU vectorised, 1 process | 209 | 127 | 76 | 55 | 43 | 35 | 29 |
| finger-spin | GPU vectorised + ckpt_unroll | 252 | 249 | 245 | 242 | 238 | 234 | 231 |
| finger-turn-hard | CPU baseline, processes | 54 | 48 | 41 | 36 | 32 | 29 | 26 |
| finger-turn-hard | GPU faithful, processes | 44 | 40 | 35 | 31 | 28 | 26 | 23 |
| finger-turn-hard | GPU optimised, processes | 39 | 35 | 31 | 27 | 24 | 22 | 20 |
| finger-turn-hard | GPU vectorised, 1 process | 209 | 127 | 76 | 55 | 43 | 35 | 29 |
| finger-turn-hard | GPU vectorised + ckpt_unroll | 252 | 249 | 245 | 242 | 238 | 234 | 231 |
| fish-swim | CPU baseline, processes | 54 | 49 | 41 | 36 | 32 | 29 | 26 |
| fish-swim | GPU faithful, processes | 43 | 39 | 34 | 31 | 28 | 25 | 23 |
| fish-swim | GPU optimised, processes | 38 | 34 | 30 | 26 | 24 | 21 | 20 |
| fish-swim | GPU vectorised, 1 process | 204 | 125 | 76 | 54 | 42 | 35 | 29 |
| fish-swim | GPU vectorised + ckpt_unroll | 250 | 246 | 239 | 233 | 227 | 222 | 216 |
| hopper-hop | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| hopper-hop | GPU faithful, processes | 44 | 40 | 35 | 31 | 28 | 25 | 23 |
| hopper-hop | GPU optimised, processes | 38 | 35 | 30 | 27 | 24 | 22 | 20 |
| hopper-hop | GPU vectorised, 1 process | 213 | 128 | 77 | 55 | 43 | 35 | 29 |
| hopper-hop | GPU vectorised + ckpt_unroll | 266 | 262 | 255 | 249 | 242 | 236 | 231 |
| humanoid-run | CPU baseline, processes | 30 | 29 | 26 | 23 | 22 | 20 | 19 |
| humanoid-run | GPU faithful, processes | 27 | 25 | 23 | 21 | 20 | 18 | 17 |
| humanoid-run | GPU optimised, processes | 24 | 23 | 21 | 19 | 17 | 16 | 15 |
| humanoid-run | GPU vectorised, 1 process | 51 | 44 | 36 | 30 | 26 | 23 | 20 |
| humanoid-run | GPU vectorised + ckpt_unroll | 54 | 53 | 52 | 52 | 51 | 50 | 49 |
| humanoid-stand | CPU baseline, processes | 30 | 29 | 26 | 23 | 22 | 20 | 19 |
| humanoid-stand | GPU faithful, processes | 27 | 25 | 23 | 21 | 20 | 18 | 17 |
| humanoid-stand | GPU optimised, processes | 24 | 23 | 21 | 19 | 17 | 16 | 15 |
| humanoid-stand | GPU vectorised, 1 process | 51 | 44 | 36 | 30 | 26 | 23 | 20 |
| humanoid-stand | GPU vectorised + ckpt_unroll | 54 | 53 | 52 | 52 | 51 | 50 | 49 |
| humanoid-walk | CPU baseline, processes | 30 | 29 | 26 | 23 | 22 | 20 | 19 |
| humanoid-walk | GPU faithful, processes | 27 | 25 | 23 | 21 | 20 | 18 | 17 |
| humanoid-walk | GPU optimised, processes | 24 | 23 | 21 | 19 | 17 | 16 | 15 |
| humanoid-walk | GPU vectorised, 1 process | 51 | 44 | 36 | 30 | 26 | 23 | 20 |
| humanoid-walk | GPU vectorised + ckpt_unroll | 54 | 53 | 52 | 52 | 51 | 50 | 49 |
| quadruped-run | CPU baseline, processes | 51 | 46 | 40 | 35 | 31 | 28 | 25 |
| quadruped-run | GPU faithful, processes | 40 | 37 | 33 | 29 | 27 | 24 | 22 |
| quadruped-run | GPU optimised, processes | 36 | 33 | 29 | 26 | 23 | 21 | 19 |
| quadruped-run | GPU vectorised, 1 process | 136 | 95 | 63 | 47 | 38 | 31 | 27 |
| quadruped-run | GPU vectorised + ckpt_unroll | 152 | 150 | 147 | 144 | 141 | 138 | 135 |
| quadruped-walk | CPU baseline, processes | 51 | 46 | 40 | 35 | 31 | 28 | 25 |
| quadruped-walk | GPU faithful, processes | 40 | 37 | 33 | 29 | 27 | 24 | 22 |
| quadruped-walk | GPU optimised, processes | 36 | 33 | 29 | 26 | 23 | 21 | 19 |
| quadruped-walk | GPU vectorised, 1 process | 136 | 95 | 63 | 47 | 38 | 31 | 27 |
| quadruped-walk | GPU vectorised + ckpt_unroll | 152 | 150 | 147 | 144 | 141 | 138 | 135 |
| reacher-easy | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| reacher-easy | GPU faithful, processes | 45 | 41 | 36 | 32 | 29 | 26 | 24 |
| reacher-easy | GPU optimised, processes | 41 | 37 | 32 | 28 | 25 | 23 | 21 |
| reacher-easy | GPU vectorised, 1 process | 228 | 133 | 79 | 56 | 43 | 35 | 30 |
| reacher-easy | GPU vectorised + ckpt_unroll | 293 | 287 | 279 | 272 | 265 | 258 | 251 |
| reacher-hard | CPU baseline, processes | 56 | 50 | 42 | 37 | 32 | 29 | 26 |
| reacher-hard | GPU faithful, processes | 45 | 41 | 36 | 32 | 29 | 26 | 24 |
| reacher-hard | GPU optimised, processes | 41 | 37 | 32 | 28 | 25 | 23 | 21 |
| reacher-hard | GPU vectorised, 1 process | 228 | 133 | 79 | 56 | 43 | 35 | 30 |
| reacher-hard | GPU vectorised + ckpt_unroll | 293 | 287 | 279 | 272 | 265 | 258 | 251 |
| walker-run | CPU baseline, processes | 52 | 47 | 40 | 35 | 31 | 28 | 26 |
| walker-run | GPU faithful, processes | 42 | 38 | 34 | 30 | 27 | 25 | 23 |
| walker-run | GPU optimised, processes | 37 | 34 | 29 | 26 | 23 | 21 | 19 |
| walker-run | GPU vectorised, 1 process | 177 | 114 | 72 | 52 | 41 | 34 | 29 |
| walker-run | GPU vectorised + ckpt_unroll | 214 | 210 | 205 | 199 | 194 | 189 | 184 |
| walker-walk | CPU baseline, processes | 52 | 47 | 40 | 35 | 31 | 28 | 26 |
| walker-walk | GPU faithful, processes | 42 | 38 | 34 | 30 | 27 | 25 | 23 |
| walker-walk | GPU optimised, processes | 37 | 34 | 29 | 26 | 23 | 21 | 19 |
| walker-walk | GPU vectorised, 1 process | 177 | 114 | 72 | 52 | 41 | 34 | 29 |
| walker-walk | GPU vectorised + ckpt_unroll | 214 | 210 | 205 | 199 | 194 | 189 | 184 |

## Mixed horizons running together

Separate processes with horizons H_1..H_N fit iff sum_i (a + b*H_i) <= free, i.e. N_max = floor(free / (a + b*mean H)). Columns: mean horizon of the mix (e.g. 17.5 = an even mix of H = 5, 10, 15, 20, 25, 30; 16 = H = 2..30 as in the sweep).

| task | mode | a | b | mean H = 5 | 10 | 16 | 17.5 | 30 |
|---|---|---|---|---|---|---|---|---|
| acrobot-swingup | CPU baseline | 386 | 16.5 | 50 | 42 | 36 | 35 | 26 |
| acrobot-swingup | GPU optimised | 537 | 19.8 | 37 | 32 | 27 | 26 | 20 |
| cartpole-swingup | CPU baseline | 367 | 16.4 | 52 | 44 | 37 | 36 | 27 |
| cartpole-swingup | GPU optimised | 547 | 19.9 | 36 | 31 | 27 | 26 | 20 |
| cartpole-swingup-sparse | CPU baseline | 367 | 16.4 | 52 | 44 | 37 | 36 | 27 |
| cartpole-swingup-sparse | GPU optimised | 545 | 20.0 | 36 | 31 | 27 | 26 | 20 |
| cheetah-run | CPU baseline | 386 | 16.5 | 50 | 42 | 36 | 34 | 26 |
| cheetah-run | GPU optimised | 554 | 20.0 | 36 | 31 | 27 | 26 | 20 |
| cup-catch | CPU baseline | 388 | 16.4 | 50 | 42 | 36 | 34 | 26 |
| cup-catch | GPU optimised | 574 | 19.9 | 35 | 30 | 26 | 25 | 20 |
| dog-run | CPU baseline | 1409 | 70.9 | 13 | 11 | 9 | 8 | 6 |
| dog-run | GPU optimised | 1628 | 74.7 | 11 | 9 | 8 | 8 | 6 |
| dog-trot | CPU baseline | 1409 | 70.9 | 13 | 11 | 9 | 8 | 6 |
| dog-trot | GPU optimised | 1628 | 74.7 | 11 | 9 | 8 | 8 | 6 |
| dog-walk | CPU baseline | 1409 | 70.9 | 13 | 11 | 9 | 8 | 6 |
| dog-walk | GPU optimised | 1628 | 74.7 | 11 | 9 | 8 | 8 | 6 |
| finger-spin | CPU baseline | 388 | 16.4 | 50 | 42 | 36 | 34 | 26 |
| finger-spin | GPU optimised | 547 | 19.9 | 36 | 31 | 27 | 26 | 20 |
| finger-turn-hard | CPU baseline | 400 | 16.4 | 49 | 41 | 35 | 34 | 26 |
| finger-turn-hard | GPU optimised | 560 | 19.9 | 35 | 31 | 26 | 25 | 20 |
| fish-swim | CPU baseline | 399 | 16.5 | 49 | 41 | 35 | 34 | 26 |
| fish-swim | GPU optimised | 579 | 19.9 | 34 | 30 | 26 | 25 | 20 |
| hopper-hop | CPU baseline | 386 | 16.5 | 50 | 42 | 36 | 34 | 26 |
| hopper-hop | GPU optimised | 567 | 19.8 | 35 | 30 | 26 | 25 | 20 |
| humanoid-run | CPU baseline | 730 | 16.9 | 29 | 26 | 23 | 23 | 19 |
| humanoid-run | GPU optimised | 910 | 21.1 | 23 | 21 | 18 | 18 | 15 |
| humanoid-stand | CPU baseline | 730 | 16.9 | 29 | 26 | 23 | 23 | 19 |
| humanoid-stand | GPU optimised | 910 | 21.1 | 23 | 21 | 18 | 18 | 15 |
| humanoid-walk | CPU baseline | 730 | 16.9 | 29 | 26 | 23 | 23 | 19 |
| humanoid-walk | GPU optimised | 910 | 21.1 | 23 | 21 | 18 | 18 | 15 |
| quadruped-run | CPU baseline | 426 | 16.5 | 46 | 39 | 34 | 33 | 25 |
| quadruped-run | GPU optimised | 609 | 20.1 | 33 | 29 | 25 | 24 | 19 |
| quadruped-walk | CPU baseline | 426 | 16.5 | 46 | 39 | 34 | 33 | 25 |
| quadruped-walk | GPU optimised | 609 | 20.1 | 33 | 29 | 25 | 24 | 19 |
| reacher-easy | CPU baseline | 388 | 16.4 | 50 | 42 | 36 | 34 | 26 |
| reacher-easy | GPU optimised | 531 | 19.7 | 37 | 32 | 27 | 26 | 21 |
| reacher-hard | CPU baseline | 388 | 16.4 | 50 | 42 | 36 | 34 | 26 |
| reacher-hard | GPU optimised | 531 | 19.7 | 37 | 32 | 27 | 26 | 21 |
| walker-run | CPU baseline | 412 | 16.5 | 47 | 40 | 34 | 33 | 26 |
| walker-run | GPU optimised | 590 | 20.0 | 34 | 29 | 25 | 25 | 19 |
| walker-walk | CPU baseline | 412 | 16.5 | 47 | 40 | 34 | 33 | 26 |
| walker-walk | GPU optimised | 590 | 20.0 | 34 | 29 | 25 | 25 | 19 |

## Seeds x horizon settings: k seeds at EACH of H = 2, 5, 10, 15, 20, 25, 30 (7 settings, 7k runs)

Largest k such that all 7k runs fit in VRAM at the same time. (a) one process per run; (b) one vectorised process per horizon setting holding its k seeds (V1(H) + (k-1) m(H) each; m measured at H = 5 and 30, linear in H); (c) as (b) with activation checkpointing. **No-slowdown limit for any of these sets: 1 run in total** (sections above / chapter 4).

| task | (a) processes: k | total runs | (b) vectorised: k | total runs | (c) vectorised + ckpt: k | total runs |
|---|---|---|---|---|---|---|
| acrobot-swingup | 4 | 28 | 7 | 49 | 32 | 224 |
| cartpole-swingup | 3 | 21 | 7 | 49 | 35 | 245 |
| cartpole-swingup-sparse | 3 | 21 | 7 | 49 | 35 | 245 |
| cheetah-run | 3 | 21 | 6 | 42 | 30 | 210 |
| cup-catch | 3 | 21 | 6 | 42 | 33 | 231 |
| dog-run | 1 | 7 | 1 | 7 | 2 | 14 |
| dog-trot | 1 | 7 | 1 | 7 | 2 | 14 |
| dog-walk | 1 | 7 | 1 | 7 | 2 | 14 |
| finger-spin | 3 | 21 | 6 | 42 | 29 | 203 |
| finger-turn-hard | 3 | 21 | 6 | 42 | 29 | 203 |
| fish-swim | 3 | 21 | 6 | 42 | 28 | 196 |
| hopper-hop | 3 | 21 | 6 | 42 | 30 | 210 |
| humanoid-run | 2 | 14 | 3 | 21 | 6 | 42 |
| humanoid-stand | 2 | 14 | 3 | 21 | 6 | 42 |
| humanoid-walk | 2 | 14 | 3 | 21 | 6 | 42 |
| quadruped-run | 3 | 21 | 5 | 35 | 17 | 119 |
| quadruped-walk | 3 | 21 | 5 | 35 | 17 | 119 |
| reacher-easy | 4 | 28 | 7 | 49 | 33 | 231 |
| reacher-hard | 4 | 28 | 7 | 49 | 33 | 231 |
| walker-run | 3 | 21 | 6 | 42 | 24 | 168 |
| walker-walk | 3 | 21 | 6 | 42 | 24 | 168 |

## Empirical packing tests (launch runs until CUDA OOM)

| experiment | runs that fit | first failure | GPU memory used when it failed (MiB) | mean process VRAM (MiB) |
|---|---|---|---|---|
| cross_gpu_opt_mps (MPS) | 18 | OOM | 23952 | 1203 |
| mixed_gpu_opt_mps_dog-run_H5-10-20-30 (MPS) | 8 | OOM | 23834 | 2840 |
| mixed_gpu_opt_mps_humanoid-run_H5-10-20-30 (MPS) | 18 | OOM | 24058 | 1231 |
| mixed_gpu_opt_mps_walker-walk_H5-10-20-30 (MPS) | 25 | OOM | 24041 | 906 |
| same_cpu_dog-run_H5 | 13 | OOM | 23831 | 1754 |
| same_cpu_walker-walk_H5 | 46 | OOM | 23935 | 494 |
| same_gpu_opt_dog-run_H5 | 11 | OOM | 23412 | 1996 |
| same_gpu_opt_mps_cartpole-swingup_H5 (MPS) | 35 | OOM | 24062 | 648 |
| same_gpu_opt_mps_dog-run_H5 (MPS) | 11 | OOM | 23391 | 1996 |
| same_gpu_opt_mps_humanoid-run_H5 (MPS) | 22 | OOM | 24053 | 1012 |
| same_gpu_opt_mps_walker-walk_H5 (MPS) | 33 | OOM | 24067 | 690 |
| same_gpu_opt_walker-walk_H5 | 33 | OOM | 23844 | 688 |
| vec_cartpole-swingup_H30 (one process, seeds S) | S <= 36 | OOM at S = 38 | – | – |
| vec_cartpole-swingup_H5 (one process, seeds S) | S <= 160 | OOM at S = 168 | – | – |
| vec_dog-run_H30 (one process, seeds S) | S <= 6 | OOM at S = 7 | – | – |
| vec_dog-run_H30_ckpt_unroll-true (one process, seeds S) | S <= 16 | OOM at S = 17 | – | – |
| vec_dog-run_H5 (one process, seeds S) | S <= 15 | OOM at S = 16 | – | – |
| vec_humanoid-run_H30 (one process, seeds S) | S <= 22 | OOM at S = 23 | – | – |
| vec_humanoid-run_H5 (one process, seeds S) | S <= 44 | OOM at S = 46 | – | – |
| vec_walker-walk_H30 (one process, seeds S) | S <= 34 | OOM at S = 36 | – | – |
| vec_walker-walk_H30_ckpt_unroll-true (one process, seeds S) | S <= 208 | OOM at S = 216 | – | – |
| vec_walker-walk_H5 (one process, seeds S) | S <= 136 | OOM at S = 144 | – | – |
