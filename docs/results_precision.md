# Reduced precision (bf16 / fp16 / TF32) and activation checkpointing: speed and peak VRAM

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Single seed, vectorised-agent code path for every variant (so fp32 is an exact reference), GPU env, CUDA graphs, host pacing. speed-up = fp32 time / variant time; dVRAM = variant process VRAM - fp32 process VRAM (negative = saving). Tasks of one domain have identical cost (same networks / buffers), one representative per domain is measured.

## acrobot-swingup

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 11.2 | 1.10x | 1.09x | 0.97x | 1.09x | 0.93x | 578 | +0 | +6 | +12 | +10 | -14 |
| 5 | 17.0 | 1.13x | 1.10x | 1.02x | 1.10x | 0.90x | 644 | +0 | +8 | +14 | +8 | -64 |
| 10 | 27.3 | 1.17x | 1.14x | 1.09x | 1.13x | 0.87x | 756 | +0 | +12 | +16 | +16 | -158 |
| 15 | 37.2 | 1.18x | 1.14x | 1.13x | 1.14x | 0.86x | 868 | +0 | +16 | +8 | +12 | -246 |
| 20 | 47.4 | 1.20x | 1.16x | 1.15x | 1.17x | 0.86x | 974 | +2 | +12 | +6 | +6 | -336 |
| 25 | 57.7 | 1.21x | 1.18x | 1.16x | 1.18x | 0.85x | 1086 | +0 | +18 | +6 | +22 | -422 |
| 30 | 68.1 | 1.22x | 1.18x | 1.18x | 1.19x | 0.85x | 1202 | +0 | +10 | -8 | +16 | -516 |

## cartpole-swingup

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 14.6 | 1.05x | 1.06x | 0.97x | 1.05x | 0.94x | 588 | +0 | +10 | +16 | +10 | -12 |
| 5 | 20.6 | 1.12x | 1.07x | 1.01x | 1.07x | 0.92x | 652 | +0 | +8 | +22 | +14 | -62 |
| 10 | 30.4 | 1.13x | 1.09x | 1.08x | 1.10x | 0.88x | 766 | +0 | +8 | +10 | +10 | -160 |
| 15 | 40.4 | 1.15x | 1.12x | 1.11x | 1.12x | 0.87x | 880 | +0 | +4 | +8 | +18 | -248 |
| 20 | 50.6 | 1.17x | 1.13x | 1.13x | 1.13x | 0.86x | 990 | +0 | +4 | +6 | +14 | -340 |
| 25 | 60.8 | 1.18x | 1.15x | 1.15x | 1.14x | 0.85x | 1104 | -2 | +10 | -2 | +12 | -430 |
| 30 | 71.8 | 1.19x | 1.15x | 1.17x | 1.16x | 0.86x | 1212 | +0 | +12 | -10 | +6 | -518 |

## cheetah-run

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 10.1 | 1.16x | 1.03x | 0.97x | 1.05x | 0.90x | 596 | +0 | +4 | +16 | +8 | -16 |
| 5 | 16.3 | 1.21x | 1.08x | 1.08x | 1.09x | 0.89x | 660 | +0 | +10 | +18 | +10 | -66 |
| 10 | 26.4 | 1.26x | 1.10x | 1.14x | 1.10x | 0.87x | 772 | +0 | +14 | +14 | +18 | -158 |
| 15 | 36.7 | 1.27x | 1.11x | 1.16x | 1.11x | 0.85x | 884 | +0 | +20 | +18 | +22 | -246 |
| 20 | 47.1 | 1.29x | 1.13x | 1.19x | 1.12x | 0.85x | 998 | +0 | +22 | +4 | +16 | -338 |
| 25 | 57.6 | 1.31x | 1.14x | 1.21x | 1.14x | 0.85x | 1108 | +0 | +28 | +2 | +30 | -430 |
| 30 | 67.9 | 1.31x | 1.14x | 1.22x | 1.14x | 0.85x | 1220 | -2 | +24 | -6 | +32 | -520 |

## cup-catch

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 17.5 | 1.07x | 1.03x | 0.99x | 1.03x | 0.96x | 612 | +0 | +12 | +24 | -8 | -10 |
| 5 | 22.9 | 1.12x | 1.05x | 1.03x | 1.04x | 0.90x | 678 | +0 | +20 | +22 | -4 | -62 |
| 10 | 32.8 | 1.17x | 1.05x | 1.08x | 1.06x | 0.89x | 796 | +0 | +6 | +10 | -10 | -160 |
| 15 | 43.1 | 1.21x | 1.09x | 1.11x | 1.09x | 0.88x | 906 | +0 | +12 | +12 | -10 | -248 |
| 20 | 53.4 | 1.23x | 1.10x | 1.15x | 1.10x | 0.87x | 1018 | +0 | +6 | +0 | -12 | -342 |
| 25 | 63.4 | 1.25x | 1.12x | 1.17x | 1.12x | 0.86x | 1132 | +0 | +6 | -12 | -10 | -434 |
| 30 | 74.4 | 1.27x | 1.13x | 1.19x | 1.13x | 0.86x | 1238 | +0 | +4 | -2 | -14 | -514 |

## dog-run

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 24.8 | 1.10x | 1.18x | 1.09x | 1.18x | 0.92x | 1800 | +2 | -40 | -30 | -464 | -80 |
| 5 | 37.7 | 1.13x | 1.25x | 1.18x | 1.24x | 0.86x | 2036 | +2 | -70 | -64 | -492 | -294 |
| 10 | 60.7 | 1.16x | 1.33x | 1.30x | 1.33x | 0.84x | 2454 | +2 | -132 | -128 | -556 | -656 |
| 15 | 83.5 | 1.18x | 1.37x | 1.36x | 1.37x | 0.84x | 2866 | +4 | -210 | -204 | -624 | -1024 |
| 20 | 106.4 | 1.18x | 1.40x | 1.39x | 1.40x | 0.83x | 3290 | +4 | -290 | -298 | -708 | -1416 |
| 25 | 128.6 | 1.19x | 1.40x | 1.40x | 1.41x | 0.81x | 3708 | +4 | -370 | -382 | -798 | -1790 |
| 30 | 152.3 | 1.20x | 1.42x | 1.43x | 1.43x | 0.82x | 4104 | +4 | -426 | -448 | -864 | -2164 |

## finger-spin

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 9.2 | 1.18x | 1.06x | 0.99x | 1.04x | 0.94x | 588 | +0 | +10 | +18 | +8 | -12 |
| 5 | 15.2 | 1.21x | 1.08x | 1.05x | 1.07x | 0.88x | 656 | +0 | +6 | +16 | +6 | -66 |
| 10 | 25.5 | 1.27x | 1.11x | 1.14x | 1.12x | 0.87x | 766 | +0 | +18 | +20 | +18 | -158 |
| 15 | 35.8 | 1.27x | 1.11x | 1.16x | 1.11x | 0.86x | 882 | +0 | +12 | +6 | +6 | -248 |
| 20 | 46.2 | 1.29x | 1.13x | 1.19x | 1.12x | 0.85x | 988 | +0 | +14 | +10 | +10 | -334 |
| 25 | 56.5 | 1.31x | 1.13x | 1.20x | 1.13x | 0.85x | 1102 | +0 | +0 | -2 | +6 | -428 |
| 30 | 67.0 | 1.32x | 1.14x | 1.22x | 1.13x | 0.85x | 1208 | +0 | +8 | -10 | +4 | -512 |

## fish-swim

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 17.0 | 1.03x | 0.98x | 0.95x | 1.00x | 0.92x | 620 | +0 | +6 | +16 | +4 | -16 |
| 5 | 23.3 | 1.10x | 1.06x | 1.03x | 1.05x | 0.92x | 688 | +0 | +8 | +18 | +8 | -68 |
| 10 | 33.0 | 1.12x | 1.06x | 1.07x | 1.05x | 0.88x | 800 | +0 | +8 | +16 | +8 | -158 |
| 15 | 43.2 | 1.14x | 1.08x | 1.11x | 1.08x | 0.88x | 910 | +0 | +10 | +16 | +16 | -248 |
| 20 | 53.5 | 1.16x | 1.08x | 1.13x | 1.08x | 0.87x | 1024 | +0 | +18 | +12 | +14 | -342 |
| 25 | 64.0 | 1.18x | 1.09x | 1.15x | 1.10x | 0.86x | 1140 | +0 | +12 | -6 | +6 | -436 |
| 30 | 74.5 | 1.19x | 1.11x | 1.17x | 1.11x | 0.86x | 1240 | +0 | +30 | -4 | +36 | -512 |

## hopper-hop

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 12.8 | 1.07x | 1.04x | 0.98x | 1.05x | 0.93x | 608 | +0 | +2 | +16 | +2 | -16 |
| 5 | 18.9 | 1.13x | 1.07x | 1.06x | 1.06x | 0.90x | 672 | +0 | +8 | +22 | +8 | -64 |
| 10 | 28.9 | 1.14x | 1.08x | 1.11x | 1.07x | 0.88x | 782 | +0 | +14 | +22 | +16 | -156 |
| 15 | 39.4 | 1.18x | 1.10x | 1.15x | 1.09x | 0.86x | 898 | +0 | +10 | +12 | +14 | -248 |
| 20 | 49.6 | 1.18x | 1.11x | 1.17x | 1.11x | 0.86x | 1008 | +0 | +18 | +2 | +12 | -336 |
| 25 | 60.2 | 1.21x | 1.12x | 1.19x | 1.13x | 0.86x | 1122 | +0 | +16 | -6 | +16 | -430 |
| 30 | 70.4 | 1.21x | 1.13x | 1.20x | 1.13x | 0.85x | 1234 | +0 | +12 | -8 | +14 | -518 |

## humanoid-run

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16.1 | 1.11x | 1.05x | 1.01x | 1.06x | 0.95x | 946 | +0 | +16 | +28 | -112 | -10 |
| 5 | 24.3 | 1.14x | 1.08x | 1.07x | 1.07x | 0.91x | 1024 | +0 | +14 | +30 | -116 | -70 |
| 10 | 38.8 | 1.18x | 1.12x | 1.15x | 1.11x | 0.90x | 1138 | +0 | +26 | +38 | -100 | -158 |
| 15 | 53.4 | 1.20x | 1.14x | 1.20x | 1.14x | 0.89x | 1268 | +0 | +28 | +26 | -106 | -260 |
| 20 | 67.8 | 1.22x | 1.15x | 1.22x | 1.15x | 0.89x | 1380 | +0 | +54 | +36 | -82 | -338 |
| 25 | 82.8 | 1.22x | 1.16x | 1.24x | 1.16x | 0.89x | 1490 | +0 | +74 | +36 | -60 | -428 |
| 30 | 97.3 | 1.23x | 1.17x | 1.25x | 1.17x | 0.88x | 1604 | +0 | +70 | +32 | -54 | -512 |

## quadruped-run

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 14.7 | 1.07x | 1.05x | 0.99x | 1.04x | 0.94x | 648 | +0 | +12 | +20 | -10 | -10 |
| 5 | 20.9 | 1.11x | 1.04x | 1.06x | 1.05x | 0.92x | 722 | +0 | +6 | +16 | -14 | -70 |
| 10 | 30.7 | 1.14x | 1.08x | 1.10x | 1.07x | 0.88x | 832 | -2 | +16 | +16 | -2 | -162 |
| 15 | 40.9 | 1.17x | 1.10x | 1.14x | 1.10x | 0.87x | 942 | +0 | +24 | +12 | -4 | -250 |
| 20 | 51.3 | 1.18x | 1.12x | 1.17x | 1.12x | 0.86x | 1054 | +0 | +22 | +12 | -2 | -338 |
| 25 | 61.7 | 1.19x | 1.13x | 1.19x | 1.12x | 0.86x | 1168 | +0 | +28 | +2 | +6 | -432 |
| 30 | 72.3 | 1.20x | 1.13x | 1.20x | 1.13x | 0.86x | 1280 | +0 | +26 | +0 | +2 | -518 |

## reacher-easy

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 9.1 | 1.20x | 1.08x | 0.98x | 1.08x | 0.93x | 570 | +0 | +10 | +20 | +8 | -12 |
| 5 | 14.9 | 1.19x | 1.08x | 1.04x | 1.08x | 0.88x | 636 | +0 | +8 | +20 | +10 | -64 |
| 10 | 25.4 | 1.26x | 1.12x | 1.13x | 1.11x | 0.86x | 746 | +0 | +10 | +12 | +14 | -156 |
| 15 | 35.5 | 1.28x | 1.13x | 1.17x | 1.13x | 0.86x | 858 | +0 | +12 | +16 | +14 | -242 |
| 20 | 46.3 | 1.31x | 1.14x | 1.19x | 1.14x | 0.86x | 968 | +0 | +10 | +6 | +10 | -334 |
| 25 | 56.4 | 1.30x | 1.15x | 1.21x | 1.14x | 0.85x | 1084 | +0 | +6 | -8 | +8 | -430 |
| 30 | 66.8 | 1.31x | 1.15x | 1.21x | 1.15x | 0.85x | 1192 | +0 | +6 | -12 | +4 | -512 |

## walker-walk

| H | fp32 ms | TF32 matmul speed-up | bf16 autocast speed-up | fp16 autocast + loss scaling speed-up | bf16 autocast + bf16 replay buffer speed-up | fp32 + activation checkpointing (exact) speed-up | fp32 VRAM | TF32 matmul dVRAM | bf16 autocast dVRAM | fp16 autocast + loss scaling dVRAM | bf16 autocast + bf16 replay buffer dVRAM | fp32 + activation checkpointing (exact) dVRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 13.7 | 1.10x | 1.05x | 0.97x | 1.04x | 0.92x | 632 | +0 | +6 | +16 | -8 | -16 |
| 5 | 19.7 | 1.15x | 1.07x | 1.03x | 1.05x | 0.91x | 700 | +0 | +8 | +14 | -6 | -66 |
| 10 | 30.1 | 1.22x | 1.09x | 1.10x | 1.08x | 0.87x | 808 | +0 | +16 | +18 | -2 | -156 |
| 15 | 40.2 | 1.23x | 1.10x | 1.13x | 1.10x | 0.87x | 924 | +0 | +18 | +6 | +4 | -250 |
| 20 | 50.4 | 1.25x | 1.12x | 1.16x | 1.12x | 0.86x | 1038 | +0 | +18 | +4 | +4 | -342 |
| 25 | 60.9 | 1.27x | 1.13x | 1.19x | 1.13x | 0.86x | 1142 | +0 | +26 | +6 | +20 | -426 |
| 30 | 71.3 | 1.29x | 1.14x | 1.20x | 1.13x | 0.85x | 1258 | +0 | +32 | -8 | +6 | -518 |

## Averages over the 12 domains

| variant | mean speed-up H=5 | mean speed-up H=30 | mean dVRAM H=5 (MiB) | mean dVRAM H=30 (MiB) |
|---|---|---|---|---|
| TF32 matmul | 1.14x | 1.24x | +0 | +0 |
| bf16 autocast | 1.09x | 1.16x | +3 | -16 |
| fp16 autocast + loss scaling | 1.06x | 1.22x | +12 | -40 |
| bf16 autocast + bf16 replay buffer | 1.08x | 1.17x | -47 | -68 |
| fp32 + activation checkpointing (exact) | 0.90x | 0.85x | -85 | -653 |
