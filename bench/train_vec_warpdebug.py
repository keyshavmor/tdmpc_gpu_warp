"""Fault diagnostic: src/train_vec.py with all warp kernels compiled in warp debug mode.

train_vec.py with every warp / mujoco_warp kernel compiled in warp DEBUG mode (bounds-checked array
accesses, verified launches) - to locate a rare illegal-memory-access fault seen in long vectorised runs.
An out-of-bounds index inside a warp kernel then raises a warp error naming the kernel and array instead
of a later, unattributable CUDA fault. Takes the same command-line overrides as src/train_vec.py
(kernels are recompiled in debug mode, so the first run compiles for a while and the run is slower).

Diagnostic for the open issue "Rare GPU fault in long multi-world (vectorised) runs" in
docs/08_known_issues.md, section 8.1 (`illegal memory access` / `unspecified launch failure` in train_vec.py
processes, mostly under CUDA MPS; cause unconfirmed).

Environment: tdmpc-warp (a CUDA GPU).
Usage: python bench/train_vec_warpdebug.py task=walker-walk num_seeds=3 seed=1
Output: the usual train_vec.py logs (logs/<task>/state/<exp_name>/<seed>/) and warp diagnostics on stderr.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import warp as wp
wp.config.mode = 'debug'
# (verify_cuda cannot be combined with CUDA-graph capture; debug mode keeps bounds checking)
import train_vec
from cfg import parse_cfg
from pathlib import Path
train_vec.train(parse_cfg(Path(os.path.join(os.path.dirname(__file__), '..', 'cfgs'))))
