"""Minimal CUDA process: create a CUDA context, allocate, run a few small kernels, synchronise and exit normally.

Used by bench/mps_fault_hunt.py (--non_mps_script bench/cuda_ping.py) as a non-MPS process that contains none of
this repository's code (no warp, no MuJoCo Warp, no CUDA graphs), to test whether starting / ending an ordinary CUDA
context next to MPS clients is enough to trigger the MPS fault (docs/08_known_issues.md).

Usage: python bench/cuda_ping.py [max_seconds=5]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import sys
import time
import torch

seconds = 5.0
for arg in sys.argv[1:]:
	if arg.startswith('max_seconds='):
		seconds = float(arg.split('=', 1)[1])
t0 = time.time()
x = torch.randn(1024, 1024, device='cuda')
while time.time() - t0 < seconds:
	x = torch.tanh(x @ x.T * 1e-3)
	torch.cuda.synchronize()
	time.sleep(0.05)
torch.cuda.synchronize()
