"""Minimal graphics-context process: create and destroy a Vulkan device repeatedly (`vulkaninfo --summary`).

Used by bench/mps_fault_hunt.py (--non_mps_script bench/vulkan_ping.py) to test whether graphics / compute contexts of
desktop programs (browsers, editors, file managers open GPU contexts when windows open and close) starting and ending
next to MPS clients can trigger the MPS fault (docs/08_known_issues.md). Contains no CUDA code.

Usage: python bench/vulkan_ping.py [max_seconds=20]

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import subprocess
import sys
import time

seconds = 20.0
for arg in sys.argv[1:]:
	if arg.startswith('max_seconds='):
		seconds = float(arg.split('=', 1)[1])
t0 = time.time()
while time.time() - t0 < seconds:
	subprocess.run(['vulkaninfo', '--summary'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	time.sleep(1.0)
