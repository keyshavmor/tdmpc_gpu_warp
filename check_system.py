"""Check that a Linux machine is set up for tdmpc_gpu_warp and which optimisations it can use.

Checks the NVIDIA driver (version >= 580: warp-lang 1.18 is built with CUDA 13, torch with CUDA 12.8), the GPU (compute capability; bf16 and
TF32 need >= 8.0), the CUDA MPS control binary (parallel runs), the GPU compute mode, libcuda (blocking
sync), the optional profiling / debugging tools (nsys, compute-sanitizer) and the Python packages
(torch with CUDA, warp, mujoco, mujoco_warp with the GPU noslip solver, dm_control). Prints one line per
check: OK / WARN / FAIL. No GPU work is run apart from initialising CUDA.

Usage (inside the tdmpc-warp environment, from the repository root):
	python check_system.py
Machine set-up instructions: docs/USER_GUIDE.md, section 1.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import ctypes
import os
import shutil
import subprocess
import sys

results = []


def report(status, what, detail=''):
	"""Record and print one check result (status: OK / WARN / FAIL)."""
	results.append(status)
	print(f'[{status:4}] {what}' + (f': {detail}' if detail else ''))


def smi(query):
	"""Return one nvidia-smi --query-gpu field of GPU 0, or None."""
	try:
		out = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
		return out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else None
	except (OSError, subprocess.SubprocessError):
		return None


# ---------------------------------------------------------------- system level
if shutil.which('nvidia-smi') is None:
	report('FAIL', 'NVIDIA driver', 'nvidia-smi not found - install the NVIDIA driver (docs/USER_GUIDE.md 1.1)')
else:
	drv = smi('driver_version')
	ok = drv is not None and int(drv.split('.')[0]) >= 580
	report('OK' if ok else 'FAIL', 'NVIDIA driver', f'{drv} (need >= 580: warp-lang 1.18 is built with CUDA 13, torch with CUDA 12.8)')
	report('OK', 'GPU', f'{smi("name")}, {smi("memory.total")}')
	mode = smi('compute_mode')
	report('OK' if mode == 'Default' else 'WARN', 'compute mode', f'{mode} (Default: several processes and MPS work without root)')
try:
	ctypes.CDLL('libcuda.so.1')
	report('OK', 'libcuda.so.1', 'blocking-sync context flag available (host threads sleep while waiting)')
except OSError:
	report('FAIL', 'libcuda.so.1', 'not loadable - driver user-space libraries missing')
mps = shutil.which('nvidia-cuda-mps-control')
report('OK' if mps else 'WARN', 'CUDA MPS (parallel runs)', mps or 'nvidia-cuda-mps-control not found - Ubuntu package nvidia-compute-utils-<driver>')
for tool, use in (('nsys', 'profiling (optional)'), ('compute-sanitizer', 'memory checking (optional)')):
	p = shutil.which(tool) or (f'/usr/local/cuda/bin/{tool}' if os.path.exists(f'/usr/local/cuda/bin/{tool}') else None)
	report('OK' if p else 'WARN', f'{tool}: {use}', p or 'not found (CUDA toolkit / Nsight Systems; only needed for profiling or debugging)')

# ---------------------------------------------------------------- python level
try:
	import torch
	if torch.cuda.is_available():
		cc = torch.cuda.get_device_capability()
		report('OK', 'torch + CUDA', f'torch {torch.__version__}, CUDA {torch.version.cuda}, capability {cc[0]}.{cc[1]}')
		report('OK' if cc >= (8, 0) else 'WARN', 'bf16 / TF32 tensor cores (precision options)',
			   'available' if cc >= (8, 0) else 'need compute capability >= 8.0 (Ampere or newer)')
	else:
		report('FAIL', 'torch + CUDA', f'torch {torch.__version__} cannot use the GPU')
except ImportError as e:
	report('FAIL', 'torch', str(e))
try:
	import warp as wp
	wp.config.quiet = True
	wp.init()
	report('OK' if wp.is_cuda_available() else 'FAIL', 'warp', f'{wp.__version__}, CUDA {"available" if wp.is_cuda_available() else "NOT available"}')
except Exception as e:  # noqa: BLE001 - report any import / init problem
	report('FAIL', 'warp', str(e))
try:
	import mujoco
	import mujoco_warp
	from mujoco_warp._src import noslip  # noqa: F401 - present only in the local checkout
	report('OK', 'mujoco / mujoco_warp', f'mujoco {mujoco.__version__}, mujoco_warp with GPU noslip solver ({mujoco_warp.__file__})')
except ImportError as e:
	report('FAIL', 'mujoco / mujoco_warp', f'{e} - run ./scripts/setup_mujoco_warp.sh (patched MuJoCo Warp with the GPU noslip solver)')
try:
	import dm_control  # noqa: F401
	report('OK', 'dm_control', 'importable')
except ImportError as e:
	report('FAIL', 'dm_control', str(e))

print()
if 'FAIL' in results:
	print('Some required components are missing (FAIL) - see docs/USER_GUIDE.md section 1.')
	sys.exit(1)
print('All required components found.' + (' Optional components missing (WARN) - see docs/USER_GUIDE.md section 1.' if 'WARN' in results else ''))
