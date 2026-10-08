"""Fault diagnostic: capture a GPU core dump of the rare multi-world fault and inspect it with cuda-gdb.

Capture a GPU core dump of the rare illegal-memory-access fault (fast repro: 6 vectorised walker trainings
under MPS, short seeding, frequent evaluation) and print the faulting kernel with cuda-gdb.
The training processes run with CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 (a lightweight dump without memory
contents) under a private MPS daemon; the MPS server log is polled for up to 15 minutes. On a fault the
faulting process's last output lines and, for the first two dumps, `info cuda kernels` / backtrace /
the faulting instruction are printed. Work directory: /tmp/tdmpc_gpu_warp/coredump (logs, dumps, MPS).

Diagnostic for the open issue "Rare GPU fault in long multi-world (vectorised) runs" in
docs/08_known_issues.md, section 8.1 (`illegal memory access` / `unspecified launch failure` in train_vec.py
processes, mostly under CUDA MPS; cause unconfirmed).

Environment: launched from any Python 3; the training processes use $TDMPC_WARP_PYTHON (default: the interpreter running this script).
Needs nvidia-cuda-mps-control and /usr/local/cuda/bin/cuda-gdb.
Usage: python bench/fault_coredump.py
Output: stdout. Result (docs/08_known_issues.md, section 8.1): no fault was captured within the 15-minute
window.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import glob, os, re, shutil, subprocess, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
W = '/tmp/tdmpc_gpu_warp/coredump'; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=f'{W}/mps/pipe', CUDA_MPS_LOG_DIRECTORY=f'{W}/mps/log',
		   CUDA_ENABLE_COREDUMP_ON_EXCEPTION='1', CUDA_COREDUMP_FILE=f'{W}/core_%p',
		   CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory')
os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env)
procs = {}
for i in range(6):
	cmd = [PY, 'src/train_vec.py', 'task=walker-walk', 'num_seeds=3', f'seed={7 * i + 1}', 'seed_steps=1000', 'eval_freq=4000',
		   'eval_episodes=2', f'exp_name=core{i}', f'log_root={W}/logs']
	p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(f'{W}/out{i}.log', 'w'), stderr=subprocess.STDOUT)
	procs[p.pid] = i
	time.sleep(10)
t0, fault = time.time(), None
while time.time() - t0 < 900:  # 15-minute window
	time.sleep(5)
	m = re.search(r'Client \{PID: (\d+), Context: \d+\} encountered a fatal GPU error', open(f'{W}/mps/log/server.log').read())
	if m:
		fault = int(m.group(1)); break
time.sleep(30)  # let the dump be written
for p in procs:
	try: os.kill(p, 9)
	except ProcessLookupError: pass
subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env)
print('fault pid', fault, 'process index', procs.get(fault), flush=True)
cores = sorted(glob.glob(f'{W}/core_*'))
print('core dumps:', [(c, os.path.getsize(c)) for c in cores], flush=True)
if fault is not None:
	print(''.join(open(f'{W}/out{procs[fault]}.log').read().splitlines(True)[-15:]), flush=True)
for c in cores[:2]:
	r = subprocess.run(['/usr/local/cuda/bin/cuda-gdb', '-batch', '-ex', f'target cudacore {c}', '-ex', 'info cuda kernels',
						'-ex', 'bt', '-ex', 'x/4i $pc'], capture_output=True, text=True, timeout=600)
	print(f'===== cuda-gdb {c}\n', r.stdout[-6000:], r.stderr[-2000:], flush=True)
