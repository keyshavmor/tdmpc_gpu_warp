"""Fault diagnostic: reproduce the rare GPU fault with N concurrent vectorised trainings under MPS.

Reproduce the rare MPS fault: N vectorised training processes under MPS with frequent evaluations
(eval_freq=2000 env steps, 2 eval episodes, seed_steps=1000), with graph-captured eval planning (default)
or eager eval planning (eval_graphs=false). Reports fatal GPU errors from the MPS server log.
  python bench/mps_fault_repro.py {graphs|eager} [N] [minutes]      (defaults: N=8, 15 minutes)
Processes alternate walker-walk / cartpole-swingup, 3 seeds each, started 5 s apart under a private MPS
daemon (/tmp/tdmpc_gpu_warp/mps_repro; training logs under /tmp/tdmpc_gpu_warp/repro_logs_<mode>). The run ends at
the first fatal error, when all processes have exited, or when the time is up; the number of completed
evaluations is counted from the processes' output.

Diagnostic for the open issue "Rare GPU fault in long multi-world (vectorised) runs" in
docs/08_known_issues.md, section 8.1 (`illegal memory access` / `unspecified launch failure` in train_vec.py
processes, mostly under CUDA MPS; cause unconfirmed).

Environment: launched from any Python 3; the training processes use $TDMPC_WARP_PYTHON (default: the interpreter running this script).
Needs nvidia-cuda-mps-control (user-level MPS, no root).
Output: one summary line on stdout. Observations summarised in docs/08_known_issues.md, section 8.1.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, subprocess, sys, time, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
mode = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8; minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 15
MPS = '/tmp/tdmpc_gpu_warp/mps_repro'; shutil.rmtree(MPS, ignore_errors=True)
env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=f'{MPS}/pipe', CUDA_MPS_LOG_DIRECTORY=f'{MPS}/log')
os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env)
logroot = f'/tmp/tdmpc_gpu_warp/repro_logs_{mode}'; shutil.rmtree(logroot, ignore_errors=True)
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
tasks = ['walker-walk', 'cartpole-swingup', 'walker-walk', 'cartpole-swingup', 'walker-walk', 'cartpole-swingup', 'walker-walk', 'cartpole-swingup']
procs = []
for i in range(N):
	cmd = [PY, 'src/train_vec.py', f'task={tasks[i % len(tasks)]}', 'num_seeds=3', f'seed={10 * i}', 'seed_steps=1000', 'eval_freq=2000',
		   'eval_episodes=2', f'exp_name=repro{i}', f'log_root={logroot}'] + (['eval_graphs=false'] if mode == 'eager' else [])
	procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(f'{MPS}/out{i}.log', 'w'), stderr=subprocess.STDOUT))
	time.sleep(5)
t0 = time.time(); fatal = False
while time.time() - t0 < minutes * 60:
	time.sleep(10)
	if 'fatal' in open(f'{MPS}/log/server.log').read().lower():  # MPS server reports a client's fatal GPU error
		fatal = True; break
	if all(p.poll() is not None for p in procs):
		break
for p in procs:
	if p.poll() is None:
		p.kill()
evals = sum(open(f).read().count('eval S') for f in glob.glob(f'{MPS}/out*.log'))
subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env)
print(f'mode={mode} N={N}: fatal GPU error={fatal} after {time.time() - t0:.0f}s, evaluations completed={evals}', flush=True)
