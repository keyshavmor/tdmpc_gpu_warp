"""Fault diagnostic: bisect which CUDA-graph configuration triggers the rare multi-world GPU fault.

Bisect the rare illegal-memory-access fault: 6 vectorised walker trainings under MPS (short seeding,
frequent evaluation), 2 x default, 2 x agent CUDA graphs off, 2 x env CUDA graph off. On a fault the MPS
server log names the faulting PID -> configuration. Relaunches after a fault until the time budget is used.
Each round starts a private MPS daemon (pipe/log under /tmp/tdmpc_gpu_warp/mps_bisect), launches the 6
train_vec.py processes (3 seeds each, staggered by 15 s, logs under /tmp/tdmpc_gpu_warp/bisect_logs), polls the
MPS server log every 10 s, then kills all processes and stops the daemon.

Diagnostic for the open issue "Rare GPU fault in long multi-world (vectorised) runs" in
docs/08_known_issues.md, section 8.1 (`illegal memory access` / `unspecified launch failure` in train_vec.py
processes, mostly under CUDA MPS; cause unconfirmed).

Environment: launched from any Python 3; the training processes use $TDMPC_WARP_PYTHON (default: the interpreter running this script).
Needs nvidia-cuda-mps-control (user-level MPS, no root).
Usage: python bench/fault_bisect.py [budget_minutes]      (default 45)
Writes results/equivalence/fault_bisect.json (faulting configuration per round, rounds, minutes); progress
on stdout. Observations summarised in docs/08_known_issues.md, section 8.1 (faults also occurred with the
agent's CUDA graphs disabled).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import os, re, subprocess, sys, time, shutil, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 45
# configurations under test (2 processes each): all CUDA graphs / agent graphs off / env (physics) graph off
CFGS = {'default': [], 'agent_eager': ['cuda_graph_agent=false'], 'env_eager': ['cuda_graph_env=false']}
faults, rounds, t_start = [], 0, time.time()
while time.time() - t_start < budget * 60:
	MPS = '/tmp/tdmpc_gpu_warp/mps_bisect'; shutil.rmtree(MPS, ignore_errors=True)
	env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=f'{MPS}/pipe', CUDA_MPS_LOG_DIRECTORY=f'{MPS}/log')
	os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
	subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env)
	procs = {}
	for name, extra in CFGS.items():
		for k in range(2):
			cmd = [PY, 'src/train_vec.py', 'task=walker-walk', 'num_seeds=3', f'seed={100 * rounds + 10 * k + len(procs)}', 'seed_steps=1000',
				   'eval_freq=4000', 'eval_episodes=2', f'exp_name=bisect_{name}_{k}', 'log_root=/tmp/tdmpc_gpu_warp/bisect_logs'] + extra
			p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			procs[p.pid] = name
			time.sleep(15)
	fault_pid = None
	while time.time() - t_start < budget * 60:
		time.sleep(10)
		log = open(f'{MPS}/log/server.log').read()
		# the MPS server logs the PID of the client that hit the fatal error (all clients die with it)
		m = re.search(r'Client \{PID: (\d+), Context: \d+\} encountered a fatal GPU error', log)
		if m:
			fault_pid = int(m.group(1)); break
		if all(p.poll() is not None for p in [subprocess.Popen(['true'])]) and False:  # (disabled check)
			break
	for pid in procs:
		try: os.kill(pid, 9)
		except ProcessLookupError: pass
	subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env); time.sleep(3)
	rounds += 1
	if fault_pid is None:
		print(f'round {rounds}: no fault until the time budget ran out ({(time.time() - t_start) / 60:.0f} min)', flush=True)
		break
	faults.append(procs.get(fault_pid, f'unknown pid {fault_pid}'))
	print(f'round {rounds}: fault in config "{faults[-1]}" after {(time.time() - t_start) / 60:.0f} min total', flush=True)
json.dump(dict(faults=faults, rounds=rounds, minutes=(time.time() - t_start) / 60),
		  open(os.path.join(ROOT, 'results', 'equivalence', 'fault_bisect.json'), 'w'), indent=1)
print('bisect done', faults, flush=True)
