"""Tests of the MPS safeguards: foreign-process guard of src/launch_parallel.py and the graceful stop of src/train.py.

1. Parser: on a recorded `nvidia-smi -q -x` snapshot taken during the mixed MPS / non-MPS fault experiment
   (bench/fixtures/nvidia_smi_mixed_mps.xml) the guard must report exactly the two ordinary CUDA processes and
   ignore the MPS server, the MPS clients and the desktop programs.
2. Guard, live: an ordinary CUDA process (bench/cuda_ping.py, 60 s) runs; the launcher is started with --mps and
   --foreign wait and must not start any run before that process has ended (no MPS client exists yet, so this
   test cannot itself trigger the fault).
3. Graceful stop: a training run receives SIGTERM; it must exit with code 0 after synchronising the GPU, and a
   resume=true run must continue from its checkpoint.

Environment: tdmpc-warp (a CUDA GPU not used by other jobs during the test).
Usage: python bench/test_mps_guard.py
Writes results/fault_hunt/mps_guard_test.json. Results: docs/08_known_issues.md, docs/USER_GUIDE.md (section 4.4).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os, re, shutil, signal, subprocess, sys, tempfile, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from launch_parallel import gpu_processes, foreign_compute

res = {}
# 1. parser on the recorded snapshot
procs = gpu_processes(open(os.path.join(ROOT, 'bench', 'fixtures', 'nvidia_smi_mixed_mps.xml')).read())
found = foreign_compute(procs)
res['parser'] = {'processes': len(procs), 'types': sorted({p['type'] for p in procs}),
				 'foreign_found': [(p['pid'], p['type']) for p in found],
				 'pass': [p['pid'] for p in found] == [1850316, 1850710]}

# 2. live guard: foreign process first, then the launcher must wait for it
tmp = tempfile.mkdtemp(prefix='guard_test_')
ping = subprocess.Popen([sys.executable, os.path.join(ROOT, 'bench', 'cuda_ping.py'), 'max_seconds=60'], cwd=ROOT)
t_ping = time.time()
ping_end = {}
import threading
threading.Thread(target=lambda: (ping.wait(), ping_end.setdefault('t', time.time())), daemon=True).start()  # exact exit time
for _ in range(60):  # wait until the ping process shows up on the GPU
	if any(p['pid'] == ping.pid for p in foreign_compute(gpu_processes())):
		break
	time.sleep(1)
seen_before = any(p['pid'] == ping.pid for p in foreign_compute(gpu_processes()))
lau = subprocess.Popen([sys.executable, os.path.join(ROOT, 'src', 'launch_parallel.py'), '--task', 'cartpole-swingup', '--seeds', '1',
						'--mps', '--foreign', 'wait', '--checkpoint_minutes', '0', '--log_root', tmp, '--exp_name', 'guard',
						'--', 'max_env_steps=2000', 'eval_freq=100000', 'eval_episodes=1'],
					   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out, _ = lau.communicate(timeout=900)
ping.wait()
time.sleep(0.5)
t_ping_end = ping_end['t']  # when the foreign process actually exited
ev = list(open(os.path.join(tmp, 'launcher_events_cartpole-swingup.csv')).read().splitlines())[1:]
t_start = [float(l.split(',')[1]) for l in ev if ',start,' in l]
res['guard_live'] = {'foreign_visible': seen_before, 'warned': 'non-MPS CUDA processes on the GPU' in out,
					 'run_started_after_foreign_ended': bool(t_start) and t_start[0] >= t_ping_end,
					 'wait_s': (t_start[0] - t_ping) if t_start else None,
					 'launcher_exit': lau.returncode}
res['guard_live']['pass'] = all(res['guard_live'][k] is True for k in ('foreign_visible', 'warned', 'run_started_after_foreign_ended')) and lau.returncode == 0

# 3. graceful stop + resume (no MPS)
args = [sys.executable, '-u', 'src/train.py', 'task=cartpole-swingup', 'seed=2', 'graph_agent=true', 'pace_host=true',
		'checkpoint_minutes=0.05', 'max_env_steps=6000', 'eval_freq=100000', 'eval_episodes=1', f'log_root={tmp}', 'exp_name=stop']
o1 = os.path.join(tmp, 'stop1.out')
pr = subprocess.Popen(args, cwd=ROOT, stdout=open(o1, 'w'), stderr=subprocess.STDOUT)
while 'Checkpoint at env step' not in open(o1).read():
	assert pr.poll() is None, open(o1).read()[-1500:]
	time.sleep(0.5)
time.sleep(2)
pr.send_signal(signal.SIGTERM); rc1 = pr.wait(timeout=120)
txt1 = open(o1).read()
o2 = os.path.join(tmp, 'stop2.out')
rc2 = subprocess.run(args + ['resume=true'], cwd=ROOT, stdout=open(o2, 'w'), stderr=subprocess.STDOUT).returncode
txt2 = open(o2).read()
res['graceful_stop'] = {'exit_code': rc1, 'synchronised_message': 'GPU synchronised' in txt1,
						'resumed': 'Resumed from' in txt2, 'resume_exit_code': rc2,
						'completed': 'Training completed successfully' in txt2}
res['graceful_stop']['pass'] = rc1 == 0 and res['graceful_stop']['synchronised_message'] and res['graceful_stop']['resumed'] and rc2 == 0
res['pass'] = all(res[k]['pass'] for k in ('parser', 'guard_live', 'graceful_stop'))
print(json.dumps(res, indent=1)); print('PASS' if res['pass'] else 'FAIL')
os.makedirs(os.path.join(ROOT, 'results', 'fault_hunt'), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, 'results', 'fault_hunt', 'mps_guard_test.json'), 'w'), indent=1)
shutil.rmtree(tmp, ignore_errors=True)
