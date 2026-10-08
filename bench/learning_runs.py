"""Launch the learning-equivalence runs (up to 100k env steps each) as detached processes.

Job set (see jobs()):
  vec fp32 (5 seeds) vs vec fp32 + batched_update + skip_substep_sensors (5 seeds): walker-walk cartpole-swingup
  CPU original (bench/train_cpu_baseline.py, `tdmpc` env, 1 seed):                  walker-walk cartpole-swingup
  GPU faithful (src/train.py, original eager agent, 1 seed):                         cartpole-swingup
  vec bf16 / fp16 autocast (3 seeds, seeds 101-103):                                 walker-walk
  reference: the 5-seed curves of the original paper code in ../tdmpc/results/*.csv
GPU jobs run src/train_vec.py / src/train.py with the `tdmpc-warp` interpreter and share one CUDA MPS
daemon (private pipe dir /tmp/tdmpc_gpu_warp/mps_learning); the CPU anchors run the original code with the
`tdmpc` interpreter. Every job gets max_env_steps=100000 and log_root=results/learning, i.e. logs in
results/learning/<task>/state/<exp_name>/<seed>/eval.log; stdout/stderr in results/learning/logs/<job>.log;
pids in results/learning/pids.json.

  python bench/learning_runs.py start | status | stop
  python bench/learning_runs.py start_gpu          # restart only the GPU jobs under a fresh MPS daemon
  python bench/learning_runs.py start_gpu_nomps    # restart only the GPU jobs without MPS (time-sliced)

Evaluated by bench/analyze_learning.py -> docs/results_learning.md; reported in docs/06_optimisations.md
(learning curves of the exact optimisations) and docs/07_precision.md (7.4, bf16 / fp16 accuracy).
Under MPS a fatal GPU error in one client takes down all clients (docs/04_multi_run_capacity.md, 4.7),
hence the restart commands.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import json, os, subprocess, sys, time, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LR = os.path.join(ROOT, 'results', 'learning')
PYW = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
PYC = os.environ.get('TDMPC_CPU_PYTHON', os.path.expanduser('~/anaconda3/envs/tdmpc/bin/python'))
MPS = '/tmp/tdmpc_gpu_warp/mps_learning'  # short path: MPS pipes are unix sockets (~108-character path limit)
STEPS = 100000  # early stop (max_env_steps) of every job
# wider task lists, currently not referenced by the lean jobs() set below
FP32_TASKS = ['walker-walk', 'cartpole-swingup', 'cup-catch', 'finger-spin', 'reacher-easy', 'cheetah-run']
PREC_TASKS = ['walker-walk', 'cartpole-swingup', 'cup-catch']


def jobs():
	"""Lean set: learning-mechanism A/B for every optimisation + precision accuracy + original-code anchors."""
	J = []
	OPT = ['batched_update=true', 'skip_substep_sensors=true']
	for t in ('walker-walk', 'cartpole-swingup'):
		# exact-maths GPU agent (vectorised, CUDA graphs) vs + batched update + sensor skipping: 5 seeds each
		J.append((f'vec_fp32_{t}', [PYW, 'src/train_vec.py', f'task={t}', 'num_seeds=5', 'seed=1', 'exp_name=vec_fp32'], 'gpu'))
		J.append((f'vec_fp32_opt_{t}', [PYW, 'src/train_vec.py', f'task={t}', 'num_seeds=5', 'seed=1', 'exp_name=vec_fp32_opt'] + OPT, 'gpu'))
		# anchors: the original agent on the GPU env, and the untouched original code on CPU MuJoCo
		if t == 'cartpole-swingup':  # eager original agent on the GPU env (the exact vec agent covers walker)
			J.append((f'gpu_faithful_{t}', [PYW, 'src/train.py', f'task={t}', 'seed=1', 'exp_name=gpu_faithful'], 'gpu'))
		J.append((f'cpu_original_{t}', [PYC, 'bench/train_cpu_baseline.py', f'task={t}', 'seed=1', 'exp_name=cpu_original'], 'cpu'))
	for prec in ('bf16', 'fp16'):  # reduced-precision learning accuracy (changes numerics)
		J.append((f'vec_{prec}_walker-walk', [PYW, 'src/train_vec.py', 'task=walker-walk', 'num_seeds=3', 'seed=101', f'precision={prec}', f'exp_name=vec_{prec}'], 'gpu'))
	return J


def start():
	"""Start the MPS daemon and launch all jobs detached (new session), 3 s apart; record their pids."""
	os.makedirs(os.path.join(LR, 'logs'), exist_ok=True)
	env = dict(os.environ)
	# GPU jobs get the MPS pipe/log directories (they attach to the private daemon); CPU jobs the plain env
	envg =dict(env, CUDA_MPS_PIPE_DIRECTORY=os.path.join(MPS, 'pipe'), CUDA_MPS_LOG_DIRECTORY=os.path.join(MPS, 'log'))
	os.makedirs(envg['CUDA_MPS_PIPE_DIRECTORY'], exist_ok=True); os.makedirs(envg['CUDA_MPS_LOG_DIRECTORY'], exist_ok=True)
	subprocess.run(['nvidia-cuda-mps-control', '-d'], env=envg)
	pids = {}
	for name, cmd, kind in jobs():
		cmd = cmd + [f'max_env_steps={STEPS}', f'log_root={LR}']
		log = open(os.path.join(LR, 'logs', f'{name}.log'), 'w')
		p = subprocess.Popen(cmd, cwd=ROOT, env=envg if kind == 'gpu' else env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		pids[name] = p.pid
		time.sleep(3)
	json.dump(pids, open(os.path.join(LR, 'pids.json'), 'w'), indent=1)
	print('started', len(pids))


def start_gpu_nomps(stagger=10):
	"""(Re)start the GPU jobs WITHOUT MPS (time-sliced, fault-isolated processes)."""
	import shutil
	envg = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=os.path.join(MPS, 'pipe'))
	subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=envg)  # stop the daemon
	time.sleep(2)
	# no CUDA_MPS_* variables: the processes create ordinary CUDA contexts (no MPS daemon involved)
	env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS')}
	pids = {k: v for k, v in json.load(open(os.path.join(LR, 'pids.json'))).items() if k.startswith('cpu_')}
	for name, cmd, kind in jobs():
		if kind != 'gpu':
			continue
		exp = [c.split('=', 1)[1] for c in cmd if c.startswith('exp_name=')][0]
		task = [c.split('=', 1)[1] for c in cmd if c.startswith('task=')][0]
		shutil.rmtree(os.path.join(LR, task, 'state', exp), ignore_errors=True)  # discard the partial run's logs
		cmd = cmd + [f'max_env_steps={STEPS}', f'log_root={LR}']
		log = open(os.path.join(LR, 'logs', f'{name}.log'), 'w')
		p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		pids[name] = p.pid
		time.sleep(stagger)
	json.dump(pids, open(os.path.join(LR, 'pids.json'), 'w'), indent=1)
	print('restarted gpu jobs without MPS')


def start_gpu(stagger=25):
	"""(Re)start only the GPU jobs (staggered start under one MPS daemon); CPU jobs keep running."""
	import shutil
	envg = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=os.path.join(MPS, 'pipe'), CUDA_MPS_LOG_DIRECTORY=os.path.join(MPS, 'log'))
	subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=envg)
	time.sleep(2)
	shutil.rmtree(MPS, ignore_errors=True)  # fresh pipe / log directories for the new daemon
	os.makedirs(envg['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(envg['CUDA_MPS_LOG_DIRECTORY'])
	subprocess.run(['nvidia-cuda-mps-control', '-d'], env=envg)
	pids = json.load(open(os.path.join(LR, 'pids.json')))
	for name, cmd, kind in jobs():
		if kind != 'gpu':
			continue
		exp = [c.split('=', 1)[1] for c in cmd if c.startswith('exp_name=')][0]
		task = [c.split('=', 1)[1] for c in cmd if c.startswith('task=')][0]
		shutil.rmtree(os.path.join(LR, task, 'state', exp), ignore_errors=True)
		cmd = cmd + [f'max_env_steps={STEPS}', f'log_root={LR}']
		log = open(os.path.join(LR, 'logs', f'{name}.log'), 'w')
		p = subprocess.Popen(cmd, cwd=ROOT, env=envg, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		pids[name] = p.pid
		time.sleep(stagger)
	json.dump(pids, open(os.path.join(LR, 'pids.json'), 'w'), indent=1)
	print('restarted gpu jobs')


def status():
	"""Print, per job, whether its process is alive and the last line of its log; return the number alive."""
	pids =json.load(open(os.path.join(LR, 'pids.json')))
	alive = {n: os.path.exists(f'/proc/{p}') for n, p in pids.items()}
	for n in pids:
		last = open(os.path.join(LR, 'logs', f'{n}.log')).read().strip().splitlines()[-1:] or ['']
		print(f'{"RUN " if alive[n] else "done"} {n:32s} {last[0][:110]}')
	print('alive', sum(alive.values()), '/', len(alive))
	return sum(alive.values())


def stop():
	"""Shut down the learning runs' MPS daemon (does not kill the training processes themselves)."""
	envg =dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=os.path.join(MPS, 'pipe'))
	subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=envg)


if __name__ == '__main__':
	{'start': start, 'start_gpu': start_gpu, 'start_gpu_nomps': start_gpu_nomps, 'status': status, 'stop': stop}[sys.argv[1]]()
