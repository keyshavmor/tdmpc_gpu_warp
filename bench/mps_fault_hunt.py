"""Measure the rate of the rare MPS GPU fault for one configuration ("variant"), for root-cause bisection.

Each round starts a private CUDA MPS daemon, launches N identical training processes of the variant (staggered),
and watches the MPS server log for a fatal GPU error. On a fault it records the time since the round started and
since the most recent process start, the MPS-reported client and the kernel's Xid record (pid of the faulting
channel, fault type), kills all processes, stops MPS and starts the next round, until the time budget is used.
Rounds without a fault end at --round_minutes. Output: fault count, exposure (process-hours) and the per-round
timeline, from which the fault rate per process-hour and its correlation with process start-ups follow.

Variants: cfg overrides after `--` (e.g. cuda_graph_env=false) and/or diagnostic switches of
bench/fault_variant_wrapper.py given with --diag (no_conditional, mempool_keep, gc_off, sync_gc).
Mixed mode (--non_mps K): K additional processes of the same kind run WITHOUT MPS (ordinary CUDA contexts) on the
same GPU at the same time, optionally restarted every --non_mps_restart s - to test whether faults are triggered by
MPS clients and non-MPS CUDA processes sharing the GPU.
Churn mode (--churn_interval N): besides the N steady processes, an extra short-lived process of the same kind is
started every N s and killed after --churn_life s, to test whether faults are triggered by MPS clients starting /
exiting while others run (the faults seen so far clustered around process start-ups).

Environment: tdmpc-warp; needs nvidia-cuda-mps-control and `journalctl -k` (read access to the kernel log).
Usage:
	python bench/mps_fault_hunt.py --variant baseline --budget_min 60
	python bench/mps_fault_hunt.py --variant no_conditional --diag no_conditional --budget_min 60
	python bench/mps_fault_hunt.py --variant env_eager --budget_min 60 -- cuda_graph_env=false
Writes results/fault_hunt/<variant>.json. Results: docs/08_known_issues.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = os.environ.get('TDMPC_WARP_PYTHON', sys.executable)
DIAG = {'no_conditional': 'TDMPC_DIAG_NO_CONDITIONAL', 'mempool_keep': 'TDMPC_DIAG_MEMPOOL_KEEP',
		'gc_off': 'TDMPC_DIAG_GC_OFF', 'sync_gc': 'TDMPC_DIAG_SYNC_GC'}

p = argparse.ArgumentParser()
p.add_argument('--variant', required=True)
p.add_argument('--diag', nargs='*', default=[], choices=list(DIAG))
p.add_argument('--script', default='src/train_vec.py', help='src/train_vec.py (vectorised), src/train.py (single seed) or bench/env_stress.py (physics only)')
p.add_argument('--task', default='walker-walk')
p.add_argument('--nproc', type=int, default=6)
p.add_argument('--num_seeds', type=int, default=3, help='seeds per process (train_vec.py only)')
p.add_argument('--stagger', type=float, default=15.0)
p.add_argument('--round_minutes', type=float, default=20.0)
p.add_argument('--budget_min', type=float, default=60.0)
p.add_argument('--churn_interval', type=float, default=0.0, help='start an extra short-lived process every N s (0 = off)')
p.add_argument('--churn_life', type=float, default=60.0, help='lifetime of a churn process in seconds')
p.add_argument('--churn_script', default=None, help='script of the churn processes (default: --script)')
p.add_argument('--churn_end', choices=['kill', 'graceful'], default='kill', help='churn processes end by SIGKILL, or exit normally after synchronising (env_stress.py / cuda_ping.py max_seconds)')
p.add_argument('--non_mps', type=int, default=0, help='additional processes started WITHOUT MPS (own CUDA context) alongside the MPS clients')
p.add_argument('--non_mps_restart', type=float, default=0.0, help='restart each non-MPS process every N s (0 = keep running)')
p.add_argument('--non_mps_end', choices=['kill', 'graceful'], default='kill', help='how a non-MPS process ends at restart: SIGKILL, or a normal exit after synchronising (bench/env_stress.py max_seconds)')
p.add_argument('--non_mps_script', default=None, help='script of the non-MPS processes (default: --script)')
p.add_argument('--non_mps_diag', nargs='*', default=None, choices=list(DIAG), help='diagnostic switches for the non-MPS processes only (default: same as --diag)')
p.add_argument('--non_mps_extra', nargs='*', default=[], help='extra key=value options for the non-MPS processes only')
a, extra = p.parse_known_args()
extra = [x for x in extra if x != '--']
out_dir = os.path.join(ROOT, 'results', 'fault_hunt'); os.makedirs(out_dir, exist_ok=True)
MPS = '/tmp/tdmpc_gpu_warp/mps_hunt'
rounds, t_budget = [], time.time()


def xid_since(t0):
	"""Kernel Xid lines since unix time t0 (empty if journalctl is unavailable)."""
	try:
		since = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
		out = subprocess.run(['journalctl', '-k', '--since', since], capture_output=True, text=True, timeout=60).stdout
		return [l for l in out.splitlines() if 'Xid' in l]
	except (OSError, subprocess.SubprocessError):
		return []


_Popen = subprocess.Popen
t_started, ended = {}, {}
def _tracked_popen(*args, **kw):
	p = _Popen(*args, **kw)
	if args and isinstance(args[0], list) and 'fault_variant_wrapper.py' in ' '.join(args[0]):
		t_started[p.pid] = time.time()
	return p
subprocess.Popen = _tracked_popen
_kill = os.kill
def _tracked_kill(pid, sig):
	ended.setdefault(pid, time.time())
	return _kill(pid, sig)
os.kill = _tracked_kill

while time.time() - t_budget < a.budget_min * 60:
	shutil.rmtree(MPS, ignore_errors=True)
	env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=f'{MPS}/pipe', CUDA_MPS_LOG_DIRECTORY=f'{MPS}/log')
	for k in a.diag:
		env[DIAG[k]] = '1'
	os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
	subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env)
	t_round = time.time(); procs, starts = {}, []
	r = len(rounds)
	fault = None
	for i in range(a.nproc):
		seed = 1000 * r + 10 * i + 1
		args = [f'task={a.task}', f'seed={seed}', f'exp_name=hunt_{a.variant}', 'log_root=/tmp/tdmpc_gpu_warp/hunt_logs',
				'eval_freq=10000', 'eval_episodes=1', 'max_env_steps=10000000'] + extra
		if a.script.endswith('src/train.py'):
			args += ['graph_agent=true', 'pace_host=true']  # recommended single-seed configuration
		else:
			args.append(f'num_seeds={a.num_seeds}')  # train_vec.py / env_stress.py: worlds (seeds) per process
		pr = subprocess.Popen([PY, os.path.join('bench', 'fault_variant_wrapper.py'), a.script] + args, cwd=ROOT, env=env,
							  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		procs[pr.pid] = i; starts.append(time.time() - t_round)
		# watch for a fault during the staggered start as well
		t_wait = time.time()
		while time.time() - t_wait < a.stagger:
			time.sleep(1)
			log = open(f'{MPS}/log/server.log').read() if os.path.exists(f'{MPS}/log/server.log') else ''
			if 'encountered a fatal GPU error' in log:
				break
		if 'encountered a fatal GPU error' in (open(f'{MPS}/log/server.log').read() if os.path.exists(f'{MPS}/log/server.log') else ''):
			break
	env_plain = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS') and not k.startswith('TDMPC_DIAG')}
	for k in (a.diag if a.non_mps_diag is None else a.non_mps_diag):
		env_plain[DIAG[k]] = '1'
	plain, nplain = {}, 0
	def start_plain():
		global nplain
		nplain += 1
		pargs = [f'task={a.task}', f'seed={1000 * r + 800 + nplain}', f'exp_name=hunt_{a.variant}_plain',
				 'log_root=/tmp/tdmpc_gpu_warp/hunt_logs', 'eval_freq=10000', 'eval_episodes=1', 'max_env_steps=10000000'] + extra + a.non_mps_extra
		pscript = a.non_mps_script or a.script
		pargs += ['graph_agent=true', 'pace_host=true'] if pscript.endswith('src/train.py') else [f'num_seeds={a.num_seeds}']
		if a.non_mps_end == 'graceful' and a.non_mps_restart > 0:
			pargs.append(f'max_seconds={a.non_mps_restart}')  # env_stress.py exits by itself after synchronising
		pp = subprocess.Popen([PY, os.path.join('bench', 'fault_variant_wrapper.py'), pscript] + pargs, cwd=ROOT, env=env_plain,
							  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		plain[pp.pid] = time.time(); procs[pp.pid] = f'non_mps{nplain}'; starts.append(time.time() - t_round)
	for _ in range(a.non_mps):
		start_plain(); time.sleep(a.stagger)
	churn, next_churn, nchurn = {}, time.time() + a.churn_interval, 0
	while fault is None and time.time() - t_round < a.round_minutes * 60 and time.time() - t_budget < a.budget_min * 60:
		if a.non_mps_restart > 0:
			for pid, t_s in list(plain.items()):
				ended_now = subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode != 0
				if a.non_mps_end == 'kill' and time.time() - t_s > a.non_mps_restart:
					try: os.kill(pid, 9)
					except ProcessLookupError: pass
					del plain[pid]; start_plain()
				elif a.non_mps_end == 'graceful' and ended_now:  # exited normally (max_seconds): start the next one
					ended.setdefault(pid, time.time()); del plain[pid]; start_plain()
		if a.churn_interval > 0:
			for pid, t_s in list(churn.items()):  # end churn processes after their lifetime
				if a.churn_end == 'kill' and time.time() - t_s > a.churn_life:
					try: os.kill(pid, 9)
					except ProcessLookupError: pass
					del churn[pid]
				elif a.churn_end == 'graceful' and subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode != 0:
					ended.setdefault(pid, time.time()); del churn[pid]  # exited by itself after max_seconds
			if time.time() >= next_churn:
				nchurn += 1
				cargs = [f'task={a.task}', f'seed={1000 * r + 500 + nchurn}', f'exp_name=hunt_{a.variant}_churn',
						 'log_root=/tmp/tdmpc_gpu_warp/hunt_logs', 'eval_freq=10000', 'eval_episodes=1', 'max_env_steps=10000000'] + extra
				cscript = a.churn_script or a.script
				cargs += ['graph_agent=true', 'pace_host=true'] if cscript.endswith('src/train.py') else [f'num_seeds={a.num_seeds}']
				if a.churn_end == 'graceful':
					cargs.append(f'max_seconds={a.churn_life}')
				pc = subprocess.Popen([PY, os.path.join('bench', 'fault_variant_wrapper.py'), cscript] + cargs, cwd=ROOT, env=env,
									  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
				churn[pc.pid] = time.time(); procs[pc.pid] = f'churn{nchurn}'; starts.append(time.time() - t_round)
				next_churn = time.time() + a.churn_interval
		log = open(f'{MPS}/log/server.log').read() if os.path.exists(f'{MPS}/log/server.log') else ''
		m = re.search(r'\[([0-9-]+ [0-9:.]+) Server \d+\] Client \{PID: (\d+), Context: \d+\} encountered a fatal GPU error', log)
		if m:
			t_f = time.mktime(time.strptime(m.group(1).split('.')[0], '%Y-%m-%d %H:%M:%S')) - t_round
			fault = dict(t_since_round_s=t_f, mps_client_index=procs.get(int(m.group(2))),
						 t_since_last_start_s=t_f - max([s for s in starts if s <= t_f] or [0]))
			break
		for pid in list(plain):
			if a.non_mps_end == 'graceful' and a.non_mps_restart > 0:
				break  # ends are expected; faults show up in the MPS log / Xid records
			if subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode != 0:
				fault = dict(t_since_round_s=time.time() - t_round, note_non_mps=f'non-MPS process {procs[pid]} died', mps_client_index=procs[pid])
				break
		if fault is not None:
			break
		steady = [pid for pid, i in procs.items() if isinstance(i, int)]
		dead = [pid for pid in steady if subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode != 0]
		if len(dead) == len(steady):
			fault = dict(t_since_round_s=time.time() - t_round, note='all processes exited without an MPS fault record')
			break
		time.sleep(2)
	t_end = time.time()
	for pid in procs:
		try: os.kill(pid, 9)
		except ProcessLookupError: pass
	time.sleep(2)
	subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env); time.sleep(3)
	alive_s = sum(max(0.0, (t_end - t_round) - s) for s in starts)  # upper bound; exact lifetimes below
	ended.update({pid: t_end for pid in procs if pid not in ended})
	lifetimes = [ended[pid] - t_started[pid] for pid in procs if pid in t_started]
	rec = dict(round=r, duration_s=t_end - t_round, process_hours=sum(lifetimes) / 3600, lifetimes_s=lifetimes, starts_s=starts,
			   n_starts=len(starts), non_mps=a.non_mps, fault=fault,
			   xid=xid_since(t_round))  # every round: also catches faults in non-MPS processes
	if fault and rec['xid']:
		m = re.search(r'pid=(\d+)', rec['xid'][0])
		if m:
			rec['fault']['xid_process_index'] = procs.get(int(m.group(1)))
	rounds.append(rec)
	print(f'[{a.variant}] round {r}: ' + (f"FAULT at {fault['t_since_round_s']:.0f} s ({fault.get('t_since_last_start_s', float('nan')):.0f} s after a start)"
		  if fault else f'no fault in {rec["duration_s"] / 60:.1f} min'), flush=True)
	json.dump(dict(variant=a.variant, diag=a.diag, extra=extra, script=a.script, task=a.task, nproc=a.nproc, num_seeds=a.num_seeds,
				   faults=sum(1 for x in rounds if x['fault'] and 'note' not in x['fault']), process_hours=sum(x['process_hours'] for x in rounds),
				   minutes=(time.time() - t_budget) / 60, rounds=rounds), open(os.path.join(out_dir, f'{a.variant}.json'), 'w'), indent=1)
print(f'[{a.variant}] done', flush=True)
