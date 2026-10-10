"""Run several single-seed training runs in parallel on one GPU, with automatic restart after a crash.

Starts the CUDA MPS daemon (optional, needed for a parallel speed-up), launches one `src/train.py` process per
seed with staggered starts, checkpointing enabled (`checkpoint_minutes`) and, when a run exits with an error
(e.g. the rare GPU fault of docs/08_known_issues.md, which under MPS ends all co-running runs), relaunches it with
`resume=true` so it continues from its last checkpoint. If restarted runs cannot connect to MPS, the MPS daemon is
restarted once no run is alive. Every start / exit / restart is appended to an event log (CSV).
With --mps the launcher guards against the trigger of the MPS fault (docs/08_known_issues.md): ordinary (non-MPS)
CUDA processes on the same GPU. It waits for them before starting (--foreign wait, default; or warn / abort) and logs
any that appear during the run ('foreign_process' events). Stopping the launcher (Ctrl-C / SIGTERM) stops the runs
gracefully (they synchronise the GPU before exiting).

Usage (from the repository root, tdmpc-warp environment):
	python src/launch_parallel.py --task walker-walk --seeds 1 2 3 4 5 --mps
	python src/launch_parallel.py --task dog-run --seeds 1 2 3 4 5 --mps --checkpoint_minutes 20 -- horizon=10
Arguments after `--` are passed to every run (any train.py option). Defaults: CUDA-graph agent + host pacing
(`graph_agent=true pace_host=true`), checkpoint every 30 minutes, logs under ./logs as for train.py, event log
<log_root>/launcher_events_<task>.csv. Without --mps the runs are time-sliced by the GPU (no speed-up over running
them one after another, docs/04).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class MPS:
	"""User-level CUDA MPS control daemon with a short pipe directory (unix-socket path limit ~108 chars)."""

	def __init__(self, directory):
		self.pipe, self.log = os.path.join(directory, 'pipe'), os.path.join(directory, 'log')
		self.env = {'CUDA_MPS_PIPE_DIRECTORY': self.pipe, 'CUDA_MPS_LOG_DIRECTORY': self.log}

	def _ctl(self, cmd=None, daemon=False):
		"""Run nvidia-cuda-mps-control with this daemon's pipe/log directories (`cmd` on stdin, `-d` if daemon); returns the CompletedProcess."""
		env = dict(os.environ, **self.env)
		args = ['nvidia-cuda-mps-control'] + (['-d'] if daemon else [])
		return subprocess.run(args, input=cmd, env=env, capture_output=True, text=True, timeout=60)

	def start(self):
		"""Start the daemon (no-op if one already listens on this pipe directory)."""
		os.makedirs(self.pipe, exist_ok=True); os.makedirs(self.log, exist_ok=True)
		if os.path.exists(os.path.join(self.pipe, 'control')):
			try:
				if self._ctl('get_server_list\n').returncode == 0:
					return
			except subprocess.SubprocessError:
				pass
		self._ctl(daemon=True)
		time.sleep(2)

	def stop(self):
		"""Ask the daemon to quit (waits for its clients to finish)."""
		try:
			self._ctl('quit\n')
		except (subprocess.SubprocessError, OSError):
			pass
		time.sleep(3)

	def restart(self):
		self.stop(); self.start()


def gpu_processes(xml_text=None):
	"""Processes on the GPUs as reported by `nvidia-smi -q -x`: list of dicts (pid, type, name).

	type: 'C' = ordinary CUDA (compute) context, 'M+C' = MPS client, 'G' = graphics, 'C+G' = both (desktop programs)."""
	if xml_text is None:
		try:
			xml_text = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=60).stdout
		except (OSError, subprocess.SubprocessError):
			return []
	try:
		root = ET.fromstring(xml_text)
	except ET.ParseError:
		return []
	out = []
	for gpu in root.findall('gpu'):
		procs = gpu.find('processes')
		for p in (procs.findall('process_info') if procs is not None else []):
			out.append({'pid': int(p.findtext('pid') or -1), 'type': (p.findtext('type') or '').strip(),
						'name': (p.findtext('process_name') or '').strip()})
	return out


def foreign_compute(procs):
	"""Ordinary (non-MPS) CUDA compute processes, excluding the MPS server itself.

	Such processes starting, running or ending next to MPS clients trigger the fatal MPS fault of
	docs/08_known_issues.md; MPS clients ('M+C') and long-lived desktop programs ('G', 'C+G') are not reported."""
	return [p for p in procs if p['type'] == 'C' and 'nvidia-cuda-mps-server' not in p['name']]


class Run:
	"""One single-seed training process (src/train.py) and its restart bookkeeping."""

	def __init__(self, seed, base_args, log_dir, task):
		self.seed, self.base_args, self.task = seed, base_args, task
		self.out_path = os.path.join(log_dir, f'launcher_{task}_s{seed}.out')
		self.proc, self.starts, self.done, self.start_time = None, 0, False, 0.0

	def launch(self, python, env, resume):
		"""Start (or restart with resume=true) the training process; output is appended to out_path."""
		args = [python, '-u', os.path.join(ROOT, 'src', 'train.py'), f'seed={self.seed}'] + self.base_args
		if resume:
			args.append('resume=true')
		out = open(self.out_path, 'a')
		out.write(f'\n===== launch {self.starts + 1} at {time.strftime("%Y-%m-%d %H:%M:%S")} (resume={resume}) =====\n'); out.flush()
		self.proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
		self.starts += 1
		self.start_time = time.time()

	def last_env_step(self, work_dir):
		"""Last env step recorded in the run's train.log (or -1)."""
		try:
			with open(os.path.join(work_dir, 'train.log')) as f:
				lines = f.read().strip().splitlines()
			return int(lines[-1].split(',')[1]) if len(lines) > 1 else -1
		except (OSError, ValueError, IndexError):
			return -1


def main():
	"""Parse the arguments, start MPS (optional), launch the runs and supervise them until done.

	Every 5 s: reports foreign (non-MPS) CUDA processes (with --mps), detects finished / crashed runs and
	relaunches crashed ones with resume=true after --restart_delay (up to --max_restarts); the MPS daemon is
	restarted when restarts keep failing quickly and no run is alive. On the time limit or SIGINT/SIGTERM the
	remaining runs are terminated and MPS is stopped.
	"""
	p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	p.add_argument('--task', required=True)
	p.add_argument('--seeds', type=int, nargs='+', required=True)
	p.add_argument('--mps', action='store_true', help='start CUDA MPS so the runs execute concurrently')
	p.add_argument('--mps_dir', default=f'/tmp/tdmpc_gpu_warp/mps_{os.getuid()}', help='short directory for MPS pipes/logs')
	p.add_argument('--stagger', type=float, default=20.0, help='seconds between run starts')
	p.add_argument('--checkpoint_minutes', type=float, default=30.0)
	p.add_argument('--max_restarts', type=int, default=20, help='per run')
	p.add_argument('--restart_delay', type=float, default=30.0, help='seconds to wait after a crash before restarting')
	p.add_argument('--time_limit_h', type=float, default=0.0, help='stop all runs after this many hours (0 = none)')
	p.add_argument('--no_graph_agent', action='store_true', help='use the original (eager) agent instead')
	p.add_argument('--python', default=sys.executable)
	p.add_argument('--log_root', default=os.path.join(ROOT, 'logs'))
	p.add_argument('--exp_name', default='default')
	p.add_argument('--events', default=None, help='event log CSV (default <log_root>/launcher_events_<task>.csv)')
	p.add_argument('--foreign', choices=['wait', 'warn', 'abort'], default='wait',
				   help='with --mps: what to do if ordinary (non-MPS) CUDA processes use the GPU before the start '
						'(they trigger the fatal MPS fault); during the run they are always logged and reported')
	a, extra = p.parse_known_args()
	extra = [x for x in extra if x != '--']

	os.makedirs(a.log_root, exist_ok=True)
	events_path = a.events or os.path.join(a.log_root, f'launcher_events_{a.task}.csv')
	new = not os.path.exists(events_path)
	ev_f = open(events_path, 'a', newline='')
	ev = csv.writer(ev_f)
	if new:
		ev.writerow(['time', 'unix_time', 'event', 'seed', 'pid', 'returncode', 'env_step', 'alive_runs', 'note'])

	def log(event, run=None, rc='', note=''):
		"""Append one event row to the event-log CSV and print it (alive run count and the run's last env step)."""
		alive = sum(1 for r in runs if r.proc is not None and r.proc.poll() is None)
		step = run.last_env_step(work_dir(run)) if run else ''
		ev.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), f'{time.time():.1f}', event, run.seed if run else '',
					 run.proc.pid if run and run.proc else '', rc, step, alive, note])
		ev_f.flush()
		print(f'[launcher] {time.strftime("%H:%M:%S")} {event} seed={run.seed if run else "-"} rc={rc} step={step} {note}', flush=True)

	agent_args = [] if a.no_graph_agent else ['graph_agent=true', 'pace_host=true']
	base = [f'task={a.task}', f'exp_name={a.exp_name}', f'log_root={a.log_root}',
			f'checkpoint_minutes={a.checkpoint_minutes}'] + agent_args + extra
	work_dir = lambda r: os.path.join(a.log_root, a.task, 'state', a.exp_name, str(r.seed))
	runs = [Run(s, base, a.log_root, a.task) for s in a.seeds]

	env = dict(os.environ)
	mps = None
	if a.mps:  # guard: no ordinary CUDA process may share the GPU with the MPS clients (docs/08)
		while True:
			foreign = foreign_compute(gpu_processes())
			if not foreign:
				break
			names = ', '.join(f"{p['pid']} ({os.path.basename(p['name'].split()[0])})" for p in foreign)
			print(f'[launcher] WARNING: non-MPS CUDA processes on the GPU: {names}. Under MPS they can crash all runs.', flush=True)
			if a.foreign == 'abort':
				sys.exit('[launcher] aborted (--foreign abort): stop them, or route them through MPS')
			if a.foreign == 'warn':
				break
			print('[launcher] waiting for them to finish (--foreign wait) ...', flush=True)
			time.sleep(30)
		mps = MPS(a.mps_dir); mps.start(); env.update(mps.env)
		log('mps_start', note=a.mps_dir)

	stop = {'flag': False}
	def on_signal(signum, frame):
		stop['flag'] = True
	signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)

	t0 = time.time()
	for i, r in enumerate(runs):
		if i:
			time.sleep(a.stagger)
		r.launch(a.python, env, resume=os.path.exists(os.path.join(work_dir(r), 'checkpoint.pt')))
		log('start', r)

	pending = {}  # seed -> earliest restart time
	quick_failures = 0
	seen_foreign, t_guard = set(), 0.0
	while not stop['flag']:
		time.sleep(5)
		if mps is not None and time.time() - t_guard > 30:  # report foreign CUDA processes during the run
			t_guard = time.time()
			for p in foreign_compute(gpu_processes()):
				if p['pid'] not in seen_foreign:
					seen_foreign.add(p['pid'])
					log('foreign_process', note=f"pid {p['pid']} {p['name'][:80]} (non-MPS CUDA process next to MPS clients)")
		if a.time_limit_h and time.time() - t0 > a.time_limit_h * 3600:
			log('time_limit'); break
		for r in runs:
			if r.done or r.proc is None or r.seed in pending:
				continue
			rc = r.proc.poll()
			if rc is None:
				continue
			if rc == 0:
				r.done = True; log('finished', r, rc); continue
			short = time.time() - r.start_time < 120
			quick_failures = quick_failures + 1 if short else 0
			log('crash', r, rc, note='within 120 s of start' if short else '')
			if r.starts > a.max_restarts:
				r.done = True; log('gave_up', r, rc); continue
			pending[r.seed] = time.time() + a.restart_delay
		if all(r.done for r in runs):
			break
		due = [r for r in runs if r.seed in pending and time.time() >= pending[r.seed]]
		if due:
			alive = [r for r in runs if r.proc is not None and r.proc.poll() is None]
			if mps is not None and quick_failures >= 2 and not alive:
				mps.restart(); quick_failures = 0; log('mps_restart')
			for i, r in enumerate(due):
				if i:
					time.sleep(a.stagger)
				del pending[r.seed]
				r.launch(a.python, env, resume=True)
				log('restart', r)

	for r in runs:  # time limit or signal: stop the remaining runs (their last checkpoint is kept)
		if r.proc is not None and r.proc.poll() is None:
			r.proc.terminate()
	for r in runs:
		if r.proc is not None:
			try:
				r.proc.wait(timeout=60)
			except subprocess.TimeoutExpired:
				r.proc.kill()
	if mps is not None:
		mps.stop(); log('mps_stop')
	log('done', note=f'{sum(r.starts for r in runs)} launches, {sum(1 for r in runs if r.done)} runs finished/ended')
	ev_f.close()


if __name__ == '__main__':
	main()
