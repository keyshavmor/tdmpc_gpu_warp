"""Run several TD-MPC runs concurrently on the GPU and measure throughput / contention / memory.

  python bench/multi_run.py --spec SPEC [SPEC ...] [--duration 60] [--mps] [--out results/multi/x.json]
SPEC = mode:task:horizon[:extra1,extra2]  e.g.  gpu:walker-walk:5:graph_agent=true   cpu:walker-walk:5
       (a leading integer multiplier is allowed: 4*gpu:walker-walk:5)

Workers (bench/run_worker.py) build + warm up independently, then start the timed phase together.
Reported: per-run agent steps/s, CPU cores per run, mean GPU utilisation, peak total GPU memory,
per-process VRAM, failures/OOMs. With --mps the CUDA MPS daemon is started in a private pipe
directory (only these workers connect to it) and stopped afterwards.

Run the driver in the `tdmpc-warp` env; gpu workers use the `tdmpc-warp` interpreter, cpu workers (the
original CPU-MuJoCo code) the `tdmpc` interpreter. Further options:
  --sm_pct p    MPS SM partitioning: every client gets CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = p (needs --mps)
  --stagger s   seconds between worker launches (default: all at once; the timed phase starts together anyway)
  --threads n   CPU-thread pinning of cpu workers (0 = original default threading; gpu workers always 1)
Example (2 optimised runs with different horizons sharing the GPU under MPS):
  python bench/multi_run.py --spec gpu:walker-walk:5:graph_agent=true,pace_host=true
      gpu:walker-walk:30:graph_agent=true,pace_host=true --mps --duration 30 --out results/multi/x.json

Building block of bench/contention_study.py, bench/interference_study.py, bench/interference_sameH.py
and bench/partition_study.py, whose results are reported in docs/04_multi_run_capacity.md
(raw tables: docs/results_capacity.md, written by bench/analyze_interference.py).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = {'gpu': os.environ.get('TDMPC_WARP_PYTHON', sys.executable),
	  'cpu': os.environ.get('TDMPC_CPU_PYTHON', os.path.expanduser('~/anaconda3/envs/tdmpc/bin/python'))}


def parse_specs(specs):
	"""Expand 'N*mode:task:H[:extra1,extra2]' strings into a list of run dicts (one per worker)."""
	out = []
	for s in specs:
		mult = 1
		if '*' in s:
			m, s = s.split('*', 1); mult = int(m)
		parts = s.split(':')
		mode, task, h = parts[0], parts[1], int(parts[2])
		extra = parts[3].split(',') if len(parts) > 3 and parts[3] else []
		out += [dict(mode=mode, task=task, horizon=h, extra=extra)] * mult
	return out


def smi(query):
	"""nvidia-smi --query-<query> as a list of rows (list of strings, no units)."""
	o =subprocess.run(['nvidia-smi', f'--query-{query}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
	return [l.split(', ') for l in o.strip().splitlines() if l.strip()]


def cpu_times():
	"""System-wide (total, idle) CPU jiffies from /proc/stat; differences give the system CPU utilisation."""
	with open('/proc/stat') as f:
		v = [float(x) for x in f.readline().split()[1:]]
	return sum(v), v[3] + v[4]  # total, idle+iowait


class Sampler(threading.Thread):
	"""Polls nvidia-smi every 0.5 s: GPU utilisation, total memory used, and the peak VRAM of each worker pid."""

	def __init__(self, pids):
		super().__init__(daemon=True)
		self.pids, self.util, self.mem, self.proc_mem, self.stop = pids, [], [], {}, False

	def run(self):
		"""Every 0.5 s until `stop` is set: append GPU utilisation and total memory used, and update the peak VRAM of each worker pid (nvidia-smi errors are skipped)."""
		while not self.stop:
			try:
				u, m = smi('gpu=utilization.gpu,memory.used')[0]
				self.util.append(float(u)); self.mem.append(float(m))
				for pid, mem in smi('compute-apps=pid,used_memory'):
					if int(pid) in self.pids:
						self.proc_mem[int(pid)] = max(self.proc_mem.get(int(pid), 0), float(mem))
			except Exception:
				pass
			time.sleep(0.5)


def main():
	"""Start MPS (optional), launch the workers, release the start barrier, collect and summarise results."""
	p = argparse.ArgumentParser()
	p.add_argument('--spec', nargs='+', required=True)
	p.add_argument('--duration', type=float, default=60)
	p.add_argument('--mps', action='store_true')
	p.add_argument('--threads', type=int, default=1)
	p.add_argument('--stagger', type=float, default=0.0, help='seconds between worker launches')
	p.add_argument('--out', default=None)
	p.add_argument('--timeout', type=float, default=1800)
	p.add_argument('--sm_pct', type=float, default=None, help='MPS: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per client (SM partitioning)')
	a = p.parse_args()
	runs = parse_specs(a.spec)
	os.makedirs('/tmp/tdmpc_gpu_warp', exist_ok=True)  # short path: unix-socket length limit of MPS pipes
	sync = tempfile.mkdtemp(prefix='mr_', dir=os.environ.get('MR_TMP', '/tmp/tdmpc_gpu_warp'))
	env = dict(os.environ)
	mps_dir = None
	if a.mps:
		# private, user-level MPS daemon: the control daemon and every client find each other through
		# CUDA_MPS_PIPE_DIRECTORY, so only the workers launched with this env attach to it (no root needed,
		# other GPU processes on the machine are unaffected)
		mps_dir = tempfile.mkdtemp(prefix='mps_', dir='/tmp/tdmpc_gpu_warp')  # short path: unix socket limit
		env['CUDA_MPS_PIPE_DIRECTORY'] = os.path.join(mps_dir, 'pipe')
		env['CUDA_MPS_LOG_DIRECTORY'] = os.path.join(mps_dir, 'log')
		os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
		subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True)  # -d: start as daemon
		time.sleep(1)
	if a.sm_pct is not None:
		# read by each MPS client at context creation: caps the fraction of SMs its kernels may occupy
		assert a.mps, '--sm_pct needs --mps'
		env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(a.sm_pct)
	base_mem = float(smi('gpu=memory.used')[0][0])
	procs = []
	try:
		for i, r in enumerate(runs):
			cmd = [PY[r['mode']], os.path.join(HERE, 'run_worker.py'), '--mode', r['mode'], '--task', r['task'],
				   '--horizon', str(r['horizon']), '--sync', sync, '--id', str(i), '--duration', str(a.duration),
				   '--threads', str(a.threads if r['mode'] == 'gpu' else a.threads), '--seed', str(i + 1)]
			if r['extra']:
				cmd += ['--extra'] + r['extra']
			procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(os.path.join(sync, f'err.{i}'), 'w')))
			if a.stagger:
				time.sleep(a.stagger)
		t_launch = time.time()
		# start barrier: wait until every worker is built + warmed up (ready.<i>) or has died
		while True:
			ready = sum(os.path.exists(os.path.join(sync, f'ready.{i}')) for i in range(len(runs)))
			dead = [i for i, pr in enumerate(procs) if pr.poll() is not None and not os.path.exists(os.path.join(sync, f'ready.{i}'))]
			if ready + len(dead) >= len(runs) or time.time() - t_launch > a.timeout:
				break
			time.sleep(0.2)
		setup_s = time.time() - t_launch
		sampler = Sampler({pr.pid for pr in procs}); sampler.start()
		time.sleep(1.0)
		c_tot0, c_idle0 = cpu_times()
		open(os.path.join(sync, 'go'), 'w').close()  # release all workers into the timed phase at once
		for pr in procs:
			try:
				pr.wait(timeout=a.duration + a.timeout)
			except subprocess.TimeoutExpired:
				pr.kill()
		c_tot1, c_idle1 = cpu_times()
		sampler.stop = True
		results = []
		for i in range(len(runs)):
			f = os.path.join(sync, f'result.{i}.json')
			if os.path.exists(f):
				results.append(json.load(open(f)))
			else:
				err = open(os.path.join(sync, f'err.{i}')).read()[-800:]
				results.append(dict(id=str(i), ok=False, oom='out of memory' in err.lower(), error=err))
		ok = [r for r in results if r.get('ok')]
		for r in ok:  # peak nvidia-smi VRAM of the worker's process during the timed phase
			r['process_vram_mib'] = sampler.proc_mem.get(r.get('pid'), float('nan'))
		summary = dict(
			specs=a.spec, mps=a.mps, sm_pct=a.sm_pct, n_runs=len(runs), n_ok=len(ok), n_oom=sum(bool(r.get('oom')) for r in results),
			duration_s=a.duration, setup_s=setup_s,
			steps_per_s_each=[r['steps_per_s'] for r in ok],
			steps_per_s_total=sum(r['steps_per_s'] for r in ok),
			cpu_cores_each=[r['cpu_cores'] for r in ok],
			system_cpu_util=1 - (c_idle1 - c_idle0) / max(1e-9, c_tot1 - c_tot0),
			gpu_util_mean=sum(sampler.util) / max(1, len(sampler.util)),
			gpu_mem_peak_mib=max(sampler.mem) if sampler.mem else float('nan'), gpu_mem_baseline_mib=base_mem,
			runs=results)
		print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}), flush=True)
		if a.out:
			os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
			json.dump(summary, open(a.out, 'w'), indent=1)
	finally:
		for pr in procs:
			if pr.poll() is None:
				pr.kill()
		if a.mps:  # stop the private daemon (same pipe directory) and remove its directories
			subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env)
			time.sleep(1)
			shutil.rmtree(mps_dir, ignore_errors=True)
		shutil.rmtree(sync, ignore_errors=True)


if __name__ == '__main__':
	main()
