"""Empirical packing test: how many TD-MPC runs fit on the GPU before CUDA OOM.

Launches runs one after another; each run builds env + agent + full replay buffer and performs
warm-up plan/env/update steps (so its peak memory is allocated and held by the caching allocator),
then holds its memory. The next run is launched only once the previous one is ready. Stops at the
first run that fails with CUDA out-of-memory (or at --max).
  python bench/pack_test.py --cycle mode:task:H[:extra,...] [mode:task:H ...] [--mps] [--max 200] --out f.json
The --cycle specs are launched round-robin (e.g. several horizons -> mixed-horizon packing).
Spec format as in bench/multi_run.py (without the N* multiplier), e.g.
  python bench/pack_test.py --cycle gpu:walker-walk:5:graph_agent=true,pace_host=true --mps --out results/packing/x.json

Workers are bench/run_worker.py in packing mode (--duration 0): each writes its result after warm-up and
holds its memory until the driver creates the `stop` file. Run the driver in the `tdmpc-warp` env (cpu
specs use the `tdmpc` interpreter). Recorded per run: nvidia-smi process VRAM and total GPU memory used
after it started; for the failing run: OOM flag, error and memory in use. Driven by
bench/packing_study.py; the packing tests validate the computed memory limits of
docs/04_multi_run_capacity.md (tables in docs/results_memory.md, "Empirical packing tests").

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import sys
import argparse, json, os, shutil, subprocess, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = {'gpu': os.environ.get('TDMPC_WARP_PYTHON', sys.executable),
	  'cpu': os.environ.get('TDMPC_CPU_PYTHON', os.path.expanduser('~/anaconda3/envs/tdmpc/bin/python'))}


def smi(q):
	"""nvidia-smi --query-<q> as a list of rows (list of strings, no units)."""
	o =subprocess.run(['nvidia-smi', f'--query-{q}', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
	return [l.split(', ') for l in o.strip().splitlines() if l.strip()]


def main():
	"""Launch held runs one at a time until the first failure (or --max) and write the packing summary."""
	p = argparse.ArgumentParser()
	p.add_argument('--cycle', nargs='+', required=True)
	p.add_argument('--mps', action='store_true')
	p.add_argument('--max', type=int, default=200)
	p.add_argument('--out', required=True)
	a = p.parse_args()
	specs = []
	for s in a.cycle:
		parts = s.split(':')
		specs.append(dict(mode=parts[0], task=parts[1], horizon=int(parts[2]),
						  extra=parts[3].split(',') if len(parts) > 3 and parts[3] else []))
	os.makedirs('/tmp/tdmpc_gpu_warp', exist_ok=True)  # short path: unix-socket length limit of MPS pipes
	sync = tempfile.mkdtemp(prefix='pk_', dir='/tmp/tdmpc_gpu_warp')
	env = dict(os.environ)
	mps_dir = None
	if a.mps:
		# private user-level MPS daemon; only workers started with this env (same pipe dir) connect to it
		mps_dir = tempfile.mkdtemp(prefix='mps_', dir='/tmp/tdmpc_gpu_warp')
		env['CUDA_MPS_PIPE_DIRECTORY'] = os.path.join(mps_dir, 'pipe'); env['CUDA_MPS_LOG_DIRECTORY'] = os.path.join(mps_dir, 'log')
		os.makedirs(env['CUDA_MPS_PIPE_DIRECTORY']); os.makedirs(env['CUDA_MPS_LOG_DIRECTORY'])
		subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True); time.sleep(1)
	base = float(smi('gpu=memory.used')[0][0]); total = float(smi('gpu=memory.total')[0][0])
	procs, runs, failure = [], [], None
	try:
		for i in range(a.max):
			sp = specs[i % len(specs)]
			cmd = [PY[sp['mode']], os.path.join(HERE, 'run_worker.py'), '--mode', sp['mode'], '--task', sp['task'],
				   '--horizon', str(sp['horizon']), '--sync', sync, '--id', str(i), '--duration', '0', '--seed', str(i + 1)]
			if sp['mode'] == 'cpu':
				cmd += ['--threads', '0']
			if sp['extra']:
				cmd += ['--extra'] + sp['extra']
			pr = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(os.path.join(sync, f'err.{i}'), 'w'))
			procs.append(pr)
			# sequential: the next run starts only after this one holds its peak memory (or has died)
			while not os.path.exists(os.path.join(sync, f'ready.{i}')) and pr.poll() is None:
				time.sleep(0.2)
			res_f = os.path.join(sync, f'result.{i}.json')
			res = json.load(open(res_f)) if os.path.exists(res_f) else None
			used = float(smi('gpu=memory.used')[0][0])
			if res is None or not res.get('ok'):
				err = open(os.path.join(sync, f'err.{i}')).read()[-1500:]
				failure = dict(index=i, spec=sp, oom=('out of memory' in err.lower()) or bool(res and res.get('oom')),
							   error=(res or {}).get('error', err[-400:]), gpu_mem_used_mib=used)
				print(f'run {i} FAILED (oom={failure["oom"]}) at {used:.0f} MiB used', flush=True)
				break
			pm = {int(p_): float(m) for p_, m in smi('compute-apps=pid,used_memory')}
			res['process_vram_mib'] = pm.get(res['pid'], float('nan'))
			res['gpu_mem_used_after_mib'] = used
			runs.append(res)
			print(f'run {i} ok: {sp["task"]} H={sp["horizon"]} process {res["process_vram_mib"]:.0f} MiB, GPU used {used:.0f}/{total:.0f} MiB', flush=True)
		summary = dict(cycle=a.cycle, mps=a.mps, gpu_mem_baseline_mib=base, gpu_mem_total_mib=total,
					   n_fit=len(runs), failure=failure, runs=runs)
		os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
		json.dump(summary, open(a.out, 'w'), indent=1)
		print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}), flush=True)
	finally:
		open(os.path.join(sync, 'stop'), 'w').close()  # tell all held workers to release memory and exit
		time.sleep(2)
		for pr in procs:
			if pr.poll() is None:
				pr.kill()
		if a.mps:
			subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env); time.sleep(1)
			shutil.rmtree(mps_dir, ignore_errors=True)
		shutil.rmtree(sync, ignore_errors=True)


if __name__ == '__main__':
	main()
