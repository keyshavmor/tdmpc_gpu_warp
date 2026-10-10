"""Run a training script with diagnostic variants applied (for bench/mps_fault_hunt.py).

Applies the variants selected by environment variables, then runs the given script (src/train.py or
src/train_vec.py) unchanged via runpy, so the repository code itself is not modified:
	TDMPC_DIAG_NO_CONDITIONAL=1   MuJoCo Warp solver without CUDA-graph conditional nodes (fixed iteration count
	                              instead of a captured while-loop; mujoco_warp's own non-conditional path)
	TDMPC_DIAG_MEMPOOL_KEEP=1     warp's CUDA memory pool never releases memory back to the driver
	TDMPC_DIAG_GC_OFF=1           Python's cyclic garbage collector disabled (objects freed only by refcount)
	TDMPC_DIAG_SYNC_GC=1          gc.collect() every 30 s (forces garbage collection to happen during the run)
Usage: python bench/fault_variant_wrapper.py src/train_vec.py task=walker-walk num_seeds=3 ...

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import gc
import os
import runpy
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
active = []
if os.environ.get('TDMPC_DIAG_NO_CONDITIONAL') == '1':
	import mujoco_warp as mjw
	_put = mjw.put_model
	def put_model(*a, **k):
		"""mujoco_warp.put_model that also sets opt.graph_conditional = False (solver without CUDA-graph conditional nodes)."""
		m = _put(*a, **k)
		m.opt.graph_conditional = False
		return m
	mjw.put_model = put_model
	active.append('no_conditional')
if os.environ.get('TDMPC_DIAG_MEMPOOL_KEEP') == '1':
	import warp as wp
	wp.init()
	wp.set_mempool_release_threshold('cuda:0', 2 ** 40)  # far above the GPU memory: never trim the pool
	active.append('mempool_keep')
if os.environ.get('TDMPC_DIAG_GC_OFF') == '1':
	gc.disable()
	active.append('gc_off')
if os.environ.get('TDMPC_DIAG_SYNC_GC') == '1':
	def _collector():
		"""Background loop: gc.collect() every 30 s for the lifetime of the process."""
		while True:
			time.sleep(30)
			gc.collect()
	threading.Thread(target=_collector, daemon=True).start()
	active.append('gc_every_30s')
print(f'[fault_variant_wrapper] variants: {active or ["none"]}', flush=True)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(os.path.join(ROOT, script) if not os.path.isabs(script) else script, run_name='__main__')
