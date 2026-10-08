#!/bin/bash
# phase 3 (after the OOM study): vectorised memory, precision sweep, packing, contention, learning runs
#
# One of the unattended batch pipelines that ran the measurement phases of tdmpc_gpu_warp (TD-MPC on GPU
# MuJoCo / MuJoCo Warp) on the RTX 3090. Waits until the phase-2 OOM-horizon study has finished, then
# runs sequentially (all drivers with the `tdmpc-warp` env interpreter):
#   bench/vec_memory.py         seeds per vectorised process      -> results/vec_memory/   (docs/results_memory.md)
#   bench/sweep_h.py vec_*      precision / checkpointing sweep   -> results/sweep_h/vec_*/ (docs/07_precision.md)
#   bench/packing_study.py      runs until CUDA OOM               -> results/packing/      (docs/results_memory.md)
#   bench/contention_study.py   N concurrent seeds, throughput    -> results/contention/   (docs/04_multi_run_capacity.md)
# (the learning runs named above were started later, by the phase-4 / final pipelines.)
# Logs: results/logs/<step>.log; prints "PHASE3 DONE" at the end (pipeline_phase4.sh waits for it).
#   nohup bash bench/pipeline_phase3.sh > results/logs/phase3.log 2>&1 &
#
# Author of the GPU port: Keyshav Mor.
# Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
# Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
# (MIT License, Copyright (c) 2021 Nicklas Hansen).
cd "$(dirname "$0")/.."
PY=${TDMPC_WARP_PYTHON:-python}  # interpreter of the tdmpc-warp environment
# one representative task per domain (tasks of a domain share networks / buffer sizes)
REPS="acrobot-swingup cartpole-swingup cheetah-run cup-catch dog-run finger-spin fish-swim hopper-hop humanoid-run quadruped-run reacher-easy walker-walk"
# wait while the phase-2 OOM-horizon study is still running (the adjacent quoted strings form one
# pgrep pattern; split so that the full pattern never appears literally in this text)
while pgrep -f "bench/oom""_horizon.py" > /dev/null; do sleep 30; done
echo "== vec memory"; $PY bench/vec_memory.py > results/logs/vec_memory.log 2>&1; echo "vecmem rc=$?"
echo "== precision sweep"; $PY bench/sweep_h.py --modes vec_fp32 vec_tf32 vec_bf16 vec_fp16 vec_bf16_buf vec_ckpt --tasks $REPS > results/logs/precision_sweep.log 2>&1; echo "prec rc=$?"
echo "== packing"; $PY bench/packing_study.py > results/logs/packing.log 2>&1; echo "packing rc=$?"
echo "== contention"; $PY bench/contention_study.py > results/logs/contention.log 2>&1; echo "contention rc=$?"
echo "PHASE3 DONE"
