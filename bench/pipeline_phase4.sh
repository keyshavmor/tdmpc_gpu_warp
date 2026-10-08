#!/bin/bash
# phase 4 (after phase 3): interference study, multi-seed precision, AMP memory analysis, learning runs
#
# One of the unattended batch pipelines that ran the measurement phases of tdmpc_gpu_warp (TD-MPC on GPU
# MuJoCo / MuJoCo Warp) on the RTX 3090. Waits for "PHASE3 DONE" in results/logs/phase3.log, then runs
# sequentially (all with the `tdmpc-warp` env interpreter):
#   bench/interference_study.py   no-slowdown capacity, mixed H  -> results/interference/  (docs/04_multi_run_capacity.md)
#   bench/bench_step.py (vec)     multi-seed precision, H = 5     -> results/precision_vec/ (docs/07_precision.md, 7.2)
#   bench/test_amp_memory.py      why autocast saves little memory                         (docs/07_precision.md, 7.3)
#   bench/learning_runs.py start  detached learning runs          -> results/learning/      (docs/results_learning.md)
# pipeline_phase4b.sh / pipeline_final.sh repeat the later steps (the precision loop and the study drivers
# skip results that already exist).
# Logs: results/logs/<step>.log; prints "PHASE4 DONE" at the end.
#
# Author of the GPU port: Keyshav Mor.
# Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
# Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
# (MIT License, Copyright (c) 2021 Nicklas Hansen).
cd "$(dirname "$0")/.."
PY=${TDMPC_WARP_PYTHON:-python}  # interpreter of the tdmpc-warp environment
until grep -q "PHASE3 DONE" results/logs/phase3.log; do sleep 30; done  # chain after phase 3 (GPU exclusive)
echo "== interference"; $PY bench/interference_study.py > results/logs/interference.log 2>&1; echo "interference rc=$?"
echo "== multi-seed precision"
mkdir -p results/precision_vec
# "<task> <S>": S seeds in ONE vectorised process (compute-saturated regime), each precision once;
# a result file that already exists is skipped
for spec in "walker-walk 16" "dog-run 8" "humanoid-run 16" "cartpole-swingup 16"; do
  set -- $spec
  for prec in fp32 tf32 bf16 fp16; do
    f=results/precision_vec/$1_S$2_$prec.json
    [ -f $f ] || $PY bench/bench_step.py --mode gpu --task $1 --horizon 5 --k 40 --out $f --extra vec_agent=true num_seeds=$2 pace_host=true precision=$prec >> results/logs/precision_vec.log 2>&1
  done
done
echo "precvec rc=$?"
echo "== amp memory"; $PY bench/test_amp_memory.py > results/logs/amp_memory.log 2>&1; echo "ampmem rc=$?"
echo "== learning runs"; $PY bench/learning_runs.py start > results/logs/learning_start.log 2>&1; echo "learning started rc=$?"
echo "PHASE4 DONE"
