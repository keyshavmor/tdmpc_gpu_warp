#!/bin/bash
# final phase: same-H interference, SM partitioning, multi-seed precision, AMP memory, learning runs
#
# One of the unattended batch pipelines that ran the measurement phases of tdmpc_gpu_warp (TD-MPC on GPU
# MuJoCo / MuJoCo Warp) on the RTX 3090. Runs sequentially (all with the `tdmpc-warp` env interpreter):
#   bench/interference_sameH.py   2 runs of the same H (H = 2, 30; 50, 100) -> results/interference_sameH/ (docs/04_multi_run_capacity.md, 4.4)
#   bench/partition_study.py      MPS SM partitioning                    -> results/partition/      (docs/04_multi_run_capacity.md, 4.6)
#   bench/bench_step.py (vec)     multi-seed precision, H = 5            -> results/precision_vec/  (docs/07_precision.md, 7.2)
#                                 walker-walk 16 seeds and dog-run 8 seeds, fp32 / tf32 / bf16 / fp16
#   bench/test_amp_memory.py      why autocast saves little memory                                 (docs/07_precision.md, 7.3)
#   bench/learning_runs.py start  detached learning runs                 -> results/learning/       (docs/results_learning.md)
# The study drivers are resumable, so steps already completed by earlier phases are skipped.
# Logs: results/logs/<step>.log; prints "FINAL PIPELINE DONE" at the end.
#
# Author of the GPU port: Keyshav Mor.
# Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
# Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
# (MIT License, Copyright (c) 2021 Nicklas Hansen).
cd "$(dirname "$0")/.."
PY=${TDMPC_WARP_PYTHON:-python}  # interpreter of the tdmpc-warp environment
echo "== same-H interference"; $PY bench/interference_sameH.py > results/logs/interference_sameH.log 2>&1; echo "sameH rc=$?"
echo "== partition"; $PY bench/partition_study.py > results/logs/partition.log 2>&1; echo "partition rc=$?"
echo "== multi-seed precision"
mkdir -p results/precision_vec
# "<task> <S>": S seeds in ONE vectorised process (compute-saturated regime), each precision once;
# a result file that already exists is skipped
for spec in "walker-walk 16" "dog-run 8"; do
  set -- $spec
  for prec in fp32 tf32 bf16 fp16; do
    f=results/precision_vec/$1_S$2_$prec.json
    [ -f $f ] || $PY bench/bench_step.py --mode gpu --task $1 --horizon 5 --k 40 --out $f --extra vec_agent=true num_seeds=$2 pace_host=true precision=$prec >> results/logs/precision_vec.log 2>&1
  done
done
echo "precvec rc=$?"
echo "== amp memory"; $PY bench/test_amp_memory.py > results/logs/amp_memory.log 2>&1; echo "ampmem rc=$?"
echo "== learning runs"; $PY bench/learning_runs.py start > results/logs/learning_start.log 2>&1; echo "learning started rc=$?"
echo "FINAL PIPELINE DONE"
