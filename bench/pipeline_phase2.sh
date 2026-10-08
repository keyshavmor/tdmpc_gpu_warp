#!/bin/bash
# phase 2: final-code validation, equivalence tests, OOM-horizon study (GPU exclusive, sequential)
#
# One of the unattended batch pipelines that ran the measurement phases of tdmpc_gpu_warp (TD-MPC on GPU
# MuJoCo / MuJoCo Warp) on the RTX 3090. Steps, all with the `tdmpc-warp` env interpreter, one after the
# other so that each has the GPU to itself:
#   bench/validate_env.py       env port vs dm_control, final code  -> results/validation_final/ (docs/03_validation.md)
#   bench/test_batched_env.py   N-world env vs N single envs        (docs/03_validation.md, 3.4)
#   bench/test_ckpt.py          activation checkpointing is exact
#   bench/test_graph_agent.py   CUDA-graph agent == eager agent (dog-run, humanoid-run, cartpole-swingup)
#   bench/test_vec_agent.py     vectorised agent == per-seed agents (dog-run)
#   bench/oom_horizon.py        single-run OOM horizon              -> results/oom_horizon/ (docs/results_memory.md)
# Logs: results/logs/<step>.log; "<step> rc=<exit code>" lines report success per step.
#   nohup bash bench/pipeline_phase2.sh > results/logs/phase2.log 2>&1 &
#
# Author of the GPU port: Keyshav Mor.
# Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
# Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
# (MIT License, Copyright (c) 2021 Nicklas Hansen).
cd "$(dirname "$0")/.."
PY=${TDMPC_WARP_PYTHON:-python}  # interpreter of the tdmpc-warp environment
mkdir -p results/validation_final
echo "== validation (final code)"; $PY bench/validate_env.py > results/logs/validate_final.log 2>&1; echo "validation rc=$?"
echo "== batched env"; $PY bench/test_batched_env.py cartpole-swingup reacher-hard finger-turn-hard fish-swim walker-walk cup-catch cheetah-run > results/logs/batched_env.log 2>&1; echo "batched rc=$?"
echo "== ckpt"; $PY bench/test_ckpt.py > results/logs/ckpt.log 2>&1; echo "ckpt rc=$?"
echo "== graph agent equivalence (more tasks)"; for t in dog-run humanoid-run cartpole-swingup; do $PY bench/test_graph_agent.py $t 5 > results/logs/graph_agent_$t.log 2>&1; done; echo "graph rc=$?"
echo "== vec agent equivalence (dog)"; $PY bench/test_vec_agent.py dog-run 5 3 > results/logs/vec_agent_dog.log 2>&1; echo "vec rc=$?"
echo "== OOM horizon"; $PY bench/oom_horizon.py > results/logs/oom_horizon.log 2>&1; echo "oom rc=$?"
echo "PHASE2 DONE"
