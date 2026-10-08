#!/usr/bin/env bash
# Learning-curve re-runs after the PER replacement-flag fix (algorithm/tdmpc_graph.py, algorithm/tdmpc_vec.py now
# sample with replacement until the buffer is full, exactly like the original np.random.choice(replace=not full)).
#   phase 1: the recommended configuration - separate single-seed processes with the CUDA-graph agent
#            (src/train.py graph_agent=true pace_host=true), 5 seeds x {cartpole-swingup, walker-walk}, under CUDA MPS
#   phase 2: the vectorised agent (src/train_vec.py, 5 seeds per process), without MPS
# Every run stops at 100k env steps; logs in results/learning/<task>/state/{graph_fixed,vec_fp32_fixed}/<seed>/,
# console output in results/learning/logs/. Analysed by bench/analyze_learning.py -> docs/results_learning.md.
# Environment: tdmpc-warp. Usage: nohup bash bench/pipeline_learning_fix.sh > /dev/null 2>&1 &
# Options (environment variables): PHASES="1 2" (default both), USE_MPS=1 (phase 1 under MPS; 0 = plain time-slicing).
# Note: the first attempt of phase 1 under MPS ended after ~3 min with an illegal-memory-access fault in one
# single-world process that tore down all MPS clients (docs/08_known_issues.md); phase 1 was then re-run with USE_MPS=0.
# Author of the GPU port: Keyshav Mor. Based on TD-MPC by Nicklas Hansen et al. (https://github.com/nicklashansen/tdmpc).
set -u
cd "$(dirname "$0")/.."
PY=${TDMPC_WARP_PYTHON:-python}  # interpreter of the tdmpc-warp environment
LR=$PWD/results/learning; mkdir -p $LR/logs
COMMON="max_env_steps=100000 log_root=$LR"

PHASES=${PHASES:-"1 2"}; USE_MPS=${USE_MPS:-1}

# ---- phase 1: single-seed CUDA-graph agent (under MPS if USE_MPS=1)
if [[ " $PHASES " == *" 1 "* ]]; then
if [ "$USE_MPS" = 1 ]; then
export CUDA_MPS_PIPE_DIRECTORY=/tmp/tdmpc_gpu_warp/mps_lfix/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/tdmpc_gpu_warp/mps_lfix/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
fi
pids=()
for task in walker-walk cartpole-swingup; do
  for seed in 1 2 3 4 5; do
    $PY src/train.py task=$task seed=$seed graph_agent=true pace_host=true exp_name=graph_fixed $COMMON \
      > $LR/logs/graph_fixed_${task}_s$seed.log 2>&1 &
    pids+=($!); sleep 15            # stagger start-up (graph capture)
  done
done
wait "${pids[@]}"
if [ "$USE_MPS" = 1 ]; then echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
sleep 5
fi

# ---- phase 2: vectorised agent, no MPS
if [[ " $PHASES " == *" 2 "* ]]; then
pids=()
for task in walker-walk cartpole-swingup; do
  $PY src/train_vec.py task=$task num_seeds=5 seed=1 exp_name=vec_fp32_fixed $COMMON > $LR/logs/vec_fp32_fixed_$task.log 2>&1 &
  pids+=($!); sleep 15
done
wait "${pids[@]}"
fi
echo done > $LR/logs/pipeline_learning_fix.done
