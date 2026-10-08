#!/usr/bin/env bash
# Set up MuJoCo Warp with the GPU noslip solver of tdmpc_gpu_warp (needed by the dog tasks; all other tasks also
# run with upstream MuJoCo Warp at this commit, but the patched checkout is what every result was measured with).
#
# Clones https://github.com/google-deepmind/mujoco_warp at the tested commit next to this repository
# (default: ../mujoco_warp), applies patches/mujoco_warp/0001-gpu-noslip-solver.patch and, if a Python
# environment is active, installs it in editable mode.
#
# Usage (from the repository root):   ./scripts/setup_mujoco_warp.sh [target_dir] [--no-install]
#
# Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp). Author of the GPU port: Keyshav Mor.
# Based on TD-MPC by Nicklas Hansen et al. (https://github.com/nicklashansen/tdmpc).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
COMMIT=2c09ab908e5cd50cdabd1194713ba58e601899ee
TARGET=${1:-../mujoco_warp}
INSTALL=1; [[ " $* " == *" --no-install "* ]] && INSTALL=0
[[ "$TARGET" == --no-install ]] && TARGET=../mujoco_warp

if [ -d "$TARGET/.git" ]; then
  echo "using existing checkout $TARGET"
else
  git clone https://github.com/google-deepmind/mujoco_warp.git "$TARGET"
fi
cd "$TARGET"
if [ -f mujoco_warp/_src/noslip.py ]; then
  echo "noslip patch already present in $TARGET"
else
  git checkout --quiet "$COMMIT"
  git apply "$REPO_ROOT/patches/mujoco_warp/0001-gpu-noslip-solver.patch"
  echo "applied GPU noslip patch on top of $COMMIT"
fi
if [ "$INSTALL" = 1 ]; then
  if [ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && command -v uv >/dev/null; then uv pip install --no-deps -e .; else pip install --no-deps -e .; fi
  else
    echo "no active Python environment: install later with  pip install --no-deps -e $TARGET"
  fi
fi
