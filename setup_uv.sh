#!/usr/bin/env bash
# Create the Python environment of tdmpc_gpu_warp with uv (alternative to conda + environment.yaml).
# GPU port by Keyshav Mor; based on TD-MPC by Nicklas Hansen et al. (https://github.com/nicklashansen/tdmpc).
#
# Usage (from the tdmpc_gpu_warp directory):   ./setup_uv.sh
# Result: .venv/ with Python 3.11 and the exact package versions of uv.lock (torch 2.11 + CUDA 12.8,
# warp-lang 1.18, mujoco 3.15, dm_control 1.0.48, patched MuJoCo Warp in ../mujoco_warp, editable).
# Afterwards:  source .venv/bin/activate   (or prefix commands with `uv run`)
# Note: `uv sync` (exact mode) would uninstall the editable mujoco_warp again - use `uv sync --inexact` or rerun
# this script after changing dependencies.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found - install the NVIDIA driver first (docs/USER_GUIDE.md section 1)"

uv python install 3.11
if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi
# MuJoCo Warp + GPU noslip patch in ../mujoco_warp, installed with --no-deps (its pyproject pins mujoco to a
# nightly index; all its dependencies are already in pyproject.toml / uv.lock)
VIRTUAL_ENV=$PWD/.venv ./scripts/setup_mujoco_warp.sh ../mujoco_warp

uv run python - <<'PY'
import torch, warp as wp, mujoco, mujoco_warp
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name() if torch.cuda.is_available() else '-')
wp.init()
print('warp', wp.__version__, '| mujoco', mujoco.__version__, '| mujoco_warp from', mujoco_warp.__file__)
from mujoco_warp._src import noslip  # GPU noslip solver (needed by dog-*)
print('GPU noslip solver: available')
PY
echo "done: source .venv/bin/activate"
