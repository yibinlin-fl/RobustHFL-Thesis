#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
env_dir="${ROBUSTHFL_VENV:-$PWD/.venv-openi}"
"$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info < (3, 13), sys.version'
"$python_bin" -m venv --system-site-packages "$env_dir"
py="$env_dir/bin/python"
if [[ -n "${TORCH_WHEEL_INDEX:-}" ]]; then
    : "${TORCH_VERSION:?Set TORCH_VERSION when choosing a CUDA wheel}"
    : "${TORCHVISION_VERSION:?Set TORCHVISION_VERSION for the matching pair}"
    "$py" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_WHEEL_INDEX"
fi
"$py" -m pip install -r "$repo/requirements-openi.txt"
cd -- "$repo"
"$py" -m scripts.openi.preflight "$@"
printf 'Python interpreter: %s\n' "$py"
