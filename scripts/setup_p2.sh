#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${P2_MJCF:?Set P2_MJCF to astro_p2_30dof_primitive_collision.xml}"
venv=${UNIMATE_VENV:-/data/getting/.venvs/unimate}
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.10 --seed "$venv"
fi
uv pip install --python "$venv/bin/python" "setuptools<81" wheel pip
uv pip install --python "$venv/bin/python" --index-strategy unsafe-best-match \
  --no-build-isolation -r requirements.txt -r requirements-p2.txt
source "$venv/bin/activate"
python -m unimate.robots.p2 prepare --mjcf "$P2_MJCF"
export HF_HOME=${HF_HOME:-/data/getting/unimate-cache/huggingface}
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Linzhan/UniMate", local_dir="outputs", allow_patterns=[
    "unimate_uniml3d_f60_v3/config.json", "unimate_uniml3d_f60_v3/dataset_stats.npy",
    "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt"])
snapshot_download("google/flan-t5-base", allow_patterns=["config.json", "model.safetensors",
    "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json"])
PY
python -m pytest -q tests/test_p2.py
