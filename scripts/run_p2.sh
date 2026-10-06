#!/usr/bin/env bash
# Run from any directory after activating the UniMate environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR PROMPT [PROMPT ...]" >&2
  echo "Example: NUM_REPETITIONS=3 $0 outputs/p2/walk \"An object walks forward.\"" >&2
  exit 2
fi
out=$1
shift
if [[ -e "$out" ]]; then
  echo "Output already exists: $out; choose a new directory to preserve earlier samples." >&2
  exit 2
fi
asset=${P2_ASSET:-outputs/rig/astro_p2}
exp=${UNIMATE_EXP:-outputs/unimate_uniml3d_f60_v3}
python -m unimate.inference.sample --exp_dir "$exp" --asset "$asset" \
  --prompt "$@" --num_repetitions "${NUM_REPETITIONS:-3}" \
  --seed "${SEED:-10}" --batch_size "${BATCH_SIZE:-4}" \
  --cfg_scale "${CFG_SCALE:-3}" --only_save_motion --output_dir "$out"
mkdir -p "$out/robot"
shopt -s nullglob
motions=("$out"/motions/*.npy)
if [[ ${#motions[@]} -eq 0 ]]; then
  echo "Sampler produced no motions" >&2
  exit 1
fi
for motion in "${motions[@]}"; do
  name=$(basename "$motion" .npy)
  python -m unimate.robots.p2 convert --asset "$asset" --motion "$motion" \
    --output "$out/robot/$name.npz" --ground
  if [[ ${RENDER:-1} == 1 ]]; then
    MUJOCO_GL=${MUJOCO_GL:-egl} python -m unimate.robots.p2 render \
      --asset "$asset" --motion "$out/robot/$name.npz" --output "$out/robot/$name.mp4"
  fi
done
printf "P2 candidates and quality reports: %s/robot\n" "$out"
