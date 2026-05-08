#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEEDS_CSV="${SEEDS:-42,43,44}"
GPUS_CSV="${GPUS:-1,3,4}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
RUN_ROOT="${RUN_ROOT:-data/outputs/pusht_image_train3_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-image_pusht_diffusion_policy_cnn.yaml}"
DATA_URL="${DATA_URL:-https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip}"
SMOKE="${SMOKE:-0}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [[ "${#SEEDS[@]}" -ne "${#GPUS[@]}" ]]; then
  echo "SEEDS and GPUS must have the same length; got ${SEEDS_CSV} and ${GPUS_CSV}" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

bootstrap_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi
  for conda_sh in \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck source=/dev/null
      source "${conda_sh}"
      return
    fi
  done
}

ensure_data() {
  mkdir -p data
  if [[ -d data/pusht/pusht_cchi_v7_replay.zarr ]]; then
    log "Using existing Push-T dataset at data/pusht/pusht_cchi_v7_replay.zarr"
    return
  fi

  log "Downloading Push-T dataset"
  wget -c -O data/pusht.zip "${DATA_URL}"
  log "Extracting Push-T dataset"
  if command -v unzip >/dev/null 2>&1; then
    unzip -n data/pusht.zip -d data
  else
    python - <<'PY'
from zipfile import ZipFile
with ZipFile("data/pusht.zip") as zf:
    zf.extractall("data")
PY
  fi
}

ensure_config() {
  if [[ -s "${CONFIG_PATH}" ]]; then
    log "Using existing config ${CONFIG_PATH}"
    return
  fi
  log "Downloading image Push-T config"
  wget -O "${CONFIG_PATH}" \
    https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
}

write_env_snapshot() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  {
    echo "timestamp=$(date -Is)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "seeds=${SEEDS_CSV}"
    echo "gpus=${GPUS_CSV}"
    echo "conda_env=${CONDA_ENV}"
    echo "smoke=${SMOKE}"
    echo
    nvidia-smi || true
    echo
    conda run -n "${CONDA_ENV}" python - <<'PY' || true
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
  } > "${out_dir}/env_info.txt"
}

launch_one() {
  local seed="$1"
  local gpu="$2"
  local run_dir="${RUN_ROOT}/seed_${seed}"
  mkdir -p "${run_dir}"

  local overrides=(
    "--config-dir=."
    "--config-name=${CONFIG_PATH}"
    "training.seed=${seed}"
    "training.device=cuda:0"
    "logging.mode=offline"
    "hydra.run.dir=${run_dir}"
  )

  if [[ "${SMOKE}" == "1" ]]; then
    overrides+=(
      "training.debug=true"
      "training.num_epochs=2"
      "training.max_train_steps=3"
      "training.max_val_steps=3"
    )
  fi

  local cmd
  printf -v cmd '%q ' env CUDA_VISIBLE_DEVICES="${gpu}" WANDB_MODE=offline conda run -n "${CONDA_ENV}" python train.py "${overrides[@]}"
  cmd="${cmd}2>&1 | tee ${run_dir}/train.log"

  log "Launching seed ${seed} on physical GPU ${gpu}; output ${run_dir}"
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "dp_pusht_s${seed}" "cd '${REPO_ROOT}' && ${cmd}"
    echo "tmux: dp_pusht_s${seed}" > "${run_dir}/launcher.txt"
  else
    nohup bash -lc "cd '${REPO_ROOT}' && ${cmd}" > "${run_dir}/nohup.log" 2>&1 &
    echo "pid: $!" > "${run_dir}/launcher.txt"
  fi
}

main() {
  bootstrap_conda
  command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
  command -v wget >/dev/null 2>&1 || { echo "wget not found" >&2; exit 1; }

  ensure_data
  ensure_config
  mkdir -p "${RUN_ROOT}"
  write_env_snapshot "${RUN_ROOT}"

  for idx in "${!SEEDS[@]}"; do
    launch_one "${SEEDS[$idx]}" "${GPUS[$idx]}"
  done

  log "Launched ${#SEEDS[@]} run(s). Run root: ${RUN_ROOT}"
  if command -v tmux >/dev/null 2>&1; then
    tmux ls | grep 'dp_pusht_' || true
  fi
}

main "$@"
