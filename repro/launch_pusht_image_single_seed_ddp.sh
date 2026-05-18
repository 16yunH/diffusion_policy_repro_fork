#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---- user settings ----
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-4}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
CONFIG_PATH="${CONFIG_PATH:-image_pusht_diffusion_policy_cnn.yaml}"
DATA_URL="${DATA_URL:-https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RUN_DIR="${RUN_DIR:-}"
SESSION_NAME="${SESSION_NAME:-dp_pusht_ddp_s${SEED}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SMOKE="${SMOKE:-0}"

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

find_resume_dir() {
  find data/outputs -maxdepth 2 -type d -name "seed_${SEED}" 2>/dev/null \
    | while read -r candidate; do
        if [[ -f "${candidate}/checkpoints/latest.ckpt" ]]; then
          printf '%s\n' "${candidate}"
        fi
      done \
    | sort \
    | tail -n 1
}

resolve_run_dir() {
  if [[ -n "${RUN_DIR}" ]]; then
    printf '%s\n' "${RUN_DIR}"
    return
  fi

  if [[ "${AUTO_RESUME}" == "1" ]]; then
    local resume_dir
    resume_dir="$(find_resume_dir || true)"
    if [[ -n "${resume_dir}" ]]; then
      log "Auto-resuming from ${resume_dir}"
      printf '%s\n' "${resume_dir}"
      return
    fi
  fi

  printf 'data/outputs/pusht_image_ddp_seed_%s_%s\n' "${SEED}" "$(date +%Y%m%d_%H%M%S)"
}

write_env_snapshot() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  {
    echo "timestamp=$(date -Is)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "seed=${SEED}"
    echo "num_gpus=${NUM_GPUS}"
    echo "conda_env=${CONDA_ENV}"
    echo "auto_resume=${AUTO_RESUME}"
    echo "run_dir=${out_dir}"
    echo "num_workers=${NUM_WORKERS}"
    echo "val_num_workers=${VAL_NUM_WORKERS}"
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
  } > "${out_dir}/ddp_env_info.txt"
}

main() {
  bootstrap_conda
  command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
  command -v wget >/dev/null 2>&1 || { echo "wget not found" >&2; exit 1; }

  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session ${SESSION_NAME} already exists; attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
  fi

  ensure_data
  ensure_config

  local run_dir
  run_dir="$(resolve_run_dir)"
  mkdir -p "${run_dir}"
  write_env_snapshot "${run_dir}"

  local overrides=(
    "--config-dir=."
    "--config-name=${CONFIG_PATH}"
    "training.seed=${SEED}"
    "training.ddp=true"
    "logging.mode=offline"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${VAL_NUM_WORKERS}"
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
  printf -v cmd '%q ' env WANDB_MODE=offline conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${NUM_GPUS}" train.py "${overrides[@]}"
  cmd="${cmd}2>&1 | tee -a ${run_dir}/train.log"

  log "Launching seed ${SEED} on ${NUM_GPUS} GPUs via torchrun; output ${run_dir}"
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${SESSION_NAME}" "cd '${REPO_ROOT}' && ${cmd}"
    echo "tmux: ${SESSION_NAME}" > "${run_dir}/ddp_launcher.txt"
    log "Attach with: tmux attach -t ${SESSION_NAME}"
  else
    nohup bash -lc "cd '${REPO_ROOT}' && ${cmd}" > "${run_dir}/ddp_nohup.log" 2>&1 &
    echo "pid: $!" > "${run_dir}/ddp_launcher.txt"
    log "Started background PID $!"
  fi

  log "Monitor log: tail -f ${run_dir}/train.log"
}

main "$@"
