#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ======= user settings (override via env) =======
USE_DDP="${USE_DDP:-true}"                  # true = multi-GPU DDP per seed; false = 1 GPU per seed in parallel
SEEDS="${SEEDS:-42 43 44}"                  # space-separated seed list
GPU_IDS="${GPU_IDS:-1,2,3}"                 # GPUs to use (comma-separated)
NUM_GPUS="${NUM_GPUS:-3}"                   # number of GPUs for DDP
CONFIG_NAME="${CONFIG_NAME:-image_pusht_diffusion_policy_cnn.yaml}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
AUTO_RESUME="${AUTO_RESUME:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
# overrides appended to every training command
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# GPU utilization optimizations (in config, documented here for visibility):
#   dataloader.batch_size=128 (was 64) → effective batch 128×3=384 with DDP
#   dataloader.num_workers=12 (was 8) → faster data feeding
#   training.val_every=5 (was 1)       → 5× less validation overhead
#   training.early_stop_patience=500    → auto-stop when converged
# Target: ~6-8s/epoch with 85-95% GPU utilization

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# ------ conda bootstrap ------
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
  log "ERROR: conda not found" >&2
  exit 1
}

# ------ find existing output dir for a seed (auto-resume) ------
find_resume_dir() {
  local seed="$1"
  find data/outputs -maxdepth 2 -type d -name "pusht_image_train3_ddp_seed${seed}_*" 2>/dev/null \
    | while read -r candidate; do
        if [[ -f "${candidate}/checkpoints/latest.ckpt" ]]; then
          printf '%s\n' "${candidate}"
        fi
      done \
    | sort \
    | tail -n 1
}

# ------ launch a single seed with DDP (blocking) ------
run_seed_ddp() {
  local seed="$1"
  local run_dir

  if [[ "${AUTO_RESUME}" == "1" ]]; then
    run_dir="$(find_resume_dir "${seed}" || true)"
    if [[ -n "${run_dir}" ]]; then
      log "Seed ${seed}: auto-resuming from ${run_dir}"
    fi
  fi

  if [[ -z "${run_dir:-}" ]]; then
    run_dir="data/outputs/pusht_image_train3_ddp_seed${seed}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${run_dir}"
  fi

  local overrides=(
    "--config-dir=."
    "--config-name=${CONFIG_NAME}"
    "training.seed=${seed}"
    "training.ddp=true"
    "logging.mode=disabled"
    "hydra.run.dir=${run_dir}"
  )
  [[ -n "${EXTRA_OVERRIDES}" ]] && overrides+=(${EXTRA_OVERRIDES})

  log "Seed ${seed}: launching torchrun on GPUs ${GPU_IDS} (nproc=${NUM_GPUS})"
  log "Seed ${seed}: run dir ${run_dir}"
  log "Seed ${seed}: log tail -f ${run_dir}/train.log"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  WANDB_MODE=disabled \
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train.py "${overrides[@]}" \
    2>&1 | tee "${run_dir}/train.log"

  local exit_code=$?
  log "Seed ${seed}: torchrun exited with code ${exit_code}"
  return ${exit_code}
}

# ------ launch a single seed on one GPU (non-blocking, for parallel mode) ------
run_seed_single() {
  local seed="$1"
  local gpu="$2"

  local run_dir="data/outputs/pusht_image_train3_seed${seed}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${run_dir}"

  local overrides=(
    "--config-dir=."
    "--config-name=${CONFIG_NAME}"
    "training.seed=${seed}"
    "training.ddp=false"
    "logging.mode=disabled"
    "hydra.run.dir=${run_dir}"
  )
  [[ -n "${EXTRA_OVERRIDES}" ]] && overrides+=(${EXTRA_OVERRIDES})

  CUDA_VISIBLE_DEVICES="${gpu}" \
  WANDB_MODE=disabled \
  nohup python train.py "${overrides[@]}" \
    > "${run_dir}/train.log" 2>&1 &

  echo $!
}

# ======= main =======
main() {
  bootstrap_conda

  if [[ "${USE_DDP}" == "true" ]]; then
    # ---- sequential DDP mode: one seed at a time, all GPUs ----
    log "============================================"
    log "DDP sequential mode: ${NUM_GPUS} GPUs (${GPU_IDS}) per seed"
    log "Seeds: ${SEEDS}"
    log "Config: ${CONFIG_NAME}"
	    log "Optimizations: batch_size=128, val_every=5, num_workers=12"
	    log "Early stop patience: 500 (from config)"
    log "============================================"

    for seed in ${SEEDS}; do
      log "========== Starting seed ${seed} =========="
      if run_seed_ddp "${seed}"; then
        log "Seed ${seed}: completed successfully"
      else
        log "Seed ${seed}: FAILED with exit code $? — continuing to next seed"
      fi
      # increment master port to avoid conflict if multiple scripts run
      MASTER_PORT=$((MASTER_PORT + 1))
      sleep 5
    done

    log "All seeds finished."

  else
    # ---- parallel single-GPU mode: all seeds at once, 1 GPU each ----
    IFS=',' read -ra gpu_arr <<< "${GPU_IDS}"
    local seeds_arr=(${SEEDS})
    local num_seeds=${#seeds_arr[@]}

    if [[ ${#gpu_arr[@]} -lt ${num_seeds} ]]; then
      log "ERROR: need at least ${num_seeds} GPUs (have ${#gpu_arr[@]}: ${GPU_IDS})" >&2
      exit 1
    fi

    log "============================================"
    log "Parallel single-GPU mode: ${num_seeds} seeds on GPUs ${GPU_IDS}"
    log "============================================"

    local pids=()
    for i in $(seq 0 $((num_seeds - 1))); do
      local seed="${seeds_arr[$i]}"
      local gpu="${gpu_arr[$i]}"
      pid=$(run_seed_single "${seed}" "${gpu}")
      pids+=("${pid}")
      log "Seed ${seed}: GPU ${gpu}, PID ${pid}"
      sleep 3
    done

    log "All seeds launched. PIDs: ${pids[*]}"
    log "Monitor: tail -f data/outputs/pusht_image_train3_seed*/train.log"
  fi
}

main "$@"
