#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
MIN_GPU_COUNT="${MIN_GPU_COUNT:-1}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_ENV="${SKIP_ENV:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval/image_pusht_dp_cnn_train0}"
PRETRAIN_DIR="${PRETRAIN_DIR:-data/pretrained/image_pusht_diffusion_policy_cnn}"
CONFIG_URL="${CONFIG_URL:-https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml}"
CKPT_URL="${CKPT_URL:-https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt}"
CKPT_PATH="${CKPT_PATH:-${PRETRAIN_DIR}/epoch=0500-test_mean_score=0.884.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-${PRETRAIN_DIR}/config.yaml}"

mkdir -p "${PRETRAIN_DIR}" "${OUTPUT_DIR}" data/repro

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
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

install_system_deps() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    log "Skipping apt dependency installation because SKIP_APT=1"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; assuming system rendering dependencies are already installed"
    return
  fi

  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      log "sudo not found; set SKIP_APT=1 after installing libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf"
      exit 1
    fi
    sudo_cmd=(sudo)
  fi

  log "Installing Mujoco/rendering system dependencies"
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf wget unzip git
}

setup_conda_env() {
  if [[ "${SKIP_ENV}" == "1" ]]; then
    log "Skipping conda env creation because SKIP_ENV=1"
    return
  fi

  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    log "Conda environment ${CONDA_ENV} already exists"
    return
  fi

  if command -v mamba >/dev/null 2>&1; then
    log "Creating ${CONDA_ENV} with mamba"
    mamba env create -f conda_environment.yaml
  else
    log "Creating ${CONDA_ENV} with conda"
    conda env create -f conda_environment.yaml
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  if [[ -s "${dest}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
    log "Using existing ${dest}"
    return
  fi
  log "Downloading ${url}"
  wget -c -O "${dest}" "${url}"
}

write_env_info() {
  {
    echo "timestamp=$(date -Is)"
    echo "repo_root=${REPO_ROOT}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo
    echo "uname:"
    uname -a || true
    echo
    echo "disk:"
    df -h . || true
    echo
    echo "nvidia-smi:"
    nvidia-smi || true
    echo
    echo "conda:"
    conda --version || true
    echo
    echo "python:"
    conda run -n "${CONDA_ENV}" python --version || true
    echo
    echo "torch_cuda:"
    conda run -n "${CONDA_ENV}" python - <<'PY' || true
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("torch", torch.__version__)
PY
  } > data/repro/env_info.txt

  git rev-parse HEAD > data/repro/git_commit.txt
}

run_import_checks() {
  log "Running Python import checks in ${CONDA_ENV}"
  conda run -n "${CONDA_ENV}" python - <<'PY'
import torch
print("torch.cuda.is_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("torch.cuda.device_name=", torch.cuda.get_device_name(0))
import diffusion_policy
import gym
import mujoco_py
print("imports_ok")
PY
}

main() {
  bootstrap_conda

  require_cmd git
  require_cmd wget
  require_cmd conda
  require_cmd nvidia-smi

  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  log "Detected ${gpu_count} NVIDIA GPU(s); using DEVICE=${DEVICE}"
  if [[ "${gpu_count}" -lt "${MIN_GPU_COUNT}" ]]; then
    echo "Expected at least ${MIN_GPU_COUNT} GPU(s), found ${gpu_count}" >&2
    exit 1
  fi

  install_system_deps
  setup_conda_env
  download_file "${CONFIG_URL}" "${CONFIG_PATH}"
  download_file "${CKPT_URL}" "${CKPT_PATH}"
  write_env_info
  run_import_checks

  log "Running Push-T image pretrained checkpoint eval"
  mkdir -p "${OUTPUT_DIR}"
  set -o pipefail
  conda run -n "${CONDA_ENV}" python eval.py \
    --checkpoint "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

  log "Collecting result summary"
  conda run -n "${CONDA_ENV}" python repro/summarize_pusht_eval.py \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --env-info data/repro/env_info.txt \
    --summary-md data/repro/pusht_image_pretrained_eval_summary.md

  log "Done. Summary: data/repro/pusht_image_pretrained_eval_summary.md"
}

main "$@"
