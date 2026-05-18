# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) paper (Chi et al.) for reproducing Push-T simulation experiments. The project trains visuomotor policies using diffusion models to control a robot pushing a T-shaped block to a target pose.

## Environment Setup

The conda environment is `robodiff` (from `conda_environment_pusht.yaml`). Activate with:
```bash
source ~/mambaforge/etc/profile.d/conda.sh && conda activate robodiff
```

The training data lives at `data/pusht/pusht_cchi_v7_replay.zarr` (must be downloaded separately from the project page).

## Key Entry Points

- **Training**: `python train.py --config-name=<config_name>` (Hydra-based, configs in `diffusion_policy/config/`)
- **Evaluation**: `python eval.py --checkpoint <path.ckpt> -o <output_dir>` (standalone eval with deterministic env seeds)
- **DDP training** (multi-GPU): `torchrun --nproc_per_node=<N> train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=<S> training.ddp=true logging.mode=disabled hydra.run.dir=<dir>`

For Push-T image-based experiments, use config `image_pusht_diffusion_policy_cnn.yaml` (at repo root), which overrides the defaults for the hybrid diffusion UNet workspace.

## Architecture

### Core abstraction: Workspace

`BaseWorkspace` (`diffusion_policy/workspace/base_workspace.py`) is the central orchestrator. Each method (diffusion policy, IBC, BET, Robomimic) has its own workspace subclass. The workspace owns:
- `model` — the policy being trained
- `ema_model` — exponential moving average copy (if `training.use_ema`)
- `optimizer` — AdamW typically

Checkpoint save/load is handled by `BaseWorkspace.save_checkpoint()` / `load_checkpoint()`, which serializes `state_dicts` of all torch modules and optionally pickles select attributes. Checkpoints are self-contained (include the Hydra config) and can be loaded by `eval.py` without the training code.

### Config system

Uses [Hydra](https://hydra.cc/) with `--config-dir` pointing to `diffusion_policy/config/`. The root `train.py` registers an `eval` resolver for arbitrary Python expressions in configs and auto-detects the workspace class via `_target_`. A common pattern: use a separate YAML at repo root that overrides select fields (e.g., `image_pusht_diffusion_policy_cnn.yaml`).

Task configs in `diffusion_policy/config/task/` define the dataset, env runner, and agent observation/action shapes for each robot environment.

### Policy hierarchy

All policies inherit from `BaseImagePolicy` or `BaseLowdimPolicy`:

```
BaseImagePolicy / BaseLowdimPolicy
  ├── DiffusionUnetImagePolicy          (CNN visual encoder + UNet1D diffusion, images only)
  ├── DiffusionUnetHybridImagePolicy    (CNN visual encoder + UNet1D, images + low-dim agent pos)
  ├── DiffusionUnetLowdimPolicy         (UNet1D diffusion, low-dim only)
  ├── DiffusionUnetVideoPolicy          (video-style diffusion)
  ├── DiffusionTransformerHybridImagePolicy
  ├── DiffusionTransformerLowdimPolicy
  ├── IbcDfoHybridImagePolicy / IbcDfoLowdimPolicy   (energy-based)
  ├── BetLowdimPolicy                   (behavior transformers)
  └── RobomimicImagePolicy / RobomimicLowdimPolicy
```

The two key methods on every policy:
- `predict_action(obs_dict)` — inference; returns `{"action": tensor(B, Ta, Da)}`
- `compute_loss(batch)` — training; returns scalar loss

### Diffusion UNet structure

The hybrid policy (`DiffusionUnetHybridImagePolicy`) uses:
1. A ResNet-18 visual encoder (from robomimic) to extract image features + a linear encoder for `agent_pos`
2. A `ConditionalUnet1D` that denoises action sequences conditioned on observation features
3. A `DDPMScheduler` (from HuggingFace `diffusers`) for the noise schedule

### Dataset

`PushTImageDataset` reads from a zarr archive and returns sequences of `(obs_dict, action)` where `obs_dict` contains `image` and `agent_pos`. The `SequenceSampler` handles episode boundaries and padding.

### Evaluation

`PushTImageRunner` runs the policy on 50 test seeds via `AsyncVectorEnv`, records videos via `VideoRecordingWrapper`, and computes `max_reward` per seed. The aggregate `test/mean_score` is the average of per-seed max rewards.

Rollout happens both during training (logged to wandb/json) and via standalone `eval.py`.

### DDP utilities

`diffusion_policy/common/ddp_util.py` — safe to call in both single-GPU and DDP modes. Key functions:
- `init_ddp()` — initializes NCCL from torchrun env vars (no-op otherwise)
- `is_main_process()` — True on rank 0, used for gating wandb, checkpointing, rollouts
- `wrap_ddp(model)` — wraps in DistributedDataParallel
- `prepare_dataloader(dataset, ...)` — adds DistributedSampler when DDP is enabled

Checkpoint save/load auto-detects DDP wrappers and saves/loads `model.module.state_dict()` for portability.

## Reproducibility (repro/)

Scripts for launching Push-T training and evaluation:
- `repro/launch_multi_seed.sh` — launches multiple seeds (sequential DDP or parallel single-GPU)
- `launch_pusht_image_single_seed.sh` — single-seed single-GPU training
- `launch_pusht_image_single_seed_ddp.sh` — single-seed multi-GPU DDP via torchrun
- `repro/summarize_pusht_eval.py` — aggregates eval results across seeds
- `repro/run_pusht_image_pretrained_eval.sh` — runs eval on a pretrained checkpoint

## Deterministic evaluation

`PushTImageRunner.run()` creates a `torch.Generator` seeded to 0 and passes it to `policy.predict_action()`. All diffusion policy `predict_action()` methods accept `generator=None` and forward it to `conditional_sample()` to ensure reproducible rollouts.

## Early stopping

Configured via `training.early_stop_patience` (default 500), `early_stop_min_delta` (0.005), and `early_stop_metric` ("test/mean_score"). State tracked via `_no_improve_count`, `_best_score`, `_best_epoch` — all included in checkpoint `include_keys` for correct resume behavior.

## GPU utilization notes

Push-T is a lightweight task (28 batches/epoch with 90 episodes). The dominant overheads are:
- DDP gradient all-reduce synchronization between batches
- Environment rollout (pygame physics + video encoding on CPU, ~2 min every 50 epochs)
- Validation (rank-0-only forward pass, other GPUs idle)
- Short epochs amplify these fixed costs

Config optimizations applied: `batch_size=128`, `val_every=5`, `num_workers=12` improve utilization from ~50% to ~85-95%.
