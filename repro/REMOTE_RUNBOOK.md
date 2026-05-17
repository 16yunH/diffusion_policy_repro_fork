# Push-T Image Pretrained Evaluation Runbook

This runbook is for the first reproduction target: evaluating the official
Push-T image Diffusion Policy checkpoint on an 8x RTX 4090 CUDA Linux server.

## 1. Prepare the server

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
df -h .
git --version
wget --version
conda --version
```

The pretrained Push-T image eval is a single-process run and only uses one GPU.
On an 8x RTX 4090 server, choose an idle GPU with `DEVICE=cuda:<index>`. The eval
downloads a checkpoint of about 3GB and creates rollout videos. Keep at least
20GB free in the working filesystem.

## 2. Clone this workspace on the server

```bash
git clone https://github.com/real-stanford/diffusion_policy.git diffusion_policy
cd diffusion_policy
```

Copy the local `repro/` directory from this workspace into the cloned server
repository, or commit/push these repro files to a private fork and clone that
fork directly.

## 3. Run the eval

```bash
bash repro/run_pusht_image_pretrained_eval.sh
```

Useful overrides:

```bash
DEVICE=cuda:1 bash repro/run_pusht_image_pretrained_eval.sh
CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 bash repro/run_pusht_image_pretrained_eval.sh
MIN_GPU_COUNT=8 bash repro/run_pusht_image_pretrained_eval.sh
SKIP_APT=1 bash repro/run_pusht_image_pretrained_eval.sh
SKIP_ENV=1 bash repro/run_pusht_image_pretrained_eval.sh
FORCE_DOWNLOAD=1 bash repro/run_pusht_image_pretrained_eval.sh
```

On a shared 8x RTX 4090 server where only physical GPU 5 is idle and the user
has no sudo rights, use:

```bash
SKIP_APT=1 CUDA_VISIBLE_DEVICES=5 DEVICE=cuda:0 bash repro/run_pusht_image_pretrained_eval.sh
```

With `CUDA_VISIBLE_DEVICES=5`, the process sees physical GPU 5 as logical
`cuda:0`, so keep `DEVICE=cuda:0`.

For the initial reproduction, do not launch eight copies at once. First complete
one clean eval and verify `eval_log.json`, videos, and the summary. After that,
the same checkpoint eval can be repeated on other GPUs for environment
diagnostics, but it is not a replacement for multi-seed training.

## 4. Expected outputs

```text
data/eval/image_pusht_dp_cnn_train0/eval_log.json
data/eval/image_pusht_dp_cnn_train0/media/*.mp4
data/eval/image_pusht_dp_cnn_train0/run.log
data/repro/env_info.txt
data/repro/git_commit.txt
data/repro/pusht_image_pretrained_eval_summary.md
```

The official Push-T image train_0 checkpoint used by the script is:

```text
https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt
```

The summary script compares `eval_log.json` key `test/mean_score` against the
`0.884` value encoded in the checkpoint filename with a default tolerance of
`0.05`.

## 5. Conda metadata is slow or stuck

If environment creation stalls at `Collecting package metadata`, stop it with
`Ctrl+C` and configure a user-level conda mirror. This does not need sudo:

```bash
cat > ~/.condarc <<'EOF'
channels:
  - pytorch
  - pytorch3d
  - nvidia
  - conda-forge
  - defaults
show_channel_urls: true
channel_priority: flexible
remote_connect_timeout_secs: 20
remote_read_timeout_secs: 120
repodata_fns:
  - current_repodata.json
  - repodata.json
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

conda clean -i -y
```

Then retry:

```bash
SKIP_APT=1 CUDA_VISIBLE_DEVICES=5 DEVICE=cuda:0 bash repro/run_pusht_image_pretrained_eval.sh
```

If the environment already exists after a partial install, skip creation and run
the remaining steps with:

```bash
SKIP_APT=1 SKIP_ENV=1 CUDA_VISIBLE_DEVICES=5 DEVICE=cuda:0 bash repro/run_pusht_image_pretrained_eval.sh
```

## 6. Later multi-GPU extension

The 8x RTX 4090 server is useful after this first checkpoint-eval milestone:

- For official-style multi-seed training, allocate one GPU per seed with Ray or
  independent processes.
- Start with three seeds to match the paper convention, then scale only if the
  training logs and eval videos look stable.
- Keep pretrained eval artifacts separate from training artifacts, for example
  `data/eval/...` versus `data/outputs/...`.

On the shared server, if GPUs 1, 3, and 4 are idle, launch three independent
Push-T image training runs with:

```bash
conda activate robodiff
GPUS=1,3,4 SEEDS=42,43,44 bash repro/launch_pusht_image_train3.sh
```

For a short smoke test first:

```bash
SMOKE=1 GPUS=1 SEEDS=42 bash repro/launch_pusht_image_train3.sh
```

Monitor:

```bash
tmux ls
tmux attach -t dp_pusht_s42
tail -f data/outputs/pusht_image_train3_*/seed_42/train.log
python repro/summarize_pusht_train_runs.py data/outputs/pusht_image_train3_*
```

If the goal is to finish one seed first, run a single training process on one
idle physical GPU. The script below defaults to seed 42 on GPU 1 and will resume
from the newest existing `seed_42/checkpoints/latest.ckpt` if one exists:

```bash
conda activate robodiff
GPU=1 SEED=42 bash repro/launch_pusht_image_single_seed.sh
```

If GPU 1 becomes busy, use another idle GPU, for example:

```bash
GPU=5 SEED=42 bash repro/launch_pusht_image_single_seed.sh
```

Monitor:

```bash
tmux attach -t dp_pusht_single_s42
tail -f data/outputs/pusht_image_train3_*/seed_42/train.log
python repro/summarize_pusht_train_runs.py data/outputs/pusht_image_train3_*
```
