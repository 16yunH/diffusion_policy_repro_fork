param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [string]$RemoteDir = "~/diffusion_policy",
    [string]$Device = "cuda:0",
    [switch]$SkipApt,
    [switch]$SkipEnv
)

$ErrorActionPreference = "Stop"

Write-Host "Creating remote directory $RemoteDir on $Remote"
ssh $Remote "mkdir -p $RemoteDir"

Write-Host "Syncing repro scripts"
scp -r "$PSScriptRoot" "${Remote}:$RemoteDir/"

Write-Host "Ensuring official repository exists on remote"
ssh $Remote @"
set -e
remote_dir="\$(eval echo "$RemoteDir")"
mkdir -p "\$remote_dir"
if [ ! -d "\$remote_dir/.git" ]; then
  tmp_dir="\$(mktemp -d)"
  git clone --depth 1 https://github.com/real-stanford/diffusion_policy.git "\$tmp_dir"
  cp -a "\$tmp_dir"/. "\$remote_dir"/
  rm -rf "\$tmp_dir"
fi
"@

$skipAptValue = if ($SkipApt) { "1" } else { "0" }
$skipEnvValue = if ($SkipEnv) { "1" } else { "0" }

Write-Host "Launching remote eval"
ssh $Remote "remote_dir=\$(eval echo '$RemoteDir') && cd \"\$remote_dir\" && DEVICE=$Device SKIP_APT=$skipAptValue SKIP_ENV=$skipEnvValue bash repro/run_pusht_image_pretrained_eval.sh"
