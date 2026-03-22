#!/bin/bash
#SBATCH -J mmlphuman_s02
#SBATCH -p ai_training
#SBATCH -A ai_training
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --exclude=dx-ai-node15,dx-ai-node18
#SBATCH -o /home/hanlong/project/mmlphuman/logs/%x-%j.out

set -euo pipefail

cd /home/hanlong/project/mmlphuman
mkdir -p logs output/subject02

export PYTHONUNBUFFERED=1

resume_args=()
latest_ckpt=$(find ./output/subject02 -maxdepth 1 -type f -name 'chkpnt*.pth' | sort -V | tail -n 1)
if [[ -n "${latest_ckpt}" ]]; then
  echo "Resuming from ${latest_ckpt}"
  resume_args+=(--resume_checkpoint "${latest_ckpt}")
fi

/home/hanlong/miniconda3/bin/conda run --no-capture-output -n mmlphuman python -u train.py \
  --config ./config/subject02.yaml \
  --data_dir ./subject02 \
  --out_dir ./output/subject02 \
  "${resume_args[@]}"
