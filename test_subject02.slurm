#!/bin/bash
#SBATCH -J mmlphuman_test_s02
#SBATCH -p ai_training
#SBATCH -A ai_training
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=dx-ai-node15,dx-ai-node18
#SBATCH -o /home/hanlong/project/mmlphuman/logs/%x-%j.out

set -euo pipefail

cd /home/hanlong/project/mmlphuman
mkdir -p logs render/subject02_trainview

export PYTHONUNBUFFERED=1

/home/hanlong/miniconda3/bin/conda run --no-capture-output -n mmlphuman python -u test.py \
  --config ./config/subject02.yaml \
  --model_dir ./output/subject02 \
  --out_dir ./render/subject02_trainview \
  --data_dir ./subject02
