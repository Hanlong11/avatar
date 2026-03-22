#!/bin/bash
#SBATCH -J mmlphuman_novel_s02
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
mkdir -p logs render/subject02_novelviewpose

export PYTHONUNBUFFERED=1

/home/hanlong/miniconda3/bin/conda run --no-capture-output -n mmlphuman python -u script/generate_subject02_novel_camera.py

/home/hanlong/miniconda3/bin/conda run --no-capture-output -n mmlphuman python -u test.py \
  --config ./config/subject02_novel.yaml \
  --model_dir ./output/subject02 \
  --out_dir ./render/subject02_novelviewpose \
  --cam_path ./render/subject02_novelviewpose/cam_novel_mid_00_01.json \
  --pose_path ./subject02/smpl_params.npz \
  --test
