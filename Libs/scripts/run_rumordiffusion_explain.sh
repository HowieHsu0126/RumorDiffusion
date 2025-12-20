#!/bin/bash
#SBATCH --job-name=rd-explain
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_explain_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG="Libs/configs/exp.yaml"
MODEL_CFG="Libs/configs/baselines/rumordiffusion.yaml"
CKPT_PATH="Output/ckpts/twitter15/rumordiffusion/best.pt"
OUT_PATH="Output/explain/twitter15_rd.jsonl"
SPLIT="test"

python3 -m Libs.exp.rumordiffusion_explain \
  --exp "${EXP_CFG}" \
  --model "${MODEL_CFG}" \
  --ckpt "${CKPT_PATH}" \
  --split "${SPLIT}" \
  --out "${OUT_PATH}"
