#!/bin/bash
#SBATCH --job-name=rd-multi
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_multi_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG="Libs/configs/exp.yaml"
MODEL_CFG="Libs/configs/baselines/rumordiffusion.yaml"
SEEDS="${SEEDS_OVERRIDE:-42,43,44}"
DATASETS=(${DATASETS_OVERRIDE:-twitter15 twitter16 weibo pheme})

for ds in "${DATASETS[@]}"; do
  echo "==== Running RumorDiffusion on ${ds} with seeds ${SEEDS} ===="
  python3 -m Libs.exp.run \
    --exp "${EXP_CFG}" \
    --model "${MODEL_CFG}" \
    --seeds "${SEEDS}" \
    --dataset_override "${ds}"
done

