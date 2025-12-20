#!/bin/bash
#SBATCH --job-name=rd-transfer
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_transfer_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG=${EXP_CFG:-"Libs/configs/exp.yaml"}
MODEL_CFG=${MODEL_CFG:-"Libs/configs/baselines/rumordiffusion.yaml"}
SEEDS="${SEEDS_OVERRIDE:-42,43,44}"

# Source ckpt and target datasets list
CKPT_PATH=${CKPT_PATH:-"Output/ckpts/twitter15/rumordiffusion/best.pt"}
TARGETS=(${TARGETS_OVERRIDE:-twitter16 weibo pheme})

for tgt in "${TARGETS[@]}"; do
  CMD="python3 -m Libs.exp.run --exp ${EXP_CFG} --model ${MODEL_CFG} --seeds ${SEEDS} --eval_only --ckpt ${CKPT_PATH} --dataset_override ${tgt}"
  echo "Running transfer: source_ckpt=${CKPT_PATH} -> target=${tgt} seeds=${SEEDS}"
  eval ${CMD}
done

