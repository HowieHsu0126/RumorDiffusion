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

CMD="python3 -m Libs.exp.run --exp ${EXP_CFG} --model ${MODEL_CFG} --seeds ${SEEDS} --transfer_eval --lp_edges_policy require_existing"
echo "Running transfer_eval using config ${EXP_CFG}, seeds=${SEEDS}"
eval ${CMD}

