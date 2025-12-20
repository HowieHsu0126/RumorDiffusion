#!/bin/bash
#SBATCH --job-name=motivation
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/motivation_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

CONFIG=${CONFIG:-"Libs/configs/motivation.yaml"}
EXPERIMENT=${EXPERIMENT_OVERRIDE:-""}  # nodecount / timecutoff / both

CMD="python3 -m Libs.exp.motivation_experiment --config ${CONFIG}"
if [ -n "${EXPERIMENT}" ]; then
  CMD+=" --experiment ${EXPERIMENT}"
fi

echo "Running motivation experiment:"
echo "  config: ${CONFIG}"
echo "  experiment: ${EXPERIMENT:-auto}"

eval ${CMD}

