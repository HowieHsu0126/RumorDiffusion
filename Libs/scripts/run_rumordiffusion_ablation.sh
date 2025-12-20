#!/bin/bash
#SBATCH --job-name=rd-ablate
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_ablate_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG="Libs/configs/exp.yaml"
MODEL_CFGS="Libs/configs/baselines/rumordiffusion.yaml,Libs/configs/baselines/rumordiffusion_ablate_ddgan.yaml,Libs/configs/baselines/rumordiffusion_ablate_pair.yaml,Libs/configs/baselines/rumordiffusion_ablate_text.yaml,Libs/configs/baselines/rumordiffusion_ablate_temporal.yaml"
SEEDS="${SEEDS_OVERRIDE:-42,43,44}"
DATASET_OVERRIDE="${DATASET_OVERRIDE:-}"

CMD="python3 -m Libs.exp.run --exp ${EXP_CFG} --model ${MODEL_CFGS} --seeds ${SEEDS}"
if [ -n "$DATASET_OVERRIDE" ]; then
  CMD+=" --dataset_override ${DATASET_OVERRIDE}"
fi
echo "Running: $CMD"
eval $CMD
