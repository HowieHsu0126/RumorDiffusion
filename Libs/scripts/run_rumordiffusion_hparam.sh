#!/bin/bash
#SBATCH --job-name=rd-hparam
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_hparam_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG=${EXP_CFG:-"Libs/configs/exp.yaml"}
MODEL_CFGS_DEFAULT="Libs/configs/baselines/rumordiffusion.yaml,Libs/configs/baselines/rumordiffusion_hparam_depth.yaml,Libs/configs/baselines/rumordiffusion_hparam_guidance.yaml"
MODEL_CFGS="${MODEL_CFGS_OVERRIDE:-$MODEL_CFGS_DEFAULT}"
SEEDS_DEFAULT=$(python3 - "${EXP_CFG}" <<'PY'
import yaml
import sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
SEEDS="${SEEDS_OVERRIDE:-$SEEDS_DEFAULT}"
DATASET_OVERRIDE="${DATASET_OVERRIDE:-}"

CMD="python3 -m Libs.exp.run --exp ${EXP_CFG} --model ${MODEL_CFGS} --seeds ${SEEDS}"
if [ -n "$DATASET_OVERRIDE" ]; then
  CMD+=" --dataset_override ${DATASET_OVERRIDE}"
fi
echo "Running: $CMD"
eval $CMD
