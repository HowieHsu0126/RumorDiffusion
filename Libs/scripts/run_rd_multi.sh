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

EXP_CFG="${EXP_CFG:-Libs/configs/exp.yaml}"
MODEL_CFG="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
SEEDS_DEFAULT=$(python3 - "${EXP_CFG}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
SEEDS="${SEEDS_OVERRIDE:-$SEEDS_DEFAULT}"
DATASETS=(${DATASETS_OVERRIDE:-twitter15 twitter16 weibo pheme})

for ds in "${DATASETS[@]}"; do
  echo "==== Running RumorDiffusion on ${ds} with seeds ${SEEDS} ===="
  python3 -m Libs.exp.run \
    --exp "${EXP_CFG}" \
    --model "${MODEL_CFG}" \
    --seeds "${SEEDS}" \
    --dataset_override "${ds}"
done
