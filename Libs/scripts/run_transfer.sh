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
SEEDS_DEFAULT=$(python3 - "${EXP_CFG}" <<'PY'
import yaml
import sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
SEEDS="${SEEDS_OVERRIDE:-$SEEDS_DEFAULT}"

CMD="python3 -m Libs.exp.run --exp ${EXP_CFG} --model ${MODEL_CFG} --seeds ${SEEDS} --transfer_eval --lp_edges_policy require_existing"
echo "Running transfer_eval using config ${EXP_CFG}, seeds=${SEEDS}"
eval ${CMD}
