#!/bin/bash
set -euo pipefail
#SBATCH --job-name=rd-ablate
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_ablate_%j.out

set +u
source /share/home/hwxu/miniconda3/bin/activate hw
set -u
cd /share/home/hwxu/Researches/D2Plus

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

EXP_CFG=${EXP_CFG:-"Libs/configs/exp.yaml"}
MODEL_CFGS="${MODEL_CFGS_OVERRIDE:-Libs/configs/baselines/rumordiffusion.yaml,Libs/configs/baselines/rumordiffusion_ablate_ddgan.yaml,Libs/configs/baselines/rumordiffusion_ablate_pair.yaml,Libs/configs/baselines/rumordiffusion_ablate_text.yaml,Libs/configs/baselines/rumordiffusion_ablate_temporal.yaml}"
SEEDS_DEFAULT=$(python3 - "${EXP_CFG}" <<'PY'
import yaml
import sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
SEEDS="${SEEDS_OVERRIDE:-$SEEDS_DEFAULT}"
DATASET_OVERRIDE="${DATASET_OVERRIDE:-}"

ARGS=(python3 -m Libs.exp.run --exp "${EXP_CFG}" --model "${MODEL_CFGS}" --seeds "${SEEDS}")
if [ -n "$DATASET_OVERRIDE" ]; then
  ARGS+=(--dataset_override "${DATASET_OVERRIDE}")
fi
log "Running RumorDiffusion ablation"
log "EXP_CFG: ${EXP_CFG}"
log "MODEL_CFGS: ${MODEL_CFGS}"
log "Seeds: ${SEEDS}"
if [ -n "${DATASET_OVERRIDE}" ]; then
  log "Dataset override: ${DATASET_OVERRIDE}"
fi
log "Python: $(python3 --version)"
log "Git HEAD: $(git rev-parse HEAD)"
"${ARGS[@]}"
