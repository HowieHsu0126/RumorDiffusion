#!/bin/bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  PARTITION=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("partition", "cpu"))
PY
)
  NODELIST=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("nodelist", ""))
PY
)
  GRES=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("gres", ""))
PY
)
  TIME=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("time", "48:00:00"))
PY
)
  LOG_DIR=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("log_dir", "Output/logs"))
PY
)
  SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  DATASETS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
datasets = cfg.get("multi_run", {}).get("datasets", ["twitter15", "twitter16", "weibo", "pheme"])
print(" ".join(str(ds) for ds in datasets))
PY
)
  DATASETS_VALUE="${DATASETS_OVERRIDE:-${DATASETS_DEFAULT}}"
  MODEL_CFG_VALUE="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
  mkdir -p "${LOG_DIR}"
  sbatch \
    --job-name=rd_multi \
    --partition="${PARTITION}" \
    ${NODELIST:+--nodelist=${NODELIST}} \
    ${GRES:+--gres=${GRES}} \
    --time="${TIME}" \
    --output="${LOG_DIR}/rd_multi_%j.log" \
    --error="${LOG_DIR}/rd_multi_%j.err" \
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",MODEL_CFG="${MODEL_CFG_VALUE}",SEEDS_OVERRIDE="${SEEDS}",DATASETS_OVERRIDE="${DATASETS_VALUE}" \
    Libs/scripts/run_rd_multi.sh sbatch
  exit $?
fi

set +u
source /share/home/hwxu/miniconda3/bin/activate hw
set -u
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG="${EXP_CFG_DEFAULT}"
MODEL_CFG="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
SEEDS_DEFAULT=$(python3 - "${EXP_CFG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(str(s) for s in cfg.get("experiment", {}).get("seeds", [42])))
PY
)
SEEDS="${SEEDS_OVERRIDE:-$SEEDS_DEFAULT}"
DATASETS=(${DATASETS_OVERRIDE:-twitter15 twitter16 weibo pheme})

for ds in "${DATASETS[@]}"; do
  log "Dataset: ${ds}; EXP_CFG: ${EXP_CFG}; MODEL_CFG: ${MODEL_CFG}; Seeds: ${SEEDS}"
  log "Python: $(python3 --version)"
  log "Git HEAD: $(git rev-parse HEAD)"
  python3 -m Libs.exp.run \
    --exp "${EXP_CFG}" \
    --model "${MODEL_CFG}" \
    --seeds "${SEEDS}" \
    --dataset_override "${ds}"
done
