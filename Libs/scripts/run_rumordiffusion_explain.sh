#!/bin/bash
set -euo pipefail
#SBATCH --job-name=rd-explain
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_explain_%j.out

set +u
source /share/home/hwxu/miniconda3/bin/activate hw
set -u
cd /share/home/hwxu/Researches/D2Plus

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

EXP_CFG=${EXP_CFG:-"Libs/configs/exp.yaml"}
MODEL_CFG=${MODEL_CFG:-"Libs/configs/baselines/rumordiffusion.yaml"}
DATASET_NAME=${DATASET_NAME_OVERRIDE:-$(python3 - "${EXP_CFG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("dataset", {}).get("name", "twitter15"))
PY
)}
SEED=${SEED_OVERRIDE:-$(python3 - "${EXP_CFG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42])
print(seeds[0] if seeds else 42)
PY
)}
CKPT_ROOT=$(python3 - "${EXP_CFG}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("ckpt_dir", os.path.join("Output", "ckpts")))
PY
)
RESULTS_DIR=$(python3 - "${EXP_CFG}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("results_dir", os.path.join("Output", "results")))
PY
)
CKPT_PATH=${CKPT_PATH_OVERRIDE:-"${CKPT_ROOT}/${DATASET_NAME}/rumordiffusion/seed${SEED}/best.pt"}
OUT_PATH=${OUT_PATH_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_explain.jsonl"}
TRACE_OUT=${TRACE_OUT_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_case_traces.jsonl"}
SELECTED_OUT=${SELECTED_OUT_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_selected_cases.json"}
SPLIT=${SPLIT_OVERRIDE:-test}
SOURCE_IDS=${SOURCE_IDS_OVERRIDE:-}

log "Running RumorDiffusion explain"
log "EXP_CFG: ${EXP_CFG}"
log "MODEL_CFG: ${MODEL_CFG}"
log "Checkpoint: ${CKPT_PATH}"
log "Output: ${OUT_PATH}"
log "Trace output: ${TRACE_OUT}"
log "Selected output: ${SELECTED_OUT}"
log "Split: ${SPLIT}"
log "Python: $(python3 --version)"
log "Git HEAD: $(git rev-parse HEAD)"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

ARGS=(
  --exp "${EXP_CFG}"
  --model "${MODEL_CFG}"
  --ckpt "${CKPT_PATH}"
  --split "${SPLIT}"
  --out "${OUT_PATH}"
  --trace_out "${TRACE_OUT}"
  --selected_out "${SELECTED_OUT}"
)

if [ -n "${SOURCE_IDS}" ]; then
  ARGS+=(--source_ids "${SOURCE_IDS}")
fi

python3 -m Libs.exp.rumordiffusion_explain \
  "${ARGS[@]}"
