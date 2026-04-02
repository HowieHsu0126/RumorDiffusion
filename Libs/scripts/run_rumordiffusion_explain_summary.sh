#!/bin/bash
#SBATCH --job-name=rd-explain-summary
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/rd_explain_summary_%j.out

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus

EXP_CFG=${EXP_CFG:-"Libs/configs/exp.yaml"}
DATASET_NAME=${DATASET_NAME_OVERRIDE:-$(python3 - "${EXP_CFG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("dataset", {}).get("name", "twitter15"))
PY
)}
RESULTS_DIR=$(python3 - "${EXP_CFG}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("results_dir", os.path.join("Output", "results")))
PY
)
INPUT_PATH=${INPUT_PATH:-"${RESULTS_DIR}/${DATASET_NAME}_explain.jsonl"}
OUT_PATH=${OUT_PATH:-"${RESULTS_DIR}/${DATASET_NAME}_explain_summary.csv"}
TOPK=${TOPK:-20}

echo "Summarizing explain outputs:"
echo "  input: ${INPUT_PATH}"
echo "  out:   ${OUT_PATH}"
echo "  topk:  ${TOPK}"

python3 -m Libs.exp.explain_summary \
  --input "${INPUT_PATH}" \
  --out "${OUT_PATH}" \
  --topk "${TOPK}"
