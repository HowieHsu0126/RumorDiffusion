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

INPUT_PATH=${INPUT_PATH:-"Output/explain/twitter15_rd.jsonl"}
OUT_PATH=${OUT_PATH:-"Output/explain/twitter15_rd_summary.csv"}
TOPK=${TOPK:-20}

echo "Summarizing explain outputs:"
echo "  input: ${INPUT_PATH}"
echo "  out:   ${OUT_PATH}"
echo "  topk:  ${TOPK}"

python3 -m Libs.exp.explain_summary \
  --input "${INPUT_PATH}" \
  --out "${OUT_PATH}" \
  --topk "${TOPK}"

