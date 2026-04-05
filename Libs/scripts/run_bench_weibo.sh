#!/bin/bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  PARTITION=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("partition", "gpu1"))
PY
)
  NODELIST=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("nodelist", ""))
PY
)
  GRES=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("gres", "gpu:1"))
PY
)
  TIME=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("time", "48:00:00"))
PY
)
  LOG_DIR=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("sbatch", {}).get("log_dir", "Output/logs"))
PY
)
  CUDA_DEV=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
devs = cfg.get("sbatch", {}).get("cuda_devices", [0,1,2,3])
print(devs[1] if len(devs) > 1 else (devs[0] if len(devs) > 0 else 0))
PY
)
  SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3047, 126])
print(",".join(str(s) for s in seeds))
PY
)
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  MODELS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import yaml
import sys

cfg = yaml.safe_load(open(sys.argv[1]))
models_cfg = cfg.get("bench_models", {})
models = models_cfg.get("weibo", models_cfg.get("default", ["rumordiffusion"]))
print(" ".join(models))
PY
)
  MODELS="${MODELS_OVERRIDE:-${MODELS_DEFAULT}}"
  EVAL_ONLY="${EVAL_ONLY_OVERRIDE:-0}"
  CKPT_OVERRIDE_VALUE="${CKPT_OVERRIDE:-}"
  mkdir -p "${LOG_DIR}"
  sbatch \
    --job-name=bench_weibo \
    --partition="${PARTITION}" \
    ${NODELIST:+--nodelist=${NODELIST}} \
    ${GRES:+--gres=${GRES}} \
    --time="${TIME}" \
    --output="${LOG_DIR}/bench_weibo_%j.log" \
    --error="${LOG_DIR}/bench_weibo_%j.err" \
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",SEEDS_OVERRIDE="${SEEDS}",MODELS_OVERRIDE="${MODELS}",EVAL_ONLY_OVERRIDE="${EVAL_ONLY}",CKPT_OVERRIDE="${CKPT_OVERRIDE_VALUE}",CUDA_VISIBLE_DEVICES=${CUDA_DEV} \
    Libs/scripts/run_bench_weibo.sh sbatch
  exit $?
fi

set +u
source /share/home/hwxu/miniconda3/bin/activate hw
set -u
cd /share/home/hwxu/Researches/D2Plus
mkdir -p Output/logs

SEEDS="${SEEDS_OVERRIDE:-42,3047,126}"
MODELS=(${MODELS_OVERRIDE})
EVAL_ONLY="${EVAL_ONLY_OVERRIDE:-0}"
CKPT_OVERRIDE_VALUE="${CKPT_OVERRIDE:-}"

CONFIGS=""
for model in "${MODELS[@]}"; do
    if [ -z "$CONFIGS" ]; then
        CONFIGS="Libs/configs/baselines/${model}.yaml"
    else
        CONFIGS="${CONFIGS},Libs/configs/baselines/${model}.yaml"
    fi
done

log "Dataset: weibo"
log "EXP_CFG: ${EXP_CFG_DEFAULT}"
log "Models: ${MODELS[*]}"
log "Seeds: ${SEEDS}"
log "Python: $(python3 --version)"
log "Git HEAD: $(git rev-parse HEAD)"

RUN_ARGS=()
if [ "${EVAL_ONLY}" = "1" ]; then
    RUN_ARGS+=(--eval_only)
fi
if [ -n "${CKPT_OVERRIDE_VALUE}" ]; then
    RUN_ARGS+=(--ckpt "${CKPT_OVERRIDE_VALUE}")
fi

python3 -m Libs.exp.run \
    --exp "${EXP_CFG_DEFAULT}" \
    --model "$CONFIGS" \
    --dataset_override "weibo" \
    --seeds "${SEEDS}" \
    "${RUN_ARGS[@]}"
