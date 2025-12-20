#!/bin/bash

#!/bin/bash

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  PARTITION=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
print(cfg.get("sbatch", {}).get("partition", "gpu1"))
PY
)
  NODELIST=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
print(cfg.get("sbatch", {}).get("nodelist", ""))
PY
)
  GRES=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
print(cfg.get("sbatch", {}).get("gres", "gpu:1"))
PY
)
  TIME=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
print(cfg.get("sbatch", {}).get("time", "48:00:00"))
PY
)
  LOG_DIR=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
print(cfg.get("sbatch", {}).get("log_dir", "Output/logs"))
PY
)
  CUDA_DEV=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
devs = cfg.get("sbatch", {}).get("cuda_devices", [0,1,2,3])
print(devs[2] if len(devs) > 2 else (devs[-1] if len(devs) > 0 else 0))
PY
)
  SEEDS=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3047, 126])
print(",".join(str(s) for s in seeds))
PY
)
  MODELS=$(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open("Libs/configs/exp.yaml"))
models_cfg = cfg.get("bench_models", {})
models = models_cfg.get("twitter15", models_cfg.get("default", ["rumordiffusion"]))
print(" ".join(models))
PY
)
  mkdir -p "${LOG_DIR}"
  sbatch \
    --job-name=bench_twitter15 \
    --partition="${PARTITION}" \
    ${NODELIST:+--nodelist=${NODELIST}} \
    --gres="${GRES}" \
    --time="${TIME}" \
    --output="${LOG_DIR}/bench_twitter15_%j.log" \
    --error="${LOG_DIR}/bench_twitter15_%j.err" \
    --export=ALL,SEEDS_OVERRIDE="${SEEDS}",MODELS_OVERRIDE="${MODELS}",CUDA_VISIBLE_DEVICES=${CUDA_DEV} \
    Libs/scripts/run_bench_twitter15.sh sbatch
  exit $?
fi

source /share/home/hwxu/miniconda3/bin/activate hw
cd /share/home/hwxu/Researches/D2Plus
mkdir -p Output/logs

SEEDS="${SEEDS_OVERRIDE:-42,3047,126}"
MODELS=(${MODELS_OVERRIDE})

CONFIGS=""
for model in "${MODELS[@]}"; do
    if [ -z "$CONFIGS" ]; then
        CONFIGS="Libs/configs/baselines/${model}.yaml"
    else
        CONFIGS="${CONFIGS},Libs/configs/baselines/${model}.yaml"
    fi
done

echo "Starting benchmark for twitter15 dataset"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS}"
echo "=========================================="

python3 -m Libs.exp.run \
    --exp Libs/configs/exp.yaml \
    --model "$CONFIGS" \
    --dataset_override "twitter15" \
    --seeds "${SEEDS}"

