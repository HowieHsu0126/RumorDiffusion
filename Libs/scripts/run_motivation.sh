#!/bin/bash
set -euo pipefail
#SBATCH --job-name=motivation
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/hwxu/Researches/D2Plus/Output/logs/motivation_%j.out

set +u
source /share/home/hwxu/miniconda3/bin/activate hw
set -u
cd /share/home/hwxu/Researches/D2Plus

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

CONFIG=${CONFIG:-"Libs/configs/motivation.yaml"}
EXPERIMENT=${EXPERIMENT_OVERRIDE:-""}

ARGS=(python3 -m Libs.exp.motivation_experiment --config "${CONFIG}")
if [ -n "${EXPERIMENT}" ]; then
  ARGS+=(--experiment "${EXPERIMENT}")
fi

log "Running motivation experiment"
log "Config: ${CONFIG}"
log "Experiment: ${EXPERIMENT:-auto}"
log "Python: $(python3 --version)"
log "Git HEAD: $(git rev-parse HEAD)"
"${ARGS[@]}"
