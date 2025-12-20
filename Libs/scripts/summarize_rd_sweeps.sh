#!/bin/bash

# Aggregate mean±std for rd_sweeps.csv over specified datasets/models/seeds.
# Usage: optional env overrides:
#   SEEDS_OVERRIDE (default "42,3047,126")
#   DATASETS_OVERRIDE (default "pheme,weibo,twitter15,twitter16")
#   MODELS_OVERRIDE (default "rumordiffusion")
#   CSV_PATH (default "Output/results/rd_sweeps.csv")

set -e

SEEDS="${SEEDS_OVERRIDE:-42,3047,126}"
DATASETS="${DATASETS_OVERRIDE:-pheme,weibo,twitter15,twitter16}"
MODELS="${MODELS_OVERRIDE:-rumordiffusion}"
CSV_PATH="${CSV_PATH:-Output/results/rd_sweeps.csv}"

python3 - <<'PY'
import csv
import math
import os
import re
from datetime import datetime

csv_path = os.environ.get("CSV_PATH", "Output/results/rd_sweeps.csv")
seeds_env = os.environ.get("SEEDS_OVERRIDE", "42,3047,126")
datasets_env = os.environ.get("DATASETS_OVERRIDE", "pheme,weibo,twitter15,twitter16")
models_env = os.environ.get("MODELS_OVERRIDE", "rumordiffusion")

seeds = [s.strip() for s in seeds_env.split(",") if s.strip()]
datasets = [d.strip() for d in datasets_env.split(",") if d.strip()]
models = [m.strip() for m in models_env.split(",") if m.strip()]
seeds_tag = "_".join(seeds) + "_meanstd"

numeric_cols = [
    "best_val_macro_f1",
    "test_loss",
    "test_nll",
    "test_acc",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_weighted_f1",
    "test_weighted_precision",
    "test_weighted_recall",
    "ece",
    "calibration_temp",
]

def parse_float(text: str):
    s = str(text).strip()
    if s == "":
        return None
    if not re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", s, re.IGNORECASE):
        return None
    return float(s)

def mean_std(values):
    if len(values) == 0:
        return ""
    mean_val = sum(values) / float(len(values))
    var = sum((v - mean_val) ** 2 for v in values) / float(len(values))
    std_val = math.sqrt(var)
    return f"{mean_val:.6f}±{std_val:.6f}"

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    data = list(reader)

rows_out = []
now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for ds in datasets:
    for model in models:
        filtered = [r for r in data if r.get("dataset") == ds and r.get("model") == model and r.get("seed") in seeds]
        if len(filtered) == 0:
            continue
        agg = {}
        for col in numeric_cols:
            vals = [parse_float(r.get(col, "")) for r in filtered]
            vals = [v for v in vals if v is not None]
            agg[col] = mean_std(vals)
        row = {
            "timestamp": now_ts,
            "dataset": ds,
            "model": model,
            "model_cfg": filtered[0].get("model_cfg", ""),
            "exp_cfg": filtered[0].get("exp_cfg", ""),
            "ckpt_path": "aggregate",
            "log_path": "aggregate_meanstd",
            "seed": seeds_tag,
            "best_val_macro_f1": agg.get("best_val_macro_f1", ""),
            "test_loss": agg.get("test_loss", ""),
            "test_nll": agg.get("test_nll", ""),
            "test_acc": agg.get("test_acc", ""),
            "test_macro_f1": agg.get("test_macro_f1", ""),
            "test_macro_precision": agg.get("test_macro_precision", ""),
            "test_macro_recall": agg.get("test_macro_recall", ""),
            "test_weighted_f1": agg.get("test_weighted_f1", ""),
            "test_weighted_precision": agg.get("test_weighted_precision", ""),
            "test_weighted_recall": agg.get("test_weighted_recall", ""),
            "ece": agg.get("ece", ""),
            "calibration_temp": agg.get("calibration_temp", ""),
        }
        rows_out.append(row)

if len(rows_out) > 0:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        for r in rows_out:
            writer.writerow(r)
PY

