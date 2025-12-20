# RumorDiffusion: Prefix-Conditioned Diffusion for Early Rumor Detection Under Sparse and Incomplete Social Cascades

<!-- Highlights
--
- All knobs live in YAML (`Libs/configs/exp*.yaml`, `Libs/configs/baselines/*.yaml`); no hardcoded params.
- Supports multi-dataset (Twitter15/16 4-class, Weibo/PHEME 2-class) via `Libs/configs/dataset.yaml`.
- One entrypoint runner (`python -m Libs.exp.run`) for RumorDiffusion and baselines; optional SLURM helper scripts.
- Robustness/early-observation sweeps, calibration, and interpretability utilities included. -->

## Repository Layout
-
- `Input/raw/` — raw datasets (Twitter-style: `tree/*.txt`, `label.txt`, `source_tweets.txt`; JSON-style: `.json` with `source` + `comment`).
- `Input/processed/<dataset>/` — processed PyG data (`pyg/data.pt`, `pyg/slices.pt`, `splits.json`, optional `text_embs.pt`).
- `Input/sentence_transformers_all_MiniLM_L6v2/` — local text encoder checkpoint used by preprocessing (adjust path in YAML if different).
- `Libs/configs/` — experiment configs (`exp.yaml`, `exp_smoke.yaml`), dataset config, and model/baseline configs in `baselines/`.
- `Libs/exp/` — training/eval runner (`run.py`) and analysis tools.
- `Libs/data/` — dataset loader (`dataset.py`) and preprocessor (`processor.py`).
- `Libs/models/` — RumorDiffusion and baseline models.
- `Libs/scripts/` — SLURM helpers for benchmarks/ablation.
- `Output/` — checkpoints (`ckpts/`), logs (`logs/`), and results (`results/*.csv`, explain outputs).

## Installation

----
1) Install PyTorch and PyG matching your CUDA/CPU setup (follow official install guides).  
2) Install Python deps:
```
pip install pyyaml tqdm transformers sentence-transformers wandb
```
3) If you use SLURM helpers, ensure `sbatch` is available.

## Data

----
Download all data (raw data, processed data, and sentence transformer weights) from [Google Drive](https://drive.google.com/file/d/1rSFOLfTwe-TWrCEIPfYK7j4X_kosaX28/view?usp=sharing) and extract to `Input/` directory.

The extracted structure should include:
- `Input/raw/` — raw datasets (Twitter-style: `tree/*.txt`, `label.txt`, `source_tweets.txt`; JSON-style: `.json` with `source` + `comment`).
- `Input/processed/<dataset>/` — processed PyG data (`pyg/data.pt`, `pyg/slices.pt`, `splits.json`, optional `text_embs.pt`).
- `Input/sentence_transformers_all_MiniLM_L6v2/` — text encoder checkpoint used by preprocessing (adjust path in YAML if different).

## Preprocessing


Generate processed PyG data from raw:
```
python -m Libs.data.processor \
  --config Libs/configs/dataset.yaml \
  --dataset twitter15
```
Regenerate only `splits.json` using existing processed data:
```
python -m Libs.data.processor \
  --config Libs/configs/dataset.yaml \
  --dataset twitter15 \
  --regen_splits_only
```
Adjust `--dataset` to one of `twitter15`, `twitter16`, `weibo`, `pheme`. Update paths in YAML if your `Input/` location differs.

Quickstart (smoke test)
-------
Runs 1 epoch, 1 batch, 1 sample (per `exp_smoke.yaml`), RumorDiffusion model:
```
python -m Libs.exp.run \
  --exp Libs/configs/exp_smoke.yaml \
  --model Libs/configs/baselines/rumordiffusion.yaml
```
Edit the YAML paths (`dataset.root`, `outputs.*`) if your data/output directories differ.

## Training & Evaluation

-----
Single model, multi-seed (defaults from `exp.yaml`):
```
python -m Libs.exp.run \
  --exp Libs/configs/exp.yaml \
  --model Libs/configs/baselines/rumordiffusion.yaml
```
Override dataset or seeds:
```
python -m Libs.exp.run \
  --exp Libs/configs/exp.yaml \
  --model Libs/configs/baselines/rumordiffusion.yaml \
  --dataset_override twitter16 \
  --seeds 42,43,44
```
Run multiple models in one go (comma-separated YAMLs):
```
python -m Libs.exp.run \
  --exp Libs/configs/exp.yaml \
  --model Libs/configs/baselines/bigcn.yaml,Libs/configs/baselines/gcn.yaml
```
Eval-only with a checkpoint:
```
python -m Libs.exp.run \
  --exp Libs/configs/exp.yaml \
  --model Libs/configs/baselines/rumordiffusion.yaml \
  --eval_only \
  --ckpt Output/ckpts/twitter15/rumordiffusion/best.pt
```
Outputs:
- Checkpoints: `Output/ckpts/<dataset>/<model>/seed*/best.pt`
- Logs: `Output/logs/`
- Metrics: `Output/results/<dataset>.csv` (+ `rd_sweeps.csv`, `transfer.csv` when applicable)

## Benchmark scripts (SLURM, optional)

----
Helper scripts read `Libs/configs/exp.yaml` and submit jobs. Example (Twitter15):
```
sbatch Libs/scripts/run_bench_twitter15.sh
```
You can override env vars (e.g., `SEEDS_OVERRIDE`, `MODELS_OVERRIDE`, `DATASETS_OVERRIDE`) when submitting. Use these only in SLURM environments; otherwise call `python -m Libs.exp.run` directly.

Interpretability export (optional)
--
Generate per-sample logits/probs and token stats for a trained RumorDiffusion checkpoint:
```
python -m Libs.exp.rumordiffusion_explain \
  --exp Libs/configs/exp.yaml \
  --model Libs/configs/baselines/rumordiffusion.yaml \
  --ckpt Output/ckpts/twitter15/rumordiffusion/best.pt \
  --split test \
  --out Output/explain/twitter15_rd.jsonl
```
Summarize top confident/uncertain samples:
```
python -m Libs.exp.explain_summary \
  --input Output/explain/twitter15_rd.jsonl \
  --out Output/explain/twitter15_rd_summary.csv \
  --topk 20
```

## Citation


This work is under review (ACM T-IST). Citation details will be added when available.