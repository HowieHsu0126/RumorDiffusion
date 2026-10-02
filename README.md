# RumorDiffusion

**Prefix-conditioned diffusion for early rumor detection.**

Anonymous review documentation. The local workspace is named `D2Plus`.

RumorDiffusion models plausible future social cascades from sparse observed
prefixes and uses that representation for early rumor detection. The canonical
protocol consists of CTDM pretraining followed by an arbiter with frozen CTDM
parameters. Experimental variants must be identified separately.

## Protocol

| Component | Canonical specification |
| --- | --- |
| Datasets | Twitter15, Twitter16, Weibo, PHEME |
| Seeds | 42, 3407, 126, 218, 91 |
| Twitter observation horizons | 10, 20, 30 minutes |
| Weibo/PHEME observation horizons | 1, 2, 3 hops |
| Model selection | Validation partition only |

Splits are cascade-level. Test labels are report-only. Preprocessing, text
embeddings, and normalization must match the registered input manifests.

## Environment

Use Python 3.10 or newer, with mutually compatible PyTorch and PyTorch Geometric
installations for the intended CPU/CUDA runtime. The implementation also uses
NumPy, pandas, scikit-learn, PyYAML, tqdm, Transformers, sentence-transformers,
and optional Weights & Biases logging.

```bash
python -m pip install numpy pandas scikit-learn pyyaml tqdm transformers sentence-transformers wandb pytest
```

Install PyTorch/PyG for the selected runtime before running model commands.
This repository does not provide a fully pinned cross-platform environment lock.

## Entry Points

```bash
python -m Libs.data.processor --help
python -m Libs.exp.run --help
python Libs/scripts/prepare_canonical_inputs.py --help
python Libs/scripts/validate_canonical_results.py --help
python -m pytest tests -q
```

Training specifications are in `Libs/configs/`; implementations and baselines
are in `Libs/models/`. Provision each dataset and text encoder through its
authorized source, then set paths in the corresponding YAML configuration.

## Evidence and Reproduction

The result chain is source manifest, processed manifest, split manifest,
resolved configuration, code revision, checkpoint, per-example predictions,
metrics, and aggregate tables. Missing or inconsistent links invalidate the
affected result. Legacy CSVs and smoke runs are not canonical performance evidence.

Raw cascades, encoder weights, checkpoints, and generated results are provisioned
separately from this source snapshot. Record observation horizons in their native
units and retain per-seed outputs for all comparisons.

## Review and Licensing

Author names and identifying download links are omitted from this README.
The hosting account and existing history require separate anonymity review.
No project-wide license file is declared here; dataset and dependency terms
continue to apply.
