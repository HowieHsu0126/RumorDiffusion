import argparse
import collections
import copy
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from Libs.data.dataset import GraphDataset
from Libs.exp.trainer import (CGVAEConfig, CGVAETrainer, LinkPredConfig,
                              LinkPredTrainer, Trainer, TrainerConfig,
                              unpack_checkpoint_payload)
from Libs.models.baselines.APPNPNet import APPNPNet
from Libs.models.baselines.BiGCN import BiGCN
from Libs.models.baselines.CTDM_syn import CTDMSyn
from Libs.models.baselines.DGAFake import DGAFakeConfig, DGAFakeModel
from Libs.models.baselines.GAT import GAT
from Libs.models.baselines.GATv2 import GATv2
from Libs.models.baselines.GCN import GCN
from Libs.models.baselines.GCN2Net import GCN2Net
from Libs.models.baselines.GraphSAGE import GraphSAGE
from Libs.models.baselines.ResGCN import ResGCN
from Libs.models.baselines.RootPooling import RootPooling
from Libs.models.baselines.RumorDiffusion import RumorDiffusion
from Libs.models.baselines.TemporalGCN import TemporalGCN
from Libs.models.baselines.TextOnly import TextOnlyClassifier
from Libs.models.baselines.TGAT import TGAT
from Libs.models.baselines.TGN import TGN
from Libs.models.modules.rumor_diffusion import _extract_source_ids_from_batch
from Libs.utils.metrics import compute_accuracy, compute_ece, compute_macro_f1, metrics_from_logits
from Libs.utils.seed import set_global_seed

# 屏蔽警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_yaml_like(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_text(text: str) -> str:
    """Returns the SHA256 digest for a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Dict[str, Any]) -> str:
    """Hashes a JSON-serializable mapping with stable key ordering."""
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def compute_split_signature(dataset_root: str) -> str:
    """Computes a stable signature for the split definition of a processed dataset."""
    splits_path = os.path.join(dataset_root, "splits.json")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"Missing split definition: {splits_path}")
    with open(splits_path, "r", encoding="utf-8") as f:
        return sha256_text(f.read())


def resolve_dataset_label_mapping(dataset_name: str) -> Dict[str, int]:
    """Returns the label mapping configured for a dataset in `dataset.yaml`."""
    dataset_cfg = read_yaml_like(os.path.join("Libs", "configs", "dataset.yaml"))
    label_cfg = dataset_cfg.get("dataset", {}).get("labels", {})
    overrides = label_cfg.get("overrides", {}) if isinstance(label_cfg.get("overrides", {}), dict) else {}
    mapping = overrides.get(dataset_name, {}).get("map", label_cfg.get("map", {}))
    if not isinstance(mapping, dict):
        raise TypeError(f"Invalid label mapping for dataset={dataset_name}")
    return {str(key): int(value) for key, value in mapping.items()}


def build_checkpoint_metadata(
    model_name: str,
    stage: str,
    dataset_name: str,
    dataset_root: str,
    seed: int,
    label_mapping: Dict[str, int],
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds a strict metadata contract for serialized checkpoints."""
    return {
        "format_version": 1,
        "model_name": str(model_name),
        "stage": str(stage),
        "dataset_name": str(dataset_name),
        "seed": int(seed),
        "split_signature": compute_split_signature(dataset_root),
        "label_mapping": {str(key): int(value) for key, value in label_mapping.items()},
        "config_hash": canonical_json_hash(config_payload),
    }


def validate_checkpoint_metadata(
    ckpt_path: str,
    metadata: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    ignored_keys: Optional[Sequence[str]] = None,
) -> None:
    """Validates a loaded checkpoint against the expected dataset/model contract."""
    ignored = {str(key) for key in (ignored_keys or [])}
    missing_keys = [key for key in expected_metadata.keys() if key not in ignored and key not in metadata]
    if missing_keys:
        raise ValueError(
            f"Checkpoint {ckpt_path} is missing metadata keys: {', '.join(sorted(missing_keys))}"
        )
    mismatches: List[str] = []
    for key, expected_value in expected_metadata.items():
        if key in ignored:
            continue
        observed_value = metadata.get(key)
        if observed_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, observed={observed_value!r}"
            )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Checkpoint metadata mismatch for {ckpt_path}: {mismatch_text}")


def load_checkpoint_state(
    ckpt_path: str,
    device: str,
    expected_metadata: Optional[Dict[str, Any]] = None,
    validate_metadata: bool = False,
    allow_legacy: bool = True,
    ignored_metadata_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Loads a checkpoint payload and optionally enforces metadata validation."""
    payload = torch.load(ckpt_path, map_location=device)
    state_dict, metadata_raw = unpack_checkpoint_payload(payload)
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    if validate_metadata:
        if len(metadata) == 0:
            if allow_legacy:
                return state_dict, metadata
            raise ValueError(
                f"Checkpoint {ckpt_path} has no metadata payload; "
                "recreate or migrate the checkpoint before evaluation."
            )
        if expected_metadata is None:
            raise ValueError("Expected checkpoint metadata must be provided when validation is enabled.")
        validate_checkpoint_metadata(
            ckpt_path,
            metadata,
            expected_metadata,
            ignored_keys=ignored_metadata_keys,
        )
    return state_dict, metadata


def resolve_max_nodes_setting(scanned_max_nodes: int, configured_max_nodes: int, context: str) -> int:
    """Resolves `max_nodes` without silent truncation."""
    if int(scanned_max_nodes) <= 0:
        raise ValueError(f"{context}: scanned max_nodes must be positive")
    if int(configured_max_nodes) <= 0:
        return int(scanned_max_nodes)
    if int(scanned_max_nodes) > int(configured_max_nodes):
        raise ValueError(
            f"{context}: scanned max_nodes={int(scanned_max_nodes)} exceeds configured max_nodes={int(configured_max_nodes)}. "
            "Use max_nodes: 0 to resolve from the training split or raise the configured budget."
        )
    return int(configured_max_nodes)


def checkpoint_relevant_rumordiffusion_cfg(rd_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Drops evaluation-only fields before hashing RumorDiffusion checkpoints."""
    filtered = copy.deepcopy(rd_cfg)
    for key in ("checkpoint", "evaluation", "explain"):
        if key in filtered:
            filtered.pop(key)
    text_cfg = filtered.get("text", {})
    if isinstance(text_cfg, dict) and "robust_text" in text_cfg:
        text_cfg = dict(text_cfg)
        text_cfg.pop("robust_text")
        filtered["text"] = text_cfg
    return filtered


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two mappings with override taking precedence.

    - Only merges when both sides are dict; other types are replaced by override.
    - Leaves input objects unmodified by constructing a new dict.
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return dict(base)

    merged: Dict[str, Any] = {}
    # Keys from both dicts
    for k in set(base.keys()) | set(override.keys()):
        if k in override and isinstance(override[k], dict) and isinstance(base.get(k), dict):
            merged[k] = _deep_merge(base[k], override[k])
        elif k in override:
            merged[k] = override[k]
        else:
            merged[k] = base[k]
    return merged


def collect_logits(model: torch.nn.Module, loader: DataLoader, device: str, amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and targets from a loader without altering model state."""
    model.eval()
    logits_list = []
    target_list = []
    use_amp = amp and str(device).startswith("cuda")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(batch)
            else:
                out = model(batch)
            logits_list.append(out.detach().cpu())
            target_list.append(batch.y.view(-1).cpu())
    if len(logits_list) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(target_list, dim=0)


def tune_temperature(logits: torch.Tensor, targets: torch.Tensor, init_temp: float = 1.0, max_iter: int = 100, lr: float = 0.01) -> float:
    """Optimize a single temperature parameter on validation logits."""
    if logits.numel() == 0:
        return float(init_temp)
    device = logits.device
    temp = torch.tensor([init_temp], device=device,
                        requires_grad=True, dtype=logits.dtype)
    optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits / temp.clamp_min(1e-3), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temp.detach().clamp_min(1e-3).item())


def append_results_csv(csv_path: str, headers: list[str], row: list[str]) -> None:
    """Appends one row to a CSV, creating header if missing."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow(row)


def append_rows_csv(csv_path: str, headers: list[str], rows: list[list[str]]) -> None:
    """Appends multiple rows to a CSV, creating header if missing."""
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerows(rows)


def append_dict_rows_csv(csv_path: str, headers: List[str], rows: List[Dict[str, str]]) -> None:
    """Appends multiple dict rows to a CSV, creating header if missing."""
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_headers = reader.fieldnames or []
            if existing_headers == headers:
                with open(csv_path, "a", encoding="utf-8", newline="") as fa:
                    writer = csv.DictWriter(fa, fieldnames=headers, extrasaction="ignore")
                    writer.writerows(rows)
                return
            for row in reader:
                normalized = {key: row.get(key, "") for key in headers}
                existing_rows.append(normalized)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        if existing_rows:
            writer.writerows(existing_rows)
        writer.writerows(rows)


def parse_seeds(arg_seeds: str, exp_cfg: Dict[str, Any]) -> List[int]:
    """Resolves seed list from CLI or exp config."""
    if arg_seeds:
        return [int(s.strip()) for s in arg_seeds.split(",") if s.strip()]
    exp_block = exp_cfg.get("experiment", {})
    if isinstance(exp_block, dict) and "seeds" in exp_block:
        return [int(s) for s in exp_block.get("seeds", [])]
    return [int(exp_block.get("seed", 42)) if isinstance(exp_block, dict) else 42]


def aggregate_numeric(values: List[float]) -> tuple[float, float]:
    """Returns (mean, std) for numeric list; std=0 if len<2."""
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Returns the q-th percentile from a sorted numeric sequence."""
    assert 0.0 <= float(q) <= 1.0
    if len(sorted_values) == 0:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = int(round(float(q) * float(len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def summarize_latency_ms(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Summarizes a latency distribution in milliseconds."""
    if len(latencies_ms) == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(float(v) for v in latencies_ms)
    return {
        "mean_ms": float(sum(ordered) / float(len(ordered))),
        "p95_ms": percentile(ordered, 0.95),
    }


def profile_cuda_boundary(device: str) -> None:
    """Synchronizes CUDA when profiling GPU latency."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_eval_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    amp: bool = False,
    detailed: bool = True,
    class_names: Optional[Dict[int, str]] = None,
    compute_calibration: bool = False,
    ece_bins: int = 15,
    profile: bool = False,
    profile_repeat: int = 1,
    profile_warmup_batches: int = 0,
    profile_max_batches: int = 0,
) -> Dict[str, Any]:
    """Collects logits, targets, source ids, and optional latency profile."""
    model.eval()
    use_amp = bool(amp) and str(device).startswith("cuda")
    logits_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []
    source_ids: List[str] = []
    per_sample_latency_ms: List[float] = []
    total_loss = 0.0
    total = 0
    profiled_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            source_ids.extend(_extract_source_ids_from_batch(batch))
            logits = None
            should_profile = bool(profile) and (
                int(profile_max_batches) <= 0 or profiled_batches < int(profile_max_batches)
            )
            if should_profile and batch_idx < int(profile_warmup_batches):
                should_profile = False
            if should_profile:
                repeat_count = max(1, int(profile_repeat))
                for _ in range(repeat_count):
                    profile_cuda_boundary(device)
                    start = time.perf_counter()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits_profile = model(batch)
                    else:
                        logits_profile = model(batch)
                    profile_cuda_boundary(device)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    batch_size = max(1, int(batch.y.view(-1).size(0)))
                    per_sample_latency_ms.append(elapsed_ms / float(batch_size))
                    logits = logits_profile
                profiled_batches += 1
            else:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(batch)
                else:
                    logits = model(batch)

            assert logits is not None
            loss = criterion(logits, batch.y.view(-1))
            logits_list.append(logits.detach().cpu())
            target_list.append(batch.y.view(-1).cpu())
            bsz = int(batch.y.view(-1).size(0))
            total_loss += float(loss.item()) * float(bsz)
            total += bsz

    if len(logits_list) == 0:
        return {
            "metrics": {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0},
            "logits": torch.empty((0, 0)),
            "targets": torch.empty((0,), dtype=torch.long),
            "source_ids": [],
            "latency_ms": [],
        }

    logits_cat = torch.cat(logits_list, dim=0)
    targets_cat = torch.cat(target_list, dim=0)
    if len(source_ids) != int(targets_cat.size(0)):
        raise ValueError(
            f"Prediction export alignment mismatch: collected {len(source_ids)} source ids "
            f"for {int(targets_cat.size(0))} targets."
        )
    metrics = metrics_from_logits(
        logits_cat,
        targets_cat,
        num_classes=int(logits_cat.size(1)),
        detailed=bool(detailed),
        class_names=class_names,
        compute_calibration=bool(compute_calibration),
        ece_bins=int(ece_bins),
    )
    metrics["loss"] = total_loss / max(1, total)
    return {
        "metrics": metrics,
        "logits": logits_cat,
        "targets": targets_cat,
        "source_ids": source_ids,
        "latency_ms": per_sample_latency_ms,
    }


def export_prediction_rows(
    csv_path: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    tag: str,
    source_ids: Sequence[str],
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Exports per-sample predictions for later significance testing and case analysis."""
    headers = [
        "dataset",
        "model",
        "seed",
        "tag",
        "source_id",
        "target",
        "pred",
        "confidence",
        "probabilities",
    ]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    if len(source_ids) != int(targets.size(0)):
        raise ValueError(
            f"Cannot export predictions: {len(source_ids)} source ids vs {int(targets.size(0))} targets."
        )
    rows: List[Dict[str, str]] = []
    for idx, source_id in enumerate(source_ids):
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": str(seed),
                "tag": tag,
                "source_id": str(source_id),
                "target": str(int(targets[idx].item())),
                "pred": str(int(preds[idx].item())),
                "confidence": f"{float(probs[idx, preds[idx]].item()):.6f}",
                "probabilities": json.dumps([float(v) for v in probs[idx].tolist()]),
            }
        )
    append_dict_rows_csv(csv_path, headers, rows)


def load_prediction_rows(
    csv_path: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    tag: str,
) -> Dict[str, Dict[str, Any]]:
    """Loads exported per-sample predictions keyed by source id."""
    if not os.path.isfile(csv_path):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset", "") != dataset_name:
                continue
            if row.get("model", "") != model_name:
                continue
            if row.get("seed", "") != str(seed):
                continue
            if row.get("tag", "") != tag:
                continue
            source_id = row.get("source_id", "")
            rows[source_id] = {
                "target": int(row["target"]),
                "pred": int(row["pred"]),
                "confidence": float(row["confidence"]),
            }
    return rows


def bootstrap_compare_predictions(
    current_rows: Dict[str, Dict[str, Any]],
    baseline_rows: Dict[str, Dict[str, Any]],
    metric_name: str,
    n_bootstrap: int,
    random_seed: int,
) -> Dict[str, float]:
    """Computes paired bootstrap significance between two aligned prediction sets."""
    shared_ids = sorted(set(current_rows.keys()) & set(baseline_rows.keys()))
    if len(shared_ids) == 0:
        return {"delta": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "count": 0.0}
    targets = torch.tensor([int(current_rows[sid]["target"]) for sid in shared_ids], dtype=torch.long)
    preds_current = torch.tensor([int(current_rows[sid]["pred"]) for sid in shared_ids], dtype=torch.long)
    preds_baseline = torch.tensor([int(baseline_rows[sid]["pred"]) for sid in shared_ids], dtype=torch.long)
    metric_name = str(metric_name).lower()

    def metric_value(preds: torch.Tensor, target_subset: torch.Tensor) -> float:
        if metric_name == "acc":
            return compute_accuracy(preds, target_subset)
        if metric_name == "macro_f1":
            num_classes = int(
                max(
                    int(target_subset.max().item()),
                    int(preds.max().item()),
                ) + 1
            )
            return compute_macro_f1(preds, target_subset, num_classes=num_classes)
        raise ValueError(f"Unsupported significance metric: {metric_name}")

    delta_full = float(metric_value(preds_current, targets) - metric_value(preds_baseline, targets))
    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    sample_count = int(targets.numel())
    deltas: List[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample_idx = torch.randint(low=0, high=sample_count, size=(sample_count,), generator=generator)
        sampled_targets = targets[sample_idx]
        deltas.append(
            float(
                metric_value(preds_current[sample_idx], sampled_targets)
                - metric_value(preds_baseline[sample_idx], sampled_targets)
            )
        )
    ordered = sorted(deltas)
    non_positive = sum(1 for delta in ordered if delta <= 0.0)
    non_negative = sum(1 for delta in ordered if delta >= 0.0)
    tail_count = min(non_positive, non_negative)
    return {
        "delta": delta_full,
        "p_value": float(min(1.0, 2.0 * (tail_count + 1) / float(len(ordered) + 1))),
        "ci_low": percentile(ordered, 0.025),
        "ci_high": percentile(ordered, 0.975),
        "count": float(sample_count),
    }


def build_revision_row(
    timestamp: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    artifact_type: str,
    tag: str,
    values: Dict[str, Any],
) -> Dict[str, str]:
    """Builds one normalized revision-artifact row."""
    row = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "model": model_name,
        "seed": str(seed),
        "artifact_type": artifact_type,
        "tag": tag,
        "obs_minutes": "",
        "samples": "",
        "tau_low": "",
        "tau_high": "",
        "gamma": "",
        "coverage": "",
        "acc_resolved": "",
        "resolved_macro_f1": "",
        "resolved_ece": "",
        "mean_prob": "",
        "mean_var": "",
        "latency_mean_ms": "",
        "latency_p95_ms": "",
        "compare_model": "",
        "metric": "",
        "delta": "",
        "p_value": "",
        "ci_low": "",
        "ci_high": "",
        "notes": "",
    }
    for key, value in values.items():
        if key not in row:
            continue
        if isinstance(value, float):
            row[key] = f"{value:.6f}"
        else:
            row[key] = str(value)
    return row


def run_multi_seed_jobs(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    """Runs one child process per seed to avoid mixed-state multi-seed execution."""
    for seed in seeds:
        cmd = [
            sys.executable,
            "-m",
            "Libs.exp.run",
            "--exp",
            args.exp,
            "--model",
            args.model,
            "--seeds",
            str(int(seed)),
        ]
        if args.eval_only:
            cmd.append("--eval_only")
        if args.ckpt:
            cmd.extend(["--ckpt", args.ckpt])
        if args.dataset_override:
            cmd.extend(["--dataset_override", args.dataset_override])
        if args.source_dataset_override:
            cmd.extend(["--source_dataset_override", args.source_dataset_override])
        if args.lp_edges_policy:
            cmd.extend(["--lp_edges_policy", args.lp_edges_policy])
        subprocess.run(cmd, check=True)


def run_transfer_eval(exp_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Orchestrates transfer evaluation by spawning eval-only runs per pair/model/seed."""
    transfer_cfg = exp_cfg.get("transfer", {})
    if not isinstance(transfer_cfg, dict) or len(transfer_cfg.get("pairs", [])) == 0:
        raise ValueError("Transfer config is missing pairs.")

    outputs_cfg = exp_cfg.get("outputs", {})
    base_ckpt_dir = outputs_cfg.get("ckpt_dir", os.path.join("Output", "ckpts"))
    ckpt_name = str(transfer_cfg.get("ckpt_name", "best.pt"))
    lp_policy = str(transfer_cfg.get("lp_edges_policy", args.lp_edges_policy))
    seeds = parse_seeds(args.seeds, exp_cfg)

    transfer_models = transfer_cfg.get("models")
    if not transfer_models:
        transfer_models = [p.strip() for p in args.model.split(",") if p.strip()]
    model_cfgs = [read_yaml_like(p) for p in transfer_models]

    for pair in transfer_cfg.get("pairs", []):
        source = str(pair["source"])
        target = str(pair["target"])
        for model_path, model_cfg in zip(transfer_models, model_cfgs):
            model_name = str(model_cfg["name"]).lower()
            for seed in seeds:
                ckpt_path = os.path.join(
                    base_ckpt_dir,
                    source,
                    model_name,
                    f"seed{seed}",
                    ckpt_name,
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "Libs.exp.run",
                    "--exp",
                    args.exp,
                    "--model",
                    model_path,
                    "--seeds",
                    str(seed),
                    "--eval_only",
                    "--ckpt",
                    ckpt_path,
                    "--dataset_override",
                    target,
                    "--source_dataset_override",
                    source,
                    "--lp_edges_policy",
                    lp_policy,
                ]
                print(f"[transfer] {source}->{target} model={model_name} seed={seed} ckpt={ckpt_path}")
                subprocess.run(cmd, check=True)


def build_model(name: str, cfg: Dict[str, Any], num_features: int, num_classes: int):
    name = name.lower()
    if name == "textonly":
        m = TextOnlyClassifier(
            in_channels=int(
                cfg.get("text_dim", cfg.get("hidden", num_features))),
            num_classes=num_classes,
            hidden=int(cfg.get("hidden", 256)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "dga_fake":
        cfg_struct = DGAFakeConfig.from_dict(cfg)
        m = DGAFakeModel(cfg_struct, num_classes=num_classes)
        return m
    if name == "sage":
        m = GraphSAGE(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "appnp":
        m = APPNPNet(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
            appnp_alpha=float(cfg.get("alpha", 0.1)),
            appnp_k=int(cfg.get("k", 10)),
        )
        return m
    if name == "gcn2":
        m = GCN2Net(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
            alpha=float(cfg.get("alpha", 0.1)),
            theta=float(cfg.get("theta", 0.5)),
        )
        return m
    if name == "gatv2":
        m = GATv2(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            heads=int(cfg.get("heads", 4)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "temporalgcn":
        m = TemporalGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "root_pool":
        m = RootPooling(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "gcn":
        m = GCN(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
        )
        return m
    if name == "ctdm_syn":
        m = CTDMSyn(
            in_channels=num_features,
            num_classes=num_classes,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            dropout=float(cfg.get("dropout", 0.3)),
            T=int(cfg.get("T", 200)),
            obs_minutes=int(cfg.get("obs_minutes", 30)),
            t_embed_dim=int(cfg.get("t_embed_dim", cfg.get("hidden", 128))),
            film_hidden=int(cfg.get("film_hidden", 128)),
            max_nodes=int(cfg["max_nodes"]),
        )
        # Optional inference stride
        if "t_stride" in cfg:
            setattr(m, "t_stride", int(cfg["t_stride"]))
        if "sampling_eta" in cfg:
            m.sampling_eta = float(cfg["sampling_eta"])
        return m
    if name == "gat":
        m = GAT(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            heads=int(cfg["heads"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
        )
        return m
    if name == "tgat":
        m = TGAT(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            heads=int(cfg["heads"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
        )
        return m
    if name == "bigcn":
        m = BiGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "resgcn":
        m = ResGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "lp_gcn":
        # LP+GCN is just GCN with LP edges enabled
        m = GCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    raise ValueError(f"Unknown model name: {name}")


def setup_logging(log_dir: str, dataset: str, model_name: str, seed: int | None = None) -> str:
    os.makedirs(log_dir, exist_ok=True)
    job_name = os.environ.get("SLURM_JOB_NAME", "").strip()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    log_path = ""
    if job_name and job_id:
        log_path = os.path.join(log_dir, f"{job_name}_{job_id}.log")
    logger = logging.getLogger()
    for h in list(logger.handlers):
        if hasattr(h, "flush"):
            h.flush()
        if hasattr(h, "close"):
            h.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="D2Plus baseline runner")
    parser.add_argument("--exp", type=str, required=True,
                        help="Path to exp.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to baseline model yaml")
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma separated seeds to override config (e.g., 42,43,44)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only run evaluation using provided checkpoint(s)")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint path override for eval_only or transfer runs")
    parser.add_argument("--dataset_override", type=str, default="",
                        help="Override dataset name from exp.yaml for transfer evaluation")
    parser.add_argument("--source_dataset_override", type=str, default="",
                        help="Override dataset.name for transfer source dataset")
    parser.add_argument("--transfer_eval", action="store_true",
                        help="Run transfer evaluation across dataset pairs defined in exp transfer config")
    parser.add_argument("--lp_edges_policy", type=str, default="auto",
                        choices=["auto", "require_existing"],
                        help="LP edges handling: auto-generate when missing, or require existing edges")
    args = parser.parse_args()

    exp_cfg = read_yaml_like(args.exp)
    model_paths: List[str] = [p.strip() for p in args.model.split(",")]
    model_cfgs = [read_yaml_like(p) for p in model_paths]
    seeds = parse_seeds(args.seeds, exp_cfg)
    train_max_samples = int(exp_cfg.get("training", {}).get("max_samples", 0))

    if args.transfer_eval:
        run_transfer_eval(exp_cfg, args)
        return
    if len(seeds) > 1:
        run_multi_seed_jobs(args, seeds)
        return

    dataset_cfg = exp_cfg["dataset"]
    if args.source_dataset_override:
        dataset_cfg = {**dataset_cfg, "name": args.source_dataset_override}
    source_dataset_name = dataset_cfg["name"]
    dataset_name = args.dataset_override if args.dataset_override else source_dataset_name
    dataset_root = os.path.join(dataset_cfg["root"], dataset_name)
    use_lp_edges = bool(dataset_cfg.get("use_lp_edges", False))
    lp_edges_policy = str(args.lp_edges_policy)
    prefix_cfg = dataset_cfg.get("prefix", {})
    observability_cfg = dataset_cfg.get("observability", {})
    observability_sweep_cfg = dataset_cfg.get("observability_sweep", {})
    lp_cfg_dict = exp_cfg.get("link_prediction", {})
    outputs_cfg = exp_cfg.get("outputs", {})
    base_ckpt_dir = outputs_cfg.get(
        "ckpt_dir", os.path.join("Output", "ckpts"))
    base_log_dir = outputs_cfg.get("log_dir", os.path.join("Output", "logs"))
    base_results_dir = outputs_cfg.get(
        "results_dir", os.path.join("Output", "results"))
    sweeps_csv = os.path.join(base_results_dir, "rd_sweeps.csv")
    transfer_csv_default = os.path.join(base_results_dir, "transfer.csv")
    transfer_csv = exp_cfg.get("transfer", {}).get("results_path", transfer_csv_default)
    prediction_csv = outputs_cfg.get(
        "prediction_csv", os.path.join(base_results_dir, "predictions.csv"))
    revision_artifacts_csv = outputs_cfg.get(
        "revision_artifacts_csv", os.path.join(base_results_dir, "revision_artifacts.csv"))
    revision_headers = [
        "timestamp",
        "dataset",
        "model",
        "seed",
        "artifact_type",
        "tag",
        "obs_minutes",
        "samples",
        "tau_low",
        "tau_high",
        "gamma",
        "coverage",
        "acc_resolved",
        "resolved_macro_f1",
        "resolved_ece",
        "mean_prob",
        "mean_var",
        "latency_mean_ms",
        "latency_p95_ms",
        "compare_model",
        "metric",
        "delta",
        "p_value",
        "ci_low",
        "ci_high",
        "notes",
    ]

    # Device configuration
    device_cfg = exp_cfg.get("device", {})
    force_cpu = device_cfg.get("force_cpu", False)
    if force_cpu:
        device = "cpu"
    else:
        requested_device = str(
            device_cfg.get("type", "cuda" if torch.cuda.is_available() else "cpu")
        ).lower()
        device = "cpu" if requested_device == "cuda" and not torch.cuda.is_available() else requested_device

    # Set CUDA device if specified
    if device == "cuda" and torch.cuda.is_available():
        gpu_id = device_cfg.get("gpu_id", 0)
        torch.cuda.set_device(gpu_id)
        logging.info(f"Using CUDA device {gpu_id}")
    else:
        logging.info(f"Using device: {device}")

    ds_train = GraphDataset(
        root=dataset_root,
        split="train",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
    )
    ds_val = GraphDataset(
        root=dataset_root,
        split="val",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
    )
    ds_test = GraphDataset(
        root=dataset_root,
        split="test",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
    )

    num_features = ds_train.num_features
    num_classes = ds_train.num_classes
    dataset_label_mapping = resolve_dataset_label_mapping(source_dataset_name)

    os.makedirs(base_results_dir, exist_ok=True)
    csv_path = os.path.join(base_results_dir, f"{dataset_name}.csv")
    sweeps_headers = [
        "timestamp",
        "dataset",
        "model",
        "seed",
        "tag",
        "obs_minutes",
        "drop_node",
        "drop_edge",
        "edge_drop_rate",
        "text_noise_std",
        "text_dropout",
        "train_seconds",
        "eval_seconds",
        "loss",
        "nll",
        "acc",
        "macro_f1",
        "ece",
        "calibration_temp",
    ]
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt_filename = str(exp_cfg.get("transfer", {}).get("ckpt_name", "best.pt"))

    for model_cfg, model_path in zip(model_cfgs, model_paths):
        name = model_cfg["name"].lower()
        lp_forced = name == "lp_gcn"
        use_lp = True if lp_forced else use_lp_edges
        robust_text_cfg: Dict[str, Any] = {}  # default to avoid undefined access
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            set_global_seed(int(seed))

        ds_train = GraphDataset(
            root=dataset_root,
            split="train",
            use_lp_edges=use_lp,
            prefix_cfg=prefix_cfg,
            observability_cfg=observability_cfg,
        )
        ds_val = GraphDataset(
            root=dataset_root,
            split="val",
            use_lp_edges=use_lp,
            prefix_cfg=prefix_cfg,
            observability_cfg=observability_cfg,
        )
        ds_test = GraphDataset(
            root=dataset_root,
            split="test",
            use_lp_edges=use_lp,
            prefix_cfg=prefix_cfg,
            observability_cfg=observability_cfg,
        )

        num_features = ds_train.num_features
        num_classes = ds_train.num_classes

        log_path = setup_logging(
            log_dir=base_log_dir, dataset=dataset_name, model_name=name, seed=int(seed))
        logging.info(
            f"Dataset: {dataset_name}; Model: {name}; LP: {use_lp}; Seed: {seed}")

        lp_require_existing = (lp_edges_policy == "require_existing" and name == "lp_gcn")
        # Optional pre-step: if LP is required but lp_edges are missing, train LP and generate edges
        if use_lp:
            lp_dir = os.path.join(dataset_root, "lp_edges")
            need_generate = not os.path.isdir(lp_dir) or len(
                [p for p in os.listdir(lp_dir) if p.endswith('.pt')]) == 0
            if lp_require_existing and need_generate:
                raise ValueError(f"LP edges required for LP-GCN transfer, but missing in {lp_dir}")
            if not lp_require_existing and need_generate:
                logging.info(
                    "LP edges missing; training link predictor and generating edges...")
                # Build TGN from model_cfg if provided; else from lp_cfg_dict
                tgn_hidden = int(lp_cfg_dict.get(
                    "hidden", model_cfg["model"].get("hidden", 128)))
                tgn_layers = int(lp_cfg_dict.get(
                    "num_layers", model_cfg["model"].get("num_layers", 3)))
                tgn_heads = int(lp_cfg_dict.get(
                    "heads", model_cfg["model"].get("heads", 2)))
                tgn_dropout = float(lp_cfg_dict.get(
                    "dropout", model_cfg["model"].get("dropout", 0.3)))

                tgn = TGN(
                    in_channels=num_features,
                    hidden_channels=tgn_hidden,
                    num_layers=tgn_layers,
                    heads=tgn_heads,
                    dropout=tgn_dropout,
                )

                lp_cfg = LinkPredConfig(
                    epochs=int(lp_cfg_dict.get(
                        "epochs", exp_cfg["training"].get("epochs", 20))),
                    learning_rate=float(lp_cfg_dict.get(
                        "lr", exp_cfg["training"].get("lr", 3e-4))),
                    weight_decay=float(lp_cfg_dict.get(
                        "weight_decay", exp_cfg["training"].get("weight_decay", 0.0))),
                    batch_size=int(lp_cfg_dict.get(
                        "batch_size", exp_cfg["training"].get("batch_size", 1))),
                    grad_clip=float(lp_cfg_dict.get(
                        "grad_clip", exp_cfg["training"].get("grad_clip", 1.0))),
                    amp=bool(lp_cfg_dict.get(
                        "amp", exp_cfg["training"].get("amp", True))),
                    early_stopping_patience=int(lp_cfg_dict.get(
                        "early_stopping_patience", exp_cfg["training"].get("early_stopping_patience", 15))),
                    num_workers=int(lp_cfg_dict.get(
                        "num_workers", exp_cfg["training"].get("num_workers", 0))),
                    device=device,
                    heads=tgn_heads,
                    hidden=tgn_hidden,
                    num_layers=tgn_layers,
                    dropout=tgn_dropout,
                    topk_per_node=int(lp_cfg_dict.get("topk_per_node", 2)),
                )

                def make_loader(ds: GraphDataset, shuffle: bool) -> DataLoader:
                    return DataLoader(
                        ds,
                        batch_size=lp_cfg.batch_size,
                        shuffle=shuffle,
                        num_workers=lp_cfg.num_workers,
                        pin_memory=(device == "cuda"),
                        persistent_workers=lp_cfg.num_workers > 0,
                    )

                lp_trainer = LinkPredTrainer(model=tgn, cfg=lp_cfg)
                lp_ckpt_dir = os.path.join(
                    "Output", "ckpts", dataset_name, "tgn", f"seed{seed}")
                lp_ckpt_path = os.path.join(lp_ckpt_dir, "best.pt")
                lp_trainer.fit(make_loader(ds_train, shuffle=True), save_path=lp_ckpt_path,
                               epochs=lp_cfg.epochs, early_stopping_patience=lp_cfg.early_stopping_patience)

                for split_name, ds_split in ("train", ds_train), ("val", ds_val), ("test", ds_test):
                    LinkPredTrainer.generate_edges_for_split(
                        lp_trainer.model, ds_split, topk_per_node=lp_cfg.topk_per_node, out_dir=lp_dir, device=torch.device(device), num_workers=lp_cfg.num_workers)

                # Reload datasets with new LP edges
                ds_train = GraphDataset(
                    root=dataset_root,
                    split="train",
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                )
                ds_val = GraphDataset(
                    root=dataset_root,
                    split="val",
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                )
                ds_test = GraphDataset(
                    root=dataset_root,
                    split="test",
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                )
                num_features = ds_train.num_features
                num_classes = ds_train.num_classes

        # If CTDM-syn, scan max nodes from train split and cap at configured max_nodes, then inject into model config
        if name == "ctdm_syn":
            max_nodes_scan = 0
            for raw_idx in ds_train.indices():
                sample = ds_train.get(raw_idx)
                max_nodes_scan = max(max_nodes_scan, int(sample.num_nodes))
            max_nodes_config = int(model_cfg["model"].get("max_nodes", 0) or 0)
            max_nodes_final = resolve_max_nodes_setting(
                scanned_max_nodes=max_nodes_scan,
                configured_max_nodes=max_nodes_config,
                context="CTDM-syn",
            )
            model_cfg["model"]["max_nodes"] = int(max_nodes_final)
            logging.info(
                f"CTDM-syn: scanned max_nodes={max_nodes_scan}, resolved to {max_nodes_final} "
                f"(config value: {max_nodes_config})")

        # Build model for classification or generative baseline
        m = None
        class_weights = None
        if name == "rumordiffusion":
            # Stage 1: train CTDM if needed; then generate potentials
            logging.info("Running RumorDiffusion two-stage training...")

            # 1) Scan max_nodes from train split
            rd_cfg_exp = exp_cfg.get("rumordiffusion", {})
            rd_cfg_model = model_cfg.get("model", {})
            rd_cfg = _deep_merge(rd_cfg_model, rd_cfg_exp)
            ctdm_cfg = rd_cfg.get("ctdm", {})
            checkpoint_cfg = rd_cfg.get("checkpoint", {}) if isinstance(rd_cfg.get("checkpoint", {}), dict) else {}
            validate_ckpt_metadata = bool(checkpoint_cfg.get("validate_metadata", False))
            allow_legacy_ckpt = bool(checkpoint_cfg.get("allow_legacy", True))
            ctdm_ckpt_root = str(checkpoint_cfg.get("ctdm_dir", base_ckpt_dir))
            max_nodes_scan = 0
            for split_idx in range(len(ds_train)):
                sample = ds_train.get(split_idx)
                max_nodes_scan = max(max_nodes_scan, int(sample.num_nodes))
            max_nodes_config = int(ctdm_cfg.get("max_nodes", 0) or 0)
            max_nodes_final = resolve_max_nodes_setting(
                scanned_max_nodes=max_nodes_scan,
                configured_max_nodes=max_nodes_config,
                context="RumorDiffusion CTDM",
            )
            logging.info(
                f"RumorDiffusion CTDM: scanned max_nodes={max_nodes_scan}, resolved to {max_nodes_final} "
                f"(config value: {max_nodes_config})")

            # 2) Build CTDM
            bb_cfg = ctdm_cfg.get("backbone", {}) if isinstance(
                ctdm_cfg.get("backbone", {}), dict) else {}
            # Optionally compute degree histogram for PNA
            deg_histogram = None
            if str(bb_cfg.get("name", "gcn")).lower() == "pna":
                logging.info("Computing degree histogram for PNA...")
                from torch_geometric.utils import degree as pyg_degree
                deg_counter = collections.Counter()
                for raw_idx in range(len(ds_train)):
                    data = ds_train.get(raw_idx)
                    if hasattr(data, "edge_index") and data.edge_index is not None and int(data.edge_index.numel()) > 0:
                        deg = pyg_degree(data.edge_index[1].to(
                            torch.long), num_nodes=int(data.num_nodes), dtype=torch.long)
                        deg_counter.update([int(v.item()) for v in deg])
                max_degree = max(deg_counter.keys()) if len(
                    deg_counter) > 0 else 0
                deg_hist = torch.zeros(max_degree + 1, dtype=torch.long)
                for d, c in deg_counter.items():
                    if d <= max_degree:
                        deg_hist[d] = int(c)
                deg_histogram = deg_hist
                logging.info(
                    f"Degree histogram computed: max_deg={int(deg_histogram.argmax().item())}")

            # Optional text fusion from exp config
            text_cfg = rd_cfg.get("text", {})
            text_enabled = bool(text_cfg.get("enabled", False)) if isinstance(
                text_cfg, dict) else False
            text_dim = int(text_cfg.get("text_dim", 0)) if text_enabled else 0
            text_fuse = str(text_cfg.get("fuse_mode", "add")
                            ) if text_enabled else "add"
            robust_text_cfg = text_cfg.get(
                "robust_text", {}) if isinstance(text_cfg, dict) else {}

            ctdm = CTDMSyn(
                in_channels=num_features,
                num_classes=num_classes,
                hidden_channels=int(ctdm_cfg.get("hidden", 128)),
                num_layers=int(ctdm_cfg.get("num_layers", 3)),
                dropout=float(ctdm_cfg.get("dropout", 0.3)),
                T=int(ctdm_cfg.get("T", 200)),
                obs_minutes=int(ctdm_cfg.get("obs_minutes", 30)),
                t_embed_dim=int(ctdm_cfg.get(
                    "t_embed_dim", ctdm_cfg.get("hidden", 128))),
                film_hidden=int(ctdm_cfg.get("film_hidden", 128)),
                max_nodes=int(max_nodes_final),
                backbone_name=str(bb_cfg.get("name", "gcn")).lower(),
                backbone_heads=int(bb_cfg.get("heads", 1)),
                backbone_dropout=float(bb_cfg.get(
                    "dropout", ctdm_cfg.get("dropout", 0.3))),
                deg_histogram=deg_histogram,
                pna_aggregators=bb_cfg.get("aggregators", None),
                pna_scalers=bb_cfg.get("scalers", None),
                text_dim=(text_dim if text_dim > 0 else None),
                text_fuse=text_fuse,
            )
            if isinstance(robust_text_cfg, dict):
                noise_list = robust_text_cfg.get("noise_std", [])
                drop_list = robust_text_cfg.get("dropout_p", [])
                if len(noise_list) > 0:
                    ctdm.text_noise_std = float(noise_list[0])
                if len(drop_list) > 0:
                    ctdm.text_dropout_p = float(drop_list[0])
            if "t_stride" in ctdm_cfg:
                setattr(ctdm, "t_stride", int(ctdm_cfg["t_stride"]))
            # Inject CTDM auxiliary configs
            if "parameterization" in ctdm_cfg:
                ctdm.parameterization = str(
                    ctdm_cfg["parameterization"]).lower()
            # Defaults per plan: guidance_p=0.2, guidance_scale=2.0
            ctdm.guidance_p = float(ctdm_cfg.get("guidance_p", 0.2))
            ctdm.guidance_scale = float(ctdm_cfg.get("guidance_scale", 2.0))
            ctdm.sampling_eta = float(ctdm_cfg.get("sampling_eta", 0.0))
            logging.info(
                f"RD-CTDM cfg -> T={ctdm.T}, stride={getattr(ctdm, 't_stride', 1)}, "
                f"guidance_scale={ctdm.guidance_scale}, guidance_p={ctdm.guidance_p}, "
                f"sampling_eta={ctdm.sampling_eta}"
            )
            if "use_attn_readout" in ctdm_cfg:
                ctdm.use_attn_readout = bool(ctdm_cfg["use_attn_readout"])
            # Multi-window training block
            mw_cfg = ctdm_cfg.get("multiwindow", {})
            if isinstance(mw_cfg, dict):
                # Defaults per plan: enabled=True, min=10, max=30
                ctdm.multiwindow_enabled = bool(mw_cfg.get("enabled", True))
                ctdm.cutoff_min = int(mw_cfg.get("min", 10))
                ctdm.cutoff_max = int(mw_cfg.get("max", 30))
            # Sparse candidate pairs block
            sp_cfg = ctdm_cfg.get("sparse_pairs", {})
            if isinstance(sp_cfg, dict):
                # Defaults per plan: enabled=True, strategy=b, topk=8
                ctdm.sparse_pairs_enabled = bool(sp_cfg.get("enabled", True))
                ctdm.sparse_pairs_strategy = str(
                    sp_cfg.get("strategy", "b")).lower()
                ctdm.sparse_pairs_topk = int(sp_cfg.get("topk", 8))
                if "lambda_time" in sp_cfg:
                    ctdm.sparse_pairs_lambda_time = float(
                        sp_cfg["lambda_time"])
                if "lambda_degree" in sp_cfg:
                    ctdm.sparse_pairs_lambda_degree = float(
                        sp_cfg["lambda_degree"])

            # CTDM regularizer weights and prototype settings (optional overrides)
            reg_cfg = ctdm_cfg.get("regularizers", {})
            if isinstance(reg_cfg, dict):
                if "lambda_proto" in reg_cfg:
                    ctdm.lambda_proto = float(reg_cfg["lambda_proto"])
                if "temperature" in reg_cfg:
                    ctdm.prototype_temperature = float(reg_cfg["temperature"])
                if "momentum" in reg_cfg:
                    ctdm.prototype_momentum = float(reg_cfg["momentum"])
                if "lambda_topo_bce" in reg_cfg:
                    ctdm.lambda_topo_bce = float(reg_cfg["lambda_topo_bce"])
                if "lambda_deg_kl" in reg_cfg:
                    ctdm.lambda_deg_kl = float(reg_cfg["lambda_deg_kl"])
                # New regularizer parameters for text semantic integration
                if "lambda_proto_orth" in reg_cfg:
                    ctdm.lambda_proto_orth = float(
                        reg_cfg["lambda_proto_orth"])
                if "lambda_supcon_pair" in reg_cfg:
                    ctdm.lambda_supcon_pair = float(
                        reg_cfg["lambda_supcon_pair"])
                if "pair_ids" in reg_cfg:
                    ctdm.pair_ids = tuple(reg_cfg["pair_ids"])
            # Optional forecast head for future statistics
            forecast_cfg = ctdm_cfg.get("forecast", {})
            if isinstance(forecast_cfg, dict) and forecast_cfg.get("enabled", False):
                forecast_dim = int(forecast_cfg.get("dim", 2))
                ctdm.forecast_enabled = True
                ctdm.lambda_forecast = float(forecast_cfg.get("weight", 0.0))
                ctdm.forecast_head = nn.Linear(
                    ctdm.hidden_channels, forecast_dim)
            struct_cfg = ctdm_cfg.get("structure_dropout", {})
            if isinstance(struct_cfg, dict):
                ctdm.edge_drop_rate = float(struct_cfg.get("edge", 0.0))
            struct_cfg = ctdm_cfg.get("structure_dropout", {})
            if isinstance(struct_cfg, dict):
                ctdm.edge_drop_rate = float(struct_cfg.get("edge", 0.0))

            ctdm_train_cfg = TrainerConfig(
                epochs=int(exp_cfg["training"].get("epochs_ctdm", exp_cfg["training"]["epochs"])),
                learning_rate=float(exp_cfg["training"].get("lr_ctdm", exp_cfg["training"].get("lr", 3e-4))),
                weight_decay=float(exp_cfg["training"].get("weight_decay_ctdm", exp_cfg["training"].get("weight_decay", 0.0))),
                batch_size=int(exp_cfg["training"].get("batch_size_ctdm", 1)),
                grad_clip=float(exp_cfg["training"].get("grad_clip", 0.0)),
                amp=bool(exp_cfg["training"].get("amp", True)),
                early_stopping_patience=int(
                    exp_cfg["training"].get(
                        "early_stopping_patience_ctdm",
                        exp_cfg["training"].get("early_stopping_patience", 20),
                    )
                ),
                num_workers=int(exp_cfg["training"].get("num_workers", 4)),
                device=device,
                use_tqdm=bool(exp_cfg["training"].get("use_tqdm", True)),
                class_weights=None,
                max_batches_per_epoch=exp_cfg["training"].get(
                    "max_batches_per_epoch_ctdm",
                    exp_cfg["training"].get("max_batches_per_epoch"),
                ),
                min_epochs=int(
                    exp_cfg["training"].get(
                        "min_epochs_ctdm",
                        exp_cfg["training"].get("min_epochs", 30),
                    )
                ),
            )
            ctdm_eval_cfg = exp_cfg.get("evaluation", {})
            ctdm_max_eval_samples = int(
                ctdm_eval_cfg.get("max_samples", 0)
            ) if isinstance(ctdm_eval_cfg, dict) else 0

            def _maybe_subset_ctdm(ds):
                if ctdm_max_eval_samples > 0:
                    return Subset(ds, list(range(min(len(ds), ctdm_max_eval_samples))))
                return ds

            def _make_ctdm_loader(ds, shuffle: bool) -> DataLoader:
                return DataLoader(
                    ds,
                    batch_size=ctdm_train_cfg.batch_size,
                    shuffle=shuffle,
                    num_workers=ctdm_train_cfg.num_workers,
                    pin_memory=(device == "cuda"),
                    persistent_workers=ctdm_train_cfg.num_workers > 0,
                )

            ctdm_train_ds = ds_train if train_max_samples <= 0 else Subset(
                ds_train, list(range(min(len(ds_train), train_max_samples)))
            )
            ctdm_val_ds = _maybe_subset_ctdm(ds_val)
            ctdm_pretrain_dir = os.path.join(
                ctdm_ckpt_root, source_dataset_name, "rumordiffusion_ctdm", f"seed{seed}"
            )
            os.makedirs(ctdm_pretrain_dir, exist_ok=True)
            ctdm_pretrain_path = os.path.join(ctdm_pretrain_dir, ckpt_filename)
            reuse_ctdm_pretrain = bool(exp_cfg["training"].get("reuse_ctdm_pretrain", False))
            ctdm_checkpoint_metadata = build_checkpoint_metadata(
                model_name=name,
                stage="ctdm",
                dataset_name=source_dataset_name,
                dataset_root=dataset_root,
                seed=int(seed),
                label_mapping=dataset_label_mapping,
                config_payload={
                    "ctdm": _deep_merge(ctdm_cfg, {"max_nodes": int(max_nodes_final)}),
                    "training": {
                        "epochs_ctdm": int(exp_cfg["training"].get("epochs_ctdm", exp_cfg["training"]["epochs"])),
                        "lr_ctdm": float(exp_cfg["training"].get("lr_ctdm", exp_cfg["training"].get("lr", 3e-4))),
                        "weight_decay_ctdm": float(exp_cfg["training"].get("weight_decay_ctdm", exp_cfg["training"].get("weight_decay", 0.0))),
                        "batch_size_ctdm": int(exp_cfg["training"].get("batch_size_ctdm", 1)),
                        "early_stopping_patience_ctdm": int(
                            exp_cfg["training"].get(
                                "early_stopping_patience_ctdm",
                                exp_cfg["training"].get("early_stopping_patience", 20),
                            )
                        ),
                        "min_epochs_ctdm": int(
                            exp_cfg["training"].get(
                                "min_epochs_ctdm",
                                exp_cfg["training"].get("min_epochs", 30),
                            )
                        ),
                    },
                    "prefix": prefix_cfg,
                    "observability": observability_cfg,
                },
            )
            if (not args.eval_only) and not (reuse_ctdm_pretrain and os.path.isfile(ctdm_pretrain_path)):
                logging.info("CTDM pretraining started")
                ctdm_trainer = Trainer(model=ctdm, cfg=ctdm_train_cfg)
                ctdm_fit_out = ctdm_trainer.fit(
                    train_loader=_make_ctdm_loader(ctdm_train_ds, shuffle=True),
                    val_loader=_make_ctdm_loader(ctdm_val_ds, shuffle=False),
                    save_path=ctdm_pretrain_path,
                    checkpoint_metadata=ctdm_checkpoint_metadata,
                )
                logging.info(
                    "CTDM pretraining best val macro F1: "
                    f"{ctdm_fit_out.get('best_val_macro_f1', ctdm_fit_out.get('best_val_metric', 0.0)):.4f}"
                )
            elif (not args.eval_only) and reuse_ctdm_pretrain and os.path.isfile(ctdm_pretrain_path):
                logging.info(f"Reusing existing CTDM pretraining checkpoint: {ctdm_pretrain_path}")
            if not args.eval_only and os.path.isfile(ctdm_pretrain_path):
                ctdm_state, _ = load_checkpoint_state(
                    ckpt_path=ctdm_pretrain_path,
                    device=device,
                    expected_metadata=ctdm_checkpoint_metadata,
                    validate_metadata=validate_ckpt_metadata,
                    allow_legacy=allow_legacy_ckpt,
                    ignored_metadata_keys=["config_hash"],
                )
                ctdm.load_state_dict(ctdm_state)
            elif not args.eval_only:
                raise FileNotFoundError(
                    f"Missing CTDM pretraining checkpoint at {ctdm_pretrain_path}"
                )

            # RumorDiffusion arbiter and loss config overrides
            rd_loss_cfg = rd_cfg.get("loss", {})
            # Build RumorDiffusion arbiter
            arb_cfg = rd_cfg.get("arbiter", {})
            # Get actual potential vector dimension from loaded data
            sample_data = ds_train.get(0)
            if hasattr(sample_data, 'h0') and hasattr(sample_data, 'h1'):
                actual_hidden_dim = int(sample_data.h0.size(-1))
                concat_dim = int(
                    torch.cat([sample_data.h0, sample_data.h1], dim=-1).size(-1))
            else:
                # Fallback to CTDM inference if potentials not loaded
                sample_batch = next(
                    iter(DataLoader(ds_train, batch_size=1, shuffle=False)))
                sample_batch = sample_batch.to(device)
                with torch.no_grad():
                    sample_potentials = ctdm.infer_potentials_all(sample_batch)
                    actual_hidden_dim = int(sample_potentials.size(-1))
                    h0 = sample_potentials[:, 0, :]
                    h1 = sample_potentials[:, 1, :]
                    concat_dim = int(torch.cat([h0, h1], dim=-1).size(-1))
            logging.info(
                f"Single potential vector dimension: {actual_hidden_dim}")
            logging.info(f"Concatenated input dimension: {concat_dim}")

            # Build RumorDiffusion with Transformer arbiter; pass dd_gan config if provided
            dd_cfg = rd_cfg.get("dd_gan", {})
            m = RumorDiffusion(
                hidden_channels=actual_hidden_dim,
                num_classes=num_classes,
                arbiter_hidden=int(arb_cfg.get("hidden", 256)),
                dropout=float(arb_cfg.get("dropout", 0.3)),
                ctdm=ctdm,
                freeze_ctdm=bool(arb_cfg.get("freeze_ctdm", True)),
                arbiter_type=str(arb_cfg.get("type", "transformer")).lower(),
                arbiter_readout=str(arb_cfg.get("readout", "cls")).lower(),
                token_norm=str(arb_cfg.get("token_norm", "layernorm")).lower(),
                token_dropout=float(arb_cfg.get(
                    "token_dropout", arb_cfg.get("dropout", 0.3))),
                transformer_nhead=int(arb_cfg.get(
                    "transformer", {}).get("nhead", 4)),
                transformer_nlayers=int(arb_cfg.get(
                    "transformer", {}).get("nlayers", 2)),
                transformer_dim_feedforward=int(arb_cfg.get(
                    "transformer", {}).get("dim_feedforward", 512)),
                transformer_dropout=float(arb_cfg.get("transformer", {}).get(
                    "dropout", arb_cfg.get("dropout", 0.3))),
                dd_gan_enabled=bool(dd_cfg.get("enabled", False)),
                enable_dd_gan=bool(dd_cfg.get(
                    "enable", dd_cfg.get("enabled", False))),
                lambda_gan=float(dd_cfg.get("lambda_gan", 0.0)),
                lr_d=float(dd_cfg.get("lr_d", 1.0e-4)),
                lambda_r1=float(dd_cfg.get("lambda_r1", 10.0)),
                # New parameters for text semantic integration and refined loss
                use_pair_token=bool(arb_cfg.get("use_pair_token", False)),
                pair_token_mode=str(arb_cfg.get("pair_token_mode", "diff")),
                pair_ids=tuple(arb_cfg.get("pair_ids", [1, 2])),
                tf_head_enabled=bool(arb_cfg.get("tf_head_enabled", False)),
                tf_infer_adjust_alpha=float(
                    arb_cfg.get("tf_infer_adjust_alpha", 0.0)),
                pair_margin_enabled=bool(
                    rd_loss_cfg.get("pair_margin_enabled", False)),
                pair_margin_m=float(rd_loss_cfg.get("pair_margin_m", 0.5)),
                lambda_pair_margin=float(
                    rd_loss_cfg.get("lambda_pair_margin", 0.0)),
                enable_pair_margin_loss=bool(
                    rd_loss_cfg.get("enable_pair_margin_loss", True)),
                lambda_tf_bce=float(rd_loss_cfg.get("lambda_tf_bce", 0.0)),
                use_class_embedding=bool(
                    arb_cfg.get("use_class_embedding", False)),
                class_embed_mode=str(arb_cfg.get(
                    "class_embed_mode", "concat_proj")),
                cache_potentials=bool(arb_cfg.get("cache_potentials", False)),
                infer_amp=bool(arb_cfg.get("infer_amp", False)),
                uncertainty_force_ctdm=bool(arb_cfg.get("uncertainty_force_ctdm", False)),
                loss_logging=bool(rd_loss_cfg.get("loss_logging", False)),
            )
            # Inject loss hyperparameters and class priors
            if "class_priors" in rd_loss_cfg:
                pri = torch.tensor(
                    rd_loss_cfg["class_priors"], dtype=torch.float32)
                pri = pri / pri.sum()
                m.class_priors = pri.to(device)
            # Class priors from dataset if not provided
            if m.class_priors.sum().abs().item() <= 0.0 or torch.isnan(m.class_priors.sum()).item():
                counts = [0 for _ in range(int(num_classes))]
                for split_idx in range(len(ds_train)):
                    yv = int(ds_train.get(split_idx).y.view(-1)[0].item())
                    counts[yv] += 1
                total = sum(counts)
                class_weights = [
                    float(total) / (int(num_classes) * float(c)) for c in counts]
                pri = torch.tensor([c/float(total)
                                   for c in counts], dtype=torch.float32)
                m.class_priors = pri.to(device)
            # Loss weights
            m.lambda_diffusion = float(
                rd_loss_cfg.get("lambda_diffusion", 0.0))
            m.la_tau = float(rd_loss_cfg.get("la_tau", 1.0))
            m.la_gamma = float(rd_loss_cfg.get("la_gamma", 1.5))
            m.la_smooth = float(rd_loss_cfg.get("la_smooth", 0.05))
            m.lambda_brier = float(rd_loss_cfg.get("w_brier", 0.01))
            m.lambda_consistency = float(rd_loss_cfg.get("w_cons", 0.05))
            logging.info(
                f"RD-Loss cfg -> lambda_diffusion={m.lambda_diffusion}, "
                f"la_tau={m.la_tau}, la_gamma={m.la_gamma}"
            )
            # Multi-pair margin loss configuration for handling multiple confusable class pairs
            mp_enabled = bool(rd_loss_cfg.get("enable_multi_pair_margin", rd_loss_cfg.get(
                "multi_pair_margin_enabled", False)))
            m.multi_pair_margin_list = rd_loss_cfg.get(
                "multi_pair_margin_list", []) if mp_enabled else []
            m.multi_pair_margin_values = rd_loss_cfg.get(
                "multi_pair_margin_values", []) if mp_enabled else []
            m.lambda_multi_pair_margin = float(rd_loss_cfg.get(
                "lambda_multi_pair_margin", 0.0)) if mp_enabled else 0.0
            m.enable_multi_pair_margin = mp_enabled and len(
                m.multi_pair_margin_list) > 0 and float(m.lambda_multi_pair_margin) > 0.0
            if m.enable_multi_pair_margin:
                logging.info(
                    f"Multi-pair margin loss enabled with pairs: {m.multi_pair_margin_list}, margins: {m.multi_pair_margin_values}, lambda: {m.lambda_multi_pair_margin}")
            # Optional toggles for auxiliary losses
            m.enable_pair_margin_loss = bool(rd_loss_cfg.get(
                "enable_pair_margin_loss", rd_loss_cfg.get("pair_margin_enabled", False)))
            if not m.enable_pair_margin_loss:
                m.lambda_pair_margin = 0.0
            m.enable_tf_head_loss = bool(
                rd_loss_cfg.get("enable_tf_head_loss", True))
            if not m.enable_tf_head_loss:
                m.lambda_tf_bce = 0.0
            # Summarize key loss hyperparameters for reproducibility
            logging.info(
                f"Loss cfg -> la_gamma={m.la_gamma}, la_tau={m.la_tau}, "
                f"pair_margin(enabled={m.enable_pair_margin_loss}, lambda={m.lambda_pair_margin}, m={m.pair_margin_m}), "
                f"multi_pair(enabled={m.enable_multi_pair_margin}, lambda={m.lambda_multi_pair_margin}), "
                f"tf_head(enabled={m.enable_tf_head_loss}, lambda={m.lambda_tf_bce}), "
                f"brier={m.lambda_brier}, consistency={m.lambda_consistency}, dd_gan={m.dd_gan_enabled}"
            )
        elif name == "cgvae_gcn":
            # CGVAE-GCN two-stage pipeline
            logging.info("Running CGVAE-GCN two-stage training...")

            # Stage 1: Train CVAE models and generate augmented edges
            logging.info("Stage 1: Training class-conditional VGAE models...")
            cgvae_cfg_dict = model_cfg.get("cvae", {})
            print("Creating CGVAEConfig...")
            print(f"cgvae_cfg_dict: {cgvae_cfg_dict}")
            print(f"model_cfg: {model_cfg}")
            cg_cfg = CGVAEConfig(
                epochs=int(cgvae_cfg_dict.get(
                    "epochs", exp_cfg["training"].get("epochs", 50))),
                learning_rate=float(cgvae_cfg_dict.get(
                    "lr", exp_cfg["training"].get("lr", 1e-3))),
                weight_decay=float(cgvae_cfg_dict.get(
                    "weight_decay", exp_cfg["training"].get("weight_decay", 1e-5))),
                batch_size=int(cgvae_cfg_dict.get(
                    "batch_size", exp_cfg["training"].get("batch_size", 8))),
                grad_clip=float(cgvae_cfg_dict.get(
                    "grad_clip", exp_cfg["training"].get("grad_clip", 1.0))),
                amp=bool(cgvae_cfg_dict.get(
                    "amp", exp_cfg["training"].get("amp", True))),
                num_workers=int(cgvae_cfg_dict.get(
                    "num_workers", exp_cfg["training"].get("num_workers", 0))),
                early_stopping_patience=int(cgvae_cfg_dict.get(
                    "early_stopping_patience", exp_cfg["training"].get("early_stopping_patience", 15))),
                device=device,
                hidden=int(cgvae_cfg_dict.get(
                    "hidden", model_cfg["model"].get("hidden", 256))),
                latent=int(cgvae_cfg_dict.get(
                    "latent", model_cfg["model"].get("latent", 128))),
                num_layers=int(cgvae_cfg_dict.get("num_layers", 3)),
                dropout=float(cgvae_cfg_dict.get("dropout", 0.3)),
                use_bn=bool(cgvae_cfg_dict.get("use_bn", True)),
                kl_anneal=bool(cgvae_cfg_dict.get("kl_anneal", True)),
            )
            print("Creating CGVAETrainer...")
            cg_trainer = CGVAETrainer(cfg=cg_cfg)
            print("CGVAETrainer created successfully")
            cg_ckpt_root = os.path.join(
                "Output", "ckpts", dataset_name, "cgvae_gcn_stage1")
            print(f"Checkpoint root: {cg_ckpt_root}")
            print(
                f"Number of classes: {num_classes}, Number of features: {num_features}")
            print("Starting train_all_classes...")
            cg_trainer.train_all_classes(
                ds_train, cg_ckpt_root, num_classes, num_features)

            # Generate augmented edges for all splits
            logging.info(
                "Generating augmented edges using trained CVAE models...")
            aug_edge_dir = os.path.join(
                "Input", "processed", dataset_name, "cgvae_edges")
            cg_trainer.generate_edges_for_split(
                ds_train, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "train")
            cg_trainer.generate_edges_for_split(
                ds_val, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "val")
            cg_trainer.generate_edges_for_split(
                ds_test, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "test")

            # Stage 2: Train GCN on augmented graphs
            logging.info(
                "Stage 2: Training GCN classifier on augmented graphs...")

            # Reload datasets with augmented edges
            ds_train = GraphDataset(
                root=dataset_root,
                split="train",
                use_lp_edges=True,
                lp_dir=aug_edge_dir,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
            )
            ds_val = GraphDataset(
                root=dataset_root,
                split="val",
                use_lp_edges=True,
                lp_dir=aug_edge_dir,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
            )
            ds_test = GraphDataset(
                root=dataset_root,
                split="test",
                use_lp_edges=True,
                lp_dir=aug_edge_dir,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
            )

            # Build GCN model for stage 2
            from Libs.models.baselines.GCN import GCN
            gcn_hidden = int(model_cfg["model"].get("gcn_hidden", 128))
            gcn_num_layers = int(model_cfg["model"].get("gcn_num_layers", 3))
            gcn_dropout = float(model_cfg["model"].get("gcn_dropout", 0.3))

            m = GCN(
                in_channels=num_features,
                hidden_channels=gcn_hidden,
                num_layers=gcn_num_layers,
                num_classes=num_classes,
                dropout=gcn_dropout
            ).to(device)

            # Continue with standard GCN training flow
        else:
            m = build_model(
                model_cfg["name"], model_cfg["model"], num_features, num_classes)

        trainer_cfg = TrainerConfig(
            epochs=int(exp_cfg["training"]["epochs"]),
            learning_rate=float(exp_cfg["training"].get("lr", 3e-4)),
            weight_decay=float(exp_cfg["training"].get("weight_decay", 0.0)),
            batch_size=int(exp_cfg["training"]["batch_size"]),
            grad_clip=float(exp_cfg["training"].get("grad_clip", 0.0)),
            amp=bool(exp_cfg["training"].get("amp", True)),
            early_stopping_patience=int(
                exp_cfg["training"].get("early_stopping_patience", 20)),
            num_workers=int(exp_cfg["training"].get("num_workers", 4)),
            device=device,
            use_tqdm=bool(exp_cfg["training"].get("use_tqdm", True)),
            class_weights=class_weights,
            max_batches_per_epoch=exp_cfg["training"].get(
                "max_batches_per_epoch"),
            min_epochs=int(exp_cfg["training"].get("min_epochs", 30)),
        )

        eval_cfg = exp_cfg.get("evaluation", {})
        eval_batch_size = int(eval_cfg.get("batch_size", trainer_cfg.batch_size)) if isinstance(eval_cfg, dict) else int(trainer_cfg.batch_size)

        def make_loader(ds: GraphDataset, shuffle: bool, sampler=None, batch_size: Optional[int] = None) -> DataLoader:
            resolved_batch_size = int(batch_size) if batch_size is not None else int(trainer_cfg.batch_size)
            return DataLoader(
                ds,
                batch_size=resolved_batch_size,
                shuffle=(False if sampler is not None else shuffle),
                sampler=sampler,
                num_workers=trainer_cfg.num_workers,
                pin_memory=(device == "cuda"),
                persistent_workers=trainer_cfg.num_workers > 0,
            )

        # Use regular shuffle instead of WeightedRandomSampler for balanced datasets
        # (Data is already balanced, so WeightedRandomSampler may introduce noise)
        max_eval_samples = int(eval_cfg.get(
            "max_samples", 0)) if isinstance(eval_cfg, dict) else 0

        def _maybe_subset(ds):
            if max_eval_samples > 0:
                return Subset(ds, list(range(min(len(ds), max_eval_samples))))
            return ds
        ds_val_eval = _maybe_subset(ds_val)
        ds_test_eval = _maybe_subset(ds_test)

        ds_train_eff = ds_train if train_max_samples <= 0 else Subset(
            ds_train, list(range(min(len(ds_train), train_max_samples))))
        train_loader = make_loader(ds_train_eff, shuffle=True, batch_size=trainer_cfg.batch_size)
        val_loader = make_loader(ds_val_eval, shuffle=False, batch_size=eval_batch_size)
        test_loader = make_loader(ds_test_eval, shuffle=False, batch_size=eval_batch_size)

        ckpt_dir = os.path.join(
            base_ckpt_dir, source_dataset_name, name, f"seed{seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
        actual_ckpt_path = args.ckpt if args.ckpt else ckpt_path

        wb_cfg = exp_cfg.get("training", {}).get("wandb", {})
        wb_enabled = bool(wb_cfg.get("enabled", False))
        if wb_enabled:
            import wandb

            run = wandb.init(
                project=str(wb_cfg.get("project", "D2Plus")),
                name=str(wb_cfg.get(
                    "run_name", f"{dataset_name}-{name}-seed{seed}")),
                config={
                    "dataset": dataset_name,
                    "model": name,
                    "exp": args.exp,
                    "model_cfg": model_path,
                    "seed": seed,
                },
            )
        trainer = Trainer(model=m, cfg=trainer_cfg)
        fit_out = {"best_val_macro_f1": 0.0, "best_val_metric": 0.0}
        train_seconds = 0.0
        checkpoint_metadata: Optional[Dict[str, Any]] = None
        if name == "rumordiffusion":
            checkpoint_metadata = build_checkpoint_metadata(
                model_name=name,
                stage="arbiter",
                dataset_name=source_dataset_name,
                dataset_root=dataset_root,
                seed=int(seed),
                label_mapping=dataset_label_mapping,
                config_payload={
                    "rumordiffusion": checkpoint_relevant_rumordiffusion_cfg(rd_cfg),
                    "training": {
                        "epochs": int(exp_cfg["training"]["epochs"]),
                        "lr": float(exp_cfg["training"].get("lr", 3e-4)),
                        "weight_decay": float(exp_cfg["training"].get("weight_decay", 0.0)),
                        "batch_size": int(exp_cfg["training"]["batch_size"]),
                        "early_stopping_patience": int(exp_cfg["training"].get("early_stopping_patience", 20)),
                        "min_epochs": int(exp_cfg["training"].get("min_epochs", 30)),
                    },
                    "prefix": prefix_cfg,
                    "observability": observability_cfg,
                },
            )
        if not args.eval_only:
            logging.info("Training started")
            train_start = time.time()
            fit_out = trainer.fit(train_loader=train_loader,
                                  val_loader=val_loader, save_path=ckpt_path, checkpoint_metadata=checkpoint_metadata)
            train_seconds = time.time() - train_start
            logging.info(
                f"Best val macro F1: {fit_out.get('best_val_macro_f1', fit_out.get('best_val_metric', 0.0)):.4f}")
            if wb_enabled:
                wandb.log({"best_val_macro_f1": float(fit_out.get(
                    'best_val_macro_f1', fit_out.get('best_val_metric', 0.0)))})
        else:
            logging.info("Evaluation-only run: loading checkpoint without optimizer updates")
            state, _ = load_checkpoint_state(
                ckpt_path=actual_ckpt_path,
                device=device,
                expected_metadata=checkpoint_metadata,
                validate_metadata=bool(name == "rumordiffusion" and checkpoint_metadata is not None and validate_ckpt_metadata),
                allow_legacy=bool(name != "rumordiffusion" or allow_legacy_ckpt),
            )
            trainer.model.load_state_dict(state)
            logging.info(f"Checkpoint loaded from {actual_ckpt_path}")
        eval_cfg = exp_cfg.get("evaluation", {})
        class_names = eval_cfg.get(
            "class_names", {0: "non-rumor", 1: "true", 2: "false", 3: "unverified"})
        class_names_overrides = eval_cfg.get("class_names_overrides", {})
        if isinstance(class_names_overrides, dict) and dataset_name in class_names_overrides:
            class_names = class_names_overrides[dataset_name]
        detailed_eval = True
        rd_eval_cfg = rd_cfg.get("evaluation", model_cfg.get(
            "evaluation", {})) if name == "rumordiffusion" else {}
        export_pred_cfg = eval_cfg.get("export_predictions", {}) if isinstance(
            eval_cfg, dict) else {}
        latency_cfg = eval_cfg.get("latency", {}) if isinstance(
            eval_cfg, dict) else {}
        significance_cfg = eval_cfg.get("significance", {}) if isinstance(
            eval_cfg, dict) else {}
        def persist_revision_row(row: Dict[str, str]) -> None:
            append_dict_rows_csv(revision_artifacts_csv, revision_headers, [row])

        def persist_sweep_row(row: List[str]) -> None:
            append_rows_csv(sweeps_csv, sweeps_headers, [row])

        eval_start = time.time()
        logging.info("Main test evaluation started")
        test_outputs_base = collect_eval_outputs(
            trainer.model,
            test_loader,
            device=device,
            criterion=trainer.criterion,
            amp=trainer_cfg.amp,
            detailed=detailed_eval,
            class_names=class_names,
            profile=bool(isinstance(latency_cfg, dict) and latency_cfg.get("enabled", False)),
            profile_repeat=int(latency_cfg.get("repeat", 1)) if isinstance(latency_cfg, dict) else 1,
            profile_warmup_batches=int(latency_cfg.get("warmup_batches", 0)) if isinstance(latency_cfg, dict) else 0,
            profile_max_batches=int(latency_cfg.get("max_batches", 0)) if isinstance(latency_cfg, dict) else 0,
        )
        test_metrics_base = test_outputs_base["metrics"]
        reported_logits = test_outputs_base["logits"]
        reported_targets = test_outputs_base["targets"]
        eval_seconds = time.time() - eval_start
        logging.info(f"Test loss: {test_metrics_base['loss']:.4f}")
        logging.info(f"Test accuracy: {test_metrics_base['acc']:.4f}")
        logging.info(f"Test macro F1: {test_metrics_base['macro_f1']:.4f}")
        if "report" in test_metrics_base:
            logging.info("\n=== Test Set Evaluation Report ===")
            logging.info(f"\n{test_metrics_base['report']}")
        if len(test_outputs_base["latency_ms"]) > 0:
            latency_stats = summarize_latency_ms(test_outputs_base["latency_ms"])
            persist_revision_row(
                build_revision_row(
                    timestamp=run_ts,
                    dataset_name=dataset_name,
                    model_name=name,
                    seed=int(seed),
                    artifact_type="latency_profile",
                    tag="main",
                    values={
                        "latency_mean_ms": latency_stats["mean_ms"],
                        "latency_p95_ms": latency_stats["p95_ms"],
                    },
                )
            )

        def push_sweep(tag: str, metrics: Dict[str, Any], extras: Dict[str, Any]) -> None:
            persist_sweep_row([
                run_ts,
                dataset_name,
                name,
                str(seed),
                tag,
                str(extras.get("obs_minutes", "")),
                str(extras.get("drop_node", "")),
                str(extras.get("drop_edge", "")),
                str(extras.get("edge_drop_rate", "")),
                str(extras.get("text_noise_std", "")),
                str(extras.get("text_dropout", "")),
                f"{extras.get('train_seconds', 0.0):.4f}",
                f"{extras.get('eval_seconds', 0.0):.4f}",
                f"{metrics.get('loss', 0.0):.6f}",
                f"{metrics.get('nll', 0.0):.6f}",
                f"{metrics.get('acc', 0.0):.6f}",
                f"{metrics.get('macro_f1', 0.0):.6f}",
                f"{metrics.get('ece', 0.0):.6f}",
                "" if extras.get(
                    "calibration_temp", None) is None else f"{extras['calibration_temp']:.6f}",
            ])

        def evaluate_early_stop_setting(
            obs_minutes: int,
            num_samples: int,
            tau_low_value: float,
            tau_high_value: float,
            gamma_value: float,
            rumor_classes_value: Sequence[int],
            max_batches_value: int,
        ) -> Dict[str, float]:
            total_seen = 0
            resolved = 0
            correct = 0
            mean_prob_accum = 0.0
            mean_uncert_accum = 0.0
            batch_count = 0
            latency_ms: List[float] = []
            resolved_preds: List[Tensor] = []
            resolved_targets: List[Tensor] = []
            resolved_probs: List[Tensor] = []
            obs_backup_local = int(getattr(m.ctdm, "obs_minutes", obs_minutes))
            m.ctdm.obs_minutes = int(obs_minutes)
            try:
                for batch in test_loader:
                    if int(max_batches_value) > 0 and batch_count >= int(max_batches_value):
                        break
                    batch_count += 1
                    profile_cuda_boundary(device)
                    start = time.perf_counter()
                    rumor_prob, rumor_var = trainer.model.predict_with_uncertainty(
                        batch,
                        num_samples=int(num_samples),
                        rumor_classes=list(rumor_classes_value),
                    )
                    profile_cuda_boundary(device)
                    latency_ms.append((time.perf_counter() - start) * 1000.0 / max(1, int(rumor_prob.numel())))
                    decision = torch.full_like(rumor_prob, -1, dtype=torch.long)
                    decision = torch.where(
                        (rumor_prob > float(tau_high_value)) & (rumor_var < float(gamma_value)),
                        torch.ones_like(decision),
                        decision,
                    )
                    decision = torch.where(
                        (rumor_prob < float(tau_low_value)) & (rumor_var < float(gamma_value)),
                        torch.zeros_like(decision),
                        decision,
                    )
                    target_bin = torch.isin(
                        batch.y.view(-1),
                        torch.tensor(list(rumor_classes_value), device=decision.device),
                    ).long().view_as(decision)
                    resolved_mask = decision >= 0
                    resolved += int(resolved_mask.sum().item())
                    total_seen += int(decision.numel())
                    if resolved_mask.any():
                        resolved_decision = decision[resolved_mask]
                        resolved_target = target_bin[resolved_mask]
                        correct += int((resolved_decision == resolved_target).sum().item())
                        resolved_prob = rumor_prob[resolved_mask].clamp(0.0, 1.0)
                        resolved_preds.append(resolved_decision.detach().cpu())
                        resolved_targets.append(resolved_target.detach().cpu())
                        resolved_probs.append(
                            torch.stack(
                                [1.0 - resolved_prob, resolved_prob],
                                dim=1,
                            ).detach().cpu()
                        )
                    mean_prob_accum += float(rumor_prob.mean().item())
                    mean_uncert_accum += float(rumor_var.mean().item())
            finally:
                m.ctdm.obs_minutes = obs_backup_local
            latency_stats = summarize_latency_ms(latency_ms)
            if resolved_preds:
                preds_tensor = torch.cat(resolved_preds, dim=0)
                targets_tensor = torch.cat(resolved_targets, dim=0)
                probs_tensor = torch.cat(resolved_probs, dim=0)
                resolved_macro_f1 = compute_macro_f1(preds_tensor, targets_tensor, num_classes=2)
                resolved_ece = compute_ece(probs_tensor, targets_tensor, n_bins=15)
            else:
                resolved_macro_f1 = 0.0
                resolved_ece = 0.0
            return {
                "coverage": float(resolved / max(1, total_seen)),
                "acc_resolved": float(correct / max(1, resolved)),
                "resolved_macro_f1": float(resolved_macro_f1),
                "resolved_ece": float(resolved_ece),
                "mean_prob": float(mean_prob_accum / max(1, batch_count)),
                "mean_var": float(mean_uncert_accum / max(1, batch_count)),
                "latency_mean_ms": latency_stats["mean_ms"],
                "latency_p95_ms": latency_stats["p95_ms"],
            }

        push_sweep("main", test_metrics_base, {
                   "train_seconds": train_seconds, "eval_seconds": eval_seconds})

        test_metrics = test_metrics_base
        calibration_temp = None
        cal_cfg = rd_eval_cfg.get("calibration", {}) if isinstance(
            rd_eval_cfg, dict) else {}
        if name == "rumordiffusion" and isinstance(cal_cfg, dict) and cal_cfg.get("enabled", False):
            calibration_fit_split = str(cal_cfg.get("fit_split", "val")).lower()
            if calibration_fit_split != "val":
                raise ValueError(
                    f"Unsupported calibration fit split: {calibration_fit_split}. Only 'val' is allowed."
                )
            val_logits, val_targets = collect_logits(
                trainer.model, val_loader, device, amp=trainer_cfg.amp)
            temp_opt = tune_temperature(
                val_logits.to(device),
                val_targets.to(device),
                init_temp=float(cal_cfg.get("init_temp", 1.0)),
                max_iter=int(cal_cfg.get("max_iter", 50)),
                lr=float(cal_cfg.get("lr", 0.01)),
            )
            test_logits = test_outputs_base["logits"]
            test_targets = test_outputs_base["targets"]
            logits_scaled = (test_logits.to(device) / float(temp_opt)).cpu()
            if not torch.equal(
                torch.argmax(test_logits, dim=1),
                torch.argmax(logits_scaled, dim=1),
            ):
                raise RuntimeError(
                    "Temperature scaling changed predicted labels on the stored test logits."
                )
            test_metrics_cal = metrics_from_logits(
                logits_scaled,
                test_targets,
                num_classes=num_classes,
                detailed=detailed_eval,
                class_names=class_names,
                compute_calibration=True,
                ece_bins=int(cal_cfg.get("ece_bins", 15)),
            )
            logging.info(
                f"[Calibration] Temperature={temp_opt:.3f}; "
                f"Test acc={test_metrics_cal.get('acc', 0.0):.4f}, "
                f"macro_f1={test_metrics_cal.get('macro_f1', 0.0):.4f}, "
                f"ECE={test_metrics_cal.get('ece', 0.0):.4f}"
            )
            test_metrics = test_metrics_cal
            reported_logits = logits_scaled
            reported_targets = test_targets
            calibration_temp = float(temp_opt)
            push_sweep("calibrated", test_metrics_cal, {
                       "calibration_temp": calibration_temp})
        if bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
            export_prediction_rows(
                csv_path=prediction_csv,
                dataset_name=dataset_name,
                model_name=name,
                seed=int(seed),
                tag="main",
                source_ids=test_outputs_base["source_ids"],
                logits=reported_logits,
                targets=reported_targets,
            )
        if bool(isinstance(significance_cfg, dict) and significance_cfg.get("enabled", False)) and bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
            compare_models = significance_cfg.get("baseline_models", [])
            metric_names = significance_cfg.get("metrics", ["macro_f1", "acc"])
            bootstrap_rounds = int(significance_cfg.get("n_bootstrap", 1000))
            current_rows = load_prediction_rows(
                csv_path=prediction_csv,
                dataset_name=dataset_name,
                model_name=name,
                seed=int(seed),
                tag="main",
            )
            for compare_model in compare_models:
                baseline_rows = load_prediction_rows(
                    csv_path=prediction_csv,
                    dataset_name=dataset_name,
                    model_name=str(compare_model),
                    seed=int(seed),
                    tag="main",
                )
                if len(current_rows) == 0 or len(baseline_rows) == 0:
                    continue
                for metric_name in metric_names:
                    stats = bootstrap_compare_predictions(
                        current_rows=current_rows,
                        baseline_rows=baseline_rows,
                        metric_name=str(metric_name),
                        n_bootstrap=bootstrap_rounds,
                        random_seed=int(seed),
                    )
                    persist_revision_row(
                        build_revision_row(
                            timestamp=run_ts,
                            dataset_name=dataset_name,
                            model_name=name,
                            seed=int(seed),
                            artifact_type="significance",
                            tag="main",
                            values={
                                "compare_model": str(compare_model),
                                "metric": str(metric_name),
                                "delta": stats["delta"],
                                "p_value": stats["p_value"],
                                "ci_low": stats["ci_low"],
                                "ci_high": stats["ci_high"],
                                "notes": f"paired_bootstrap_n={int(bootstrap_rounds)} shared={int(stats['count'])}",
                            },
                        )
                    )

        result_headers = [
            "timestamp",
            "dataset",
            "model",
            "model_cfg",
            "exp_cfg",
            "ckpt_path",
            "log_path",
            "seed",
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
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_results_csv(
            csv_path=csv_path,
            headers=result_headers,
            row=[
                ts,
                dataset_name,
                name,
                model_path,
                args.exp,
                actual_ckpt_path,
                log_path,
                str(seed),
                f"{fit_out.get('best_val_macro_f1', fit_out.get('best_val_metric', 0.0)):.6f}",
                f"{test_metrics.get('loss', 0.0):.6f}",
                f"{test_metrics.get('nll', 0.0):.6f}",
                f"{test_metrics.get('acc', 0.0):.6f}",
                f"{test_metrics.get('macro_f1', 0.0):.6f}",
                f"{test_metrics.get('macro_precision', 0.0):.6f}",
                f"{test_metrics.get('macro_recall', 0.0):.6f}",
                f"{test_metrics.get('weighted_f1', 0.0):.6f}",
                f"{test_metrics.get('weighted_precision', 0.0):.6f}",
                f"{test_metrics.get('weighted_recall', 0.0):.6f}",
                f"{test_metrics.get('ece', 0.0):.6f}",
                "" if calibration_temp is None else f"{calibration_temp:.6f}",
            ],
        )
        if dataset_name != source_dataset_name:
            transfer_headers = [
                "timestamp",
                "train_dataset",
                "test_dataset",
                "model",
                "seed",
                "ckpt_path",
                "acc",
                "macro_f1",
                "nll",
                "ece",
                "calibration_temp",
            ]
            append_results_csv(
                csv_path=transfer_csv,
                headers=transfer_headers,
                row=[
                    ts,
                    source_dataset_name,
                    dataset_name,
                    name,
                    str(seed),
                    actual_ckpt_path,
                    f"{test_metrics.get('acc', 0.0):.6f}",
                    f"{test_metrics.get('macro_f1', 0.0):.6f}",
                    f"{test_metrics.get('nll', 0.0):.6f}",
                    f"{test_metrics.get('ece', 0.0):.6f}",
                    "" if calibration_temp is None else f"{calibration_temp:.6f}",
                ],
            )
        if wb_enabled:
            import wandb

            wandb.log({
                "test/loss": float(test_metrics.get("loss", 0.0)),
                "test/acc": float(test_metrics.get("acc", 0.0)),
                "test/macro_f1": float(test_metrics.get("macro_f1", 0.0)),
            })
            wandb.finish()

        early_windows = rd_eval_cfg.get(
            "early_obs_minutes", []) if isinstance(rd_eval_cfg, dict) else []
        early_overrides = rd_eval_cfg.get(
            "early_obs_minutes_overrides", {}) if isinstance(rd_eval_cfg, dict) else {}
        if isinstance(early_overrides, dict) and dataset_name in early_overrides:
            early_windows = early_overrides[dataset_name]
        early_stop_cfg = rd_eval_cfg.get(
            "early_stop", {}) if isinstance(rd_eval_cfg, dict) else {}
        early_stop_over = rd_eval_cfg.get(
            "early_stop_overrides", {}) if isinstance(rd_eval_cfg, dict) else {}
        if isinstance(early_stop_over, dict) and dataset_name in early_stop_over:
            cfg_over = early_stop_over[dataset_name]
            if isinstance(cfg_over, dict):
                early_stop_cfg = {**early_stop_cfg, **cfg_over}
        early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
        early_stop_samples = int(early_stop_cfg.get(
            "samples", 4)) if early_stop_enabled else 0
        tau_low = float(early_stop_cfg.get("tau_low", 0.3)
                        ) if early_stop_enabled else 0.0
        tau_high = float(early_stop_cfg.get("tau_high", 0.7)
                         ) if early_stop_enabled else 0.0
        gamma_uncert = float(early_stop_cfg.get(
            "gamma", 0.02)) if early_stop_enabled else 0.0
        rumor_classes_cfg = early_stop_cfg.get(
            "rumor_classes", [1, 2, 3]) if early_stop_enabled else []
        early_stop_max_batches = int(early_stop_cfg.get(
            "max_batches", 0)) if isinstance(early_stop_cfg, dict) else 0
        sampling_sweep_cfg = rd_eval_cfg.get(
            "sampling_sweep", {}) if isinstance(rd_eval_cfg, dict) else {}
        threshold_sweep_cfg = rd_eval_cfg.get(
            "threshold_sweep", {}) if isinstance(rd_eval_cfg, dict) else {}
        if name == "rumordiffusion" and hasattr(m, "ctdm") and isinstance(early_windows, (list, tuple)) and len(early_windows) > 0:
            obs_backup = int(getattr(m.ctdm, "obs_minutes", 30))
            for t in early_windows:
                m.ctdm.obs_minutes = int(t)
                metrics_early = trainer.evaluate(
                    test_loader, detailed=True, class_names=class_names)
                push_sweep(f"early_{t}", metrics_early, {"obs_minutes": t})
                if early_stop_enabled and hasattr(trainer.model, "predict_with_uncertainty"):
                    early_stop_result = evaluate_early_stop_setting(
                        obs_minutes=int(t),
                        num_samples=int(early_stop_samples),
                        tau_low_value=float(tau_low),
                        tau_high_value=float(tau_high),
                        gamma_value=float(gamma_uncert),
                        rumor_classes_value=rumor_classes_cfg,
                        max_batches_value=int(early_stop_max_batches),
                    )
                    logging.info(
                        f"[EarlyStop] obs_minutes={t}, samples={early_stop_samples}, "
                        f"coverage={early_stop_result['coverage']:.4f}, acc_resolved={early_stop_result['acc_resolved']:.4f}, "
                        f"macro_f1_resolved={early_stop_result['resolved_macro_f1']:.4f}, "
                        f"ece_resolved={early_stop_result['resolved_ece']:.4f}, "
                        f"mean_prob={early_stop_result['mean_prob']:.4f}, mean_var={early_stop_result['mean_var']:.4f}, "
                        f"latency_mean_ms={early_stop_result['latency_mean_ms']:.4f}, latency_p95_ms={early_stop_result['latency_p95_ms']:.4f}"
                    )
                    persist_revision_row(
                        build_revision_row(
                            timestamp=run_ts,
                            dataset_name=dataset_name,
                            model_name=name,
                            seed=int(seed),
                            artifact_type="early_stop",
                            tag=f"early_{t}",
                            values={
                                "obs_minutes": int(t),
                                "samples": int(early_stop_samples),
                                "tau_low": float(tau_low),
                                "tau_high": float(tau_high),
                                "gamma": float(gamma_uncert),
                                **early_stop_result,
                            },
                        )
                    )
                    if isinstance(sampling_sweep_cfg, dict) and bool(sampling_sweep_cfg.get("enabled", False)):
                        for sweep_samples in sampling_sweep_cfg.get("samples", []):
                            sweep_result = evaluate_early_stop_setting(
                                obs_minutes=int(t),
                                num_samples=int(sweep_samples),
                                tau_low_value=float(tau_low),
                                tau_high_value=float(tau_high),
                                gamma_value=float(gamma_uncert),
                                rumor_classes_value=rumor_classes_cfg,
                                max_batches_value=int(sampling_sweep_cfg.get("max_batches", early_stop_max_batches)),
                            )
                            persist_revision_row(
                                build_revision_row(
                                    timestamp=run_ts,
                                    dataset_name=dataset_name,
                                    model_name=name,
                                    seed=int(seed),
                                    artifact_type="sampling_sweep",
                                    tag=f"early_{t}_samples_{int(sweep_samples)}",
                                    values={
                                        "obs_minutes": int(t),
                                        "samples": int(sweep_samples),
                                        "tau_low": float(tau_low),
                                        "tau_high": float(tau_high),
                                        "gamma": float(gamma_uncert),
                                        **sweep_result,
                                    },
                                )
                            )
                    if isinstance(threshold_sweep_cfg, dict) and bool(threshold_sweep_cfg.get("enabled", False)):
                        tau_low_list = threshold_sweep_cfg.get("tau_low", [])
                        tau_high_list = threshold_sweep_cfg.get("tau_high", [])
                        gamma_list = threshold_sweep_cfg.get("gamma", [])
                        for tau_low_value in tau_low_list:
                            for tau_high_value in tau_high_list:
                                if float(tau_low_value) >= float(tau_high_value):
                                    continue
                                for gamma_value in gamma_list:
                                    sweep_result = evaluate_early_stop_setting(
                                        obs_minutes=int(t),
                                        num_samples=int(threshold_sweep_cfg.get("samples", early_stop_samples)),
                                        tau_low_value=float(tau_low_value),
                                        tau_high_value=float(tau_high_value),
                                        gamma_value=float(gamma_value),
                                        rumor_classes_value=rumor_classes_cfg,
                                        max_batches_value=int(threshold_sweep_cfg.get("max_batches", early_stop_max_batches)),
                                    )
                                    persist_revision_row(
                                        build_revision_row(
                                            timestamp=run_ts,
                                            dataset_name=dataset_name,
                                            model_name=name,
                                            seed=int(seed),
                                            artifact_type="threshold_sweep",
                                            tag=f"early_{t}_tau{float(tau_low_value):.2f}_{float(tau_high_value):.2f}_gamma{float(gamma_value):.2f}",
                                            values={
                                                "obs_minutes": int(t),
                                                "samples": int(threshold_sweep_cfg.get("samples", early_stop_samples)),
                                                "tau_low": float(tau_low_value),
                                                "tau_high": float(tau_high_value),
                                                "gamma": float(gamma_value),
                                                **sweep_result,
                                            },
                                        )
                                    )
            m.ctdm.obs_minutes = obs_backup
        robust_cfg = rd_eval_cfg.get("robustness", {}) if isinstance(
            rd_eval_cfg, dict) else {}
        drop_sweep = robust_cfg.get("edge_dropout", []) if isinstance(
            robust_cfg, dict) else []
        if name == "rumordiffusion" and hasattr(m, "ctdm") and isinstance(drop_sweep, (list, tuple)) and len(drop_sweep) > 0:
            edge_backup = float(getattr(m.ctdm, "edge_drop_rate", 0.0))
            for p in drop_sweep:
                m.ctdm.edge_drop_rate = float(p)
                metrics_drop = trainer.evaluate(
                    test_loader, detailed=True, class_names=class_names)
                push_sweep(f"robust_edge_{p}", metrics_drop, {
                           "edge_drop_rate": p})
            m.ctdm.edge_drop_rate = edge_backup

        text_noise_list = robust_text_cfg.get(
            "noise_std", []) if isinstance(robust_text_cfg, dict) else []
        text_drop_list = robust_text_cfg.get(
            "dropout_p", []) if isinstance(robust_text_cfg, dict) else []
        if name == "rumordiffusion" and hasattr(m, "ctdm") and (len(text_noise_list) > 0 or len(text_drop_list) > 0):
            noise_candidates = text_noise_list if len(
                text_noise_list) > 0 else [0.0]
            drop_candidates = text_drop_list if len(
                text_drop_list) > 0 else [0.0]
            noise_backup = float(getattr(m.ctdm, "text_noise_std", 0.0))
            drop_backup = float(getattr(m.ctdm, "text_dropout_p", 0.0))
            for n_std in noise_candidates:
                for d_p in drop_candidates:
                    m.ctdm.text_noise_std = float(n_std)
                    m.ctdm.text_dropout_p = float(d_p)
                    metrics_text_rob = trainer.evaluate(
                        test_loader, detailed=True, class_names=class_names)
                    push_sweep(f"text_robust_noise{n_std}_drop{d_p}", metrics_text_rob, {
                               "text_noise_std": n_std, "text_dropout": d_p})
            m.ctdm.text_noise_std = noise_backup
            m.ctdm.text_dropout_p = drop_backup

        drop_node_list = observability_sweep_cfg.get(
            "drop_node", []) if isinstance(observability_sweep_cfg, dict) else []
        drop_edge_list = observability_sweep_cfg.get(
            "drop_edge", []) if isinstance(observability_sweep_cfg, dict) else []
        if len(drop_node_list) > 0 or len(drop_edge_list) > 0:
            dn_candidates = drop_node_list if len(drop_node_list) > 0 else [
                observability_cfg.get("drop_node", 0.0)]
            de_candidates = drop_edge_list if len(drop_edge_list) > 0 else [
                observability_cfg.get("drop_edge", 0.0)]
            for dn in dn_candidates:
                for de in de_candidates:
                    ds_test_obs = GraphDataset(
                        root=dataset_root,
                        split="test",
                        use_lp_edges=use_lp,
                        prefix_cfg=prefix_cfg,
                        observability_cfg={"drop_node": float(dn), "drop_edge": float(
                            de), "seed": observability_cfg.get("seed", 0)},
                    )
                    test_loader_obs = make_loader(ds_test_obs, shuffle=False)
                    metrics_obs = trainer.evaluate(
                        test_loader_obs, detailed=True, class_names=class_names)
                    push_sweep(f"robust_obs_node{dn}_edge{de}", metrics_obs, {
                               "drop_node": dn, "drop_edge": de})
        seed_results.append(
            {
                "val_macro_f1": float(fit_out.get("best_val_macro_f1", fit_out.get("best_val_metric", 0.0))),
                "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                "test_acc": float(test_metrics.get("acc", 0.0)),
                "test_nll": float(test_metrics.get("nll", 0.0)),
            }
        )

        if len(seeds) > 1 and len(seed_results) == len(seeds):
            agg_headers = result_headers
            mean_val, std_val = aggregate_numeric(
                [r["val_macro_f1"] for r in seed_results])
            mean_f1, std_f1 = aggregate_numeric(
                [r["test_macro_f1"] for r in seed_results])
            mean_acc, std_acc = aggregate_numeric(
                [r["test_acc"] for r in seed_results])
            mean_nll, std_nll = aggregate_numeric(
                [r["test_nll"] for r in seed_results])
            append_results_csv(
                csv_path=csv_path,
                headers=agg_headers,
                row=[
                    ts,
                    dataset_name,
                    f"{name}_agg",
                    model_path,
                    args.exp,
                    "",
                    "",
                    "agg",
                    f"{mean_val:.6f}±{std_val:.6f}",
                    "",
                    f"{mean_nll:.6f}±{std_nll:.6f}",
                    f"{mean_acc:.6f}±{std_acc:.6f}",
                    f"{mean_f1:.6f}±{std_f1:.6f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            )


if __name__ == "__main__":
    main()
