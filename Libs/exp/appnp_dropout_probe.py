"""Compare APPNP edge-dropout semantics on a fixed real TRAIN batch.

This is an initialization diagnostic, not a trained performance comparison.
The native implementation and the fresh-weight reference share all parameters,
inputs and dropout seeds. No optimizer step or checkpoint read is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
from types import MethodType

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn.conv.gcn_conv import gcn_norm


SEEDS = (41, 42, 43)
SOURCE_COMMIT = "a2441b4b4c09b04014c76c74df5ff86fe59cc309"


def fresh_weight_forward(self, x: Tensor, edge_index: Tensor, edge_weight=None):
    """Diagnostic Tensor-edge branch: each mask starts from normalized weights."""
    if edge_index.layout != torch.strided:
        raise ValueError("This probe requires the recorded dense edge-index input")
    if self.cached:
        raise ValueError("The probe requires the recorded uncached propagation")
    if self.normalize:
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(self.node_dim), False,
            self.add_self_loops, self.flow, dtype=x.dtype,
        )
    if edge_weight is None:
        raise ValueError("The recorded APPNP path requires normalized edge weights")
    h = x
    for _ in range(self.K):
        weights = edge_weight
        if self.dropout > 0 and self.training:
            weights = F.dropout(edge_weight, p=self.dropout)
        x = self.propagate(edge_index, x=x, edge_weight=weights)
        x = x * (1 - self.alpha)
        x = x + self.alpha * h
    return x


def finite_number(value: Tensor) -> float | None:
    result = float(value.detach().to(torch.float64))
    return result if math.isfinite(result) else None


def macro_f1(labels: Tensor, predictions: Tensor) -> tuple[float, list[list[int]]]:
    matrix = [[int(((labels == y) & (predictions == p)).sum()) for p in (0, 1)]
              for y in (0, 1)]
    values = []
    for c in (0, 1):
        denominator = sum(matrix[c]) + sum(row[c] for row in matrix)
        values.append(2 * matrix[c][c] / denominator if denominator else 0.0)
    return sum(values) / 2, matrix


def measure(model, batch: Batch, seed: int) -> dict:
    edge_trace = []

    def record_weights(module, inputs):
        weights = inputs[2]["edge_weight"].detach()
        edge_trace.append({
            "step": len(edge_trace) + 1,
            "edges": weights.numel(),
            "nonzero": int((weights != 0).sum()),
            "max_abs_weight": finite_number(weights.abs().max()),
            "finite": bool(torch.isfinite(weights).all()),
        })

    handle = model.propagation.register_propagate_forward_pre_hook(record_weights)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        gradients = [p.grad.detach().reshape(-1).to(torch.float64)
                     for p in model.parameters() if p.grad is not None]
        if len(gradients) != sum(1 for _ in model.parameters()):
            raise ValueError("A model parameter lacks its gradient")
        gradient = torch.cat(gradients)
        score, confusion = macro_f1(batch.y.view(-1), logits.detach().argmax(1))
        return {
            "train_batch_macro_f1": score,
            "confusion": confusion,
            "cross_entropy": finite_number(loss),
            "max_abs_logit": finite_number(logits.abs().max()),
            "raw_gradient_l2": finite_number(torch.linalg.vector_norm(gradient)),
            "max_abs_raw_gradient": finite_number(gradient.abs().max()),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "raw_gradients_finite": bool(torch.isfinite(gradient).all()),
            "edge_trace": edge_trace,
        }
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The diagnostic report already exists")
    if args.samples < 1:
        raise ValueError("The real TRAIN batch must be nonempty")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("This diagnostic must run on CPU")
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed_commit != SOURCE_COMMIT:
        raise ValueError("The training source commit changed")
    sys.path.insert(0, str(args.source_root))
    from Libs.data.dataset import GraphDataset
    from Libs.models.baselines.APPNPNet import APPNPNet

    control_path = args.project_root / "Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_execution_plan.json"
    control = json.loads(control_path.read_text())
    binding = next(x for x in control["case_evidence_bindings"]
                   if (x["dataset"], x["model"], x["seed"]) == ("weibo", "appnp", 41))
    config = binding["configuration"]
    model_config = binding["model_configuration"]["model"]
    dataset_config = config["dataset"]
    prefix = {"strategy": "hop", "unit": "hop", "window": 3, "include_root": True}
    dataset = GraphDataset(
        root=str(Path(dataset_config["root"]) / "weibo"), split="train",
        mode="train_arbiter", use_lp_edges=False, use_potentials=False,
        prefix_cfg=prefix, observability_cfg=dataset_config["observability"],
        text_cache_name=dataset_config["text_encoder"]["cache_name"],
        force_text_cache=dataset_config["text_encoder"]["force_text_cache"],
        text_normalize=dataset_config["text_encoder"]["normalize_embeddings"],
    )
    if len(dataset) != 3729 or args.samples > len(dataset):
        raise ValueError("The frozen real TRAIN split or probe size changed")
    graphs = [dataset.get(i) for i in range(args.samples)]
    if any(not hasattr(g, "source_id") for g in graphs):
        raise ValueError("A TRAIN cascade lacks its original source ID")
    identities = [str(g.source_id) for g in graphs]
    if len(set(identities)) != len(identities):
        raise ValueError("A real TRAIN source ID is duplicated")
    batch = Batch.from_data_list(graphs)
    if batch.x.size(1) != 8 or not torch.isfinite(batch.x).all():
        raise ValueError("The recorded structural feature input changed")
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        native = APPNPNet(
            in_channels=8, hidden_channels=model_config["hidden"],
            num_layers=model_config["num_layers"], num_classes=2,
            dropout=model_config["dropout"], appnp_alpha=model_config["alpha"],
            appnp_k=model_config["k"],
        )
        reference = copy.deepcopy(native)
        reference.propagation.forward = MethodType(fresh_weight_forward, reference.propagation)
        if not all(torch.equal(v, reference.state_dict()[k]) for k, v in native.state_dict().items()):
            raise ValueError("The paired models have different parameters")
        native.eval()
        reference.eval()
        with torch.no_grad():
            evaluation_equal = torch.equal(native(batch), reference(batch))
        if not evaluation_equal:
            raise ValueError("The reference changed evaluation behavior")
        results.append({
            "seed": seed, "parameter_states_identical": True,
            "evaluation_logits_bitwise_equal": evaluation_equal,
            "native_cumulative_dropout": measure(native, batch, seed),
            "fresh_normalized_weight_dropout": measure(reference, batch, seed),
        })
    source_paths = {
        "diagnostic_runner": Path(__file__),
        "appnp_model": args.source_root / "Libs/models/baselines/APPNPNet.py",
        "installed_appnp": Path(importlib.metadata.distribution("torch-geometric").locate_file("torch_geometric/nn/conv/appnp.py")),
    }
    report = {
        "status": "REAL_TRAIN_PAIRED_DROPOUT_PROBE_COMPLETED",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": socket.gethostname(), "allocation_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_commit": observed_commit,
        "torch_geometric_version": importlib.metadata.version("torch-geometric"),
        "scope": "Initialization diagnostic on one fixed real TRAIN batch; no performance qualification",
        "question": "Does cumulative edge-dropout scaling amplify the real TRAIN forward and raw gradients?",
        "comparison": "Same real inputs, parameters and seeds; only per-step edge-weight dropout semantics differ",
        "train_batch_selection": "First split-local cascades in the immutable TRAIN order; no metric-based selection",
        "dataset": "weibo", "split": "train", "horizon": {"unit": "hop", "value": 3},
        "source_ids": identities, "split_local_indices": list(range(args.samples)),
        "raw_indices": [dataset._indices[i] for i in range(args.samples)],
        "class_counts": {str(c): int((batch.y.view(-1) == c).sum()) for c in (0, 1)},
        "graph_count": args.samples, "node_count": batch.num_nodes,
        "edge_count": batch.edge_index.size(1),
        "max_abs_input_feature": finite_number(batch.x.abs().max()),
        "model_config": model_config, "seeds": list(SEEDS),
        "input_pins_reused_from_binding": control["current_serial_execution"]["expected_dataset_split_evidence"]["input_pins"],
        "source_files": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in source_paths.items()},
        "results": results,
        "optimizer_updates": 0, "checkpoint_reads": 0,
        "validation_or_test_predictions_read": False,
        "native_training_unchanged": True, "project_complete": False,
        "limits": "Not trained, not a representative full-split result, and not causal proof of the recorded classifier collapse",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if json.loads(args.output.read_text()) != report:
        raise ValueError("The diagnostic report readback differs")
    print(json.dumps({"status": report["status"], "graph_count": args.samples, "results": results}, allow_nan=False))


if __name__ == "__main__":
    main()
