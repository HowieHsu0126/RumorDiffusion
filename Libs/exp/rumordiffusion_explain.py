"""Lightweight interpretability runner for RumorDiffusion.

This script loads a RumorDiffusion checkpoint, runs inference on a chosen
split, and saves per-sample logits, probabilities, and token-level statistics
for post-hoc analysis. All parameters come from YAML configs to avoid
hardcoding.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader as PyGDataLoader

from Libs.data.dataset import GraphDataset
from Libs.exp.run import read_yaml_like
from Libs.models.baselines.CTDM_syn import CTDMSyn
from Libs.models.baselines.RumorDiffusion import RumorDiffusion
from Libs.models.modules.rumor_diffusion import _extract_source_ids_from_batch


def build_device(device_cfg: Dict) -> str:
    """Selects device string based on config and availability."""
    force_cpu = bool(device_cfg.get("force_cpu", False))
    if force_cpu:
        return "cpu"
    cfg_type = str(device_cfg.get("type", "cuda")).lower()
    return "cuda" if cfg_type == "cuda" and torch.cuda.is_available() else cfg_type


def build_ctdm(ctdm_cfg: Dict, num_features: int, num_classes: int) -> CTDMSyn:
    """Constructs CTDMSyn with backbone, text, and regularizer settings."""
    bb_cfg = ctdm_cfg.get("backbone", {}) if isinstance(ctdm_cfg.get("backbone", {}), dict) else {}
    text_cfg = ctdm_cfg.get("text", {}) if isinstance(ctdm_cfg.get("text", {}), dict) else {}
    text_enabled = bool(text_cfg.get("enabled", False))
    text_dim = int(text_cfg.get("text_dim", 0)) if text_enabled else 0
    text_fuse = str(text_cfg.get("fuse_mode", "add")) if text_enabled else "add"

    model = CTDMSyn(
        in_channels=num_features,
        num_classes=num_classes,
        hidden_channels=int(ctdm_cfg.get("hidden", 128)),
        num_layers=int(ctdm_cfg.get("num_layers", 3)),
        dropout=float(ctdm_cfg.get("dropout", 0.3)),
        T=int(ctdm_cfg.get("T", 200)),
        obs_minutes=int(ctdm_cfg.get("obs_minutes", 30)),
        t_embed_dim=int(ctdm_cfg.get("t_embed_dim", ctdm_cfg.get("hidden", 128))),
        film_hidden=int(ctdm_cfg.get("film_hidden", 128)),
        max_nodes=int(ctdm_cfg.get("max_nodes", 2048)),
        backbone_name=str(bb_cfg.get("name", "gcn")).lower(),
        backbone_heads=int(bb_cfg.get("heads", 1)),
        backbone_dropout=float(bb_cfg.get("dropout", ctdm_cfg.get("dropout", 0.3))),
        deg_histogram=None,
        pna_aggregators=bb_cfg.get("aggregators", None),
        pna_scalers=bb_cfg.get("scalers", None),
        text_dim=(text_dim if text_enabled else None),
        text_fuse=text_fuse,
    )
    if "t_stride" in ctdm_cfg:
        setattr(model, "t_stride", int(ctdm_cfg["t_stride"]))
    if "parameterization" in ctdm_cfg:
        model.parameterization = str(ctdm_cfg["parameterization"]).lower()
    model.guidance_p = float(ctdm_cfg.get("guidance_p", 0.2))
    model.guidance_scale = float(ctdm_cfg.get("guidance_scale", 2.0))
    if "use_attn_readout" in ctdm_cfg:
        model.use_attn_readout = bool(ctdm_cfg["use_attn_readout"])

    mw_cfg = ctdm_cfg.get("multiwindow", {}) if isinstance(ctdm_cfg.get("multiwindow", {}), dict) else {}
    model.multiwindow_enabled = bool(mw_cfg.get("enabled", False))
    model.cutoff_min = int(mw_cfg.get("min", 10))
    model.cutoff_max = int(mw_cfg.get("max", 30))

    sp_cfg = ctdm_cfg.get("sparse_pairs", {}) if isinstance(ctdm_cfg.get("sparse_pairs", {}), dict) else {}
    model.sparse_pairs_enabled = bool(sp_cfg.get("enabled", False))
    model.sparse_pairs_strategy = str(sp_cfg.get("strategy", "b")).lower()
    model.sparse_pairs_topk = int(sp_cfg.get("topk", 8))
    if "lambda_time" in sp_cfg:
        model.sparse_pairs_lambda_time = float(sp_cfg["lambda_time"])
    if "lambda_degree" in sp_cfg:
        model.sparse_pairs_lambda_degree = float(sp_cfg["lambda_degree"])

    reg_cfg = ctdm_cfg.get("regularizers", {}) if isinstance(ctdm_cfg.get("regularizers", {}), dict) else {}
    if "lambda_proto" in reg_cfg:
        model.lambda_proto = float(reg_cfg["lambda_proto"])
    if "temperature" in reg_cfg:
        model.prototype_temperature = float(reg_cfg["temperature"])
    if "momentum" in reg_cfg:
        model.prototype_momentum = float(reg_cfg["momentum"])
    if "lambda_topo_bce" in reg_cfg:
        model.lambda_topo_bce = float(reg_cfg["lambda_topo_bce"])
    if "lambda_deg_kl" in reg_cfg:
        model.lambda_deg_kl = float(reg_cfg["lambda_deg_kl"])
    if "lambda_proto_orth" in reg_cfg:
        model.lambda_proto_orth = float(reg_cfg["lambda_proto_orth"])
    if "lambda_supcon_pair" in reg_cfg:
        model.lambda_supcon_pair = float(reg_cfg["lambda_supcon_pair"])
    if "pair_ids" in reg_cfg:
        model.pair_ids = tuple(reg_cfg["pair_ids"])
    return model


def infer_hidden_channels(ds: GraphDataset, ctdm: CTDMSyn, device: torch.device) -> Tuple[int, int]:
    """Infers potential and concatenated hidden dimensions from a dataset sample."""
    sample = ds.get(0)
    if hasattr(sample, "h0") and hasattr(sample, "h1"):
        h_dim = int(sample.h0.size(-1))
        concat_dim = int(torch.cat([sample.h0, sample.h1], dim=-1).size(-1))
        return h_dim, concat_dim
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        potentials = ctdm.infer_potentials_all(batch)
    h_dim = int(potentials.size(-1))
    concat_dim = int(torch.cat([potentials[:, 0, :], potentials[:, 1, :]], dim=-1).size(-1))
    return h_dim, concat_dim


def build_rumor_diffusion(model_cfg: Dict, exp_cfg: Dict, ds: GraphDataset, device: torch.device) -> RumorDiffusion:
    """Builds RumorDiffusion model using YAML configuration."""
    ctdm_cfg = model_cfg.get("ctdm", {})
    arb_cfg = model_cfg.get("arbiter", {})
    loss_cfg = model_cfg.get("loss", {})
    dd_cfg = model_cfg.get("dd_gan", {})

    num_features = ds.num_features
    num_classes = ds.num_classes
    ctdm = build_ctdm(ctdm_cfg, num_features, num_classes)
    ctdm = ctdm.to(device)

    hidden_dim, _ = infer_hidden_channels(ds, ctdm, device)

    model = RumorDiffusion(
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        arbiter_hidden=int(arb_cfg.get("hidden", 256)),
        dropout=float(arb_cfg.get("dropout", 0.3)),
        ctdm=ctdm,
        freeze_ctdm=bool(arb_cfg.get("freeze_ctdm", True)),
        arbiter_type=str(arb_cfg.get("type", "transformer")).lower(),
        arbiter_readout=str(arb_cfg.get("readout", "cls")).lower(),
        token_norm=str(arb_cfg.get("token_norm", "layernorm")).lower(),
        token_dropout=float(arb_cfg.get("token_dropout", arb_cfg.get("dropout", 0.3))),
        transformer_nhead=int(arb_cfg.get("transformer", {}).get("nhead", 4)),
        transformer_nlayers=int(arb_cfg.get("transformer", {}).get("nlayers", 2)),
        transformer_dim_feedforward=int(arb_cfg.get("transformer", {}).get("dim_feedforward", 512)),
        transformer_dropout=float(arb_cfg.get("transformer", {}).get("dropout", arb_cfg.get("dropout", 0.3))),
        dd_gan_enabled=bool(dd_cfg.get("enabled", False)),
        enable_dd_gan=bool(dd_cfg.get("enable", dd_cfg.get("enabled", False))),
        lambda_gan=float(dd_cfg.get("lambda_gan", 0.0)),
        lr_d=float(dd_cfg.get("lr_d", 1.0e-4)),
        lambda_r1=float(dd_cfg.get("lambda_r1", 10.0)),
        use_pair_token=bool(arb_cfg.get("use_pair_token", False)),
        pair_token_mode=str(arb_cfg.get("pair_token_mode", "diff")),
        pair_ids=tuple(arb_cfg.get("pair_ids", [1, 2])),
        tf_head_enabled=bool(arb_cfg.get("tf_head_enabled", False)),
        tf_infer_adjust_alpha=float(arb_cfg.get("tf_infer_adjust_alpha", 0.0)),
        pair_margin_enabled=bool(loss_cfg.get("pair_margin_enabled", False)),
        pair_margin_m=float(loss_cfg.get("pair_margin_m", 0.5)),
        lambda_pair_margin=float(loss_cfg.get("lambda_pair_margin", 0.0)),
        enable_pair_margin_loss=bool(loss_cfg.get("enable_pair_margin_loss", True)),
        lambda_tf_bce=float(loss_cfg.get("lambda_tf_bce", 0.0)),
        use_class_embedding=bool(arb_cfg.get("use_class_embedding", False)),
        class_embed_mode=str(arb_cfg.get("class_embed_mode", "concat_proj")),
        cache_potentials=bool(arb_cfg.get("cache_potentials", False)),
        infer_amp=bool(arb_cfg.get("infer_amp", False)),
        loss_logging=bool(loss_cfg.get("loss_logging", False)),
    )

    if "class_priors" in loss_cfg:
        pri = torch.tensor(loss_cfg["class_priors"], dtype=torch.float32, device=device)
        model.class_priors = pri / pri.sum()
    model.la_tau = float(loss_cfg.get("la_tau", 1.0))
    model.la_gamma = float(loss_cfg.get("la_gamma", 1.5))
    model.la_smooth = float(loss_cfg.get("la_smooth", 0.05))
    model.lambda_brier = float(loss_cfg.get("w_brier", 0.01))
    model.lambda_consistency = float(loss_cfg.get("w_cons", 0.05))

    mp_enabled = bool(loss_cfg.get("enable_multi_pair_margin", loss_cfg.get("multi_pair_margin_enabled", False)))
    model.multi_pair_margin_list = loss_cfg.get("multi_pair_margin_list", []) if mp_enabled else []
    model.multi_pair_margin_values = loss_cfg.get("multi_pair_margin_values", []) if mp_enabled else []
    model.lambda_multi_pair_margin = float(loss_cfg.get("lambda_multi_pair_margin", 0.0)) if mp_enabled else 0.0
    model.enable_multi_pair_margin = mp_enabled and len(model.multi_pair_margin_list) > 0 and float(model.lambda_multi_pair_margin) > 0.0

    model.enable_pair_margin_loss = bool(loss_cfg.get("enable_pair_margin_loss", loss_cfg.get("pair_margin_enabled", False)))
    if not model.enable_pair_margin_loss:
        model.lambda_pair_margin = 0.0
    model.enable_tf_head_loss = bool(loss_cfg.get("enable_tf_head_loss", True))
    if not model.enable_tf_head_loss:
        model.lambda_tf_bce = 0.0

    eval_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    if "class_priors" not in loss_cfg and isinstance(eval_cfg, dict) and "class_names" in eval_cfg:
        counts = [0 for _ in range(int(num_classes))]
        for raw_idx in range(len(ds)):
            yv = int(ds.get(raw_idx).y.view(-1)[0].item())
            counts[yv] += 1
        total = sum(counts)
        pri = torch.tensor([c / float(total) for c in counts], dtype=torch.float32, device=device)
        model.class_priors = pri
    return model.to(device)


def load_checkpoint(model: RumorDiffusion, ckpt_path: str, device: torch.device) -> None:
    """Loads model weights from checkpoint path."""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


def collect_explanations(
    model: RumorDiffusion,
    loader: PyGDataLoader,
    device: torch.device,
    pair_ids: Tuple[int, int],
    out_path: str,
) -> Tuple[int, float, float]:
    """Runs inference and saves JSON lines with logits and token stats."""
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records: List[Dict] = []
    p0, p1 = int(pair_ids[0]), int(pair_ids[1])
    total_prob = 0.0
    total_margin = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tokens = model._build_tokens_from_batch(batch)
            logits, cls_out = model._encode_from_tokens(tokens)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.max(dim=1)
            margin = logits[:, p0] - logits[:, p1]
            cls_norm = cls_out.norm(dim=1)
            token_norm_mean = tokens.norm(dim=2).mean(dim=1)
            token_norm_max = tokens.norm(dim=2).max(dim=1).values
            sids = _extract_source_ids_from_batch(batch)

            labels = batch.y.view(-1).detach().cpu() if hasattr(batch, "y") else None
            for i, sid in enumerate(sids):
                rec = {
                    "source_id": sid,
                    "pred_class": int(top_idx[i].item()),
                    "pred_prob": float(top_prob[i].item()),
                    "pair_margin": float(margin[i].item()),
                    "prob_true": float(probs[i, p0].item()) if p0 < probs.size(1) else 0.0,
                    "prob_false": float(probs[i, p1].item()) if p1 < probs.size(1) else 0.0,
                    "cls_norm": float(cls_norm[i].item()),
                    "token_norm_mean": float(token_norm_mean[i].item()),
                    "token_norm_max": float(token_norm_max[i].item()),
                    "probs": [float(v) for v in probs[i].tolist()],
                }
                if labels is not None and i < labels.numel():
                    rec["label"] = int(labels[i].item())
                    rec["correct"] = int(rec["pred_class"] == rec["label"])
                records.append(rec)
                total_prob += float(top_prob[i].item())
                total_margin += float(margin[i].item())

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    count = len(records)
    mean_prob = total_prob / max(1, count)
    mean_margin = total_margin / max(1, count)
    return count, mean_prob, mean_margin


def append_results_csv(csv_path: str, headers: List[str], row: List[str]) -> None:
    """Appends interpretability summary to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow(row)


def main() -> None:
    """CLI entrypoint for RumorDiffusion interpretability export."""
    parser = argparse.ArgumentParser(description="RumorDiffusion interpretability runner")
    parser.add_argument("--exp", type=str, required=True, help="Path to exp.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to model yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for RumorDiffusion")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to explain")
    parser.add_argument("--out", type=str, default="Output/explain/rumordiffusion_explain.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    exp_cfg = read_yaml_like(args.exp)
    model_cfg = read_yaml_like(args.model)

    device_cfg = exp_cfg.get("device", {})
    device_str = build_device(device_cfg)
    device = torch.device(device_str)
    if device_str == "cuda" and torch.cuda.is_available():
        gpu_id = int(device_cfg.get("gpu_id", 0))
        torch.cuda.set_device(gpu_id)

    dataset_cfg = exp_cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_root = os.path.join(dataset_cfg["root"], dataset_name)
    use_lp_edges = bool(dataset_cfg.get("use_lp_edges", False))
    prefix_cfg = dataset_cfg.get("prefix", {})
    observability_cfg = dataset_cfg.get("observability", {})
    ds = GraphDataset(
        root=dataset_root,
        split=args.split,
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
    )
    outputs_cfg = exp_cfg.get("outputs", {})
    results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))

    model = build_rumor_diffusion(model_cfg, exp_cfg, ds, device)
    load_checkpoint(model, args.ckpt, device)

    batch_size = int(exp_cfg.get("training", {}).get("batch_size", 4))
    loader = PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)),
        pin_memory=(device_str == "cuda"),
        persistent_workers=int(exp_cfg.get("training", {}).get("num_workers", 0)) > 0,
    )

    arb_cfg = model_cfg.get("arbiter", {})
    pair_ids = tuple(arb_cfg.get("pair_ids", [1, 2]))

    count, mean_prob, mean_margin = collect_explanations(model, loader, device, pair_ids, args.out)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_headers = [
        "timestamp",
        "dataset",
        "split",
        "model_cfg",
        "exp_cfg",
        "ckpt_path",
        "out_jsonl",
        "num_records",
        "mean_pred_prob",
        "mean_pair_margin",
        "pair_ids",
    ]
    summary_row = [
        ts,
        dataset_name,
        args.split,
        args.model,
        args.exp,
        args.ckpt,
        args.out,
        str(count),
        f"{mean_prob:.6f}",
        f"{mean_margin:.6f}",
        str(list(pair_ids)),
    ]
    summary_csv = os.path.join(results_dir, f"{dataset_name}_explain.csv")
    append_results_csv(summary_csv, summary_headers, summary_row)
    print(f"Saved interpretability records to {args.out}")


if __name__ == "__main__":
    main()
