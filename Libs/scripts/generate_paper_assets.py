#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


SELECTED_MODELS = [
    "textonly",
    "bigcn",
    "tgat",
    "lp_gcn",
    "dga_fake",
    "rumordiffusion",
]

MODEL_LABELS = {
    "textonly": "TextOnly (Sentence Emb.)",
    "bigcn": "BiGCN",
    "tgat": "TGAT",
    "lp_gcn": "LP-GCN (D$^2$)",
    "dga_fake": "DGA-Fake",
    "rumordiffusion": "RumorDiffusion (Ours)",
}


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float | None:
    cell = str(value).strip()
    if cell == "" or "±" in cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def mean_confidence(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    std_value = math.sqrt(max(variance, 0.0))
    ci95 = 1.96 * std_value / math.sqrt(float(len(values)))
    return float(mean_value), float(ci95)


def format_stat(values: Sequence[float]) -> str:
    mean_value, ci95 = mean_confidence(values)
    return f"{mean_value:.3f}$\\pm${ci95:.3f}"


def filter_seed_rows(
    rows: Iterable[Dict[str, str]],
    *,
    dataset: str,
    source_dataset: str,
    model: str,
    exp_cfg_suffix: str | None = None,
) -> List[Dict[str, str]]:
    filtered = []
    for row in rows:
        if row.get("dataset", "") != dataset:
            continue
        if row.get("source_dataset", "") != source_dataset:
            continue
        if row.get("model", "") != model:
            continue
        if row.get("seed", "") == "agg":
            continue
        if exp_cfg_suffix and not row.get("exp_cfg", "").endswith(exp_cfg_suffix):
            continue
        filtered.append(row)
    return filtered


def build_main_table(results_dir: Path, datasets: Sequence[str], caption: str, label: str) -> str:
    rows_by_dataset = {dataset: read_rows(results_dir / f"{dataset}.csv") for dataset in datasets}
    body_lines: List[str] = []
    for model in SELECTED_MODELS:
        values: List[str] = [MODEL_LABELS[model]]
        for dataset in datasets:
            seed_rows = filter_seed_rows(
                rows_by_dataset[dataset],
                dataset=dataset,
                source_dataset=dataset,
                model=model,
            )
            acc_values = [parse_float(row.get("test_acc", "")) for row in seed_rows]
            f1_values = [parse_float(row.get("test_macro_f1", "")) for row in seed_rows]
            acc_values = [value for value in acc_values if value is not None]
            f1_values = [value for value in f1_values if value is not None]
            values.append(format_stat(acc_values) if acc_values else "---")
            values.append(format_stat(f1_values) if f1_values else "---")
        body_lines.append(" & ".join(values) + r" \\")
    header = [
        r"\begin{table*}[!h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{l" + "cc" * len(datasets) + r"}",
        r"\toprule",
    ]
    if len(datasets) == 2:
        header.extend(
            [
                rf"& \multicolumn{{2}}{{c}}{{{datasets[0].capitalize()}}} & \multicolumn{{2}}{{c}}{{{datasets[1].capitalize()}}} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                r"Method & Acc & Macro-F1 & Acc & Macro-F1 \\",
                r"\midrule",
            ]
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(header + body_lines + footer) + "\n"


def build_transfer_table(results_dir: Path) -> str:
    rows = read_rows(results_dir / "transfer.csv")
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("seed", "") == "agg":
            continue
        key = (row.get("train_dataset", ""), row.get("test_dataset", ""), row.get("model", ""))
        grouped[key].append(row)
    body_lines: List[str] = []
    for (train_dataset, test_dataset, model_name), model_rows in sorted(grouped.items()):
        if model_name != "rumordiffusion":
            continue
        acc_values = [parse_float(row.get("acc", "")) for row in model_rows]
        f1_values = [parse_float(row.get("macro_f1", "")) for row in model_rows]
        ece_values = [parse_float(row.get("ece", "")) for row in model_rows]
        acc_values = [value for value in acc_values if value is not None]
        f1_values = [value for value in f1_values if value is not None]
        ece_values = [value for value in ece_values if value is not None]
        body_lines.append(
            " & ".join(
                [
                    f"{train_dataset}$\\rightarrow${test_dataset}",
                    format_stat(acc_values) if acc_values else "---",
                    format_stat(f1_values) if f1_values else "---",
                    format_stat(ece_values) if ece_values else "---",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[!h]",
            r"\centering",
            r"\caption{Zero-shot transfer results generated from the paper-ready canonical result tree.}",
            r"\label{tab:paper_ready_transfer}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Transfer pair & Acc & Macro-F1 & ECE \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_rebuttal_tradeoff_figure(results_dir: Path, figure_path: Path) -> None:
    revision_rows = read_rows(results_dir / "revision_artifacts.csv")
    sampling_rows = [
        row for row in revision_rows
        if row.get("artifact_type", "") == "sampling_sweep"
        and row.get("run_kind", "") == "transfer"
        and row.get("dataset", "") == "twitter16"
        and row.get("tag", "").startswith("early_10")
    ]
    threshold_rows = [
        row for row in revision_rows
        if row.get("artifact_type", "") == "threshold_sweep"
        and row.get("run_kind", "") == "transfer"
        and row.get("dataset", "") == "twitter16"
        and row.get("tag", "").startswith("early_10")
        and row.get("samples", "") == "2"
    ]
    if len(sampling_rows) == 0 and len(threshold_rows) == 0:
        raise FileNotFoundError("Missing transfer sampling/threshold sweep rows in revision_artifacts.csv")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if sampling_rows:
        grouped_sampling: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        for row in sampling_rows:
            grouped_sampling[int(float(row["samples"]))].append(row)
        sampling_points = []
        for samples, rows_for_sample in sorted(grouped_sampling.items()):
            latency_values = [parse_float(row.get("latency_mean_ms", "")) for row in rows_for_sample]
            acc_values = [parse_float(row.get("acc_resolved", "")) for row in rows_for_sample]
            latency_values = [value for value in latency_values if value is not None]
            acc_values = [value for value in acc_values if value is not None]
            if latency_values and acc_values:
                sampling_points.append((samples, sum(latency_values) / len(latency_values), sum(acc_values) / len(acc_values)))
        axes[0].plot([point[0] for point in sampling_points], [point[1] for point in sampling_points], marker="o", label="Latency (ms)")
        axes[0].set_xlabel("Sampling budget S")
        axes[0].set_ylabel("Mean latency (ms)")
        axes[0].grid(alpha=0.3)
        twin = axes[0].twinx()
        twin.plot([point[0] for point in sampling_points], [point[2] for point in sampling_points], marker="s", color="tab:orange", label="Resolved Acc")
        twin.set_ylabel("Resolved accuracy")
        axes[0].set_title("Sampling Budget Trade-off")

    if threshold_rows:
        grouped_threshold: Dict[Tuple[float, float, float], List[Dict[str, str]]] = defaultdict(list)
        for row in threshold_rows:
            key = (float(row["tau_low"]), float(row["tau_high"]), float(row["gamma"]))
            grouped_threshold[key].append(row)
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []
        for key, rows_for_key in sorted(grouped_threshold.items()):
            coverage_values = [parse_float(row.get("coverage", "")) for row in rows_for_key]
            acc_values = [parse_float(row.get("acc_resolved", "")) for row in rows_for_key]
            coverage_values = [value for value in coverage_values if value is not None]
            acc_values = [value for value in acc_values if value is not None]
            if not coverage_values or not acc_values:
                continue
            xs.append(sum(coverage_values) / len(coverage_values))
            ys.append(sum(acc_values) / len(acc_values))
            labels.append(f"({key[0]:.1f},{key[1]:.1f},{key[2]:.2f})")
        axes[1].scatter(xs, ys, color="tab:green")
        for x_value, y_value, label in zip(xs, ys, labels):
            axes[1].annotate(label, (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel("Coverage")
        axes[1].set_ylabel("Resolved accuracy")
        axes[1].set_title("Threshold Guidance")
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and figures from canonical RD results.")
    parser.add_argument("--results-dir", default="Output/results/paper_ready", help="Canonical result directory")
    parser.add_argument("--tables-dir", default="Docs/paper/Generated", help="Directory for generated LaTeX tables")
    parser.add_argument("--figures-dir", default="Docs/paper/Figs/generated", help="Directory for generated figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    twitter_table = build_main_table(
        results_dir=results_dir,
        datasets=["twitter15", "twitter16"],
        caption="Main Results for Twitter15/16 generated from the paper-ready canonical result tree.",
        label="tab:paper_ready_twitter_results",
    )
    binary_table = build_main_table(
        results_dir=results_dir,
        datasets=["weibo", "pheme"],
        caption="Main Results for Weibo/PHEME generated from the paper-ready canonical result tree.",
        label="tab:paper_ready_binary_results",
    )
    transfer_table = build_transfer_table(results_dir)

    write_text(tables_dir / "main_results_twitter.tex", twitter_table)
    write_text(tables_dir / "main_results_binary.tex", binary_table)
    write_text(tables_dir / "transfer_results.tex", transfer_table)
    build_rebuttal_tradeoff_figure(results_dir, figures_dir / "rebuttal_tradeoff.pdf")


if __name__ == "__main__":
    main()
