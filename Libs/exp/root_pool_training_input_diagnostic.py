"""Profile the frozen Weibo MAIN root representation on the complete TRAIN split.

This runner constructs the original train_arbiter GraphDataset and calls get()
for every TRAIN record. It does not instantiate a model or fit a policy. Source
imports are deferred until the CLI has verified its immutable source checkout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

import psutil
import torch


SOURCE_COMMIT = "a2441b4b4c09b04014c76c74df5ff86fe59cc309"
CONFIGURATION_SHA256 = "4be74f3857d68936fc70cdea6afd622c77c90bb88e21bee3c764ada33d03d98e"
MODEL_CONFIGURATION_SHA256 = "3d6808f10027c4a4813dcb650e9f2f2262a43fca7ad96753a03b84d24aef5e5b"
SEEDS = (41, 42, 43)
TRAIN_COUNT = 3729
CLASS_COUNTS = {"0": 1880, "1": 1849}
FEATURE_ORDER = (
    "norm_time", "depth", "in_degree", "out_degree", "parent_delta_norm",
    "time_speed_norm", "time_gradient_norm", "root_time_delta_norm",
)
INPUT_SHA256 = {
    "raw_manifest.json": "c50a6781a51feda1cb9685a1d4054c0b045f34a8b708775304ce1af9552955d2",
    "processed_manifest.json": "d591d48b3a015ddf57c3a50f25a534e156e1f8814f3f7b81f15083848b8bd564",
    "split_manifest.json": "1c46e077a46e51f68fcb0cbabd008277a7f53e6b8a0a829e7c33277756ff4cd7",
    "splits.json": "671450ae9d212fbe5c95476b04fc46209b13915377027b734f75c3bf341a7773",
    "label_schema.json": "560c71a20f118b3987061e27d59469e5157629a7e35268eb5add7ae0843594c1",
    "pyg/data.pt": "a17ccacf57bb6dfc3b2cec177a2eaba5909836c8ce41a97de85413bcf0d983b3",
    "pyg/slices.pt": "123b19a38cb335d324e08c3a43662f7234889b68a3a568aa2bcdbe82bd112743",
    "text_embs_mpnet_canonical_v1.pt": "1f93ab6a56b407e955750f60bdda0ca6214c76032a20e185d9809a1d4c131cdb",
}
REPORT_RELATIVE = Path("Output/exploratory/rumordiffusion_canonical_v1/reports/remaining_ordinary_baseline_completion/weibo_root_pool_training_input_diagnostic.json")


def read_json(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def file_identity(path: Path) -> dict[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input must be a regular, non-symlink file: {path}")
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/share/home/hwxu/Researches/D2Plus"))
    parser.add_argument("--source-root", type=Path, default=Path("/share/home/hwxu/Researches/D2Plus-worktrees/cgvae-augmentation-reuse-source"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / REPORT_RELATIVE
    if output.exists():
        raise FileExistsError(f"The diagnostic report already exists: {output}")
    if os.environ.get("SLURM_JOB_ID") != "665759" or socket.gethostname().split(".")[0] != "cpu04":
        raise ValueError("This execution requires existing allocation665759 on CPU04")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("This execution requires one CPU and an empty CUDA device list")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("The diagnostic requires one numerical-library thread")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized before this CPU diagnostic")
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed_commit != SOURCE_COMMIT:
        raise ValueError("The immutable source commit changed")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "Libs/data/dataset.py",
         "Libs/data/processor.py", "Libs/exp/run.py", "Libs/utils/artifact_schema.py"],
        cwd=args.source_root, check=True, capture_output=True,
    )
    sys.path.insert(0, str(args.source_root))
    from Libs.data.dataset import GraphDataset
    from Libs.exp.run import resolve_dataset_prefix_cfg
    from Libs.utils.artifact_schema import canonical_json_hash, omit_optional_none_fields

    control_path = args.project_root / "Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_execution_plan.json"
    control, control_sha = read_json(control_path)
    bindings = [b for b in control["case_evidence_bindings"] if (b["dataset"], b["model"]) == ("weibo", "root_pool")]
    if sorted(b["seed"] for b in bindings) != list(SEEDS):
        raise ValueError("Exactly the three frozen Weibo RootPooling bindings are required")
    for binding in bindings:
        if binding["training_code_commit"] != SOURCE_COMMIT:
            raise ValueError("A training source binding changed")
        if binding["configuration_sha256"] != CONFIGURATION_SHA256 or canonical_json_hash(omit_optional_none_fields(binding["configuration"])) != CONFIGURATION_SHA256:
            raise ValueError("A resolved configuration binding changed")
        if binding["model_configuration_sha256"] != MODEL_CONFIGURATION_SHA256 or canonical_json_hash(binding["model_configuration"]) != MODEL_CONFIGURATION_SHA256:
            raise ValueError("A model configuration binding changed")
    config = bindings[0]["configuration"]
    if config["experiment"]["seeds"] != list(SEEDS) or config["split"]["seed"] != 42 or config["experiment"]["bootstrap_seed"] != 42:
        raise ValueError("The split, bootstrap or experiment seeds changed")
    dataset_config = config["dataset"]
    prefix = resolve_dataset_prefix_cfg(dataset_config["prefix"], "weibo")
    if (prefix["strategy"], prefix["unit"], prefix["window"], prefix["minutes"], prefix["include_root"]) != ("hop", "hop", 3, 3, True):
        raise ValueError("The diagnostic requires the recorded MAIN3hop/include_root prefix")
    observability = dataset_config["observability"]
    if observability != {"drop_node": 0.0, "drop_edge": 0.0, "seed": 42}:
        raise ValueError("The recorded observability configuration changed")
    text_config = dataset_config["text_encoder"]
    if text_config["cache_name"] != "text_embs_mpnet_canonical_v1.pt" or not text_config["force_text_cache"] or text_config["rebuild_cache"] or not text_config["normalize_embeddings"]:
        raise ValueError("The frozen normalized text cache is required")

    execution_path = args.project_root / "Output/exploratory/manuscript_baseline_completion/weibo/root_pool/execution.json"
    execution, execution_sha = read_json(execution_path)
    if (execution["dataset"], execution["model"], execution["source_commit"], str(execution["job_id"])) != ("weibo", "root_pool", SOURCE_COMMIT, "665759") or execution["protocol_seeds"] != list(SEEDS):
        raise ValueError("The original input-staging execution receipt changed")
    authority = Path(dataset_config["root"]) / "weibo"
    physical = Path(execution["physical_input_root"]) / "weibo"
    if not physical.is_absolute() or physical.is_symlink() or not physical.is_dir() or physical == authority:
        raise ValueError("The original node-local staged Weibo input is unavailable")
    pins = {}
    manifests = {}
    bulk_before = {}
    for name, expected in INPUT_SHA256.items():
        record = execution["input_files"][name]
        if record["path"] != str(authority / name) or record["sha256"] != expected:
            raise ValueError(f"The durable input pin changed: {name}")
        actual_path = physical / name
        if name.endswith(".json"):
            payload, current_sha = read_json(actual_path)
            if current_sha != expected:
                raise ValueError(f"The staged manifest conflicts with its durable pin: {name}")
            manifests[name] = payload
            verification = "small_staged_manifest_sha256_recomputed"
        else:
            bulk_before[name] = file_identity(actual_path)
            if bulk_before[name]["size_bytes"] != file_identity(authority / name)["size_bytes"]:
                raise ValueError(f"The staged bulk input size changed: {name}")
            verification = "native_GraphDataset_constructor_sha256_once" if name in ("pyg/data.pt", "pyg/slices.pt") else "qualified_execution_sha256_reused_without_rehash"
        pins[name] = {"path": record["path"], "physical_read_path": str(actual_path),
                      "sha256": expected, "verification": verification}
    raw = manifests["raw_manifest.json"]
    processed = manifests["processed_manifest.json"]
    split = manifests["split_manifest.json"]
    schema = manifests["label_schema.json"]
    if any(m["dataset"] != "weibo" for m in (raw, processed, split, schema)):
        raise ValueError("A manifest names another dataset")
    if processed["raw_manifest_sha256"] != INPUT_SHA256["raw_manifest.json"] or split["raw_manifest_sha256"] != INPUT_SHA256["raw_manifest.json"] or split["processed_manifest_sha256"] != INPUT_SHA256["processed_manifest.json"] or split["splits_sha256"] != INPUT_SHA256["splits.json"]:
        raise ValueError("The small-manifest provenance chain conflicts")
    if canonical_json_hash(schema) != processed["label_schema_sha256"] or raw["label_schema_sha256"] != processed["label_schema_sha256"]:
        raise ValueError("The canonical label-schema payload hash conflicts")
    if processed["data_sha256"] != INPUT_SHA256["pyg/data.pt"] or processed["slices_sha256"] != INPUT_SHA256["pyg/slices.pt"] or processed["time_unit"] != "hop" or split["split_seed"] != 42:
        raise ValueError("The stored tensor, time-unit or split-seed binding changed")
    train_indices = manifests["splits.json"]["train"]
    train_manifest = split["splits"]["train"]
    train_ids = train_manifest["source_ids"]
    if len(train_indices) != TRAIN_COUNT or len(set(train_indices)) != TRAIN_COUNT or any(type(i) is not int for i in train_indices) or len(train_ids) != TRAIN_COUNT or len(set(train_ids)) != TRAIN_COUNT:
        raise ValueError("The complete immutable TRAIN identity set is required")
    if train_manifest["sample_count"] != TRAIN_COUNT or train_manifest["class_distribution"] != CLASS_COUNTS or canonical_json_hash(train_ids) != train_manifest["source_id_sha256"]:
        raise ValueError("The TRAIN manifest count, classes or source-ID hash changed")

    dataset = GraphDataset(
        root=str(physical), split="train", mode="train_arbiter",
        use_lp_edges=False, use_potentials=False, prefix_cfg=prefix,
        observability_cfg=observability, text_cache_name=text_config["cache_name"],
        force_text_cache=text_config["force_text_cache"],
        text_normalize=text_config["normalize_embeddings"],
    )
    if len(dataset) != TRAIN_COUNT or list(dataset._indices) != train_indices or not dataset._text_cache_loaded:
        raise ValueError("The real TRAIN view or cached text load differs from its binding")
    rows = []
    groups = defaultdict(list)
    labels = Counter()
    for local_index in range(TRAIN_COUNT):
        graph = dataset.get(local_index)
        source_id = str(graph.source_id)
        if source_id != str(train_ids[local_index]):
            raise ValueError(f"TRAIN source identity changed at local index{local_index}")
        if graph.x.dim() != 2 or graph.x.size(1) != len(FEATURE_ORDER) or not torch.isfinite(graph.x).all():
            raise ValueError(f"TRAIN features are non-finite or have the wrong dimension: {source_id}")
        if graph.root_index.numel() != 1 or graph.y.numel() != 1 or graph.y.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Invalid TRAIN root or integer label: {source_id}")
        root_index = int(graph.root_index.item())
        label = int(graph.y.item())
        if not 0 <= root_index < graph.x.size(0) or str(label) not in CLASS_COUNTS:
            raise ValueError(f"TRAIN root or class is outside its schema: {source_id}")
        vector = graph.x[root_index].detach().tolist()
        row = {"train_local_index": local_index, "raw_index": train_indices[local_index],
               "source_id": source_id, "label": label, "root_index": root_index,
               "root_time_hops_shifted": int(graph.node_time_hops_shifted[root_index].item()),
               "root_features": vector}
        rows.append(row)
        groups[tuple(vector)].append(row)
        labels[str(label)] += 1
    if dict(sorted(labels.items())) != CLASS_COUNTS or len(rows) != TRAIN_COUNT:
        raise ValueError("The measured TRAIN rows and class counts do not reconcile")
    matrix = torch.tensor([row["root_features"] for row in rows], dtype=torch.float64)
    statistics = []
    for column, feature in enumerate(FEATURE_ORDER):
        values = matrix[:, column]
        unique_count = int(torch.unique(values).numel())
        statistics.append({"feature": feature, "finite_count": int(torch.isfinite(values).sum()),
                           "minimum": float(values.min()), "maximum": float(values.max()),
                           "mean": float(values.mean()), "population_variance": float(values.var(unbiased=False)),
                           "population_standard_deviation": float(values.std(unbiased=False)),
                           "unique_value_count": unique_count, "nonzero_count": int(torch.count_nonzero(values)),
                           "constant": unique_count == 1})
    vector_groups = []
    for vector, members in sorted(groups.items()):
        counts = dict(sorted(Counter(str(row["label"]) for row in members).items()))
        vector_groups.append({"root_features": list(vector), "sample_count": len(members),
                              "class_counts": counts, "shared_by_multiple_classes": len(counts) > 1,
                              "train_local_indices": [row["train_local_index"] for row in members]})
    cross_class_groups = [group for group in vector_groups if group["shared_by_multiple_classes"]]
    cross_class_count = sum(group["sample_count"] for group in cross_class_groups)
    if sum(group["sample_count"] for group in vector_groups) != TRAIN_COUNT or sum(sum(group["class_counts"].values()) for group in vector_groups) != TRAIN_COUNT:
        raise ValueError("The full-vector and class-group summaries do not reconcile")
    if any(file_identity(physical / name) != identity for name, identity in bulk_before.items()):
        raise ValueError("A staged bulk input changed during the diagnostic")
    report = {
        "schema_version": "1", "status": "TRAIN_ROOT_INPUT_DIAGNOSTIC_COMPLETE",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "weibo", "model": "root_pool", "split": "train", "mode": "train_arbiter",
        "training_code_commit": SOURCE_COMMIT, "configuration_binding_sha256": CONFIGURATION_SHA256,
        "configuration_bindings": [{"seed": b["seed"], "configuration_sha256": b["configuration_sha256"],
                                    "model_configuration_sha256": b["model_configuration_sha256"]} for b in sorted(bindings, key=lambda b: b["seed"])],
        "control_snapshot": {"path": str(control_path), "sha256": control_sha},
        "prefix_configuration": prefix, "prefix_resolver": "Libs.exp.run.resolve_dataset_prefix_cfg",
        "observability_configuration": observability, "text_cache_configuration": text_config,
        "text_normalization_configuration": dataset_config["text_normalization"],
        "randomized_experiment": False, "split_seed": 42, "bootstrap_seed": 42,
        "protocol_experiment_seeds": list(SEEDS), "sample_count": TRAIN_COUNT,
        "class_counts": dict(sorted(labels.items())), "feature_order": list(FEATURE_ORDER),
        "feature_statistics": statistics, "distinct_root_vectors": len(vector_groups),
        "root_vectors_shared_by_multiple_classes": len(cross_class_groups),
        "training_examples_in_cross_class_root_groups": cross_class_count,
        "root_vector_groups": vector_groups, "training_rows": rows, "input_pins": pins,
        "input_source": {"physical_input_root": str(physical), "durable_input_root": str(authority),
                         "execution_receipt": {"path": str(execution_path), "sha256": execution_sha},
                         "native_constructor_bulk_checksum_files": ["pyg/data.pt", "pyg/slices.pt"],
                         "native_constructor_bulk_checks_per_file": 1,
                         "text_cache_bulk_sha256_recomputed": False, "encoder_instantiated": False},
        "model_input_semantics": {"root_structural_features_only": True, "graph_message_passing": False,
                                  "text_embedding_used_by_classifier": False,
                                  "structural_features_recomputed_from_observed_prefix": True,
                                  "feature_time_scale": "current observed-prefix maximum shifted time; nonpositive values floored to1"},
        "model_calls": 0, "model_instances": 0, "optimizer_updates": 0,
        "policy_fitting_or_replay": False, "validation_or_test_views_constructed": False,
        "shared_collated_storage_loaded_for_train_split": True,
        "label_index_source_id_statistics_scope": "TRAIN only",
        "performance_metrics_computed": False, "main_or_early_cases_added": 0,
        "scientific_boundary": "Descriptive full TRAIN input diagnosis at MAIN3hop. Input variation and cross-class identical vectors do not establish the cause of TEST errors or qualify comparative performance. No model, optimizer, policy or held-out view is executed. Accepted evidence and manuscript values are unchanged.",
        "compute": {"node": socket.gethostname(), "allocation_job_id": os.environ["SLURM_JOB_ID"],
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "cpus_in_affinity": len(os.sched_getaffinity(0)),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_initialized": torch.cuda.is_initialized(),
                    "numerical_library_threads": 1, "python_executable": sys.executable,
                    "elapsed_seconds": time.monotonic() - start_clock,
                    "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "node_available_memory_bytes": psutil.virtual_memory().available,
                    "node_local_disk_free_bytes": psutil.disk_usage(str(physical)).free},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "project_complete": False,
    }
    content = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    observed = output.read_bytes()
    if observed != content or json.loads(observed) != report:
        raise ValueError("The exclusive diagnostic report failed exact metadata readback")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": hashlib.sha256(observed).hexdigest(),
                      "sample_count": TRAIN_COUNT, "class_counts": report["class_counts"],
                      "distinct_root_vectors": len(vector_groups), "cross_class_root_groups": len(cross_class_groups),
                      "training_examples_in_cross_class_root_groups": cross_class_count,
                      "constant_features": [s["feature"] for s in statistics if s["constant"]],
                      "compute": report["compute"], "exact_metadata_readback": True}, sort_keys=True))
    del dataset, graph, matrix


if __name__ == "__main__":
    main()
