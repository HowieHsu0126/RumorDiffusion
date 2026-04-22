import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"


@dataclass
class DatasetConfig:
    """Holds dataset processing configuration loaded from YAML-equivalent JSON.

    Attributes:
        datasets: Names of datasets to process, e.g., ["twitter15", "twitter16"].
        raw_root: Absolute path to raw datasets directory.
        processed_root: Absolute path to save processed outputs.
        split_type: Strategy for split (currently only random).
        split_ratios: Fractions for train/val/test splits.
        split_seed: Random seed to make splits reproducible.
        label_allowed: Labels allowed to include.
        label_map: Mapping from string label to integer class id.
        label_skip_unknown: If true, skip samples with unknown labels.
        cutoff_minutes: Early truncation time window in minutes.
        enforce_root_component: Keep only the connected component containing the root.
        features_structural: List of structural feature names in order.
        augment_enabled: Whether to enable augmentation.
        augment_method: Augmentation method name.
        saving_format: Output saving format identifier.
        text_encoder_path: Optional path to a sentence transformer checkpoint.
        text_normalize: Whether to L2-normalize loaded text embeddings.
        text_noise_std: Std of Gaussian noise added to text embeddings when augmentation is enabled.
        text_dropout: Feature dropout probability applied to text embeddings when augmentation is enabled.
    """

    datasets: List[str]
    raw_root: str
    processed_root: str
    split_type: str
    split_ratios: Dict[str, float]
    split_seed: int
    label_allowed: List[str]
    label_map: Dict[str, int]
    label_skip_unknown: bool
    label_overrides: Dict[str, Dict[str, object]]
    cutoff_minutes: int
    enforce_root_component: bool
    features_structural: List[str]
    augment_enabled: bool
    augment_method: str
    text_encoder_path: Optional[str]
    text_normalize: bool
    text_cache_name: str
    text_batch_size: int
    text_normalization: Dict[str, object]
    text_noise_std: float
    text_dropout: float
    saving_format: str
    time_source_default: str
    time_source_overrides: Dict[str, str]


def load_yaml(config_path: str) -> Dict:
    """Loads full YAML using yaml.safe_load for robust parsing."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> DatasetConfig:
    cfg = load_yaml(config_path)["dataset"]
    # Validate presence
    required_keys = [
        "datasets",
        "raw_root",
        "processed_root",
        "split",
        "labels",
        "early",
        "features",
        "augment",
        "saving",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing config section: {k}")

    split = cfg["split"]
    labels = cfg["labels"]
    early = cfg["early"]
    features = cfg["features"]
    augment = cfg["augment"]
    saving = cfg["saving"]
    text_cfg = cfg.get("text", {})
    time_cfg = cfg.get("time_source", {})

    return DatasetConfig(
        datasets=list(cfg["datasets"]),
        raw_root=str(cfg["raw_root"]),
        processed_root=str(cfg["processed_root"]),
        split_type=str(split["type"]),
        split_ratios={
            "train": float(split["ratios"]["train"]),
            "val": float(split["ratios"]["val"]),
            "test": float(split["ratios"]["test"]),
        },
        split_seed=int(split["seed"]),
        label_allowed=list(labels["allowed"]),
        label_map={str(k): int(v) for k, v in labels["map"].items()},
        label_skip_unknown=bool(labels["skip_unknown"]),
        label_overrides={str(k): dict(v) for k, v in labels.get("overrides", {}).items()},
        cutoff_minutes=int(early["cutoff_minutes"]),
        enforce_root_component=bool(early["enforce_root_component"]),
        features_structural=list(features["structural"]),
        augment_enabled=bool(augment["enabled"]),
        augment_method=str(augment["method"]),
        text_encoder_path=text_cfg.get("encoder_path"),
        text_normalize=bool(text_cfg.get("normalize", False)),
        text_cache_name=str(text_cfg.get("cache_name", "text_embs.pt")),
        text_batch_size=int(text_cfg.get("batch_size", 64)),
        text_normalization=dict(text_cfg.get("normalization", {})),
        text_noise_std=float(augment.get("text_noise_std", 0.0)),
        text_dropout=float(augment.get("text_dropout", 0.0)),
        saving_format=str(saving["format"]),
        time_source_default=str(time_cfg.get("default", "minutes")),
        time_source_overrides={str(k): str(v) for k, v in time_cfg.get("overrides", {}).items()},
    )


@dataclass
class LabelConfig:
    allowed: List[str]
    mapping: Dict[str, int]
    skip_unknown: bool
    numeric: bool


def resolve_label_config(dataset: str, cfg: DatasetConfig) -> LabelConfig:
    """Returns dataset-specific label mapping configuration."""
    override = cfg.label_overrides.get(dataset, {})
    allowed = list(override.get("allowed", cfg.label_allowed))
    mapping = {str(k): int(v) for k, v in override.get("map", cfg.label_map).items()}
    skip_unknown = bool(override.get("skip_unknown", cfg.label_skip_unknown))
    numeric = bool(override.get("numeric", False))
    return LabelConfig(allowed=allowed, mapping=mapping, skip_unknown=skip_unknown, numeric=numeric)


def resolve_time_source(dataset: str, cfg: DatasetConfig) -> str:
    """Returns dataset-specific time source selection."""
    return cfg.time_source_overrides.get(dataset, cfg.time_source_default)


def normalize_tweet_text(text: str, normalization_cfg: Optional[Dict[str, object]] = None) -> str:
    """Normalizes tweet text using configurable, non-destructive substitutions."""
    cfg = normalization_cfg if isinstance(normalization_cfg, dict) else {}
    normalized = str(text)
    if bool(cfg.get("lowercase", False)):
        normalized = normalized.lower()
    if bool(cfg.get("replace_url", False)):
        normalized = re.sub(r"https?://\S+|www\.\S+", " <url> ", normalized)
    if bool(cfg.get("replace_mention", False)):
        normalized = re.sub(r"@\w+", " <user> ", normalized)
    if bool(cfg.get("replace_number", False)):
        normalized = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " <num> ", normalized)
    if bool(cfg.get("keep_hashtag_text", False)):
        normalized = re.sub(r"#([A-Za-z0-9_]+)", r" \1 ", normalized)
    if bool(cfg.get("compress_repeated_chars", False)):
        normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def resolve_text_model_path(model_name_or_path: str, raw_root: str) -> str:
    """Resolves a configured model path, preferring explicit local paths when present."""
    model_text = str(model_name_or_path).strip()
    if len(model_text) == 0:
        return os.path.join(os.path.dirname(raw_root), "sentence_transformers_all_MiniLM_L6v2")
    if os.path.isdir(model_text):
        return model_text
    local_candidate = os.path.join(os.path.dirname(raw_root), model_text)
    if os.path.isdir(local_candidate):
        return local_candidate
    return model_text


def parse_source_texts(raw_dir: str) -> Dict[str, str]:
    """Parses `source_tweets.txt` into a mapping from source_id to raw text.

    The file format is tab-separated: `<source_id>\t<text>` per line.
    """
    path = os.path.join(raw_dir, "source_tweets.txt")
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            assert len(parts) == 2, f"Bad line without TAB: {line}"
            sid = parts[0].strip()
            text = parts[1].strip()
            assert len(sid) > 0, "Empty source id"
            mapping[sid] = text
    return mapping


def collect_json_source_texts(raw_dir: str) -> Dict[str, str]:
    """Collects source texts from JSON-format datasets (e.g., weibo, pheme)."""
    mapping: Dict[str, str] = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(raw_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        source = obj.get("source", {})
        if not isinstance(source, dict):
            continue
        sid = (
            source.get("tweet id")
            or source.get("tweet_id")
            or source.get("source id")
            or source.get("source_id")
            or source.get("id")
        )
        assert sid is not None, f"Missing source id in {fpath}"
        text = str(source.get("content", "")).strip()
        if len(text) == 0:
            continue
        mapping[str(sid)] = text
    assert len(mapping) > 0, f"No JSON sources found in {raw_dir}"
    return mapping


def build_text_embedding_cache(dataset: str,
                               raw_root: str,
                               processed_root: str,
                               model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                               batch_size: int = 64,
                               normalization_cfg: Optional[Dict[str, object]] = None,
                               cache_name: str = "text_embs.pt") -> str:
    """Builds and saves `{source_id -> text_emb}` cache to processed dir.

    The output file path is `{processed_root}/{dataset}/text_embs.pt` and contains:
      {"model": str, "dim": int, "ids": List[str], "emb": FloatTensor[N, D]}
    """
    raw_dir = os.path.join(raw_root, dataset)
    out_dir = os.path.join(processed_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, str(cache_name))

    # Parse all available source texts
    sid_to_text = parse_source_texts(raw_dir)
    ids = sorted(sid_to_text.keys())
    texts = [normalize_tweet_text(sid_to_text[sid], normalization_cfg) for sid in ids]

    # Encode texts using sentence-transformers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_text_model_path(model_name, raw_root)
    print(f"Loading model from: {model_path}")
    
    # Try to load with transformers directly first
    try:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        
        # Create a simple encoder function
        def encode_texts(texts, batch_size=64):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(embeddings_batch.cpu())
            return torch.cat(embeddings, dim=0)
        
        # Use the custom encoder
        emb = encode_texts(texts, batch_size=int(batch_size))
        
    except Exception as e:
        print(f"Failed to load with transformers: {e}")
        # Fallback to sentence-transformers
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_path, device=device)
        emb = encoder.encode(texts, batch_size=int(batch_size), convert_to_tensor=True, normalize_embeddings=False)
        emb = emb.to(torch.float32).cpu()
    
    # Ensure float32
    emb = emb.to(torch.float32)

    obj = {
        "model": str(model_name),
        "dim": int(emb.size(-1)),
        "ids": ids,
        "emb": emb,
    }
    torch.save(obj, out_path)
    return out_path


def _encode_texts_sentence_transformer(texts: List[str],
                                       model_path: str,
                                       batch_size: int = 64) -> Tensor:
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_path, device=device)
    emb = encoder.encode(
        texts,
        batch_size=int(batch_size),
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return emb.to(torch.float32).cpu()


def _encode_texts_transformer_mean_pool(texts: List[str],
                                        model_path: str,
                                        batch_size: int = 64) -> Tensor:
    """Encodes texts using a local sentence-transformer, with HF mean-pool fallback."""
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    if os.path.isfile(os.path.join(model_path, "modules.json")):
        return _encode_texts_sentence_transformer(texts, model_path=model_path, batch_size=batch_size)

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content in ("", "{}"):
            os.remove(adapter_cfg_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    all_emb: List[Tensor] = []
    for start in range(0, len(texts), int(batch_size)):
        batch_texts = texts[start:start + int(batch_size)]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(token_embeddings)
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            pooled = summed / counts
        all_emb.append(pooled.cpu())
    return torch.cat(all_emb, dim=0).to(torch.float32)


def build_text_embedding_cache_from_mapping(dataset: str,
                                            processed_root: str,
                                            mapping: Dict[str, str],
                                            model_path: str,
                                            batch_size: int = 64,
                                            normalization_cfg: Optional[Dict[str, object]] = None,
                                            cache_name: str = "text_embs.pt") -> str:
    """Builds text embedding cache from provided id->text mapping."""
    out_dir = os.path.join(processed_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, str(cache_name))
    ids = sorted(mapping.keys())
    texts = [normalize_tweet_text(mapping[i], normalization_cfg) for i in ids]
    emb = _encode_texts_transformer_mean_pool(texts, model_path=model_path, batch_size=batch_size)
    obj = {
        "model": str(model_path),
        "dim": int(emb.size(-1)),
        "ids": ids,
        "emb": emb,
    }
    torch.save(obj, out_path)
    return out_path


def _load_text_embedding_cache(processed_root: str, dataset: str, cache_name: str = "text_embs.pt") -> Tuple[Dict[str, int], Tensor]:
    """Loads text embedding cache and returns (id_to_index, emb_matrix)."""
    path = os.path.join(processed_root, dataset, str(cache_name))
    obj = torch.load(path, weights_only=False)
    ids: List[str] = list(obj["ids"])  # type: ignore[index]
    emb: Tensor = obj["emb"].to(torch.float32)  # type: ignore[index]
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    return id_to_idx, emb

def parse_node(token: str) -> Tuple[str, str, int]:
    """Parses a node token like ["uid","tweet","delay"].

    Returns:
        (uid, tweet_id, delay_minutes)
    """
    s = token.strip()
    assert s.startswith("[") and s.endswith("]"), f"Bad node token: {token}"
    inner = s[1:-1].strip()
    parts = [p.strip() for p in inner.split(",")]
    assert len(parts) == 3, f"Node parts != 3: {token}"

    def strip_quotes(x: str) -> str:
        x = x.strip()
        if x.startswith("'") and x.endswith("'"):
            return x[1:-1]
        if x.startswith('"') and x.endswith('"'):
            return x[1:-1]
        return x

    uid = strip_quotes(parts[0])
    tweet_id = strip_quotes(parts[1])
    delay_raw = strip_quotes(parts[2])
    # allow optional leading '-' and decimal digits
    assert re.match(r"^-?\d+(?:\.\d+)?$",
                    delay_raw) is not None, f"Delay not numeric: {delay_raw}"
    delay_minutes = int(round(float(delay_raw)))
    return uid, tweet_id, delay_minutes


def parse_tree_file(path: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[int, int]]]:
    """Parses one tree file into node list and edge index pairs.

    Returns:
        nodes: list of (uid, tweet_id, delay_minutes) in order of first appearance
        edges: list of (src_idx, dst_idx)
    """
    nodes: List[Tuple[str, str, int]] = []
    node_to_idx: Dict[Tuple[str, str, int], int] = {}
    edges: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "->" not in line:
                raise ValueError(f"Edge line missing '->': {line}")
            lhs, rhs = line.split("->", 1)
            u = parse_node(lhs.strip())
            v = parse_node(rhs.strip())

            if u not in node_to_idx:
                node_to_idx[u] = len(nodes)
                nodes.append(u)
            if v not in node_to_idx:
                node_to_idx[v] = len(nodes)
                nodes.append(v)
            edges.append((node_to_idx[u], node_to_idx[v]))

    return nodes, edges


def truncate_early(nodes: List[Tuple[str, str, int]],
                   edges: List[Tuple[int, int]],
                   cutoff_minutes: int,
                   enforce_root_component: bool) -> Tuple[List[int], List[Tuple[int, int]], int, List[int]]:
    """Applies time-based truncation and optional root-component filtering.

    Returns:
        keep_nodes: indices of nodes kept after truncation and filtering
        keep_edges: edges reindexed within kept nodes
        root_index_new: index of root node in the reindexed space
    """
    # shift times so that the earliest time becomes 0
    original_times = torch.tensor([t for (_, _, t) in nodes], dtype=torch.long)
    min_time = int(original_times.min().item())
    shifted_times = original_times - min_time
    keep_mask = shifted_times <= cutoff_minutes
    keep_nodes = [i for i, k in enumerate(keep_mask.tolist()) if k]
    if not keep_nodes:
        raise ValueError("No nodes kept after truncation")

    # Identify root as node with minimal time (prefer time==0)
    min_kept_time = int(shifted_times[keep_mask].min().item())
    candidate_indices = [i for i in keep_nodes if shifted_times[i].item() == 0]
    if candidate_indices:
        root_idx = min(candidate_indices)
    else:
        # fallback to earliest time among kept nodes
        root_idx = min((i for i in keep_nodes),
                       key=lambda x: shifted_times[x].item())

    # Filter edges by kept nodes
    kept_set = set(keep_nodes)
    filtered_edges = [(u, v)
                      for (u, v) in edges if u in kept_set and v in kept_set]

    if enforce_root_component:
        # BFS/DFS from root within filtered edges
        adj: Dict[int, List[int]] = {}
        for u, v in filtered_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        stack = [root_idx]
        visited = set([root_idx])
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        keep_nodes = sorted(list(visited))
        kept_set = set(keep_nodes)
        filtered_edges = [(u, v) for (
            u, v) in filtered_edges if u in kept_set and v in kept_set]

    # Reindex nodes to 0..N-1
    old_to_new = {old: new for new, old in enumerate(keep_nodes)}
    reindexed_edges = [(old_to_new[u], old_to_new[v])
                       for (u, v) in filtered_edges]
    root_index_new = old_to_new[root_idx]
    # provide shifted times for feature computation
    shifted_times_list = [int(shifted_times[i].item()) for i in keep_nodes]
    return keep_nodes, reindexed_edges, root_index_new, shifted_times_list


def compute_structural_features(nodes: List[Tuple[str, str, int]],
                                edges: List[Tuple[int, int]],
                                keep_nodes: List[int],
                                reindexed_edges: List[Tuple[int, int]],
                                root_index_new: int,
                                cutoff_minutes: int,
                                shifted_times_list: List[int]) -> Tensor:
    """Computes structural node features.

    Returns features in the order:
    [norm_time, depth, in_degree, out_degree, parent_delta_norm, time_speed_norm, 
     time_gradient_norm, root_time_delta_norm].
    """
    num_nodes = len(keep_nodes)
    times = torch.tensor(shifted_times_list, dtype=torch.float32)
    norm_time = times / float(max(1, cutoff_minutes))
    norm_time = torch.clamp(norm_time, 0.0, 1.0)

    # Build undirected adjacency for BFS depth
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in reindexed_edges:
        adj[u].append(v)
        adj[v].append(u)

    depth = torch.full((num_nodes,), float("inf"))
    depth[root_index_new] = 0.0
    queue = [root_index_new]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nb in adj.get(cur, []):
            if math.isinf(depth[nb].item()):
                depth[nb] = depth[cur] + 1.0
                queue.append(nb)
    # disconnected should not happen if filtered
    depth[torch.isinf(depth)] = float(num_nodes)

    in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    for u, v in reindexed_edges:
        out_degree[u] += 1.0
        in_degree[v] += 1.0

    # Parent-child time delta (minutes), normalized by cutoff_minutes.
    # For nodes without an incoming edge, the delta is 0.
    parents: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in reindexed_edges:
        parents[v].append(u)

    parent_delta = torch.zeros(num_nodes, dtype=torch.float32)
    for v in range(num_nodes):
        if parents[v]:
            p_idx = torch.tensor(parents[v], dtype=torch.long)
            p_times = times[p_idx]
            delta = times[v] - torch.min(p_times)
            parent_delta[v] = torch.clamp(delta, min=0.0)
        else:
            parent_delta[v] = 0.0
    parent_delta_norm = torch.clamp(parent_delta / float(max(1, cutoff_minutes)), 0.0, 1.0)

    # Time speed: arrival time relative to depth (captures propagation velocity)
    # Nodes with high time but low depth have fast propagation
    time_speed = torch.where(
        depth > 0,
        times / (depth + 1.0),
        times
    )
    time_speed_norm = torch.clamp(time_speed / float(max(1, cutoff_minutes)), 0.0, 1.0)

    # Time gradient: average time difference with neighbors
    time_gradient = torch.zeros(num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        neighbors = adj.get(i, [])
        if neighbors:
            neighbor_times = times[torch.tensor(neighbors, dtype=torch.long)]
            time_gradient[i] = torch.abs(times[i] - torch.mean(neighbor_times))
        else:
            time_gradient[i] = 0.0
    time_gradient_norm = torch.clamp(time_gradient / float(max(1, cutoff_minutes)), 0.0, 1.0)

    # Root time delta: direct time difference with root node
    root_time = times[root_index_new]
    root_time_delta = torch.abs(times - root_time)
    root_time_delta_norm = torch.clamp(root_time_delta / float(max(1, cutoff_minutes)), 0.0, 1.0)

    x = torch.stack([
        norm_time, depth, in_degree, out_degree, parent_delta_norm,
        time_speed_norm, time_gradient_norm, root_time_delta_norm
    ], dim=1)
    return x


def parse_json_tree(obj: Dict[str, Any], time_source: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[int, int]]]:
    """Parses JSON-format cascades with hop-based timing."""
    assert "source" in obj, "source field missing in JSON object"
    source = obj["source"]
    comments = obj.get("comment", [])

    source_uid = str(source.get("user id", source.get("user_id", "")))
    source_tid = (
        source.get("tweet id")
        or source.get("tweet_id")
        or source.get("source id")
        or source.get("source_id")
        or source.get("id")
    )
    assert source_tid is not None, "Source tweet id missing"
    nodes: List[Tuple[str, str, int]] = [(source_uid, str(source_tid), 0)]

    comment_id_to_idx: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []

    for comment in comments:
        cid_raw = comment.get("comment id", comment.get("comment_id"))
        assert cid_raw is not None, "comment id missing"
        cid = int(cid_raw)
        delay = int(comment.get("hop", 1)) if time_source == "hop" else int(comment.get("time", 1))
        user = str(comment.get("user id", comment.get("user_id", "")))
        idx = len(nodes)
        nodes.append((user, str(cid), delay))
        comment_id_to_idx[cid] = idx

    for comment in comments:
        cid_raw = comment.get("comment id", comment.get("comment_id"))
        cid = int(cid_raw)
        parent_raw = comment.get("parent", -1)
        parent_val = int(parent_raw) if parent_raw is not None else -1
        parent_idx = 0 if parent_val == -1 else comment_id_to_idx[parent_val]
        child_idx = comment_id_to_idx[cid]
        edges.append((parent_idx, child_idx))

    return nodes, edges


def build_pyg_data(nodes: List[Tuple[str, str, int]],
                   edges: List[Tuple[int, int]],
                   keep_nodes: List[int],
                   reindexed_edges: List[Tuple[int, int]],
                   root_index_new: int,
                   cutoff_minutes: int,
                   shifted_times_list: List[int],
                   label_id: int,
                   source_id: str) -> Data:
    num_nodes = len(keep_nodes)
    if num_nodes == 0:
        raise ValueError("No nodes after filtering")

    x = compute_structural_features(
        nodes,
        edges,
        keep_nodes,
        reindexed_edges,
        root_index_new,
        cutoff_minutes,
        shifted_times_list,
    )
    edge_index = torch.tensor(
        reindexed_edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label_id], dtype=torch.long)

    node_uid = [nodes[i][0] for i in keep_nodes]
    node_tid = [nodes[i][1] for i in keep_nodes]
    node_time = torch.tensor([nodes[i][2]
                             for i in keep_nodes], dtype=torch.long)
    node_time_shifted = torch.tensor(shifted_times_list, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        root_index=torch.tensor([root_index_new], dtype=torch.long),
    )
    # Attach metadata as plain attributes (PyG tolerant)
    data.source_id = source_id
    data.node_uid = node_uid
    data.node_tweet_id = node_tid
    data.node_time_minutes = node_time
    data.node_time_minutes_shifted = node_time_shifted
    return data


def read_labels(label_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Bad label line: {line}")
            label, sid = line.split(":", 1)
            mapping[sid.strip()] = label.strip()
    return mapping


def generate_splits(num_samples: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    assert abs(sum(ratios.values()) -
               1.0) < 1e-6, "Split ratios must sum to 1.0"
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_samples, generator=g).tolist()
    n_train = int(num_samples * ratios["train"])
    n_val = int(num_samples * ratios["val"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def generate_stratified_splits(labels: List[int], ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    assert len(labels) > 0, "Labels must be non-empty for stratified split"
    cls_to_indices: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        y_int = int(y)
        cls_to_indices.setdefault(y_int, []).append(idx)

    g = torch.Generator()
    g.manual_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls, idx_list in cls_to_indices.items():
        order = torch.randperm(len(idx_list), generator=g).tolist()
        shuffled = [idx_list[i] for i in order]
        n = len(shuffled)
        n_train = int(math.floor(n * ratios["train"]))
        n_val = int(math.floor(n * ratios["val"]))
        # Ensure minimal coverage when samples exist
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_val = min(n_val, max(0, n - n_train))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            if n_val > 0 and n_train + n_val > n:
                n_val = max(0, n - n_train)
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    def _shuffle(arr: List[int]) -> List[int]:
        if len(arr) == 0:
            return arr
        order = torch.randperm(len(arr), generator=g).tolist()
        return [arr[i] for i in order]

    return {
        "train": _shuffle(train_idx),
        "val": _shuffle(val_idx),
        "test": _shuffle(test_idx),
    }


def generate_splits(labels: List[int], ratios: Dict[str, float], seed: int, strategy: str) -> Dict[str, List[int]]:
    strategy_norm = str(strategy).lower()
    if strategy_norm == "stratified":
        return generate_stratified_splits(labels, ratios, seed)
    return generate_random_splits(len(labels), ratios, seed)


def generate_random_splits(num_samples: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    assert abs(sum(ratios.values()) -
               1.0) < 1e-6, "Split ratios must sum to 1.0"
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_samples, generator=g).tolist()
    n_train = int(num_samples * ratios["train"])
    n_val = int(num_samples * ratios["val"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def collate_and_save(dataset_name: str,
                     data_list: List[Data],
                     labels: List[int],
                     out_root: str,
                     split_ratios: Dict[str, float],
                     split_seed: int,
                     split_strategy: str) -> None:
    out_dir = os.path.join(out_root, dataset_name, "pyg")
    os.makedirs(out_dir, exist_ok=True)
    # Use collate without touching internal storage attributes to avoid warnings

    class _Collater(InMemoryDataset):
        def __init__(self, root: str):
            super().__init__(root)
    collater = _Collater(out_dir)
    data_obj, slices = collater.collate(data_list)
    torch.save(data_obj, os.path.join(out_dir, "data.pt"))
    torch.save(slices, os.path.join(out_dir, "slices.pt"))

    splits = generate_splits(labels, split_ratios, split_seed, split_strategy)
    with open(os.path.join(out_root, dataset_name, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def load_labels_from_processed(data_path: str, slices_path: str) -> List[int]:
    data_obj = torch.load(data_path, weights_only=False)
    slices = torch.load(slices_path, weights_only=False)
    y_slices = slices["y"]
    labels: List[int] = []
    for i in range(len(y_slices) - 1):
        start = int(y_slices[i])
        end = int(y_slices[i + 1])
        y_tensor = data_obj.y[start:end]
        labels.append(int(y_tensor.view(-1)[0].item()))
    return labels


def process_dataset(ds: str, cfg: DatasetConfig) -> None:
    label_cfg = resolve_label_config(ds, cfg)
    time_source = resolve_time_source(ds, cfg)
    raw_dir = os.path.join(cfg.raw_root, ds)
    text_model_path = resolve_text_model_path(
        cfg.text_encoder_path or "sentence_transformers_all_MiniLM_L6v2",
        cfg.raw_root,
    )

    # JSON-format datasets (weibo, pheme)
    if ds in {"weibo", "pheme"}:
        assert os.path.isdir(raw_dir), f"Missing raw dir: {raw_dir}"
        source_texts = collect_json_source_texts(raw_dir)
        cache_path = os.path.join(cfg.processed_root, ds, cfg.text_cache_name)
        if not os.path.isfile(cache_path):
            build_text_embedding_cache_from_mapping(
                dataset=ds,
                processed_root=cfg.processed_root,
                mapping=source_texts,
                model_path=text_model_path,
                batch_size=cfg.text_batch_size,
                normalization_cfg=cfg.text_normalization,
                cache_name=cfg.text_cache_name,
            )
        id_to_idx, emb_matrix = _load_text_embedding_cache(cfg.processed_root, ds, cfg.text_cache_name)

        data_list: List[Data] = []
        labels: List[int] = []
        for fname in sorted(os.listdir(raw_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(raw_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            source = obj.get("source", {})
            source_id = (
                source.get("tweet id")
                or source.get("tweet_id")
                or source.get("source id")
                or source.get("source_id")
                or source.get("id")
            )
            assert source_id is not None, f"Missing source id in {fpath}"
            if str(source_id) not in source_texts:
                # Skip samples without valid source content to keep embeddings consistent
                continue

            label_raw = source.get("label")
            assert label_raw is not None, f"Missing label for source {source_id}"
            if label_cfg.numeric:
                label_value = int(label_raw)
                label_name = "rumor" if label_value == 1 else "non-rumor"
            else:
                label_name = str(label_raw).strip()
            if label_name not in label_cfg.allowed:
                if label_cfg.skip_unknown:
                    continue
                raise ValueError(f"Disallowed label {label_raw} for source {source_id}")
            label_id = label_cfg.mapping[label_name]

            nodes, edges = parse_json_tree(obj, time_source=time_source)
            keep_nodes, re_e, root_new, shifted_times = truncate_early(
                nodes,
                edges,
                cfg.cutoff_minutes,
                cfg.enforce_root_component,
            )

            data = build_pyg_data(
                nodes,
                edges,
                keep_nodes,
                re_e,
                root_new,
                cfg.cutoff_minutes,
                shifted_times,
                label_id,
                str(source_id),
            )
            assert str(source_id) in id_to_idx, f"Text embedding missing for source {source_id}"
            row = int(id_to_idx[str(source_id)])
            vec = emb_matrix[row].to(torch.float32).unsqueeze(0)
            if cfg.text_normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            if cfg.augment_enabled and cfg.augment_method in ["text_noise", "text_aug", "text_dropout"]:
                if cfg.text_noise_std > 0:
                    vec = vec + torch.randn_like(vec) * float(cfg.text_noise_std)
                if cfg.text_dropout > 0:
                    vec = torch.nn.functional.dropout(vec, p=min(0.9, float(cfg.text_dropout)), training=True)
            data.text_emb = vec
            data_list.append(data)
            labels.append(int(label_id))

        assert len(data_list) > 0, f"No samples produced for {ds}"
        collate_and_save(
            dataset_name=ds,
            data_list=data_list,
            labels=labels,
            out_root=cfg.processed_root,
            split_ratios=cfg.split_ratios,
            split_seed=cfg.split_seed,
            split_strategy=cfg.split_type,
        )
        return

    # Twitter-style datasets (tree + label.txt)
    tree_dir = os.path.join(raw_dir, "tree")
    label_path = os.path.join(raw_dir, "label.txt")
    if not os.path.isdir(tree_dir):
        raise FileNotFoundError(f"Missing tree dir: {tree_dir}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    cache_path = os.path.join(cfg.processed_root, ds, cfg.text_cache_name)
    if not os.path.isfile(cache_path):
        build_text_embedding_cache(
            ds,
            cfg.raw_root,
            cfg.processed_root,
            model_name=text_model_path,
            batch_size=cfg.text_batch_size,
            normalization_cfg=cfg.text_normalization,
            cache_name=cfg.text_cache_name,
        )
    id_to_idx, emb_matrix = _load_text_embedding_cache(cfg.processed_root, ds, cfg.text_cache_name)

    sid_to_text = parse_source_texts(raw_dir)
    label_map_raw = read_labels(label_path)
    data_list: List[Data] = []
    labels: List[int] = []
    for fname in sorted(os.listdir(tree_dir)):
        fpath = os.path.join(tree_dir, fname)
        if not os.path.isfile(fpath):
            continue
        source_id = os.path.splitext(os.path.basename(fpath))[0]
        label_str = label_map_raw.get(source_id)
        if label_str is None:
            if label_cfg.skip_unknown:
                continue
            raise ValueError(f"Missing label for source {source_id}")
        norm_label = label_str.strip()

        if norm_label not in label_cfg.allowed:
            if label_cfg.skip_unknown:
                continue
            raise ValueError(f"Disallowed label {label_str}")
        label_id = label_cfg.mapping[norm_label]

        assert source_id in sid_to_text, f"Missing text for source {source_id}"

        nodes, edges = parse_tree_file(fpath)
        keep_nodes, re_e, root_new, shifted_times = truncate_early(
            nodes,
            edges,
            cfg.cutoff_minutes,
            cfg.enforce_root_component,
        )

        data = build_pyg_data(
            nodes,
            edges,
            keep_nodes,
            re_e,
            root_new,
            cfg.cutoff_minutes,
            shifted_times,
            label_id,
            source_id,
        )
        assert source_id in id_to_idx, f"Text embedding missing for source {source_id}"
        row = int(id_to_idx[source_id])
        vec = emb_matrix[row].to(torch.float32).unsqueeze(0)
        if cfg.text_normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        if cfg.augment_enabled and cfg.augment_method in ["text_noise", "text_aug", "text_dropout"]:
            if cfg.text_noise_std > 0:
                vec = vec + torch.randn_like(vec) * float(cfg.text_noise_std)
            if cfg.text_dropout > 0:
                vec = torch.nn.functional.dropout(vec, p=min(0.9, float(cfg.text_dropout)), training=True)
        data.text_emb = vec
        data_list.append(data)
        labels.append(int(label_id))

    if not data_list:
        raise ValueError(f"No samples produced for {ds}")

    collate_and_save(
        dataset_name=ds,
        data_list=data_list,
        labels=labels,
        out_root=cfg.processed_root,
        split_ratios=cfg.split_ratios,
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Early propagation PyG preprocessor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                        choices=["twitter15", "twitter16", "weibo", "pheme"], required=True)
    parser.add_argument("--regen_splits_only", action="store_true",
                        help="Recompute splits.json using existing processed data without rebuilding features")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset not in cfg.datasets:
        raise ValueError(
            f"Dataset {args.dataset} not in configured list: {cfg.datasets}")
    if args.regen_splits_only:
        data_path = os.path.join(cfg.processed_root, args.dataset, "pyg", "data.pt")
        slices_path = os.path.join(cfg.processed_root, args.dataset, "pyg", "slices.pt")
        assert os.path.isfile(data_path), f"Missing {data_path}"
        assert os.path.isfile(slices_path), f"Missing {slices_path}"
        labels = load_labels_from_processed(data_path, slices_path)
        splits = generate_splits(labels, cfg.split_ratios, cfg.split_seed, cfg.split_type)
        splits_path = os.path.join(cfg.processed_root, args.dataset, "splits.json")
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        print(f"Regenerated stratified splits for {args.dataset} at {splits_path}")
        return
    process_dataset(args.dataset, cfg)


if __name__ == "__main__":
    main()
