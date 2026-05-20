import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from Libs.data.processor import compute_structural_features


def _assert_tensor_2_by_n(edge_index: Tensor) -> None:
    """Asserts that an edge index tensor has shape (2, E) and dtype long.

    Args:
        edge_index: Edge index tensor to validate.
    """
    assert isinstance(edge_index, Tensor), "edge_index must be a torch.Tensor"
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
        f"edge_index must be 2 x E, got {tuple(edge_index.shape)}"
    )
    assert edge_index.dtype in (torch.long, torch.int64), (
        f"edge_index dtype must be long, got {edge_index.dtype}"
    )


def _remove_self_loops(edge_index: Tensor) -> Tensor:
    """Removes self-loops from directed edge list.

    Args:
        edge_index: Input edges of shape (2, E).

    Returns:
        Edge index without self-loops.
    """
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    return edge_index[:, mask]


def _unique_union_edges(edge_index_a: Tensor, edge_index_b: Tensor, num_nodes: int) -> Tensor:
    """Returns the union of two directed edge sets, deduplicated and loop-free.

    The result keeps no self-loops and contains each directed edge at most once.

    Args:
        edge_index_a: First edge set (2, E1).
        edge_index_b: Second edge set (2, E2).
        num_nodes: Number of nodes, used to hash pairs.

    Returns:
        A tensor of shape (2, E) with unique edges.
    """
    _assert_tensor_2_by_n(edge_index_a)
    _assert_tensor_2_by_n(edge_index_b)

    ea = _remove_self_loops(edge_index_a)
    eb = _remove_self_loops(edge_index_b)

    assert ea.numel() > 0, "edge_index_a has no valid edges"

    # Bounds check to catch bad LP edges early.
    assert int(ea.max().item()) < num_nodes, (
        f"edge_index_a has index >= num_nodes ({int(ea.max().item())} >= {num_nodes})"
    )
    if eb.numel() > 0:
        assert int(eb.max().item()) < num_nodes, (
            f"edge_index_b has index >= num_nodes ({int(eb.max().item())} >= {num_nodes})"
        )

    src = torch.cat([ea[0], eb[0]], dim=0)
    dst = torch.cat([ea[1], eb[1]], dim=0)

    # Hash pairs to 1D keys and unique them.
    keys = src.to(torch.int64) * int(num_nodes) + dst.to(torch.int64)
    uniq_keys = torch.unique(keys)
    uniq_src = torch.div(uniq_keys, int(num_nodes), rounding_mode="floor")
    uniq_dst = uniq_keys % int(num_nodes)
    out = torch.stack([uniq_src.to(torch.long), uniq_dst.to(torch.long)], dim=0)
    return out


def _stable_seed(source_id: object, base: int = 0) -> int:
    """Returns a stable integer seed derived from source_id and base."""
    text = str(source_id)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed_core = int(h[:8], 16)
    return int((seed_core + int(base)) % (2**31))


def _prefix_mask_from_strategy(times: Tensor, strategy: str, minutes: Optional[int], k: Optional[int], root_index: Optional[int], include_root: bool) -> Tensor:
    """Computes prefix node mask according to strategy."""
    times_long = times.to(torch.long)
    n = times_long.numel()
    if strategy in {"minutes", "hop", "window"}:
        cutoff = int(minutes if minutes is not None else times_long.max().item() if n > 0 else 0)
        mask = times_long <= cutoff
    elif strategy == "count":
        if n == 0:
            return torch.zeros_like(times_long, dtype=torch.bool)
        budget = int(k if k is not None else n)
        budget = max(1, min(budget, n))
        order = torch.argsort(times_long)
        chosen = order[:budget]
        mask = torch.zeros_like(times_long, dtype=torch.bool)
        mask[chosen] = True
    else:
        mask = torch.ones_like(times_long, dtype=torch.bool)
    if include_root and root_index is not None and 0 <= int(root_index) < n:
        mask[int(root_index)] = True
    return mask


def _apply_missingness(mask: Tensor, edge_index: Tensor, drop_node: float, drop_edge: float, seed: int, root_index: Optional[int], include_root: bool) -> Tuple[Tensor, Tensor]:
    """Applies node/edge dropout to observed prefix."""
    keep_nodes = mask.to(torch.bool).clone()
    if drop_node > 0.0 and keep_nodes.numel() > 0:
        gen = torch.Generator(device=keep_nodes.device)
        gen.manual_seed(int(seed))
        rand_node = torch.rand(keep_nodes.shape, generator=gen, device=keep_nodes.device)
        keep_nodes = keep_nodes & (rand_node >= float(drop_node))
        if include_root and root_index is not None and 0 <= int(root_index) < keep_nodes.numel():
            keep_nodes[int(root_index)] = True
    edge_mask = keep_nodes[edge_index[0]] & keep_nodes[edge_index[1]]
    if drop_edge > 0.0 and edge_mask.numel() > 0:
        gen_edge = torch.Generator(device=edge_mask.device)
        gen_edge.manual_seed(int(seed))
        edge_mask = edge_mask & (torch.rand(edge_mask.shape, generator=gen_edge, device=edge_mask.device) >= float(drop_edge))
    edge_filtered = edge_index[:, edge_mask]
    return keep_nodes, edge_filtered


def _reindex_edge_index(edge_index: Tensor, keep_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Reindexes an edge list into the compact index space defined by `keep_mask`."""
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    old_to_new = torch.full(
        (int(keep_mask.numel()),),
        -1,
        dtype=torch.long,
        device=keep_mask.device,
    )
    old_to_new[kept_indices] = torch.arange(kept_indices.numel(), dtype=torch.long, device=keep_mask.device)
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), kept_indices
    src = old_to_new[edge_index[0].to(torch.long)]
    dst = old_to_new[edge_index[1].to(torch.long)]
    valid = (src >= 0) & (dst >= 0)
    if not bool(valid.any()):
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), kept_indices
    return torch.stack([src[valid], dst[valid]], dim=0).contiguous(), kept_indices


def _materialize_prefix_view(data: Data, keep_mask: Tensor, edge_index: Tensor) -> Data:
    """Materializes a prefix-only graph view while preserving the original full graph."""
    keep_mask = keep_mask.to(torch.bool)
    full_edge_index = data.edge_index.to(torch.long)
    full_x = data.x.to(torch.float32)
    full_root_index = data.root_index.view(-1).to(torch.long)
    full_shifted_times = data.node_time_minutes_shifted.to(torch.long)
    full_times = data.node_time_minutes.to(torch.long)
    full_node_uid = list(data.node_uid)
    full_node_tweet_id = list(data.node_tweet_id)

    prefix_edge_index, kept_indices = _reindex_edge_index(edge_index.to(torch.long), keep_mask)
    keep_nodes = [int(index) for index in kept_indices.tolist()]
    root_old = int(full_root_index[0].item())
    if root_old not in keep_nodes:
        raise ValueError("Prefix materialization dropped the root node.")
    root_index_new = int(keep_nodes.index(root_old))
    shifted_times_list = [int(full_shifted_times[index].item()) for index in keep_nodes]
    full_nodes = [
        (str(full_node_uid[index]), str(full_node_tweet_id[index]), int(full_times[index].item()))
        for index in range(len(full_node_uid))
    ]
    full_edges = [(int(src), int(dst)) for src, dst in full_edge_index.t().tolist()]
    reindexed_edges = [(int(src), int(dst)) for src, dst in prefix_edge_index.t().tolist()]

    data.full_x = full_x
    data.full_edge_index = full_edge_index
    data.full_root_index = full_root_index
    data.full_node_time_minutes = full_times
    data.full_node_time_minutes_shifted = full_shifted_times
    data.full_node_uid = full_node_uid
    data.full_node_tweet_id = full_node_tweet_id

    data.x = compute_structural_features(
        nodes=full_nodes,
        edges=full_edges,
        keep_nodes=keep_nodes,
        reindexed_edges=reindexed_edges,
        root_index_new=root_index_new,
        feature_time_scale=max(shifted_times_list) if len(shifted_times_list) > 0 else 1,
        shifted_times_list=shifted_times_list,
    )
    data.edge_index = prefix_edge_index
    data.root_index = torch.tensor([root_index_new], dtype=torch.long)
    data.node_uid = [full_node_uid[index] for index in keep_nodes]
    data.node_tweet_id = [full_node_tweet_id[index] for index in keep_nodes]
    data.node_time_minutes = full_times[kept_indices].clone()
    data.node_time_minutes_shifted = full_shifted_times[kept_indices].clone()
    data.prefix_node_mask = torch.ones(int(len(keep_nodes)), dtype=torch.bool)
    data.prefix_edge_index = data.edge_index.clone()
    data.prefix_observed_ratio = float(keep_mask.float().mean().item())
    if hasattr(data, "t"):
        data.t = data.node_time_minutes_shifted.to(torch.float32)
    for stale_attr in ("h0", "h1", "h_cat"):
        if hasattr(data, stale_attr):
            delattr(data, stale_attr)
    return data


class GraphDataset(InMemoryDataset):
    """Graph classification dataset for D2Plus processed trees.

    This dataset loads pre-collated PyG data saved by the preprocessor and
    exposes train/val/test splits. It optionally merges link-prediction edges
    into each sample and sets time features for TGAT models.

    Attributes:
        root: Path like `Input/processed/twitter15`.
        split: Split name in {"train", "val", "test"}.
        use_lp_edges: Whether to merge predicted edges from `lp_edges/`.
        lp_dir: Directory storing LP edges, defaults to `root/lp_edges`.
        set_t_for_tgat: Whether to set `data.t` from shifted minutes.
        indices: Indices of samples within the collated storage for this split.
    """

    def __init__(
        self,
        root: str,
        split: str,
        use_lp_edges: bool = False,
        lp_dir: Optional[str] = None,
        set_t_for_tgat: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        use_potentials: bool = False,
        potentials_dir: Optional[str] = None,
        potentials_pair: Optional[List[int]] = None,
        prefix_cfg: Optional[Dict] = None,
        observability_cfg: Optional[Dict] = None,
        text_cache_name: Optional[str] = None,
        force_text_cache: bool = False,
        text_normalize: bool = False,
    ) -> None:
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self._processed_dir = os.path.join(self.root, "pyg")
        data_path = os.path.join(self._processed_dir, "data.pt")
        slices_path = os.path.join(self._processed_dir, "slices.pt")
        splits_path = os.path.join(self.root, "splits.json")

        assert os.path.isfile(data_path), f"Missing {data_path}"
        assert os.path.isfile(slices_path), f"Missing {slices_path}"
        assert os.path.isfile(splits_path), f"Missing {splits_path}"

        self.data = torch.load(data_path, weights_only=False)
        self.slices = torch.load(slices_path, weights_only=False)

        with open(splits_path, "r", encoding="utf-8") as f:
            all_splits = json.load(f)
        assert split in all_splits, f"Split '{split}' not in splits.json"
        self._indices: List[int] = list(all_splits[split])

        # Optional LP edge merging settings
        self.use_lp_edges = bool(use_lp_edges)
        self.lp_dir = lp_dir if lp_dir is not None else os.path.join(self.root, "lp_edges")
        self.set_t_for_tgat = bool(set_t_for_tgat)

        # Optional potentials loading settings
        self.use_potentials = bool(use_potentials)
        self.potentials_dir = potentials_dir if potentials_dir is not None else os.path.join(self.root, "rd_potentials")
        self.potentials_pair = potentials_pair if potentials_pair is not None else [0, 1]
        # Prefix and observability configuration
        prefix_cfg = prefix_cfg if prefix_cfg is not None else {}
        self._prefix_strategy = str(prefix_cfg.get("strategy", "none")).lower()
        self._prefix_minutes = prefix_cfg.get("window", prefix_cfg.get("minutes", None))
        self._prefix_k = prefix_cfg.get("k", None)
        self._prefix_include_root = bool(prefix_cfg.get("include_root", True))

        observability_cfg = observability_cfg if observability_cfg is not None else {}
        self._drop_edge = float(observability_cfg.get("drop_edge", 0.0))
        self._drop_node = float(observability_cfg.get("drop_node", 0.0))
        self._obs_seed = int(observability_cfg.get("seed", 0))
        self._force_text_cache = bool(force_text_cache)
        self._text_normalize = bool(text_normalize)

        # Optional text embedding cache (graph-level). Load once if present.
        resolved_text_cache_name = str(text_cache_name).strip() if text_cache_name is not None else "text_embs.pt"
        text_cache_path = os.path.join(self.root, resolved_text_cache_name)
        self._text_cache_loaded = False
        self._text_id_to_idx: Optional[Dict[str, int]] = None
        self._text_emb_matrix: Optional[Tensor] = None
        if os.path.isfile(text_cache_path):
            obj = torch.load(text_cache_path, weights_only=False)
            ids: List[str] = list(obj["ids"])  # type: ignore[index]
            emb: Tensor = obj["emb"].to(torch.float32)  # type: ignore[index]
            self._text_id_to_idx = {sid: i for i, sid in enumerate(ids)}
            self._text_emb_matrix = emb
            self._text_cache_loaded = True

        # Basic integrity check on the first sample
        if len(self._indices) > 0:
            sample = super().get(self._indices[0])
            self._validate_sample(sample)

        # Initialize num_classes as None, will be computed on first access
        self._num_classes = None

    def _validate_sample(self, data: Data) -> None:
        """Validates a single sample from storage for structural integrity.

        Args:
            data: A PyG `Data` object from storage.
        """
        assert isinstance(data, Data), "Loaded object must be torch_geometric.data.Data"
        assert hasattr(data, "x") and data.x.dim() == 2, "x must be [N, F]"
        assert hasattr(data, "edge_index"), "edge_index missing"
        assert hasattr(data, "y") and data.y.numel() == 1, "y must be a scalar class id"
        assert hasattr(data, "root_index") and data.root_index.numel() == 1, "root_index must be scalar"
        assert hasattr(data, "node_time_minutes_shifted"), "shifted time is required"
        assert hasattr(data, "source_id"), "source_id metadata is required"
        assert data.node_time_minutes_shifted.size(0) == data.num_nodes, (
            "shifted time length must equal number of nodes"
        )
        _assert_tensor_2_by_n(data.edge_index)
        assert int(data.edge_index.max().item()) < data.num_nodes, (
            "edge_index contains node id >= num_nodes"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        """Override to use split-local indexing."""
        return self.get(idx)

    def indices(self) -> List[int]:
        """Returns the indices for this split."""
        return self._indices

    def get(self, idx: int) -> Data:
        """Returns the split-specific sample with optional LP edge merging.

        Args:
            idx: Index within the current split.

        Returns:
            A `Data` object with optional `t` and merged edges.
        """
        # Map split-local index to underlying storage index
        real_idx = int(self._indices[idx])
        data = super().get(real_idx)

        if self.set_t_for_tgat:
            t = data.node_time_minutes_shifted.to(torch.float32)
            assert t.numel() == data.num_nodes, "t length must match num_nodes"
            data.t = t

        if self.use_lp_edges:
            lp_path = os.path.join(self.lp_dir, f"{data.source_id}.pt")
            if os.path.isfile(lp_path):
                lp_obj = torch.load(lp_path, weights_only=False)
                if isinstance(lp_obj, dict):
                    assert "edge_index" in lp_obj, "LP file dict must contain 'edge_index'"
                    lp_edge_index = lp_obj["edge_index"]
                else:
                    lp_edge_index = lp_obj
                _assert_tensor_2_by_n(lp_edge_index)

                merged = _unique_union_edges(
                    data.edge_index.to(torch.long),
                    lp_edge_index.to(torch.long),
                    num_nodes=data.num_nodes,
                )
                assert merged.numel() > 0, "Merged edges are empty"
                data.edge_index = merged
            else:
                # LP edges file doesn't exist, skip merging
                pass

        if self.use_potentials:
            pot_path = os.path.join(self.potentials_dir, f"{data.source_id}.pt")
            if os.path.isfile(pot_path):
                pot_obj = torch.load(pot_path, weights_only=False)
                if isinstance(pot_obj, dict):
                    assert "h" in pot_obj, "Potentials file dict must contain 'h'"
                    Hc = pot_obj["h"].to(torch.float32)
                else:
                    Hc = pot_obj.to(torch.float32)
                
                # Handle different potential tensor shapes
                if Hc.dim() == 2:
                    # Shape [C, H] - select indices from potentials_pair
                    c, hdim = int(Hc.size(0)), int(Hc.size(1))
                    sel = [int(i) for i in self.potentials_pair]
                    assert len(sel) >= 2, "potentials_pair must contain at least two indices"
                    for i in sel:
                        assert 0 <= i < c, f"Invalid potentials index {i} for C={c}"
                    if len(sel) == 2:
                        i0, i1 = sel[0], sel[1]
                        assert i0 != i1, "potentials_pair indices must be distinct"
                        data.h0 = Hc[i0].unsqueeze(0)  # [1, H]
                        data.h1 = Hc[i1].unsqueeze(0)  # [1, H]
                    else:
                        parts = [Hc[i] for i in sel]
                        data.h_cat = torch.cat(parts, dim=-1).unsqueeze(0)  # [1, K*H]
                elif Hc.dim() == 1:
                    # Shape [H] - single potential vector
                    sel = [int(i) for i in self.potentials_pair]
                    assert len(sel) == 2, "Single-vector potentials only support pair length 2"
                    data.h0 = Hc.unsqueeze(0)  # [1, H]
                    data.h1 = Hc.unsqueeze(0)  # [1, H]
                else:
                    raise ValueError(f"Unexpected potentials tensor shape: {Hc.shape}")
            else:
                # Missing potentials file; leave unset for on-the-fly computation if model supports it
                pass

        # Lazy attach graph-level text embedding if cache is available and attribute missing
        if self._text_cache_loaded:
            assert hasattr(data, "source_id"), "source_id metadata is required for text cache lookup"
            sid = str(data.source_id)
            assert self._text_id_to_idx is not None and self._text_emb_matrix is not None
            if self._force_text_cache or not hasattr(data, "text_emb"):
                assert sid in self._text_id_to_idx, f"text_emb missing for source_id={sid}"
                row = int(self._text_id_to_idx[sid])
                vec = self._text_emb_matrix[row].to(torch.float32).unsqueeze(0)
                if self._text_normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                data.text_emb = vec

        # Build prefix view and observability-aware subgraph if configured
        use_prefix_view = (self._prefix_strategy != "none") or (self._drop_edge > 0.0) or (self._drop_node > 0.0)
        if use_prefix_view:
            root_idx = int(data.root_index.view(-1)[0].item()) if hasattr(data, "root_index") else None
            prefix_mask = _prefix_mask_from_strategy(
                times=data.node_time_minutes_shifted,
                strategy=self._prefix_strategy,
                minutes=self._prefix_minutes,
                k=self._prefix_k,
                root_index=root_idx,
                include_root=self._prefix_include_root,
            )
            seed_val = _stable_seed(getattr(data, "source_id", "0"), base=self._obs_seed)
            obs_mask, edge_prefix = _apply_missingness(
                mask=prefix_mask,
                edge_index=data.edge_index.to(torch.long),
                drop_node=self._drop_node,
                drop_edge=self._drop_edge,
                seed=seed_val,
                root_index=root_idx,
                include_root=self._prefix_include_root,
            )
            data = _materialize_prefix_view(data, obs_mask, edge_prefix)

        return data

    @property
    def num_features(self) -> int:
        """Number of input node features per node."""
        if len(self._indices) == 0:
            return 0
        sample = super().get(self._indices[0])
        return int(sample.x.size(1))

    @property
    def num_classes(self) -> int:
        """Number of classes inferred from available labels."""
        if self._num_classes is None:
            # Compute on first access using raw data
            all_indices = []
            splits_path = os.path.join(self.root, "splits.json")
            with open(splits_path, "r", encoding="utf-8") as f:
                all_splits = json.load(f)
            for k in ("train", "val", "test"):
                if k in all_splits:
                    all_indices.extend(all_splits[k])
            unique_indices = sorted(set(int(i) for i in all_indices))
            assert len(unique_indices) > 0, "No indices found in splits.json"
            # Access raw data using proper PyG method
            labels: List[int] = [int(self._data.y[i].view(-1)[0].item()) for i in unique_indices]
            self._num_classes = int(max(labels) + 1)
        return self._num_classes
