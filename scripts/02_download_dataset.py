#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download TAGFN dataset from HuggingFace datasets and flatten it, then output global statistics.

Default Hub path is kayzliu/TAGFN (corresponds to kayzliu/tagfn GitHub, with repo name TAGFN on HF).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# ----- HuggingFace dataset: list of candidate repo_ids (tried in order) -----
_CANDIDATE_REPO_IDS = (
    "kayzliu/TAGFN",
    "kayzliu/tagfn",
)


def _print_manual_download_help(repo_id: str, output_dir: Path) -> None:
    """When Hub access fails, print manual download instructions."""
    print("\n" + "=" * 60)
    print("[Hint] Manual download when HuggingFace access fails")
    print("=" * 60)
    print("1) Install CLI: pip install -U huggingface_hub hf_transfer")
    print("   (Optional) Enable: export HF_HUB_ENABLE_HF_TRANSFER=1")
    print("2) Login (if dataset becomes gated): huggingface-cli login")
    print("3) Download to local directory, e.g.:")
    print(f"   huggingface-cli download {repo_id} --repo-type dataset --local-dir \"{output_dir / 'hf_snapshot'}\"")
    print("4) For China network, try mirror (self-verify compliance):")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print(f"   Then run this script again, or directly: huggingface-cli download {repo_id} ...")
    print("5) Open dataset page in browser, manually download parquet and organize:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print("=" * 60 + "\n")


def _resolve_repo_id() -> str:
    """Try repo_ids in order, return first resolvable config id; raise last error if all fail."""
    last_err: Exception | None = None
    for rid in _CANDIDATE_REPO_IDS:
        try:
            from datasets import get_dataset_config_names

            get_dataset_config_names(rid)
            return rid
        except Exception as e:  # noqa: BLE001 - only used to return next candidate
            last_err = e
    assert last_err is not None
    raise last_err


def _load_full_dataset_dict(repo_id: str, *, cache_dir: str) -> Any:
    """
    Load full DatasetDict (multiple sub-datasets like politifact / fakeddit / gossipcop).

    Parameters
    ----------
    cache_dir
        Passed to load_dataset(..., cache_dir=...) to cache parquet to specified directory.

    Returns
    -------
    datasets.DatasetDict
    """
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(repo_id)
    if not configs:
        raw = load_dataset(repo_id, cache_dir=cache_dir)
        if isinstance(raw, DatasetDict):
            return raw
        if isinstance(raw, Dataset):
            return DatasetDict({"default": raw})
        raise TypeError(f"Cannot recognize load_dataset return type: {type(raw)!r}")
    parts = {}
    for name in configs:
        # Load each config separately to avoid single huge memory usage
        parts[name] = load_dataset(repo_id, name, cache_dir=cache_dir)
    return DatasetDict(parts)


def _iter_all_splits(dataset_dict) -> Iterable[tuple[str, str, Any]]:
    """Yield (config_name, split_name, Dataset)."""
    from datasets import Dataset, DatasetDict

    if isinstance(dataset_dict, Dataset):
        yield "default", "train", dataset_dict
        return
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict or Dataset, got {type(dataset_dict)!r}")

    for cfg_name, dcfg in dataset_dict.items():
        if isinstance(dcfg, Dataset):
            yield cfg_name, "train", dcfg
            continue
        if isinstance(dcfg, DatasetDict):
            for split_name in dcfg.keys():
                yield cfg_name, split_name, dcfg[split_name]
            continue
        raise TypeError(f"Cannot iterate config {cfg_name!r} of type {type(dcfg)!r}")


def _normalize_edge_index(edge_index: Any) -> tuple[list[int], list[int]] | None:
    """
    Convert edge_index from various formats to two 1D lists (row, col).

    Common HF formats: [[...], [...]] or numpy/torch arrays.
    """
    if edge_index is None:
        return None
    if hasattr(edge_index, "tolist"):
        edge_index = edge_index.tolist()
    if isinstance(edge_index, dict):
        # Rare: Arrow arrays may be serialized as dict
        if "data" in edge_index and isinstance(edge_index["data"], list):
            edge_index = edge_index["data"]
    if not isinstance(edge_index, (list, tuple)) or len(edge_index) != 2:
        return None
    row, col = edge_index[0], edge_index[1]
    if not isinstance(row, (list, tuple)) or not isinstance(col, (list, tuple)):
        return None
    if len(row) != len(col):
        return None
    return list(map(int, row)), list(map(int, col))


def _graph_num_nodes(edge_index: Any, row: dict[str, Any]) -> int:
    """Get graph node count: prefer max(edge_index)+1, fallback to num_nodes/number_of_nodes fields."""
    ne = _normalize_edge_index(edge_index)
    if ne is not None:
        r, c = ne
        if r or c:
            return max(max(r), max(c)) + 1
    for key in ("num_nodes", "number_of_nodes", "n_nodes", "num_node"):
        v = row.get(key)
        if v is not None:
            return int(v)
    y = row.get("y")
    if isinstance(y, (list, tuple)) and len(y) > 0:
        return len(y)
    return 0


def _graph_num_edges(edge_index: Any) -> int:
    ne = _normalize_edge_index(edge_index)
    if ne is None:
        return 0
    r, c = ne
    return len(r)  # same as len(c)


def _collect_labels(y_val: Any, bucket: list[int]) -> None:
    """Expand one sample's label into bucket (supports graph-level or node-level arrays)."""
    if y_val is None:
        return
    if isinstance(y_val, (list, tuple)):
        for v in y_val:
            bucket.append(int(v))
    else:
        bucket.append(int(y_val))


def compute_global_stats(dataset_dict, *, stats_max_graphs: int | None, batch_size: int = 256) -> dict[str, Any]:
    """
    Scan all configs/splits, collect total nodes, edges, label distribution across all graphs.

    Parameters
    ----------
    stats_max_graphs
        Max graphs to count per (config, split); None means no limit.
    batch_size
        Batch size for datasets.Dataset.iter, used to speed up scanning.

    Returns
    -------
    dict
        total_graphs, total_nodes, total_edges, label_counter (Counter), per_config_lines
    """
    from tqdm import tqdm

    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    all_labels: list[int] = []
    per_cfg: dict[str, dict[str, int]] = {}

    for cfg_name, split_name, dset in _iter_all_splits(dataset_dict):
        key = f"{cfg_name}/{split_name}"
        g = n = e = 0
        limit = stats_max_graphs
        n_rows = len(dset)
        if limit is not None:
            n_rows = min(n_rows, limit)
        n_batches = (n_rows + batch_size - 1) // batch_size if n_rows else 0
        desc = f"Counting {key}"
        processed = 0
        for batch in tqdm(dset.iter(batch_size=batch_size), total=n_batches, desc=desc, unit="batch"):
            keys = list(batch.keys())
            if not keys:
                continue
            bsz = len(batch[keys[0]])
            for i in range(bsz):
                if limit is not None and processed >= limit:
                    break
                row = {k: batch[k][i] for k in keys}
                processed += 1
                g += 1
                ei = row.get("edge_index")
                nn = _graph_num_nodes(ei, row)
                ee = _graph_num_edges(ei)
                n += nn
                e += ee
                y_field = row.get("y")
                if y_field is None:
                    y_field = row.get("label")
                _collect_labels(y_field, all_labels)
            if limit is not None and processed >= limit:
                break
        total_graphs += g
        total_nodes += n
        total_edges += e
        per_cfg[key] = {"graphs": g, "nodes": n, "edges": e}

    return {
        "total_graphs": total_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "label_counter": Counter(all_labels),
        "per_config": per_cfg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TAGFN (HuggingFace datasets) to target directory and compute statistics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/workspace/data/tagfn"),
        help="Target directory for dataset save_to_disk (defaults to cursor.md unified path)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help="Specify Hub repo id explicitly; empty means auto-try between kayzliu/TAGFN and kayzliu/tagfn",
    )
    parser.add_argument(
        "--stats-max-graphs",
        type=int,
        default=None,
        metavar="N",
        help="Max N graphs to count per config/split (speed up testing); default unlimited (full scan, large datasets may be slow)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace download cache directory; default uses .hf_hub_cache under output",
    )
    args = parser.parse_args()
    output_dir: Path = args.output.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else (output_dir / ".hf_hub_cache")

    print(f"[Progress] Target directory: {output_dir}")
    print(f"[Progress] Hub cache directory: {cache_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo_id.strip() if args.repo_id else ""
    try:
        if not repo_id:
            print("[Progress] Detecting available Hub repo_id ...")
            repo_id = _resolve_repo_id()
        print(f"[Progress] Using Hub dataset: {repo_id}")

        print("[Progress] Downloading and loading all sub-datasets from Hub (may be large, please wait)...")
        ds_dict = _load_full_dataset_dict(repo_id, cache_dir=str(cache_dir))

        print(f"[Progress] Saving to disk -> {output_dir} ...")
        ds_dict.save_to_disk(str(output_dir))

        if args.stats_max_graphs is not None:
            print(f"[Progress] Scanning samples and computing statistics (max {args.stats_max_graphs} graphs per split)...")
        else:
            print("[Progress] Scanning samples and computing statistics (full scan, may be slow; use --stats-max-graphs to limit)...")
        stats = compute_global_stats(ds_dict, stats_max_graphs=args.stats_max_graphs)

        print("\n" + "=" * 60)
        print("[Result] Dataset basic statistics (all sub-datasets / all splits combined)")
        if args.stats_max_graphs is not None:
            print(f"(Note: using sampling limit of {args.stats_max_graphs} graphs per split)")
        print("=" * 60)
        print(f"  Total samples (graphs): {stats['total_graphs']}")
        print(f"  Total nodes:            {stats['total_nodes']}")
        print(f"  Total edges:            {stats['total_edges']}")
        print("  Label distribution (prefer field 'y', fallback 'label'; includes all node/graph labels):")
        if not stats["label_counter"]:
            print("    (Could not parse labels from 'y' / 'label' fields; please verify column names on Hub Data Studio)")
        else:
            for lab in sorted(stats["label_counter"]):
                cnt = stats["label_counter"][lab]
                print(f"    Label {lab}: {cnt}")
        print("\n  Per config/split details (graphs, nodes, edges):")
        for k in sorted(stats["per_config"]):
            v = stats["per_config"][k]
            print(f"    {k}: graphs={v['graphs']}, nodes={v['nodes']}, edges={v['edges']}")
        print("=" * 60 + "\n")
        print("[Completed] Data saved.")
        return 0

    except Exception as err:  # noqa: BLE001 - need unified handling and show manual download instructions
        print(f"\n[Error] HuggingFace download or processing failed: {err!r}\n", file=sys.stderr)
        fallback_repo = args.repo_id.strip() if args.repo_id else _CANDIDATE_REPO_IDS[0]
        _print_manual_download_help(fallback_repo, output_dir)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
