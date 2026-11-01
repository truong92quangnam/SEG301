"""
Prepare the mAP@54000 data split from the full 60,000-image CIFAR-10 pickle batches.

Protocol (Dubey IEEE TCSVT 2021, Table IV):
    - query    :  1,000  (100 / class)
    - train    :  5,000  (500 / class, disjoint from query)
    - database : 54,000  (all remaining: 60,000 - 1,000 - 5,000)

All three sets are mutually exclusive and their union covers all 60,000 images.

Usage:
    python scripts/prepare_data_60k.py \\
        --root cifar-10-batches-py \\
        --output outputs/splits/manifest_cifar10_60k_54k_gallery.json \\
        --seed 2021
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.cifar10_pickle_handler import load_all_batches
# ---------------------------------------------------------------------------


def generate_splits_60k(
    labels: list,
    seed: int = 2021,
    query_per_class: int = 100,
    train_per_class: int = 500,
) -> dict:
    """
    Generates deterministic, stratified, fully-disjoint splits for 60,000 images.

    Returns dict with keys: 'query_indices', 'train_indices', 'database_indices'.
    """
    assert len(labels) == 60_000, f"Expected 60,000 labels, got {len(labels)}"

    rng = np.random.RandomState(seed)
    labels_arr = np.array(labels)
    num_classes = 10

    query_indices: list = []
    train_indices: list = []
    database_indices: list = []

    for c in range(num_classes):
        cls_idx = np.where(labels_arr == c)[0].copy()
        rng.shuffle(cls_idx)

        q_idx = cls_idx[:query_per_class].tolist()
        t_idx = cls_idx[query_per_class: query_per_class + train_per_class].tolist()
        d_idx = cls_idx[query_per_class + train_per_class:].tolist()

        query_indices.extend(q_idx)
        train_indices.extend(t_idx)
        database_indices.extend(d_idx)

    # ---- Sanity checks ----
    q_set, t_set, d_set = set(query_indices), set(train_indices), set(database_indices)

    assert len(q_set) == query_per_class * num_classes, "Query size mismatch"
    assert len(t_set) == train_per_class * num_classes, "Train size mismatch"
    assert len(d_set) == 60_000 - query_per_class * num_classes - train_per_class * num_classes, \
        f"Database size mismatch: expected {60_000 - 6000} got {len(d_set)}"

    assert q_set.isdisjoint(t_set),  "Query ∩ Train  ≠ ∅"
    assert q_set.isdisjoint(d_set),  "Query ∩ DB     ≠ ∅"
    assert t_set.isdisjoint(d_set),  "Train ∩ DB     ≠ ∅"
    assert len(q_set | t_set | d_set) == 60_000, "Splits do not cover all 60,000 images"

    print(f"  Query   : {len(query_indices):,} images ({query_per_class} / class)")
    print(f"  Train   : {len(train_indices):,} images ({train_per_class} / class)")
    print(f"  Database: {len(database_indices):,} images")

    return {
        "query_indices":    [int(x) for x in query_indices],
        "train_indices":    [int(x) for x in train_indices],
        "database_indices": [int(x) for x in database_indices],
    }


def compute_manifest_hash(manifest_data: dict) -> str:
    """Returns SHA256 of the JSON-serialised manifest."""
    json_str = json.dumps(manifest_data, sort_keys=True)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare mAP@54000 manifest from CIFAR-10 pickle batches.")
    parser.add_argument("--root",   type=str, default="cifar-10-batches-py", help="Path to cifar-10-batches-py dir")
    parser.add_argument("--output", type=str, default="outputs/splits/manifest_cifar10_60k_54k_gallery.json")
    parser.add_argument("--seed",   type=int, default=2021, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Loading CIFAR-10 batches from: {args.root}")
    _, labels = load_all_batches(args.root)
    print(f"Total images loaded: {len(labels):,}")

    print("\nGenerating splits (seed={args.seed})...")
    splits = generate_splits_60k(labels, seed=args.seed)

    manifest = {
        "version": "1.0",
        "protocol": "mAP@54000 (Dubey IEEE TCSVT 2021)",
        "data_source": "pickle_batches",
        "pickle_root": args.root,
        "seed": args.seed,
        "num_total": 60_000,
        "splits": splits,
        "split_sizes": {
            "query":    len(splits["query_indices"]),
            "train":    len(splits["train_indices"]),
            "database": len(splits["database_indices"]),
        },
        "sha256": "",   # placeholder; filled below after computing hash
    }
    # Compute hash of everything except the sha256 field itself
    temp = {k: v for k, v in manifest.items() if k != "sha256"}
    manifest["sha256"] = compute_manifest_hash(temp)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {args.output}")
    print(f"SHA256: {manifest['sha256']}")


if __name__ == "__main__":
    main()

