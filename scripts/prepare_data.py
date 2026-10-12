"""
Script to prepare and inspect local CIFAR-10 data from cifar-10/ folder,
and create fixed, immutable split manifests with SHA256 checksums.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar10_handler import (
    generate_cifar10_splits,
    compute_manifest_hash,
    CIFAR10_CLASSES,
    CLASS_TO_IDX,
)


def load_local_cifar10(data_dir: str) -> Tuple[pd.DataFrame, List[int]]:
    """Loads CIFAR-10 labels and image IDs from cifar-10/trainLabels.csv."""
    csv_path = os.path.join(data_dir, "trainLabels.csv")
    assert os.path.exists(csv_path), f"trainLabels.csv not found at {csv_path}"

    print(f"Loading local labels from: {csv_path}")
    df = pd.read_csv(csv_path)
    assert len(df) == 50000, f"Expected 50,000 samples, got {len(df)}"

    int_labels = [CLASS_TO_IDX[l] for l in df['label'].values]
    return df, int_labels


def main():
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 manifests from local folder.")
    parser.add_argument("--data-dir", type=str, default="cifar-10", help="Path to cifar-10 data folder")
    parser.add_argument("--output-dir", type=str, default="outputs/splits", help="Directory to save manifests")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed for deterministic split")
    parser.add_argument("--mode", type=str, default="survey_source", choices=["survey_source", "original_paper"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Preparing CIFAR-10 Manifest (Mode: {args.mode}, Seed: {args.seed}) ---")
    df_labels, int_labels = load_local_cifar10(args.data_dir)

    print(f"Total labeled samples: {len(df_labels)}")
    unique_labels, counts = np.unique(int_labels, return_counts=True)
    print("Class label distribution across 50,000 samples:")
    for lbl, count in zip(unique_labels, counts):
        print(f"  Class {lbl} ({CIFAR10_CLASSES[lbl]}): {count} images")

    splits = generate_cifar10_splits(
        int_labels,
        mode=args.mode,
        seed=args.seed,
        query_per_class=100,
        train_per_class=500
    )

    manifest_hash = compute_manifest_hash(splits)
    print(f"\nManifest SHA256 Checksum: {manifest_hash}")

    manifest_content = {
        "dataset": "local_cifar10",
        "data_dir": args.data_dir,
        "mode": args.mode,
        "seed": args.seed,
        "sha256": manifest_hash,
        "num_query": len(splits["query_indices"]),
        "num_train": len(splits["train_indices"]),
        "num_database": len(splits["database_indices"]),
        "splits": splits,
    }

    manifest_path = os.path.join(args.output_dir, f"manifest_cifar10_{args.mode}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_content, f, indent=2)

    print(f"Manifest successfully saved to: {manifest_path}")
    print("Split summary:")
    print(f"  - Query items: {len(splits['query_indices'])} (100 per class)")
    print(f"  - Train items: {len(splits['train_indices'])} (500 per class)")
    print(f"  - Database items: {len(splits['database_indices'])} (4,400 per class)")


if __name__ == "__main__":
    main()
