"""
Extraction and Evaluation Script for HashNet on CIFAR-10.

Executes:
- Loads trained checkpoint
- Extracts binary codes and continuous embeddings for Query and Database sets
- Runs memory-safe ChunkedEvaluator to compute mAP@K
- Saves per-query metrics and run_metadata.json
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar10_handler import (
    LocalCIFAR10ImageDataset,
    get_default_transforms,
)
from src.models.hashnet import HashNetAlexNet
from src.evaluation.metrics import ChunkedEvaluator


def extract_features_and_codes(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_codes = []
    all_continuous = []
    all_labels = []
    all_orig_indices = []

    with torch.no_grad():
        for images, labels, orig_idx in loader:
            images = images.to(device)
            # Continuous and raw logits
            h, z = model(images, alpha=1.0)
            # Binary codes in {-1, +1}
            b = torch.where(z >= 0, 1.0, -1.0)

            all_codes.append(b.cpu().numpy())
            all_continuous.append(h.cpu().numpy())
            all_labels.append(labels.numpy())
            all_orig_indices.append(orig_idx.numpy())

    codes = np.concatenate(all_codes, axis=0)
    continuous = np.concatenate(all_continuous, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    orig_indices = np.concatenate(all_orig_indices, axis=0)
    return codes, continuous, labels, orig_indices


def main():
    parser = argparse.ArgumentParser(description="Extract codes and evaluate retrieval.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    bits = config["hash_bits"]
    topk = config["evaluation"]["topk"]
    chunk_size = config["evaluation"]["chunk_size"]
    out_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)

    device_str = "cuda" if (torch.cuda.is_available() and config["training"]["device"] == "cuda") else "cpu"
    device = torch.device(device_str)
    print(f"=== Evaluating Retrieval: HashNet {bits}-bit on CIFAR-10 ===")
    print(f"Device: {device} | Cutoff Top-K: {topk}")

    # Load Manifest
    with open(config["manifest_path"], "r") as f:
        manifest = json.load(f)

    df_labels = pd.read_csv(os.path.join(config["data_dir"], "trainLabels.csv"))
    _, test_transform = get_default_transforms()

    query_indices = manifest["splits"]["query_indices"]
    database_indices = manifest["splits"]["database_indices"]

    query_dataset = LocalCIFAR10ImageDataset(
        image_dir=os.path.join(config["data_dir"], "train"),
        df_labels=df_labels,
        indices=query_indices,
        transform=test_transform,
    )
    database_dataset = LocalCIFAR10ImageDataset(
        image_dir=os.path.join(config["data_dir"], "train"),
        df_labels=df_labels,
        indices=database_indices,
        transform=test_transform,
    )

    query_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=0)
    database_loader = DataLoader(database_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Load Model
    model = HashNetAlexNet(hash_bits=bits, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Extract Codes
    print("Extracting query binary codes...")
    q_codes, q_cont, q_labels, q_orig_idx = extract_features_and_codes(model, query_loader, device)
    print(f"Query codes shape: {q_codes.shape}")

    print("Extracting database binary codes...")
    d_codes, d_cont, d_labels, d_orig_idx = extract_features_and_codes(model, database_loader, device)
    print(f"Database codes shape: {d_codes.shape}")

    # Save codes for visualization demo
    codes_save_path = os.path.join(out_dir, f"extracted_codes_{bits}bit.npz")
    np.savez_compressed(
        codes_save_path,
        q_codes=q_codes,
        q_cont=q_cont,
        q_labels=q_labels,
        q_orig_idx=q_orig_idx,
        d_codes=d_codes,
        d_cont=d_cont,
        d_labels=d_labels,
        d_orig_idx=d_orig_idx,
    )
    print(f"Saved extracted codes to: {codes_save_path}")

    # Evaluate mAP
    print(f"Computing mAP@{topk} using ChunkedEvaluator...")
    evaluator = ChunkedEvaluator(topk=topk, chunk_size=chunk_size, device="cpu")
    eval_res = evaluator.evaluate(q_codes, d_codes, q_labels, d_labels)

    map_score = eval_res["mAP"]
    print(f"\n==========================================")
    print(f"Result: HashNet {bits}-bit mAP@{topk} = {map_score * 100:.2f}% ({map_score:.4f})")
    print(f"==========================================\n")

    # Save per_query_metrics.csv
    per_query_path = os.path.join(out_dir, "per_query_metrics.csv")
    with open(per_query_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_index", "original_image_id", "label", "AP"])
        for i, ap_val in enumerate(eval_res["AP_list"]):
            orig_id = df_labels.iloc[q_orig_idx[i]]["id"]
            lbl = q_labels[i]
            writer.writerow([i, orig_id, lbl, ap_val])

    # Save run_metadata.json
    metadata = {
        "model": "HashNet",
        "hash_bits": bits,
        "mAP": map_score,
        "mAP_percentage": f"{map_score * 100:.2f}%",
        "topk": topk,
        "checkpoint": args.checkpoint,
        "manifest_sha256": manifest["sha256"],
        "num_query": len(q_labels),
        "num_database": len(d_labels),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Results successfully saved to {out_dir}")


if __name__ == "__main__":
    main()
