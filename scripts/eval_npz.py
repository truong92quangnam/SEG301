"""
Fast Evaluation Script for Extracted Hash Codes (.npz).
Computes mAP@K for any specified K without re-running AlexNet feature extraction.
"""

import argparse
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.evaluation.metrics import ChunkedEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate mAP from extracted codes NPZ.")
    parser.add_argument("--npz", type=str, required=True, help="Path to extracted_codes_*.npz")
    parser.add_argument("--topk", type=int, nargs="+", default=[1000, 44000], help="List of Top-K cutoffs")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation results to JSON")
    args = parser.parse_args()

    data = np.load(args.npz)
    q_codes = data["q_codes"]
    d_codes = data["d_codes"]
    q_labels = data["q_labels"]
    d_labels = data["d_labels"]

    print(f"Loaded codes from: {args.npz}")
    print(f"Query codes: {q_codes.shape} | Database codes: {d_codes.shape}")

    results = {}
    for k in args.topk:
        actual_k = min(k, len(d_labels))
        evaluator = ChunkedEvaluator(topk=actual_k, chunk_size=200, device="cpu")
        res = evaluator.evaluate(q_codes, d_codes, q_labels, d_labels)
        map_val = res["mAP"]
        results[f"mAP@{actual_k}"] = {
            "score": map_val,
            "percentage": f"{map_val * 100:.2f}%"
        }
        print(f"-> mAP@{actual_k}: {map_val * 100:.2f}% ({map_val:.4f})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
