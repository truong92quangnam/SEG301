"""
Runner script for Multi-Task Ablation Studies:
1. Classifier Placement: Feature (4096-D) vs Hash code (32-D)
2. Hyperparameter Sensitivity: Beta in {0.1, 0.5, 1.0, 2.0}
"""

import os
import subprocess
import sys
import yaml
import numpy as np
from src.evaluation.metrics import ChunkedEvaluator

ABLATIONS = [
    ("hashnet_32bit_multitask_feat_54k", "configs/hashnet_32bit_multitask_feat_54k.yaml"),
    ("hashnet_32bit_multitask_beta01_54k", "configs/hashnet_32bit_multitask_beta01_54k.yaml"),
    ("hashnet_32bit_multitask_beta05_54k", "configs/hashnet_32bit_multitask_beta05_54k.yaml"),
    ("hashnet_32bit_multitask_beta20_54k", "configs/hashnet_32bit_multitask_beta20_54k.yaml"),
]

def main():
    python_bin = sys.executable
    results = {}

    for exp_name, cfg_path in ABLATIONS:
        print(f"\n=======================================================")
        print(f"Running Ablation: {exp_name}")
        print(f"=======================================================")
        
        # 1. Train
        cmd_train = [python_bin, "scripts/train.py", "--config", cfg_path]
        subprocess.run(cmd_train, check=True)
        
        # 2. Extract & Evaluate
        ckpt_path = f"outputs/experiments/{exp_name}/checkpoint_final.pth"
        cmd_eval = [python_bin, "scripts/evaluate.py", "--config", cfg_path, "--checkpoint", ckpt_path]
        subprocess.run(cmd_eval, check=True)

    print("\nAll ablations completed successfully!")

if __name__ == "__main__":
    main()
