"""
Training pipeline for HashNet on CIFAR-10.

Executes:
- Manifest loading and validation
- Pretrained AlexNet backbone initialization
- Differential learning rate optimization (10x on Hash Head)
- Continuation schedule progression
- Periodic validation evaluation (optional)
- Checkpoint and training log saving

Supports two data sources (set via config 'data_source'):
  - 'local_images' (default): PNG files from cifar-10/train/ directory
  - 'pickle_batches': Binary pickle files from cifar-10-batches-py/
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar10_handler import (
    LocalCIFAR10ImageDataset,
    get_default_transforms,
)
from src.datasets.cifar10_pickle_handler import (
    Cifar10PickleDataset,
    load_all_batches,
    get_default_transforms as get_pickle_transforms,
)
from src.models.hashnet import HashNetAlexNet
from src.losses.hashnet_loss import HashNetLoss, ContinuationScheduler
from src.losses.regularizers import BitDecorrelationAndBalanceLoss
from src.evaluation.metrics import ChunkedEvaluator


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ContinuationScheduler,
    global_step: int,
    device: torch.device,
    regularizer: nn.Module = None,
    cls_criterion: nn.Module = None,
    cls_weight: float = 1.0,
) -> Tuple[float, int, float, float, float]:
    model.train()
    total_loss = 0.0
    total_hash_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    current_alpha = 1.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        current_alpha = scheduler.get_alpha(global_step)
        
        optimizer.zero_grad()
        if cls_criterion is not None and model.cls_head is not None:
            h, z, cls_logits = model(images, alpha=current_alpha, return_logits=True)
            hash_loss = criterion(h, labels, alpha=current_alpha)
            cls_loss = cls_criterion(cls_logits, labels)
            loss = hash_loss + cls_weight * cls_loss
            total_hash_loss += hash_loss.item()
            total_cls_loss += cls_loss.item()
        else:
            h, z = model(images, alpha=current_alpha)
            loss = criterion(h, labels, alpha=current_alpha)
            total_hash_loss += loss.item()

        if regularizer is not None:
            reg_loss, _, _ = regularizer(h)
            loss = loss + reg_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

    epoch_loss = total_loss / max(1, num_batches)
    avg_hash_loss = total_hash_loss / max(1, num_batches)
    avg_cls_loss = total_cls_loss / max(1, num_batches) if cls_criterion is not None else 0.0
    return epoch_loss, global_step, current_alpha, avg_hash_loss, avg_cls_loss


def main():
    parser = argparse.ArgumentParser(description="Train HashNet on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="Output directory")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    bits = config["hash_bits"]
    batch_size = config["training"]["batch_size"]
    seed = config["training"]["seed"]
    gamma = config["training"]["gamma"]

    # Reproducibility seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_str = "cuda" if (torch.cuda.is_available() and config["training"]["device"] == "cuda") else "cpu"
    device = torch.device(device_str)
    print(f"=== Starting Training: HashNet {bits}-bit on CIFAR-10 ===")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")

    # Load Manifest
    manifest_path = config["manifest_path"]
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    print(f"Loaded Manifest: {manifest_path} (SHA256: {manifest['sha256'][:16]}...)")

    train_indices = manifest["splits"]["train_indices"]
    data_source = config.get("data_source", "local_images")

    if data_source == "pickle_batches":
        # Load all 60k images once, share across splits
        print(f"Loading CIFAR-10 pickle batches from: {config['data_dir']}")
        raw_data, raw_labels = load_all_batches(config["data_dir"])
        train_transform, _ = get_pickle_transforms()
        train_dataset = Cifar10PickleDataset(
            root_dir=config["data_dir"],
            data=raw_data,
            labels=raw_labels,
            indices=train_indices,
            transform=train_transform,
        )
    else:
        df_labels = pd.read_csv(os.path.join(config["data_dir"], "trainLabels.csv"))
        train_transform, _ = get_default_transforms()
        train_dataset = LocalCIFAR10ImageDataset(
            image_dir=os.path.join(config["data_dir"], "train"),
            df_labels=df_labels,
            indices=train_indices,
            transform=train_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
    )

    # Check Multi-task Classification configuration
    mt_config = config.get("multitask", {})
    use_multitask = mt_config.get("enabled", False)
    num_classes = mt_config.get("num_classes", 10) if use_multitask else None
    cls_source = mt_config.get("cls_source", "hash")
    cls_weight = mt_config.get("cls_weight", 1.0)
    cls_criterion = nn.CrossEntropyLoss().to(device) if use_multitask else None

    # Initialize Model
    model = HashNetAlexNet(
        hash_bits=bits,
        pretrained=config["pretrained"],
        num_classes=num_classes,
        cls_source=cls_source,
    ).to(device)
    criterion = HashNetLoss().to(device)
    cont_scheduler = ContinuationScheduler(gamma=gamma)

    if use_multitask:
        print(f"Active Multi-task: CrossEntropyLoss (cls_source={cls_source}, cls_weight={cls_weight}, num_classes={num_classes})")

    # Initialize Bit-Balance & Decorrelation regularizer if requested
    regularizer = None
    reg_config = config.get("regularization", {})
    if reg_config.get("use_decorr_balance", False):
        lambda_bal = reg_config.get("lambda_balance", 0.1)
        lambda_dec = reg_config.get("lambda_decorr", 0.1)
        regularizer = BitDecorrelationAndBalanceLoss(
            lambda_balance=lambda_bal, lambda_decorr=lambda_dec
        ).to(device)
        print(f"Active Regularizer: BitDecorrelationAndBalanceLoss(lambda_balance={lambda_bal}, lambda_decorr={lambda_dec})")

    # Differential learning rate: 10x for hash head and classification head
    backbone_params = list(model.features.parameters()) + list(model.classifier.parameters())
    head_params = list(model.hash_head.parameters())
    if model.cls_head is not None:
        head_params += list(model.cls_head.parameters())

    optimizer = optim.SGD([
        {"params": backbone_params, "lr": config["training"]["lr_backbone"]},
        {"params": head_params, "lr": config["training"]["lr_hash_head"]},
    ], momentum=config["training"]["momentum"], weight_decay=config["training"]["weight_decay"])

    exp_name = args.exp_name or config.get("exp_name") or f"hashnet_{bits}bit"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_csv_path = os.path.join(exp_dir, "training_logs.csv")

    log_fields = ["epoch", "step", "loss", "hash_loss", "cls_loss", "alpha", "time_sec"]
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        loss, global_step, alpha, hash_loss, cls_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            cont_scheduler,
            global_step,
            device,
            regularizer=regularizer,
            cls_criterion=cls_criterion,
            cls_weight=cls_weight,
        )
        ep_time = time.time() - ep_start

        if use_multitask:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {loss:.4f} (Hash: {hash_loss:.4f}, Cls: {cls_loss:.4f}) | Alpha: {alpha:.3f} | Step: {global_step} | Time: {ep_time:.1f}s")
        else:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {loss:.4f} | Alpha: {alpha:.3f} | Step: {global_step} | Time: {ep_time:.1f}s")

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch,
                "step": global_step,
                "loss": loss,
                "hash_loss": hash_loss,
                "cls_loss": cls_loss,
                "alpha": alpha,
                "time_sec": ep_time,
            })

    # Save final checkpoint
    checkpoint_path = os.path.join(exp_dir, "checkpoint_final.pth")
    torch.save({
        "epoch": epochs,
        "global_step": global_step,
        "hash_bits": bits,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, checkpoint_path)
    print(f"Training completed in {time.time() - start_time:.1f}s. Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
