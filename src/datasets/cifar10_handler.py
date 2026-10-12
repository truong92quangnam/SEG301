"""
CIFAR-10 Dataset Handler and Manifest Creator.

Manages data loading from local image folder (cifar-10/) with labeled images
and ensures exact and immutable splits for the experimental protocols:
- 1,000 query (100 per class)
- 5,000 train (500 per class)
- Remaining 44,000 database (4,400 per class)
Mutually exclusive: Q intersect T = empty, Q intersect D = empty, T intersect D = empty.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CIFAR10_CLASSES)}


def compute_manifest_hash(manifest_data: dict) -> str:
    """Computes SHA256 checksum of manifest content."""
    json_str = json.dumps(manifest_data, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def generate_cifar10_splits(
    labels: List[int],
    mode: str = "survey_source",
    seed: int = 2021,
    query_per_class: int = 100,
    train_per_class: int = 500,
) -> Dict[str, List[int]]:
    """
    Generates deterministic index splits for CIFAR-10.
    
    Args:
        labels: List of integer class labels (length 50,000 or 60,000).
        mode: 'survey_source' or 'original_paper'.
        seed: Random seed for deterministic selection.
        query_per_class: Number of queries per class (100 -> 1,000 total).
        train_per_class: Number of training samples per class (500 -> 5,000 total).
        
    Returns:
        Dict with keys: 'query_indices', 'train_indices', 'database_indices'.
    """
    num_samples = len(labels)
    assert num_samples in [50000, 60000], f"Expected 50,000 or 60,000 samples, got {num_samples}"
    
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    num_classes = 10
    
    class_to_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    query_indices = []
    train_indices = []
    database_indices = []
    
    for c in range(num_classes):
        cls_indices = class_to_indices[c].copy()
        rng.shuffle(cls_indices)
        
        # Pick query
        q_idx = cls_indices[:query_per_class].tolist()
        query_indices.extend(q_idx)
        
        remaining = cls_indices[query_per_class:]
        
        if mode == "survey_source":
            # 5,000 train, remaining database (completely disjoint)
            t_idx = remaining[:train_per_class].tolist()
            d_idx = remaining[train_per_class:].tolist()
            train_indices.extend(t_idx)
            database_indices.extend(d_idx)
        elif mode == "original_paper":
            d_idx = remaining.tolist()
            database_indices.extend(d_idx)
            t_idx = remaining[:train_per_class].tolist()
            train_indices.extend(t_idx)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    # Verification
    q_set = set(query_indices)
    t_set = set(train_indices)
    d_set = set(database_indices)
    
    expected_db_size = num_samples - (query_per_class * num_classes) - (train_per_class * num_classes)
    
    assert len(q_set) == query_per_class * num_classes, "Query set size mismatch"
    assert len(t_set) == train_per_class * num_classes, "Training set size mismatch"
    assert len(d_set) == expected_db_size, f"Database size mismatch: {len(d_set)} vs {expected_db_size}"
    
    assert len(q_set.intersection(t_set)) == 0, "Query and Train overlap detected!"
    assert len(q_set.intersection(d_set)) == 0, "Query and Database overlap detected!"
    if mode == "survey_source":
        assert len(t_set.intersection(d_set)) == 0, "Train and Database overlap detected in survey_source mode!"
        assert len(q_set.union(t_set).union(d_set)) == num_samples, "Splits do not cover all images!"

    return {
        "query_indices": [int(x) for x in query_indices],
        "train_indices": [int(x) for x in train_indices],
        "database_indices": [int(x) for x in database_indices],
    }


class LocalCIFAR10ImageDataset(Dataset):
    """
    Dataset loader reading images directly from cifar-10/train/<id>.png.
    """
    def __init__(
        self,
        image_dir: str,
        df_labels: pd.DataFrame,
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            image_dir: Path to directory containing <id>.png (e.g. 'cifar-10/train').
            df_labels: DataFrame with columns ['id', 'label'].
            indices: List of row indices (0-based) into df_labels.
            transform: PyTorch transforms.
        """
        self.image_dir = image_dir
        if indices is not None:
            self.df = df_labels.iloc[indices].reset_index(drop=True)
            self.original_indices = [int(i) for i in indices]
        else:
            self.df = df_labels.copy()
            self.original_indices = list(range(len(df_labels)))
            
        self.transform = transform
        self.ids = self.df['id'].values
        # Convert string labels to integer classes
        self.labels = [CLASS_TO_IDX[lbl] if isinstance(lbl, str) else int(lbl) for lbl in self.df['label'].values]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = self.labels[idx]
        orig_idx = self.original_indices[idx]
        return img, label, orig_idx


def get_default_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns (train_transform, test_transform) according to protocol."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, test_transform
