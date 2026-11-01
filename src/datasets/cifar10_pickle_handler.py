"""
CIFAR-10 Pickle Batch Dataset Handler.

Loads all 60,000 CIFAR-10 images directly from the official binary pickle files
(cifar-10-batches-py/) without requiring a PNG image directory.

Layout expected:
    cifar-10-batches-py/
        data_batch_1  ... data_batch_5   (train: 50,000 images)
        test_batch                       (test:  10,000 images)
        batches.meta

Global index mapping:
    data_batch_1: indices   0 –  9,999
    data_batch_2: indices 10,000 – 19,999
    data_batch_3: indices 20,000 – 29,999
    data_batch_4: indices 30,000 – 39,999
    data_batch_5: indices 40,000 – 49,999
    test_batch:   indices 50,000 – 59,999

mAP@54000 protocol (Dubey IEEE TCSVT 2021):
    - query    :  1,000  (100 / class, drawn from all 60k)
    - train    :  5,000  (500 / class, completely disjoint from query)
    - database : 54,000  (remaining — all images not in query or train)
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CIFAR10_CLASSES)}

# Official batch file names in load order
_BATCH_FILES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
]


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------

def load_all_batches(root_dir: str) -> Tuple[np.ndarray, List[int]]:
    """
    Loads all 60,000 CIFAR-10 images from the binary pickle files.

    Args:
        root_dir: Path to the cifar-10-batches-py directory.

    Returns:
        data   – uint8 array of shape (60000, 3072).
        labels – list of 60,000 integer class labels (0–9).
    """
    all_data: List[np.ndarray] = []
    all_labels: List[int] = []

    for fname in _BATCH_FILES:
        fpath = os.path.join(root_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Batch file not found: {fpath}")
        with open(fpath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        all_data.append(batch[b"data"])          # shape (10000, 3072)
        all_labels.extend(batch[b"labels"])      # list of 10000 ints

    data = np.concatenate(all_data, axis=0)      # (60000, 3072)
    return data, all_labels


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Cifar10PickleDataset(Dataset):
    """
    PyTorch Dataset that reads images from CIFAR-10 binary pickle batches.

    Each item is a tuple of (image_tensor, label_int, original_global_index).
    The global index refers to the position in the concatenated 60,000-image array.

    Args:
        root_dir  : Path to the cifar-10-batches-py directory.
        data      : Pre-loaded uint8 array (60000, 3072). If None, loaded lazily.
        labels    : Pre-loaded list of 60,000 integer labels. If None, loaded lazily.
        indices   : Subset of global indices to expose. If None, all 60,000 images.
        transform : Torchvision transforms to apply.
    """

    def __init__(
        self,
        root_dir: str,
        data: Optional[np.ndarray] = None,
        labels: Optional[List[int]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform

        # Load raw data if not provided (shared across splits for efficiency)
        if data is None or labels is None:
            data, labels = load_all_batches(root_dir)

        self._data = data          # (N, 3072) uint8
        self._labels = np.array(labels, dtype=np.int64)  # (N,)

        # Apply subset if requested
        if indices is not None:
            self._subset_indices = np.array(indices, dtype=np.int64)
        else:
            self._subset_indices = np.arange(len(self._labels), dtype=np.int64)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._subset_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image   : Transformed image tensor (3, H, W).
            label   : Integer class label (0–9).
            orig_idx: Global index in the 60,000-image pool.
        """
        global_idx = int(self._subset_indices[idx])
        raw = self._data[global_idx]                        # (3072,) uint8

        # CIFAR-10 stores images in CHW order, row-major (R, G, B planes)
        img_chw = raw.reshape(3, 32, 32)                    # (3, 32, 32)
        img_hwc = img_chw.transpose(1, 2, 0)                # (32, 32, 3)
        img = Image.fromarray(img_hwc, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = int(self._labels[global_idx])
        return img, label, global_idx


# ---------------------------------------------------------------------------
# Transforms (identical to cifar10_handler.py for consistent preprocessing)
# ---------------------------------------------------------------------------

def get_default_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns (train_transform, test_transform) with ImageNet normalisation."""
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

