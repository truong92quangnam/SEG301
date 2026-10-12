"""
Unit Tests for CIFAR-10 Manifest and Split Integrity.
Validates:
- No overlap between Query, Train, and Database in 'survey_source' mode
- Exact count matching (1,000 query, 5,000 train, 54,000 database)
- Determinism with fixed seed
- SHA256 manifest hash consistency
"""

import numpy as np
import pytest

from src.datasets.cifar10_handler import (
    generate_cifar10_splits,
    compute_manifest_hash,
)


def test_survey_source_split_integrity():
    """Checks exact sizes and mutually exclusive subsets for survey_source mode."""
    # 60,000 labels, 6,000 per class
    labels = []
    for c in range(10):
        labels.extend([c] * 6000)
    
    splits = generate_cifar10_splits(labels, mode="survey_source", seed=2021)
    
    q_idx = splits["query_indices"]
    t_idx = splits["train_indices"]
    d_idx = splits["database_indices"]
    
    assert len(q_idx) == 1000
    assert len(t_idx) == 5000
    assert len(d_idx) == 54000
    
    q_set = set(q_idx)
    t_set = set(t_idx)
    d_set = set(d_idx)
    
    # Mutually disjoint
    assert len(q_set.intersection(t_set)) == 0
    assert len(q_set.intersection(d_set)) == 0
    assert len(t_set.intersection(d_set)) == 0
    assert len(q_set.union(t_set).union(d_set)) == 60000


def test_split_determinism():
    """Ensures identical seeds generate identical indices and manifest hash."""
    labels = []
    for c in range(10):
        labels.extend([c] * 6000)
        
    s1 = generate_cifar10_splits(labels, mode="survey_source", seed=42)
    s2 = generate_cifar10_splits(labels, mode="survey_source", seed=42)
    
    assert s1["query_indices"] == s2["query_indices"]
    assert s1["train_indices"] == s2["train_indices"]
    assert s1["database_indices"] == s2["database_indices"]
    
    h1 = compute_manifest_hash(s1)
    h2 = compute_manifest_hash(s2)
    assert h1 == h2

