"""
Unit Tests for Retrieval Metrics & AP Calculation.
Validates:
- Synthetic known Hamming distance
- Synthetic known ranking & AP
- Zero-relevant item query policy (AP = 0.0)
- Hamming ties handling
- Equivalence between ReferenceEvaluator and ChunkedEvaluator
"""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    compute_ap_single_query,
    compute_hamming_distance_reference,
    ReferenceEvaluator,
    ChunkedEvaluator,
)


def test_hamming_distance_known():
    """Checks Hamming distance on hand-crafted bit patterns."""
    q = np.array([
        [1, -1, 1, -1],  # 1 0 1 0
        [-1, -1, -1, -1] # 0 0 0 0
    ])
    d = np.array([
        [1, -1, 1, -1],  # dist to q0 = 0, dist to q1 = 2
        [-1, 1, -1, 1],  # dist to q0 = 4, dist to q1 = 2
        [1, 1, 1, 1]     # dist to q0 = 2, dist to q1 = 4
    ])
    
    dist = compute_hamming_distance_reference(q, d)
    expected = np.array([
        [0, 4, 2],
        [2, 2, 4]
    ])
    np.testing.assert_array_equal(dist, expected)


def test_ap_single_query_known():
    """
    Checks AP calculation against known mathematical manual example.
    Retrieved: [1, 0, 1, 1, 0]
    total_relevant in DB = 4
    Ranks:
    pos 1: hit 1 -> prec 1/1 = 1.0
    pos 2: hit 0 -> prec 1/2 = 0.5
    pos 3: hit 1 -> prec 2/3 = 0.6667
    pos 4: hit 1 -> prec 3/4 = 0.75
    pos 5: hit 0 -> prec 3/5 = 0.6
    
    Sum of precision at relevant hits: 1.0 + 2/3 + 3/4 = 1.0 + 0.6666667 + 0.75 = 2.4166667
    Total relevant = 4
    AP = 2.4166667 / 4 = 0.6041667
    """
    ranked_rel = np.array([1, 0, 1, 1, 0])
    total_relevant = 4
    ap = compute_ap_single_query(ranked_rel, total_relevant, k=5)
    expected_ap = (1.0 + (2.0 / 3.0) + (3.0 / 4.0)) / 4.0
    assert pytest.approx(ap, abs=1e-5) == expected_ap


def test_zero_relevant_query():
    """Query with zero relevant items in database should strictly have AP = 0.0."""
    ranked_rel = np.array([0, 0, 0, 0])
    total_relevant = 0
    ap = compute_ap_single_query(ranked_rel, total_relevant, k=4)
    assert ap == 0.0


def test_evaluator_equivalence():
    """
    Verifies that ReferenceEvaluator and ChunkedEvaluator produce identical mAP
    on a synthetic dataset with Hamming ties and diverse classes.
    """
    np.random.seed(42)
    N_q = 50
    N_d = 200
    K = 16 # 16 bits
    
    # Random binary codes {-1, +1}
    q_codes = np.random.choice([-1, 1], size=(N_q, K))
    d_codes = np.random.choice([-1, 1], size=(N_d, K))
    
    # Labels 0..9
    q_labels = np.random.randint(0, 10, size=N_q)
    d_labels = np.random.randint(0, 10, size=N_d)
    
    ref_eval = ReferenceEvaluator(topk=N_d)
    ref_res = ref_eval.evaluate(q_codes, d_codes, q_labels, d_labels)
    
    chunk_eval = ChunkedEvaluator(topk=N_d, chunk_size=15, device="cpu")
    chunk_res = chunk_eval.evaluate(q_codes, d_codes, q_labels, d_labels)
    
    assert pytest.approx(ref_res["mAP"], abs=1e-5) == chunk_res["mAP"]
    np.testing.assert_allclose(ref_res["AP_list"], chunk_res["AP_list"], atol=1e-5)

