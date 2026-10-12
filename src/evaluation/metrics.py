"""
Retrieval Metrics and Evaluation Module.

Implements exact AP, mAP@K, and Hamming Distance calculations.
Provides two implementations:
1. ReferenceEvaluator: Exact, un-vectorized/naive implementation for ground-truth unit testing.
2. ChunkedEvaluator: Chunked, memory-safe matrix evaluation for large databases (prevents OOM).

Includes exact handling of:
- Ties policy (stable sort)
- Zero-relevant policy (AP = 0.0)
- Normalization denominator: min(R_q, K)
- Self-match policy
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch


def compute_hamming_distance_reference(
    query_codes: np.ndarray,
    database_codes: np.ndarray
) -> np.ndarray:
    """
    Computes pairwise Hamming distance between queries and database codes.
    
    Args:
        query_codes: (N_q, K) binary codes in {-1, +1} or {0, 1}
        database_codes: (N_d, K) binary codes in {-1, +1} or {0, 1}
        
    Returns:
        (N_q, N_d) matrix of integer Hamming distances.
    """
    # Standardize to {0, 1}
    q = (query_codes > 0).astype(np.int32)
    d = (database_codes > 0).astype(np.int32)
    
    # Broadcast or XOR
    N_q, K = q.shape
    N_d, _ = d.shape
    
    # D_H = sum_k (q_k != d_k)
    distances = np.zeros((N_q, N_d), dtype=np.int32)
    for i in range(N_q):
        distances[i] = np.sum(q[i] != d, axis=1)
        
    return distances


def compute_ap_single_query(
    ranked_relevance: np.ndarray,
    total_relevant: int,
    k: Optional[int] = None
) -> float:
    """
    Computes Average Precision (AP@K) for a single query.
    
    Args:
        ranked_relevance: 1D array of 0s and 1s indicating if each retrieved
                          database item is relevant (1) or not (0), sorted by distance.
        total_relevant: Total number of relevant items in the entire database (R_q).
        k: Cutoff rank K. If None, considers all ranked items.
        
    Returns:
        AP score in range [0.0, 1.0].
    """
    if total_relevant == 0:
        return 0.0
        
    if k is not None:
        ranked_relevance = ranked_relevance[:k]
        denom = min(total_relevant, k)
    else:
        denom = total_relevant
        
    if denom == 0:
        return 0.0
        
    rel_count = np.sum(ranked_relevance)
    if rel_count == 0:
        return 0.0
        
    # Positions: 1, 2, ..., N
    ranks = np.arange(1, len(ranked_relevance) + 1)
    cum_hits = np.cumsum(ranked_relevance)
    precision_at_k = cum_hits / ranks
    
    ap = np.sum(precision_at_k * ranked_relevance) / denom
    return float(ap)


class ReferenceEvaluator:
    """
    Reference (naive) evaluator for unit testing and ground-truth verification.
    """
    def __init__(self, topk: Optional[int] = None):
        self.topk = topk

    def evaluate(
        self,
        query_codes: np.ndarray,
        database_codes: np.ndarray,
        query_labels: np.ndarray,
        database_labels: np.ndarray,
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluates mAP using naive loops and numpy.
        """
        distances = compute_hamming_distance_reference(query_codes, database_codes)
        N_q = len(query_labels)
        aps = []
        
        for i in range(N_q):
            q_label = query_labels[i]
            q_dists = distances[i]
            
            # Stable sort indices
            sort_indices = np.argsort(q_dists, kind='stable')
            
            # Ground truth relevance
            retrieved_labels = database_labels[sort_indices]
            rel_vector = (retrieved_labels == q_label).astype(np.int32)
            total_rel = int(np.sum(database_labels == q_label))
            
            ap = compute_ap_single_query(rel_vector, total_rel, k=self.topk)
            aps.append(ap)
            
        mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0
        return {
            "mAP": mean_ap,
            "AP_list": aps,
        }


class ChunkedEvaluator:
    """
    Optimized, chunked Hamming distance evaluator.
    Computes distances and ranking in batches of queries to prevent OOM on GPUs / large DBs.
    """
    def __init__(self, topk: Optional[int] = 54000, chunk_size: int = 200, device: str = "cpu"):
        self.topk = topk
        self.chunk_size = chunk_size
        self.device = torch.device(device)

    def evaluate(
        self,
        query_codes: Union[np.ndarray, torch.Tensor],
        database_codes: Union[np.ndarray, torch.Tensor],
        query_labels: Union[np.ndarray, torch.Tensor],
        database_labels: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluates mAP@K efficiently.
        """
        if isinstance(query_codes, np.ndarray):
            query_codes = torch.from_numpy(query_codes)
        if isinstance(database_codes, np.ndarray):
            database_codes = torch.from_numpy(database_codes)
        if isinstance(query_labels, np.ndarray):
            query_labels = torch.from_numpy(query_labels)
        if isinstance(database_labels, np.ndarray):
            database_labels = torch.from_numpy(database_labels)

        # Standardize codes to {-1, +1} float tensor for matrix inner-product:
        # Hamming distance = 0.5 * (K - q * d^T)
        q_codes = torch.where(query_codes > 0, 1.0, -1.0).float().to(self.device)
        d_codes = torch.where(database_codes > 0, 1.0, -1.0).float().to(self.device)
        q_labels = query_labels.to(self.device)
        d_labels = database_labels.to(self.device)

        N_q, K = q_codes.shape
        N_d = d_codes.shape[0]
        cutoff = min(N_d, self.topk) if self.topk is not None else N_d

        aps = []
        
        # Precompute total relevant items in database per class label
        unique_labels = torch.unique(q_labels)
        total_rel_per_class = {
            int(lbl.item()): int((d_labels == lbl).sum().item())
            for lbl in unique_labels
        }

        for start_idx in range(0, N_q, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N_q)
            q_chunk = q_codes[start_idx:end_idx]
            q_lbl_chunk = q_labels[start_idx:end_idx]

            # Matrix multiplication: (chunk_q, K) x (K, N_d) -> (chunk_q, N_d)
            # inner_product in [-K, K]
            # Hamming distance = 0.5 * (K - inner_product)
            # Minimizing Hamming distance is equivalent to MAXIMIZING inner_product
            inner_prod = torch.matmul(q_chunk, d_codes.t())  # (chunk_q, N_d)
            
            # Convert to CPU for ranking & precision calculation
            inner_prod_cpu = inner_prod.cpu()
            
            for i in range(q_chunk.shape[0]):
                q_lbl = int(q_lbl_chunk[i].item())
                sim_scores = inner_prod_cpu[i].numpy()
                
                # Sort descending by inner product (stable sort for reproducibility)
                # In numpy, negate scores for stable ascending sort
                sorted_indices = np.argsort(-sim_scores, kind='stable')
                
                # Truncate to cutoff
                top_indices = sorted_indices[:cutoff]
                retrieved_labels = database_labels[top_indices].cpu().numpy()
                
                rel_vector = (retrieved_labels == q_lbl).astype(np.int32)
                total_rel = total_rel_per_class.get(q_lbl, 0)
                
                ap = compute_ap_single_query(rel_vector, total_rel, k=cutoff)
                aps.append(ap)

        mean_map = float(np.mean(aps)) if len(aps) > 0 else 0.0
        return {
            "mAP": mean_map,
            "AP_list": aps,
        }

