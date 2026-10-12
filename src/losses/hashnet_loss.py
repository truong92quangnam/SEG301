"""
Loss functions for Deep Hashing.

Paper: "HashNet: Deep Learning to Hash by Continuation" (ICCV 2017)
Loss: Weighted Maximum Likelihood Loss addressing pairwise similarity imbalance.
Schedule: Continuation scaling alpha_t = sqrt(1 + gamma * t).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class HashNetLoss(nn.Module):
    """
    Weighted Maximum Likelihood Loss for HashNet.
    
    L = sum_{s_ij in S} ( w_ij * log(1 + exp(alpha * u_ij)) - s_ij * alpha * u_ij )
    where:
    - u_ij = 0.5 * h_i^T h_j
    - s_ij in {0, 1}
    - w_ij weights positive and negative pairs to handle imbalance.
    """
    def __init__(self):
        super(HashNetLoss, self).__init__()

    def forward(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Computes pairwise HashNet loss for a batch.
        
        Args:
            h: Continuous hash representations of shape (B, K), h in (-1, 1).
            labels: 1D class labels of shape (B,).
            alpha: Current continuation parameter.
            
        Returns:
            Scalar loss tensor.
        """
        B = h.shape[0]
        
        # 1. Similarity matrix S in {0, 1}
        # S_ij = 1 if labels[i] == labels[j], else 0
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        S = (labels_col == labels_row).float()
        
        # 2. Pairwise inner product: u_ij = 0.5 * h_i^T h_j
        inner_prod = torch.matmul(h, h.t()) # (B, B)
        u = 0.5 * inner_prod
        
        # 3. Calculate imbalance weights w_ij
        num_positive = S.sum()
        num_total = B * B
        num_negative = num_total - num_positive
        
        # Avoid zero-division if entire batch is of the same class or all different
        if num_positive > 0 and num_negative > 0:
            c_pos = num_total / num_positive
            c_neg = num_total / num_negative
            w = torch.where(S > 0.5, c_pos, c_neg)
        else:
            w = torch.ones_like(S)
            
        # 4. Numerically stable weighted loss:
        # log(1 + exp(alpha * u)) can overflow if alpha * u is large.
        # Use log1p(exp(...)) with clamp or softplus:
        # softplus(x) = log(1 + exp(x))
        alpha_u = alpha * u
        # softplus is numerically stable for large alpha_u
        log_term = torch.nn.functional.softplus(alpha_u)
        
        loss_matrix = w * log_term - S * alpha_u
        loss = loss_matrix.mean()
        
        return loss


class ContinuationScheduler:
    """
    Continuation parameter schedule:
    alpha_t = sqrt(1.0 + gamma * t)
    where t is the current training iteration (step).
    """
    def __init__(self, gamma: float = 0.001, initial_alpha: float = 1.0):
        self.gamma = gamma
        self.initial_alpha = initial_alpha

    def get_alpha(self, step: int) -> float:
        """Returns alpha at given step."""
        return float(np.sqrt(1.0 + self.gamma * step))

