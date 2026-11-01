"""
Bit-Balance and Bit-Decorrelation Regularization for Deep Hashing.

Mathematical Foundation:
1. Bit Balance (Entropy Maximization):
   In binary coding {-1, +1}^K, information entropy of bit k is maximized when
   P(b_k = +1) = P(b_k = -1) = 0.5.
   Over a mini-batch B:
   L_balance = (1 / K) * sum_{k=1}^K ( (1 / B) * sum_{i=1}^B h_{ik} )^2

2. Bit Decorrelation (Mutual Independence / Redundancy Minimization):
   To eliminate redundant bits (especially in 32-bit and 64-bit regimes),
   the covariance between different bits must vanish:
   C = (1 / B) * H^T @ H  in R^{K x K}
   L_decorr = (1 / (K * (K - 1))) * sum_{j != k} (C_jk)^2

Total Regularizer:
   L_reg = lambda_balance * L_balance + lambda_decorr * L_decorr
"""

from typing import Tuple
import torch
import torch.nn as nn


class BitDecorrelationAndBalanceLoss(nn.Module):
    """
    Combined Bit-Balance and Bit-Decorrelation Regularizer.
    Compatible with any deep hashing architecture (HashNet, DHN, etc.).
    """
    def __init__(self, lambda_balance: float = 0.1, lambda_decorr: float = 0.1):
        super(BitDecorrelationAndBalanceLoss, self).__init__()
        self.lambda_balance = lambda_balance
        self.lambda_decorr = lambda_decorr

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes balance and decorrelation penalties.

        Args:
            h: Continuous hash representations of shape (B, K), h in (-1, 1).

        Returns:
            total_reg_loss: Scalar weighted sum of balance and decorrelation loss.
            loss_balance: Scalar unweighted balance loss.
            loss_decorr: Scalar unweighted decorrelation loss.
        """
        B, K = h.shape
        if B <= 1 or K <= 1:
            zero = torch.tensor(0.0, device=h.device)
            return zero, zero, zero

        # 1. Bit Balance Loss: mean of each bit across batch should be 0
        bit_mean = torch.mean(h, dim=0)  # Shape: (K,)
        loss_balance = torch.mean(bit_mean ** 2)

        # 2. Bit Decorrelation Loss: off-diagonal correlation -> 0
        # C_jk = (1 / B) * sum_i h_ij * h_ik
        C = torch.matmul(h.t(), h) / float(B)  # Shape: (K, K)
        
        # Zero out diagonal elements
        diag_mask = torch.eye(K, dtype=torch.bool, device=h.device)
        off_diag_C = C.masked_fill(diag_mask, 0.0)

        # Average squared correlation across K*(K-1) off-diagonal pairs
        num_off_diags = float(K * (K - 1))
        loss_decorr = torch.sum(off_diag_C ** 2) / num_off_diags

        total_reg_loss = (
            self.lambda_balance * loss_balance + self.lambda_decorr * loss_decorr
        )
        return total_reg_loss, loss_balance, loss_decorr

