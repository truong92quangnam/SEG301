"""
Loss function for Deep Hashing Network (DHN).

Paper: "Deep Hashing Network for Efficient Similarity Retrieval" (AAAI 2016)
Objective:
    L = L_pair + lambda * L_quant
where:
    L_pair = sum_ij w_ij * ( log(1 + exp(u_ij)) - s_ij * u_ij )
    L_quant = sum_i sum_k log cosh( |u_ik| - 1 )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHNLoss(nn.Module):
    """
    DHN Joint Loss: Pairwise Cross-Entropy Similarity + Log-Cosh Quantization Loss.
    """
    def __init__(self, lambda_quant: float = 0.1):
        super(DHNLoss, self).__init__()
        self.lambda_quant = lambda_quant

    def forward(
        self,
        u: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            u: continuous hash representations of shape (B, K), u in (-1, 1).
            labels: 1D class labels of shape (B,).
        Returns:
            Scalar loss tensor.
        """
        B, K = u.shape

        # 1. Similarity matrix S in {0, 1}
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        S = (labels_col == labels_row).float()

        # 2. Pairwise inner product: theta_ij = 0.5 * u_i^T u_j
        inner_prod = torch.matmul(u, u.t())
        theta = 0.5 * inner_prod

        # 3. Class balance weights
        num_positive = S.sum()
        num_total = float(B * B)
        num_negative = num_total - num_positive

        if num_positive > 0 and num_negative > 0:
            c_pos = num_total / num_positive
            c_neg = num_total / num_negative
            w = torch.where(S > 0.5, c_pos, c_neg)
        else:
            w = torch.ones_like(S)

        # 4. Numerically stable pairwise loss
        pair_loss = torch.where(
            S > 0.5,
            F.softplus(-theta),
            F.softplus(theta)
        )
        l_pair = (w * pair_loss).mean()

        # 5. Quantization loss: log cosh(|u| - 1)
        # Note: log(cosh(x)) = |x| + log(1 + exp(-2*|x|)) - log(2)
        dev = torch.abs(u) - 1.0
        # Smooth log-cosh approximation
        l_quant = torch.log(torch.cosh(dev)).mean()

        total_loss = l_pair + self.lambda_quant * l_quant
        return total_loss

