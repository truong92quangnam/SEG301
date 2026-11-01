"""
Unit tests for Bit-Decorrelation and Balance Regularization.
"""

import pytest
import torch

from src.losses.regularizers import BitDecorrelationAndBalanceLoss


def test_regularizer_shapes_and_finite():
    torch.manual_seed(42)
    reg = BitDecorrelationAndBalanceLoss(lambda_balance=0.1, lambda_decorr=0.1)

    h = torch.randn(32, 16, requires_grad=True)
    loss, l_bal, l_dec = reg(torch.tanh(h))

    assert torch.isfinite(loss).item()
    assert torch.isfinite(l_bal).item()
    assert torch.isfinite(l_dec).item()
    assert loss.item() >= 0.0

    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test_perfect_balance_and_orthogonality():
    # If H is an orthogonal Walsh-Hadamard / sign matrix with zero column sums
    # Balance loss should be 0.0, Decorr loss should be 0.0
    # Create simple 4x2 orthogonal matrix:
    # col 1: [+1, +1, -1, -1] -> sum = 0
    # col 2: [+1, -1, +1, -1] -> sum = 0
    # dot product: 1*1 + 1*(-1) + (-1)*1 + (-1)*(-1) = 1 - 1 - 1 + 1 = 0
    h = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0]
    ], dtype=torch.float32)

    reg = BitDecorrelationAndBalanceLoss(lambda_balance=1.0, lambda_decorr=1.0)
    loss, l_bal, l_dec = reg(h)

    assert abs(l_bal.item()) < 1e-6
    assert abs(l_dec.item()) < 1e-6
    assert abs(loss.item()) < 1e-6


def test_redundant_bits_penalization():
    # If all columns are identical, decorrelation penalty must be large
    h_rand = torch.randn(64, 16)
    # H with identical columns
    single_col = torch.randn(64, 1)
    h_identical = single_col.repeat(1, 16)

    reg = BitDecorrelationAndBalanceLoss(lambda_balance=0.1, lambda_decorr=0.1)
    _, _, dec_rand = reg(torch.tanh(h_rand))
    _, _, dec_ident = reg(torch.tanh(h_identical))

    assert dec_ident.item() > dec_rand.item()


def test_integration_with_hashnet_and_dhn():
    from src.models.hashnet import HashNetAlexNet
    from src.models.dhn import DHNAlexNet
    from src.losses.hashnet_loss import HashNetLoss
    from src.losses.dhn_loss import DHNLoss

    reg = BitDecorrelationAndBalanceLoss(lambda_balance=0.1, lambda_decorr=0.1)

    # 1. HashNet test
    model_hn = HashNetAlexNet(hash_bits=16, pretrained=False)
    criterion_hn = HashNetLoss()
    x = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    h, _ = model_hn(x, alpha=1.0)
    loss_hn = criterion_hn(h, labels, alpha=1.0)
    reg_loss_hn, _, _ = reg(h)
    total_loss_hn = loss_hn + reg_loss_hn
    total_loss_hn.backward()
    assert model_hn.hash_head.weight.grad is not None

    # 2. DHN test
    model_dhn = DHNAlexNet(hash_bits=16, pretrained=False)
    criterion_dhn = DHNLoss(lambda_quant=0.1)
    u, _ = model_dhn(x)
    loss_dhn = criterion_dhn(u, labels)
    reg_loss_dhn, _, _ = reg(u)
    total_loss_dhn = loss_dhn + reg_loss_dhn
    total_loss_dhn.backward()
    assert model_dhn.hash_head.weight.grad is not None

