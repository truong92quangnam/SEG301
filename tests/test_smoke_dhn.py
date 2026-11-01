"""
Smoke Test for DHN (Deep Hashing Network) Architecture and Loss.
"""

import pytest
import torch
import torch.optim as optim

from src.models.dhn import DHNAlexNet
from src.losses.dhn_loss import DHNLoss


def test_dhn_forward_and_loss():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DHNAlexNet(hash_bits=16, pretrained=False).to(device)
    criterion = DHNLoss(lambda_quant=0.1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(8, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1, 0, 2, 1, 3, 2, 0], device=device)

    optimizer.zero_grad()
    u, z = model(x)

    assert u.shape == (8, 16)
    assert z.shape == (8, 16)
    assert (u >= -1.0).all() and (u <= 1.0).all()

    loss = criterion(u, labels)
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0

    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    b = model.get_binary_code(x)
    assert ((b == 1.0) | (b == -1.0)).all()

