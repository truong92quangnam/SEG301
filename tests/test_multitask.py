"""
Unit and Integration Tests for Multi-task Classification Hashing.
"""

import pytest
import torch
import torch.nn as nn

from src.models.hashnet import HashNetAlexNet
from src.models.dhn import DHNAlexNet
from src.losses.hashnet_loss import HashNetLoss
from src.losses.dhn_loss import DHNLoss


def test_hashnet_multitask_forward():
    """Verify HashNetAlexNet forward with return_logits=True."""
    model = HashNetAlexNet(hash_bits=32, pretrained=False, num_classes=10, cls_source="hash")
    x = torch.randn(4, 3, 224, 224)
    h, z, logits = model(x, alpha=1.2, return_logits=True)

    assert h.shape == (4, 32)
    assert z.shape == (4, 32)
    assert logits.shape == (4, 10)
    assert (h >= -1.0).all() and (h <= 1.0).all()

    # Verify standard backward pass with joint loss
    labels = torch.tensor([0, 1, 2, 3])
    loss_hash = HashNetLoss()(h, labels, alpha=1.2)
    loss_cls = nn.CrossEntropyLoss()(logits, labels)
    loss_total = loss_hash + 1.0 * loss_cls

    loss_total.backward()
    assert model.cls_head.weight.grad is not None
    assert model.hash_head.weight.grad is not None


def test_dhn_multitask_forward():
    """Verify DHNAlexNet forward with return_logits=True."""
    model = DHNAlexNet(hash_bits=16, pretrained=False, num_classes=10, cls_source="hash")
    x = torch.randn(4, 3, 224, 224)
    u, z, logits = model(x, return_logits=True)

    assert u.shape == (4, 16)
    assert z.shape == (4, 16)
    assert logits.shape == (4, 10)
    assert (u >= -1.0).all() and (u <= 1.0).all()

    labels = torch.tensor([0, 1, 2, 3])
    loss_hash = DHNLoss()(u, labels)
    loss_cls = nn.CrossEntropyLoss()(logits, labels)
    loss_total = loss_hash + 1.0 * loss_cls

    loss_total.backward()
    assert model.cls_head.weight.grad is not None
    assert model.hash_head.weight.grad is not None


def test_backward_compatibility():
    """Ensure existing calling convention model(x) still works seamlessly."""
    model_hn = HashNetAlexNet(hash_bits=16, pretrained=False, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    res = model_hn(x)
    assert len(res) == 2  # (h, z)
    assert res[0].shape == (2, 16)

