"""
Smoke Test for HashNet End-to-End Pipeline (Debug Only).

Validates:
- Forward pass with continuous and raw output
- Backward pass and gradient computation
- Finite loss calculation with HashNetLoss
- ContinuationScheduler step updates
- Code binarization (output contains strictly {-1, +1})
- Checkpoint save and load
- Retrieval and mAP evaluation on small synthetic debug split
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.optim as optim

from src.models.hashnet import HashNetAlexNet
from src.losses.hashnet_loss import HashNetLoss, ContinuationScheduler
from src.evaluation.metrics import ChunkedEvaluator


def test_hashnet_forward_and_backward():
    """Checks forward pass, backward gradients, and finite loss."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tiny model without pretrained weights for speed
    model = HashNetAlexNet(hash_bits=16, pretrained=False).to(device)
    criterion = HashNetLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Synthetic batch: 8 images, shape (8, 3, 224, 224)
    x = torch.randn(8, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1, 0, 2, 1, 3, 2, 0], device=device)

    alpha = 1.5
    optimizer.zero_grad()
    h, z = model(x, alpha=alpha)

    # Assert shapes
    assert h.shape == (8, 16), f"Expected (8, 16), got {h.shape}"
    assert z.shape == (8, 16), f"Expected (8, 16), got {z.shape}"
    # Assert continuous values strictly in (-1, 1)
    assert (h >= -1.0).all() and (h <= 1.0).all()

    # Loss computation
    loss = criterion(h, labels, alpha=alpha)
    assert torch.isfinite(loss).item(), "Loss is NaN or infinite!"
    assert loss.item() > 0.0

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all().item(), f"Gradient non-finite for {name}"

    optimizer.step()


def test_binary_code_strictly_binary():
    """Checks that binarization strictly outputs values in {-1, +1}."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HashNetAlexNet(hash_bits=16, pretrained=False).to(device)
    model.eval()

    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        b = model.get_binary_code(x)

    assert b.shape == (4, 16)
    unique_vals = torch.unique(b).cpu().numpy().tolist()
    for val in unique_vals:
        assert val in [-1.0, 1.0], f"Found non-binary value {val} in binary code!"


def test_continuation_scheduler():
    """Checks continuation schedule alpha_t = sqrt(1 + gamma * t)."""
    sched = ContinuationScheduler(gamma=0.001)
    assert sched.get_alpha(0) == 1.0
    # Step 1000: sqrt(1 + 0.001 * 1000) = sqrt(2.0) = 1.4142...
    assert pytest.approx(sched.get_alpha(1000), abs=1e-4) == np.sqrt(2.0)
    assert sched.get_alpha(2000) > sched.get_alpha(1000)


def test_checkpoint_save_and_load():
    """Verifies checkpoint saving and loading restores identical weights."""
    model1 = HashNetAlexNet(hash_bits=16, pretrained=False)
    model2 = HashNetAlexNet(hash_bits=16, pretrained=False)

    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(model1.state_dict(), tmp.name)
        model2.load_state_dict(torch.load(tmp.name, weights_only=True))

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model parameters mismatch after loading checkpoint!"


def test_end_to_end_smoke_retrieval():
    """
    Simulates end-to-end forward -> binary code extraction -> retrieval -> mAP evaluation.
    Labeled strictly as 'debug only'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HashNetAlexNet(hash_bits=16, pretrained=False).to(device)
    model.eval()

    N_q = 10
    N_d = 40
    # Dummy query and database
    q_imgs = torch.randn(N_q, 3, 224, 224, device=device)
    d_imgs = torch.randn(N_d, 3, 224, 224, device=device)
    q_labels = torch.randint(0, 5, (N_q,))
    d_labels = torch.randint(0, 5, (N_d,))

    with torch.no_grad():
        q_codes = model.get_binary_code(q_imgs)
        d_codes = model.get_binary_code(d_imgs)

    evaluator = ChunkedEvaluator(topk=N_d, chunk_size=5, device="cpu")
    results = evaluator.evaluate(q_codes.cpu(), d_codes.cpu(), q_labels, d_labels)

    assert "mAP" in results
    assert 0.0 <= results["mAP"] <= 1.0
    assert len(results["AP_list"]) == N_q

