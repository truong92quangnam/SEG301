"""
Compute and compare gradient norms for HashNet 16-bit vs 32-bit:
Shows empirical evidence of continuation saturation and vanishing gradient.
"""
import torch
import numpy as np
from src.models.hashnet import HashNetAlexNet
from src.losses.hashnet_loss import HashNetLoss, ContinuationScheduler
from src.datasets.cifar10_pickle_handler import Cifar10PickleDataset, load_all_batches, get_default_transforms
import json

def analyze_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("outputs/splits/manifest_cifar10_60k_54k_gallery.json", "r") as f:
        manifest = json.load(f)

    train_indices = manifest["splits"]["train_indices"][:500] # mini batch subset
    raw_data, raw_labels = load_all_batches("cifar-10-batches-py")
    _, test_trans = get_default_transforms()
    ds = Cifar10PickleDataset("cifar-10-batches-py", raw_data, raw_labels, train_indices, transform=test_trans)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    images, labels, _ = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    results = {}
    for bits in [16, 32]:
        model = HashNetAlexNet(hash_bits=bits, pretrained=True).to(device)
        criterion = HashNetLoss()
        
        grad_norms = []
        alphas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        for a in alphas:
            model.zero_grad()
            h, z = model(images, alpha=a)
            loss = criterion(h, labels, alpha=a)
            loss.backward()
            
            # Compute gradient norm of hash head weight
            gnorm = model.hash_head.weight.grad.norm().item()
            grad_norms.append(gnorm)
            
        results[bits] = grad_norms
        print(f"Bits {bits}: grad norms across alphas: {grad_norms}")

    return results

if __name__ == "__main__":
    analyze_gradients()
