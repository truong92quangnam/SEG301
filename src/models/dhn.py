"""
Deep Hashing Network (DHN) Architecture.

Paper: "Deep Hashing Network for Efficient Similarity Retrieval" (AAAI 2016)
Backbone: AlexNet (pretrained on ImageNet)
Hash Layer: Fully connected layer mapping 4096-d feature to K bits with tanh activation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class DHNAlexNet(nn.Module):
    """
    AlexNet architecture adapted for Deep Hashing Network (DHN).
    Replaces classifier[6] with a K-bit Hash Head with tanh activation.
    """
    def __init__(self, hash_bits: int, pretrained: bool = True):
        super(DHNAlexNet, self).__init__()
        self.hash_bits = hash_bits

        if pretrained:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            base_model = models.alexnet(weights=weights)
        else:
            base_model = models.alexnet(weights=None)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        self.classifier = nn.Sequential(
            base_model.classifier[0],  # Dropout
            base_model.classifier[1],  # Linear(9216, 4096)
            base_model.classifier[2],  # ReLU
            base_model.classifier[3],  # Dropout
            base_model.classifier[4],  # Linear(4096, 4096)
            base_model.classifier[5],  # ReLU
        )

        self.hash_head = nn.Linear(4096, hash_bits, bias=True)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.hash_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            u: continuous relaxed codes in (-1, 1)^K
            z: raw logits
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)

        z = self.hash_head(feat)
        u = torch.tanh(z)
        return u, z

    def get_binary_code(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts discrete binary codes b in {-1, +1}^K for retrieval inference."""
        _, z = self.forward(x)
        b = torch.where(z >= 0, 1.0, -1.0)
        return b

