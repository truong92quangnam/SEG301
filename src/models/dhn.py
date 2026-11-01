"""
Deep Hashing Network (DHN) Architecture.

Paper: "Deep Hashing Network for Efficient Similarity Retrieval" (AAAI 2016)
Backbone: AlexNet (pretrained on ImageNet)
Hash Layer: Fully connected layer mapping 4096-d feature to K bits with tanh activation.
Multi-task Classification Head: Optional classification head to jointly optimize
cross-entropy loss with DHN pairwise and quantization loss.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models


class DHNAlexNet(nn.Module):
    """
    AlexNet architecture adapted for Deep Hashing Network (DHN).
    Replaces classifier[6] with a K-bit Hash Head with tanh activation.
    Optionally includes a multi-task classification head.
    """
    def __init__(
        self,
        hash_bits: int,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        cls_source: str = "hash",
    ):
        """
        Args:
            hash_bits: Number of binary hash bits K.
            pretrained: Whether to load ImageNet-1K pretrained weights.
            num_classes: Number of classes for multi-task classification. If None, disabled.
            cls_source: 'hash' (Linear(hash_bits -> num_classes)) or
                        'feature' (Linear(4096 -> num_classes)).
        """
        super(DHNAlexNet, self).__init__()
        self.hash_bits = hash_bits
        self.num_classes = num_classes
        self.cls_source = cls_source

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

        # Multi-task Classification Head
        if self.num_classes is not None and self.num_classes > 0:
            in_dim = hash_bits if cls_source == "hash" else 4096
            self.cls_head = nn.Linear(in_dim, self.num_classes, bias=True)
            nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.cls_head.bias, 0.0)
        else:
            self.cls_head = None

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            If return_logits is False:
                u: continuous relaxed codes in (-1, 1)^K
                z: raw logits
            If return_logits is True:
                u: continuous relaxed codes.
                z: raw logits.
                cls_logits: classification logits (B, num_classes).
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)

        z = self.hash_head(feat)
        u = torch.tanh(z)

        if return_logits:
            if self.cls_head is None:
                raise ValueError("cls_head is not configured in this DHN model.")
            cls_in = u if self.cls_source == "hash" else feat
            cls_logits = self.cls_head(cls_in)
            return u, z, cls_logits

        return u, z

    def get_binary_code(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts discrete binary codes b in {-1, +1}^K for retrieval inference."""
        _, z = self.forward(x)
        b = torch.where(z >= 0, 1.0, -1.0)
        return b

