"""
HashNet Architecture and Quantization Layers.

Paper: "HashNet: Deep Learning to Hash by Continuation" (ICCV 2017)
Backbone: AlexNet (pretrained on ImageNet)
Hash Head: Fully-Connected layer mapping 4096-d feature representations to K bits.
Continuation: tanh(alpha * z) where alpha increases over training iterations.
Multi-task Classification Head: Optional classification head to jointly optimize
cross-entropy loss with hash pairwise loss, preventing gradient vanishing and
Hamming space sparsity especially at higher bit lengths (e.g. 32-bit).
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class HashNetAlexNet(nn.Module):
    """
    AlexNet architecture adapted for HashNet with optional Multi-task Classification Head.
    Replaces classifier[6] (1000 classes) with a Hash Head (K bits).
    Optionally adds a classification head for multi-task learning.
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
            pretrained: Whether to load ImageNet-1K pretrained weights for AlexNet backbone.
            num_classes: Number of target classes for multi-task classification. If None or <= 0,
                         classification branch is disabled.
            cls_source: 'hash' (Linear(hash_bits -> num_classes)) or
                        'feature' (Linear(4096 -> num_classes)).
        """
        super(HashNetAlexNet, self).__init__()
        self.hash_bits = hash_bits
        self.num_classes = num_classes
        self.cls_source = cls_source
        
        # Load AlexNet backbone
        if pretrained:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            base_model = models.alexnet(weights=weights)
        else:
            base_model = models.alexnet(weights=None)
            
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Keep fc6 and fc7, replace fc8
        self.classifier = nn.Sequential(
            base_model.classifier[0], # Dropout
            base_model.classifier[1], # Linear(9216, 4096)
            base_model.classifier[2], # ReLU
            base_model.classifier[3], # Dropout
            base_model.classifier[4], # Linear(4096, 4096)
            base_model.classifier[5], # ReLU
        )
        
        # Hash Head (fc8 replacement)
        self.hash_head = nn.Linear(4096, hash_bits, bias=True)
        # Initialize hash_head weights using normal distribution
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
        alpha: float = 1.0,
        return_logits: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, 3, 224, 224).
            alpha: Continuation parameter (>= 1.0).
            return_logits: If True, also returns classification logits.
            
        Returns:
            If return_logits is False:
                continuous_code: h = tanh(alpha * z) in range (-1, 1).
                raw_logits: z = hash_head(features).
            If return_logits is True:
                continuous_code: h.
                raw_logits: z.
                cls_logits: classification logits (B, num_classes).
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)
        
        z = self.hash_head(feat)
        h = torch.tanh(alpha * z)

        if return_logits:
            if self.cls_head is None:
                raise ValueError("cls_head is not configured in this HashNet model.")
            cls_in = h if self.cls_source == "hash" else feat
            cls_logits = self.cls_head(cls_in)
            return h, z, cls_logits

        return h, z

    def get_binary_code(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts discrete binary codes b in {-1, +1}^K for retrieval inference.
        """
        _, z = self.forward(x, alpha=1.0)
        # Sign function: +1 if z >= 0 else -1
        b = torch.where(z >= 0, 1.0, -1.0)
        return b


