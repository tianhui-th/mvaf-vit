"""PyTorch implementation of the updated MVAF-ViT classifier.

The classifier consumes three tensors named ``global``, ``context`` and
``roi``.  All views share one ViT-B/16 encoder.  A sample-level gate fuses
the three projected class tokens, and an ROI-anchored discrepancy gate (RADG)
modulates the two absolute ROI-to-view differences before interaction
modeling.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class TorchvisionEncoder(nn.Module):
    """ViT-B/16 feature extractor with the ImageNet head removed.

    The forward path mirrors torchvision's implementation and returns the
    768-dimensional class token.  Keeping the encoder self-contained avoids a
    dependency on the repository's MUSA-specific model module and works with
    both CUDA and CPU PyTorch builds.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.encoder = model.encoder
        self.seq_length = model.seq_length
        self.out_dim = model.hidden_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected an image tensor shaped [B, 3, H, W], got {tuple(image.shape)}"
            )
        tokens = self.conv_proj(image)
        tokens = tokens.reshape(tokens.shape[0], self.out_dim, -1).permute(0, 2, 1)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        return self.encoder(tokens)[:, 0]


class MVAFViT(nn.Module):
    """Updated MVAF-ViT three-view classifier.

    The preferred call is ``model({"global": ..., "context": ..., "roi": ...})``.
    For compatibility with the earlier standalone example, positional calls
    ``model(global_view, context_view, roi_view)`` are also accepted.
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        num_classes: int = 2,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.encoder = TorchvisionEncoder(pretrained=pretrained)
        self.view_projector = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.discrepancy_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # A bias of 2.0 starts RADG close to pass-through while retaining a
        # learnable feature-level attenuation during optimization.
        nn.init.constant_(self.discrepancy_gate[-1].bias, 2.0)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _unpack_views(
        batch_or_global: Mapping[str, torch.Tensor] | torch.Tensor,
        context_view: torch.Tensor | None,
        roi_view: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_or_global, Mapping):
            if context_view is not None or roi_view is not None:
                raise TypeError("Do not pass positional views together with a view mapping")
            try:
                return (
                    batch_or_global["global"],
                    batch_or_global["context"],
                    batch_or_global["roi"],
                )
            except KeyError as error:
                raise KeyError("The view mapping must contain global, context and roi") from error
        if context_view is None or roi_view is None:
            raise TypeError("Expected a view mapping or three positional view tensors")
        return batch_or_global, context_view, roi_view

    def forward(
        self,
        batch_or_global: Mapping[str, torch.Tensor] | torch.Tensor,
        context_view: torch.Tensor | None = None,
        roi_view: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        global_view, context_view, roi_view = self._unpack_views(
            batch_or_global, context_view, roi_view
        )
        if global_view.ndim != 4 or context_view.ndim != 4 or roi_view.ndim != 4:
            raise ValueError("All views must be 4-D tensors shaped [B, 3, H, W]")
        batch_size = roi_view.shape[0]
        if not (global_view.shape[0] == context_view.shape[0] == batch_size):
            raise ValueError("The three views must have the same batch size")

        views = torch.cat([global_view, context_view, roi_view], dim=0)
        encoded = self.view_projector(self.encoder(views))
        global_token, context_token, roi_token = encoded.split(batch_size, dim=0)

        tokens = torch.stack([global_token, context_token, roi_token], dim=1)
        view_weights = self.gate(
            torch.cat([global_token, context_token, roi_token], dim=1)
        ).softmax(dim=1)
        fused_token = (tokens * view_weights.unsqueeze(-1)).sum(dim=1)

        roi_context_difference = (roi_token - context_token).abs()
        roi_global_difference = (roi_token - global_token).abs()
        discrepancy_weights = self.discrepancy_gate(
            torch.cat([roi_token, roi_context_difference, roi_global_difference], dim=1)
        ).view(batch_size, 2, -1).sigmoid()
        calibrated_context_difference = discrepancy_weights[:, 0] * roi_context_difference
        calibrated_global_difference = discrepancy_weights[:, 1] * roi_global_difference
        interaction_token = self.interaction(
            torch.cat(
                [roi_token, calibrated_context_difference, calibrated_global_difference], dim=1
            )
        )
        final_feature = torch.cat([fused_token, interaction_token], dim=1)

        return {
            "logits": self.classifier(final_feature),
            "aux_logits": self.aux_classifier(roi_token),
            "embedding": F.normalize(fused_token, dim=1),
            # These tensors are useful for representation analysis and do not
            # add a separate computation path.
            "fused_token": fused_token,
            "final_feature": final_feature,
            "gate_weights": view_weights,
            "roi_token": roi_token,
            "calibrated_discrepancy": torch.cat(
                [calibrated_context_difference, calibrated_global_difference], dim=1
            ),
            "interaction": interaction_token,
            "discrepancy_weights": discrepancy_weights,
        }


def build_model(
    name: str = "mvaf_vit",
    *,
    pretrained: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.25,
    num_classes: int = 2,
) -> MVAFViT:
    """Build the updated classifier by its public model name."""
    if name != "mvaf_vit":
        raise ValueError(f"Unknown model {name!r}; this package exposes only 'mvaf_vit'")
    return MVAFViT(
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
