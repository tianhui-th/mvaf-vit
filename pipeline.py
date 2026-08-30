"""End-to-end automatic MVAF-ViT pipeline."""

from __future__ import annotations

import torch
from torch import nn

from .localizer import SpatialBoxLocalizer, make_views
from .model import MVAFViT


class AutomaticMVAFViT(nn.Module):
    """Localizer plus the updated three-view MVAF-ViT classifier.

    The paper trains the localizer and classifier in separate stages because
    its crop operation is discrete.  ``detach_box=True`` reproduces that
    boundary in this GPU implementation: classification gradients do not
    update the localizer.  Pass a precomputed ``box`` to run only the
    classifier path while evaluating a frozen localizer.
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        context_scale: float = 2.0,
        roi_scale: float = 1.2,
        output_size: int = 224,
        detach_box: bool = True,
    ):
        super().__init__()
        self.localizer = SpatialBoxLocalizer(pretrained=pretrained)
        self.classifier = MVAFViT(
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.context_scale = context_scale
        self.roi_scale = roi_scale
        self.output_size = output_size
        self.detach_box = detach_box

    def forward(
        self,
        image: torch.Tensor,
        box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        localization: dict[str, torch.Tensor] = {}
        if box is None:
            localization = self.localizer(image)
            box = localization["box"]
        else:
            if box.shape != (image.shape[0], 3):
                raise ValueError("box must have shape [B, 3]")
            box = box.to(device=image.device)
        crop_box = box.detach() if self.detach_box else box
        views = make_views(
            image,
            crop_box,
            context_scale=self.context_scale,
            roi_scale=self.roi_scale,
            output_size=self.output_size,
        )
        output = self.classifier(views)
        output.update(localization)
        output["box"] = box
        output["global_view"] = views["global"]
        output["context_view"] = views["context"]
        output["roi_view"] = views["roi"]
        return output
