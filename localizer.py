"""GPU-compatible spatial nodule localizer and multi-scale crop utilities."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny feature extractor used by the spatial localizer."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.norm = model.classifier[0]
        self.out_dim = model.classifier[2].in_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        features = self.norm(self.avgpool(features))
        return torch.flatten(features, 1)


class SpatialBoxLocalizer(nn.Module):
    """ConvNeXt feature pyramid predicting ``(center_x, center_y, side)``.

    The center is estimated by a 56x56 soft-argmax heatmap and the normalized
    square side by attention pooling.  Outputs are normalized to ``(0, 1)``
    and can be consumed directly by :func:`make_views` on the same device.
    """

    def __init__(self, pretrained: bool = True, hidden_dim: int = 128):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.encoder = ConvNeXtEncoder(pretrained=pretrained)
        self.lateral_56 = nn.Conv2d(96, hidden_dim, kernel_size=1)
        self.lateral_28 = nn.Conv2d(192, hidden_dim, kernel_size=1)
        self.lateral_14 = nn.Conv2d(384, hidden_dim, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.side_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.zeros_(self.heatmap_head.bias)
        nn.init.zeros_(self.side_head[-1].weight)
        nn.init.constant_(self.side_head[-1].bias, torch.logit(torch.tensor(0.30)).item())

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected an image tensor shaped [B, 3, H, W], got {tuple(image.shape)}"
            )
        feature_56 = feature_28 = feature_14 = None
        features = image
        for index, layer in enumerate(self.encoder.features):
            features = layer(features)
            if index == 1:
                feature_56 = features
            elif index == 3:
                feature_28 = features
            elif index == 5:
                feature_14 = features
                break
        if feature_56 is None or feature_28 is None or feature_14 is None:
            raise RuntimeError("ConvNeXt feature pyramid stages were not produced")

        fused = self.lateral_56(feature_56)
        fused = fused + F.interpolate(
            self.lateral_28(feature_28), size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = fused + F.interpolate(
            self.lateral_14(feature_14), size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.refine(fused)
        heatmap_logits = self.heatmap_head(fused)
        batch_size, _, height, width = heatmap_logits.shape

        # Keep the coordinate and softmax accumulation in fp32 for stable
        # localization when the surrounding CUDA forward uses autocast.
        attention = torch.softmax(heatmap_logits.float().flatten(1), dim=1)
        y = (torch.arange(height, device=image.device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=image.device, dtype=torch.float32) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        center_x = (attention * grid_x.flatten().unsqueeze(0)).sum(dim=1)
        center_y = (attention * grid_y.flatten().unsqueeze(0)).sum(dim=1)
        pooled = (fused.float().flatten(2) * attention.unsqueeze(1)).sum(dim=2)
        side = self.side_head(pooled).sigmoid().flatten()
        box = torch.stack([center_x, center_y, side], dim=1)
        return {"box": box, "heatmap_logits": heatmap_logits.float()}


def _clamp_center(center: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    lower = half_extent
    upper = 1.0 - half_extent
    return torch.minimum(torch.maximum(center, lower), upper)


def square_crop(
    image: torch.Tensor,
    box: torch.Tensor,
    *,
    scale: float = 1.0,
    output_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Crop normalized square boxes on the input device using ``grid_sample``.

    ``box`` uses normalized ``(center_x, center_y, side)`` coordinates, with
    side normalized by the shorter image dimension.  Boundary-crossing crops
    are shifted inward before resizing, matching the paper's crop protocol.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    if box.ndim != 2 or box.shape != (image.shape[0], 3):
        raise ValueError("box must have shape [B, 3]")
    if scale <= 0:
        raise ValueError("scale must be positive")
    height, width = image.shape[-2:]
    output_size = output_size or (height, width)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("output_size must be a positive integer or (height, width)")

    # The localizer's side is relative to min(H, W), so convert it to pixel
    # extents before deriving the affine sampling matrix.
    side_pixels = (box[:, 2].float() * float(min(height, width)) * scale).clamp(
        min=4.0, max=float(min(height, width))
    )
    half_x = side_pixels / (2.0 * width)
    half_y = side_pixels / (2.0 * height)
    center_x = _clamp_center(box[:, 0].float(), half_x)
    center_y = _clamp_center(box[:, 1].float(), half_y)

    # grid_sample has broader dtype support for fp32 grids.  Casting only this
    # crop operation keeps CUDA half-precision model execution stable while
    # returning the original image dtype to the classifier.
    work_image = image.float()
    theta = torch.zeros(image.shape[0], 2, 3, device=image.device, dtype=torch.float32)
    theta[:, 0, 0] = side_pixels / width
    theta[:, 1, 1] = side_pixels / height
    theta[:, 0, 2] = 2.0 * center_x - 1.0
    theta[:, 1, 2] = 2.0 * center_y - 1.0
    grid = F.affine_grid(theta, (image.shape[0], image.shape[1], *output_size), align_corners=False)
    cropped = F.grid_sample(
        work_image, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return cropped.to(dtype=image.dtype)


def make_views(
    image: torch.Tensor,
    box: torch.Tensor,
    *,
    context_scale: float = 2.0,
    roi_scale: float = 1.2,
    output_size: int | tuple[int, int] = 224,
) -> dict[str, torch.Tensor]:
    """Generate the global, context and lesion-centered classifier views."""
    if isinstance(output_size, int):
        output_shape = (output_size, output_size)
    else:
        output_shape = output_size
    global_view = F.interpolate(image, size=output_shape, mode="bilinear", align_corners=False)
    return {
        "global": global_view,
        "context": square_crop(image, box, scale=context_scale, output_size=output_shape),
        "roi": square_crop(image, box, scale=roi_scale, output_size=output_shape),
    }
