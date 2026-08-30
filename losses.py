"""Training losses for the MVAF-ViT classifier and box localizer."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Compute supervised contrastive loss within the current mini-batch."""
    if embedding.ndim != 2 or labels.ndim != 1 or embedding.shape[0] != labels.shape[0]:
        raise ValueError("embedding must be [B, D] and labels must be [B]")
    if embedding.shape[0] < 3:
        return embedding.sum() * 0
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    similarity = embedding @ embedding.t() / temperature
    identity = torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity

    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * ~identity
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0
    mean_log_probability = (
        (log_probability * positive).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_log_probability[valid].mean()


def classification_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    aux_weight: float = 0.20,
    consistency_weight: float = 0.04,
    contrastive_weight: float = 0.03,
    label_smoothing: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Main CE plus ROI auxiliary CE, one-way KL, and SupCon regularization."""
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional class-index tensor")
    if any(weight < 0 for weight in (aux_weight, consistency_weight, contrastive_weight)):
        raise ValueError("loss weights must be non-negative")
    main = F.cross_entropy(output["logits"], labels, label_smoothing=label_smoothing)
    auxiliary = F.cross_entropy(
        output["aux_logits"], labels, label_smoothing=label_smoothing
    )
    teacher_probability = output["logits"].detach().softmax(dim=1)
    consistency = F.kl_div(
        output["aux_logits"].log_softmax(dim=1),
        teacher_probability,
        reduction="batchmean",
    )
    contrastive = supervised_contrastive_loss(output["embedding"], labels)
    total = (
        main
        + aux_weight * auxiliary
        + consistency_weight * consistency
        + contrastive_weight * contrastive
    )
    return total, {
        "main": main.detach(),
        "auxiliary": auxiliary.detach(),
        "consistency": consistency.detach(),
        "contrastive": contrastive.detach(),
    }


def mvaf_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    aux_weight: float = 0.20,
    consistency_weight: float = 0.04,
    contrastive_weight: float = 0.03,
    label_smoothing: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backward-compatible alias for the complete MVAF objective."""
    return classification_loss(
        output,
        labels,
        aux_weight=aux_weight,
        consistency_weight=consistency_weight,
        contrastive_weight=contrastive_weight,
        label_smoothing=label_smoothing,
    )


def square_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU of normalized square boxes represented as ``(cx, cy, side)``."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both have shape [B, 3]")
    pred_half = prediction[:, 2].clamp_min(0) * 0.5
    target_half = target[:, 2].clamp_min(0) * 0.5
    pred_left, pred_right = prediction[:, 0] - pred_half, prediction[:, 0] + pred_half
    pred_top, pred_bottom = prediction[:, 1] - pred_half, prediction[:, 1] + pred_half
    target_left, target_right = target[:, 0] - target_half, target[:, 0] + target_half
    target_top, target_bottom = target[:, 1] - target_half, target[:, 1] + target_half
    intersection = (
        (torch.minimum(pred_right, target_right) - torch.maximum(pred_left, target_left))
        .clamp_min(0)
        * (torch.minimum(pred_bottom, target_bottom) - torch.maximum(pred_top, target_top)).clamp_min(0)
    )
    union = prediction[:, 2].clamp_min(0).square() + target[:, 2].clamp_min(0).square() - intersection
    return intersection / union.clamp_min(1e-6)


def localization_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    heatmap_sigma: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth-L1 + square IoU + optional heatmap cross entropy."""
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must have shape [B, 3] as (cx, cy, side)")
    prediction = output["box"]
    regression = F.smooth_l1_loss(prediction, target, beta=0.05)
    overlap = 1.0 - square_iou(prediction, target).mean()
    total = regression + 0.50 * overlap
    items = {"regression": regression.detach(), "iou": overlap.detach()}

    if "heatmap_logits" in output:
        if heatmap_sigma <= 0:
            raise ValueError("heatmap_sigma must be positive")
        logits = output["heatmap_logits"]
        height, width = logits.shape[-2:]
        y = (torch.arange(height, device=target.device, dtype=target.dtype) + 0.5) / height
        x = (torch.arange(width, device=target.device, dtype=target.dtype) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        distance = (
            (grid_x.unsqueeze(0) - target[:, 0, None, None]).square()
            + (grid_y.unsqueeze(0) - target[:, 1, None, None]).square()
        )
        heatmap_target = torch.exp(-distance / (2.0 * heatmap_sigma**2)).flatten(1)
        heatmap_target = heatmap_target / heatmap_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        heatmap = -(
            heatmap_target * torch.log_softmax(logits.flatten(1), dim=1)
        ).sum(dim=1).mean()
        total = total + 0.10 * heatmap
        items["heatmap"] = heatmap.detach()
    return total, items
