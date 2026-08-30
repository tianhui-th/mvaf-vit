# MVAF-ViT（PyTorch GPU 实现）

本目录整理的是 revision 9 使用的更新模型：空间框局部化器、自动三视图生成、共享 ViT-B/16、三 token 自适应融合，以及 ROI-anchored discrepancy gate（RADG）。推理时只需要完整超声图像；专家框仅用于局部化器和分类器的训练监督，不作为部署输入。

## 文件

- `localizer.py`：ConvNeXt-Tiny 特征金字塔局部化器、GPU 方形裁剪和三视图生成；
- `model.py`：共享 ViT-B/16 的 MVAF-ViT 分类器；
- `pipeline.py`：`AutomaticMVAFViT` 端到端自动流水线；
- `losses.py`：定位损失、主分类损失、ROI 辅助 CE、单向 fused-to-ROI KL 和监督对比损失；
- `__init__.py`：公开导入接口。

代码只依赖 PyTorch 和 torchvision，不导入 `torch_musa`。将模型和输入移动到 `cuda` 后，局部化、`grid_sample` 裁剪和 ViT 前向均在 GPU 上执行。

## 端到端推理

```python
import torch

from mvaf_vit_model import AutomaticMVAFViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutomaticMVAFViT(pretrained=False).to(device).eval()
image = torch.randn(2, 3, 224, 224, device=device)

with torch.inference_mode():
    output = model(image)

print(output["box"].shape)       # [2, 3], normalized (cx, cy, side)
print(output["logits"].shape)    # [2, 2]
print(output["gate_weights"].shape)  # [2, 3]
```

`pretrained=True` 会加载 torchvision 的 ImageNet 权重；离线环境或单元测试使用 `pretrained=False`。输入应为 RGB、ImageNet 归一化后的 `[B, 3, H, W]` 张量。

## 分阶段训练

论文中的训练边界是先训练局部化器，再冻结局部化器训练分类器。对应的最小调用如下：

```python
from mvaf_vit_model import MVAFViT, SpatialBoxLocalizer, classification_loss
from mvaf_vit_model import localization_loss, make_views

# Stage 1: expert target is [center_x, center_y, normalized_square_side].
localizer_output = localizer(images)
loc_loss, loc_items = localization_loss(localizer_output, target_boxes)

# Stage 2: detach the predicted box, then train the classifier objective.
views = make_views(images, localizer_output["box"].detach())
classifier_output = classifier(views)
cls_loss, cls_items = classification_loss(classifier_output, labels)
```

`AutomaticMVAFViT` 默认 `detach_box=True`，因此分类损失不会反向更新局部化器；若只使用冻结框，可直接调用 `model(image, box=predicted_boxes)`。`make_views` 使用 GPU `affine_grid`/`grid_sample`，边界越界的方形框会先向图像内平移，再调整到 224×224。

## 输出

分类器输出 `logits`、`aux_logits` 和归一化 `embedding`；`gate_weights` 是全图、上下文、ROI 三 token 的样本级权重，`discrepancy_weights` 是 RADG 的特征级权重，`calibrated_discrepancy` 和 `interaction` 用于表示分析。自动流水线另外返回 `box`、三个实际输入视图和局部化热图（使用自动局部化时）。
