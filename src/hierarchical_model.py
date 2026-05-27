"""
层次化鸟类分类模型

共享 ResNet backbone + N 个级联分类头，
支持 3 级 CUB-Hierarchy: Order(目) → Family(科) → Species(种)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from src.taxonomy import TaxonomyTree


def _build_backbone(name: str):
    """构建 backbone，返回 (module, feature_dim)"""
    if name == "resnet101":
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        return m, 2048
    elif name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        return m, 2048
    else:
        raise ValueError(f"未知 backbone: {name}，可选: resnet50/resnet101")


class CascadeHead(nn.Module):
    """单个级联分类头: BN → Dropout → Linear"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.drop(x)
        return self.fc(x)


class HierarchicalHeads(nn.Module):
    """动态级联式层次化分类头（nn.ModuleList）"""

    def __init__(
        self,
        feature_dim: int,
        level_names: List[str],
        num_classes: List[int],
        dropout: float = 0.5,
        cascade: bool = True,
    ):
        super().__init__()
        self.level_names = level_names
        self.cascade = cascade
        self.heads = nn.ModuleList()
        for i, n_cls in enumerate(num_classes):
            in_dim = feature_dim if (i == 0 or not cascade) else feature_dim + num_classes[i - 1]
            self.heads.append(CascadeHead(in_dim, n_cls, dropout))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for i, head in enumerate(self.heads):
            inp = features if (i == 0 or not self.cascade) else torch.cat([features, outputs[-1]], dim=1)
            outputs.append(head(inp))
        return outputs


# ============================================================
#  完整层次化分类模型
# ============================================================

class HierarchicalBirdClassifier(nn.Module):
    """层次化鸟类分类器（动态层级 + 可选 backbone）"""

    def __init__(self, level_names: List[str], num_classes: Dict[str, int],
                 backbone_name: str = "resnet101",
                 cascade: bool = True, dropout: float = 0.5):
        super().__init__()
        self.LEVELS = level_names
        self.num_classes = num_classes
        self.cascade = cascade
        self.backbone_name = backbone_name

        self.backbone, feature_dim = _build_backbone(backbone_name)

        cls_list = [num_classes.get(lvl, 1) for lvl in level_names]
        self.heads = HierarchicalHeads(feature_dim, level_names, cls_list, dropout, cascade)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.heads(self.backbone(x))

    def forward_with_features(self, x: torch.Tensor):
        features = self.backbone(x)
        return features, self.heads(features)

    def predict_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {lvl: F.softmax(logits[i], dim=1) for i, lvl in enumerate(self.LEVELS)}

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("已冻结 backbone")

    def freeze_stages(self, up_to: int = 2):
        self.freeze_backbone()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.heads.parameters():
            p.requires_grad = True
        print("已解冻所有参数")


# ============================================================
#  分类学思维推理引擎
# ============================================================

class TaxonomicInference:
    """动态层级推理: 逐级阈值判断，不确定就停止"""

    def __init__(self, model: HierarchicalBirdClassifier, taxonomy: TaxonomyTree,
                 thresholds: Optional[Dict[str, float]] = None, device: str = "cpu"):
        self.model = model
        self.taxonomy = taxonomy
        self.device = device
        self.thresholds = thresholds or {lvl: 0.5 for lvl in model.LEVELS}

    @torch.no_grad()
    def predict(self, image: torch.Tensor, top_k_per_level: int = 3) -> Dict[str, Any]:
        self.model.eval()
        image = image.to(self.device)
        logits_list = self.model(image)

        result_path = []
        stopped_at = self.model.LEVELS[-1]
        stopped_reason = 'completed'
        all_confident = True

        for i, level in enumerate(self.model.LEVELS):
            probs = F.softmax(logits_list[i], dim=1)
            max_prob, top_idx = probs.max(dim=1)

            topk_p, topk_i = probs.topk(min(top_k_per_level, probs.size(1)), dim=1)
            topk_list = [{'name': self._name(level, topk_i[0, j].item()),
                          'confidence': round(topk_p[0, j].item() * 100, 2)}
                         for j in range(topk_i.size(1))]

            passed = max_prob.item() >= self.thresholds.get(level, 0.5)

            result_path.append({
                'level': level, 'name': self._name(level, top_idx.item()),
                'class_id': top_idx.item(), 'confidence': round(max_prob.item() * 100, 2),
                'top_k': topk_list, 'passed_threshold': passed,
            })

            if not passed:
                stopped_at, stopped_reason, all_confident = level, 'below_threshold', False
                break

        taxonomy_str = " → ".join(
            f"{r['name']} ({r['confidence']:.0f}%)"
            for r in result_path
            if not (r['level'] == stopped_at and stopped_reason == 'below_threshold')
        )
        return {'stopped_at': stopped_at, 'stopped_reason': stopped_reason,
                'path': result_path, 'full_taxonomy_string': taxonomy_str,
                'is_confident': all_confident,
                'top_prediction': result_path[-1] if result_path else None}

    def _name(self, level: str, cid: int) -> str:
        names = self.taxonomy.level_names.get(level, [])
        return names[cid] if 0 <= cid < len(names) else f"{level}_{cid}"

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, top_k_per_level: int = 3):
        return [self.predict(images[i:i + 1], top_k_per_level) for i in range(images.size(0))]


# ============================================================
#  工具函数
# ============================================================

def build_hierarchical_model(
    taxonomy: TaxonomyTree,
    backbone_name: str = "resnet101",
    cascade: bool = True,
    dropout: float = 0.5,
) -> HierarchicalBirdClassifier:
    """
    基于 TaxonomyTree 构建层次化分类模型

    Args:
        taxonomy: 已解析的分类学树
        use_se: 是否启用 SE 注意力
        cascade: 是否使用级联分类头
        dropout: Dropout 概率

    Returns:
        HierarchicalBirdClassifier 实例
    """
    nums = taxonomy.num_levels()
    # 从 taxonomy.LEVEL_ORDER 动态获取实际层级（排除 root 和 0 类别的层级）
    levels = [lvl for lvl in taxonomy.LEVEL_ORDER
              if lvl != 'root' and nums.get(lvl, 0) > 0]
    num_classes = {lvl: nums.get(lvl, 0) for lvl in levels}
    print(f"构建层次化模型 - 层级: {'→'.join(levels)}")
    print(f"  各类别数: {num_classes}")

    model = HierarchicalBirdClassifier(
        level_names=levels,
        num_classes=num_classes,
        backbone_name=backbone_name,
        cascade=cascade,
        dropout=dropout,
    )
    return model


if __name__ == "__main__":
    # 快速测试
    print("测试层次化模型...")

    # 模拟 CUB 3 级
    model = HierarchicalBirdClassifier(
        level_names=['order', 'family', 'species'],
        num_classes={'order': 13, 'family': 38, 'species': 200},
        use_se=True,
    )

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)  # List[Tensor]
    for i, lvl in enumerate(model.LEVELS):
        print(f"{lvl} logits: {logits[i].shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}, 可训练: {trainable:,}")
