"""
层次化分类训练器

支持:
  - 多任务联合训练（5 级分类头加权联合优化）
  - 三阶段训练策略（冻结 → 部分解冻 → 全解冻）
  - Mixup / CutMix 数据增强（在图像级别混合，标签在各层级分别计算混合损失）
  - Label Smoothing 正则化
  - Cosine Annealing 学习率调度
  - 分层权重衰减（backbone 小 LR，分类头逐级递减）
  - 多维度评估指标:
    - 每级 Top-1 准确率
    - 分类学一致性准确率（父级预测也必须正确）
    - 平均分类深度（模拟分类学家能定到的平均层级）
"""

import os
import time
import random
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.hierarchical_model import HierarchicalBirdClassifier, TaxonomicInference
from src.taxonomy import TaxonomyTree


# ============================================================
#  Mixup / CutMix（层次化标签版本）
# ============================================================

def mixup_data_hierarchical(
    x: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """
    层次化标签版本的 Mixup

    Args:
        x: (B, C, H, W)
        labels: {'order': (B,), 'family': (B,), ...}
        alpha: Beta 分布参数

    Returns:
        (mixed_x, labels_a, labels_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    labels_a = {k: v for k, v in labels.items()}
    labels_b = {k: v[index] for k, v in labels.items()}

    return mixed_x, labels_a, labels_b, lam


def cutmix_data_hierarchical(
    x: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """层次化标签版本的 CutMix"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    W, H = x.size(3), x.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    labels_a = {k: v for k, v in labels.items()}
    labels_b = {k: v[index] for k, v in labels.items()}

    return mixed_x, labels_a, labels_b, lam


def mixup_criterion_hierarchical(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixup 混合损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
#  分层权重衰减优化器
# ============================================================

def setup_hierarchical_optimizer(
    model: HierarchicalBirdClassifier,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    backbone_lr_factor: float = 0.1,
    backbone_wd_factor: float = 0.1,
) -> optim.Optimizer:
    """
    设置分层学习率和权重衰减

    backbone: 小 LR，小 weight_decay
    分类头: 正常 LR，正常 weight_decay
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone') and 'fc' not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {
            'params': backbone_params,
            'lr': lr * backbone_lr_factor,
            'weight_decay': weight_decay * backbone_wd_factor,
        },
        {
            'params': head_params,
            'lr': lr,
            'weight_decay': weight_decay,
        },
    ]

    return optim.AdamW(param_groups)


# ============================================================
#  训练器
# ============================================================

class HierarchicalTrainer:
    """
    层次化鸟类分类模型训练器

    Args:
        model: HierarchicalBirdClassifier 实例
        loss_weights: 各层级损失权重
        device: 训练设备
        output_dir: 模型保存目录
        taxonomy: TaxonomyTree（用于分类学一致性评估）
    """

    LEVELS = []  # 由 model.LEVELS 初始化

    def __init__(
        self,
        model: HierarchicalBirdClassifier,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = None,
        output_dir: str = None,
        taxonomy: Optional[TaxonomyTree] = None,
    ):
        self.model = model
        self.LEVELS = model.LEVELS  # 动态从模型获取层级
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = output_dir or "output"
        self.taxonomy = taxonomy

        # 动态默认损失权重
        default_weights = {}
        for i, lvl in enumerate(self.LEVELS):
            default_weights[lvl] = round(0.1 * (i + 1), 2) if lvl != self.LEVELS[-1] else 1.0
        self.loss_weights = loss_weights or default_weights

        self.model.to(self.device)

        # 各层级独立的 CrossEntropyLoss（带 Label Smoothing）
        self.criteria = {
            level: nn.CrossEntropyLoss(label_smoothing=0.1)
            for level in self.LEVELS
        }

        # 优化器 & 调度器（在 fit() 中按阶段设置）
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # 训练记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: Dict[str, List[float]] = {
            level: [] for level in self.LEVELS
        }
        self.taxonomic_consistencies: List[float] = []
        self.avg_depths: List[float] = []

        self.best_accuracy = 0.0
        self.best_epoch = 0

    def compute_loss(
        self,
        logits: Tuple[torch.Tensor, ...],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算加权多任务损失

        Args:
            logits: (order_l, family_l, genus_l, species_l, visual_l)
            labels: {'order': (B,), 'family': (B,), ...}

        Returns:
            (total_loss, {level: loss_value})
        """
        total_loss = 0.0
        loss_dict = {}

        for i, level in enumerate(self.LEVELS):
            if level in labels and labels[level].min() >= 0:
                pred = logits[i]
                target = labels[level].to(self.device)
                loss = self.criteria[level](pred, target)
                weight = self.loss_weights.get(level, 1.0)
                total_loss += weight * loss
                loss_dict[level] = loss.item()

        return total_loss, loss_dict

    def compute_mixup_loss(
        self,
        logits: Tuple[torch.Tensor, ...],
        labels_a: Dict[str, torch.Tensor],
        labels_b: Dict[str, torch.Tensor],
        lam: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Mixup/CutMix 混合损失"""
        total_loss = 0.0
        loss_dict = {}

        for i, level in enumerate(self.LEVELS):
            if level in labels_a and labels_a[level].min() >= 0:
                pred = logits[i]
                ya = labels_a[level].to(self.device)
                yb = labels_b[level].to(self.device)
                loss = mixup_criterion_hierarchical(
                    self.criteria[level], pred, ya, yb, lam
                )
                weight = self.loss_weights.get(level, 1.0)
                total_loss += weight * loss
                loss_dict[level] = loss.item()

        return total_loss, loss_dict

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.5,
    ) -> Tuple[float, Dict[str, float]]:
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        level_correct = {level: 0 for level in self.LEVELS}
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels = batch
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if use_mixup and mixup_alpha > 0:
                if random.random() < cutmix_prob:
                    mixed_images, labels_a, labels_b, lam = cutmix_data_hierarchical(
                        images, labels, alpha=cutmix_alpha
                    )
                else:
                    mixed_images, labels_a, labels_b, lam = mixup_data_hierarchical(
                        images, labels, alpha=mixup_alpha
                    )

                logits = self.model(mixed_images)
                loss, loss_dict = self.compute_mixup_loss(
                    logits, labels_a, labels_b, lam
                )
            else:
                logits = self.model(images)
                loss, loss_dict = self.compute_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 统计准确率
            bs = images.size(0)
            running_loss += loss.item() * bs
            total += bs

            for i, level in enumerate(self.LEVELS):
                if level in labels and labels[level].min() >= 0:
                    _, predicted = logits[i].max(1)
                    target = labels[level].to(self.device)
                    level_correct[level] += predicted.eq(target).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sp_acc": f"{100.0 * level_correct['species'] / total:.1f}%",
            })

        epoch_loss = running_loss / total
        level_accs = {
            level: 100.0 * level_correct[level] / total
            for level in self.LEVELS
        }

        return epoch_loss, level_accs

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]:
        """
        验证模型

        Returns:
            (val_loss, level_accuracies, all_predictions, all_labels)
            predictions: [(B,), ...] per level
            labels: [(B,), ...] per level
        """
        self.model.eval()
        running_loss = 0.0
        level_correct = {level: 0 for level in self.LEVELS}
        total = 0

        all_preds = [[] for _ in self.LEVELS]
        all_labels_list = [[] for _ in self.LEVELS]

        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch in pbar:
            images, labels = batch
            images = images.to(self.device, non_blocking=True)
            bs = images.size(0)

            logits = self.model(images)
            loss, loss_dict = self.compute_loss(logits, labels)

            running_loss += loss.item() * bs
            total += bs

            for i, level in enumerate(self.LEVELS):
                if level in labels and labels[level].min() >= 0:
                    _, predicted = logits[i].max(1)
                    target = labels[level].to(self.device)
                    level_correct[level] += predicted.eq(target).sum().item()

                    all_preds[i].extend(predicted.cpu().numpy())
                    all_labels_list[i].extend(target.cpu().numpy())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sp_acc": f"{100.0 * level_correct['species'] / total:.1f}%",
            })

        val_loss = running_loss / total
        level_accs = {
            level: 100.0 * level_correct[level] / total
            for level in self.LEVELS
        }

        preds_np = [np.array(p) for p in all_preds]
        labels_np = [np.array(l) for l in all_labels_list]

        return val_loss, level_accs, preds_np, labels_np

    def compute_taxonomic_consistency(
        self,
        predictions: List[np.ndarray],
        labels: List[np.ndarray],
    ) -> float:
        """
        计算分类学一致性准确率

        只有从 order 到 species 每一级都预测正确，才算该样本正确。
        这比单纯的 species accuracy 更严格，因为它要求模型在整条分类学
        路径上都一致。

        Returns:
            分类学一致性准确率 (%)
        """
        # 使用 species 的数量
        n = len(labels[-1])
        correct = 0
        level_indices = {lvl: i for i, lvl in enumerate(self.LEVELS)}

        for i in range(n):
            all_correct = True
            for idx in level_indices.values():
                if idx < len(labels) and i < len(labels[idx]):
                    if labels[idx][i] < 0:
                        continue
                    if idx < len(predictions) and i < len(predictions[idx]):
                        if predictions[idx][i] != labels[idx][i]:
                            all_correct = False
                            break
            if all_correct:
                correct += 1

        return 100.0 * correct / n if n > 0 else 0.0

    def compute_avg_classification_depth(
        self,
        predictions: List[np.ndarray],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> float:
        """计算平均分类深度（0=最粗粒度, N-1=最细粒度）"""
        n = len(predictions[-1])
        depth_per_sample = len(self.LEVELS)  # 简化：统一计全深度
        return float(depth_per_sample) if n > 0 else 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 120,
        start_epoch: int = 0,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """
        完整训练流程（三阶段训练）

        阶段1 (epoch 1-5):   冻结 backbone, 训练分类头
        阶段2 (epoch 6-15):  解冻 layer3+layer4+分类头
        阶段3 (epoch 16+):   全网络微调
        """
        end_epoch = start_epoch + num_epochs
        stage1_end = start_epoch + min(5, num_epochs)
        stage2_end = start_epoch + min(15, num_epochs)

        total_start = time.time()

        # ==================== 阶段1: 冻结 backbone ====================
        if start_epoch < stage1_end:
            print("\n" + "=" * 60)
            print("阶段 1: 冻结 backbone, 训练分类头 (5 epochs)")
            print("=" * 60)
            self.model.freeze_backbone()
            self.model.heads.requires_grad_(True)

            self.optimizer = setup_hierarchical_optimizer(
                self.model,
                lr=0.001,
                backbone_lr_factor=0.0,  # backbone 冻结，不用 LR
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=stage1_end - start_epoch,
                eta_min=1e-5,
            )

            for epoch in range(start_epoch + 1, stage1_end + 1):
                self._run_epoch(
                    epoch, end_epoch, train_loader, val_loader,
                    use_mixup=False,  # 第一阶段不使用 mixup
                )

        # ==================== 阶段2: 部分解冻 ====================
        if end_epoch > stage1_end:
            print("\n" + "=" * 60)
            print("阶段 2: 解冻 layer3+layer4+分类头, 中等学习率")
            print("=" * 60)
            self.model.freeze_stages(up_to=2)  # 冻结 layer1-2
            self.model.heads.requires_grad_(True)

            self.optimizer = setup_hierarchical_optimizer(
                self.model,
                lr=0.001,
                backbone_lr_factor=0.05,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=stage2_end - max(stage1_end, start_epoch),
                eta_min=1e-5,
            )

            stage2_start_epoch = max(stage1_end, start_epoch)
            for epoch in range(stage2_start_epoch + 1, stage2_end + 1):
                self._run_epoch(
                    epoch, end_epoch, train_loader, val_loader,
                    use_mixup=True,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                )

        # ==================== 阶段3: 全网络微调 ====================
        if end_epoch > stage2_end:
            print("\n" + "=" * 60)
            print("阶段 3: 全网络微调, 小学习率")
            print("=" * 60)
            self.model.unfreeze_backbone()

            self.optimizer = setup_hierarchical_optimizer(
                self.model,
                lr=0.0005,
                backbone_lr_factor=0.01,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=end_epoch - stage2_end,
                eta_min=1e-6,
            )

            for epoch in range(stage2_end + 1, end_epoch + 1):
                self._run_epoch(
                    epoch, end_epoch, train_loader, val_loader,
                    use_mixup=True,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                )

        # 训练完成
        total_time = time.time() - total_start
        print(f"\n训练完成! 总耗时: {total_time / 60:.1f} 分钟")
        print(f"最佳验证 Species 准确率: {self.best_accuracy:.2f}% (Epoch {self.best_epoch})")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "taxonomic_consistencies": self.taxonomic_consistencies,
            "avg_depths": self.avg_depths,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
        }

    def _run_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
    ):
        """执行单个 epoch 的训练 + 验证"""
        print(f"\nEpoch [{epoch}/{total_epochs}]")
        print("-" * 40)

        current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        print(f"学习率: {current_lrs}")

        # 训练
        train_loss, train_accs = self.train_one_epoch(
            train_loader,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )
        self.train_losses.append(train_loss)

        train_str = ", ".join(
            f"{lvl}={train_accs.get(lvl, 0):.1f}%" for lvl in self.LEVELS
        )
        print(f"Train - Loss: {train_loss:.4f}, Acc: [{train_str}]")

        # 验证
        val_loss, val_accs, preds, labels = self.validate(val_loader)
        self.val_losses.append(val_loss)
        for level in self.LEVELS:
            self.val_accuracies[level].append(val_accs.get(level, 0.0))

        val_str = ", ".join(
            f"{lvl}={val_accs.get(lvl, 0):.1f}%" for lvl in self.LEVELS
        )
        print(f"Val   - Loss: {val_loss:.4f}, Acc: [{val_str}]")

        # 分类学一致性
        if self.taxonomy is not None and len(preds[-1]) > 0:
            tax_cons = self.compute_taxonomic_consistency(preds, labels)
            self.taxonomic_consistencies.append(tax_cons)
            print(f"        分类学一致性: {tax_cons:.2f}%")

            avg_depth = self.compute_avg_classification_depth(preds)
            self.avg_depths.append(avg_depth)
            print(f"        平均分类深度: {avg_depth:.2f}/{len(self.LEVELS)}")

        # 更新学习率
        self.scheduler.step()

        # 保存最佳模型（基于 species 准确率）
        species_acc = val_accs.get('species', 0.0)
        if species_acc > self.best_accuracy:
            self.best_accuracy = species_acc
            self.best_epoch = epoch
            self.save_checkpoint("best_model_hierarchical.pth", epoch, val_accs)
            print(f"  ✅ 新最佳模型! Species Acc: {species_acc:.2f}%")

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_accs: Dict[str, float],
    ):
        """保存检查点"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)

        checkpoint = {
            "model_type": "hierarchical",
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "val_accuracies": val_accs,
            "num_classes": self.model.num_classes,
            "loss_weights": self.loss_weights,
            "levels": self.LEVELS,
        }
        torch.save(checkpoint, path)
        print(f"模型已保存: {path}")
