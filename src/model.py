"""
ResNet50 模型构建与训练模块

基于 torchvision 预训练的 ResNet50, 替换最后的全连接层以适应 CUB-200-2011 的 200 类分类任务。
改进版本:
  - SE-Net 通道注意力机制增强特征提取
  - 改进分类头 (BN + Dropout + 两层FC)
  - Label Smoothing 正则化
  - Mixup / CutMix 数据增强
  - Cosine Annealing 学习率调度
  - 测试时增强 (TTA)
  - 分层权重衰减策略
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from tqdm import tqdm

from src import config


# ============================================================
#  SE-Net 通道注意力模块
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    通过全局平均池化 -> FC -> ReLU -> FC -> Sigmoid 的路径,
    自适应地学习每个通道的重要性权重, 增强有用特征、抑制冗余特征。

    Args:
        channels: 输入通道数
        reduction: 压缩比 (默认16, 即中间层通道数 = channels // reduction)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)  # 保证最少8通道
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


def _add_se_to_residual_block(block: nn.Module, reduction: int = 16):
    """
    给 ResNet 的 BasicBlock / Bottleneck 添加 SE 注意力

    在残差分支的最后一个卷积之后、shortcut 相加之前插入 SE 模块。
    """
    # 获取输出通道数
    if hasattr(block, 'conv3'):
        # Bottleneck (ResNet50/101/152)
        out_channels = block.conv3.out_channels
    elif hasattr(block, 'conv2'):
        # BasicBlock (ResNet18/34)
        out_channels = block.conv2.out_channels
    else:
        return block

    block.se = SEBlock(out_channels, reduction=reduction)

    # 保存原始 forward
    original_forward = block.forward

    def new_forward(*args, **kwargs):
        # 调用原始 forward, 但在残差相加前插入 SE
        # 这里需要重新实现 forward 逻辑
        pass

    # 更简洁的方式: 替换 Bottleneck 的 forward
    if isinstance(block, models.resnet.Bottleneck):
        _patch_bottleneck_forward(block)
    elif isinstance(block, models.resnet.BasicBlock):
        _patch_basicblock_forward(block)

    return block


def _patch_bottleneck_forward(block: nn.Module):
    """替换 Bottleneck 的 forward 方法, 在残差相加前插入 SE"""

    def se_forward(x):
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

        # SE attention
        out = block.se(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        out = block.relu(out)

        return out

    block.forward = se_forward


def _patch_basicblock_forward(block: nn.Module):
    """替换 BasicBlock 的 forward 方法, 在残差相加前插入 SE"""

    def se_forward(x):
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)

        # SE attention
        out = block.se(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        out = block.relu(out)

        return out

    block.forward = se_forward


# ============================================================
#  ResNet50 + SE 鸟类分类模型
# ============================================================

class ResNet50BirdClassifier(nn.Module):
    """
    基于 ResNet50 + SE-Net 的鸟类分类模型

    改进点:
    1. 在每个残差块后添加 SE 通道注意力
    2. 改进分类头: BN -> Dropout -> FC -> ReLU -> Dropout -> FC
    3. 使用 ImageNet 预训练权重进行迁移学习
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True,
                 use_se: bool = True, se_reduction: int = 16):
        super().__init__()

        self.use_se = use_se

        # 加载预训练 ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # 添加 SE 注意力模块
        if self.use_se:
            self._add_se_blocks(reduction=se_reduction)

        # 替换最后的全连接层 (改进版: BN + Dropout + 两层FC)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes),
        )

        # 初始化新增层的权重
        self._init_weights()

    def _add_se_blocks(self, reduction: int = 16):
        """给 ResNet50 的所有残差块添加 SE 注意力"""
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.backbone, name)
            for block in layer:
                _add_se_to_residual_block(block, reduction=reduction)
        print(f"已添加 SE-Net 注意力模块 (reduction={reduction})")

    def _init_weights(self):
        """初始化新增全连接层的权重"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """冻结 backbone 参数, 仅训练全连接层 (用于第一阶段微调)"""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("已冻结 backbone 参数, 仅训练分类头")

    def unfreeze_backbone(self):
        """解冻所有参数 (用于第二阶段微调)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("已解冻所有参数, 进行全网络微调")


# ============================================================
#  Mixup / CutMix 数据增强
# ============================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.4) -> tuple:
    """
    Mixup 数据增强

    将两张图像按比例混合: x_mixed = lambda * x_i + (1 - lambda) * x_j
    对应标签也按比例混合。

    Args:
        x: 输入图像 batch
        y: 标签 batch
        alpha: Beta 分布参数, lambda ~ Beta(alpha, alpha)

    Returns:
        (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0) -> tuple:
    """
    CutMix 数据增强

    从一张图像中裁剪矩形区域, 粘贴到另一张图像上。
    相比 Mixup, CutMix 保留了更多局部特征信息。

    Args:
        x: 输入图像 batch, (B, C, H, W)
        y: 标签 batch
        alpha: Beta 分布参数

    Returns:
        (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # 生成随机裁剪区域
    W, H = x.size(3), x.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 计算裁剪边界
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # 混合图像
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 根据面积比调整 lambda
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix 的混合损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
#  测试时增强 (TTA)
# ============================================================

class TestTimeAugmentation:
    """
    测试时增强 (Test-Time Augmentation)

    对同一张测试图像进行多次不同的增强变换 (多尺度裁剪、水平翻转等),
    然后对预测概率取平均, 提升推理准确率。

    Args:
        model: 训练好的模型
        device: 推理设备
        scales: 多尺度裁剪的缩放比例列表
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 scales: list = None):
        self.model = model
        self.device = device
        self.scales = scales or config.TTA_SCALES

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        对一个 batch 的图像进行 TTA 预测

        Args:
            images: (B, C, H, W), 已归一化的图像

        Returns:
            avg_probs: (B, num_classes), 平均预测概率
        """
        self.model.eval()
        B, C, H, W = images.shape
        all_probs = []

        for scale in self.scales:
            # 多尺度裁剪
            new_size = int(max(H, W) * scale)
            if abs(scale - 1.0) > 0.01:
                scaled = F.interpolate(images, size=(new_size, new_size),
                                       mode='bilinear', align_corners=False)
                # 中心裁剪回原始尺寸
                if new_size > H:
                    start = (new_size - H) // 2
                    scaled = scaled[:, :, start:start + H, start:start + W]
                else:
                    # 如果缩小了, 需要padding
                    pad_h = (H - new_size) // 2
                    pad_w = (W - new_size) // 2
                    scaled = F.pad(scaled, (pad_w, W - new_size - pad_w,
                                            pad_h, H - new_size - pad_h),
                                   value=0)
            else:
                scaled = images

            # 原图预测
            outputs = self.model(scaled.to(self.device))
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)

            # 水平翻转预测
            flipped = torch.flip(scaled, dims=[3])
            outputs_flip = self.model(flipped.to(self.device))
            probs_flip = F.softmax(outputs_flip, dim=1)
            all_probs.append(probs_flip)

        # 取平均
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs


# ============================================================
#  训练器
# ============================================================

class Trainer:
    """
    模型训练器 (改进版)

    改进点:
    1. Label Smoothing 正则化
    2. Mixup / CutMix 数据增强
    3. Cosine Annealing 学习率调度
    4. 分层权重衰减 (backbone 小, 分类头大)
    5. 测试时增强 (TTA)
    6. 三阶段训练策略 (冻结 -> 部分解冻 -> 全解冻)

    Args:
        model: ResNet50BirdClassifier 实例
        device: 训练设备 (cuda / cpu)
        output_dir: 模型保存目录
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        output_dir: str = None,
    ):
        self.model = model
        self.device = device or config.DEVICE
        self.output_dir = output_dir or config.OUTPUT_DIR

        # 确定实际可用设备
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用, 自动切换到 CPU")
            self.device = "cpu"

        self.model.to(self.device)

        # 损失函数 (带 Label Smoothing)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.LABEL_SMOOTHING
        )

        # 分层权重衰减优化器
        self._setup_optimizer()

        # 学习率调度器
        self._setup_scheduler()

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def _setup_optimizer(self):
        """设置分层权重衰减的优化器"""
        # backbone 参数使用较小的 weight_decay
        # 分类头参数使用正常的 weight_decay
        backbone_params = []
        fc_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name or 'se' in name:
                fc_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': backbone_params, 'weight_decay': config.BACKBONE_WEIGHT_DECAY,
             'lr': config.LEARNING_RATE * 0.1},
            {'params': fc_params, 'weight_decay': config.WEIGHT_DECAY,
             'lr': config.LEARNING_RATE},
        ]

        self.optimizer = optim.AdamW(param_groups)

    def _setup_scheduler(self):
        """设置学习率调度器"""
        if config.USE_COSINE_LR:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=1e-6,
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.LR_STEP_SIZE,
                gamma=config.LR_GAMMA,
            )

    def train_one_epoch(self, train_loader, use_mixup: bool = True) -> float:
        """训练一个 epoch (支持 Mixup/CutMix)"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播 (可能使用 Mixup/CutMix)
            self.optimizer.zero_grad()

            if use_mixup and config.MIXUP_ALPHA > 0:
                # 随机选择 Mixup 或 CutMix
                if random.random() < config.CUTMIX_PROB:
                    mixed_images, y_a, y_b, lam = cutmix_data(
                        images, labels, alpha=config.CUTMIX_ALPHA
                    )
                else:
                    mixed_images, y_a, y_b, lam = mixup_data(
                        images, labels, alpha=config.MIXUP_ALPHA
                    )
                outputs = self.model(mixed_images)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.1f}%",
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader, use_tta: bool = False) -> tuple:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器
            use_tta: 是否使用测试时增强

        Returns:
            (val_loss, val_accuracy, all_predictions, all_labels)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # TTA
        tta_handler = None
        if use_tta and config.USE_TTA:
            tta_handler = TestTimeAugmentation(self.model, self.device)

        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if tta_handler is not None:
                # TTA 预测
                probs = tta_handler.predict(images)
                _, predicted = probs.max(1)
                # 计算 loss 用原始输出
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.1f}%",
            })

        val_loss = running_loss / total
        val_acc = 100.0 * correct / total
        return val_loss, val_acc, np.array(all_preds), np.array(all_labels)

    def fit(self, train_loader, val_loader, num_epochs: int = None, start_epoch: int = 0) -> dict:
        """
        完整训练流程 (三阶段训练策略)

        阶段1: 冻结 backbone, 训练分类头 (5 epochs)
        阶段2: 解冻 layer3+layer4+分类头, 中等学习率微调 (10 epochs)
        阶段3: 全网络微调, 小学习率 (剩余 epochs)
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

        end_epoch = start_epoch + num_epochs
        stage1_epochs = min(5, end_epoch)
        stage2_epochs = min(15, end_epoch)

        total_start = time.time()

        # === 阶段1: 冻结 backbone ===
        if start_epoch < stage1_epochs:
            print("\n" + "=" * 60)
            print("阶段 1: 冻结 backbone, 训练分类头 (5 epochs)")
            print("=" * 60)
            self.model.freeze_backbone()
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=5, eta_min=1e-5,
            )

            for epoch in range(start_epoch + 1, stage1_epochs + 1):
                self._train_epoch(train_loader, val_loader, epoch, end_epoch,
                                  use_mixup=False)

        # === 阶段2: 部分解冻 (layer3 + layer4 + fc) ===
        if end_epoch > stage1_epochs:
            print("\n" + "=" * 60)
            print("阶段 2: 解冻 layer3+layer4+分类头, 中等学习率微调")
            print("=" * 60)
            self._partial_unfreeze()
            self.optimizer = optim.AdamW([
                {'params': self._get_backbone_params(), 'lr': config.LEARNING_RATE * 0.05,
                 'weight_decay': config.BACKBONE_WEIGHT_DECAY},
                {'params': self._get_fc_params(), 'lr': config.LEARNING_RATE * 0.5,
                 'weight_decay': config.WEIGHT_DECAY},
            ])
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=stage2_epochs - stage1_epochs, eta_min=1e-5,
            )

            stage2_start = max(stage1_epochs, start_epoch)
            for epoch in range(stage2_start + 1, stage2_epochs + 1):
                self._train_epoch(train_loader, val_loader, epoch, end_epoch,
                                  use_mixup=True)

        # === 阶段3: 全网络微调 ===
        if end_epoch > stage2_epochs:
            print("\n" + "=" * 60)
            print("阶段 3: 全网络微调, 小学习率")
            print("=" * 60)
            self.model.unfreeze_backbone()
            self.optimizer = optim.AdamW([
                {'params': self._get_backbone_params(), 'lr': config.LEARNING_RATE * 0.01,
                 'weight_decay': config.BACKBONE_WEIGHT_DECAY},
                {'params': self._get_fc_params(), 'lr': config.LEARNING_RATE * 0.1,
                 'weight_decay': config.WEIGHT_DECAY},
            ])
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=end_epoch - stage2_epochs, eta_min=1e-6,
            )

            for epoch in range(stage2_epochs + 1, end_epoch + 1):
                self._train_epoch(train_loader, val_loader, epoch, end_epoch,
                                  use_mixup=True)

        total_time = time.time() - total_start
        print(f"\n训练完成! 总耗时: {total_time / 60:.1f} 分钟")
        print(f"最佳验证准确率: {self.best_accuracy:.2f}% (Epoch {self.best_epoch})")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
        }

    def _partial_unfreeze(self):
        """部分解冻: 只解冻 layer3, layer4 和 fc"""
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name or 'se' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"部分解冻: layer3+layer4+fc+se, 可训练参数: {trainable:,}")

    def _get_backbone_params(self):
        """获取 backbone 参数 (不含 fc)"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'fc' not in name:
                params.append(param)
        return params

    def _get_fc_params(self):
        """获取分类头参数"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'fc' in name:
                params.append(param)
        return params

    def _train_epoch(self, train_loader, val_loader, epoch: int, total_epochs: int,
                     use_mixup: bool = True):
        """单个 epoch 的训练和验证"""
        print(f"\nEpoch [{epoch}/{total_epochs}]")
        print("-" * 40)

        # 当前学习率
        current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        print(f"学习率: {current_lrs}")

        # 训练
        train_loss, train_acc = self.train_one_epoch(train_loader, use_mixup=use_mixup)
        self.train_losses.append(train_loss)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # 验证 (不使用TTA, 节省训练时间)
        val_loss, val_acc, _, _ = self.validate(val_loader, use_tta=False)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 更新学习率
        self.scheduler.step()

        # 保存最佳模型
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            self.best_epoch = epoch
            self.save_model("best_model.pth")
            print(f">>> 保存最佳模型 (Acc: {val_acc:.2f}%)")

        # 定期保存检查点
        if epoch % 10 == 0:
            self.save_model(f"checkpoint_epoch{epoch}.pth")

    def save_model(self, filename: str):
        """保存模型权重"""
        save_path = os.path.join(self.output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
        }, save_path)

    def load_model(self, checkpoint_path: str):
        """加载模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载模型: {checkpoint_path}")
        print(f"  准确率: {checkpoint.get('best_accuracy', 'N/A')}%")
        print(f"  Epoch: {checkpoint.get('best_epoch', 'N/A')}")
