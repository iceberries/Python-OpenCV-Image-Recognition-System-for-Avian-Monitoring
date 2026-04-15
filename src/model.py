"""
ResNet50 模型构建与训练模块

基于 torchvision 预训练的 ResNet50, 替换最后的全连接层以适应 CUB-200-2011 的 200 类分类任务。
包含完整的训练、验证和模型保存逻辑。
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tqdm import tqdm

from src import config


class ResNet50BirdClassifier(nn.Module):
    """
    基于 ResNet50 的鸟类分类模型

    使用 ImageNet 预训练权重进行迁移学习, 将最后的全连接层替换为 200 维输出。
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()

        # 加载预训练 ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # 初始化新增层的权重
        self._init_weights()

    def _init_weights(self):
        """初始化新增全连接层的权重"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
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


class Trainer:
    """
    模型训练器

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

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA,
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def train_one_epoch(self, train_loader) -> float:
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
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
    def validate(self, val_loader) -> tuple:
        """
        验证模型

        Returns:
            (val_loss, val_accuracy, all_predictions, all_labels)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
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

    def fit(self, train_loader, val_loader, num_epochs: int = None) -> dict:
        """
        完整训练流程 (两阶段微调)

        阶段1: 冻结 backbone, 训练分类头 (5 epochs)
        阶段2: 解冻全部参数, 全网络微调

        Returns:
            训练历史记录
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

        total_start = time.time()

        # === 阶段1: 冻结 backbone, 训练分类头 ===
        print("\n" + "=" * 60)
        print("阶段 1: 冻结 backbone, 训练分类头")
        print("=" * 60)
        self.model.freeze_backbone()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=config.LR_GAMMA,
        )

        stage1_epochs = min(5, num_epochs)
        for epoch in range(1, stage1_epochs + 1):
            self._train_epoch(train_loader, val_loader, epoch, stage1_epochs)

        # === 阶段2: 全网络微调 ===
        print("\n" + "=" * 60)
        print("阶段 2: 解冻全部参数, 全网络微调")
        print("=" * 60)
        self.model.unfreeze_backbone()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE * 0.1,  # 降低学习率
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA,
        )

        stage2_epochs = num_epochs - stage1_epochs
        for epoch in range(stage1_epochs + 1, num_epochs + 1):
            self._train_epoch(train_loader, val_loader, epoch, num_epochs)

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

    def _train_epoch(self, train_loader, val_loader, epoch: int, total_epochs: int):
        """单个 epoch 的训练和验证"""
        print(f"\nEpoch [{epoch}/{total_epochs}]")
        print("-" * 40)

        # 当前学习率
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"学习率: {current_lr:.6f}")

        # 训练
        train_loss, train_acc = self.train_one_epoch(train_loader)
        self.train_losses.append(train_loss)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # 验证
        val_loss, val_acc, _, _ = self.validate(val_loader)
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
        if epoch % 5 == 0:
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
