"""
鸟类图像识别系统 - 主程序入口

使用方法:
    python -m src.main              # 使用默认参数训练
    python -m src.main --epochs 50  # 指定训练轮次
    python -m src.main --eval       # 仅评估 (需已有模型权重)
    python -m src.main --batch_size 64  # 指定批次大小
"""

import os
import sys
import argparse
import random
import numpy as np

import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.dataset import create_dataloaders, get_class_names
from src.model import ResNet50BirdClassifier, Trainer


def set_seed(seed: int = 42):
    """设置随机种子, 保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CUB-200-2011 鸟类图像识别系统 (ResNet50)")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集根目录 (默认自动检测)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"批次大小 (默认: {config.BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help=f"DataLoader 工作线程数 (默认: {config.NUM_WORKERS})")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"训练轮次 (默认: {config.NUM_EPOCHS})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"初始学习率 (默认: {config.LEARNING_RATE})")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                        help=f"权重衰减 (默认: {config.WEIGHT_DECAY})")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help=f"随机种子 (默认: {config.RANDOM_SEED})")

    # 模式选项
    parser.add_argument("--eval", action="store_true",
                        help="仅评估模式 (需已有模型权重)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型权重路径 (默认使用最佳模型)")
    parser.add_argument("--no_bbox", action="store_true",
                        help="不使用边界框裁剪")
    parser.add_argument("--no_aug", action="store_true",
                        help="不使用数据增强")

    # 模型选项
    parser.add_argument("--no_pretrained", action="store_true",
                        help="不使用预训练权重 (从头训练)")
    parser.add_argument("--num_classes", type=int, default=config.NUM_CLASSES,
                        help=f"类别数 (默认: {config.NUM_CLASSES})")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 显示训练配置
    print("=" * 60)
    print("  CUB-200-2011 鸟类图像识别系统")
    print("  模型: ResNet50 (迁移学习)")
    print("=" * 60)
    print(f"  数据集目录:     {args.data_dir or config.DATASET_ROOT}")
    print(f"  批次大小:       {args.batch_size}")
    print(f"  训练轮次:       {args.epochs}")
    print(f"  学习率:         {args.lr}")
    print(f"  预训练权重:     {'是' if not args.no_pretrained else '否'}")
    print(f"  边界框裁剪:     {'是' if not args.no_bbox else '否'}")
    print(f"  数据增强:       {'是' if not args.no_aug else '否'}")
    print(f"  设备:           {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.NUM_EPOCHS = args.epochs

    # 数据集路径
    data_dir = args.data_dir if args.data_dir else config.DATASET_ROOT
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"错误: 数据集目录不存在或结构不正确: {data_dir}")
        print(f"请确保路径下存在 images/ 文件夹")
        sys.exit(1)

    # 加载数据
    print("\n加载数据集...")
    train_loader, test_loader, class_names = create_dataloaders(
        root_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_bounding_box=not args.no_bbox,
        use_augmentation=not args.no_aug,
    )
    print(f"训练集: {len(train_loader.dataset)} 张图像, {len(train_loader)} 个批次")
    print(f"测试集: {len(test_loader.dataset)} 张图像, {len(test_loader)} 个批次")

    # 构建模型
    print("\n构建 ResNet50 模型...")
    model = ResNet50BirdClassifier(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(model=model, output_dir=config.OUTPUT_DIR)

    # 评估模式
    if args.eval:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"错误: 模型文件不存在: {checkpoint_path}")
            sys.exit(1)
        trainer.load_model(checkpoint_path)
        val_loss, val_acc, preds, labels = trainer.validate(test_loader)
        print(f"\n测试集结果: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        return

    # 训练
    print("\n开始训练...")
    history = trainer.fit(train_loader, test_loader, num_epochs=args.epochs)

    # 最终评估
    print("\n最终评估...")
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    trainer.load_model(checkpoint_path)
    val_loss, val_acc, preds, labels = trainer.validate(test_loader)
    print(f"\n最终测试集结果: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")

    # 打印 Top-5 准确率
    print(f"\n训练历史:")
    for epoch, (loss, acc) in enumerate(
        zip(history["val_losses"], history["val_accuracies"]), 1
    ):
        marker = " *" if acc == max(history["val_accuracies"]) else ""
        print(f"  Epoch {epoch:3d}: Val Loss={loss:.4f}, Val Acc={acc:.2f}%{marker}")


if __name__ == "__main__":
    main()
