"""
鸟类图像识别系统 - 主程序入口

改进版本:
  - 支持 SE-Net 注意力增强
  - 支持 Mixup/CutMix 数据增强
  - 支持 Label Smoothing
  - 支持 Cosine Annealing 学习率调度
  - 支持测试时增强 (TTA)
  - 三阶段训练策略
"""

import os
import sys
import argparse
import random
import numpy as np

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.dataset import create_dataloaders
from src.model import ResNet50BirdClassifier, Trainer, TestTimeAugmentation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CUB-200-2011 鸟类图像识别系统 (改进版)")

    # 基础参数
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)

    # 模型参数
    parser.add_argument("--num_classes", type=int, default=config.NUM_CLASSES)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_se", action="store_true",
                        help="禁用 SE-Net 注意力模块")

    # 数据增强参数
    parser.add_argument("--no_bbox", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_mixup", action="store_true",
                        help="禁用 Mixup/CutMix 数据增强")
    parser.add_argument("--mixup_alpha", type=float, default=config.MIXUP_ALPHA)
    parser.add_argument("--cutmix_alpha", type=float, default=config.CUTMIX_ALPHA)

    # 训练参数
    parser.add_argument("--label_smoothing", type=float, default=config.LABEL_SMOOTHING)
    parser.add_argument("--no_cosine_lr", action="store_true",
                        help="使用 StepLR 替代 Cosine Annealing")

    # 评估参数
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--tta", action="store_true",
                        help="使用测试时增强进行评估")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.NUM_EPOCHS = args.epochs
    config.LABEL_SMOOTHING = args.label_smoothing
    config.MIXUP_ALPHA = 0 if args.no_mixup else args.mixup_alpha
    config.CUTMIX_ALPHA = 0 if args.no_mixup else args.cutmix_alpha
    config.USE_COSINE_LR = not args.no_cosine_lr
    config.USE_SE_ATTENTION = not args.no_se

    print("=" * 60)
    print("  CUB-200-2011 鸟类图像识别系统 (改进版)")
    print("  模型: ResNet50 + SE-Net + 改进分类头")
    print("=" * 60)
    print(f"  批次大小:       {args.batch_size}")
    print(f"  训练轮次:       {args.epochs}")
    print(f"  学习率:         {args.lr}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  SE-Net 注意力:  {'是' if not args.no_se else '否'}")
    print(f"  Mixup/CutMix:   {'是' if not args.no_mixup else '否'}")
    print(f"  Cosine LR:      {'是' if not args.no_cosine_lr else '否'}")
    print(f"  预训练权重:     {'是' if not args.no_pretrained else '否'}")
    print(f"  边界框裁剪:     {'是' if not args.no_bbox else '否'}")
    print(f"  数据增强:       {'是' if not args.no_aug else '否'}")
    print(f"  测试时增强:     {'是' if args.tta else '否'}")
    print(f"  设备:           {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    data_dir = args.data_dir if args.data_dir else config.DATASET_ROOT
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"错误: 数据集目录不存在: {data_dir}")
        sys.exit(1)

    print("\n加载数据集...")
    train_loader, test_loader, class_names = create_dataloaders(
        root_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_bounding_box=not args.no_bbox,
        use_augmentation=not args.no_aug,
    )
    print(f"训练集: {len(train_loader.dataset)} 张图像")
    print(f"测试集: {len(test_loader.dataset)} 张图像")

    print("\n构建 ResNet50 + SE-Net 模型...")
    model = ResNet50BirdClassifier(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        use_se=not args.no_se,
    )

    # ========== 恢复训练逻辑 ==========
    start_epoch = 0
    resume_checkpoint = None

    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n恢复训练: 加载检查点 {args.resume}")
            resume_checkpoint = torch.load(args.resume, map_location=config.DEVICE)
            model.load_state_dict(resume_checkpoint['model_state_dict'])

            start_epoch = resume_checkpoint.get('best_epoch', 0)
            best_acc = resume_checkpoint.get('best_accuracy', 0.0)
            print(f"  ✓ 已加载模型权重 (原最佳: {best_acc:.2f}%, Epoch {start_epoch})")

            # 微调模式：降低学习率
            original_lr = args.lr
            args.lr = args.lr * 0.1
            config.LEARNING_RATE = args.lr
            print(f"  ✓ 学习率调整: {original_lr} -> {args.lr}")
        else:
            print(f"警告: 检查点不存在: {args.resume}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(model=model, output_dir=config.OUTPUT_DIR)

    # 恢复训练器状态
    if resume_checkpoint:
        if 'optimizer_state_dict' in resume_checkpoint:
            trainer.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        trainer.best_accuracy = resume_checkpoint.get('best_accuracy', 0.0)
        trainer.best_epoch = resume_checkpoint.get('best_epoch', 0)
        print(f"  ✓ 已恢复训练状态 (best_acc={trainer.best_accuracy:.2f}%)")

    # 评估模式
    if args.eval:
        checkpoint_path = args.checkpoint or os.path.join(config.OUTPUT_DIR, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"错误: 模型文件不存在: {checkpoint_path}")
            sys.exit(1)
        trainer.load_model(checkpoint_path)

        print("\n评估模式...")
        if args.tta:
            print("使用测试时增强 (TTA)...")
        val_loss, val_acc, _, _ = trainer.validate(test_loader, use_tta=args.tta)
        print(f"\n测试集结果: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        return

    # 训练
    print(f"\n开始训练 (从 Epoch {start_epoch + 1} 开始)...")
    history = trainer.fit(train_loader, test_loader, num_epochs=args.epochs, start_epoch=start_epoch)

    # 最终评估 (使用 TTA)
    print("\n最终评估...")
    best_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        trainer.load_model(best_path)
        val_loss, val_acc, _, _ = trainer.validate(test_loader, use_tta=True)
        print(f"\n最终测试集结果 (TTA): Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
    else:
        print("警告: 未找到最佳模型文件")


if __name__ == "__main__":
    main()
