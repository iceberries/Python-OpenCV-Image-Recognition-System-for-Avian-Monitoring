"""
鸟类图像识别 - 预测/推理模块

使用训练好的 ResNet50 + SE-Net 模型对单张图像或目录下所有图像进行预测。
支持测试时增强 (TTA) 提升预测准确率。

使用方法:
    # 预测单张图像
    python -m src.predict --image path/to/bird.jpg
    # 预测目录下所有图像
    python -m src.predict --dir path/to/images/
    # 指定模型路径
    python -m src.predict --image path/to/bird.jpg --checkpoint output/best_model.pth
    # 显示 Top-K 结果
    python -m src.predict --image path/to/bird.jpg --top_k 10
    # 使用 TTA 提升准确率
    python -m src.predict --image path/to/bird.jpg --tta
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as config
from src.dataset import get_class_names
from src.model import ResNet50BirdClassifier, Trainer, TestTimeAugmentation
from src.preprocessing import resize_image, normalize_image, to_tensor, apply_clahe


def load_and_preprocess_image(image_path: str, use_clahe: bool = True) -> torch.Tensor:
    """
    加载并预处理单张图像用于预测

    Args:
        image_path: 图像文件路径
        use_clahe: 是否使用 CLAHE 对比度增强

    Returns:
        预处理后的 Tensor (1, C, H, W)
    """
    # 使用 Pillow 读取图像
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # 使用 OpenCV 调整大小
    image = resize_image(image, config.INPUT_SIZE)

    # 使用 CLAHE 增强对比度
    if use_clahe:
        image = apply_clahe(image, clip_limit=2.0)

    # 归一化
    image = normalize_image(image)

    # 转为 Tensor (C, H, W) -> (1, C, H, W)
    tensor = to_tensor(image)
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    return tensor


def predict_single(
    model: torch.nn.Module,
    image_path: str,
    class_names: list,
    device: str,
    top_k: int = 5,
    use_tta: bool = False,
) -> list:
    """
    对单张图像进行预测

    Args:
        model: 训练好的模型
        image_path: 图像路径
        class_names: 类别名称列表
        device: 推理设备
        top_k: 返回前 K 个预测结果
        use_tta: 是否使用测试时增强

    Returns:
        [(class_name, probability), ...] 排序后的预测结果
    """
    # 预处理
    tensor = load_and_preprocess_image(image_path).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        if use_tta:
            # 使用 TTA
            tta = TestTimeAugmentation(model, device)
            probabilities = tta.predict(tensor)
        else:
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

    # 获取 Top-K 结果
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = class_names[idx] if idx < len(class_names) else f"Unknown_{idx}"
        results.append((class_name, float(prob) * 100))

    return results


def predict_directory(
    model: torch.nn.Module,
    dir_path: str,
    class_names: list,
    device: str,
    top_k: int = 5,
    use_tta: bool = False,
) -> dict:
    """
    对目录下所有图像进行预测

    Args:
        model: 训练好的模型
        dir_path: 图像目录路径
        class_names: 类别名称列表
        device: 推理设备
        top_k: 返回前 K 个预测结果
        use_tta: 是否使用测试时增强

    Returns:
        {image_path: [(class_name, probability), ...]}
    """
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(dir_path, ext)))
        image_files.extend(glob(os.path.join(dir_path, "**", ext), recursive=True))

    image_files = sorted(set(image_files))
    print(f"在 {dir_path} 中找到 {len(image_files)} 张图像")

    results = {}
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")
        try:
            preds = predict_single(model, image_path, class_names, device, top_k, use_tta)
            results[image_path] = preds
            for rank, (name, prob) in enumerate(preds, 1):
                print(f"  {rank}. {name} ({prob:.2f}%)")
        except Exception as e:
            print(f"  错误: {e}")

    return results


def visualize_prediction(image_path: str, results: list, save_path: str = None):
    """
    可视化预测结果 (使用 OpenCV)

    Args:
        image_path: 图像路径
        results: predict_single 返回的结果
        save_path: 保存路径 (None 则显示)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 创建结果面板
    panel_h = h
    panel_w = max(w + 300, 600)
    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 240

    # 放置图像
    panel[:h, :w] = image

    # 写入预测结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 40

    cv2.putText(panel, "Prediction Results", (w + 20, y_offset),
                font, 0.8, (0, 0, 0), 2)
    y_offset += 50

    for rank, (name, prob) in enumerate(results, 1):
        # 颜色编码: 越高概率越绿
        color = (0, int(200 * prob / 100), 0) if rank == 1 else (80, 80, 80)
        text = f"{rank}. {name}"
        prob_text = f"{prob:.1f}%"

        cv2.putText(panel, text, (w + 20, y_offset),
                    font, 0.5, color, 1)
        cv2.putText(panel, prob_text, (w + 20, y_offset + 22),
                    font, 0.45, (100, 100, 100), 1)
        y_offset += 50

    if save_path:
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, panel_bgr)
        print(f"结果已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="鸟类图像预测 (支持 TTA)")

    parser.add_argument("--image", type=str, default=None,
                        help="单张图像路径")
    parser.add_argument("--dir", type=str, default=None,
                        help="图像目录路径")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.OUTPUT_DIR, "best_model.pth"),
                        help="模型权重路径")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集根目录 (用于获取类别名称)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="显示 Top-K 预测结果 (默认: 5)")
    parser.add_argument("--tta", action="store_true",
                        help="使用测试时增强 (TTA) 提升准确率")
    parser.add_argument("--no_se", action="store_true",
                        help="模型不使用 SE-Net 注意力模块")
    parser.add_argument("--visualize", action="store_true",
                        help="可视化预测结果")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="可视化结果保存目录")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("请指定 --image 或 --dir 参数")
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        print("请先运行训练: python -m src.main")
        sys.exit(1)

    # 加载类别名称
    data_dir = args.data_dir or config.DATASET_ROOT
    class_names = get_class_names(data_dir)
    print(f"加载了 {len(class_names)} 个类别")

    # 构建模型并加载权重
    device = config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = ResNet50BirdClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        use_se=not args.no_se,
    )
    trainer = Trainer(model=model, device=device)
    trainer.load_model(args.checkpoint)

    print(f"\n使用设备: {device}")
    print(f"SE-Net 注意力: {'是' if not args.no_se else '否'}")
    print(f"测试时增强 (TTA): {'是' if args.tta else '否'}")
    print("=" * 60)

    # 单张图像预测
    if args.image:
        print(f"\n预测图像: {args.image}")
        results = predict_single(model, args.image, class_names, device, args.top_k, args.tta)
        print(f"\nTop-{args.top_k} 预测结果:")
        for rank, (name, prob) in enumerate(results, 1):
            marker = " <--" if rank == 1 else ""
            print(f"  {rank}. {name:40s} ({prob:.2f}%){marker}")

        if args.visualize:
            save_path = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(args.image))[0]
                save_path = os.path.join(args.save_dir, f"{basename}_result.jpg")
            visualize_prediction(args.image, results, save_path)

    # 目录预测
    if args.dir:
        predict_directory(model, args.dir, class_names, device, args.top_k, args.tta)


if __name__ == "__main__":
    main()
