"""
CUB-200-2011 数据集加载模块

使用 Pillow 读取图像, 结合 OpenCV 和 scikit-image 进行图像预处理,
基于数据集自带的标注文件(images.txt, train_test_split.txt 等)构建 PyTorch Dataset。
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import skimage
from skimage import exposure

from src import config
from src.preprocessing import (
    crop_by_bounding_box,
    resize_image,
    normalize_image,
    apply_data_augmentation,
    to_tensor,
)


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 鸟类细粒度分类数据集

    Args:
        root_dir: 数据集根目录 (CUB_200_2011/CUB_200_2011/)
        split: 'train' 或 'test'
        transform: 额外的 torchvision transforms (可选)
        use_bounding_box: 是否使用边界框裁剪
        use_augmentation: 是否使用数据增强 (仅训练集有效)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        use_bounding_box: bool = True,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_bounding_box = use_bounding_box
        self.use_augmentation = use_augmentation and (split == "train")
        self.images_dir = os.path.join(root_dir, "images")

        # 解析标注文件
        self.image_paths = []  # 图像相对路径
        self.labels = []        # 类别标签 (0-indexed)
        self.bboxes = []        # 边界框 (x, y, w, h)

        self._parse_annotations()

        assert len(self.image_paths) == len(self.labels) == len(self.bboxes), (
            f"数据不一致: images={len(self.image_paths)}, "
            f"labels={len(self.labels)}, bboxes={len(self.bboxes)}"
        )

    def _parse_annotations(self):
        """解析 CUB-200-2011 的标注文件"""
        root = self.root_dir

        # 1. 读取图像路径列表: image_id -> image_path
        image_id_to_path = {}
        with open(os.path.join(root, "images.txt"), "r") as f:
            for line in f:
                image_id, image_path = line.strip().split(" ", 1)
                image_id_to_path[int(image_id)] = image_path

        # 2. 读取类别标签: image_id -> class_id (1-indexed)
        image_id_to_class = {}
        with open(os.path.join(root, "image_class_labels.txt"), "r") as f:
            for line in f:
                image_id, class_id = line.strip().split(" ")
                image_id_to_class[int(image_id)] = int(class_id) - 1  # 转为 0-indexed

        # 3. 读取训练/测试划分: image_id -> is_training (1=train, 0=test)
        image_id_to_split = {}
        with open(os.path.join(root, "train_test_split.txt"), "r") as f:
            for line in f:
                image_id, is_train = line.strip().split(" ")
                image_id_to_split[int(image_id)] = int(is_train)

        # 4. 读取边界框: image_id -> (x, y, w, h)
        image_id_to_bbox = {}
        with open(os.path.join(root, "bounding_boxes.txt"), "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                image_id = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                image_id_to_bbox[image_id] = (x, y, w, h)

        # 5. 根据划分筛选数据
        target_split = 1 if self.split == "train" else 0
        for image_id, image_path in image_id_to_path.items():
            if image_id_to_split.get(image_id, -1) == target_split:
                self.image_paths.append(image_path)
                self.labels.append(image_id_to_class[image_id])
                self.bboxes.append(image_id_to_bbox.get(image_id, (0, 0, 0, 0)))

        print(f"[{self.split.upper()}] 加载了 {len(self.image_paths)} 张图像, {len(set(self.labels))} 个类别")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        try:
            # 1. 使用 Pillow 读取图像
            image_path = os.path.join(self.images_dir, self.image_paths[idx])
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)  # H x W x C, uint8, RGB

            # 2. 获取标注信息
            label = self.labels[idx]
            bbox = self.bboxes[idx]

            # 3. 使用 OpenCV 按边界框裁剪
            if self.use_bounding_box:
                image = crop_by_bounding_box(image, bbox)

            # 4. 调整大小到模型输入尺寸
            image = resize_image(image, config.INPUT_SIZE)

            # 5. 数据增强 (使用 scikit-image 和 OpenCV)
            if self.use_augmentation:
                image = apply_data_augmentation(image)

            # 6. 归一化
            image = normalize_image(image)

            # 7. 转为 PyTorch Tensor (C x H x W)
            image = to_tensor(image)

            # 8. 额外的 torchvision transforms (可选)
            if self.transform is not None:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # 出错时返回第一张图作为兜底, 避免训练中断
            print(f"\n警告: 加载图像失败 {self.image_paths[idx]}: {e}, 使用兜底图像")
            return self.__getitem__(0)


def get_class_names(root_dir: str) -> list:
    """获取所有类别名称"""
    class_names = []
    with open(os.path.join(root_dir, "classes.txt"), "r") as f:
        for line in f:
            _, class_name = line.strip().split(" ", 1)
            class_names.append(class_name)
    return class_names


def create_dataloaders(
    root_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_bounding_box: bool = True,
    use_augmentation: bool = True,
) -> tuple:
    """
    创建训练集和测试集的 DataLoader

    Returns:
        (train_loader, test_loader, class_names)
    """
    if root_dir is None:
        root_dir = config.DATASET_ROOT

    train_dataset = CUB200Dataset(
        root_dir=root_dir,
        split="train",
        use_bounding_box=use_bounding_box,
        use_augmentation=use_augmentation,
    )

    test_dataset = CUB200Dataset(
        root_dir=root_dir,
        split="test",
        use_bounding_box=use_bounding_box,
        use_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = get_class_names(root_dir)

    return train_loader, test_loader, class_names
