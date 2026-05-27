"""
CUB-200-2011 层次化数据集

复用现有 CUB-200 图像（CUB_200_2011/images/），
结合 cvjena CUB-Hierarchy 的 3 级分类学标签 (Order→Family→Species)。

每个样本返回: (image, {order, family, species})
"""

import os
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    resize_image,
    normalize_image,
    apply_data_augmentation,
    to_tensor,
    apply_clahe,
)
from src.taxonomy import TaxonomyTree


class CUBHierarchyDataset(Dataset):
    """
    CUB-200 层次化数据集 (3 级: Order→Family→Species)

    Args:
        root_dir: CUB-200 数据集根目录 (含 images/ 和 images.txt 等)
        hierarchy_dir: CUB-Hierarchy 目录 (含 cub.parent-child.txt, classes.txt)
        taxonomy: 已解析的 TaxonomyTree (parse_cub_hierarchy() 后)
        split: 'train' | 'test'
        input_size: 输入尺寸
        use_augmentation: 训练时是否增强
    """

    def __init__(
        self,
        root_dir: str,
        hierarchy_dir: str,
        taxonomy: TaxonomyTree,
        split: str = "train",
        input_size: int = 224,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.taxonomy = taxonomy
        self.split = split
        self.input_size = input_size
        self.use_augmentation = use_augmentation and (split == "train")
        self.images_dir = os.path.join(root_dir, "images")

        self.image_paths: List[str] = []
        self.species_ids: List[int] = []  # CUB-200 原始 class_id (0~199)
        self.hierarchy_labels: List[Dict[str, int]] = []

        self._parse_cub_dataset(hierarchy_dir)
        self._build_hierarchy_labels()

    def _parse_cub_dataset(self, hierarchy_dir: str):
        """解析 CUB-200 标注 + 匹配层级"""
        r = self.root_dir

        # --- image_id → path ---
        id_to_path: Dict[int, str] = {}
        with open(os.path.join(r, "images.txt"), 'r') as f:
            for line in f:
                img_id, path = line.strip().split(' ', 1)
                id_to_path[int(img_id)] = path

        # --- image_id → species_label (1-indexed → 0-indexed) ---
        id_to_species: Dict[int, int] = {}
        with open(os.path.join(r, "image_class_labels.txt"), 'r') as f:
            for line in f:
                img_id, cls = line.strip().split(' ')
                id_to_species[int(img_id)] = int(cls) - 1

        # --- image_id → split (1=train, 0=test) ---
        id_to_split: Dict[int, int] = {}
        with open(os.path.join(r, "train_test_split.txt"), 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split(' ')
                id_to_split[int(img_id)] = int(is_train)

        target = 1 if self.split == "train" else 0

        for img_id, species_id in id_to_species.items():
            if id_to_split.get(img_id, 1) != target:
                continue
            if img_id not in id_to_path:
                continue
            self.image_paths.append(id_to_path[img_id])
            self.species_ids.append(species_id)

    def _build_hierarchy_labels(self):
        """将 CUB-200 class_id 映射到 3 级层级标签"""
        for species_id in self.species_ids:
            # 在 CUB-Hierarchy 中，species 节点的 taxonomy ID = CUB class_id
            node = self.taxonomy.get_node(species_id)
            if node is None:
                # 回退：尝试通过名称查找
                labels = {'order': -1, 'family': -1, 'species': species_id}
            else:
                path = node.path_to_root()
                labels = {'order': -1, 'family': -1, 'species': species_id}
                for n in path:
                    if n.level == 'order':
                        labels['order'] = self.taxonomy.get_level_idx('order', n.id)
                    elif n.level == 'family':
                        labels['family'] = self.taxonomy.get_level_idx('family', n.id)
            self.hierarchy_labels.append(labels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        rel_path = self.image_paths[index]
        full_path = os.path.join(self.images_dir, rel_path)
        labels = self.hierarchy_labels[index]

        image = Image.open(full_path).convert("RGB")
        image = np.array(image)

        image = resize_image(image, self.input_size)
        if self.use_augmentation:
            image = apply_data_augmentation(image)
        image = apply_clahe(image, clip_limit=2.0)
        image = normalize_image(image)
        image_tensor = torch.from_numpy(to_tensor(image)).float()

        return image_tensor, labels

    def get_num_classes(self) -> Dict[str, int]:
        nums = self.taxonomy.num_levels()
        return {
            'order': nums.get('order', 0),
            'family': nums.get('family', 0),
            'species': nums.get('species', 0),
        }


def collate_cub_hierarchy(batch: List[Tuple[torch.Tensor, Dict[str, int]]]):
    """批次整理: dict → 独立 tensor"""
    images = torch.stack([item[0] for item in batch])
    labels = {
        'order': torch.tensor([item[1].get('order', -1) for item in batch], dtype=torch.long),
        'family': torch.tensor([item[1].get('family', -1) for item in batch], dtype=torch.long),
        'species': torch.tensor([item[1].get('species', -1) for item in batch], dtype=torch.long),
    }
    return images, labels


def create_cub_hierarchy_dataloaders(
    cub_root: str,
    hierarchy_dir: str,
    taxonomy: TaxonomyTree,
    batch_size: int = 16,
    num_workers: int = 4,
    input_size: int = 224,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """创建 CUB-Hierarchy 训练/验证 DataLoader"""
    train_ds = CUBHierarchyDataset(
        cub_root, hierarchy_dir, taxonomy, "train", input_size, use_augmentation
    )
    val_ds = CUBHierarchyDataset(
        cub_root, hierarchy_dir, taxonomy, "test", input_size, False
    )

    print(f"CUB-Hierarchy 训练集: {len(train_ds)} 张")
    print(f"CUB-Hierarchy 验证集: {len(val_ds)} 张")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True,
                   collate_fn=collate_cub_hierarchy),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True,
                   collate_fn=collate_cub_hierarchy),
    )
