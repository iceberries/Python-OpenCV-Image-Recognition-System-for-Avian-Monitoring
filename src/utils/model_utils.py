"""
模型辅助工具函数

- 加载 config.json
- 计算模型参数量
- GPU 显存监控
- 类别名加载
"""
import json
import os
from typing import Dict, List, Optional


def load_model_config(model_dir: str) -> Dict:
    """
    加载模型目录下的 config.json

    Args:
        model_dir: 模型权重所在目录

    Returns:
        配置字典，文件不存在时返回空字典
    """
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def count_parameters(model) -> Dict[str, int]:
    """
    计算模型参数量

    Args:
        model: PyTorch 模型

    Returns:
        {"total": 总参数量, "trainable": 可训练参数量}
    """
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    except Exception:
        return {"total": 0, "trainable": 0}


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """
    获取 GPU 显存使用信息 (MB)

    Returns:
        {"total_mb": 总显存, "used_mb": 已用显存, "free_mb": 空闲显存}
        无 GPU 时返回 None
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
        used = torch.cuda.memory_allocated(0) / 1024 / 1024
        free = total - used
        return {"total_mb": total, "used_mb": used, "free_mb": free}
    except Exception:
        return None


def load_class_names(classes_file: str = None, dataset_root: str = None) -> List[str]:
    """
    加载类别名称列表

    优先级: classes_file > dataset_root/classes.txt > src/config 默认

    Args:
        classes_file: 类别文件完整路径
        dataset_root: 数据集根目录

    Returns:
        类别名称列表
    """
    # 1. 直接指定文件
    if classes_file and os.path.exists(classes_file):
        names = []
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    names.append(parts[1] if len(parts) > 1 else parts[0])
        return names

    # 2. 数据集目录下的 classes.txt
    if dataset_root:
        path = os.path.join(dataset_root, "classes.txt")
        if os.path.exists(path):
            return load_class_names(classes_file=path)

    # 3. 默认路径
    try:
        from src.config import CLASSES_FILE
        if os.path.exists(CLASSES_FILE):
            return load_class_names(classes_file=CLASSES_FILE)
    except ImportError:
        pass

    # 4. CUB-200-2011 标准结构
    from src.config import PROJECT_ROOT
    default_path = os.path.join(
        PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011", "classes.txt"
    )
    if os.path.exists(default_path):
        return load_class_names(classes_file=default_path)

    return []


def format_parameter_count(count: int) -> str:
    """格式化参数量显示"""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
