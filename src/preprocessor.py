"""
可配置、可组合的图像预处理流水线模块

通过 YAML 配置文件或字典初始化预处理管线，适配分类模型输入。
支持多种 resize 策略、归一化方式、数据增强操作，
所有操作基于 OpenCV 和 NumPy，不引入 PyTorch/TensorFlow 依赖。

典型用法::

    from src.preprocessor import PreprocessingPipeline

    # 从 YAML 配置初始化
    pipeline = PreprocessingPipeline.from_yaml("config/preprocess.yaml")

    # 推理模式
    tensor = pipeline.preprocess(image, mode="infer")

    # 训练模式（含数据增强）
    tensor = pipeline.preprocess(image, mode="train")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ImageNet 归一化参数 (与 torchvision transforms.Normalize 一致)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 默认填充颜色（灰色 128）
DEFAULT_PAD_COLOR: Tuple[int, int, int] = (128, 128, 128)


class PreprocessingPipeline:
    """
    可配置的图像预处理流水线

    通过配置字典或 YAML 文件初始化，支持以下处理步骤：

    1. **resize** — 尺寸调整
       - ``stretch``: 直接拉伸到目标尺寸
       - ``pad``: 保持长宽比，短边用灰色(128)填充
       - ``crop``: 中心裁剪后缩放

    2. **normalize** — 归一化
       - ``imagenet``: ImageNet 均值/标准差标准化
       - ``minmax``: 缩放到 [0, 1]
       - ``none``: 仅转为 float32，不做归一化

    3. **to_tensor** — 维度转换 (H, W, C) → (C, H, W)

    4. **augment** — 训练模式数据增强
       - 随机水平翻转
       - 随机旋转 ±15°
       - 颜色抖动（亮度/对比度 ±10%）

    Args:
        config: 预处理配置字典，包含 resize / normalize / augment 等字段
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = self._validate_and_fill_defaults(config)
        self._config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PreprocessingPipeline":
        """
        从 YAML 配置文件创建流水线实例

        Args:
            yaml_path: YAML 配置文件路径

        Returns:
            PreprocessingPipeline 实例

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 解析失败
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"YAML 配置文件顶层应为字典，收到: {type(config).__name__}")

        return cls(config)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, mode: str = "infer") -> np.ndarray:
        """
        对输入图像执行预处理流水线

        Args:
            image: 原始 RGB 图像 (H, W, 3), uint8
            mode: 处理模式
                - ``infer``: 仅做 resize + normalize + to_tensor
                - ``train``: 额外启用 augmentation

        Returns:
            预处理后的图像 (C, H, W), float32

        Raises:
            ValueError: mode 不合法或图像格式不符合预期
        """
        if mode not in ("infer", "train"):
            raise ValueError(f"mode 必须为 'infer' 或 'train'，收到: {mode!r}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"输入图像应为 (H, W, 3) 的 RGB 图像，收到形状: {image.shape}"
            )

        result = image.copy()

        # 1. 训练模式：先做数据增强（在 uint8 空间操作）
        if mode == "train":
            result = self._apply_augmentation(result)

        # 2. Resize
        result = self._apply_resize(result)

        # 3. Normalize
        result = self._apply_normalize(result)

        # 4. To Tensor: (H, W, C) → (C, H, W)
        result = np.transpose(result, (2, 0, 1)).copy()

        return result.astype(np.float32)

    def save_config(self, save_path: Union[str, Path]) -> None:
        """
        将当前配置序列化为 JSON 文件，以便复现

        Args:
            save_path: JSON 文件保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 将 numpy 数组转为列表以便 JSON 序列化
        serializable = self._make_serializable(self._config)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info("配置已保存至: %s", save_path)

    @property
    def config(self) -> Dict[str, Any]:
        """返回当前配置的深拷贝"""
        import copy
        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # 配置校验与默认值填充
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_fill_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """校验配置字典并填充默认值"""
        cfg = config.copy()

        # ---- resize 配置 ----
        resize_cfg = cfg.get("resize", {})
        resize_mode = resize_cfg.get("mode", "stretch")
        if resize_mode not in ("stretch", "pad", "crop"):
            raise ValueError(
                f"resize.mode 必须为 'stretch'/'pad'/'crop'，收到: {resize_mode!r}"
            )
        resize_cfg.setdefault("mode", resize_mode)
        resize_cfg.setdefault("target_size", 224)
        resize_cfg.setdefault("pad_color", list(DEFAULT_PAD_COLOR))
        resize_cfg.setdefault("interpolation", "cubic")
        cfg["resize"] = resize_cfg

        # ---- normalize 配置 ----
        norm_cfg = cfg.get("normalize", {})
        norm_mode = norm_cfg.get("mode", "imagenet")
        if norm_mode not in ("imagenet", "minmax", "none"):
            raise ValueError(
                f"normalize.mode 必须为 'imagenet'/'minmax'/'none'，收到: {norm_mode!r}"
            )
        norm_cfg.setdefault("mode", norm_mode)
        # 自定义均值/标准差（仅 imagenet 模式使用）
        norm_cfg.setdefault("mean", IMAGENET_MEAN.tolist())
        norm_cfg.setdefault("std", IMAGENET_STD.tolist())
        cfg["normalize"] = norm_cfg

        # ---- to_tensor 配置 ----
        cfg.setdefault("to_tensor", True)

        # ---- augment 配置（训练模式） ----
        aug_cfg = cfg.get("augment", {})
        aug_cfg.setdefault("horizontal_flip", {"prob": 0.5})
        aug_cfg.setdefault("rotation", {"max_angle": 15, "prob": 0.5})
        aug_cfg.setdefault("color_jitter", {
            "brightness": 0.1,
            "contrast": 0.1,
            "prob": 0.5,
        })
        cfg["augment"] = aug_cfg

        return cfg

    # ------------------------------------------------------------------
    # Resize 策略
    # ------------------------------------------------------------------

    def _get_interpolation(self, name: str) -> int:
        """将插值方式名称映射为 OpenCV 常量"""
        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        if name not in mapping:
            raise ValueError(f"不支持的插值方式: {name!r}，可选: {list(mapping.keys())}")
        return mapping[name]

    def _apply_resize(self, image: np.ndarray) -> np.ndarray:
        """
        根据配置执行尺寸调整

        Args:
            image: (H, W, C), uint8

        Returns:
            调整后的图像 (target_size, target_size, C), uint8
        """
        cfg = self._config["resize"]
        mode = cfg["mode"]
        target_size = cfg["target_size"]
        pad_color = tuple(cfg["pad_color"])
        interp = self._get_interpolation(cfg["interpolation"])

        if mode == "stretch":
            return self._resize_stretch(image, target_size, interp)
        elif mode == "pad":
            return self._resize_pad(image, target_size, pad_color, interp)
        elif mode == "crop":
            return self._resize_crop(image, target_size, interp)
        else:
            raise ValueError(f"未知的 resize 模式: {mode!r}")

    @staticmethod
    def _resize_stretch(
        image: np.ndarray, target_size: int, interpolation: int
    ) -> np.ndarray:
        """
        直接拉伸到目标正方形尺寸

        Args:
            image: (H, W, C), uint8
            target_size: 目标边长
            interpolation: OpenCV 插值方式

        Returns:
            (target_size, target_size, C), uint8
        """
        return cv2.resize(image, (target_size, target_size), interpolation=interpolation)

    @staticmethod
    def _resize_pad(
        image: np.ndarray,
        target_size: int,
        pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> np.ndarray:
        """
        保持长宽比缩放，短边居中填充灰色(128)

        将图像较长边缩放到 target_size，较短边用 pad_color 填充，
        图像居中放置。

        Args:
            image: (H, W, C), uint8
            target_size: 目标正方形边长
            pad_color: 填充颜色 (R, G, B)
            interpolation: OpenCV 插值方式

        Returns:
            (target_size, target_size, C), uint8
        """
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 确保缩放后尺寸不超过目标
        new_h = min(new_h, target_size)
        new_w = min(new_w, target_size)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # 创建填充画布，居中放置
        canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        return canvas

    @staticmethod
    def _resize_crop(
        image: np.ndarray, target_size: int, interpolation: int
    ) -> np.ndarray:
        """
        中心裁剪为正方形后缩放到目标尺寸

        以图像中心为基准，取较短边长度的正方形区域，
        然后缩放到 target_size × target_size。

        Args:
            image: (H, W, C), uint8
            target_size: 目标边长
            interpolation: OpenCV 插值方式

        Returns:
            (target_size, target_size, C), uint8
        """
        h, w = image.shape[:2]
        crop_size = min(h, w)

        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        cropped = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

        return cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)

    # ------------------------------------------------------------------
    # Normalize 策略
    # ------------------------------------------------------------------

    def _apply_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        根据配置执行归一化

        Args:
            image: (H, W, C), uint8 或 float32

        Returns:
            归一化后的 float32 图像
        """
        cfg = self._config["normalize"]
        mode = cfg["mode"]

        if mode == "imagenet":
            mean = np.array(cfg["mean"], dtype=np.float32)
            std = np.array(cfg["std"], dtype=np.float32)
            image = image.astype(np.float32) / 255.0
            image = (image - mean) / std
        elif mode == "minmax":
            image = image.astype(np.float32) / 255.0
        elif mode == "none":
            image = image.astype(np.float32)
        else:
            raise ValueError(f"未知的 normalize 模式: {mode!r}")

        return image

    # ------------------------------------------------------------------
    # Augmentation（训练模式）
    # ------------------------------------------------------------------

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        应用训练模式的数据增强

        按顺序执行：随机水平翻转 → 随机旋转 → 颜色抖动

        Args:
            image: (H, W, C), uint8

        Returns:
            增强后的图像 (H, W, C), uint8
        """
        aug_cfg = self._config["augment"]

        # 1. 随机水平翻转
        image = self._random_horizontal_flip(image, aug_cfg.get("horizontal_flip", {}))

        # 2. 随机旋转
        image = self._random_rotation(image, aug_cfg.get("rotation", {}))

        # 3. 颜色抖动
        image = self._color_jitter(image, aug_cfg.get("color_jitter", {}))

        return image

    @staticmethod
    def _random_horizontal_flip(
        image: np.ndarray, cfg: Dict[str, Any]
    ) -> np.ndarray:
        """
        随机水平翻转

        Args:
            image: (H, W, C), uint8
            cfg: 包含 ``prob`` 键的配置字典

        Returns:
            可能翻转后的图像
        """
        prob = cfg.get("prob", 0.5)
        if np.random.random() < prob:
            image = cv2.flip(image, 1)
        return image

    @staticmethod
    def _random_rotation(image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """
        随机旋转

        以图像中心为旋转中心，在 [-max_angle, +max_angle] 范围内随机旋转。
        旋转后空白区域用灰色(128)填充。

        Args:
            image: (H, W, C), uint8
            cfg: 包含 ``max_angle`` 和 ``prob`` 键的配置字典

        Returns:
            可能旋转后的图像
        """
        prob = cfg.get("prob", 0.5)
        if np.random.random() >= prob:
            return image

        max_angle = cfg.get("max_angle", 15)
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后的画布大小，确保不裁剪
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)

        # 调整平移量使图像居中
        rotation_matrix[0, 2] += (new_w - w) / 2.0
        rotation_matrix[1, 2] += (new_h - h) / 2.0

        # 用灰色填充旋转后的空白区域
        image = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
        )

        return image

    @staticmethod
    def _color_jitter(image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """
        颜色抖动：随机调整亮度和对比度

        亮度调整：对每个像素加减偏移量
        对比度调整：以 128 为中心缩放像素值范围

        Args:
            image: (H, W, C), uint8
            cfg: 包含 ``brightness``, ``contrast``, ``prob`` 键的配置字典

        Returns:
            可能调整后的图像
        """
        prob = cfg.get("prob", 0.5)
        if np.random.random() >= prob:
            return image

        brightness_delta = cfg.get("brightness", 0.1)
        contrast_delta = cfg.get("contrast", 0.1)

        result = image.astype(np.float32)

        # 亮度调整：随机偏移 [-delta * 255, +delta * 255]
        if brightness_delta > 0:
            brightness_offset = np.random.uniform(
                -brightness_delta * 255, brightness_delta * 255
            )
            result += brightness_offset

        # 对比度调整：以 128 为中心缩放
        if contrast_delta > 0:
            contrast_factor = np.random.uniform(
                1.0 - contrast_delta, 1.0 + contrast_delta
            )
            result = (result - 128.0) * contrast_factor + 128.0

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # 配置序列化
    # ------------------------------------------------------------------

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """递归地将配置中不可 JSON 序列化的对象转为可序列化类型"""
        if isinstance(obj, dict):
            return {k: PreprocessingPipeline._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PreprocessingPipeline._make_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
