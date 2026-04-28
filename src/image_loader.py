"""
图像读取模块

提供健壮的图像加载功能，支持常见格式，处理异常情况，
并自动处理灰度图和带透明通道的图像。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# 支持的图像格式扩展名
SUPPORTED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ImageLoader:
    """
    健壮的图像读取器

    支持常见图像格式（.jpg, .jpeg, .png, .bmp, .tiff, .webp），
    自动处理灰度图转 RGB、带 Alpha 通道的 PNG 丢弃或填充白色背景，
    并提供完善的异常处理机制。

    Args:
        alpha_mode: Alpha 通道处理方式。
            - "discard": 直接丢弃 Alpha 通道（默认）
            - "white": 用白色背景填充透明区域
        max_workers: 批量读取时的最大线程数，默认为 4
    """

    def __init__(
        self,
        alpha_mode: str = "discard",
        max_workers: int = 4,
    ) -> None:
        if alpha_mode not in ("discard", "white"):
            raise ValueError(f"alpha_mode 必须为 'discard' 或 'white'，收到: {alpha_mode!r}")
        if max_workers < 1:
            raise ValueError(f"max_workers 必须为正整数，收到: {max_workers}")

        self._alpha_mode = alpha_mode
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def load_single(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        读取单张图片，返回 RGB 格式 ndarray (H, W, 3)

        Args:
            path: 图像文件路径（字符串或 Path 对象）

        Returns:
            RGB 格式的 numpy 数组，dtype=uint8，形状 (H, W, 3)；
            若文件损坏或无法解码则返回 None

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 文件格式不支持或文件为空（0 字节）时抛出
        """
        path = Path(path)

        # ---- 前置校验 ----
        self._validate_path(path)

        # ---- 读取图像 ----
        try:
            image = self._read_image(path)
        except UnidentifiedImageError:
            logger.error("文件损坏或格式无法识别: %s", path)
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("读取图像失败: %s — %s", path, exc)
            return None

        if image is None:
            logger.error("图像解码失败: %s", path)
            return None

        # ---- 统一通道处理 ----
        image = self._normalize_channels(image)

        return image

    def load_batch(self, paths: List[Union[str, Path]]) -> List[Optional[np.ndarray]]:
        """
        批量读取图片，使用多线程加速

        按照输入 paths 的顺序返回结果列表。若某张图片读取失败，
        对应位置为 None，不中断批量流程。

        Args:
            paths: 图像文件路径列表

        Returns:
            与 paths 等长的列表，元素为 np.ndarray 或 None
        """
        results: List[Optional[np.ndarray]] = [None] * len(paths)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_index = {
                executor.submit(self.load_single, p): idx for idx, p in enumerate(paths)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("批量读取中发生异常 [index=%d]: %s", idx, exc)
                    results[idx] = None

        return results

    def get_info(self, path: Union[str, Path]) -> Dict[str, object]:
        """
        返回图片信息

        Args:
            path: 图像文件路径

        Returns:
            包含以下键的字典：
            - size: (width, height)
            - channels: 通道数
            - format: 图像格式（如 'JPEG', 'PNG'）
            - file_size: 文件大小（字节）

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 文件格式不支持或文件为空时抛出
        """
        path = Path(path)
        self._validate_path(path)

        try:
            with Image.open(path) as img:
                width, height = img.size
                channels = len(img.getbands())
                fmt = img.format or "UNKNOWN"
        except UnidentifiedImageError:
            logger.error("文件损坏或格式无法识别: %s", path)
            raise ValueError(f"文件损坏或格式无法识别: {path}") from None
        except Exception as exc:
            logger.error("获取图片信息失败: %s — %s", path, exc)
            raise

        file_size = path.stat().st_size

        return {
            "size": (width, height),
            "channels": channels,
            "format": fmt,
            "file_size": file_size,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_path(self, path: Path) -> None:
        """校验路径是否存在、是否为空文件、格式是否支持"""
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"文件为空（0 字节）: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的图像格式: {ext!r}，"
                f"支持的格式: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    def _read_image(self, path: Path) -> Optional[np.ndarray]:
        """
        使用 Pillow 读取图像并转为 numpy 数组

        优先使用 Pillow 读取以获取更好的格式兼容性，
        然后转为 numpy 数组（RGB 顺序）。
        """
        img = Image.open(path)
        img.load()  # 强制加载像素数据，以便尽早发现损坏文件

        # Pillow 读取后为 RGB 或 RGBA，转为 numpy
        arr = np.array(img, dtype=np.uint8)
        return arr

    def _normalize_channels(self, image: np.ndarray) -> np.ndarray:
        """
        统一图像通道数为 3（RGB）

        - 灰度图 (H, W) → (H, W, 3)
        - RGBA (H, W, 4) → (H, W, 3)，根据 alpha_mode 处理
        - RGB (H, W, 3) → 不变
        """
        if image.ndim == 2:
            # 灰度图 → RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA → RGB
            if self._alpha_mode == "white":
                # 用白色背景填充透明区域
                rgb = image[:, :, :3].copy()
                alpha = image[:, :, 3:4].astype(np.float32) / 255.0
                white = np.full_like(rgb, 255, dtype=np.uint8)
                image = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            else:
                # 直接丢弃 Alpha 通道
                image = image[:, :, :3].copy()
        elif image.ndim == 3 and image.shape[2] == 1:
            # 单通道 (H, W, 1) → RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image
