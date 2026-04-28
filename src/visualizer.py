"""
鸟类图像识别系统 - 可视化模块

支持分类结果叠加、热力图叠加、对比网格拼接，
并内置中文标签、置信度颜色编码、文字描边等功能。
"""

import copy
import os
import platform
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    """识别结果可视化工具类。

    所有绘制方法均对输入图像做深拷贝，不修改原图。
    使用 PIL 绘制中文文字（带黑色描边），兼容所有平台。
    """

    # ------------------------------------------------------------------
    # 颜色常量 (BGR)
    # ------------------------------------------------------------------
    _COLOR_GREEN  = (0, 200, 0)
    _COLOR_YELLOW = (0, 200, 255)
    _COLOR_RED    = (0, 0, 220)
    _COLOR_WHITE  = (255, 255, 255)
    _COLOR_BLACK  = (0, 0, 0)
    _COLOR_BAR_BG = (50, 50, 50)

    # 颜色常量 (RGB) — 供 PIL 使用
    _RGB_GREEN  = (0, 200, 0)
    _RGB_YELLOW = (200, 200, 0)
    _RGB_RED    = (220, 0, 0)
    _RGB_WHITE  = (255, 255, 255)
    _RGB_BLACK  = (0, 0, 0)

    def __init__(self, font_scale: float = 0.8, thickness: int = 2):
        """
        Args:
            font_scale:  全局文字缩放比例（影响 PIL 字号）。
            thickness:   全局文字粗细（描边宽度）。
        """
        self._font_scale = font_scale
        self._thickness = thickness
        self._base_font_size = max(int(font_scale * 24), 14)
        self._pil_font = self._load_chinese_font(self._base_font_size)

    # ------------------------------------------------------------------
    # 字体加载
    # ------------------------------------------------------------------
    @staticmethod
    def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont:
        """尝试加载支持中文的 TrueType 字体。"""
        system = platform.system()

        if system == "Windows":
            font_dirs = [
                os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            ]
            candidates = ["simhei.ttf", "msyh.ttc", "msyhbd.ttc", "simsun.ttc"]
        elif system == "Darwin":
            font_dirs = [
                "/System/Library/Fonts",
                "/Library/Fonts",
                "/System/Library/Fonts/Supplemental",
                os.path.expanduser("~/Library/Fonts"),
            ]
            candidates = [
                "PingFang.ttc",
                "STHeiti Light.ttc",
                "STHeiti Medium.ttc",
                "Arial Unicode.ttf",
                "Heiti.ttc",
            ]
        else:  # Linux
            font_dirs = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                "/usr/share/fonts/truetype",
                "/usr/share/fonts/opentype",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
            ]
            candidates = [
                "NotoSansCJK-Regular.ttc",
                "NotoSansCJKsc-Regular.otf",
                "NotoSansSC-Regular.otf",
                "WenQuanYiMicroHei.ttf",
                "WenQuanYiZenHei.ttf",
                "DroidSansFallbackFull.ttf",
            ]

        for d in font_dirs:
            if not os.path.isdir(d):
                continue
            for c in candidates:
                path = os.path.join(d, c)
                if os.path.isfile(path):
                    try:
                        return ImageFont.truetype(path, size)
                    except Exception:
                        continue

        # 回退：PIL 默认字体
        print("[Visualizer] 警告: 未找到中文字体，使用默认字体（中文可能显示为方框）")
        return ImageFont.load_default()

    def _get_scaled_font(self, scale: float) -> ImageFont.FreeTypeFont:
        """获取按比例缩放的字体。"""
        scaled_size = max(int(self._base_font_size * scale), 10)
        if hasattr(self._pil_font, 'path'):
            try:
                return ImageFont.truetype(self._pil_font.path, scaled_size)
            except Exception:
                pass
        # 如果无法从路径创建，用基础字体
        return self._pil_font

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _put_text(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: Optional[float] = None,
        thickness: Optional[int] = None,
    ) -> None:
        """在 BGR 图像上绘制带黑色描边的文字（通过 PIL）。

        Args:
            img:         BGR numpy 数组，会被原地修改。
            text:        要绘制的文字。
            org:         文字左上角坐标 (x, y)。
            color:       BGR 颜色元组。
            font_scale:  相对于 self._font_scale 的缩放比例。
            thickness:   描边宽度（未使用，由 font_scale 间接控制）。
        """
        scale = font_scale / self._font_scale if font_scale else 1.0
        font = self._get_scaled_font(scale) if scale != 1.0 else self._pil_font
        stroke_w = thickness if thickness else self._thickness

        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # RGB 颜色
        rgb_color = (color[2], color[1], color[0])

        # PIL text 坐标是左上角，描边使用 stroke_width + stroke_fill
        draw.text(
            org,
            text,
            font=font,
            fill=rgb_color,
            stroke_width=stroke_w,
            stroke_fill=self._RGB_BLACK,
        )

        # 写回 BGR
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(img, result)

    def _confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """根据置信度返回 BGR 颜色。>90% 绿色，70-90% 黄色，<70% 红色。"""
        if confidence > 0.9:
            return self._COLOR_GREEN
        elif confidence >= 0.7:
            return self._COLOR_YELLOW
        else:
            return self._COLOR_RED

    def _get_text_size(
        self, text: str, font_scale: Optional[float] = None,
    ) -> Tuple[int, int]:
        """使用 PIL 精确计算文字宽高（像素）。"""
        scale = font_scale / self._font_scale if font_scale else 1.0
        font = self._get_scaled_font(scale) if scale != 1.0 else self._pil_font

        # 创建临时图像计算文字尺寸
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def draw_classification_result(
        self,
        image: np.ndarray,
        class_name: str,
        confidence: float,
        top_k: Optional[List[Dict[str, Union[str, float]]]] = None,
    ) -> np.ndarray:
        """在图像上绘制分类结果。

        Args:
            image:       输入 BGR 图像 (H, W, 3)。
            class_name:  主分类类别名称。
            confidence:  主分类置信度 [0, 1]。
            top_k:       可选 Top-K 列表，每项为 {"class": str, "confidence": float}。

        Returns:
            绘制后的图像（深拷贝）。
        """
        canvas = copy.deepcopy(image)
        color = self._confidence_color(confidence)
        pct = confidence * 100

        # ---- 左上角主分类结果 ----
        label = f"{class_name}  {pct:.1f}%"
        pad = 8
        text_w, text_h = self._get_text_size(label)
        box_h = text_h + pad * 2
        box_w = text_w + pad * 2 + self._thickness * 2  # 考虑描边宽度

        # 半透明背景条
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), self._COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        self._put_text(canvas, label, (pad, pad), color)

        # ---- 右侧 Top-K 排行榜 ----
        if top_k and len(top_k) > 0:
            canvas = self._draw_top_k_panel(canvas, top_k)

        return canvas

    def _draw_top_k_panel(
        self,
        canvas: np.ndarray,
        top_k: List[Dict[str, Union[str, float]]],
    ) -> np.ndarray:
        """在图像右侧绘制 Top-K 水平柱状排行榜。"""
        h, w = canvas.shape[:2]
        max_bars = min(len(top_k), 5)
        bar_height = 28
        panel_pad = 10
        bar_gap = 4
        label_area_w = 160
        bar_area_w = min(200, w // 3)

        panel_w = label_area_w + bar_area_w + panel_pad * 3 + 80  # 额外空间给百分比
        panel_h = panel_pad * 2 + max_bars * (bar_height + bar_gap)

        # 半透明面板
        x0 = w - panel_w - panel_pad
        y0 = panel_pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), self._COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        small_scale = self._font_scale * 0.55

        for idx, item in enumerate(top_k[:max_bars]):
            name = str(item.get("class", ""))
            conf = float(item.get("confidence", 0.0))
            color = self._confidence_color(conf)

            by = y0 + panel_pad + idx * (bar_height + bar_gap)
            bx_label = x0 + panel_pad
            bx_bar = x0 + panel_pad + label_area_w

            # 类别名
            self._put_text(
                canvas, name,
                (bx_label, by + 3),
                self._COLOR_WHITE,
                font_scale=small_scale,
                thickness=max(1, self._thickness - 1),
            )

            # 柱状条背景
            cv2.rectangle(
                canvas,
                (bx_bar, by),
                (bx_bar + bar_area_w, by + bar_height),
                self._COLOR_BAR_BG,
                -1,
            )
            # 柱状条前景
            bar_w = int(bar_area_w * conf)
            if bar_w > 0:
                cv2.rectangle(
                    canvas,
                    (bx_bar, by),
                    (bx_bar + bar_w, by + bar_height),
                    color,
                    -1,
                )
            # 百分比文字
            pct_text = f"{conf * 100:.1f}%"
            self._put_text(
                canvas, pct_text,
                (bx_bar + bar_area_w + 6, by + 3),
                self._COLOR_WHITE,
                font_scale=small_scale,
                thickness=max(1, self._thickness - 1),
            )

        return canvas

    def draw_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """将热力图叠加到原图上（JET colormap）。

        Args:
            image:   输入 BGR 图像 (H, W, 3) uint8。
            heatmap: 热力图 (h, w) 或 (h, w, 1)，值域 [0, 1] 或 [0, 255]。
            alpha:   热力图叠加透明度。

        Returns:
            叠加后的图像（深拷贝）。
        """
        canvas = copy.deepcopy(image)
        h, w = canvas.shape[:2]

        # squeeze & 归一化
        hm = heatmap.squeeze()
        if hm.dtype != np.uint8:
            if hm.max() <= 1.0:
                hm = (hm * 255).astype(np.uint8)
            else:
                hm = hm.astype(np.uint8)

        # 尺寸对齐：双线性插值
        if hm.shape[0] != h or hm.shape[1] != w:
            hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)

        # JET colormap → BGR
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # 加权叠加
        blended = cv2.addWeighted(canvas, 1.0, hm_color, alpha, 0)
        return blended

    def create_comparison_grid(
        self,
        images: Sequence[np.ndarray],
        titles: Sequence[str],
        rows: int = 1,
    ) -> np.ndarray:
        """将多张图拼接为对比网格。

        自动统一尺寸（以第一张图为基准），每张图下方带标题。

        Args:
            images: 图像列表，每个为 (H, W, 3) BGR。
            titles: 标题列表，长度应与 images 相同。
            rows:   网格行数。

        Returns:
            拼接后的网格图像。
        """
        n = len(images)
        if n == 0:
            raise ValueError("images 不能为空")
        if len(titles) != n:
            raise ValueError("titles 长度必须与 images 相同")

        cols = int(np.ceil(n / rows))

        # 以第一张图尺寸为基准
        ref_h, ref_w = images[0].shape[:2]
        title_bar_h = 36
        cell_h = ref_h + title_bar_h
        cell_w = ref_w

        # 画布
        grid_w = cols * cell_w
        grid_h = rows * cell_h
        grid = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)  # 深灰背景

        small_scale = self._font_scale * 0.65

        for idx, (img, title) in enumerate(zip(images, titles)):
            r = idx // cols
            c = idx % cols
            if r >= rows:
                break

            # 统一尺寸
            resized = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

            y0 = r * cell_h
            x0 = c * cell_w

            # 图像区域
            grid[y0 : y0 + ref_h, x0 : x0 + ref_w] = resized

            # 标题条
            title_y0 = y0 + ref_h
            cv2.rectangle(
                grid,
                (x0, title_y0),
                (x0 + cell_w, title_y0 + title_bar_h),
                self._COLOR_BLACK,
                -1,
            )
            self._put_text(
                grid, title,
                (x0 + 6, title_y0 + 4),
                self._COLOR_WHITE,
                font_scale=small_scale,
                thickness=max(1, self._thickness - 1),
            )

        return grid

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    @staticmethod
    def save(image: np.ndarray, path: str, dpi: int = 150) -> None:
        """保存图像为高质量 PNG。

        Args:
            image: BGR 图像。
            path:  保存路径。
            dpi:   分辨率（写入 PNG 元数据）。
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        success = cv2.imwrite(path, image, encode_param)
        if not success:
            raise IOError(f"无法保存图像至 {path}")

    @staticmethod
    def to_bytes(image: np.ndarray, fmt: str = "png") -> bytes:
        """将图像转为字节流。

        Args:
            image: BGR 图像。
            fmt:   格式后缀，如 "png", "jpg", "webp"。

        Returns:
            编码后的字节流。
        """
        ext = fmt.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        params = []
        if ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif ext in ("jpeg", "webp"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        success, buf = cv2.imencode(f".{ext}", image, params)
        if not success:
            raise ValueError(f"图像编码失败: format={fmt}")
        return buf.tobytes()


# ======================================================================
# 独立可运行 demo
# ======================================================================
def _demo() -> None:
    """使用随机数据模拟识别结果，演示 Visualizer 全部功能。"""
    print("=" * 60)
    print("  Visualizer Demo — 鸟类识别可视化")
    print("=" * 60)

    viz = Visualizer(font_scale=0.8, thickness=2)
    np.random.seed(42)

    # ---- 模拟数据 ----
    H, W = 360, 480
    base_img = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)

    class_name = "家燕 (Hirundo rustica)"
    confidence = 0.93

    top_k = [
        {"class": "家燕",   "confidence": 0.93},
        {"class": "金腰燕", "confidence": 0.04},
        {"class": "灰沙燕", "confidence": 0.015},
        {"class": "崖沙燕", "confidence": 0.009},
        {"class": "毛脚燕", "confidence": 0.006},
    ]

    # ---- 1. 分类结果叠加 ----
    print("\n[1] draw_classification_result ...")
    result_cls = viz.draw_classification_result(base_img, class_name, confidence, top_k)
    viz.save(result_cls, "output/demo_classification.png")
    print("    -> 保存至 output/demo_classification.png")

    # ---- 2. 热力图叠加 ----
    print("[2] draw_heatmap ...")
    heatmap_small = np.random.rand(64, 64).astype(np.float32)
    heatmap_small[20:44, 20:44] = np.random.rand(24, 24).astype(np.float32) * 0.5 + 0.5
    heatmap_small = cv2.GaussianBlur(heatmap_small, (15, 15), 0)

    result_hm = viz.draw_heatmap(base_img, heatmap_small, alpha=0.45)
    viz.save(result_hm, "output/demo_heatmap.png")
    print("    -> 保存至 output/demo_heatmap.png")

    # ---- 3. 对比网格 ----
    print("[3] create_comparison_grid ...")
    images_grid = [
        base_img,
        result_cls,
        result_hm,
        viz.draw_heatmap(base_img, heatmap_small, alpha=0.7),
    ]
    titles_grid = ["原图", "分类结果", "热力图 α=0.45", "热力图 α=0.70"]

    grid = viz.create_comparison_grid(images_grid, titles_grid, rows=2)
    viz.save(grid, "output/demo_comparison_grid.png")
    print("    -> 保存至 output/demo_comparison_grid.png")

    # ---- 4. 字节流导出 ----
    print("[4] to_bytes ...")
    data = viz.to_bytes(result_cls, fmt="png")
    print(f"    -> PNG 字节流大小: {len(data) / 1024:.1f} KB")

    # ---- 5. 不同置信度颜色 ----
    print("[5] 置信度颜色测试 ...")
    conf_tests = [
        ("高置信度鸟", 0.96),
        ("中置信度鸟", 0.78),
        ("低置信度鸟", 0.45),
    ]
    conf_images = []
    conf_titles = []
    for name, conf in conf_tests:
        img_c = viz.draw_classification_result(base_img, name, conf)
        conf_images.append(img_c)
        conf_titles.append(f"置信度 {conf*100:.0f}%")

    grid_conf = viz.create_comparison_grid(conf_images, conf_titles, rows=1)
    viz.save(grid_conf, "output/demo_confidence_colors.png")
    print("    -> 保存至 output/demo_confidence_colors.png")

    print("\n全部 Demo 完成！请查看 output/ 目录。")


if __name__ == "__main__":
    _demo()
