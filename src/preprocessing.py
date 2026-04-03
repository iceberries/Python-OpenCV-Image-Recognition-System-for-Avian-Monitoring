"""
图像预处理与增强工具模块

整合 OpenCV (cv2)、scikit-image (skimage)、numpy 提供完整的图像处理管线:

=== 图像预处理 ===
- 尺寸调整 (等比例缩放、Letterbox 填充、长边缩放)
- 归一化 (ImageNet 标准化、自定义均值/方差)
- 直方图均衡化 (全局直方图均衡化、CLAHE、自适应直方图均衡化)
- 边界框裁剪
- Tensor 转换

=== 图像增强 ===
- 对比度增强 (Gamma 校正、对数变换、Sigmoid 校正、CLAHE)
- 锐化 (USM 锐化、拉普拉斯锐化、非锐化掩蔽)
- 去噪 (高斯滤波、中值滤波、双边滤波、非局部均值去噪)
- 形变 (仿射变换、透视变换、弹性形变、网格扭曲)
- 颜色抖动 (亮度、对比度、饱和度、色调)
- 空间变换 (旋转、翻转、裁剪、缩放)

=== 完整管线 ===
- apply_preprocessing(): 推理时的标准预处理
- apply_data_augmentation(): 训练时的数据增强管线
"""

import numpy as np
import cv2
import skimage
from skimage import exposure, transform, filters, restoration
import random

import config

# ImageNet 归一化参数 (与 torchvision transforms.Normalize 一致)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
#  一、基础操作
# ============================================================

def crop_by_bounding_box(image: np.ndarray, bbox: tuple, padding: int = 0) -> np.ndarray:
    """
    根据边界框裁剪图像 (OpenCV)

    Args:
        image: (H, W, C), uint8, RGB
        bbox: (x, y, width, height)
        padding: 边界框扩展像素数
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    x = max(0, int(x) - padding)
    y = max(0, int(y) - padding)
    bw = min(int(bw) + 2 * padding, w - x)
    bh = min(int(bh) + 2 * padding, h - y)

    x2 = min(x + bw, w)
    y2 = min(y + bh, h)
    cropped = image[y:y2, x:x2]

    return cropped if cropped.size > 0 else image


def to_tensor(image: np.ndarray) -> np.ndarray:
    """numpy (H, W, C) -> PyTorch (C, H, W)"""
    return np.transpose(image, (2, 0, 1)).copy()


# ============================================================
#  二、尺寸调整 (预处理)
# ============================================================

def resize_image(image: np.ndarray, target_size: int = 224,
                 interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    直接缩放到正方形目标尺寸 (OpenCV)

    Args:
        image: (H, W, C), uint8
        target_size: 目标正方形边长
        interpolation: OpenCV 插值方式
    """
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)


def resize_keep_aspect_ratio(image: np.ndarray, target_size: int = 224,
                              pad_color: tuple = (128, 128, 128),
                              interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    等比例缩放 (Letterbox 填充), 保持长宽比不变形 (OpenCV)

    将图像较长边缩放到 target_size, 较短边用 pad_color 填充。

    Args:
        image: (H, W, C), uint8
        target_size: 目标尺寸 (正方形)
        pad_color: 填充颜色 (RGB)
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 创建填充画布
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas


def resize_longest_side(image: np.ndarray, max_size: int = 800,
                         interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    按长边等比例缩放, 短边自适应 (OpenCV)

    Args:
        image: (H, W, C), uint8
        max_size: 长边目标长度
    """
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image  # 不放大
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


# ============================================================
#  三、归一化 (预处理)
# ============================================================

def normalize_image(image: np.ndarray,
                    mean: np.ndarray = None,
                    std: np.ndarray = None) -> np.ndarray:
    """
    归一化: uint8 [0,255] -> float32, 然后减均值除标准差

    Args:
        image: (H, W, C), uint8
        mean: 均值, 默认 ImageNet
        std: 标准差, 默认 ImageNet
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Min-Max 归一化到 [0, 1] (numpy)

    Args:
        image: (H, W, C), uint8/float
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image.astype(np.float32) - img_min) / (img_max - img_min)


def standardize_image(image: np.ndarray) -> np.ndarray:
    """
    Z-Score 标准化: (x - mean) / std (numpy)

    Args:
        image: (H, W, C), uint8
    """
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    if std == 0:
        return image - mean
    return (image - mean) / std


# ============================================================
#  四、直方图均衡化 (预处理/增强)
# ============================================================

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    全局直方图均衡化 (OpenCV)

    将每个颜色通道独立进行直方图均衡化, 适用于光照不均匀的图像。

    Args:
        image: (H, W, C), uint8, RGB
    """
    channels = []
    for i in range(3):
        channels.append(cv2.equalizeHist(image[:, :, i]))
    return cv2.merge(channels)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE 对比度受限自适应直方图均衡化 (OpenCV)

    在 LAB 色彩空间的 L 通道上操作, 避免色彩失真。
    相比全局均衡化, CLAHE 能更好地处理局部对比度差异。

    Args:
        image: (H, W, C), uint8, RGB
        clip_limit: 对比度限制阈值, 值越大对比度增强越强
        tile_grid_size: 网格尺寸, 将图像分成 tile_grid_size 大小的网格分别处理
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def histogram_equalization_yuv(image: np.ndarray) -> np.ndarray:
    """
    YUV 空间直方图均衡化 (OpenCV)

    在 YUV 色彩空间的 Y (亮度) 通道上均衡化, 保留色彩信息。

    Args:
        image: (H, W, C), uint8, RGB
    """
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def histogram_equalization_hsv(image: np.ndarray) -> np.ndarray:
    """
    HSV 空间直方图均衡化 (OpenCV)

    对 V (明度) 通道进行均衡化。

    Args:
        image: (H, W, C), uint8, RGB
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adaptive_histogram_equalization(image: np.ndarray,
                                    clip_limit: float = 2.0,
                                    kernel_size: int = None) -> np.ndarray:
    """
    scikit-image 自适应直方图均衡化

    Args:
        image: (H, W, C), uint8, RGB
        clip_limit: 对比度限制
        kernel_size: 自适应窗口大小, 默认为图像最小维度的 1/4
    """
    if kernel_size is None:
        kernel_size = max(2, min(image.shape[0], image.shape[1]) // 4)
    # skimage 的 adapthist 输入范围需为 [0, 1]
    img_float = image.astype(np.float32) / 255.0
    result = exposure.equalize_adapthist(img_float, clip_limit=clip_limit,
                                          kernel_size=kernel_size)
    return (result * 255).astype(np.uint8)


# ============================================================
#  五、对比度增强 (增强)
# ============================================================

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma 校正 (scikit-image)

    gamma < 1: 提亮暗部; gamma > 1: 压暗亮部

    Args:
        image: (H, W, C), uint8
        gamma: 校正系数
    """
    return exposure.adjust_gamma(image, gamma=gamma).astype(np.uint8)


def adjust_log(image: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    对数变换, 增强暗部细节 (scikit-image)

    公式: gain * log(1 + x)

    Args:
        image: (H, W, C), uint8
        gain: 增益系数
    """
    return exposure.adjust_log(image, gain=gain).astype(np.uint8)


def adjust_sigmoid(image: np.ndarray, cutoff: float = 0.5,
                    gain: float = 10.0) -> np.ndarray:
    """
    Sigmoid 校正, 调整对比度曲线 (scikit-image)

    Args:
        image: (H, W, C), uint8
        cutoff: 截止点 (0~1), 控制曲线中心
        gain: 增益, 控制曲线陡峭程度
    """
    return exposure.adjust_sigmoid(image, cutoff=cutoff, gain=gain).astype(np.uint8)


def enhance_contrast_stretching(image: np.ndarray, percent_low: float = 2.0,
                                 percent_high: float = 98.0) -> np.ndarray:
    """
    对比度拉伸 / 百分比截断拉伸 (scikit-image)

    将像素值拉伸到 [0, 255], 裁剪掉 percent_low~percent_high 范围外的像素。

    Args:
        image: (H, W, C), uint8
        percent_low: 低端截断百分位
        percent_high: 高端截断百分位
    """
    p_low, p_high = np.percentile(image, (percent_low, percent_high))
    result = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
#  六、锐化 (增强)
# ============================================================

def usm_sharpen(image: np.ndarray, sigma: float = 1.0,
                amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    Unsharp Masking 锐化 (OpenCV)

    算法: sharpened = original + amount * (original - blurred)

    Args:
        image: (H, W, C), uint8, RGB
        sigma: 高斯模糊标准差, 控制锐化范围
        amount: 锐化强度 (0~5)
        threshold: 锐化阈值, 差异小于此值不锐化 (降噪效果)
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    # 阈值处理: 只锐化差异明显的区域
    if threshold > 0:
        diff = cv2.absdiff(image, blurred)
        mask = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY),
                              threshold, 255, cv2.THRESH_BINARY)[1]
        mask_3ch = cv2.merge([mask, mask, mask])
        sharpened = np.where(mask_3ch > 0, sharpened, image)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, kernel_size: int = 3,
                       strength: float = 1.0) -> np.ndarray:
    """
    拉普拉斯锐化 (OpenCV)

    使用拉普拉斯算子检测边缘, 叠加到原图上增强边缘。

    Args:
        image: (H, W, C), uint8
        kernel_size: 拉普拉斯核大小
        strength: 锐化强度
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    if kernel_size == 5:
        kernel = np.array([
            [0,  0, -1,  0,  0],
            [0, -1,  2, -1,  0],
            [-1, 2, -4,  2, -1],
            [0, -1,  2, -1,  0],
            [0,  0, -1,  0,  0]
        ], dtype=np.float32)

    # 对每个通道分别卷积
    sharpened = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        sharpened[:, :, c] = cv2.filter2D(
            image[:, :, c].astype(np.float32), cv2.CV_32F, kernel
        )

    result = image.astype(np.float32) + strength * sharpened
    return np.clip(result, 0, 255).astype(np.uint8)


def detail_enhance(image: np.ndarray, sigma_s: float = 10,
                    sigma_r: float = 0.15) -> np.ndarray:
    """
    细节增强 (OpenCV bilateralFilter 的高效实现)

    使用双边滤波保留边缘, 同时增强细节层。

    Args:
        image: (H, W, C), uint8
        sigma_s: 空间域标准差
        sigma_r: 值域标准差 (越小保留越多的边缘)
    """
    base = cv2.bilateralFilter(image, -1, sigma_s, sigma_r)
    detail = image.astype(np.float32) - base.astype(np.float32)
    result = image.astype(np.float32) + 1.5 * detail
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
#  七、去噪 (增强)
# ============================================================

def denoise_gaussian(image: np.ndarray, kernel_size: int = 3,
                      sigma: float = 1.0) -> np.ndarray:
    """
    高斯滤波去噪 (OpenCV)

    适用于去除高斯噪声, 但会平滑边缘。

    Args:
        image: (H, W, C), uint8
        kernel_size: 卷积核大小 (必须为奇数)
        sigma: 高斯核标准差
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def denoise_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    中值滤波去噪 (OpenCV)

    非常适合去除椒盐噪声, 同时较好地保留边缘。

    Args:
        image: (H, W, C), uint8
        kernel_size: 卷积核大小 (必须为奇数)
    """
    return cv2.medianBlur(image, kernel_size)


def denoise_bilateral(image: np.ndarray, d: int = 9,
                       sigma_color: float = 75,
                       sigma_space: float = 75) -> np.ndarray:
    """
    双边滤波去噪 (OpenCV)

    保边去噪: 空间距离近 + 颜色值相近的像素才参与滤波, 有效保留边缘。

    Args:
        image: (H, W, C), uint8
        d: 滤波邻域直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(image: np.ndarray, h: float = 10,
                 template_window: int = 7,
                 search_window: int = 21) -> np.ndarray:
    """
    非局部均值去噪 (OpenCV)

    搜索图像中相似的 patch 进行加权平均, 去噪效果最好但速度较慢。

    Args:
        image: (H, W, C), uint8
        h: 滤波强度, 值越大去噪越强但图像越模糊 (通常 10~30)
        template_window: 模板窗口大小 (奇数)
        search_window: 搜索窗口大小 (奇数)
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


def denoise_tv_chambolle(image: np.ndarray, weight: float = 0.1) -> np.ndarray:
    """
    TV (全变分) 去噪 (scikit-image)

    基于变分法的去噪方法, 能很好地保留图像边缘结构。

    Args:
        image: (H, W, C), uint8
        weight: 去噪权重, 越大去噪越强 (通常 0.1~0.3)
    """
    img_float = image.astype(np.float32) / 255.0
    denoised = restoration.denoise_tv_chambolle(img_float, weight=weight,
                                                 channel_axis=2)
    return (denoised * 255).astype(np.uint8)


def denoise_bayesian(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    贝叶斯阈值小波去噪 (scikit-image)

    使用贝叶斯收缩阈值, 适合处理高斯噪声。

    Args:
        image: (H, W, C), uint8
        threshold: 去噪阈值 (越低去噪越强)
    """
    img_float = image.astype(np.float32) / 255.0
    denoised = restoration.denoise_wavelet(img_float, method="BayesShrink",
                                            channel_axis=2,
                                            convert2ycbcr=True)
    return (denoised * 255).astype(np.uint8)


# ============================================================
#  八、形变 / 几何变换 (增强)
# ============================================================

def random_affine_transform(image: np.ndarray,
                             shear_range: float = 0.2,
                             scale_range: tuple = (0.85, 1.15),
                             translate_range: tuple = (0.1, 0.1),
                             target_size: int = None) -> np.ndarray:
    """
    随机仿射变换 (scikit-image)

    包含: 错切、缩放、平移的组合变换。

    Args:
        image: (H, W, C), uint8
        shear_range: 错切角度范围 (弧度)
        scale_range: 缩放范围
        translate_range: (tx, ty) 平移比例 (相对于图像尺寸)
        target_size: 输出尺寸, None 则保持原尺寸
    """
    h, w = image.shape[:2]
    transform_matrix = transform.AffineTransform(
        shear=random.uniform(-shear_range, shear_range),
        scale=random.uniform(*scale_range),
        translation=(
            random.uniform(-translate_range[0], translate_range[0]) * w,
            random.uniform(-translate_range[1], translate_range[1]) * h,
        ),
    )
    warped = transform.warp(image, transform_matrix, mode="reflect",
                             preserve_range=True)

    if target_size is not None:
        warped = cv2.resize(warped.astype(np.uint8), (target_size, target_size),
                             interpolation=cv2.INTER_CUBIC)

    return warped.astype(np.uint8)


def elastic_transform(image: np.ndarray,
                       alpha: float = 50.0,
                       sigma: float = 5.0) -> np.ndarray:
    """
    弹性形变 (OpenCV + numpy)

    生成随机的位移场, 使图像产生弹性扭曲效果,
    模拟鸟类的自然姿态变化。

    Args:
        image: (H, W, C), uint8
        alpha: 形变强度, 值越大扭曲越明显
        sigma: 高斯模糊标准差, 控制形变的平滑程度
    """
    h, w = image.shape[:2]

    # 生成随机位移场
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

    # 高斯模糊使位移场平滑
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # 创建网格坐标
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # 应用重映射
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
    return warped


def random_perspective_transform(image: np.ndarray,
                                  distortion_scale: float = 0.1,
                                  target_size: int = None) -> np.ndarray:
    """
    随机透视变换 (OpenCV)

    模拟不同拍摄角度带来的透视畸变。

    Args:
        image: (H, W, C), uint8
        distortion_scale: 畸变程度 (0~0.5)
        target_size: 输出尺寸
    """
    h, w = image.shape[:2]

    # 四个角的偏移量
    half_w, half_h = w // 2, h // 2
    d = distortion_scale

    # 源四角坐标
    src_pts = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])

    # 随机偏移后的四角坐标
    dst_pts = np.float32([
        [random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
        [w + random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
        [w + random.uniform(-d * w, d * w), h + random.uniform(-d * h, d * h)],
        [random.uniform(-d * w, d * w), h + random.uniform(-d * h, d * h)],
    ])

    # 透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT_101)

    if target_size is not None:
        warped = cv2.resize(warped, (target_size, target_size),
                             interpolation=cv2.INTER_CUBIC)
    return warped


def grid_distortion(image: np.ndarray, steps: int = 10,
                     distort_limit: float = 0.3) -> np.ndarray:
    """
    网格扭曲 (OpenCV)

    沿水平和垂直方向交替应用正弦波位移,
    产生类似水面折射的扭曲效果。

    Args:
        image: (H, W, C), uint8
        steps: 网格步数
        distort_limit: 扭曲幅度
    """
    h, w = image.shape[:2]

    # 水平方向扭曲
    x_step = w // steps
    distort_x = np.zeros((steps + 1, 1), dtype=np.float32)
    for i in range(steps + 1):
        distort_x[i, 0] = random.uniform(-distort_limit, distort_limit) * x_step

    # 垂直方向扭曲
    y_step = h // steps
    distort_y = np.zeros((steps + 1, 1), dtype=np.float32)
    for i in range(steps + 1):
        distort_y[i, 0] = random.uniform(-distort_limit, distort_limit) * y_step

    # 构建映射坐标
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)

    # 逐行/列添加扰动
    for i in range(steps + 1):
        y_start = i * y_step
        y_end = min((i + 1) * y_step, h)
        map_y[y_start:y_end, :] += distort_y[i]

        x_start = i * x_step
        x_end = min((i + 1) * x_step, w)
        map_x[:, x_start:x_end] += distort_x[i]

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
    return warped


def random_rotation(image: np.ndarray, max_angle: int = 15) -> np.ndarray:
    """
    随机旋转 (scikit-image)

    Args:
        image: (H, W, C), uint8
        max_angle: 最大旋转角度
    """
    if max_angle > 0:
        angle = random.uniform(-max_angle, max_angle)
        image = transform.rotate(image, angle, mode="reflect",
                                 preserve_range=True).astype(np.uint8)
    return image


# ============================================================
#  九、颜色增强
# ============================================================

def random_horizontal_flip(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """随机水平翻转 (OpenCV)"""
    if random.random() < prob:
        image = cv2.flip(image, 1)
    return image


def random_color_jitter(image: np.ndarray,
                         brightness: float = 0.2,
                         contrast: float = 0.2,
                         saturation: float = 0.2,
                         hue: float = 0.1) -> np.ndarray:
    """
    随机颜色抖动 (OpenCV HSV + scikit-image)

    Args:
        image: (H, W, C), uint8
        brightness: 亮度变化范围
        contrast: 对比度变化范围
        saturation: 饱和度变化范围
        hue: 色调变化范围
    """
    # HSV 空间操作: 色调 + 饱和度
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    if hue > 0:
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue) * 179) % 179

    if saturation > 0:
        scale = 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)

    hsv = hsv.astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 亮度 (scikit-image gamma)
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        image = exposure.adjust_gamma(image, gamma=1.0 / factor).astype(np.uint8)

    # 对比度 (scikit-image rescale_intensity)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        v_min, v_max = np.percentile(image, (2, 98))
        image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        mean_val = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip(
            (image - mean_val) * factor + mean_val, 0, 255
        ).astype(np.uint8)

    return image


def random_gaussian_noise(image: np.ndarray,
                           mean: float = 0,
                           std_range: tuple = (5, 25)) -> np.ndarray:
    """
    添加随机高斯噪声 (numpy)

    Args:
        image: (H, W, C), uint8
        mean: 噪声均值
        std_range: 噪声标准差随机范围
    """
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================
#  十、完整管线
# ============================================================

def apply_preprocessing(image: np.ndarray,
                         target_size: int = 224,
                         use_clahe: bool = True,
                         use_letterbox: bool = False) -> np.ndarray:
    """
    推理时的标准预处理管线

    流程: CLAHE -> 缩放 -> 归一化 -> 转 Tensor

    Args:
        image: (H, W, C), uint8, RGB
        target_size: 目标尺寸
        use_clahe: 是否应用 CLAHE 直方图均衡化
        use_letterbox: 是否使用 Letterbox 保持长宽比

    Returns:
        (C, H, W), float32
    """
    # 1. CLAHE 对比度增强
    if use_clahe:
        image = apply_clahe(image, clip_limit=2.0)

    # 2. 尺寸调整
    if use_letterbox:
        image = resize_keep_aspect_ratio(image, target_size)
    else:
        image = resize_image(image, target_size)

    # 3. 归一化
    image = normalize_image(image)

    # 4. 转 Tensor
    image = to_tensor(image)

    return image


def apply_data_augmentation(image: np.ndarray) -> np.ndarray:
    """
    训练时的完整数据增强管线

    流程:
    1. CLAHE 对比度增强
    2. 随机裁剪并缩放
    3. 随机仿射变换 (错切/缩放)
    4. 随机水平翻转
    5. 随机颜色抖动 (亮度/对比度/饱和度/色调)
    6. 随机弹性形变
    7. 随机轻微旋转
    8. 随机高斯噪声

    Args:
        image: (H, W, C), uint8, RGB

    Returns:
        增强后的图像 (H, W, C), uint8
    """
    # 1. CLAHE 对比度增强
    image = apply_clahe(image, clip_limit=2.0)

    # 2. 随机裁剪并缩放
    h, w = image.shape[:2]
    scale = random.uniform(0.8, 1.0)
    crop_h, crop_w = int(h * scale), int(w * scale)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    image = image[y:y + crop_h, x:x + crop_w]
    image = cv2.resize(image, (config.INPUT_SIZE, config.INPUT_SIZE),
                        interpolation=cv2.INTER_CUBIC)

    # 3. 随机仿射变换
    if random.random() < 0.3:
        image = random_affine_transform(
            image, shear_range=0.15, scale_range=(0.9, 1.1),
            translate_range=(0.05, 0.05), target_size=config.INPUT_SIZE,
        )

    # 4. 随机水平翻转
    image = random_horizontal_flip(image, prob=config.RANDOM_FLIP_PROB)

    # 5. 随机颜色抖动
    jitter = config.COLOR_JITTER
    image = random_color_jitter(
        image,
        brightness=jitter[0], contrast=jitter[1],
        saturation=jitter[2], hue=jitter[3],
    )

    # 6. 随机弹性形变 (低概率, 计算开销大)
    if random.random() < 0.2:
        image = elastic_transform(image, alpha=30.0, sigma=4.0)

    # 7. 随机轻微旋转
    if random.random() < 0.5:
        image = random_rotation(image, max_angle=10)

    # 8. 随机高斯噪声
    if random.random() < 0.2:
        image = random_gaussian_noise(image, std_range=(3, 15))

    return image
