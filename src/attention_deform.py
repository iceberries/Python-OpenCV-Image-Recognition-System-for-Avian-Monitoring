"""
注意力引导的定向形变模块

核心思路:
    1. 使用 Grad-CAM 从 ResNet50 提取注意力热力图 (关注哪些区域对分类贡献大)
    2. 用注意力权重调制形变强度:
       - 注意力高的区域 (鸟的主体): 形变强度低, 保持关键特征
       - 注意力低的区域 (背景/边缘): 形变强度高, 增加多样性
    3. 支持: 注意力引导弹性形变、注意力引导局部遮挡、注意力引导裁剪

使用方式:
    # 作为数据增强插件
    from src.attention_deform import AttentionGuidedAugmentor

    augmentor = AttentionGuidedAugmentor(model, device="cpu")
    augmented_image = augmentor(image_np)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  一、Grad-CAM 注意力热力图生成
# ============================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping

    利用目标层 (最后一个卷积层) 的梯度和激活值,
    生成空间注意力热力图, 指示图像中对分类决策贡献大的区域。

    Args:
        model: PyTorch 模型 (ResNet50)
        target_layer: 目标卷积层, 默认取 ResNet50 的 layer4
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # 默认使用 ResNet50 的最后一个残差块
        if target_layer is None:
            # ResNet50 的 layer4 是最后一组残差块, 输出 2048 通道
            target_layer = self._find_last_conv_layer(model)
            if target_layer is None:
                raise ValueError("无法自动定位最后一个卷积层, 请手动指定 target_layer")

        self.target_layer = target_layer
        self._register_hooks()

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """递归查找模型中最后一个卷积层 (通常是 ResNet 的 layer4)"""
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        """注册前向/反向钩子, 捕获激活值和梯度"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                  target_class: int = None) -> np.ndarray:
        """
        生成 Grad-CAM 注意力热力图

        Args:
            input_tensor: (1, C, H, W), 归一化后的图像 Tensor
            target_class: 目标类别索引, None 则使用模型预测的类别

        Returns:
            attention_map: (H, W), float32, 范围 [0, 1], 值越大表示注意力越强
        """
        # 前向传播
        output = self.model(input_tensor)

        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播: 计算目标类别的梯度
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM: 加权求和
        # weights = global_avg_pool(gradients)  -> (1, C, 1, 1)
        # cam = weights * activations            -> (1, C, H', W')
        # cam = sum(cam, axis=C)                -> (1, H', W')
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU: 只保留正向贡献
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_upsampled(self, input_tensor: torch.Tensor,
                            target_size: tuple,
                            target_class: int = None) -> np.ndarray:
        """
        生成并上采样到指定尺寸的注意力热力图

        Args:
            input_tensor: (1, C, H, W)
            target_size: (width, height)
            target_class: 目标类别

        Returns:
            attention_map: (height, width), [0, 1]
        """
        cam = self.generate(input_tensor, target_class)
        cam_upsampled = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
        return cam_upsampled


# ============================================================
#  二、注意力引导的弹性形变
# ============================================================

def attention_guided_elastic_deform(image: np.ndarray,
                                     attention_map: np.ndarray,
                                     alpha: float = 40.0,
                                     sigma: float = 5.0,
                                     mode: str = "inverse") -> np.ndarray:
    """
    注意力引导的弹性形变

    思路: 用注意力图调制位移场的强度
    - mode="inverse":  注意力高的区域形变小 (保护关键特征), 背景形变大
    - mode="direct":   注意力高的区域形变大 (增加关键区域的多样性, 更难的正样本)
    - mode="balanced": 所有区域适度形变, 注意力区域略强

    Args:
        image: (H, W, C), uint8, RGB
        attention_map: (H, W), float32, [0, 1], 1=高注意力
        alpha: 形变强度基数
        sigma: 高斯模糊标准差
        mode: "inverse" / "direct" / "balanced"

    Returns:
        形变后的图像 (H, W, C), uint8
    """
    h, w = image.shape[:2]

    # 确保注意力图与图像同尺寸
    if attention_map.shape[:2] != (h, w):
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 生成基础随机位移场
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

    # 高斯平滑位移场
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    # 根据注意力图调制位移强度
    if mode == "inverse":
        # 注意力高 -> 位移小, 注意力低 -> 位移大
        # weight = 1 - attention (背景区域全形变, 主体区域保护)
        weight = 1.0 - attention_map
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        # 保留最小 20% 的形变, 避免完全不动
        weight = 0.2 + 0.8 * weight
    elif mode == "direct":
        # 注意力高 -> 位移大 (增加关键区域难度)
        weight = attention_map.copy()
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        weight = 0.2 + 0.8 * weight
    else:  # balanced
        weight = np.ones_like(attention_map) * 0.7

    # 应用调制权重
    dx = dx * weight * alpha
    dy = dy * weight * alpha

    # 再次平滑, 消除权重边界处的跳变
    dx = cv2.GaussianBlur(dx, (0, 0), sigma * 0.5)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma * 0.5)

    # 创建映射坐标并重映射
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101)
    return warped


# ============================================================
#  三、注意力引导的局部遮挡
# ============================================================

def attention_guided_occlusion(image: np.ndarray,
                                attention_map: np.ndarray,
                                occlusion_ratio: float = 0.15,
                                mode: str = "background") -> np.ndarray:
    """
    注意力引导的局部遮挡 (CutOut / Hide-and-Seek)

    - mode="background": 遮挡注意力低的区域 (背景), 让模型聚焦主体
    - mode="foreground": 遮挡注意力高的区域 (主体), 训练模型利用全局信息
    - mode="mixed":      随机遮挡前景或背景

    Args:
        image: (H, W, C), uint8
        attention_map: (H, W), [0, 1]
        occlusion_ratio: 遮挡面积占图像的比例
        mode: "background" / "foreground" / "mixed"

    Returns:
        遮挡后的图像
    """
    h, w = image.shape[:2]
    result = image.copy()

    if attention_map.shape[:2] != (h, w):
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 决定遮挡模式
    if mode == "mixed":
        mode = random.choice(["background", "foreground"])

    # 构建选择概率图
    if mode == "background":
        prob_map = 1.0 - attention_map  # 背景概率高
    else:
        prob_map = attention_map.copy()  # 前景概率高

    # 归一化
    prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)

    # 生成遮挡区域: 在概率高的区域放置遮挡块
    total_pixels = h * w
    occlude_pixels = int(total_pixels * occlusion_ratio)

    # 将概率图展平并按概率排序
    flat_prob = prob_map.flatten()
    # 采样: 按概率加权随机选取像素
    indices = np.random.choice(
        total_pixels,
        size=min(occlude_pixels, total_pixels),
        replace=False,
        p=flat_prob / flat_prob.sum()
    )

    # 用随机块遮挡 (而非单像素, 更真实)
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[indices] = True

    # 形态学膨胀, 将离散像素扩展为块状区域
    kernel_size = max(3, int(np.sqrt(occlude_pixels) * 0.1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (kernel_size, kernel_size))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    # 用均值颜色填充遮挡区域
    mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
    result[mask] = mean_color

    return result


# ============================================================
#  四、注意力引导的仿射变换
# ============================================================

def attention_guided_affine(image: np.ndarray,
                             attention_map: np.ndarray,
                             max_angle: float = 10.0,
                             max_scale: float = 0.15,
                             max_translate: float = 0.05) -> np.ndarray:
    """
    注意力引导的仿射变换

    生成多个候选仿射变换, 选择对注意力区域影响最小的那个,
    即变换后注意力区域的像素值变化最小。

    Args:
        image: (H, W, C), uint8
        attention_map: (H, W), [0, 1]
        max_angle: 最大旋转角度
        max_scale: 最大缩放比例
        max_translate: 最大平移比例

    Returns:
        变换后的图像
    """
    h, w = image.shape[:2]
    if attention_map.shape[:2] != (h, w):
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 生成 N 个候选变换
    best_warped = None
    min_cost = float("inf")

    for _ in range(5):
        angle = random.uniform(-max_angle, max_angle)
        scale = 1.0 + random.uniform(-max_scale, max_scale)
        tx = random.uniform(-max_translate, max_translate) * w
        ty = random.uniform(-max_translate, max_translate) * h

        # 仿射变换矩阵
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[:, 2] += [tx, ty]

        warped = cv2.warpAffine(image, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)

        # 评估代价: 注意力区域的像素差异 (越小越好, 表示关键区域被保留得越好)
        warped_attn = cv2.warpAffine(attention_map.astype(np.float32), M, (w, h),
                                      borderMode=cv2.BORDER_REFLECT_101)
        cost = -np.sum(attention_map * warped_attn)  # 负的互相关

        if cost < min_cost:
            min_cost = cost
            best_warped = warped

    return best_warped


# ============================================================
#  五、注意力热力图可视化
# ============================================================

def visualize_attention(image: np.ndarray,
                         attention_map: np.ndarray,
                         alpha: float = 0.4) -> np.ndarray:
    """
    将注意力热力图叠加到原图上 (OpenCV)

    Args:
        image: (H, W, C), uint8, RGB
        attention_map: (H, W), float32, [0, 1]
        alpha: 热力图透明度

    Returns:
        叠加后的可视化图像 (H, W, C), uint8, RGB
    """
    h, w = image.shape[:2]
    if attention_map.shape[:2] != (h, w):
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 生成伪彩色热力图 (蓝->绿->黄->红)
    heatmap = (attention_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加
    result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return result


# ============================================================
#  六、完整的注意力引导增强器
# ============================================================

class AttentionGuidedAugmentor:
    """
    注意力引导的定向形变增强器

    将 Grad-CAM 注意力与形变操作结合, 实现智能数据增强。

    使用方式:
        model = ResNet50BirdClassifier(pretrained=True)
        augmentor = AttentionGuidedAugmentor(model, device="cpu")

        for image in images:
            augmented = augmentor.elastic(image)          # 注意力引导弹性形变
            augmented = augmentor.occlusion(image)         # 注意力引导遮挡
            augmented = augmentor.affine(image)            # 注意力引导仿射
            augmented = augmentor.auto(image)              # 随机选择一种
            heatmap = augmentor.get_heatmap(image)         # 获取注意力热力图
            vis = augmentor.visualize(image)               # 可视化注意力叠加

    Args:
        model: PyTorch 模型 (需能前向传播)
        device: 推理设备
        target_size: 输入图像尺寸
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                  target_size: int = 224):
        self.device = device
        self.target_size = target_size
        self.model = model.to(device).eval()

        # 创建 Grad-CAM
        self.grad_cam = GradCAM(self.model)

        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像为模型输入 Tensor"""
        # 缩放到目标尺寸
        if image.shape[:2] != (self.target_size, self.target_size):
            image = cv2.resize(image, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_CUBIC)
        # 归一化
        img_float = image.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def get_attention_map(self, image: np.ndarray) -> np.ndarray:
        """
        获取图像的注意力热力图

        Args:
            image: (H, W, C), uint8, RGB

        Returns:
            attention_map: (H, W), [0, 1]
        """
        input_tensor = self._preprocess(image)
        with torch.no_grad():
            # 先获取预测类别
            output = self.model(input_tensor)
            target_class = output.argmax(dim=1).item()

        # 需要 enable grad 来计算 Grad-CAM
        input_tensor.requires_grad = True
        attention_map = self.grad_cam.generate_upsampled(
            input_tensor,
            target_size=(image.shape[1], image.shape[0]),
            target_class=target_class,
        )
        return attention_map

    def elastic(self, image: np.ndarray,
                 alpha: float = 40.0, sigma: float = 5.0,
                 mode: str = "inverse") -> np.ndarray:
        """
        注意力引导弹性形变

        Args:
            image: (H, W, C), uint8
            alpha: 形变强度
            sigma: 平滑度
            mode: "inverse"(保护主体) / "direct"(增强主体难度)

        Returns:
            形变后的图像
        """
        attention_map = self.get_attention_map(image)
        return attention_guided_elastic_deform(
            image, attention_map, alpha=alpha, sigma=sigma, mode=mode
        )

    def occlusion(self, image: np.ndarray,
                   occlusion_ratio: float = 0.15,
                   mode: str = "background") -> np.ndarray:
        """
        注意力引导遮挡

        Args:
            image: (H, W, C), uint8
            occlusion_ratio: 遮挡面积比例
            mode: "background"(遮背景) / "foreground"(遮主体) / "mixed"

        Returns:
            遮挡后的图像
        """
        attention_map = self.get_attention_map(image)
        return attention_guided_occlusion(
            image, attention_map,
            occlusion_ratio=occlusion_ratio,
            mode=mode,
        )

    def affine(self, image: np.ndarray,
               max_angle: float = 10.0,
               max_scale: float = 0.15,
               max_translate: float = 0.05) -> np.ndarray:
        """
        注意力引导仿射变换

        Args:
            image: (H, W, C), uint8
            max_angle: 最大旋转角度
            max_scale: 最大缩放
            max_translate: 最大平移

        Returns:
            变换后的图像
        """
        attention_map = self.get_attention_map(image)
        return attention_guided_affine(
            image, attention_map,
            max_angle=max_angle,
            max_scale=max_scale,
            max_translate=max_translate,
        )

    def auto(self, image: np.ndarray) -> np.ndarray:
        """
        随机选择一种注意力引导增强

        随机选择: 弹性形变(inverse) / 遮挡(background) / 遮挡(foreground) / 仿射
        """
        aug_fn = random.choice([
            lambda: self.elastic(image, mode="inverse"),
            lambda: self.elastic(image, mode="direct"),
            lambda: self.occlusion(image, mode="background"),
            lambda: self.occlusion(image, mode="foreground"),
            lambda: self.affine(image),
        ])
        return aug_fn()

    def get_heatmap(self, image: np.ndarray) -> np.ndarray:
        """获取原始注意力热力图 (灰度)"""
        return (self.get_attention_map(image) * 255).astype(np.uint8)

    def visualize(self, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """获取注意力热力图叠加可视化 (彩色)"""
        attention_map = self.get_attention_map(image)
        return visualize_attention(image, attention_map, alpha=alpha)
