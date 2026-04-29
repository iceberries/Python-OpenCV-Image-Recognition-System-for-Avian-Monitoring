"""
模型管理器 - 单例模式

功能:
  - 异步加载 PyTorch 模型（不阻塞 UI）
  - 单图/批量推理接口
  - 模型热切换（切换权重文件重新加载）
  - 加载状态信号
  - GPU OOM 自动降级 CPU
  - Grad-CAM 热力图生成（可选）
"""
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from src.config import OUTPUT_DIR, NUM_CLASSES, INPUT_SIZE, USE_SE_ATTENTION
from src.preprocessing import resize_image, normalize_image, to_tensor, apply_clahe
from src.utils.model_utils import load_class_names, count_parameters, get_gpu_memory_info


class ModelLoadWorker(QThread):
    """异步模型加载线程"""
    progress = pyqtSignal(int)            # 加载进度 0~100
    finished = pyqtSignal(bool, str)      # (success, message)

    def __init__(self, manager: "ModelManager", checkpoint_path: str, device: str):
        super().__init__()
        self.manager = manager
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self):
        try:
            self.progress.emit(10)
            # 构建模型
            from src.model import ResNet50BirdClassifier
            model = ResNet50BirdClassifier(
                num_classes=NUM_CLASSES,
                pretrained=False,
                use_se=USE_SE_ATTENTION,
            )
            self.progress.emit(40)

            # 加载权重
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            self.progress.emit(70)

            # 移到目标设备
            model.to(self.device)
            model.eval()
            self.progress.emit(85)

            # 计算参数量
            param_info = count_parameters(model)

            # 更新 manager 状态
            old_model = self.manager._model
            self.manager._model = model
            self.manager._device = self.device
            self.manager._param_info = param_info
            self.manager._checkpoint_meta = {
                "best_accuracy": checkpoint.get("best_accuracy", "N/A"),
                "best_epoch": checkpoint.get("best_epoch", "N/A"),
            }

            # 释放旧模型
            if old_model is not None:
                del old_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.progress.emit(100)
            self.finished.emit(True, f"模型加载成功: {os.path.basename(self.checkpoint_path)}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # GPU OOM, 尝试降级到 CPU
                try:
                    self.device = "cpu"
                    # 重新执行（递归只允许一次降级）
                    self.run_cpu_fallback()
                except Exception as e2:
                    self.finished.emit(False, f"GPU OOM 且 CPU 降级失败: {e2}")
            else:
                self.finished.emit(False, f"模型加载失败: {e}")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {e}")

    def run_cpu_fallback(self):
        """GPU OOM 后的 CPU 降级"""
        from src.model import ResNet50BirdClassifier
        model = ResNet50BirdClassifier(
            num_classes=NUM_CLASSES,
            pretrained=False,
            use_se=USE_SE_ATTENTION,
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to("cpu")
        model.eval()

        old_model = self.manager._model
        self.manager._model = model
        self.manager._device = "cpu"
        self.manager._param_info = count_parameters(model)
        self.manager._checkpoint_meta = {
            "best_accuracy": checkpoint.get("best_accuracy", "N/A"),
            "best_epoch": checkpoint.get("best_epoch", "N/A"),
        }
        if old_model is not None:
            del old_model
        torch.cuda.empty_cache()

        self.progress.emit(100)
        self.finished.emit(True, "模型加载成功 (GPU OOM, 已降级到 CPU)")


class ModelManager(QObject):
    """
    模型管理器（单例）

    - 异步加载模型
    - 单图/批量推理
    - 模型热切换
    - GPU OOM 自动降级
    """

    # 信号
    modelLoaded = pyqtSignal(bool, str)           # (success, message)
    modelLoadingProgress = pyqtSignal(int)         # (percent 0~100)

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = cls.__new__(cls)
                    super(ModelManager, obj).__init__()
                    obj._model = None
                    obj._device = "cpu"
                    obj._class_names = []
                    obj._param_info = {"total": 0, "trainable": 0}
                    obj._checkpoint_meta = {}
                    obj._load_worker = None
                    obj._grad_cam = None
                    obj._infer_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    # ==================== 模型加载 ====================

    def load_model_async(self, checkpoint_path: str = None, device: str = None):
        """
        异步加载模型（不阻塞 UI）

        Args:
            checkpoint_path: 权重文件路径，默认 output/best_model.pth
            device: 推理设备，默认自动检测
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            self.modelLoaded.emit(False, f"模型文件不存在: {checkpoint_path}")
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 取消正在进行的加载
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.quit()
            self._load_worker.wait(3000)

        # 加载类别名
        self._class_names = load_class_names()

        self._load_worker = ModelLoadWorker(self, checkpoint_path, device)
        self._load_worker.progress.connect(self.modelLoadingProgress.emit)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.start()

    def _on_load_finished(self, success: bool, message: str):
        """加载完成回调"""
        from src.core.app_state import AppState
        state = AppState.get()
        state.is_model_ready = success
        if success:
            state.current_model_name = message
            # 初始化 Grad-CAM（延迟，按需）
            self._grad_cam = None
        self.modelLoaded.emit(success, message)

    def load_model_sync(self, checkpoint_path: str = None, device: str = None) -> Tuple[bool, str]:
        """
        同步加载模型（阻塞调用，仅用于初始化或测试）

        Returns:
            (success, message)
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            return False, f"模型文件不存在: {checkpoint_path}"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from src.model import ResNet50BirdClassifier
            model = ResNet50BirdClassifier(
                num_classes=NUM_CLASSES,
                pretrained=False,
                use_se=USE_SE_ATTENTION,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            old_model = self._model
            self._model = model
            self._device = device
            self._param_info = count_parameters(model)
            self._checkpoint_meta = {
                "best_accuracy": checkpoint.get("best_accuracy", "N/A"),
                "best_epoch": checkpoint.get("best_epoch", "N/A"),
            }
            self._class_names = load_class_names()
            self._grad_cam = None

            if old_model is not None:
                del old_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            from src.core.app_state import AppState
            AppState.get().is_model_ready = True
            AppState.get().current_model_name = os.path.basename(checkpoint_path)
            self.modelLoaded.emit(True, f"模型加载成功: {os.path.basename(checkpoint_path)}")
            return True, f"模型加载成功: {os.path.basename(checkpoint_path)}"

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                return self.load_model_sync(checkpoint_path, device="cpu")
            return False, f"模型加载失败: {e}"
        except Exception as e:
            return False, f"模型加载失败: {e}"

    # ==================== 推理接口 ====================

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def param_info(self) -> Dict[str, int]:
        return self._param_info

    @property
    def checkpoint_meta(self) -> Dict:
        return self._checkpoint_meta

    def preprocess(self, image: np.ndarray, use_clahe: bool = True) -> torch.Tensor:
        """
        图像预处理: RGB uint8 (H,W,3) -> Tensor (1,C,H,W)

        Args:
            image: RGB 图像, uint8
            use_clahe: 是否使用 CLAHE 对比度增强

        Returns:
            (1, C, H, W) float32 tensor
        """
        if use_clahe:
            image = apply_clahe(image, clip_limit=2.0)
        image = resize_image(image, INPUT_SIZE)
        image = normalize_image(image)
        image = to_tensor(image)
        return torch.from_numpy(image).unsqueeze(0).float()

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5,
        use_clahe: bool = True,
        return_heatmap: bool = False,
    ) -> Dict:
        """
        单图推理

        Args:
            image: RGB 图像 (H, W, 3), uint8
            top_k: 返回 Top-K 结果
            use_clahe: 是否 CLAHE 预处理
            return_heatmap: 是否生成 Grad-CAM 热力图

        Returns:
            {
                "class_name": str,
                "confidence": float (0~1),
                "top_k": [{"class_name": str, "confidence": float}, ...],
                "latency_ms": float,
                "heatmap": Optional[np.ndarray (H,W) float [0,1]],
            }
        """
        if self._model is None:
            raise RuntimeError("模型未加载")

        t0 = time.perf_counter()

        # 预处理
        input_tensor = self.preprocess(image, use_clahe=use_clahe)
        input_tensor = input_tensor.to(self._device)

        # 推理
        with torch.no_grad():
            output = self._model(input_tensor)
            probs = F.softmax(output, dim=1)

        # 后处理
        probs_np = probs.cpu().numpy()[0]
        top_indices = np.argsort(probs_np)[::-1][:top_k]

        top_k_results = []
        for idx in top_indices:
            name = self._class_names[idx] if idx < len(self._class_names) else f"Class_{idx}"
            top_k_results.append({
                "class_name": name,
                "confidence": float(probs_np[idx]),
            })

        predicted_class = top_k_results[0]["class_name"] if top_k_results else "Unknown"
        confidence = top_k_results[0]["confidence"] if top_k_results else 0.0
        latency_ms = (time.perf_counter() - t0) * 1000

        # Grad-CAM 热力图（可选）
        heatmap = None
        if return_heatmap:
            heatmap = self._generate_heatmap(image, input_tensor)

        # 清理 GPU 缓存
        if self._device == "cuda":
            torch.cuda.empty_cache()

        return {
            "class_name": predicted_class,
            "confidence": confidence,
            "top_k": top_k_results,
            "latency_ms": latency_ms,
            "heatmap": heatmap,
        }

    def predict_batch(
        self,
        images: List[np.ndarray],
        top_k: int = 5,
        use_clahe: bool = True,
    ) -> List[Dict]:
        """
        批量推理

        Args:
            images: RGB 图像列表
            top_k: 返回 Top-K
            use_clahe: 是否 CLAHE

        Returns:
            每张图的结果字典列表
        """
        results = []
        for img in images:
            try:
                result = self.predict(img, top_k=top_k, use_clahe=use_clahe)
                result["error"] = None
            except Exception as e:
                result = {
                    "class_name": "Error",
                    "confidence": 0.0,
                    "top_k": [],
                    "latency_ms": 0.0,
                    "heatmap": None,
                    "error": str(e),
                }
            results.append(result)
        return results

    def _generate_heatmap(
        self, original_image: np.ndarray, input_tensor: torch.Tensor
    ) -> Optional[np.ndarray]:
        """
        生成 Grad-CAM 热力图

        Args:
            original_image: 原始 RGB 图像 (H, W, 3)
            input_tensor: 预处理后的输入 (1, C, H, W)

        Returns:
            热力图 (H, W) float32 [0, 1]，失败返回 None
        """
        try:
            if self._grad_cam is None:
                from src.attention_deform import GradCAM
                # 尝试获取 backbone 的 layer4
                if hasattr(self._model, 'backbone'):
                    target = self._model.backbone.layer4
                else:
                    target = None
                self._grad_cam = GradCAM(self._model, target_layer=target)

            heatmap = self._grad_cam.generate_upsampled(
                input_tensor.clone().detach().to(self._device),
                target_size=(original_image.shape[1], original_image.shape[0]),
            )
            return heatmap
        except Exception:
            # Grad-CAM 失败不阻塞推理
            return None

    def get_model_info(self) -> Dict:
        """
        获取模型信息摘要

        Returns:
            模型信息字典
        """
        info = {
            "model_name": "ResNet50 + SE-Attention",
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "device": self._device,
            "use_se": USE_SE_ATTENTION,
            "param_total": self._param_info.get("total", 0),
            "param_trainable": self._param_info.get("trainable", 0),
            "class_count": len(self._class_names),
            "checkpoint_meta": self._checkpoint_meta,
            "gpu_info": get_gpu_memory_info(),
        }
        return info

    def release(self):
        """释放模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
        self._grad_cam = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from src.core.app_state import AppState
        AppState.get().is_model_ready = False
