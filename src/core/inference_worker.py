"""
推理工作线程 - 避免 UI 卡顿

支持:
  - 单图模式: run_single
  - 批量模式: run_batch
  - 安全取消（QMutex + bool 标志位）
  - 推理完成后自动释放 GPU 显存
"""
import gc
import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QMutex, pyqtSignal

from src.core.model_manager import ModelManager


class InferenceWorker(QThread):
    """
    推理工作线程

    信号:
        singleResultReady(result: dict, original_image: np.ndarray)
        batchProgress(current: int, total: int, result: dict)
        batchFinished(results: List[dict])
        inferenceError(error_msg: str)
    """

    # 信号
    singleResultReady = pyqtSignal(dict, object)      # result, original_image
    batchProgress = pyqtSignal(int, int, dict)         # current, total, result
    batchFinished = pyqtSignal(list)                   # results list
    inferenceError = pyqtSignal(str)                   # error_msg

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._cancelled = False
        self._mode = None  # "single" or "batch"

        # 单图参数
        self._single_image: Optional[np.ndarray] = None
        self._return_heatmap: bool = False

        # 批量参数
        self._batch_images: List[np.ndarray] = []
        self._batch_filenames: List[str] = []

        # 推理参数
        self._top_k: int = 5
        self._use_clahe: bool = True

    # ==================== 配置方法 ====================

    def setup_single(
        self,
        image: np.ndarray,
        top_k: int = 5,
        use_clahe: bool = True,
        return_heatmap: bool = False,
    ):
        """配置单图推理"""
        self._mode = "single"
        self._single_image = image
        self._top_k = top_k
        self._use_clahe = use_clahe
        self._return_heatmap = return_heatmap

    def setup_batch(
        self,
        images: List[np.ndarray],
        filenames: List[str] = None,
        top_k: int = 5,
        use_clahe: bool = True,
    ):
        """配置批量推理"""
        self._mode = "batch"
        self._batch_images = images
        self._batch_filenames = filenames or [f"image_{i}" for i in range(len(images))]
        self._top_k = top_k
        self._use_clahe = use_clahe

    # ==================== 取消 ====================

    def cancel(self):
        """安全取消推理"""
        self._mutex.lock()
        self._cancelled = True
        self._mutex.unlock()

    def _is_cancelled(self) -> bool:
        self._mutex.lock()
        val = self._cancelled
        self._mutex.unlock()
        return val

    # ==================== 运行 ====================

    def run(self):
        if self._mode == "single":
            self._run_single()
        elif self._mode == "batch":
            self._run_batch()

    def _run_single(self):
        """单图推理"""
        try:
            manager = ModelManager.get()
            if not manager.is_ready:
                self.inferenceError.emit("模型未加载，请先加载模型")
                return

            result = manager.predict(
                self._single_image,
                top_k=self._top_k,
                use_clahe=self._use_clahe,
                return_heatmap=self._return_heatmap,
            )

            # 添加延迟明细
            latency_ms = result.get("latency_ms", 0)
            result["latency"] = {
                "preprocess": 0,
                "inference": latency_ms,
                "postprocess": 0,
            }

            if not self._is_cancelled():
                self.singleResultReady.emit(result, self._single_image)

        except Exception as e:
            if not self._is_cancelled():
                self.inferenceError.emit(f"推理失败: {e}")
        finally:
            self._cleanup()

    def _run_batch(self):
        """批量推理"""
        results = []
        total = len(self._batch_images)
        manager = ModelManager.get()

        if not manager.is_ready:
            self.inferenceError.emit("模型未加载，请先加载模型")
            return

        for i, image in enumerate(self._batch_images):
            if self._is_cancelled():
                break

            try:
                result = manager.predict(
                    image,
                    top_k=self._top_k,
                    use_clahe=self._use_clahe,
                    return_heatmap=False,
                )
                result["filename"] = self._batch_filenames[i] if i < len(self._batch_filenames) else f"image_{i}"
                result["latency"] = {
                    "preprocess": 0,
                    "inference": result.get("latency_ms", 0),
                    "postprocess": 0,
                }
                result["error"] = None

            except Exception as e:
                result = {
                    "filename": self._batch_filenames[i] if i < len(self._batch_filenames) else f"image_{i}",
                    "class_name": "Error",
                    "confidence": 0.0,
                    "top_k": [],
                    "latency_ms": 0.0,
                    "latency": {"preprocess": 0, "inference": 0, "postprocess": 0},
                    "error": str(e),
                }

            results.append(result)

            if not self._is_cancelled():
                self.batchProgress.emit(i + 1, total, result)

            # 每处理完一张释放内存
            del image
            if i % 10 == 9:
                gc.collect()

        if not self._is_cancelled():
            self.batchFinished.emit(results)

        self._cleanup()

    def _cleanup(self):
        """清理资源"""
        # 释放引用
        self._single_image = None
        self._batch_images = []
        self._batch_filenames = []
        gc.collect()

        # 释放 GPU 显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
