"""
外挂翻译模块 (i18n)

从 CUB-Hierarchy/translations_zh.json 加载中文翻译。
查不到则回退到英文原名，不抛异常。

用法:
    from src.i18n import I18n
    i18n = I18n.load("CUB-Hierarchy/")
    chinese = i18n.t("order", "Passeriformes")  # → "雀形目"
"""

import os
import json
from typing import Dict, Optional


class I18n:
    """翻译管理器（单例）"""

    _instance: Optional["I18n"] = None

    @classmethod
    def get(cls) -> "I18n":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def load(cls, hierarchy_dir: str) -> "I18n":
        """从 hierarchy_dir 加载 translations_zh.json"""
        inst = cls.get()
        inst._translations: Dict[str, Dict[str, str]] = {}
        path = os.path.join(hierarchy_dir, "translations_zh.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                inst._translations = json.load(f)
        return inst

    def t(self, level: str, name: str) -> str:
        """翻译单个名称，无翻译则返回原名"""
        return self._translations.get(level, {}).get(name, name)
