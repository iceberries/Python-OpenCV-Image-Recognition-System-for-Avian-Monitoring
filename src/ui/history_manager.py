"""
历史记录管理器 - 数据层 + 业务层

功能:
  - HistoryRecord 数据类
  - SQLite3 持久化存储（线程安全）
  - CRUD、条件查询、分页、排序
  - CSV/JSON 导出
  - 统计概览
  - 自动清理过期记录
  - 缩略图按日期目录存储
"""
import csv
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize


# ==================== 数据模型 ====================

@dataclass
class HistoryRecord:
    """历史记录数据类"""
    id: Optional[int] = None
    timestamp: str = ""               # ISO 格式
    filename: str = ""
    image_path: str = ""              # 原始图片路径
    thumbnail_path: str = ""          # 缩略图路径
    predicted_class: str = ""
    confidence: float = 0.0
    top_k_results: str = "[]"         # JSON 字符串
    notes: str = ""

    def get_top_k(self) -> List[Dict]:
        """解析 top_k_results JSON"""
        try:
            return json.loads(self.top_k_results)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_top_k(self, results: List[Dict]):
        """设置 top_k_results 为 JSON"""
        self.top_k_results = json.dumps(results, ensure_ascii=False)


# ==================== 数据库管理 ====================

class HistoryDB:
    """
    SQLite 数据库管理（线程安全）

    - 元数据存 data/history.db
    - 缩略图存 data/thumbnails/YYYY-MM/
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        filename        TEXT    NOT NULL DEFAULT '',
        image_path      TEXT    NOT NULL DEFAULT '',
        thumbnail_path  TEXT    NOT NULL DEFAULT '',
        predicted_class TEXT    NOT NULL DEFAULT '',
        confidence      REAL    NOT NULL DEFAULT 0.0,
        top_k_results   TEXT    NOT NULL DEFAULT '[]',
        notes           TEXT    NOT NULL DEFAULT ''
    );
    """

    _CREATE_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);
    CREATE INDEX IF NOT EXISTS idx_history_class ON history(predicted_class);
    CREATE INDEX IF NOT EXISTS idx_history_confidence ON history(confidence);
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: 数据库文件路径，默认 data/history.db
        """
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "history.db")

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(self._CREATE_TABLE_SQL + self._CREATE_INDEX_SQL)
                conn.commit()
            finally:
                conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> HistoryRecord:
        """将数据库行转换为 HistoryRecord"""
        return HistoryRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            filename=row["filename"],
            image_path=row["image_path"],
            thumbnail_path=row["thumbnail_path"],
            predicted_class=row["predicted_class"],
            confidence=row["confidence"],
            top_k_results=row["top_k_results"],
            notes=row["notes"],
        )

    # ==================== 增删改 ====================

    def save(self, record: HistoryRecord) -> int:
        """
        保存单条记录

        Returns:
            插入的记录 ID
        """
        with self._lock:
            conn = self._get_conn()
            try:
                if not record.timestamp:
                    record.timestamp = datetime.now().isoformat()

                cursor = conn.execute(
                    """INSERT INTO history
                       (timestamp, filename, image_path, thumbnail_path,
                        predicted_class, confidence, top_k_results, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.timestamp, record.filename, record.image_path,
                        record.thumbnail_path, record.predicted_class,
                        record.confidence, record.top_k_results, record.notes,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def update(self, record: HistoryRecord) -> bool:
        """更新已有记录"""
        if record.id is None:
            return False
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE history SET
                       timestamp=?, filename=?, image_path=?, thumbnail_path=?,
                       predicted_class=?, confidence=?, top_k_results=?, notes=?
                       WHERE id=?""",
                    (
                        record.timestamp, record.filename, record.image_path,
                        record.thumbnail_path, record.predicted_class,
                        record.confidence, record.top_k_results, record.notes,
                        record.id,
                    ),
                )
                conn.commit()
                return conn.total_changes > 0
            finally:
                conn.close()

    def delete(self, record_id: int) -> bool:
        """删除单条记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                # 先获取缩略图路径用于删除文件
                row = conn.execute(
                    "SELECT thumbnail_path FROM history WHERE id=?", (record_id,)
                ).fetchone()
                if row and row["thumbnail_path"]:
                    self._delete_thumbnail_file(row["thumbnail_path"])

                conn.execute("DELETE FROM history WHERE id=?", (record_id,))
                conn.commit()
                return conn.total_changes > 0
            finally:
                conn.close()

    def delete_batch(self, ids: List[int]) -> int:
        """
        批量删除记录

        Returns:
            删除的记录数
        """
        if not ids:
            return 0
        with self._lock:
            conn = self._get_conn()
            try:
                # 获取缩略图路径
                placeholders = ",".join("?" * len(ids))
                rows = conn.execute(
                    f"SELECT thumbnail_path FROM history WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                for row in rows:
                    if row["thumbnail_path"]:
                        self._delete_thumbnail_file(row["thumbnail_path"])

                conn.execute(
                    f"DELETE FROM history WHERE id IN ({placeholders})", ids
                )
                conn.commit()
                return conn.total_changes
            finally:
                conn.close()

    # ==================== 查询 ====================

    def query(
        self,
        filters: Optional[Dict] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "timestamp DESC",
    ) -> Tuple[List[HistoryRecord], int]:
        """
        条件查询

        Args:
            filters: 筛选条件字典，支持:
                - date_start: str  起始日期 (YYYY-MM-DD)
                - date_end: str    结束日期 (YYYY-MM-DD)
                - predicted_class: str  类别名（精确匹配）
                - confidence_min: float 最低置信度
                - confidence_max: float 最高置信度
                - filename: str    文件名模糊搜索
            limit:  每页条数
            offset: 偏移量
            order_by: 排序字段

        Returns:
            (记录列表, 总匹配数)
        """
        filters = filters or {}
        where_clauses = []
        params = []

        # 日期范围
        if filters.get("date_start"):
            where_clauses.append("DATE(timestamp) >= ?")
            params.append(filters["date_start"])
        if filters.get("date_end"):
            where_clauses.append("DATE(timestamp) <= ?")
            params.append(filters["date_end"])

        # 类别
        if filters.get("predicted_class"):
            where_clauses.append("predicted_class = ?")
            params.append(filters["predicted_class"])

        # 置信度区间
        if filters.get("confidence_min") is not None:
            where_clauses.append("confidence >= ?")
            params.append(float(filters["confidence_min"]))
        if filters.get("confidence_max") is not None:
            where_clauses.append("confidence <= ?")
            params.append(float(filters["confidence_max"]))

        # 文件名模糊搜索
        if filters.get("filename"):
            where_clauses.append("filename LIKE ?")
            params.append(f"%{filters['filename']}%")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        conn = self._get_conn()
        try:
            # 总数
            count_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM history {where_sql}", params
            ).fetchone()
            total = count_row["cnt"]

            # 数据
            rows = conn.execute(
                f"SELECT * FROM history {where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

            records = [self._row_to_record(row) for row in rows]
            return records, total
        finally:
            conn.close()

    def get_by_id(self, record_id: int) -> Optional[HistoryRecord]:
        """按 ID 获取单条记录"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM history WHERE id=?", (record_id,)).fetchone()
            return self._row_to_record(row) if row else None
        finally:
            conn.close()

    # ==================== 统计 ====================

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            {
                total: int,
                today_count: int,
                class_distribution: List[Dict],  # Top-10 类别分布
                avg_confidence: float,
                date_range: Tuple[str, str],
            }
        """
        conn = self._get_conn()
        try:
            # 总数
            total = conn.execute("SELECT COUNT(*) as cnt FROM history").fetchone()["cnt"]

            if total == 0:
                return {
                    "total": 0,
                    "today_count": 0,
                    "class_distribution": [],
                    "avg_confidence": 0.0,
                    "date_range": ("", ""),
                }

            # 今日识别数
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM history WHERE DATE(timestamp)=?",
                (today,),
            ).fetchone()["cnt"]

            # 类别分布 Top-10
            rows = conn.execute(
                """SELECT predicted_class, COUNT(*) as cnt
                   FROM history
                   GROUP BY predicted_class
                   ORDER BY cnt DESC
                   LIMIT 10"""
            ).fetchall()
            class_distribution = [
                {"class": row["predicted_class"], "count": row["cnt"]}
                for row in rows
            ]

            # 平均置信度
            avg_conf = conn.execute(
                "SELECT AVG(confidence) as avg_c FROM history"
            ).fetchone()["avg_c"] or 0.0

            # 日期范围
            date_row = conn.execute(
                "SELECT MIN(DATE(timestamp)) as min_d, MAX(DATE(timestamp)) as max_d FROM history"
            ).fetchone()
            date_range = (date_row["min_d"] or "", date_row["max_d"] or "")

            return {
                "total": total,
                "today_count": today_count,
                "class_distribution": class_distribution,
                "avg_confidence": round(avg_conf, 4),
                "date_range": date_range,
            }
        finally:
            conn.close()

    def get_all_classes(self) -> List[str]:
        """获取所有已识别的类别列表"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT predicted_class FROM history ORDER BY predicted_class"
            ).fetchall()
            return [row["predicted_class"] for row in rows]
        finally:
            conn.close()

    # ==================== 导出 ====================

    def export_csv(self, path: Optional[str] = None, filters: Optional[Dict] = None) -> str:
        """
        导出为 CSV

        Args:
            path: 输出路径，默认 data/export_history_YYYYMMDD.csv
            filters: 可选筛选条件

        Returns:
            导出文件路径
        """
        records, _ = self.query(filters=filters, limit=100000, offset=0)
        if not path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(
                data_dir, f"export_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "时间", "文件名", "原始路径", "缩略图路径",
                "预测类别", "置信度", "Top-K 结果", "备注",
            ])
            for r in records:
                writer.writerow([
                    r.id, r.timestamp, r.filename, r.image_path,
                    r.thumbnail_path, r.predicted_class,
                    f"{r.confidence:.4f}", r.top_k_results, r.notes,
                ])

        return path

    def export_json(self, path: Optional[str] = None, filters: Optional[Dict] = None) -> str:
        """
        导出为 JSON

        Args:
            path: 输出路径，默认 data/export_history_YYYYMMDD.json
            filters: 可选筛选条件

        Returns:
            导出文件路径
        """
        records, _ = self.query(filters=filters, limit=100000, offset=0)
        if not path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(
                data_dir, f"export_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        data = [asdict(r) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path

    # ==================== 清理 ====================

    def cleanup(self, retention_days: int = 90) -> int:
        """
        清理过期记录

        Args:
            retention_days: 保留天数，默认 90 天

        Returns:
            删除的记录数
        """
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                # 获取要删除记录的缩略图路径
                rows = conn.execute(
                    "SELECT thumbnail_path FROM history WHERE timestamp < ?",
                    (cutoff,),
                ).fetchall()
                for row in rows:
                    if row["thumbnail_path"]:
                        self._delete_thumbnail_file(row["thumbnail_path"])

                conn.execute("DELETE FROM history WHERE timestamp < ?", (cutoff,))
                conn.commit()
                return conn.total_changes
            finally:
                conn.close()

    # ==================== 缩略图管理 ====================

    @staticmethod
    def get_thumbnail_dir() -> str:
        """获取缩略图根目录"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        thumb_root = os.path.join(project_root, "data", "thumbnails")
        os.makedirs(thumb_root, exist_ok=True)
        return thumb_root

    def save_thumbnail(self, image: np.ndarray, filename: str) -> str:
        """
        保存缩略图到 data/thumbnails/YYYY-MM/ 目录

        Args:
            image: RGB numpy 数组
            filename: 原始文件名

        Returns:
            缩略图保存路径
        """
        now = datetime.now()
        month_dir = os.path.join(
            self.get_thumbnail_dir(), now.strftime("%Y-%m")
        )
        os.makedirs(month_dir, exist_ok=True)

        # 生成唯一文件名
        base, ext = os.path.splitext(filename)
        if not ext or ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            ext = ".jpg"
        thumb_name = f"{base}_{now.strftime('%H%M%S')}{ext}"
        thumb_path = os.path.join(month_dir, thumb_name)

        # 缩放并保存
        h, w = image.shape[:2]
        max_size = 200
        scale = min(max_size / h, max_size / w) if max(h, w) > max_size else 1.0
        new_w, new_h = int(w * scale), int(h * scale)

        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        pixmap.save(thumb_path)

        return thumb_path

    @staticmethod
    def _delete_thumbnail_file(thumb_path: str):
        """删除缩略图文件"""
        try:
            if os.path.isfile(thumb_path):
                os.remove(thumb_path)
        except OSError:
            pass


# ==================== 全局单例 ====================

_history_db: Optional[HistoryDB] = None


def get_history_db(db_path: Optional[str] = None) -> HistoryDB:
    """获取全局 HistoryDB 单例"""
    global _history_db
    if _history_db is None:
        _history_db = HistoryDB(db_path)
    return _history_db
