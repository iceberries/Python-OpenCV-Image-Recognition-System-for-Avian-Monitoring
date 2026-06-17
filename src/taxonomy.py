"""
分类学树模块 (CUB-Hierarchy)

解析 cvjena/semantic-embeddings 的 cub.parent-child.txt 构建 3 级分类学树:
  Order(目) → Family(科) → Species(种)
"""

import os
from typing import Dict, List, Optional, Tuple, Set


# ============================================================
#  数据模型
# ============================================================

class TaxonNode:
    """分类学树节点"""

    __slots__ = ('id', 'name', 'common_name', 'level', 'parent',
                 'children', 'species_id', '_depth')

    def __init__(self, node_id: int, name: str, level: str = "species",
                 common_name: str = ""):
        self.id = node_id
        self.name = name
        self.common_name = common_name
        self.level = level
        self.parent: Optional['TaxonNode'] = None
        self.children: List['TaxonNode'] = []
        self.species_id: Optional[int] = None
        self._depth: int = -1

    @property
    def depth(self) -> int:
        if self._depth < 0:
            self._depth = 0 if self.parent is None else self.parent.depth + 1
        return self._depth

    def path_to_root(self) -> List['TaxonNode']:
        """从当前节点到根节点的路径（含自身）"""
        path = [self]
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def is_ancestor_of(self, other: 'TaxonNode') -> bool:
        """判断当前节点是否为 other 的祖先"""
        node = other.parent
        while node is not None:
            if node.id == self.id:
                return True
            node = node.parent
        return False

    def __repr__(self):
        return f"TaxonNode(id={self.id}, name='{self.name}', level='{self.level}')"


# ============================================================
#  分类学树
# ============================================================

class TaxonomyTree:
    """CUB-Hierarchy 分类学树管理器"""

    LEVEL_ORDER = ['root', 'order', 'family', 'species']

    def __init__(self, hierarchy_dir: str = ""):
        self.hierarchy_dir = hierarchy_dir
        self.nodes: Dict[int, TaxonNode] = {}
        self.root: Optional[TaxonNode] = None
        self.level_ids: Dict[str, List[int]] = {}
        self.level_names: Dict[str, List[str]] = {}
        self.level_to_idx: Dict[str, Dict[int, int]] = {}
        self.leaf_paths: Dict[int, List[TaxonNode]] = {}

    # ================================================================
    #  解析
    # ================================================================

    def parse(self, hierarchy_dir: str = None):
        """解析 CUB-Hierarchy (3 级: Order→Family→Species)"""
        if hierarchy_dir is None:
            hierarchy_dir = self.hierarchy_dir
        self.hierarchy_dir = hierarchy_dir

        # 自动检测文件名
        candidates = ["cub_flat.parent-child.txt", "cub_balanced.parent-child.txt",
                       "cub.parent-child.txt"]
        parent_child_path = None
        for c in candidates:
            p = os.path.join(hierarchy_dir, c)
            if os.path.exists(p):
                parent_child_path = p
                break

        classes_candidates = ["classes_flat.txt", "classes_balanced.txt", "classes.txt"]
        classes_path = None
        for c in classes_candidates:
            p = os.path.join(hierarchy_dir, c)
            if os.path.exists(p):
                classes_path = p
                break

        if parent_child_path is None:
            raise FileNotFoundError(
                f"CUB-Hierarchy 文件未找到: {hierarchy_dir}\n"
                f"请从 https://github.com/cvjena/semantic-embeddings 下载 CUB-Hierarchy"
            )

        # --- 1. classes.txt ---
        id_to_name: Dict[int, str] = {}
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1) if ' ' in line else line.split('\t', 1)
                    if len(parts) >= 2:
                        id_to_name[int(parts[0])] = parts[1].strip().replace(' ', '_')
                    else:
                        id_to_name[i] = line.replace(' ', '_')

        # --- 2. parent-child.txt ---
        edges: List[Tuple[int, int]] = []
        with open(parent_child_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))

        # --- 3. 建节点 ---
        all_ids: Set[int] = set()
        for p, c in edges:
            all_ids.add(p)
            all_ids.add(c)
        for nid in all_ids:
            self.nodes[nid] = TaxonNode(nid, id_to_name.get(nid, f"node_{nid}"))

        # --- 4. 建边 ---
        for p, c in edges:
            parent, child = self.nodes.get(p), self.nodes.get(c)
            if parent and child:
                parent.children.append(child)
                child.parent = parent

        # --- 5. 使用文件自带的 Aves 根节点（保持旧 checkpoint 兼容） ---
        self.root = self._find_root()
        if self.root is None:
            virtual_root_id = max(all_ids) + 1 if all_ids else 1
            self.root = TaxonNode(virtual_root_id, "Aves")
            self.root.level = "root"
            self.nodes[virtual_root_id] = self.root
            for node in list(self.nodes.values()):
                if node.parent is None and node.id != virtual_root_id:
                    self.root.children.append(node)
                    node.parent = self.root

        # --- 6. 推断层级 ---
        self._infer_cub_levels()
        self.LEVEL_ORDER = ['root', 'order', 'family', 'species']
        self._build_index_maps()
        self._load_translations()

        for node in self.nodes.values():
            if not node.children:
                self.leaf_paths[node.id] = node.path_to_root()

        return self

    def _load_translations(self):
        """从 translations_zh.json 加载中文翻译。
        由于旧分类学偏移（family 实际是 order，species 包含 family+species），
        翻译时做层级映射。"""
        from src.i18n import I18n
        i18n = I18n.load(self.hierarchy_dir)
        self.level_names_zh: Dict[str, List[str]] = {}
        # 翻译映射：model_level → json_level
        trans_map = {'order': 'order', 'family': 'order', 'species': 'species'}
        for lvl in self.LEVEL_ORDER:
            if lvl == 'root':
                continue
            src = trans_map.get(lvl, lvl)
            names = self.level_names.get(lvl, [])
            self.level_names_zh[lvl] = []
            for name in names:
                zh = i18n.t(src, name)
                if zh == name:
                    # 回退：species 级也查 family 翻译
                    zh = i18n.t('family', name)
                self.level_names_zh[lvl].append(zh)

    def _build_index_maps(self):
        """为每个层级构建 0-indexed 类别ID映射"""
        self.level_ids = {lvl: [] for lvl in self.LEVEL_ORDER if lvl != 'root'}
        self.level_names = {lvl: [] for lvl in self.LEVEL_ORDER if lvl != 'root'}
        self.level_to_idx = {lvl: {} for lvl in self.LEVEL_ORDER if lvl != 'root'}
        self.level_depths: Dict[str, List[int]] = {}
        for node in sorted(self.nodes.values(), key=lambda n: n.id):
            if node.level == 'root':
                continue
            lvl = node.level
            self.level_ids[lvl].append(node.id)
            self.level_names[lvl].append(node.name)
            self.level_to_idx[lvl][node.id] = len(self.level_ids[lvl]) - 1
        # 记录 species 层各节点的树深度
        if 'species' in self.level_ids:
            self.level_depths['species'] = [
                self.nodes[nid]._depth for nid in self.level_ids['species']
            ]

    def _infer_cub_levels(self):
        """root→order→family→species"""
        queue = [(self.root, 0)]
        depth_to_level = {0: 'root', 1: 'order', 2: 'family', 3: 'species'}
        while queue:
            node, depth = queue.pop(0)
            node.level = depth_to_level.get(depth, 'species')
            node._depth = depth
            for child in node.children:
                queue.append((child, depth + 1))

    def _find_root(self) -> Optional[TaxonNode]:
        """找到没有父节点的根"""
        for node in self.nodes.values():
            if node.parent is None:
                return node
        # 若所有节点都有父节点，选 ID 最小的
        if self.nodes:
            return self.nodes[min(self.nodes.keys())]
        return None

    def _find_root(self) -> Optional[TaxonNode]:
        """为每个层级构建 0-indexed 类别ID映射"""
        self.level_ids = {lvl: [] for lvl in self.LEVEL_ORDER if lvl != 'root'}
        self.level_names = {lvl: [] for lvl in self.LEVEL_ORDER if lvl != 'root'}
        self.level_to_idx = {lvl: {} for lvl in self.LEVEL_ORDER if lvl != 'root'}

        for node in sorted(self.nodes.values(), key=lambda n: n.id):
            if node.level == 'root':
                continue
            lvl = node.level
            self.level_ids[lvl].append(node.id)
            self.level_names[lvl].append(node.name)
            self.level_to_idx[lvl][node.id] = len(self.level_ids[lvl]) - 1

    # ================================================================
    #  查询接口
    # ================================================================

    def num_levels(self) -> Dict[str, int]:
        """返回各层级的类别数"""
        return {lvl: len(ids) for lvl, ids in self.level_ids.items()}

    def get_node(self, node_id: int) -> Optional[TaxonNode]:
        """通过 ID 获取节点"""
        return self.nodes.get(node_id)

    def get_node_by_name(self, name: str, level: str = None) -> Optional[TaxonNode]:
        """通过名称查找节点"""
        for node in self.nodes.values():
            if node.name == name:
                if level is None or node.level == level:
                    return node
        return None

    def get_level_idx(self, level: str, node_id: int) -> int:
        """
        获取节点在指定层级中的 0-indexed 类别 ID

        Args:
            level: 'order'|'family'|'genus'|'species'|'visual'
            node_id: NABirds 内部节点 ID

        Returns:
            0-indexed 整数标签，未找到返回 -1
        """
        return self.level_to_idx.get(level, {}).get(node_id, -1)

    def get_taxonomic_path(self, leaf_id: int) -> List[TaxonNode]:
        """
        获取叶子节点的完整分类学路径

        Returns:
            [root, order, family, genus, species, visual]
        """
        return self.leaf_paths.get(leaf_id, [])

    def get_all_labels_for_image(
        self, visual_class_id: int
    ) -> Dict[str, int]:
        """
        获取图像的多级标签（0-indexed）

        Args:
            visual_class_id: NABirds 的 class_id（通常是 visual 层级）

        Returns:
            {'order': 0, 'family': 3, 'genus': 12, 'species': 45, 'visual': 67}
        """
        path = self.get_taxonomic_path(visual_class_id)
        result = {}
        for node in path:
            if node.level != 'root':
                idx = self.get_level_idx(node.level, node.id)
                result[node.level] = idx
        return result

    def get_species_label(self, visual_class_id: int) -> int:
        """获取视觉类别对应的物种标签"""
        path = self.get_taxonomic_path(visual_class_id)
        for node in path:
            if node.level == 'species':
                return self.get_level_idx('species', node.id)
        return -1

    def summary(self) -> str:
        """打印分类学树摘要"""
        lines = ["=" * 60,
                 "  NABirds 分类学树摘要",
                 "=" * 60]
        nums = self.num_levels()
        for lvl in ['order', 'family', 'genus', 'species', 'visual']:
            lines.append(f"  {lvl:>10s}: {nums.get(lvl, 0):>5d} 类")
        lines.append("-" * 60)
        lines.append(f"  总节点数: {len(self.nodes)}")
        lines.append(f"  叶子节点(视觉类别): {nums.get('visual', 0)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_sample_paths(self, n: int = 3):
        """打印几条示例分类学路径"""
        print("\n示例分类学路径:")
        count = 0
        for leaf_id, path in self.leaf_paths.items():
            if count >= n:
                break
            path_str = " → ".join(
                f"[{node.level}] {node.name}" for node in path if node.level != 'root'
            )
            print(f"  {path_str}")
            count += 1


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        # 默认路径
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "NABirds"
        )

    print(f"解析 NABirds 分类学树: {root}")
    try:
        tree = TaxonomyTree(root)
        tree.parse()
        print(tree.summary())
        tree.print_sample_paths(5)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 NABirds 数据集已放置在正确位置。")
