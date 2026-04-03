"""
全局配置文件
"""
import os

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集根目录 (注意双层嵌套结构: CUB_200_2011/CUB_200_2011/)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "CUB_200_2011", "CUB_200_2011")

# 图像目录
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")

# 输出目录 (保存模型权重、日志等)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据集配置 ====================
NUM_CLASSES = 200
NUM_IMAGES = 11788

# 数据集标注文件
IMAGES_FILE = os.path.join(DATASET_ROOT, "images.txt")
CLASSES_FILE = os.path.join(DATASET_ROOT, "classes.txt")
IMAGE_CLASS_LABELS_FILE = os.path.join(DATASET_ROOT, "image_class_labels.txt")
TRAIN_TEST_SPLIT_FILE = os.path.join(DATASET_ROOT, "train_test_split.txt")
BOUNDING_BOXES_FILE = os.path.join(DATASET_ROOT, "bounding_boxes.txt")

# ==================== 模型配置 ====================
# 输入图像尺寸 (ResNet50 标准输入)
INPUT_SIZE = 224

# 批次大小
BATCH_SIZE = 32

# 训练轮次
NUM_EPOCHS = 30

# 初始学习率
LEARNING_RATE = 0.001

# 学习率衰减因子
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# 权重衰减
WEIGHT_DECAY = 1e-4

# 预训练权重
PRETRAINED = True

# ==================== 数据增强配置 ====================
# 训练集随机裁剪尺寸
TRAIN_CROP_SIZE = 224
# 测试集中心裁剪尺寸
TEST_CROP_SIZE = 224
# 随机水平翻转概率
RANDOM_FLIP_PROB = 0.5
# 颜色抖动参数
COLOR_JITTER = (0.2, 0.2, 0.2, 0.1)

# ==================== 设备配置 ====================
DEVICE = "cuda"  # 如果没有 GPU 会自动回退到 CPU
NUM_WORKERS = 4  # DataLoader 工作线程数

# ==================== 随机种子 ====================
RANDOM_SEED = 42
