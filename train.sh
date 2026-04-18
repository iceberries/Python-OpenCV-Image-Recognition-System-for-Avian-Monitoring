#!/bin/bash
# ============================================================
#  鸟类图像识别系统 - 一键训练脚本 (Docker / 宿主机通用)
#  用法:
#    Docker 内:  docker run --gpus all ... (自动执行)
#    宿主机:    bash train.sh [--cpu] [--epochs N] [--batch_size N]
# ============================================================

set -e

# ==================== 可选参数 ====================
EPOCHS=30
BATCH_SIZE=32
USE_CPU=false
PROJECT_DIR="/app"              # Docker 内固定路径
CUDA_VERSION="cu118"

for arg in "$@"; do
    case $arg in
        --cpu)        USE_CPU=true ;;
        --epochs=*)   EPOCHS="${arg#*=}" ;;
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --help|-h)
            echo "用法: bash train.sh [选项]"
            echo "  --cpu          强制使用 CPU (仅宿主机模式生效)"
            echo "  --epochs=N     训练轮次 (默认 30)"
            echo "  --batch_size=N 批次大小 (默认 32)"
            exit 0 ;;
    esac
done

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ==================== 检测运行环境 ====================
IN_DOCKER=false
[[ -f /.dockerenv ]] && IN_DOCKER=true

# ============================================================
#  Docker 模式: 环境已就绪，只需处理 GPU 切换
# ============================================================
setup_docker_env() {
    info "运行环境: Docker 容器"
    python --version

    # GPU 自动切换: CPU 镜像 → GPU 容器时覆盖安装 CUDA 版 PyTorch
    if ! $USE_CPU && command -v nvidia-smi &>/dev/null; then
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            success "GPU 版 PyTorch 已就绪"
        else
            warn "检测到 GPU，正在切换为 CUDA 版 PyTorch (~2分钟)..."
            pip install torch torchvision \
                --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" \
                --force-reinstall --no-deps -q
            success "CUDA 版 PyTorch 安装完成"
        fi
        python -c "
import torch
print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
    else
        info "使用 CPU 模式训练"
    fi
}

# ============================================================
#  宿主机模式: 安装 Conda + Python 环境
# ============================================================
install_conda() {
    if command -v conda &>/dev/null; then
        success "Conda 已安装: $(conda --version)"
        return 0
    fi
    info "安装 Miniconda..."
    local installer="$HOME/miniconda3_installer.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
    bash "$installer" -b -p "$HOME/miniconda3"
    rm -f "$installer"
    success "Miniconda 安装完成"
}

setup_conda_env() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    local env_name="bird"

    if conda env list | grep -q "^${env_name}\s"; then
        info "Conda 环境 '$env_name' 已存在"
    else
        info "创建 Conda 环境: $env_name ..."
        conda create -n "$env_name" python=3.10 -y
        success "环境创建完成"
    fi
    conda activate "$env_name"

    info "安装 Python 依赖..."
    pip install --upgrade pip -q
    pip install numpy opencv-python Pillow scikit-image matplotlib tqdm scipy networkx PyWavelets -q

    if $USE_CPU || ! command -v nvidia-smi &>/dev/null; then
        info "安装 PyTorch (CPU)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    else
        info "安装 PyTorch (CUDA ${CUDA_VERSION})..."
        pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" -q
    fi
    success "依赖安装完成"
}

# ============================================================
#  数据集检查 (已下载则跳过)
# ============================================================
check_dataset() {
    local data_path="$PROJECT_DIR/CUB_200_2011/CUB_200_2011"

    if [[ -d "$data_path/images" ]]; then
        local img_count=$(find "$data_path/images" -type f 2>/dev/null | wc -l)
        success "数据集已就绪 (图像数: $img_count)"
        return 0
    fi

    # 尝试从腾讯云 COS 下载
    if [[ -d "$PROJECT_DIR/CUB_200_2011" ]]; then
        warn "数据集目录存在但 images/ 缺失，尝试从腾讯云下载..."
        mkdir -p "$PROJECT_DIR/CUB_200_2011"
        local tmp_zip="/tmp/CUB_200_2011.tgz"
        if wget -q --show-progress -O "$tmp_zip" "http://u.yd.tencentyun.com/CUB_200_2011.tgz"; then
            tar -xzf "$tmp_zip" -C "$PROJECT_DIR/CUB_200_2011/"
            rm -f "$tmp_zip"
            success "数据集下载并解压完成"
            return 0
        fi
    fi

    error "数据集缺失! 请将 CUB_200_2011 放入 $PROJECT_DIR/CUB_200_2011/ 目录"
}

# ============================================================
#  训练
# ============================================================
run_training() {
    echo ""
    echo "============================================================"
    echo "   开始训练"
    echo "============================================================"
    printf "  %-10s %s\n" "轮次:" "$EPOCHS"
    printf "  %-10s %s\n" "批次:" "$BATCH_SIZE"
    printf "  %-10s %s\n" "设备:" "$(if $USE_CPU; then echo 'CPU'; else echo 'GPU (auto)'; fi)"
    echo "============================================================"
    echo ""

    cd "$PROJECT_DIR"

    # main.py 通过 config.DEVICE 自动选择设备, 无需传 --cpu
    python -m src.main \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE"

    echo ""
    if [[ -f "$PROJECT_DIR/output/best_model.pth" ]]; then
        success "训练完成! 最佳模型: output/best_model.pth"
        ls -lh "$PROJECT_DIR/output/"
    else
        error "训练结束但未找到模型文件"
    fi
}

# ============================================================
#  主流程
# ============================================================
main() {
    echo ""
    echo "============================================================"
    echo "   鸟类图像识别系统 - 一键训练"
    echo "============================================================"

    if $IN_DOCKER; then
        setup_docker_env
    else
        info "运行环境: 宿主机"
        PROJECT_DIR="$HOME/Python-OpenCV-Image-Recognition-System-for-Avian-Monitoring"
        if [[ ! -d "$PROJECT_DIR/src" ]]; then
            error "未找到项目代码: $PROJECT_DIR"
        fi
        success "项目目录: $PROJECT_DIR"
        install_conda
        setup_conda_env
    fi

    check_dataset
    run_training

    echo ""
    echo "============================================================"
    success "   全部完成!"
    echo "============================================================"
    echo "  模型权重: $PROJECT_DIR/output/best_model.pth"
    echo "============================================================"
}

main
