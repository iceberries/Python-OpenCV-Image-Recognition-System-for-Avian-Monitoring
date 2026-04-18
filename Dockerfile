# ============================================
#  CUB-200-2011 鸟类识别 - CPU 基础镜像
#  CPU 机构建 → GPU 服务器运行 (自动切换 CUDA)
#  用法: bash build.sh          (自动尝试多个镜像源)
#   或:  docker build --build-arg REGISTRY=xxx -t bird:v1 .
# ============================================

ARG REGISTRY=mirror.ccs.tencentyun.com
FROM ${REGISTRY}/python:3.10-slim

# ====== Layer 1: 系统依赖 ======
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-dev \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ======清华 pip 源 ======
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.timeout 120
RUN pip install --upgrade pip

# ====== 复制项目代码 ======
COPY requirements.txt .
COPY src/ ./src/
COPY train.sh .
RUN chmod +x train.sh
COPY verify_env.py .

# ====== Layer 2: Python 依赖 ======
# CPU 版 PyTorch 构建体积小，到 GPU 服务器后自动切换为 CUDA 版
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==4.9.0.80 \
    Pillow==10.2.0 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    tqdm==4.66.1 \
    scipy==1.12.0 \
    networkx==3.2.1 \
    PyWavelets==1.5.0

# ====== 创建必要目录 ======
RUN mkdir -p /app/output /app/CUB_200_2011

# ====== 数据卷挂载点 ======
VOLUME ["/app/CUB_200_2011", "/app/output"]

CMD ["bash", "train.sh"]
