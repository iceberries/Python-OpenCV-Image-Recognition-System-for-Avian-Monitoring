#!/bin/bash
# ============================================================
#  自动尝试多个 Docker 镜像源，构建成功即停止
#  用法: bash build.sh [镜像名:标签]
# ============================================================

set -e

IMAGE="${1:-bird:v1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 镜像源列表 (按优先级排序，腾讯云内网优先)
REGISTRIES=(
    "mirror.ccs.tencentyun.com"
    "docker.1ms.run"
    "docker.xuanyuan.me"
    "docker.m.daocloud.io"
)

echo "============================================================"
echo "   自动构建 Docker 镜像: ${IMAGE}"
echo "============================================================"

for REGISTRY in "${REGISTRIES[@]}"; do
    echo ""
    echo -e "\033[0;36m>>> 尝试镜像源: ${REGISTRY}\033[0m"
    
    if docker build --build-arg "REGISTRY=${REGISTRY}" -t "$IMAGE" "$SCRIPT_DIR"; then
        echo ""
        echo -e "\033[0;32m[OK] 构建成功! 使用镜像源: ${REGISTRY}\033[0m"
        echo -e "\033[0;32m     镜像: ${IMAGE}\033[0m"
        echo ""
        echo "  运行:  docker run -it --rm -v \$(pwd)/CUB_200_2011:/app/CUB_200_2011 -v \$(pwd)/output:/app/output ${IMAGE}"
        echo "  导出:  docker save ${IMAGE} | gzip > bird-v1.tar.gz"
        exit 0
    else
        echo -e "\033[0;33m[WARN] ${REGISTRY} 失败，尝试下一个...\033[0m"
    fi
done

echo ""
echo -e "\033[0;31m[ERROR] 所有镜像源均失败，请检查网络或手动指定:\033[0m"
echo "  docker build --build-arg REGISTRY=你的镜像源 -t ${IMAGE} ."
exit 1
