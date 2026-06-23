#!/bin/bash
# TaijiCore Docker构建脚本

set -e

VERSION=${1:-"latest"}
IMAGE_NAME="taiji"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"

echo "🔨 Building Docker image: ${IMAGE_TAG}"
docker build -t ${IMAGE_TAG} .

echo "✅ Build complete!"
echo "📦 Image: ${IMAGE_TAG}"
echo ""
echo "启动容器栈:"
echo "  docker-compose up -d"
echo ""
echo "查看日志:"
echo "  docker-compose logs -f taiji"
echo ""
echo "停止容器:"
echo "  docker-compose down"