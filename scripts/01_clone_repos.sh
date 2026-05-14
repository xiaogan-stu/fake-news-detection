#!/bin/bash
set -e

PROJECT_DIR="/mnt/workspace/fake-news-detection"
echo "=== 克隆 MHR 仓库（提供双曲流形核心代码）==="

# 克隆 MHR
if [ -d "/tmp/MHR" ]; then
    echo "MHR 已存在，跳过克隆"
else
    git clone https://github.com/guoxinyu0617/MHR /tmp/MHR
    echo "✅ MHR 克隆完成"
fi

# 查看 MHR 实际目录结构
echo ""
echo "=== MHR 仓库结构 ==="
find /tmp/MHR -type f -name "*.py" | head -30

# 复制 manifolds 文件
echo ""
echo "=== 复制流形代码 ==="
MANIFOLD_DST="$PROJECT_DIR/src/models/manifolds"

# 尝试常见路径
for src_path in \
    "/tmp/MHR/manifolds" \
    "/tmp/MHR/models/manifolds" \
    "/tmp/MHR/hgnn/manifolds" \
    "/tmp/MHR/src/manifolds"
do
    if [ -d "$src_path" ]; then
        echo "找到 manifolds 目录: $src_path"
        cp -r $src_path/. $MANIFOLD_DST/
        echo "✅ 流形文件已复制到 $MANIFOLD_DST"
        break
    fi
done

# 同样处理 layers
LAYERS_DST="$PROJECT_DIR/src/models/layers"
for src_path in \
    "/tmp/MHR/layers" \
    "/tmp/MHR/models/layers" \
    "/tmp/MHR/hgnn/layers" \
    "/tmp/MHR/src/layers"
do
    if [ -d "$src_path" ]; then
        echo "找到 layers 目录: $src_path"
        cp -r $src_path/. $LAYERS_DST/
        echo "✅ 图层文件已复制到 $LAYERS_DST"
        break
    fi
done

echo ""
echo "=== 最终文件列表 ==="
echo "manifolds:"
ls $MANIFOLD_DST/
echo "layers:"
ls $LAYERS_DST/