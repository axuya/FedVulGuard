#!/bin/bash
set -e

echo "========================================"
echo "Phase 2 环境配置（修正版）"
echo "========================================"

# 1. 基础依赖
pip install solc-select py-solc-x slither-analyzer networkx scikit-learn matplotlib seaborn tqdm pandas

# 2. 查看 torch 版本
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# 3. 正确下载 solc 0.8.0 官方静态二进制
mkdir -p ~/.solc/0.8.0 && cd ~/.solc/0.8.0
if [ ! -f solc ]; then
    echo "🚀 下载 solc 0.8.0 ..."
    wget -q https://github.com/ethereum/solidity/releases/download/v0.8.0/solc-static-linux -O solc
    chmod +x solc
fi

# 4. 验证
~/.solc/0.8.0/solc --version

# 5. 设置环境变量
export SOLC_BINARY=~/.solc/0.8.0/solc
echo "export SOLC_BINARY=$SOLC_BINARY" >> ~/.bashrc

# 6. 可选：pygraphviz（若之前失败）
pip install --no-cache-dir pygraphviz 2>/dev/null || echo "⚠️ pygraphviz 需系统 graphviz-dev，跳过"

echo "========================================"
echo "✅ 完成！solc 路径：$SOLC_BINARY"
echo "========================================"