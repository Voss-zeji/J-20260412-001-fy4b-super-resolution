#!/bin/bash
# 远程GPU服务器环境安装脚本

echo "======================================"
echo "设置FY-4B超分辨率项目环境"
echo "======================================"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: python3 未安装"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python版本: $PYTHON_VERSION"

# 安装pip（如果没有）
if ! command -v pip3 &> /dev/null; then
    echo "安装pip..."
    apt-get update -qq
    apt-get install -y -qq python3-pip
fi

# 升级pip
python3 -m pip install --upgrade pip -q

# 安装PyTorch (CUDA 13.0)
echo ""
echo "安装PyTorch (CUDA 13.0)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip3 install numpy h5py pandas matplotlib seaborn scikit-image tqdm -q

# 验证安装
echo ""
echo "验证PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "======================================"
echo "环境安装完成"
echo "======================================"
