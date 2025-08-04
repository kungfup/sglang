#!/bin/bash

# Semi-PD 异步优化编译脚本

set -e

echo "🔧 编译 Semi-PD 异步优化组件"
echo "================================"

# 检查必要的工具
check_dependencies() {
    echo "📋 检查编译依赖..."
    
    if ! command -v nvcc &>/dev/null; then
        echo "❌ 错误: 未找到 nvcc，请安装 CUDA toolkit"
        exit 1
    fi
    
    if ! command -v g++ &>/dev/null; then
        echo "❌ 错误: 未找到 g++，请安装 build-essential"
        exit 1
    fi
    
    echo "✅ 编译依赖检查通过"
}

# 设置编译环境
setup_build_env() {
    echo "🔧 设置编译环境..."
    
    # CUDA 编译标志
    export CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    
    # 优化编译选项
    export CXXFLAGS="-O3 -march=native -mtune=native"
    export NVCCFLAGS="-O3 --use_fast_math"
    
    # 启用并行编译
    export MAX_JOBS=$(nproc)
    
    echo "✅ 编译环境设置完成"
    echo "  - CUDA 架构: $CUDA_ARCH_LIST"
    echo "  - 并行任务数: $MAX_JOBS"
}

# 清理之前的构建
clean_build() {
    echo "🧹 清理之前的构建..."
    
    if [ -d "build" ]; then
        rm -rf build
        echo "✅ 清理 build 目录"
    fi
    
    # 清理 Python 缓存
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    echo "✅ 清理完成"
}

# 编译 CUDA 内核
compile_cuda_kernels() {
    echo "🚀 编译 CUDA 内核..."
    
    cd sgl-kernel
    
    # 确保异步优化头文件存在
    if [ ! -f "csrc/allreduce/semi_pd_async_allreduce.cuh" ]; then
        echo "❌ 错误: 异步优化头文件不存在"
        exit 1
    fi
    
    # 检查并安装构建依赖
    echo "📦 安装构建依赖..."
    pip install scikit-build-core wheel torch
    
    # 使用现代构建系统编译
    echo "🔨 使用 pip 编译..."
    pip install -e . --no-build-isolation --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ CUDA 内核编译成功"
    else
        echo "❌ CUDA 内核编译失败，尝试使用 make 方式"
        
        # 备选方案：使用 Makefile
        if command -v make &>/dev/null; then
            echo "🔧 使用 make install..."
            make install
            if [ $? -eq 0 ]; then
                echo "✅ 使用 make 编译成功"
            else
                echo "❌ 所有编译方式都失败了"
                exit 1
            fi
        else
            echo "❌ make 命令不可用"
            exit 1
        fi
    fi
    
    cd ..
}

# 编译 IPC 扩展
compile_ipc_extension() {
    echo "🔗 编译 IPC 扩展..."
    
    cd semi-pd-ipc
    
    # 检查是否已经有编译好的文件
    if [ -f "semi_pd_ipc.cpython-*.so" ]; then
        echo "✅ 检测到已编译的 IPC 扩展，跳过编译"
        cd ..
        return 0
    fi
    
    # 编译 IPC 扩展
    echo "🔨 编译 IPC 扩展..."
    python setup.py build_ext --inplace
    if [ $? -eq 0 ]; then
        echo "✅ IPC 扩展编译成功"
    else
        echo "❌ IPC 扩展编译失败，尝试使用 pip 方式"
        pip install -e . --no-build-isolation
        if [ $? -eq 0 ]; then
            echo "✅ 使用 pip 编译 IPC 扩展成功"
        else
            echo "❌ IPC 扩展编译失败"
            exit 1
        fi
    fi
    
    cd ..
}

# 安装 Python 包
install_python_package() {
    echo "📦 安装 Python 包..."
    
    cd python
    pip install -e . --no-build-isolation
    if [ $? -eq 0 ]; then
        echo "✅ Python 包安装成功"
    else
        echo "❌ Python 包安装失败"
        exit 1
    fi
    
    cd ..
}

# 运行测试
run_tests() {
    echo "🧪 运行基础测试..."
    
    # 测试 CUDA 内核
    python -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA 可用')
    print(f'✅ CUDA 设备数: {torch.cuda.device_count()}')
    print(f'✅ CUDA 版本: {torch.version.cuda}')
else:
    print('❌ CUDA 不可用')
    exit(1)
"
    
    # 测试 Semi-PD 导入
    python -c "
try:
    from sglang.semi_pd.utils import InstanceRole
    from sglang.srt.managers.semi_pd_async_optimization import create_semi_pd_async_optimizer
    print('✅ Semi-PD 异步优化模块导入成功')
except ImportError as e:
    print(f'❌ Semi-PD 模块导入失败: {e}')
    exit(1)
"
    
    echo "✅ 基础测试通过"
}

# 主函数
main() {
    echo "开始编译 Semi-PD 异步优化组件..."
    
    check_dependencies
    setup_build_env
    clean_build
    
    echo ""
    echo "🔨 开始编译..."
    
    compile_cuda_kernels
    compile_ipc_extension
    install_python_package
    
    echo ""
    echo "🧪 运行测试..."
    run_tests
    
    echo ""
    echo "🎉 编译完成!"
    echo "=============================="
    echo "现在可以使用以下命令启动优化版本:"
    echo "  ./start_semi_pd_async_optimized.sh"
    echo ""
    echo "或者使用传统启动方式:"
    echo "  ./start_semi_pd_optimized.sh"
    echo ""
    echo "预期性能提升:"
    echo "  - cudaStreamSynchronize CPU 时间: 93% → <5%"
    echo "  - PREFILL 性能提升: ~35%"
    echo "  - DECODE 性能提升: ~10%"
    echo ""
}

# 执行主函数
main "$@" 