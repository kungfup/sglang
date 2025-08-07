#!/bin/bash
# 临时禁用CUDA Graph的启动脚本
# 用于对比测试性能差异

echo "🚫 启动Semi-PD（禁用CUDA Graph）"
echo "用于对比测试与修复后的性能差异"

# 在你的启动命令中添加 --disable-cuda-graph 参数
# 例如：
# python -m sglang.launch_server \
#   --model-path your_model \
#   --enable-semi-pd \
#   --disable-cuda-graph \  # 添加这个参数
#   --other-args

echo "📊 对比指标："
echo "- 禁用CUDA Graph的QPS和延迟"
echo "- 启用CUDA Graph（修复后）的QPS和延迟"
echo "- cudaGraphLaunch的CPU时间占比"
