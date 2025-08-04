#!/bin/bash

echo "🚀 Semi-PD 张量同步优化 - 快速验证"
echo "=" * 50

# 设置环境
export PYTHONPATH="/home/yzh/semi_pd_migration/sglang_0.4.8/python:$PYTHONPATH"
export SEMI_PD_ASYNC_OPT_ENABLED=1
export SEMI_PD_CACHE_SIZE=2000
export SEMI_PD_CACHE_TTL=0.1

echo "📊 运行性能测试..."
python test_tensor_sync_optimization.py

echo ""
echo "✅ 测试完成！如果看到 60-90% 的性能提升，说明优化有效。"
echo ""
echo "🎯 接下来可以："
echo "  1. 运行实际的 Semi-PD 服务: ./start_semi_pd_async_optimized.sh"
echo "  2. 查看完整文档: cat SOLUTION_SUMMARY.md"
echo "  3. 进行负载测试: python verify_async_optimization.py" 