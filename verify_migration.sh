#!/bin/bash
# 验证迁移的文件和修改

echo "=========================================="
echo "SGLang Migration Verification"
echo "=========================================="
echo ""

# 检查文件是否存在
echo "1. Checking if modified files exist..."
files=(
    "python/sglang/srt/distributed/parallel_state.py"
    "python/sglang/srt/managers/mm_utils.py"
    "python/sglang/srt/managers/multimodal_processor.py"
    "python/sglang/srt/managers/schedule_batch.py"
    "python/sglang/srt/managers/scheduler.py"
    "python/sglang/srt/managers/vit_worker.py"
    "python/sglang/srt/model_executor/forward_batch_info.py"
    "python/sglang/srt/model_executor/model_runner.py"
    "python/sglang/srt/utils/common.py"
    "python/sglang/srt/server_args.py"
    "python/sglang/srt/models/qwen2.py"
    "python/sglang/srt/models/qwen2_5_vl.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file NOT FOUND"
        all_exist=false
    fi
done
echo ""

# 检查 Python 语法
echo "2. Checking Python syntax..."
syntax_ok=true
for file in "${files[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "  ✓ $file syntax OK"
    else
        echo "  ✗ $file syntax ERROR"
        python -m py_compile "$file"
        syntax_ok=false
    fi
done
echo ""

# 检查关键函数是否存在
echo "3. Checking for key functions/classes..."

echo "  Checking parallel_state.py..."
if grep -q "def get_pipeline_model_parallel_layer_split" python/sglang/srt/distributed/parallel_state.py; then
    echo "    ✓ get_pipeline_model_parallel_layer_split found"
else
    echo "    ✗ get_pipeline_model_parallel_layer_split NOT FOUND"
fi

if grep -q "def get_pp_indices" python/sglang/srt/distributed/parallel_state.py; then
    echo "    ✓ get_pp_indices found"
else
    echo "    ✗ get_pp_indices NOT FOUND"
fi

if grep -q "_PIPELINE_GLOBAL_CONFIG" python/sglang/srt/distributed/parallel_state.py; then
    echo "    ✓ _PIPELINE_GLOBAL_CONFIG found"
else
    echo "    ✗ _PIPELINE_GLOBAL_CONFIG NOT FOUND"
fi

echo "  Checking mm_utils.py..."
if grep -q "def tensor_hash" python/sglang/srt/managers/mm_utils.py; then
    echo "    ✓ tensor_hash found"
else
    echo "    ✗ tensor_hash NOT FOUND"
fi

if grep -q "prioritizes using the GPU" python/sglang/srt/managers/mm_utils.py; then
    echo "    ✓ GPU fallback logic found"
else
    echo "    ✗ GPU fallback logic NOT FOUND"
fi

echo "  Checking schedule_batch.py..."
if grep -q "2\*\*31 - 1" python/sglang/srt/managers/schedule_batch.py; then
    echo "    ✓ Updated pad_value calculation found"
else
    echo "    ✗ Updated pad_value calculation NOT FOUND"
fi

echo "  Checking scheduler.py..."
if grep -q "finally:" python/sglang/srt/managers/scheduler.py; then
    echo "    ✓ finally block found"
else
    echo "    ✗ finally block NOT FOUND"
fi

if grep -q "pg.barrier()" python/sglang/srt/managers/scheduler.py; then
    echo "    ✓ PP group cleanup logic found"
else
    echo "    ✗ PP group cleanup logic NOT FOUND"
fi

echo "  Checking vit_worker.py..."
if grep -q "class ViTWorkerManager" python/sglang/srt/managers/vit_worker.py; then
    echo "    ✓ ViTWorkerManager class found"
else
    echo "    ✗ ViTWorkerManager class NOT FOUND"
fi

if grep -q "class ViTWorkerThread" python/sglang/srt/managers/vit_worker.py; then
    echo "    ✓ ViTWorkerThread class found"
else
    echo "    ✗ ViTWorkerThread class NOT FOUND"
fi

echo "  Checking forward_batch_info.py..."
if grep -q "missing_len = extend_seq_len - actual_len" python/sglang/srt/model_executor/forward_batch_info.py; then
    echo "    ✓ mrope_positions padding logic found"
else
    echo "    ✗ mrope_positions padding logic NOT FOUND"
fi

echo "  Checking model_runner.py..."
if grep -q "pipeline_model_parallel_layer_split" python/sglang/srt/model_executor/model_runner.py; then
    echo "    ✓ layer_split parameter found"
else
    echo "    ✗ layer_split parameter NOT FOUND"
fi

if grep -q "Recreated PP group with NCCL backend" python/sglang/srt/model_executor/model_runner.py; then
    echo "    ✓ PP NCCL optimization found"
else
    echo "    ✗ PP NCCL optimization NOT FOUND"
fi

echo "  Checking server_args.py..."
if grep -q "pipeline_model_parallel_layer_split" python/sglang/srt/server_args.py; then
    echo "    ✓ layer_split argument found"
else
    echo "    ✗ layer_split argument NOT FOUND"
fi

echo "  Checking qwen2_5_vl.py..."
if grep -q "vit_async_enabled" python/sglang/srt/models/qwen2_5_vl.py; then
    echo "    ✓ ViT async optimization found"
else
    echo "    ✗ ViT async optimization NOT FOUND"
fi

if grep -q "_prepare_initial_embeddings" python/sglang/srt/models/qwen2_5_vl.py; then
    echo "    ✓ PP embedding preparation found"
else
    echo "    ✗ PP embedding preparation NOT FOUND"
fi

if grep -q "% vit_merger_window_size" python/sglang/srt/models/qwen2_5_vl.py; then
    echo "    ✓ Padding fix found"
else
    echo "    ✗ Padding fix NOT FOUND"
fi

echo ""

# 统计修改行数
echo "4. Counting modified lines..."
echo "  parallel_state.py: $(wc -l < python/sglang/srt/distributed/parallel_state.py) lines"
echo "  mm_utils.py: $(wc -l < python/sglang/srt/managers/mm_utils.py) lines"
echo "  schedule_batch.py: $(wc -l < python/sglang/srt/managers/schedule_batch.py) lines"
echo "  scheduler.py: $(wc -l < python/sglang/srt/managers/scheduler.py) lines"
echo "  vit_worker.py: $(wc -l < python/sglang/srt/managers/vit_worker.py) lines"
echo "  forward_batch_info.py: $(wc -l < python/sglang/srt/model_executor/forward_batch_info.py) lines"
echo "  model_runner.py: $(wc -l < python/sglang/srt/model_executor/model_runner.py) lines"
echo "  common.py: $(wc -l < python/sglang/srt/utils/common.py) lines"
echo "  server_args.py: $(wc -l < python/sglang/srt/server_args.py) lines"
echo "  qwen2_5_vl.py: $(wc -l < python/sglang/srt/models/qwen2_5_vl.py) lines"
echo ""

# 总结
echo "=========================================="
if [ "$all_exist" = true ] && [ "$syntax_ok" = true ]; then
    echo "✓ All checks passed!"
    echo "Migration appears to be successful."
else
    echo "✗ Some checks failed!"
    echo "Please review the errors above."
fi
echo "=========================================="

