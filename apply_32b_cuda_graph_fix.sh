#!/bin/bash
# 应用CUDA Graph性能修复补丁

SGLANG_DIR="/home/yzh/semi_pd_migration/sglang_0.4.8"
TARGET_FILE="$SGLANG_DIR/python/sglang/srt/model_executor/cuda_graph_runner.py"

echo "🔧 应用32B模型CUDA Graph性能修复..."

# 1. 备份原文件
cp $TARGET_FILE ${TARGET_FILE}.backup_32b
echo "✅ 备份完成"

# 2. 应用主要修复
echo "📝 修复stream管理问题..."

# 修复replay_prepare函数中的stream管理
python3 << 'EOF'
import re

file_path = "/home/yzh/semi_pd_migration/sglang_0.4.8/python/sglang/srt/model_executor/cuda_graph_runner.py"

with open(file_path, 'r') as f:
    content = f.read()

# 修复1：恢复stream包装
# 查找replay_prepare函数，添加stream context
pattern1 = r'(def replay_prepare.*?
.*?# Common inputs)'
replacement1 = r'
        # Use captured stream to avoid cross-stream waits
        with torch.cuda.stream(self.stream):'

# 修复2：移除过多的日志
# 移除[CG-STREAM]和[CG-LAUNCH]日志
content = re.sub(r'.*\[CG-STREAM\].*
', '', content)
content = re.sub(r'.*\[CG-LAUNCH\].*
', '', content)
content = re.sub(r'.*_t0 = _t\.perf_counter\(\).*
', '', content)
content = re.sub(r'.*_dt = \(_t\.perf_counter.*
', '', content)

# 修复3：优化replay函数
replay_fix = """
    def replay(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        if self.update_mode == CudaGraphRunnerMode.UPDATE_TORCH_INPUT:
            self.input_ids[:self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[:self.raw_num_token].copy_(forward_batch.positions)
        
        # Ensure replay on captured stream
        with torch.cuda.stream(self.stream):
            self.graphs[self.bs].replay()
        
        output = self.output_buffers[self.bs]
"""

# 查找并替换replay函数
replay_pattern = r'def replay\(self,.*?
.*?output = self\.output_buffers\[self\.bs\]'
content = re.sub(replay_pattern, replay_fix.strip(), content, flags=re.DOTALL)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Stream管理修复完成")
EOF

# 3. 应用内存优化
echo "📝 应用32B模型特定优化..."

python3 << 'EOF'
file_path = "/home/yzh/semi_pd_migration/sglang_0.4.8/python/sglang/srt/model_executor/cuda_graph_runner.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# 在类初始化中添加模型大小检测
for i, line in enumerate(lines):
    if "__init__" in line and "CudaGraphRunner" in lines[max(0, i-5):i]:
        # 添加模型大小属性
        insert_line = i + 10  # 在__init__函数内部添加
        lines.insert(insert_line, "        # Detect model size for optimization
")
        lines.insert(insert_line + 1, "        self.model_size_gb = self._estimate_model_size()
")
        break

# 添加模型大小估算函数
model_size_func = """
    def _estimate_model_size(self):
        """Estimate model size in GB"""
        try:
            total_params = sum(p.numel() for p in self.model_runner.model.parameters())
            size_gb = (total_params * 2) / (1024 ** 3)  # FP16
            return size_gb
        except:
            return 0
"""

# 找到合适的位置插入
for i, line in enumerate(lines):
    if "def capture(" in line:
        lines.insert(i, model_size_func + "
")
        break

with open(file_path, 'w') as f:
    f.writelines(lines)

print("✅ 32B模型优化完成")
EOF

echo "
✅ 修复应用完成！

下一步：
1. 重启Semi-PD服务
2. 测试32B模型性能
"
