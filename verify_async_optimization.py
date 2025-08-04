#!/usr/bin/env python3
"""
Semi-PD 异步优化性能验证脚本
用于验证移除 MSCCL++ 强制流同步后的性能提升
"""

import os
import sys
import time
import json
import argparse
import subprocess
import torch
import torch.profiler
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemiPDPerformanceVerifier:
    """Semi-PD 性能验证器"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 2):
        self.model_path = model_path
        self.tp_size = tensor_parallel_size
        self.verification_results = {}
        
    def verify_optimization_environment(self) -> Dict[str, bool]:
        """验证优化环境设置"""
        logger.info("🔍 验证优化环境设置...")
        
        checks = {}
        
        # 检查异步优化标志
        checks['async_opt_enabled'] = os.environ.get('SEMI_PD_ASYNC_OPT_ENABLED') == '1'
        
        # 检查 CUDA 设置
        checks['cuda_available'] = torch.cuda.is_available()
        checks['cuda_async_enabled'] = os.environ.get('CUDA_LAUNCH_BLOCKING') != '1'
        
        # 检查内存配置
        checks['memory_config_set'] = (
            'SEMI_PD_PREFILL_MEMORY_POOL_SIZE' in os.environ and
            'SEMI_PD_DECODE_MEMORY_POOL_SIZE' in os.environ
        )
        
        # 检查编译优化
        try:
            from sglang.srt.managers.semi_pd_async_optimization import get_semi_pd_async_optimizer
            checks['async_module_available'] = True
        except ImportError:
            checks['async_module_available'] = False
            
        # 输出检查结果
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}: {passed}")
            
        return checks
        
    def run_performance_profile(self, duration: int = 60, with_optimization: bool = True) -> str:
        """运行性能分析"""
        profile_name = f"semipd_{'optimized' if with_optimization else 'baseline'}_{int(time.time())}"
        
        logger.info(f"🚀 运行性能分析: {profile_name}")
        
        # 设置环境变量
        env = os.environ.copy()
        if with_optimization:
            env['SEMI_PD_ASYNC_OPT_ENABLED'] = '1'
            env['CUDA_LAUNCH_BLOCKING'] = '0'
        else:
            env['SEMI_PD_DISABLE_ASYNC_OPT'] = '1'
            env['CUDA_LAUNCH_BLOCKING'] = '0'
            
        # 构建启动命令
        cmd = [
            sys.executable, '-m', 'sglang.launch_server',
            '--model-path', self.model_path,
            '--host', '127.0.0.1',
            '--port', '8001',
            '--tp-size', str(self.tp_size),
            '--semi-pd',
            '--disable-cuda-graph-for-prefill',
            '--enable-cuda-graph',
            '--max-total-tokens', '8192',
            '--profile',
            '--profile-dir', f'./profile_{profile_name}',
        ]
        
        # 启动服务器
        logger.info(f"启动命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            # 等待服务器启动
            time.sleep(10)
            
            # 运行负载测试
            self._run_load_test(duration)
            
            # 等待分析完成
            time.sleep(5)
            
        finally:
            # 停止服务器
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                
        return f'./profile_{profile_name}'
        
    def _run_load_test(self, duration: int):
        """运行负载测试"""
        logger.info(f"📊 运行 {duration} 秒负载测试...")
        
        # 构建测试脚本
        test_script = f"""
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

def send_request():
    url = "http://127.0.0.1:8001/generate"
    data = {{
        "text": "请解释一下深度学习中的注意力机制原理，并说明其在Transformer架构中的作用。",
        "max_new_tokens": 128,
        "temperature": 0.7
    }}
    
    try:
        response = requests.post(url, json=data, timeout=30)
        return response.status_code == 200
    except:
        return False

# 运行负载测试
start_time = time.time()
success_count = 0
total_count = 0

with ThreadPoolExecutor(max_workers=4) as executor:
    while time.time() - start_time < {duration}:
        future = executor.submit(send_request)
        total_count += 1
        if future.result():
            success_count += 1
        time.sleep(0.5)

print(f"负载测试完成: {{success_count}}/{{total_count}} 成功")
"""
        
        # 运行测试脚本
        with open('/tmp/load_test.py', 'w') as f:
            f.write(test_script)
            
        subprocess.run([sys.executable, '/tmp/load_test.py'], 
                      capture_output=True, text=True)
                      
    def analyze_profile_results(self, profile_dir: str) -> Dict[str, float]:
        """分析性能分析结果"""
        logger.info(f"📈 分析性能分析结果: {profile_dir}")
        
        results = {}
        
        # 查找性能分析文件
        profile_files = []
        if os.path.exists(profile_dir):
            for file in os.listdir(profile_dir):
                if file.endswith('.txt') and 'stats' in file:
                    profile_files.append(os.path.join(profile_dir, file))
                    
        if not profile_files:
            logger.warning(f"未找到性能分析文件在 {profile_dir}")
            return results
            
        # 分析每个文件
        for profile_file in profile_files:
            process_type = 'UNKNOWN'
            if 'PREFILL' in profile_file:
                process_type = 'PREFILL'
            elif 'DECODE' in profile_file:
                process_type = 'DECODE'
                
            logger.info(f"分析 {process_type} 性能文件: {profile_file}")
            
            # 解析性能数据
            sync_time_percent = 0.0
            total_cuda_time = 0.0
            
            try:
                with open(profile_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    if 'cudaStreamSynchronize' in line:
                        # 提取同步时间百分比
                        parts = line.split()
                        if len(parts) >= 3:
                            sync_time_percent = float(parts[1].rstrip('%'))
                            
                    elif 'Self CUDA time total:' in line:
                        # 提取总 CUDA 时间
                        parts = line.split(':')
                        if len(parts) >= 2:
                            time_str = parts[1].strip()
                            # 解析时间格式 (例如: "22.501s")
                            if time_str.endswith('s'):
                                total_cuda_time = float(time_str[:-1])
                                
            except Exception as e:
                logger.error(f"解析性能文件失败: {e}")
                
            results[f'{process_type}_sync_time_percent'] = sync_time_percent
            results[f'{process_type}_total_cuda_time'] = total_cuda_time
            
        return results
        
    def compare_performance(self, baseline_results: Dict, optimized_results: Dict) -> Dict[str, float]:
        """比较性能结果"""
        logger.info("📊 比较性能结果...")
        
        comparison = {}
        
        for key in baseline_results:
            if key in optimized_results:
                baseline_val = baseline_results[key]
                optimized_val = optimized_results[key]
                
                if baseline_val > 0:
                    if 'sync_time_percent' in key:
                        # 对于同步时间百分比，降低是好的
                        improvement = (baseline_val - optimized_val) / baseline_val * 100
                        comparison[f'{key}_improvement_percent'] = improvement
                    elif 'total_cuda_time' in key:
                        # 对于总时间，降低是好的
                        improvement = (baseline_val - optimized_val) / baseline_val * 100
                        comparison[f'{key}_improvement_percent'] = improvement
                        
        return comparison
        
    def generate_report(self, baseline_results: Dict, optimized_results: Dict, comparison: Dict) -> str:
        """生成性能报告"""
        report = []
        report.append("=" * 60)
        report.append("Semi-PD 异步优化性能验证报告")
        report.append("=" * 60)
        report.append("")
        
        # 基线性能
        report.append("📊 基线性能 (未优化):")
        for key, value in baseline_results.items():
            if 'sync_time_percent' in key:
                report.append(f"  - {key}: {value:.2f}%")
            elif 'total_cuda_time' in key:
                report.append(f"  - {key}: {value:.3f}s")
        report.append("")
        
        # 优化后性能
        report.append("🚀 优化后性能:")
        for key, value in optimized_results.items():
            if 'sync_time_percent' in key:
                report.append(f"  - {key}: {value:.2f}%")
            elif 'total_cuda_time' in key:
                report.append(f"  - {key}: {value:.3f}s")
        report.append("")
        
        # 性能提升
        report.append("📈 性能提升:")
        for key, improvement in comparison.items():
            status = "✅" if improvement > 0 else "❌" if improvement < -5 else "⚠️"
            report.append(f"  {status} {key}: {improvement:+.2f}%")
        report.append("")
        
        # 总结
        report.append("📝 总结:")
        prefill_sync_improvement = comparison.get('PREFILL_sync_time_percent_improvement_percent', 0)
        decode_sync_improvement = comparison.get('DECODE_sync_time_percent_improvement_percent', 0)
        
        if prefill_sync_improvement > 80:
            report.append("  ✅ PREFILL 进程同步开销显著降低")
        elif prefill_sync_improvement > 50:
            report.append("  ⚠️  PREFILL 进程同步开销有所改善")
        else:
            report.append("  ❌ PREFILL 进程同步开销改善不明显")
            
        if decode_sync_improvement > 50:
            report.append("  ✅ DECODE 进程同步开销显著降低")
        elif decode_sync_improvement > 20:
            report.append("  ⚠️  DECODE 进程同步开销有所改善")
        else:
            report.append("  ❌ DECODE 进程同步开销改善不明显")
            
        report.append("")
        report.append("🎯 建议:")
        if prefill_sync_improvement < 80:
            report.append("  - 检查 SEMI_PD_ASYNC_OPT_ENABLED 环境变量是否设置")
            report.append("  - 确认 mscclpp_allreduce.cuh 已正确修改")
            report.append("  - 验证异步内存句柄管理器是否正常工作")
            
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Semi-PD 异步优化性能验证')
    parser.add_argument('--model-path', required=True, help='模型路径')
    parser.add_argument('--tp-size', type=int, default=2, help='张量并行大小')
    parser.add_argument('--duration', type=int, default=60, help='测试持续时间(秒)')
    parser.add_argument('--skip-baseline', action='store_true', help='跳过基线测试')
    
    args = parser.parse_args()
    
    verifier = SemiPDPerformanceVerifier(args.model_path, args.tp_size)
    
    # 验证环境
    env_checks = verifier.verify_optimization_environment()
    if not all(env_checks.values()):
        logger.warning("⚠️  环境检查未完全通过，可能影响验证结果")
        
    baseline_results = {}
    optimized_results = {}
    
    try:
        # 运行基线测试
        if not args.skip_baseline:
            logger.info("🔄 运行基线性能测试...")
            baseline_dir = verifier.run_performance_profile(args.duration, with_optimization=False)
            baseline_results = verifier.analyze_profile_results(baseline_dir)
        
        # 运行优化测试
        logger.info("🚀 运行优化性能测试...")
        optimized_dir = verifier.run_performance_profile(args.duration, with_optimization=True)
        optimized_results = verifier.analyze_profile_results(optimized_dir)
        
        # 比较结果
        if baseline_results and optimized_results:
            comparison = verifier.compare_performance(baseline_results, optimized_results)
            
            # 生成报告
            report = verifier.generate_report(baseline_results, optimized_results, comparison)
            print(report)
            
            # 保存报告
            with open('semi_pd_optimization_report.txt', 'w') as f:
                f.write(report)
            logger.info("📄 报告已保存到 semi_pd_optimization_report.txt")
            
        else:
            logger.error("❌ 无法获取完整的性能数据")
            
    except KeyboardInterrupt:
        logger.info("用户中断验证过程")
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        
    logger.info("🏁 性能验证完成")


if __name__ == '__main__':
    main() 