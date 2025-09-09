#!/usr/bin/env python3
"""
SGLang TBO 验证脚本
用于验证 Two-Batch Overlap 是否真正在 dense 模型上运行
"""

import asyncio
import aiohttp
import json
import time
import uuid
import re
import subprocess
import sys
from pathlib import Path

class TBOVerifier:
    def __init__(self, server_url="http://127.0.0.1:30000", log_file="/tmp/sglang_tbo_verify.log"):
        self.server_url = server_url
        self.log_file = log_file
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_unique_prompt(self, length=1000):
        """生成唯一的长prompt确保prefill模式"""
        base_prompt = "请详细解释深度学习中的注意力机制原理，包括其数学公式、实现细节和应用场景。" * (length // 50)
        unique_id = str(uuid.uuid4())
        return f"[ID:{unique_id}] {base_prompt} 请针对这个唯一ID {unique_id} 给出详细回答。"
    
    async def send_request(self, prompt_length=1000, max_tokens=50):
        """发送请求到SGLang服务器"""
        prompt = self.generate_unique_prompt(prompt_length)
        
        payload = {
            "model": "/home/yzh/model/Qwen/Qwen2.5-32B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False
        }
        
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    return {
                        "success": True,
                        "elapsed_time": elapsed,
                        "prompt_tokens": len(prompt.split()),
                        "response": result.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
                    }
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_logs(self):
        """分析日志文件，检查TBO执行情况"""
        if not Path(self.log_file).exists():
            return {"error": "日志文件不存在"}
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
        
        # 检查关键TBO指标
        indicators = {
            "服务器启动": "Application startup complete" in log_content,
            "TBO启用配置": "enable_two_batch_overlap=True" in log_content,
            "TBO门控检查": "tbo_gate_global:" in log_content,
            "TBO准备阶段": "TboForwardBatchPreparer.prepare" in log_content,
            "TBO执行路径": "forward use TBO:" in log_content,
            "微批次分割": "split_inputs:" in log_content,
            "重叠操作执行": "execute_overlapped_operations" in log_content,
            "TBO操作调用": any(op in log_content for op in [
                "op_input_norm_and_qkv", "op_attention_compute", 
                "op_attention_output_proj_and_allreduce", "op_post_attn_norm_and_gate_up",
                "op_mlp_down_proj_and_allreduce", "op_residual_add_final"
            ]),
            "小批次门控": "disable_tbo_small_token:" in log_content,
            "标准逐层前向": "using standard layer-by-layer forward" in log_content,
            "CUDA流管理": "stage" in log_content and ("ops=" in log_content),
            "RMSNorm融合核": "fused_add_rmsnorm" in log_content or "rmsnorm" in log_content,
        }
        
        # 提取关键数值
        can_run_tbo_matches = re.findall(r'can_run_tbo=(\w+)', log_content)
        forward_mode_matches = re.findall(r'global_forward_mode=(\d+)', log_content)
        token_length_matches = re.findall(r'token_len=(\d+)', log_content)
        
        return {
            "indicators": indicators,
            "can_run_tbo_values": can_run_tbo_matches,
            "forward_modes": forward_mode_matches,
            "token_lengths": token_length_matches,
            "tbo_activated": "can_run_tbo=True" in log_content and "forward use TBO: can_run_tbo=True" in log_content,
            "fallback_reasons": self._extract_fallback_reasons(log_content)
        }
    
    def _extract_fallback_reasons(self, log_content):
        """提取TBO回退原因"""
        reasons = []
        if "disable_tbo_small_token:" in log_content:
            matches = re.findall(r'disable_tbo_small_token: token_len=(\d+) < min_tokens=(\d+)', log_content)
            reasons.extend([f"小批次门控: token_len={m[0]} < min_tokens={m[1]}" for m in matches])
        
        if "using standard layer-by-layer forward" in log_content:
            reasons.append("使用标准逐层前向传播")
            
        if "force disable TBO" in log_content:
            reasons.append("强制禁用TBO")
            
        return reasons
    
    def print_verification_report(self, test_results, log_analysis):
        """打印验证报告"""
        print("\n" + "="*60)
        print("SGLang TBO 验证报告")
        print("="*60)
        
        # 测试结果
        print(f"\n📊 请求测试结果:")
        for i, result in enumerate(test_results, 1):
            if result["success"]:
                print(f"  测试 {i}: ✅ 成功 ({result['elapsed_time']:.2f}s, {result['prompt_tokens']} tokens)")
                print(f"    响应: {result['response']}")
            else:
                print(f"  测试 {i}: ❌ 失败 - {result['error']}")
        
        # 日志分析
        print(f"\n📋 TBO 执行指标:")
        if "error" in log_analysis:
            print(f"  ❌ {log_analysis['error']}")
            return
            
        indicators = log_analysis["indicators"]
        for key, value in indicators.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}")
        
        print(f"\n🎯 关键状态:")
        print(f"  TBO激活状态: {'✅ 已激活' if log_analysis['tbo_activated'] else '❌ 未激活'}")
        
        if log_analysis["can_run_tbo_values"]:
            print(f"  can_run_tbo历史: {', '.join(log_analysis['can_run_tbo_values'])}")
        
        if log_analysis["forward_modes"]:
            mode_names = {1: "EXTEND(prefill)", 2: "DECODE", 3: "TARGET_VERIFY"}
            modes = [f"{mode}({mode_names.get(int(mode), 'UNKNOWN')})" for mode in log_analysis["forward_modes"]]
            print(f"  前向模式: {', '.join(modes)}")
        
        if log_analysis["token_lengths"]:
            print(f"  Token长度: {', '.join(log_analysis['token_lengths'])}")
        
        if log_analysis["fallback_reasons"]:
            print(f"\n⚠️  TBO回退原因:")
            for reason in log_analysis["fallback_reasons"]:
                print(f"    - {reason}")
        
        # TBO真实性判断
        print(f"\n🔍 TBO真实性判断:")
        tbo_really_running = (
            log_analysis["tbo_activated"] and
            indicators["TBO操作调用"] and
            not indicators["标准逐层前向"] and
            indicators["CUDA流管理"]
        )
        
        if tbo_really_running:
            print("  🎉 TBO 确实在运行！")
            print("    - TBO已激活且调用了TBO专用操作")
            print("    - 没有回退到标准逐层前向传播")
            print("    - 检测到CUDA流管理")
        else:
            print("  ⚠️  TBO 可能没有真正运行")
            if not log_analysis["tbo_activated"]:
                print("    - TBO未激活")
            if not indicators["TBO操作调用"]:
                print("    - 未检测到TBO专用操作调用")
            if indicators["标准逐层前向"]:
                print("    - 回退到了标准逐层前向传播")
            if not indicators["CUDA流管理"]:
                print("    - 未检测到CUDA流管理")
    
    async def run_verification(self):
        """运行完整的TBO验证流程"""
        print("🚀 开始 SGLang TBO 验证...")
        
        # 等待服务器启动
        print("⏳ 等待服务器启动...")
        await asyncio.sleep(10)
        
        # 执行多个测试
        test_results = []
        test_configs = [
            {"length": 500, "max_tokens": 20, "name": "短prompt测试"},
            {"length": 1500, "max_tokens": 50, "name": "长prompt测试"},
            {"length": 2000, "max_tokens": 30, "name": "超长prompt测试"},
        ]
        
        for config in test_configs:
            print(f"📤 执行 {config['name']}...")
            result = await self.send_request(config["length"], config["max_tokens"])
            result["test_name"] = config["name"]
            test_results.append(result)
            await asyncio.sleep(2)  # 避免请求过快
        
        # 分析日志
        print("📊 分析执行日志...")
        await asyncio.sleep(2)  # 等待日志写入
        log_analysis = self.analyze_logs()
        
        # 生成报告
        self.print_verification_report(test_results, log_analysis)
        
        return test_results, log_analysis

async def main():
    """主函数"""
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = "http://127.0.0.1:30000"
    
    async with TBOVerifier(server_url) as verifier:
        await verifier.run_verification()

if __name__ == "__main__":
    asyncio.run(main()) 