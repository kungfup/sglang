#!/usr/bin/env python3
"""
测试多模态功能的脚本

用法：
    python test_multimodal.py --text-only    # 测试纯文本
    python test_multimodal.py --image <path> # 测试图像+文本
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image(image_path: str) -> str:
    """将图像文件编码为base64字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_text_only(base_url: str = "http://127.0.0.1:30019"):
    """测试纯文本请求"""
    print("=" * 60)
    print("测试 1: 纯文本请求")
    print("=" * 60)
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "Qwen2.5-VL-32B-Instruct",
        "messages": [
            {"role": "user", "content": "你好！请用一句话介绍一下你自己。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    print(f"\n请求URL: {url}")
    print(f"请求内容: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("\n发送请求...")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ 请求成功！")
        print(f"响应状态码: {response.status_code}")
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"\n模型回复:\n{content}")
            
            # 检查usage信息
            if "usage" in result:
                usage = result["usage"]
                print(f"\nToken使用情况:")
                print(f"  - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  - Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"\n⚠️ 响应格式异常: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 处理响应时出错: {e}")
        return False


def test_multimodal(image_path: str, base_url: str = "http://127.0.0.1:30019"):
    """测试多模态（图像+文本）请求"""
    print("=" * 60)
    print("测试 2: 多模态请求（图像+文本）")
    print("=" * 60)
    
    # 检查图像文件是否存在
    if not Path(image_path).exists():
        print(f"\n❌ 图像文件不存在: {image_path}")
        return False
    
    print(f"\n图像文件: {image_path}")
    print("正在编码图像...")
    
    try:
        # 编码图像
        image_base64 = encode_image(image_path)
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        
        print(f"图像编码完成，大小: {len(image_base64)} 字符")
        
        # 构造请求
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": "Qwen2.5-VL-32B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        },
                        {
                            "type": "text",
                            "text": "请详细描述这张图片的内容。"
                        }
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        
        print(f"\n请求URL: {url}")
        print("请求内容: [包含图像数据，已省略]")
        print("\n发送请求...")
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ 请求成功！")
        print(f"响应状态码: {response.status_code}")
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"\n模型回复:\n{content}")
            
            # 检查usage信息
            if "usage" in result:
                usage = result["usage"]
                print(f"\nToken使用情况:")
                print(f"  - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  - Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"\n⚠️ 响应格式异常: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return False
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ 无法读取图像文件: {image_path}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ 处理时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试Semi-PD多模态功能")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="只测试纯文本请求"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="测试多模态请求时使用的图像文件路径"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:30019",
        help="服务器基础URL（默认: http://127.0.0.1:30019）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试，默认运行文本测试
    if not args.text_only and not args.image:
        args.text_only = True
    
    results = []
    
    # 测试纯文本
    if args.text_only:
        success = test_text_only(args.base_url)
        results.append(("纯文本测试", success))
    
    # 测试多模态
    if args.image:
        success = test_multimodal(args.image, args.base_url)
        results.append(("多模态测试", success))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 返回退出码
    all_passed = all(success for _, success in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

