#!/usr/bin/env python3
"""
简单的多模态测试脚本，用于调试
"""

import base64
import json
import sys

import requests


def create_test_image():
    """创建一个简单的测试图像（1x1像素的红色图片）"""
    from PIL import Image
    import io
    
    # 创建一个1x1的红色图片
    img = Image.new('RGB', (100, 100), color='red')
    
    # 转换为bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # 编码为base64
    return base64.b64encode(img_byte_arr).decode('utf-8')


def test_multimodal():
    """测试多模态请求"""
    print("=" * 60)
    print("测试多模态请求（带调试日志）")
    print("=" * 60)
    
    # 创建测试图像
    print("\n创建测试图像...")
    image_base64 = create_test_image()
    print(f"图像编码完成，大小: {len(image_base64)} 字符")
    
    # 构造请求
    url = "http://127.0.0.1:30019/v1/chat/completions"
    payload = {
        "model": "Qwen2.5-VL-32B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请描述这张图片。"
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
    }
    
    print(f"\n请求URL: {url}")
    print("请求内容: [包含图像数据]")
    print("\n发送请求...")
    print("请查看服务器日志中的 [MM_DEBUG] 标记")
    print("=" * 60)
    
    try:
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
                
                # 分析token数量
                prompt_tokens = usage.get('prompt_tokens', 0)
                if prompt_tokens < 100:
                    print(f"\n⚠️ 警告: Prompt tokens ({prompt_tokens}) 太少！")
                    print("   这表明图像可能没有被处理成图像tokens。")
                    print("   正常情况下，一张图像应该产生数百个tokens。")
                else:
                    print(f"\n✅ Prompt tokens ({prompt_tokens}) 看起来正常，图像可能被正确处理了。")
        else:
            print(f"\n⚠️ 响应格式异常: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return False
        
        return True
        
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


if __name__ == "__main__":
    success = test_multimodal()
    sys.exit(0 if success else 1)

