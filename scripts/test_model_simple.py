"""
简单测试模型
============
快速验证模型是否正常工作
"""
import os
import sys
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def main():
    print("=" * 60)
    print("简单测试模型")
    print("=" * 60)

    # 加载模型
    model_path = "taiji_data/evolution_data/best"
    print(f"\n  加载模型: {model_path}")

    from taiji.loader import load_model
    model, tokenizer = load_model(model_path, device="cpu")
    model.eval()

    print(f"  模型加载成功!")

    # 测试 tokenizer
    print(f"\n  测试 Tokenizer:")
    test_text = "你好，我是态极"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"    输入: {test_text}")
    print(f"    编码: {encoded[:10]}...")
    print(f"    解码: {decoded}")

    # 测试前向传播
    print(f"\n  测试前向传播:")
    input_tensor = torch.tensor([encoded[:50]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_tensor)
        if hasattr(output, 'logits'):
            print(f"    输出形状: {output.logits.shape}")
            print(f"    前向传播成功!")
        else:
            print(f"    输出类型: {type(output)}")

    # 测试生成（简单）
    print(f"\n  测试生成:")
    try:
        if hasattr(model, 'generate'):
            output_ids = model.generate(input_tensor, max_new_tokens=20, temperature=0.7)
            response = tokenizer.decode(output_ids[0].tolist())
            # 处理编码问题
            try:
                print(f"    生成成功: {response[:100]}...")
            except UnicodeEncodeError:
                safe_response = response[:100].encode('ascii', 'replace').decode('ascii')
                print(f"    生成成功: {safe_response}...")
        else:
            print(f"    模型没有 generate 方法，跳过生成测试")
    except Exception as e:
        print(f"    生成测试失败: {e}")

    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
