"""
测试模型性能
============
测试新训练的 1B 模型的对话能力
"""
import os
import sys
import json
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from taiji.config import ModelConfig
from taiji.architecture import ModelSelf
from taiji.tokenizer import ModelSelfTokenizer
from taiji.loader import load_model

def test_model():
    print("=" * 60)
    print("测试模型性能")
    print("=" * 60)

    # 加载模型
    model_path = "taiji_data/evolution_data/best"
    print(f"\n  加载模型: {model_path}")

    model, tokenizer = load_model(model_path, device="cpu")
    model.eval()

    # 测试用例
    test_cases = [
        {
            "name": "基础对话",
            "input": "你好，你是谁？",
        },
        {
            "name": "知识问答",
            "input": "什么是机器学习？",
        },
        {
            "name": "代码生成",
            "input": "用Python写一个快速排序算法",
        },
        {
            "name": "逻辑推理",
            "input": "如果所有的猫都是动物，所有的动物都需要食物，那么猫需要食物吗？",
        },
        {
            "name": "创意写作",
            "input": "写一首关于春天的诗",
        },
    ]

    print("\n  测试结果:")
    print("-" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n  [{i}] {test['name']}")
        print(f"  输入: {test['input']}")

        # 编码输入
        input_ids = tokenizer.encode(test['input'])
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # 生成回答
        with torch.no_grad():
            try:
                # 尝试使用 generate 方法
                if hasattr(model, 'generate'):
                    output_ids = model.generate(input_tensor, max_new_tokens=200, temperature=0.7, top_p=0.9)
                    response = tokenizer.decode(output_ids[0].tolist())
                else:
                    # 前向传播
                    output = model(input_tensor)
                    if hasattr(output, 'logits'):
                        # 取最后一个 token 的 logits
                        logits = output.logits[:, -1, :]
                        # 采样
                        probs = torch.softmax(logits / 0.7, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        # 简单生成几个 token
                        generated = [input_ids[-1]]
                        for _ in range(50):
                            input_t = torch.tensor([generated[-100:]], dtype=torch.long)
                            out = model(input_t)
                            if hasattr(out, 'logits'):
                                logits = out.logits[:, -1, :]
                                probs = torch.softmax(logits / 0.7, dim=-1)
                                next_tok = torch.multinomial(probs, 1).item()
                                generated.append(next_tok)
                                if next_tok == tokenizer.eos_token_id:
                                    break
                        response = tokenizer.decode(generated)
                    else:
                        response = "[无法生成回答]"
            except Exception as e:
                response = f"[生成失败: {str(e)}]"

        # 处理编码问题
        try:
            print(f"  输出: {response[:200]}...")
        except UnicodeEncodeError:
            # 如果有编码问题，只显示 ASCII 字符
            safe_response = response[:200].encode('ascii', 'replace').decode('ascii')
            print(f"  输出: {safe_response}...")
        print()

    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
