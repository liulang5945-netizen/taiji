"""
态极记忆 token 训练数据生成器
================================

生成教态极使用 <mem_read>/<mem_write> 特殊 token 的训练数据。

目标：让模型学会：
1. 读取记忆槽中的信息来回答问题
2. 将重要信息写入记忆槽
3. 在多轮对话中维护上下文

生成方式：
    python -m taiji.data.generate_memory_data
"""
import os
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger("MemoryDataGenerator")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "taiji_data" / "training_data"


def generate_read_samples(count: int = 200) -> list:
    """生成记忆读取训练样本"""
    samples = []

    templates = [
        {
            "memory": "用户的名字是张三，他是一名Python开发者",
            "question": "我叫什么名字？",
            "answer": "你叫张三。",
            "mem_slot": 0,
        },
        {
            "memory": "项目使用 FastAPI 框架，数据库是 PostgreSQL",
            "question": "我们的项目用的什么框架？",
            "answer": "项目使用 FastAPI 框架。",
            "mem_slot": 1,
        },
        {
            "memory": "用户的偏好：喜欢简洁的回答，不要冗长的解释",
            "question": "请介绍一下 Python",
            "answer": "Python 是一种简洁、易读的高级编程语言。",
            "mem_slot": 2,
        },
        {
            "memory": "上次用户问了关于排序算法的问题，我推荐了快速排序",
            "question": "上次我们讨论了什么？",
            "answer": "上次你问了排序算法的问题，我推荐了快速排序。",
            "mem_slot": 0,
        },
        {
            "memory": "项目目录结构：src/ 包含源代码，tests/ 包含测试，docs/ 包含文档",
            "question": "项目的目录结构是怎样的？",
            "answer": "项目有三个主要目录：src/ 存放源代码，tests/ 存放测试，docs/ 存放文档。",
            "mem_slot": 3,
        },
        {
            "memory": "用户正在开发一个聊天机器人，使用了 WebSocket 实现实时通信",
            "question": "我在做什么项目？",
            "answer": "你在开发一个聊天机器人，使用了 WebSocket 实现实时通信。",
            "mem_slot": 0,
        },
        {
            "memory": "错误日志：TypeError: cannot read property 'map' of undefined，出现在 line 42",
            "question": "最近有什么错误？",
            "answer": "最近有一个 TypeError：cannot read property 'map' of undefined，出现在第 42 行。",
            "mem_slot": 1,
        },
        {
            "memory": "用户的学习目标：掌握机器学习基础，能独立完成数据分析项目",
            "question": "我的学习目标是什么？",
            "answer": "你的学习目标是掌握机器学习基础，能独立完成数据分析项目。",
            "mem_slot": 2,
        },
    ]

    for _ in range(count):
        t = random.choice(templates)
        # 添加一些变化
        variations = [
            t,
            {
                "memory": t["memory"],
                "question": f"根据记忆，{t['question']}",
                "answer": t["answer"],
                "mem_slot": t["mem_slot"],
            },
            {
                "memory": t["memory"],
                "question": f"你还记得吗？{t['question']}",
                "answer": f"记得。{t['answer']}",
                "mem_slot": t["mem_slot"],
            },
        ]
        sample = random.choice(variations)
        samples.append(sample)

    return samples


def generate_write_samples(count: int = 200) -> list:
    """生成记忆写入训练样本"""
    samples = []

    templates = [
        {
            "conversation": "用户：我叫李四\n助手：你好李四！",
            "write_content": "用户的名字是李四",
            "slot": 0,
        },
        {
            "conversation": "用户：我们的项目用的是 React\n助手：好的，React 是一个流行的前端框架。",
            "write_content": "项目使用 React 框架",
            "slot": 1,
        },
        {
            "conversation": "用户：请帮我记住，我喜欢用 VS Code\n助手：好的，已记住你喜欢用 VS Code。",
            "write_content": "用户喜欢用 VS Code 编辑器",
            "slot": 2,
        },
        {
            "conversation": "用户：刚才的代码有个 bug，在第 15 行\n助手：找到了，是变量名拼写错误。",
            "write_content": "代码 bug 在第 15 行，变量名拼写错误",
            "slot": 3,
        },
        {
            "conversation": "用户：我要学习 Go 语言\n助手：Go 是 Google 开发的编程语言，以并发和性能著称。",
            "write_content": "用户想学习 Go 语言",
            "slot": 0,
        },
        {
            "conversation": "用户：我的邮箱是 test@example.com\n助手：好的，已记住你的邮箱。",
            "write_content": "用户邮箱：test@example.com",
            "slot": 1,
        },
    ]

    for _ in range(count):
        t = random.choice(templates)
        samples.append(t)

    return samples


def generate_multi_turn_samples(count: int = 100) -> list:
    """生成多轮对话记忆维护样本"""
    samples = []

    scenarios = [
        {
            "turns": [
                {"role": "user", "content": "我叫王五"},
                {"role": "assistant", "content": "你好王五！"},
                {"role": "user", "content": "我在做数据分析项目"},
                {"role": "assistant", "content": "数据分析是个很好的方向。"},
                {"role": "user", "content": "你还记得我叫什么吗？"},
            ],
            "answer": "你叫王五。",
            "memory_state": "用户名字=王五，项目=数据分析",
        },
        {
            "turns": [
                {"role": "user", "content": "帮我看看 main.py 文件"},
                {"role": "assistant", "content": "文件内容：def main(): print('hello')"},
                {"role": "user", "content": "把 print 改成 logger.info"},
                {"role": "assistant", "content": "已修改。"},
                {"role": "user", "content": "现在文件里有什么？"},
            ],
            "answer": "文件里有：def main(): logger.info('hello')",
            "memory_state": "main.py 内容已修改，print 改为 logger.info",
        },
        {
            "turns": [
                {"role": "user", "content": "搜索一下 Python 异步编程"},
                {"role": "assistant", "content": "Python 异步编程使用 asyncio 库..."},
                {"role": "user", "content": "给我一个例子"},
                {"role": "assistant", "content": "async def main(): await asyncio.sleep(1)"},
                {"role": "user", "content": "刚才搜索的什么关键词？"},
            ],
            "answer": "刚才搜索的关键词是 Python 异步编程。",
            "memory_state": "搜索历史=Python 异步编程",
        },
    ]

    for _ in range(count):
        s = random.choice(scenarios)
        samples.append(s)

    return samples


def format_as_training_data(read_samples, write_samples, multi_turn_samples) -> list:
    """格式化为态极训练数据格式"""
    training_data = []

    # 读取样本
    for s in read_samples:
        training_data.append({
            "type": "memory_read",
            "instruction": f"[记忆槽{s['mem_slot']}] {s['memory']}\n\n问题：{s['question']}",
            "output": s["answer"],
            "metadata": {
                "skill": "memory_read",
                "slot": s["mem_slot"],
            }
        })

    # 写入样本
    for s in write_samples:
        training_data.append({
            "type": "memory_write",
            "instruction": f"对话：{s['conversation']}\n\n请将重要信息写入记忆。",
            "output": f"<mem_write slot=\"{s['slot']}\">{s['write_content']}</mem_write>",
            "metadata": {
                "skill": "memory_write",
                "slot": s["slot"],
            }
        })

    # 多轮对话样本
    for s in multi_turn_samples:
        conversation = ""
        for turn in s["turns"]:
            role = "用户" if turn["role"] == "user" else "助手"
            conversation += f"{role}: {turn['content']}\n"

        training_data.append({
            "type": "memory_maintain",
            "instruction": f"对话历史：\n{conversation}\n\n记忆状态：{s['memory_state']}\n\n请回答最后一个问题：",
            "output": s["answer"],
            "metadata": {
                "skill": "memory_maintain",
            }
        })

    return training_data


def main():
    """生成记忆训练数据"""
    logger.info("Generating memory training data...")

    read_samples = generate_read_samples(200)
    write_samples = generate_write_samples(200)
    multi_turn_samples = generate_multi_turn_samples(100)

    training_data = format_as_training_data(read_samples, write_samples, multi_turn_samples)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "memory_training_data.jsonl"

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(training_data)} memory training samples -> {output_path}")

    # 统计
    types = {}
    for item in training_data:
        t = item["type"]
        types[t] = types.get(t, 0) + 1

    for t, count in types.items():
        logger.info(f"  {t}: {count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
