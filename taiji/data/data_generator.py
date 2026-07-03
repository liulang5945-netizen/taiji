"""
ModelSelf 训练数据自动生成器

通过模板扩展和教师模型蒸馏两种方式生成大规模 ReAct 训练数据。
解决种子数据量不足的根本问题。
"""
import json
import random
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger("DataGenerator")


# ===== 任务模板库（覆盖所有工具场景） =====

TASK_TEMPLATES = [
    # === 文件读取 ===
    {"template": "读取 {path} 文件的内容", "tools": ["read_local_file"]},
    {"template": "帮我看看 {path} 文件里写了什么", "tools": ["read_local_file"]},
    {"template": "打开 {path} 并告诉我它的内容", "tools": ["read_local_file"]},

    # === 文件创建 ===
    {"template": "在 {path} 创建文件，内容是 {content}", "tools": ["write_file"]},
    {"template": "创建一个名为 {path} 的文件，写入以下内容：{content}", "tools": ["write_file"]},
    {"template": "帮我新建一个文件 {path}，内容是 {content}", "tools": ["write_file"]},

    # === 文件修改 ===
    {"template": "将 {path} 中的 {old_text} 改为 {new_text}", "tools": ["read_local_file", "edit_file"]},
    {"template": "修改 {path} 文件，把 {old_text} 替换成 {new_text}", "tools": ["read_local_file", "edit_file"]},

    # === 目录列表 ===
    {"template": "列出 {dir} 目录下的所有文件", "tools": ["list_directory"]},
    {"template": "看看 {dir} 文件夹里有什么", "tools": ["list_directory"]},

    # === Python 执行 ===
    {"template": "用 Python 计算 {expr}", "tools": ["execute_python"]},
    {"template": "写一段 Python 代码来计算 {expr} 并运行", "tools": ["execute_python"]},
    {"template": "帮我用 Python {desc}", "tools": ["execute_python"]},

    # === 搜索 ===
    {"template": "搜索一下 {query}", "tools": ["search"]},
    {"template": "帮我查一下 {query} 的相关信息", "tools": ["search"]},

    # === 网页读取 ===
    {"template": "读取 {url} 的内容", "tools": ["read_webpage"]},
    {"template": "帮我打开 {url} 并总结主要内容", "tools": ["read_webpage"]},

    # === 知识学习 ===
    {"template": "学习一下 {topic} 的知识", "tools": ["search", "learn_knowledge"]},
    {"template": "帮我搜索并保存 {topic} 的相关知识", "tools": ["search", "learn_knowledge"]},

    # === 知识查询 ===
    {"template": "查一下之前学过的关于 {query} 的知识", "tools": ["query_knowledge"]},
    {"template": "我记不起来了，关于 {query} 我学过什么？", "tools": ["query_knowledge"]},

    # === 多步骤文件操作 ===
    {"template": "先读取 {path}，然后把 {old_text} 改成 {new_text}", "tools": ["read_local_file", "edit_file"]},
    {"template": "创建 {path} 文件，写入 {content}，然后读取验证", "tools": ["write_file", "read_local_file"]},
    {"template": "在 {path} 中搜索 {keyword} 并替换为 {new_word}", "tools": ["read_local_file", "edit_file"]},

    # === 多步任务 ===
    {"template": "先搜索 {query}，然后把搜索结果保存到 {path}", "tools": ["search", "write_file"]},
    {"template": "搜索 {query} 并把要点写入 {path}", "tools": ["search", "write_file"]},
    {"template": "读取 {path} 文件，然后修改其中的 {old_text} 为 {new_text}", "tools": ["read_local_file", "edit_file"]},

    # === 无工具回答 ===
    {"template": "什么是 {concept}？请直接回答", "tools": []},
    {"template": "解释一下 {concept}", "tools": []},
    {"template": "为什么 {concept} 很重要？", "tools": []},
]

# ===== 参数池 =====

FILE_PATHS = [
    "README.md", "config.json", "src/main.py", "test.txt",
    "docs/index.html", "app_settings.json", "requirements.txt",
    "taiji/architecture.py", "agent/agent.py", "api/routes_chat.py",
    "version.json", "data/config.txt", "notes.md", "circle.py",
]

FILE_CONTENTS = [
    "Hello World", "# 项目文档\n\n这是一个测试项目。",
    "version=1.0.0\nauthor=Taiji", "print('Hello World')",
    "# 配置文件\n\nmode=development\ndebug=true",
    "def main():\n    pass", "这是一段测试文本，用于验证文件操作。",
]

OLD_TEXTS = [
    "Hello World", "version=1.0.0", "mode=development",
    "print('Hello World')", "测试文本",
]
NEW_TEXTS = [
    "Hello Python", "version=2.0.0", "mode=production",
    "print('Hello Python')", "修改后的文本",
]

DIRECTORIES = [".", "src", "docs", "taiji", "agent", "api"]

EXPRESSIONS = [
    "1 到 100 的和", "斐波那契数列前 20 项", "判断 17 是否是素数",
    "计算 [3,1,4,1,5,9,2,6] 的平均值", "生成一个随机密码",
    "排序 [5,2,8,1,9,3]", "计算 2 的 10 次方",
    "把 [1,2,3,4,5] 每个元素平方",
]

PYTHON_DESCRIPTIONS = [
    "写一个函数判断素数", "写一个排序算法",
    "计算圆的面积和周长", "统计当前目录下 .py 文件数量",
    "生成 8 位随机密码",
]

QUERIES = [
    "Python 异步编程", "机器学习入门", "Docker 教程",
    "PyTorch 安装方法", "Git 常用命令", "什么是 Transformer",
    "Python 装饰器", "REST API 设计原则",
    "PyTorch 最新版本", "LoRA 微调",
]

URLS = [
    "https://docs.python.org/3/",
    "https://pytorch.org/get-started/locally/",
    "https://github.com/",
]

TOPICS = [
    "Python 装饰器", "机器学习基础", "Docker 容器化",
    "Git 工作流", "设计模式",
]

CONCEPTS = [
    "Python", "机器学习", "深度学习和神经网络的区别",
    "API", "递归", "什么是 Transformer 架构",
    "Git 的基本操作", "Docker",
]


def generate_tool_thought(action: str, args: dict, step_num: int, is_final: bool = False) -> str:
    """根据工具和参数生成合理的思考内容"""
    if is_final:
        thoughts = [
            "任务完成，总结结果给用户。",
            "所有步骤执行完毕，现在整理最终回答。",
            "目标达成，将结果呈现给用户。",
            "已完成所有操作，返回最终结果。",
        ]
        return random.choice(thoughts)

    action_thoughts = {
        "read_local_file": [
            "用户想读取文件内容，让我使用 read_local_file 工具。",
            "需要查看文件内容，调用 read_local_file。",
            "先读取文件内容看看里面是什么。",
        ],
        "write_file": [
            "用户想创建新文件，使用 write_file 工具。",
            "需要创建一个新文件，调用 write_file。",
            "让我用 write_file 写入文件内容。",
        ],
        "edit_file": [
            "需要修改文件内容，使用 edit_file 工具。",
            "用户想替换文本，调用 edit_file。",
            "用 edit_file 修改文件中的指定内容。",
        ],
        "list_directory": [
            "用户想查看目录内容，使用 list_directory 工具。",
            "需要列出目录下的文件，调用 list_directory。",
            "让我查看这个目录里有什么文件。",
        ],
        "execute_python": [
            "用户想运行 Python 代码，使用 execute_python 工具。",
            "需要执行 Python 代码，直接调用 execute_python。",
            "让我用 Python 来计算/处理这个任务。",
        ],
        "search": [
            "用户想搜索信息，使用 search 工具。",
            "需要查找相关资料，调用 search 工具。",
            "让我搜索一下这个主题。",
        ],
        "read_webpage": [
            "用户想读取网页内容，使用 read_webpage 工具。",
            "需要查看网页内容，调用 read_webpage。",
        ],
        "learn_knowledge": [
            "搜索到了相关资料，现在用 learn_knowledge 保存知识。",
            "获取到信息了，用 learn_knowledge 记录下来。",
        ],
        "query_knowledge": [
            "用户想查询已学知识，使用 query_knowledge 工具。",
            "需要从知识库中检索信息，调用 query_knowledge。",
        ],
    }

    if action in action_thoughts:
        return random.choice(action_thoughts[action])
    return f"接下来需要执行 {action} 来完成这一步。"


def generate_final_answer(action: Optional[str], observation_placeholder: str = "{observation}") -> str:
    """生成最终的回答文本"""
    if action == "read_local_file":
        answers = [
            f"文件内容如下：\n{observation_placeholder}",
            f"读取结果：\n{observation_placeholder}",
        ]
        return random.choice(answers)
    elif action == "write_file":
        return "文件创建成功。"
    elif action == "edit_file":
        return "修改完成。"
    elif action == "list_directory":
        return f"目录内容：\n{observation_placeholder}"
    elif action == "execute_python":
        return f"执行结果：\n{observation_placeholder}"
    elif action == "search":
        return f"搜索结果：\n{observation_placeholder}"
    elif action == "read_webpage":
        return f"网页内容摘要：\n{observation_placeholder}"
    elif action == "learn_knowledge":
        return "知识已学习并保存。"
    elif action == "query_knowledge":
        return f"查询结果：\n{observation_placeholder}"
    else:
        return "已完成所有操作。"
    

def generate_direct_answer(concept: str) -> str:
    """生成无工具时的直接回答"""
    templates = [
        f"{concept} 是一个重要的概念。\n\n其主要特点包括：\n1. ...\n2. ...\n3. ...\n\n如果需要更详细的信息，我可以帮你搜索。",
        f"关于 {concept}，\n\n这是计算机科学/编程中的一个基础概念。\n\n它的核心思想是...\n\n你可以进一步搜索了解更多细节。",
        f"{concept} 的基本介绍：\n\n- 定义：...\n- 用途：...\n- 示例：...\n\n有什么具体方面想了解的吗？",
    ]
    return random.choice(templates)


def generate_react_sample(
    template: dict,
    param_pool: dict,
    max_steps: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    根据模板和参数池生成一条 ReAct 样本。
    
    返回 None 表示参数随机采样失败（跳过）。
    """
    tools = template["tools"]
    template_str = template["template"]

    if not tools:
        # 无工具的直接回答
        concept = random.choice(CONCEPTS)
        task = template_str.format(concept=concept)
        answer = generate_direct_answer(concept)
        return {
            "task": task,
            "steps": [{"thought": "这是一个基础知识问题，直接回答即可。", "final_answer": answer}],
        }

    # 采样参数
    params = {}
    if "{path}" in template_str:
        params["path"] = random.choice(FILE_PATHS)
    if "{content}" in template_str:
        params["content"] = random.choice(FILE_CONTENTS)
    if "{old_text}" in template_str or "{old}" in template_str:
        params["old_text"] = random.choice(OLD_TEXTS)
    if "{new_text}" in template_str or "{new}" in template_str:
        params["new_text"] = random.choice(NEW_TEXTS)
    if "{dir}" in template_str:
        params["dir"] = random.choice(DIRECTORIES)
    if "{expr}" in template_str:
        params["expr"] = random.choice(EXPRESSIONS)
    if "{desc}" in template_str:
        params["desc"] = random.choice(PYTHON_DESCRIPTIONS)
    if "{query}" in template_str:
        params["query"] = random.choice(QUERIES)
    if "{url}" in template_str:
        params["url"] = random.choice(URLS)
    if "{topic}" in template_str:
        params["topic"] = random.choice(TOPICS)
    if "{concept}" in template_str:
        params["concept"] = random.choice(CONCEPTS)
    if "{keyword}" in template_str:
        params["keyword"] = random.choice(["TODO", "FIXME", "version"])
    if "{new_word}" in template_str:
        params["new_word"] = random.choice(["已完成", "已修复", "version=2.0"])
    if "{action1}" in template_str:
        params["action1"] = random.choice(["搜索信息", "读取文件", "创建文件"])
    if "{action2}" in template_str:
        params["action2"] = random.choice(["保存结果", "修改配置", "验证内容"])
    if "{path}" not in params and "path" in template_str:
        params["path"] = random.choice(FILE_PATHS)

    # 格式化任务描述
    try:
        task = template_str.format(**params)
    except KeyError:
        # 参数不足，跳过这个样本
        return None

    # 构建步骤
    steps = []
    for i, tool in enumerate(tools):
        action_args = {}

        # 为每种工具生成适合的参数
        if tool == "read_local_file":
            path = params.get("path", random.choice(FILE_PATHS))
            action_args["input"] = path
        elif tool == "write_file":
            path = params.get("path", random.choice(FILE_PATHS))
            content = params.get("content", random.choice(FILE_CONTENTS))
            action_args["input"] = f"{path} | {content}"
        elif tool == "edit_file":
            path = params.get("path", random.choice(FILE_PATHS))
            old = params.get("old_text", random.choice(OLD_TEXTS))
            new = params.get("new_text", random.choice(NEW_TEXTS))
            action_args["input"] = f"{path} | {old} | {new}"
        elif tool == "list_directory":
            dir_name = params.get("dir", random.choice(DIRECTORIES))
            action_args["input"] = dir_name
        elif tool == "execute_python":
            code = params.get("expr", random.choice(EXPRESSIONS))
            action_args["input"] = f"# {code}\nprint('result')"
        elif tool == "search":
            query = params.get("query", random.choice(QUERIES))
            action_args["input"] = query
        elif tool == "read_webpage":
            url = params.get("url", random.choice(URLS))
            action_args["input"] = url
        elif tool == "learn_knowledge":
            topic = params.get("topic", random.choice(TOPICS))
            action_args["input"] = f"{topic}是关于{topic}的知识。"
        elif tool == "query_knowledge":
            query = params.get("query", random.choice(QUERIES))
            action_args["input"] = query

        thought = generate_tool_thought(tool, action_args, i)
        is_last = (i == len(tools) - 1)

        step = {"thought": thought, "action": tool, "action_args": action_args}

        # 如果这是最后一个工具，紧接着给出最终回答
        if is_last:
            final_thought = generate_tool_thought(tool, action_args, i, is_final=True)
            final_answer = generate_final_answer(tool)
            step["final_answer"] = final_answer

        steps.append(step)

    # 如果只有一个工具且没有最终回答（不会发生，但安全处理）
    if steps and "final_answer" not in steps[-1]:
        steps[-1]["final_answer"] = "任务完成。"

    return {"task": task, "steps": steps}


def generate_bulk_react_data(
    num_samples: int = 1000,
    templates: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    批量生成 ReAct 训练数据。

    Args:
        num_samples: 目标样本数量
        templates: 模板列表，默认为 TASK_TEMPLATES

    Returns:
        生成的 ReAct 样本列表
    """
    if templates is None:
        templates = TASK_TEMPLATES

    samples = []
    attempts = 0
    max_attempts = num_samples * 5  # 防止死循环

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        template = random.choice(templates)
        sample = generate_react_sample(template, {})

        if sample is not None:
            samples.append(sample)

    logger.info(
        f"Generated {len(samples)} ReAct samples "
        f"(attempts: {attempts}, skip rate: {(attempts-len(samples))/max(attempts,1)*100:.0f}%)"
    )
    return samples


def generate_bulk_conversation_data(num_samples: int = 500) -> List[Dict[str, Any]]:
    """批量生成对话训练数据"""
    samples = []

    for i in range(num_samples):
        concept = random.choice(CONCEPTS)
        system_prompt = "你是 Taiji AI 助手，一个智能的本地 AI 助手。"

        user_questions = [
            f"帮我解释一下{concept}",
            f"什么是{concept}？",
            f"能讲讲{concept}吗？",
            f"你好，请问{concept}是什么？",
            f"我需要了解{concept}",
        ]

        # 用多轮对话
        num_rounds = random.randint(1, 3)
        messages = [{"role": "system", "content": system_prompt}]

        for r in range(num_rounds):
            question = random.choice(user_questions)
            answer = generate_direct_answer(concept)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        samples.append({"messages": messages})

    logger.info(f"Generated {len(samples)} conversation samples")
    return samples


def export_data(
    react_samples: List[Dict],
    conv_samples: List[Dict],
    output_dir: str = "taiji/training_data",
):
    """
    导出训练数据到 JSONL 文件。

    Args:
        react_samples: ReAct 样本列表
        conv_samples: 对话样本列表
        output_dir: 输出目录
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "react_data.jsonl"), "w", encoding="utf-8") as f:
        for item in react_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "conversation_data.jsonl"), "w", encoding="utf-8") as f:
        for item in conv_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported {len(react_samples)} ReAct samples to {output_dir}/react_data.jsonl")
    print(f"Exported {len(conv_samples)} conversation samples to {output_dir}/conversation_data.jsonl")


def load_exported_data(data_dir: str) -> tuple:
    """
    加载已导出的训练数据。

    Args:
        data_dir: 数据目录

    Returns:
        (react_samples, conv_samples)
    """
    import os

    react_samples = []
    conv_samples = []

    react_path = os.path.join(data_dir, "react_data.jsonl")
    if os.path.exists(react_path):
        with open(react_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    react_samples.append(json.loads(line))

    conv_path = os.path.join(data_dir, "conversation_data.jsonl")
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv_samples.append(json.loads(line))

    return react_samples, conv_samples