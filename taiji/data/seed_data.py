"""
种子训练数据

人工构造的 ReAct 样本，用于初始训练。
覆盖 Taiji 的核心工具和常见使用场景。
"""
from typing import List, Dict, Any


def get_seed_react_data() -> List[Dict[str, Any]]:
    """
    获取种子 ReAct 训练数据。

    每条样本的格式:
    {
        "task": "用户任务描述",
        "steps": [
            {"thought": "...", "action": "tool_name", "action_args": {"input": "..."}},
            {"thought": "...", "action": "tool_name", "action_args": {"input": "..."}},
            {"thought": "...", "final_answer": "..."},
        ]
    }
    """
    return [
        # === 文件操作 ===
        {
            "task": "读取当前目录下的 requirements.txt 文件",
            "steps": [
                {
                    "thought": "用户想读取 requirements.txt 文件，我需要使用 read_local_file 工具。",
                    "action": "read_local_file",
                    "action_args": {"input": "requirements.txt"},
                },
                {
                    "thought": "文件内容已获取，直接返回给用户。",
                    "final_answer": "以下是 requirements.txt 的内容：\n{observation}",
                },
            ],
        },
        {
            "task": "在 agent_workspace 目录下创建一个 hello.py 文件，内容是打印 Hello World",
            "steps": [
                {
                    "thought": "用户想创建一个 Python 文件，我需要使用 write_file 工具。格式是 '路径 | 内容'。",
                    "action": "write_file",
                    "action_args": {"input": "hello.py | print('Hello World')"},
                },
                {
                    "thought": "文件创建成功，告知用户。",
                    "final_answer": "已创建 hello.py 文件，内容为 `print('Hello World')`。",
                },
            ],
        },
        {
            "task": "列出当前工作目录的所有文件",
            "steps": [
                {
                    "thought": "用户想查看目录内容，使用 list_directory 工具。",
                    "action": "list_directory",
                    "action_args": {"input": "."},
                },
                {
                    "thought": "目录列表已获取，格式化后返回。",
                    "final_answer": "当前目录下的文件：\n{observation}",
                },
            ],
        },
        {
            "task": "把 hello.py 里的 Hello World 改成 Hello Python",
            "steps": [
                {
                    "thought": "用户想编辑文件内容，使用 edit_file 工具。格式是 '路径 | 旧文本 | 新文本'。",
                    "action": "edit_file",
                    "action_args": {"input": "hello.py | Hello World | Hello Python"},
                },
                {
                    "thought": "编辑成功。",
                    "final_answer": "已将 hello.py 中的 'Hello World' 修改为 'Hello Python'。",
                },
            ],
        },
        {
            "task": "读取 config.json 文件并告诉我里面的 model_name 是什么",
            "steps": [
                {
                    "thought": "用户想查看配置文件内容，先读取文件。",
                    "action": "read_local_file",
                    "action_args": {"input": "config.json"},
                },
                {
                    "thought": "文件内容已获取，从 JSON 中提取 model_name 字段。",
                    "final_answer": "config.json 中的 model_name 为：{observation}",
                },
            ],
        },
        {
            "task": "在 docs 目录下创建一个 README.md 文件",
            "steps": [
                {
                    "thought": "用户想在 docs 目录下创建 README.md，使用 write_file 工具。",
                    "action": "write_file",
                    "action_args": {"input": "docs/README.md | # 文档目录\n\n本目录存放项目文档。"},
                },
                {
                    "thought": "文件创建成功。",
                    "final_answer": "已在 docs 目录下创建 README.md 文件。",
                },
            ],
        },
        {
            "task": "查看 taiji 目录下有哪些文件",
            "steps": [
                {
                    "thought": "用户想查看特定目录的内容，使用 list_directory 工具。",
                    "action": "list_directory",
                    "action_args": {"input": "taiji"},
                },
                {
                    "thought": "目录列表已获取。",
                    "final_answer": "taiji 目录下的文件：\n{observation}",
                },
            ],
        },
        {
            "task": "把 version.json 里的 version 改成 2.0.0",
            "steps": [
                {
                    "thought": "用户想修改 JSON 文件中的版本号。先读取当前内容。",
                    "action": "read_local_file",
                    "action_args": {"input": "version.json"},
                },
                {
                    "thought": "获取到内容，现在用 edit_file 修改版本号。",
                    "action": "edit_file",
                    "action_args": {"input": "version.json | 1.0.0 | 2.0.0"},
                },
                {
                    "thought": "版本号已修改。",
                    "final_answer": "已将 version.json 中的版本号修改为 2.0.0。",
                },
            ],
        },

        # === 代码执行 ===
        {
            "task": "用 Python 计算 1 到 100 的和",
            "steps": [
                {
                    "thought": "用户想用 Python 计算，我需要使用 execute_python 工具执行代码。",
                    "action": "execute_python",
                    "action_args": {"input": "result = sum(range(1, 101))\nprint(f'1到100的和为: {result}')"},
                },
                {
                    "thought": "代码执行成功，返回结果。",
                    "final_answer": "1 到 100 的和为：{observation}",
                },
            ],
        },
        {
            "task": "写一个 Python 函数判断素数，然后测试 17 是否是素数",
            "steps": [
                {
                    "thought": "用户想写一个素数判断函数并测试。我用 execute_python 执行完整代码。",
                    "action": "execute_python",
                    "action_args": {"input": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(f'17 是素数: {is_prime(17)}')"},
                },
                {
                    "thought": "代码执行成功。",
                    "final_answer": "素数判断函数已编写并测试。{observation}",
                },
            ],
        },
        {
            "task": "用 Python 计算斐波那契数列的前 20 项",
            "steps": [
                {
                    "thought": "用户想计算斐波那契数列，使用 execute_python 工具。",
                    "action": "execute_python",
                    "action_args": {"input": "def fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(20))"},
                },
                {
                    "thought": "代码执行成功。",
                    "final_answer": "斐波那契数列前 20 项：\n{observation}",
                },
            ],
        },
        {
            "task": "用 Python 生成一个随机密码，要求 16 位，包含大小写字母和数字",
            "steps": [
                {
                    "thought": "用户想生成随机密码，使用 Python 的 secrets 和 string 模块。",
                    "action": "execute_python",
                    "action_args": {"input": "import secrets\nimport string\nchars = string.ascii_letters + string.digits\npassword = ''.join(secrets.choice(chars) for _ in range(16))\nprint(f'随机密码: {password}')"},
                },
                {
                    "thought": "密码生成成功。",
                    "final_answer": "已生成随机密码：{observation}",
                },
            ],
        },
        {
            "task": "写一个 Python 脚本统计当前目录下所有 .py 文件的数量",
            "steps": [
                {
                    "thought": "用户想统计 Python 文件数量，使用 execute_python。",
                    "action": "execute_python",
                    "action_args": {"input": "import os\npy_files = [f for f in os.listdir('.') if f.endswith('.py')]\nprint(f'Python 文件数量: {len(py_files)}')\nfor f in py_files:\n    print(f'  - {f}')"},
                },
                {
                    "thought": "统计完成。",
                    "final_answer": "{observation}",
                },
            ],
        },
        {
            "task": "用 Python 计算一个列表 [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5] 的平均值、中位数和标准差",
            "steps": [
                {
                    "thought": "用户想做统计计算，使用 statistics 模块。",
                    "action": "execute_python",
                    "action_args": {"input": "import statistics\ndata = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nprint(f'平均值: {statistics.mean(data):.2f}')\nprint(f'中位数: {statistics.median(data):.2f}')\nprint(f'标准差: {statistics.stdev(data):.2f}')"},
                },
                {
                    "thought": "计算完成。",
                    "final_answer": "{observation}",
                },
            ],
        },

        # === 网络搜索 ===
        {
            "task": "搜索一下 Python 3.12 有什么新特性",
            "steps": [
                {
                    "thought": "用户想了解 Python 3.12 的新特性，我需要使用 search 工具搜索。",
                    "action": "search",
                    "action_args": {"input": "Python 3.12 新特性"},
                },
                {
                    "thought": "搜索结果已获取，整理后返回给用户。",
                    "final_answer": "根据搜索结果，Python 3.12 的主要新特性包括：\n{observation}",
                },
            ],
        },
        {
            "task": "帮我搜索一下 Taiji 项目的 GitHub 地址",
            "steps": [
                {
                    "thought": "用户想找 Taiji 的 GitHub 地址，使用 search 工具。",
                    "action": "search",
                    "action_args": {"input": "Taiji AI assistant GitHub"},
                },
                {
                    "thought": "搜索完成，返回结果。",
                    "final_answer": "搜索结果：\n{observation}",
                },
            ],
        },
        {
            "task": "搜索 PyTorch 最新版本的安装方法",
            "steps": [
                {
                    "thought": "用户想安装 PyTorch，搜索最新的安装指南。",
                    "action": "search",
                    "action_args": {"input": "PyTorch latest version install guide 2024"},
                },
                {
                    "thought": "搜索结果已获取。",
                    "final_answer": "PyTorch 最新安装方法：\n{observation}",
                },
            ],
        },
        {
            "task": "搜索一下如何使用 LoRA 微调大语言模型",
            "steps": [
                {
                    "thought": "用户想了解 LoRA 微调，搜索相关教程。",
                    "action": "search",
                    "action_args": {"input": "LoRA fine-tune LLM tutorial"},
                },
                {
                    "thought": "搜索到相关资料。",
                    "final_answer": "LoRA 微调大语言模型的方法：\n{observation}",
                },
            ],
        },

        # === 网页读取 ===
        {
            "task": "帮我读取 https://docs.python.org/3/ 这个网页的内容",
            "steps": [
                {
                    "thought": "用户想读取一个网页，使用 read_webpage 工具。",
                    "action": "read_webpage",
                    "action_args": {"input": "https://docs.python.org/3/"},
                },
                {
                    "thought": "网页内容已获取，摘要返回。",
                    "final_answer": "网页内容摘要：\n{observation}",
                },
            ],
        },
        {
            "task": "读取 https://pytorch.org/get-started/locally/ 获取 PyTorch 安装命令",
            "steps": [
                {
                    "thought": "用户想从 PyTorch 官网获取安装命令。",
                    "action": "read_webpage",
                    "action_args": {"input": "https://pytorch.org/get-started/locally/"},
                },
                {
                    "thought": "网页内容已获取。",
                    "final_answer": "PyTorch 安装指南：\n{observation}",
                },
            ],
        },

        # === 项目创建 ===
        {
            "task": "创建一个名为 calculator 的 Python 项目",
            "steps": [
                {
                    "thought": "用户想创建一个 Python 项目，使用 create_project 工具。格式是 '类型 | 名称'。",
                    "action": "create_project",
                    "action_args": {"input": "python-script | calculator"},
                },
                {
                    "thought": "项目创建成功。",
                    "final_answer": "已创建 Python 项目 'calculator'，包含 main.py 和 README.md 文件。",
                },
            ],
        },
        {
            "task": "创建一个 Flask Web 项目，名字叫 myapp",
            "steps": [
                {
                    "thought": "用户想创建 Flask 项目，使用 create_project 工具。",
                    "action": "create_project",
                    "action_args": {"input": "web-app | myapp"},
                },
                {
                    "thought": "项目创建成功。",
                    "final_answer": "已创建 Flask Web 项目 'myapp'，包含 app.py、templates/ 和 static/ 目录。",
                },
            ],
        },

        # === 代码分析 ===
        {
            "task": "分析一下 main.py 这个文件的代码结构",
            "steps": [
                {
                    "thought": "用户想分析代码，使用 analyze_code 工具。",
                    "action": "analyze_code",
                    "action_args": {"input": "main.py"},
                },
                {
                    "thought": "分析结果已获取。",
                    "final_answer": "main.py 的代码分析结果：\n{observation}",
                },
            ],
        },
        {
            "task": "分析 taiji/architecture.py 的代码质量",
            "steps": [
                {
                    "thought": "用户想分析特定文件的代码质量。",
                    "action": "analyze_code",
                    "action_args": {"input": "taiji/architecture.py"},
                },
                {
                    "thought": "分析完成。",
                    "final_answer": "代码质量分析：\n{observation}",
                },
            ],
        },

        # === 依赖安装 ===
        {
            "task": "帮我安装 requests 库",
            "steps": [
                {
                    "thought": "用户想安装 Python 包，使用 install_dependency 工具。",
                    "action": "install_dependency",
                    "action_args": {"input": "requests"},
                },
                {
                    "thought": "安装完成。",
                    "final_answer": "requests 库安装完成。",
                },
            ],
        },
        {
            "task": "安装 torch 和 torchvision",
            "steps": [
                {
                    "thought": "用户想安装 PyTorch 相关包。",
                    "action": "install_dependency",
                    "action_args": {"input": "torch torchvision"},
                },
                {
                    "thought": "安装完成。",
                    "final_answer": "torch 和 torchvision 安装完成。",
                },
            ],
        },

        # === 知识学习 ===
        {
            "task": "学习一下 Python 装饰器的用法",
            "steps": [
                {
                    "thought": "用户想学习 Python 装饰器，先搜索相关资料。",
                    "action": "search",
                    "action_args": {"input": "Python decorator 装饰器 教程"},
                },
                {
                    "thought": "搜索到了相关资料，使用 learn_knowledge 保存。",
                    "action": "learn_knowledge",
                    "action_args": {"input": "Python装饰器是一种设计模式，允许在不修改原函数代码的情况下扩展函数功能。使用 @语法糖，例如 @decorator。常见用途：日志、权限校验、缓存。"},
                },
                {
                    "thought": "知识已学习，整理后返回。",
                    "final_answer": "Python 装饰器是一种强大的设计模式：\n\n1. 使用 @decorator 语法糖\n2. 可以在不修改原函数的情况下扩展功能\n3. 常见用途：日志记录、权限校验、缓存\n\n示例：\n```python\ndef my_decorator(func):\n    def wrapper():\n        print('before')\n        func()\n        print('after')\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print('hello')\n```",
                },
            ],
        },

        # === 知识查询 ===
        {
            "task": "查一下之前学过的关于装饰器的知识",
            "steps": [
                {
                    "thought": "用户想查询之前学过的知识，使用 query_knowledge 工具。",
                    "action": "query_knowledge",
                    "action_args": {"input": "装饰器"},
                },
                {
                    "thought": "查询结果已获取。",
                    "final_answer": "根据知识库记录：\n{observation}",
                },
            ],
        },

        # === 命令执行 ===
        {
            "task": "查看当前 Python 版本",
            "steps": [
                {
                    "thought": "用户想查看 Python 版本，使用 run_command 工具。",
                    "action": "run_command",
                    "action_args": {"input": "python --version"},
                },
                {
                    "thought": "命令执行完成。",
                    "final_answer": "当前 Python 版本：{observation}",
                },
            ],
        },
        {
            "task": "查看当前目录的磁盘使用情况",
            "steps": [
                {
                    "thought": "用户想查看磁盘使用情况。",
                    "action": "run_command",
                    "action_args": {"input": "dir"},
                },
                {
                    "thought": "命令执行完成。",
                    "final_answer": "当前目录内容：\n{observation}",
                },
            ],
        },
        {
            "task": "查看 pip 已安装的包列表",
            "steps": [
                {
                    "thought": "用户想查看已安装的 Python 包。",
                    "action": "run_command",
                    "action_args": {"input": "pip list"},
                },
                {
                    "thought": "命令执行完成。",
                    "final_answer": "已安装的 Python 包：\n{observation}",
                },
            ],
        },
        {
            "task": "查看当前 git 分支和状态",
            "steps": [
                {
                    "thought": "用户想查看 git 状态。",
                    "action": "run_command",
                    "action_args": {"input": "git status"},
                },
                {
                    "thought": "命令执行完成。",
                    "final_answer": "Git 状态：\n{observation}",
                },
            ],
        },

        # === 多步骤任务 ===
        {
            "task": "创建一个 test.txt 文件，写入今天的日期，然后读取验证",
            "steps": [
                {
                    "thought": "这是一个多步骤任务。先获取今天的日期。",
                    "action": "execute_python",
                    "action_args": {"input": "from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d'))"},
                },
                {
                    "thought": "获取到日期，现在创建文件。",
                    "action": "write_file",
                    "action_args": {"input": "test.txt | 今天的日期是: {observation}"},
                },
                {
                    "thought": "文件创建成功，现在读取验证。",
                    "action": "read_local_file",
                    "action_args": {"input": "test.txt"},
                },
                {
                    "thought": "验证完成，文件内容正确。",
                    "final_answer": "已完成：\n1. 获取今天的日期\n2. 创建 test.txt 文件\n3. 验证文件内容：\n{observation}",
                },
            ],
        },
        {
            "task": "搜索 Python 异步编程的资料，然后把要点保存到 notes.md",
            "steps": [
                {
                    "thought": "用户想搜索并保存资料。先搜索。",
                    "action": "search",
                    "action_args": {"input": "Python async await 异步编程 教程"},
                },
                {
                    "thought": "搜索到资料，现在保存要点到文件。",
                    "action": "write_file",
                    "action_args": {"input": "notes.md | # Python 异步编程要点\n\n- 使用 async def 定义协程函数\n- 使用 await 等待异步操作完成\n- 使用 asyncio.run() 启动事件循环\n- 常用库: aiohttp, aiofiles"},
                },
                {
                    "thought": "文件保存成功。",
                    "final_answer": "已搜索 Python 异步编程资料并保存到 notes.md 文件。",
                },
            ],
        },
        {
            "task": "先读取 app_settings.json，然后把 device 字段改成 cuda",
            "steps": [
                {
                    "thought": "用户想修改配置文件。先读取当前内容。",
                    "action": "read_local_file",
                    "action_args": {"input": "app_settings.json"},
                },
                {
                    "thought": "获取到配置内容，现在修改 device 字段。",
                    "action": "edit_file",
                    "action_args": {"input": "app_settings.json | \"device\": \"cpu\" | \"device\": \"cuda\""},
                },
                {
                    "thought": "修改完成。",
                    "final_answer": "已将 app_settings.json 中的 device 字段修改为 cuda。",
                },
            ],
        },
        {
            "task": "创建一个 Python 脚本计算圆的面积和周长，然后运行测试",
            "steps": [
                {
                    "thought": "用户想创建并运行一个计算圆的脚本。先创建文件。",
                    "action": "write_file",
                    "action_args": {"input": "circle.py | import math\n\ndef circle_info(radius):\n    area = math.pi * radius ** 2\n    circumference = 2 * math.pi * radius\n    return area, circumference\n\narea, circ = circle_info(5)\nprint(f'半径=5: 面积={area:.2f}, 周长={circ:.2f}')"},
                },
                {
                    "thought": "文件创建成功，现在运行测试。",
                    "action": "execute_python",
                    "action_args": {"input": "import math\n\ndef circle_info(radius):\n    area = math.pi * radius ** 2\n    circumference = 2 * math.pi * radius\n    return area, circumference\n\narea, circ = circle_info(5)\nprint(f'半径=5: 面积={area:.2f}, 周长={circ:.2f}')"},
                },
                {
                    "thought": "运行成功。",
                    "final_answer": "圆的计算结果：\n{observation}",
                },
            ],
        },
        {
            "task": "创建一个 data 目录，然后在其中创建 config.txt 文件",
            "steps": [
                {
                    "thought": "用户想创建目录和文件。先创建 config.txt 文件（write_file 会自动创建目录）。",
                    "action": "write_file",
                    "action_args": {"input": "data/config.txt | # 配置文件\nversion=1.0\nmode=production"},
                },
                {
                    "thought": "文件创建成功，验证目录结构。",
                    "action": "list_directory",
                    "action_args": {"input": "data"},
                },
                {
                    "thought": "验证完成。",
                    "final_answer": "已创建 data 目录和 config.txt 文件。\n目录内容：\n{observation}",
                },
            ],
        },

        # === 直接回答（不需要工具）===
        {
            "task": "什么是 Python？",
            "steps": [
                {
                    "thought": "这是一个基础知识问题，不需要使用工具，直接回答。",
                    "final_answer": "Python 是一种高级、解释型、通用的编程语言。它的主要特点包括：\n\n1. **简洁易读**：使用缩进而非大括号\n2. **动态类型**：变量不需要声明类型\n3. **丰富的标准库**：内置大量模块\n4. **跨平台**：支持 Windows、Linux、macOS\n5. **广泛应用**：Web 开发、数据科学、AI、自动化等",
                },
            ],
        },
        {
            "task": "解释一下什么是递归",
            "steps": [
                {
                    "thought": "这是编程概念问题，直接回答即可。",
                    "final_answer": "递归是一种编程技巧，指函数调用自身。\n\n**两个要素：**\n1. 基准条件（base case）：停止递归的条件\n2. 递归步骤（recursive step）：将问题分解为更小的子问题\n\n**示例 - 阶乘：**\n```python\ndef factorial(n):\n    if n <= 1:  # 基准条件\n        return 1\n    return n * factorial(n - 1)  # 递归步骤\n```\n\n**注意事项：**\n- 必须有基准条件，否则会无限递归\n- 递归深度过大会导致栈溢出\n- 某些场景下迭代更高效",
                },
            ],
        },
        {
            "task": "什么是 Transformer 架构？",
            "steps": [
                {
                    "thought": "这是 AI 基础知识问题，直接回答。",
                    "final_answer": "Transformer 是一种深度学习架构，由 Google 在 2017 年提出。\n\n**核心组件：**\n1. **自注意力机制**（Self-Attention）：让模型关注输入序列中的不同位置\n2. **多头注意力**：并行多个注意力头，捕获不同维度的信息\n3. **前馈网络**（FFN）：对每个位置独立处理\n4. **位置编码**：注入序列顺序信息\n\n**变体：**\n- Encoder-only: BERT（理解型）\n- Decoder-only: GPT、LLaMA（生成型）\n- Encoder-decoder: T5、BART（翻译型）\n\n**为什么重要：**\n- 并行计算，比 RNN 快\n- 长距离依赖建模能力强\n- 是当前大语言模型的基础架构",
                },
            ],
        },
        {
            "task": "解释一下 Git 的基本操作",
            "steps": [
                {
                    "thought": "这是版本控制基础知识，直接回答。",
                    "final_answer": "Git 基本操作：\n\n**初始化与克隆：**\n- `git init` — 初始化新仓库\n- `git clone <url>` — 克隆远程仓库\n\n**日常工作流：**\n- `git add .` — 暂存所有更改\n- `git commit -m \"msg\"` — 提交更改\n- `git push` — 推送到远程\n- `git pull` — 拉取远程更新\n\n**分支管理：**\n- `git branch <name>` — 创建分支\n- `git checkout <name>` — 切换分支\n- `git merge <name>` — 合并分支\n\n**查看状态：**\n- `git status` — 查看工作区状态\n- `git log --oneline` — 查看提交历史",
                },
            ],
        },
        {
            "task": "什么是 API？",
            "steps": [
                {
                    "thought": "这是基础概念问题，直接回答。",
                    "final_answer": "API（Application Programming Interface，应用程序编程接口）是软件组件之间交互的接口。\n\n**通俗理解：**\nAPI 就像餐厅的菜单——你不需要知道厨房怎么做菜，只需要通过菜单点餐。\n\n**常见类型：**\n1. **REST API** — 基于 HTTP，使用 JSON 格式\n2. **GraphQL** — 灵活查询，客户端指定数据结构\n3. **WebSocket** — 全双工通信，实时性好\n\n**示例（REST API）：**\n```\nGET /api/users     → 获取用户列表\nPOST /api/users    → 创建新用户\nPUT /api/users/1   → 更新用户\nDELETE /api/users/1 → 删除用户\n```",
                },
            ],
        },
        {
            "task": "如何提高代码的可读性？",
            "steps": [
                {
                    "thought": "这是编程最佳实践问题，直接回答。",
                    "final_answer": "提高代码可读性的方法：\n\n1. **命名规范**\n   - 变量名要有意义：`user_count` 而非 `n`\n   - 函数名用动词：`get_user()`、`calculate_total()`\n\n2. **函数设计**\n   - 每个函数只做一件事\n   - 函数体不超过 20 行\n   - 参数不超过 3-4 个\n\n3. **注释**\n   - 解释「为什么」而非「是什么」\n   - 复杂逻辑前加注释\n\n4. **代码结构**\n   - 相关代码放在一起\n   - 使用空行分隔逻辑块\n   - 保持一致的缩进风格\n\n5. **避免魔法数字**\n   - 用常量替代：`MAX_RETRY = 3`\n\n6. **遵循 PEP 8**\n   - Python 官方风格指南",
                },
            ],
        },
        {
            "task": "帮我用中文解释一下机器学习的基本概念",
            "steps": [
                {
                    "thought": "这是基础知识问题，直接用中文回答。",
                    "final_answer": "机器学习是让计算机从数据中自动学习规律的技术。\n\n**三大类型：**\n\n1. **监督学习** — 有标签数据\n   - 分类：判断邮件是否是垃圾邮件\n   - 回归：预测房价\n\n2. **无监督学习** — 无标签数据\n   - 聚类：客户分群\n   - 降维：数据压缩\n\n3. **强化学习** — 试错学习\n   - 游戏 AI、机器人控制\n\n**基本流程：**\n1. 收集数据\n2. 特征工程\n3. 选择模型\n4. 训练模型\n5. 评估效果\n6. 部署应用",
                },
            ],
        },
    ] + get_extended_react_data() + get_edge_case_react_data() + get_complex_react_data()


def get_seed_conversation_data() -> List[Dict[str, Any]]:
    """
    获取种子对话训练数据。

    格式:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
    }
    """
    return [
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手，一个智能的本地 AI 助手。你可以帮助用户完成文件操作、代码编写、信息搜索等任务。"},
                {"role": "user", "content": "你好，你能做什么？"},
                {"role": "assistant", "content": "你好！我是 Taiji AI 助手，我可以帮你完成很多任务：\n\n📁 **文件操作** — 读取、创建、编辑、删除文件\n💻 **代码执行** — 运行 Python 代码\n🔍 **信息搜索** — 搜索互联网获取信息\n🌐 **网页读取** — 读取网页内容\n📦 **项目管理** — 创建项目、安装依赖\n🧠 **知识学习** — 学习和查询知识\n\n有什么我可以帮你的吗？"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "帮我写一个快速排序算法"},
                {"role": "assistant", "content": "好的，这是一个 Python 实现的快速排序算法：\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n# 测试\narr = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(arr))  # [1, 1, 2, 3, 6, 8, 10]\n```\n\n**算法原理：**\n1. 选择一个基准元素（pivot）\n2. 将数组分为三部分：小于、等于、大于基准\n3. 递归排序左右两部分\n4. 合并结果\n\n**时间复杂度：**\n- 平均：O(n log n)\n- 最坏：O(n²)（当数组已排序时）"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是机器学习？"},
                {"role": "assistant", "content": "机器学习（Machine Learning）是人工智能的一个分支，它让计算机能够从数据中自动学习规律，而不需要显式编程。\n\n**三大类型：**\n\n1. **监督学习** — 有标签数据训练\n   - 分类：预测类别（如垃圾邮件识别）\n   - 回归：预测数值（如房价预测）\n\n2. **无监督学习** — 无标签数据\n   - 聚类：将数据分组\n   - 降维：减少特征数量\n\n3. **强化学习** — 通过试错学习\n   - 游戏 AI、机器人控制\n\n**常用算法：**\n- 线性回归、逻辑回归\n- 决策树、随机森林\n- 支持向量机（SVM）\n- 神经网络、深度学习\n\n**应用：**\n- 图像识别、语音识别\n- 推荐系统、自然语言处理\n- 自动驾驶、医疗诊断"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "Python 中的列表和元组有什么区别？"},
                {"role": "assistant", "content": "列表（list）和元组（tuple）的主要区别：\n\n| 特性 | 列表 list | 元组 tuple |\n|------|----------|------------|\n| 可变性 | 可变（mutable） | 不可变（immutable） |\n| 语法 | `[1, 2, 3]` | `(1, 2, 3)` |\n| 性能 | 较慢 | 较快 |\n| 用途 | 数据会变化时 | 数据固定时 |\n| 方法 | append, remove, sort 等 | 只有 count, index |\n\n**示例：**\n```python\n# 列表 - 可修改\nlst = [1, 2, 3]\nlst.append(4)  # [1, 2, 3, 4]\n\n# 元组 - 不可修改\ntup = (1, 2, 3)\n# tup.append(4)  # 报错！\n```\n\n**选择建议：**\n- 需要修改数据 → 用列表\n- 数据不应该被修改 → 用元组\n- 作为字典的键 → 必须用元组"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "如何处理 Python 中的异常？"},
                {"role": "assistant", "content": "Python 异常处理使用 try-except 语句：\n\n**基本语法：**\n```python\ntry:\n    # 可能出错的代码\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    # 处理特定异常\n    print(f'错误: {e}')\nexcept Exception as e:\n    # 处理所有其他异常\n    print(f'未知错误: {e}')\nelse:\n    # 没有异常时执行\n    print('成功')\nfinally:\n    # 无论如何都执行\n    print('结束')\n```\n\n**常见异常类型：**\n- `ValueError` — 值错误\n- `TypeError` — 类型错误\n- `FileNotFoundError` — 文件不存在\n- `KeyError` — 字典键不存在\n- `IndexError` — 索引越界\n\n**最佳实践：**\n1. 捕获具体异常，不要用裸 except\n2. 不要忽略异常（至少记录日志）\n3. 用 EAFP 风格（先尝试，再处理）"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "帮我写一个 Python 的 HTTP 服务器"},
                {"role": "assistant", "content": "这是一个简单的 Python HTTP 服务器：\n\n**方法一：使用内置模块（最简单）**\n```python\nfrom http.server import HTTPServer, SimpleHTTPRequestHandler\n\nserver = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)\nprint('服务器运行在 http://localhost:8000')\nsimpleHTTPServer.serve_forever()\n```\n\n**方法二：使用 Flask（功能更强）**\n```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return 'Hello, World!'\n\n@app.route('/api/data')\ndef get_data():\n    return jsonify({'message': 'Hello from API'})\n\nif __name__ == '__main__':\n    app.run(host='0.0.0.0', port=8000, debug=True)\n```\n\n**方法三：异步服务器（高性能）**\n```python\nimport asyncio\nfrom aiohttp import web\n\nasync def handle(request):\n    return web.Response(text='Hello!')\n\napp = web.Application()\napp.router.add_get('/', handle)\nweb.run_app(app, port=8000)\n```"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 Docker？"},
                {"role": "assistant", "content": "Docker 是一个容器化平台，可以将应用及其依赖打包成一个轻量级、可移植的容器。\n\n**核心概念：**\n\n1. **镜像（Image）** — 只读模板，包含应用代码和运行环境\n2. **容器（Container）** — 镜像的运行实例\n3. **Dockerfile** — 定义如何构建镜像的脚本\n4. **Docker Compose** — 管理多容器应用\n\n**常用命令：**\n```bash\ndocker build -t myapp .    # 构建镜像\ndocker run -p 8000:8000 myapp  # 运行容器\ndocker ps                   # 查看运行中的容器\ndocker stop <id>            # 停止容器\ndocker images               # 查看所有镜像\n```\n\n**优势：**\n- 环境一致性（开发=生产）\n- 快速部署和扩展\n- 资源隔离\n- 版本控制\n\n**vs 虚拟机：**\nDocker 更轻量，启动更快，占用资源更少"},
            ],
        },
    ] + get_extended_conversation_data() + get_diverse_conversation_data()


def get_extended_react_data() -> List[Dict[str, Any]]:
    """扩展 ReAct 训练数据：覆盖更多场景和边界情况"""
    return [
        # === 错误处理 ===
        {
            "task": "读取一个不存在的文件 data.csv",
            "steps": [
                {"thought": "用户想读取 data.csv，先尝试读取。", "action": "read_local_file", "action_args": {"input": "data.csv"}},
                {"thought": "文件不存在，应该告知用户并提供替代方案。", "final_answer": "data.csv 文件不存在。请确认文件路径是否正确，或者我可以帮你创建一个示例 CSV 文件。"},
            ],
        },
        {
            "task": "运行一段有语法错误的代码 print('hello'",
            "steps": [
                {"thought": "用户给了代码，但缺少右括号。我应该先尝试运行，然后报告错误。", "action": "execute_python", "action_args": {"input": "print('hello'"}},
                {"thought": "代码有语法错误，告知用户具体问题。", "final_answer": "代码有语法错误：缺少右括号。应改为 `print('hello')`。"},
            ],
        },
        {
            "task": "安装一个不存在的包 nonexistpkg12345",
            "steps": [
                {"thought": "用户想安装一个包，尝试安装。", "action": "install_dependency", "action_args": {"input": "nonexistpkg12345"}},
                {"thought": "安装失败，包不存在。告知用户。", "final_answer": "安装失败：nonexistpkg12345 包不存在。请检查包名是否正确。"},
            ],
        },

        # === 文件批量操作 ===
        {
            "task": "把当前目录下所有 .txt 文件的内容合并到 combined.txt",
            "steps": [
                {"thought": "先列出目录找到所有 txt 文件。", "action": "list_directory", "action_args": {"input": "."}},
                {"thought": "用 Python 脚本合并所有 txt 文件。", "action": "execute_python", "action_args": {"input": "import os\nimport glob\n\ntxt_files = glob.glob('*.txt')\ncombined = []\nfor f in txt_files:\n    with open(f, 'r', encoding='utf-8') as fh:\n        combined.append(f'=== {f} ===')\n        combined.append(fh.read())\n        combined.append('')\n\nwith open('combined.txt', 'w', encoding='utf-8') as out:\n    out.write('\\n'.join(combined))\n\nprint(f'已合并 {len(txt_files)} 个文件到 combined.txt')"}},
                {"thought": "合并完成。", "final_answer": "{observation}"},
            ],
        },
        {
            "task": "给当前目录下所有 .py 文件的开头添加一行注释 # Auto-generated header",
            "steps": [
                {"thought": "先找到所有 py 文件。", "action": "execute_python", "action_args": {"input": "import glob\npy_files = glob.glob('*.py')\nprint('\\n'.join(py_files))"}},
                {"thought": "用 Python 批量添加头部注释。", "action": "execute_python", "action_args": {"input": "import glob\n\nheader = '# Auto-generated header\\n'\nfor f in glob.glob('*.py'):\n    with open(f, 'r', encoding='utf-8') as fh:\n        content = fh.read()\n    if not content.startswith(header):\n        with open(f, 'w', encoding='utf-8') as fh:\n            fh.write(header + content)\n        print(f'已添加: {f}')\n    else:\n        print(f'跳过（已有）: {f}')"}},
                {"thought": "操作完成。", "final_answer": "已为所有 .py 文件添加头部注释。"},
            ],
        },

        # === 数据处理 ===
        {
            "task": "读取 data.json 并统计里面有多少条记录",
            "steps": [
                {"thought": "先读取 JSON 文件。", "action": "read_local_file", "action_args": {"input": "data.json"}},
                {"thought": "解析 JSON 并统计。", "action": "execute_python", "action_args": {"input": "import json\nwith open('data.json', 'r') as f:\n    data = json.load(f)\nif isinstance(data, list):\n    print(f'共 {len(data)} 条记录')\nelse:\n    print(f'数据类型: {type(data).__name__}')"}},
                {"thought": "统计完成。", "final_answer": "{observation}"},
            ],
        },
        {
            "task": "把 CSV 文件 data.csv 转换成 JSON 格式",
            "steps": [
                {"thought": "先读取 CSV 文件内容。", "action": "read_local_file", "action_args": {"input": "data.csv"}},
                {"thought": "用 Python 转换格式。", "action": "execute_python", "action_args": {"input": "import csv\nimport json\n\nwith open('data.csv', 'r', encoding='utf-8') as f:\n    reader = csv.DictReader(f)\n    data = list(reader)\n\nwith open('data.json', 'w', encoding='utf-8') as f:\n    json.dump(data, f, ensure_ascii=False, indent=2)\n\nprint(f'已转换 {len(data)} 条记录到 data.json')"}},
                {"thought": "转换完成。", "final_answer": "已将 CSV 转换为 JSON 格式，共 {observation} 条记录。"},
            ],
        },
        {
            "task": "生成一个包含 100 条随机用户数据的 JSON 文件",
            "steps": [
                {"thought": "用 Python 生成随机用户数据。", "action": "execute_python", "action_args": {"input": "import json\nimport random\nimport string\n\ndef random_name():\n    first = random.choice(['Alice','Bob','Charlie','Diana','Eve','Frank','Grace','Henry'])\n    last = random.choice(['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller'])\n    return f'{first} {last}''\n\nusers = []\nfor i in range(100):\n    users.append({\n        'id': i + 1,\n        'name': random_name(),\n        'email': f'user{i+1}@example.com',\n        'age': random.randint(18, 65),\n    })\n\nwith open('users.json', 'w', encoding='utf-8') as f:\n    json.dump(users, f, ensure_ascii=False, indent=2)\n\nprint(f'已生成 {len(users)} 条用户数据到 users.json')"}},
                {"thought": "数据生成完成。", "final_answer": "已生成包含 100 条随机用户数据的 users.json 文件。"},
            ],
        },

        # === 搜索 + 分析 ===
        {
            "task": "搜索 Python 的 asyncio 库的用法，然后写一个示例脚本",
            "steps": [
                {"thought": "先搜索 asyncio 的用法。", "action": "search", "action_args": {"input": "Python asyncio 异步编程 示例"}},
                {"thought": "搜索到了资料，现在写一个示例脚本。", "action": "write_file", "action_args": {"input": "async_example.py | import asyncio\n\nasync def fetch_data(name, delay):\n    print(f'{name} 开始')\n    await asyncio.sleep(delay)\n    print(f'{name} 完成')\n    return f'{name} 的数据'\n\nasync def main():\n    tasks = [\n        fetch_data('任务A', 2),\n        fetch_data('任务B', 1),\n        fetch_data('任务C', 3),\n    ]\n    results = await asyncio.gather(*tasks)\n    print(f'结果: {results}')\n\nasyncio.run(main())"}},
                {"thought": "脚本创建成功，运行测试。", "action": "execute_python", "action_args": {"input": "import asyncio\n\nasync def fetch_data(name, delay):\n    print(f'{name} 开始')\n    await asyncio.sleep(delay)\n    print(f'{name} 完成')\n    return f'{name} 的数据'\n\nasync def main():\n    tasks = [\n        fetch_data('任务A', 2),\n        fetch_data('任务B', 1),\n        fetch_data('任务C', 3),\n    ]\n    results = await asyncio.gather(*tasks)\n    print(f'结果: {results}')\n\nasyncio.run(main())"}},
                {"thought": "运行成功。", "final_answer": "已创建并测试 asyncio 示例脚本：\n{observation}"},
            ],
        },

        # === 调试场景 ===
        {
            "task": "这段代码运行报错了，帮我看看：def add(a, b): return a + b; print(add('1', 2))",
            "steps": [
                {"thought": "用户代码有类型错误，字符串和数字不能相加。先运行确认。", "action": "execute_python", "action_args": {"input": "def add(a, b):\n    return a + b\nprint(add('1', 2))"}},
                {"thought": "确认是 TypeError，给用户解释并修复。", "action": "execute_python", "action_args": {"input": "def add(a, b):\n    return int(a) + int(b)\nprint(add('1', 2))  # 输出: 3"}},
                {"thought": "修复成功。", "final_answer": "原始代码的问题：字符串 '1' 和数字 2 不能直接相加。修复方法是将参数转为 int。修复后输出：3"},
            ],
        },
        {
            "task": "帮我找 main.py 里的 bug，运行后结果不对",
            "steps": [
                {"thought": "先读取 main.py 看看代码。", "action": "read_local_file", "action_args": {"input": "main.py"}},
                {"thought": "分析代码逻辑，运行看看。", "action": "execute_python", "action_args": {"input": "exec(open('main.py').read())"}},
                {"thought": "根据运行结果和代码分析，找出问题。", "final_answer": "分析了 main.py 的代码，发现以下可能的问题：\n{observation}\n\n请告诉我具体的预期结果和实际结果，我可以进一步定位。"},
            ],
        },

        # === 项目管理 ===
        {
            "task": "查看当前项目的 Python 依赖是否都已安装",
            "steps": [
                {"thought": "先读取 requirements.txt。", "action": "read_local_file", "action_args": {"input": "requirements.txt"}},
                {"thought": "用 pip check 检查依赖。", "action": "run_command", "action_args": {"input": "pip check"}},
                {"thought": "检查完成。", "final_answer": "依赖检查结果：\n{observation}"},
            ],
        },
        {
            "task": "创建一个 requirements.txt，包含 requests, flask, pytest",
            "steps": [
                {"thought": "用户想创建 requirements.txt。", "action": "write_file", "action_args": {"input": "requirements.txt | requests>=2.28.0\nflask>=3.0.0\npytest>=7.0.0"}},
                {"thought": "文件创建成功。", "final_answer": "已创建 requirements.txt，包含 requests、flask、pytest 三个依赖。"},
            ],
        },

        # === Git 操作 ===
        {
            "task": "查看当前 git 仓库的最近 5 次提交记录",
            "steps": [
                {"thought": "用户想看 git 提交记录。", "action": "run_command", "action_args": {"input": "git log --oneline -5"}},
                {"thought": "命令执行完成。", "final_answer": "最近 5 次提交：\n{observation}"},
            ],
        },
        {
            "task": "创建一个新的 git 分支叫 feature/new-api",
            "steps": [
                {"thought": "用户想创建 git 分支。", "action": "run_command", "action_args": {"input": "git checkout -b feature/new-api"}},
                {"thought": "分支创建成功。", "final_answer": "已创建并切换到新分支 feature/new-api。"},
            ],
        },

        # === 网络请求 ===
        {
            "task": "用 Python 请求 httpbin.org/get 看看返回什么",
            "steps": [
                {"thought": "用户想测试 HTTP 请求。", "action": "execute_python", "action_args": {"input": "import requests\nresp = requests.get('https://httpbin.org/get', timeout=10)\nprint(resp.json())"}},
                {"thought": "请求成功。", "final_answer": "HTTP 响应：\n{observation}"},
            ],
        },

        # === 文件搜索 ===
        {
            "task": "在当前目录下找到所有包含 'TODO' 的 Python 文件",
            "steps": [
                {"thought": "用 Python 搜索文件内容。", "action": "execute_python", "action_args": {"input": "import os\nimport glob\n\nfor f in glob.glob('**/*.py', recursive=True):\n    try:\n        with open(f, 'r', encoding='utf-8') as fh:\n            for i, line in enumerate(fh, 1):\n                if 'TODO' in line:\n                    print(f'{f}:{i}: {line.strip()}')\n    except Exception:\n        pass"}},
                {"thought": "搜索完成。", "final_answer": "搜索结果：\n{observation}"},
            ],
        },

        # === 正则表达式 ===
        {
            "task": "用正则表达式从文本中提取所有邮箱地址",
            "steps": [
                {"thought": "用 Python 正则表达式提取邮箱。", "action": "execute_python", "action_args": {"input": "import re\n\ntext = '联系方式: alice@example.com, bob@test.org, 无效@, charlie@domain.com'\nemails = re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)\nprint(f'找到 {len(emails)} 个邮箱:')\nfor e in emails:\n    print(f'  {e}')"}},
                {"thought": "提取完成。", "final_answer": "{observation}"},
            ],
        },

        # === JSON 处理 ===
        {
            "task": "读取 config.json，把所有值转成大写后保存",
            "steps": [
                {"thought": "先读取配置文件。", "action": "read_local_file", "action_args": {"input": "config.json"}},
                {"thought": "用 Python 处理 JSON 值。", "action": "execute_python", "action_args": {"input": "import json\n\nwith open('config.json', 'r', encoding='utf-8') as f:\n    data = json.load(f)\n\ndef uppercase_values(obj):\n    if isinstance(obj, dict):\n        return {k: uppercase_values(v) for k, v in obj.items()}\n    elif isinstance(obj, list):\n        return [uppercase_values(i) for i in obj]\n    elif isinstance(obj, str):\n        return obj.upper()\n    return obj\n\nresult = uppercase_values(data)\nwith open('config_upper.json', 'w', encoding='utf-8') as f:\n    json.dump(result, f, ensure_ascii=False, indent=2)\nprint('已保存到 config_upper.json')"}},
                {"thought": "处理完成。", "final_answer": "已将 config.json 中的字符串值全部转为大写，保存到 config_upper.json。"},
            ],
        },

        # === 系统信息 ===
        {
            "task": "查看当前系统的 CPU 和内存使用情况",
            "steps": [
                {"thought": "用 Python 获取系统信息。", "action": "execute_python", "action_args": {"input": "import psutil\nimport platform\n\nprint(f'系统: {platform.system()} {platform.release()}')\nprint(f'CPU 核心数: {psutil.cpu_count()}')\nprint(f'CPU 使用率: {psutil.cpu_percent(interval=1)}%')\nmem = psutil.virtual_memory()\nprint(f'总内存: {mem.total / 1024**3:.1f} GB')\nprint(f'已用: {mem.used / 1024**3:.1f} GB ({mem.percent}%)')"}},
                {"thought": "信息获取成功。", "final_answer": "系统信息：\n{observation}"},
            ],
        },

        # === 文本处理 ===
        {
            "task": "统计 README.md 里每个单词出现的次数，找出最常见的 10 个",
            "steps": [
                {"thought": "先读取 README.md。", "action": "read_local_file", "action_args": {"input": "README.md"}},
                {"thought": "用 Python 统计词频。", "action": "execute_python", "action_args": {"input": "from collections import Counter\nimport re\n\nwith open('README.md', 'r', encoding='utf-8') as f:\n    text = f.read().lower()\n\nwords = re.findall(r'\\b[a-z]+\\b', text)\n# 过滤停用词\nstop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'it', 'this', 'that', 'with'}\nwords = [w for w in words if w not in stop_words and len(w) > 2]\n\ntop10 = Counter(words).most_common(10)\nfor word, count in top10:\n    print(f'  {word}: {count}')"}},
                {"thought": "统计完成。", "final_answer": "README.md 中最常见的 10 个单词：\n{observation}"},
            ],
        },

        # === 定时任务 ===
        {
            "task": "写一个 Python 脚本，每 5 秒检查一次某个 URL 是否可访问",
            "steps": [
                {"thought": "用户想写一个 URL 监控脚本。", "action": "write_file", "action_args": {"input": "url_monitor.py | import requests\nimport time\n\nurl = 'http://localhost:8000'\ninterval = 5\n\nprint(f'开始监控 {url}，每 {interval} 秒检查一次')\nwhile True:\n    try:\n        resp = requests.get(url, timeout=3)\n        print(f'[{time.strftime(\"%H:%M:%S\")}] OK - 状态码: {resp.status_code}')\n    except Exception as e:\n        print(f'[{time.strftime(\"%H:%M:%S\")}] FAIL - {e}')\n    time.sleep(interval)"}},
                {"thought": "脚本创建成功。", "final_answer": "已创建 URL 监控脚本 url_monitor.py。运行方式：`python url_monitor.py`"},
            ],
        },

        # === 数据可视化（文本） ===
        {
            "task": "用 Python 画一个简单的柱状图显示在终端",
            "steps": [
                {"thought": "用 ASCII 字符在终端画柱状图。", "action": "execute_python", "action_args": {"input": "data = {'Python': 85, 'Java': 70, 'C++': 60, 'JavaScript': 75, 'Go': 50, 'Rust': 45}\n\nmax_val = max(data.values())\nfor name, val in data.items():\n    bar = '█' * int(val / max_val * 30)\n    print(f'{name:>12} | {bar} {val}')"}},
                {"thought": "图表生成成功。", "final_answer": "编程语言流行度柱状图：\n{observation}"},
            ],
        },

        # === 文件压缩 ===
        {
            "task": "把 data 目录打包成 zip 文件",
            "steps": [
                {"thought": "用 Python 的 zipfile 模块打包。", "action": "execute_python", "action_args": {"input": "import zipfile\nimport os\n\ndef zip_directory(zipf, directory):\n    for root, dirs, files in os.walk(directory):\n        for file in files:\n            filepath = os.path.join(root, file)\n            arcname = os.path.relpath(filepath, '.')\n            zipf.write(filepath, arcname)\n\nwith zipfile.ZipFile('data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:\n    zip_directory(zipf, 'data')\n\nsize = os.path.getsize('data.zip')\nprint(f'已打包: data.zip ({size / 1024:.1f} KB)')"}},
                {"thought": "打包完成。", "final_answer": "已将 data 目录打包为 data.zip。"},
            ],
        },

        # === 环境变量 ===
        {
            "task": "查看当前系统的 PATH 环境变量",
            "steps": [
                {"thought": "用 Python 读取环境变量。", "action": "execute_python", "action_args": {"input": "import os\npath = os.environ.get('PATH', '')\nfor p in path.split(os.pathsep):\n    print(p)"}},
                {"thought": "获取成功。", "final_answer": "PATH 环境变量：\n{observation}"},
            ],
        },

        # === 代码重构 ===
        {
            "task": "把 utils.py 里重复的 try-except 代码提取成一个装饰器",
            "steps": [
                {"thought": "先读取 utils.py 分析代码。", "action": "read_local_file", "action_args": {"input": "utils.py"}},
                {"thought": "分析重复的错误处理模式，创建装饰器。", "action": "write_file", "action_args": {"input": "error_decorator.py | import functools\nimport logging\n\ndef safe_execute(default=None, log_error=True):\n    \"\"\"装饰器：安全执行函数，捕获异常并返回默认值\"\"\"\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            try:\n                return func(*args, **kwargs)\n            except Exception as e:\n                if log_error:\n                    logging.error(f'{func.__name__} 失败: {e}')\n                return default\n        return wrapper\n    return decorator\n\n# 使用示例\n@safe_execute(default=[])\ndef read_config(path):\n    import json\n    with open(path) as f:\n        return json.load(f)"}},
                {"thought": "装饰器创建成功。", "final_answer": "已创建错误处理装饰器 error_decorator.py。使用 @safe_execute(default=默认值) 可以自动捕获异常。"},
            ],
        },

        # === 数据库模拟 ===
        {
            "task": "用 SQLite 创建一个用户表并插入几条数据",
            "steps": [
                {"thought": "用 Python 的 sqlite3 模块操作数据库。", "action": "execute_python", "action_args": {"input": "import sqlite3\n\nconn = sqlite3.connect('users.db')\nc = conn.cursor()\n\nc.execute('''CREATE TABLE IF NOT EXISTS users\n    (id INTEGER PRIMARY KEY, name TEXT, email TEXT, age INTEGER)''')\n\nusers = [\n    (1, 'Alice', 'alice@example.com', 30),\n    (2, 'Bob', 'bob@example.com', 25),\n    (3, 'Charlie', 'charlie@example.com', 35),\n]\nc.executemany('INSERT OR REPLACE INTO users VALUES (?,?,?,?)', users)\nconn.commit()\n\nc.execute('SELECT * FROM users')\nfor row in c.fetchall():\n    print(row)\nconn.close()"}},
                {"thought": "数据库操作成功。", "final_answer": "已创建 SQLite 数据库 users.db 并插入数据：\n{observation}"},
            ],
        },

        # === API 测试 ===
        {
            "task": "写一个简单的 Flask API 并测试 /hello 接口",
            "steps": [
                {"thought": "创建 Flask API 文件。", "action": "write_file", "action_args": {"input": "test_api.py | from flask import Flask, jsonify\napp = Flask(__name__)\n\n@app.route('/hello')\ndef hello():\n    return jsonify({'message': 'Hello, World!'})\n\n@app.route('/add/<int:a>/<int:b>')\ndef add(a, b):\n    return jsonify({'result': a + b})\n\nif __name__ == '__main__':\n    app.run(port=5000)"}},
                {"thought": "用 Python 测试 API 逻辑（不启动服务器）。", "action": "execute_python", "action_args": {"input": "from flask import Flask\napp = Flask(__name__)\n\n@app.route('/hello')\ndef hello():\n    return '{\"message\": \"Hello, World!\"}'\n\nwith app.test_client() as client:\n    resp = client.get('/hello')\n    print(f'状态码: {resp.status_code}')\n    print(f'响应: {resp.data.decode()}')"}},
                {"thought": "测试成功。", "final_answer": "已创建 Flask API 并测试：\n{observation}"},
            ],
        },

        # === 性能测试 ===
        {
            "task": "比较列表推导式和 for 循环的性能差异",
            "steps": [
                {"thought": "用 Python 的 timeit 模块比较两种写法的性能。", "action": "execute_python", "action_args": {"input": "import timeit\n\n# 列表推导式\ndef list_comp():\n    return [x**2 for x in range(10000)]\n\n# for 循环\ndef for_loop():\n    result = []\n    for x in range(10000):\n        result.append(x**2)\n    return result\n\nt1 = timeit.timeit(list_comp, number=1000)\nt2 = timeit.timeit(for_loop, number=1000)\n\nprint(f'列表推导式: {t1:.4f}s')\nprint(f'for 循环: {t2:.4f}s')\nprint(f'推导式快 {t2/t1:.1f}x')"}},
                {"thought": "性能测试完成。", "final_answer": "性能对比结果：\n{observation}"},
            ],
        },

        # === 多步文件操作 ===
        {
            "task": "创建一个备份目录，把所有 .json 文件复制过去，然后列出备份内容",
            "steps": [
                {"thought": "先创建备份目录。", "action": "execute_python", "action_args": {"input": "import os\nimport shutil\nimport glob\n\nos.makedirs('backup', exist_ok=True)\n\nfor f in glob.glob('*.json'):\n    shutil.copy2(f, 'backup/')\n    print(f'已备份: {f}')\n\nprint(f'\\n备份目录内容:')\nfor f in os.listdir('backup'):\n    size = os.path.getsize(os.path.join('backup', f))\n    print(f'  {f} ({size} bytes)')"}},
                {"thought": "备份完成。", "final_answer": "已创建 backup 目录并备份所有 JSON 文件：\n{observation}"},
            ],
        },
    ]


def get_extended_conversation_data() -> List[Dict[str, Any]]:
    """扩展对话训练数据：覆盖更多知识领域和对话场景"""
    return [
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "Python 的 GIL 是什么？"},
                {"role": "assistant", "content": "GIL（Global Interpreter Lock，全局解释器锁）是 CPython 解释器中的一个机制。\n\n**是什么：**\nGIL 是一个互斥锁，确保同一时刻只有一个线程执行 Python 字节码。\n\n**为什么存在：**\n- 保护 Python 对象的引用计数\n- 简化 CPython 的内存管理\n- 防止多线程同时修改对象\n\n**影响：**\n- CPU 密集型任务：多线程无法利用多核（用 multiprocessing 解决）\n- I/O 密集型任务：影响不大（I/O 时会释放 GIL）\n\n**解决方案：**\n1. `multiprocessing` — 多进程绕过 GIL\n2. `asyncio` — 异步 I/O\n3. `C 扩展` — NumPy 等库在 C 层释放 GIL\n4. `subprocess` — 外部进程\n5. Python 3.13+ 正在实验性移除 GIL"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 RESTful API？"},
                {"role": "assistant", "content": "RESTful API 是一种基于 REST（Representational State Transfer）架构风格的 API 设计规范。\n\n**核心原则：**\n1. **无状态** — 每个请求包含所有必要信息\n2. **统一接口** — 使用标准 HTTP 方法\n3. **资源导向** — URL 表示资源，HTTP 方法表示操作\n\n**HTTP 方法对应操作：**\n- `GET /users` — 获取用户列表\n- `GET /users/1` — 获取单个用户\n- `POST /users` — 创建用户\n- `PUT /users/1` — 更新用户（全量）\n- `PATCH /users/1` — 更新用户（部分）\n- `DELETE /users/1` — 删除用户\n\n**最佳实践：**\n- URL 用名词复数：`/users` 而非 `/getUser`\n- 状态码语义化：200 成功、404 未找到、500 服务器错误\n- 版本化：`/api/v1/users`\n- 分页：`/users?page=1&limit=10`"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "解释一下 TCP 和 UDP 的区别"},
                {"role": "assistant", "content": "TCP 和 UDP 是两种主要的传输层协议：\n\n| 特性 | TCP | UDP |\n|------|-----|-----|\n| 连接 | 面向连接（三次握手） | 无连接 |\n| 可靠性 | 可靠（确认、重传） | 不可靠 |\n| 顺序 | 保证顺序 | 不保证 |\n| 速度 | 较慢 | 较快 |\n| 开销 | 较大 | 较小 |\n| 适用 | 文件传输、网页、邮件 | 视频、游戏、DNS |\n\n**TCP 三次握手：**\n1. 客户端 → SYN → 服务器\n2. 服务器 → SYN+ACK → 客户端\n3. 客户端 → ACK → 服务器\n\n**选择建议：**\n- 数据完整性重要 → TCP（HTTP、FTP、SMTP）\n- 实时性重要、丢包可接受 → UDP（视频直播、在线游戏、DNS）"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 SQL 注入？如何防御？"},
                {"role": "assistant", "content": "SQL 注入是一种代码注入攻击，攻击者通过在输入中插入恶意 SQL 语句来操纵数据库。\n\n**攻击示例：**\n```sql\n-- 正常查询\nSELECT * FROM users WHERE name = 'Alice'\n\n-- 注入攻击\n输入: ' OR '1'='1\nSELECT * FROM users WHERE name = '' OR '1'='1'  -- 返回所有用户！\n```\n\n**防御方法：**\n\n1. **参数化查询（最有效）**\n```python\n# 安全\ncursor.execute(\"SELECT * FROM users WHERE name = ?\", (user_input,))\n\n# 危险\ncursor.execute(f\"SELECT * FROM users WHERE name = '{user_input}'\")\n```\n\n2. **ORM 框架** — SQLAlchemy、Django ORM 自动防注入\n3. **输入验证** — 白名单过滤特殊字符\n4. **最小权限** — 数据库用户只给必要权限\n5. **WAF** — Web 应用防火墙\n6. **转义** — 对特殊字符转义（最后手段）"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "解释一下什么是设计模式中的单例模式"},
                {"role": "assistant", "content": "单例模式（Singleton Pattern）确保一个类只有一个实例，并提供全局访问点。\n\n**Python 实现：**\n\n```python\n# 方法1：使用 __new__\nclass Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n\n# 方法2：使用装饰器\ndef singleton(cls):\n    instances = {}\n    def get_instance(*args, **kwargs):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kwargs)\n        return instances[cls]\n    return get_instance\n\n@singleton\nclass Database:\n    pass\n\n# 方法3：使用模块（Python 推荐）\n# config.py\n# class Config:\n#     ...\n# config = Config()  # 模块级实例\n```\n\n**适用场景：**\n- 数据库连接池\n- 配置管理器\n- 日志记录器\n- 线程池\n\n**注意事项：**\n- 多线程环境需要加锁\n- 不利于单元测试\n- Python 中用模块级变量通常更简单"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是容器编排？Kubernetes 是什么？"},
                {"role": "assistant", "content": "容器编排是自动化管理容器化应用的部署、扩缩容和运维的过程。\n\n**Kubernetes（K8s）** 是最流行的容器编排平台：\n\n**核心概念：**\n- **Pod** — 最小部署单元，包含一个或多个容器\n- **Service** — 为 Pod 提供稳定的网络访问\n- **Deployment** — 管理 Pod 的副本数和更新策略\n- **Namespace** — 资源隔离\n\n**核心功能：**\n1. **自动部署和回滚**\n2. **自动扩缩容** — 根据 CPU/内存自动调整\n3. **服务发现和负载均衡**\n4. **自动恢复** — 容器崩溃自动重启\n5. **配置管理** — ConfigMap 和 Secret\n\n**常用命令：**\n```bash\nkubectl get pods              # 查看 Pod\nkubectl apply -f app.yaml     # 部署应用\nkubectl scale deploy/app --replicas=3  # 扩容\nkubectl rollout undo deploy/app  # 回滚\n```\n\n**vs Docker Compose：**\n- Docker Compose：单机、简单\n- Kubernetes：多机、生产级、功能强大"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "Python 的生成器和迭代器有什么区别？"},
                {"role": "assistant", "content": "迭代器和生成器都是用于遍历数据的工具，但实现方式不同：\n\n**迭代器（Iterator）：**\n```python\nclass MyIterator:\n    def __init__(self, max_val):\n        self.max = max_val\n        self.current = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.current >= self.max:\n            raise StopIteration\n        self.current += 1\n        return self.current\n```\n\n**生成器（Generator）：**\n```python\ndef my_generator(max_val):\n    current = 0\n    while current < max_val:\n        current += 1\n        yield current\n```\n\n**区别：**\n| 特性 | 迭代器 | 生成器 |\n|------|--------|--------|\n| 实现 | `__iter__` + `__next__` | `yield` 关键字 |\n| 代码量 | 较多 | 极少 |\n| 内存 | 需要存储状态 | 惰性求值，省内元 |\n| 用途 | 自定义复杂遍历 | 简单序列生成 |\n\n**生成器优势：**\n- 内存高效（不一次性生成所有值）\n- 代码简洁\n- 支持无限序列\n\n**常见用法：**\n```python\n# 读取大文件\ndef read_large_file(path):\n    with open(path) as f:\n        for line in f:\n            yield line.strip()\n```"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是微服务架构？"},
                {"role": "assistant", "content": "微服务架构是将应用拆分为一组小型、独立服务的架构风格。\n\n**核心特点：**\n- 每个服务独立部署和扩展\n- 服务间通过 API 通信（通常 REST 或 gRPC）\n- 每个服务有自己的数据库\n- 团队可以独立开发和发布\n\n**vs 单体架构：**\n| 特性 | 单体 | 微服务 |\n|------|------|--------|\n| 部署 | 整体部署 | 独立部署 |\n| 扩展 | 整体扩展 | 按需扩展 |\n| 技术栈 | 统一 | 可以不同 |\n| 复杂度 | 简单 | 高（分布式） |\n| 团队 | 大团队协调 | 小团队自治 |\n\n**常见组件：**\n- API Gateway — 统一入口\n- 服务注册/发现 — Eureka、Consul\n- 配置中心 — Nacos、Apollo\n- 链路追踪 — Jaeger、Zipkin\n- 消息队列 — Kafka、RabbitMQ\n\n**适用场景：**\n- 大型团队（10+ 开发者）\n- 高并发、需要独立扩展\n- 需要快速迭代\n\n**不适用：**\n- 小项目（增加复杂度无意义）\n- 团队经验不足"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "如何优化 Python 代码的性能？"},
                {"role": "assistant", "content": "Python 性能优化的常用方法：\n\n**1. 数据结构选择**\n- 查找用 `set`/`dict`（O(1)）而非 `list`（O(n)）\n- 频繁插入用 `deque` 而非 `list`\n- 不可变数据用 `tuple` 而非 `list`\n\n**2. 算法优化**\n- 避免嵌套循环（O(n²) → O(n log n)）\n- 用空间换时间（缓存、哈希表）\n- 使用 `functools.lru_cache` 缓存函数结果\n\n**3. 内置函数和库**\n- 用 `map()`、`filter()` 替代循环\n- 用 `join()` 替代字符串拼接\n- 用 NumPy 替代纯 Python 数值计算\n\n**4. 并发处理**\n- I/O 密集：`asyncio` 或 `threading`\n- CPU 密集：`multiprocessing`\n\n**5. 代码级优化**\n```python\n# 慢\nresult = []\nfor x in data:\n    result.append(x * 2)\n\n# 快\nresult = [x * 2 for x in data]\n\n# 更快（数值计算）\nimport numpy as np\nresult = np.array(data) * 2\n```\n\n**6. 工具**\n- `cProfile` — 函数级性能分析\n- `line_profiler` — 逐行分析\n- `memory_profiler` — 内存分析\n- `PyPy` — 替代解释器（JIT 编译）"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 WebSocket？和 HTTP 有什么区别？"},
                {"role": "assistant", "content": "WebSocket 是一种全双工通信协议，允许服务器和客户端实时双向通信。\n\n**vs HTTP：**\n| 特性 | HTTP | WebSocket |\n|------|------|----------|\n| 通信方向 | 单向（请求-响应） | 双向 |\n| 连接 | 短连接（每次新建） | 长连接（持久） |\n| 开销 | 每次带完整头部 | 仅首次握手开销大 |\n| 实时性 | 轮询/SSE | 原生实时 |\n| 适用 | 网页、API | 聊天、游戏、实时数据 |\n\n**握手过程：**\n```\n客户端 → HTTP Upgrade: websocket → 服务器\n服务器 → 101 Switching Protocols → 客户端\n→ 双向 WebSocket 通信开始\n```\n\n**Python 示例：**\n```python\nimport asyncio\nimport websockets\n\nasync def handler(websocket):\n    async for message in websocket:\n        await websocket.send(f'Echo: {message}')\n\nasyncio.run(websockets.serve(handler, 'localhost', 8765))\n```\n\n**适用场景：**\n- 聊天应用\n- 实时股票行情\n- 在线游戏\n- 协作编辑\n- 通知推送"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 OAuth 2.0？"},
                {"role": "assistant", "content": "OAuth 2.0 是一种授权框架，允许第三方应用在用户授权下访问其资源，而无需暴露密码。\n\n**核心角色：**\n- **资源所有者** — 用户\n- **客户端** — 第三方应用\n- **授权服务器** — 颁发令牌\n- **资源服务器** — 存储用户数据\n\n**授权流程（授权码模式）：**\n1. 用户点击「用微信登录」\n2. 跳转到微信授权页面\n3. 用户确认授权\n4. 微信返回授权码给应用\n5. 应用用授权码换取 access_token\n6. 用 access_token 访问用户信息\n\n**常见模式：**\n- **授权码模式** — 最安全，适合 Web 应用\n- **隐式模式** — 适合前端 SPA（已不推荐）\n- **密码模式** — 信任的应用（已不推荐）\n- **客户端凭证** — 服务间通信\n\n**Token 类型：**\n- `access_token` — 短期有效（几小时）\n- `refresh_token` — 长期有效，用于刷新 access_token\n\n**vs JWT：**\n- OAuth 2.0 是授权框架\n- JWT 是令牌格式\n- OAuth 2.0 可以使用 JWT 作为 token 格式"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 Redis？主要用来做什么？"},
                {"role": "assistant", "content": "Redis 是一个开源的内存数据结构存储系统，可用作数据库、缓存和消息中间件。\n\n**核心特点：**\n- 基于内存，读写极快（10万+ QPS）\n- 支持多种数据结构\n- 支持持久化（RDB + AOF）\n- 支持主从复制和集群\n\n**数据结构：**\n- **String** — 缓存、计数器、分布式锁\n- **Hash** — 对象存储（用户信息）\n- **List** — 消息队列、最新列表\n- **Set** — 去重、交集/并集\n- **Sorted Set** — 排行榜、延迟队列\n- **Bitmap** — 用户签到、在线状态\n\n**常见用途：**\n1. **缓存** — 数据库查询结果缓存\n2. **会话存储** — 用户登录状态\n3. **分布式锁** — `SETNX`\n4. **消息队列** — Pub/Sub 或 Stream\n5. **排行榜** — Sorted Set\n6. **限流** — 滑动窗口\n\n**vs Memcached：**\n- Redis 数据结构更丰富\n- Redis 支持持久化\n- Redis 支持 Pub/Sub\n- Memcached 更简单，纯缓存场景更高效"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是 CI/CD？"},
                {"role": "assistant", "content": "CI/CD 是现代软件开发的核心实践：\n\n**CI（持续集成）：**\n开发者频繁（每天多次）将代码合并到主分支，每次合并自动触发构建和测试。\n\n**CD（持续交付/部署）：**\n- **持续交付**：代码随时可以部署到生产环境（需手动确认）\n- **持续部署**：代码自动部署到生产环境\n\n**典型流程：**\n```\n代码提交 → 自动构建 → 单元测试 → 代码检查 → 部署到测试环境 → 集成测试 → 部署到生产\n```\n\n**常用工具：**\n- **GitHub Actions** — GitHub 原生\n- **GitLab CI** — GitLab 原生\n- **Jenkins** — 老牌，插件丰富\n- **CircleCI** — 云端，配置简单\n\n**GitHub Actions 示例：**\n```yaml\nname: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3\n      - uses: actions/setup-python@v4\n        with:\n          python-version: '3.11'\n      - run: pip install -r requirements.txt\n      - run: pytest\n```\n\n**好处：**\n- 尽早发现问题\n- 减少手动操作\n- 提高发布频率和质量"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "Python 里的深拷贝和浅拷贝有什么区别？"},
                {"role": "assistant", "content": "深拷贝和浅拷贝的区别在于对嵌套对象的处理：\n\n**浅拷贝（Shallow Copy）：**\n只复制对象本身，不复制内部引用的对象。\n```python\nimport copy\n\na = [[1, 2], [3, 4]]\nb = copy.copy(a)\n\nb[0][0] = 99\nprint(a)  # [[99, 2], [3, 4]] — a 也被改了！\n```\n\n**深拷贝（Deep Copy）：**\n递归复制所有嵌套对象。\n```python\na = [[1, 2], [3, 4]]\nb = copy.deepcopy(a)\n\nb[0][0] = 99\nprint(a)  # [[1, 2], [3, 4]] — a 不受影响\n```\n\n**创建方式：**\n| 方式 | 类型 |\n|------|------|\n| `copy.copy(x)` | 浅拷贝 |\n| `copy.deepcopy(x)` | 深拷贝 |\n| `list(x)` / `x[:]` | 浅拷贝 |\n| `dict(x)` / `x.copy()` | 浅拷贝 |\n\n**选择建议：**\n- 对象只有基本类型 → 浅拷贝够用\n- 对象包含嵌套的可变类型 → 用深拷贝\n- 性能敏感 → 浅拷贝更快"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是消息队列？有什么用？"},
                {"role": "assistant", "content": "消息队列（Message Queue）是一种异步通信机制，用于在应用之间传递消息。\n\n**核心概念：**\n- **生产者** — 发送消息\n- **消费者** — 接收消息\n- **队列** — 存储消息的缓冲区\n\n**解决的问题：**\n1. **异步处理** — 用户下单后异步发送邮件\n2. **流量削峰** — 秒杀场景缓冲请求\n3. **应用解耦** — 订单系统和库存系统独立\n4. **可靠通信** — 消息持久化，不丢失\n\n**常见消息队列：**\n| 特性 | RabbitMQ | Kafka | RocketMQ |\n|------|----------|-------|----------|\n| 模型 | 队列 | 发布-订阅 | 队列+发布订阅 |\n| 吞吐 | 万级 | 百万级 | 十万级 |\n| 延迟 | 微秒 | 毫秒 | 毫秒 |\n| 适用 | 业务消息 | 日志/大数据 | 电商/金融 |\n\n**Python 示例（RabbitMQ）：**\n```python\nimport pika\n\n# 生产者\nconn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))\nchannel = conn.channel()\nchannel.queue_declare(queue='task_queue')\nchannel.basic_publish(exchange='', routing_key='task_queue', body=b'Hello')\n```"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "如何保护 API 的安全？"},
                {"role": "assistant", "content": "API 安全的最佳实践：\n\n**1. 身份认证**\n- API Key — 简单场景\n- OAuth 2.0 — 第三方授权\n- JWT — 无状态令牌\n- mTLS — 双向证书认证\n\n**2. 授权**\n- RBAC（基于角色）\n- ABAC（基于属性）\n- Scope（OAuth 权限范围）\n\n**3. 传输安全**\n- 强制 HTTPS\n- TLS 1.2+\n- 证书固定\n\n**4. 输入验证**\n- 参数类型检查\n- 长度限制\n- 白名单验证\n- 防 SQL 注入/XSS\n\n**5. 限流和防滥用**\n- 速率限制（Rate Limiting）\n- IP 黑名单\n- 请求大小限制\n\n**6. 日志和监控**\n- 记录所有 API 调用\n- 异常行为告警\n- 审计日志\n\n**7. 其他**\n- CORS 配置\n- 敏感数据脱敏\n- 版本管理\n- 错误信息不暴露内部细节"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是正则表达式？举几个常用例子"},
                {"role": "assistant", "content": "正则表达式（Regular Expression）是用于匹配字符串模式的工具。\n\n**基础语法：**\n| 符号 | 含义 | 示例 |\n|------|------|------|\n| `.` | 任意字符 | `a.c` 匹配 abc、a1c |\n| `*` | 0 次或多次 | `ab*c` 匹配 ac、abc、abbc |\n| `+` | 1 次或多次 | `ab+c` 匹配 abc、abbc |\n| `?` | 0 次或 1 次 | `ab?c` 匹配 ac、abc |\n| `\\d` | 数字 | `\\d+` 匹配 123 |\n| `\\w` | 字母/数字/下划线 | `\\w+` 匹配 hello |\n| `\\s` | 空白字符 | `\\s+` 匹配空格、制表符 |\n| `[]` | 字符集 | `[abc]` 匹配 a、b、c |\n| `^` | 开头 | `^Hello` 匹配以 Hello 开头 |\n| `$` | 结尾 | `end$` 匹配以 end 结尾 |\n\n**常用例子：**\n```python\nimport re\n\n# 邮箱\nre.match(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', 'test@example.com')\n\n# 手机号（中国）\nre.match(r'1[3-9]\\d{9}', '13812345678')\n\n# IP 地址\nre.match(r'(\\d{1,3}\\.){3}\\d{1,3}', '192.168.1.1')\n\n# 提取数字\nre.findall(r'\\d+', '价格是100元，数量是5个')  # ['100', '5']\n\n# 替换\nre.sub(r'\\s+', ' ', 'hello   world')  # 'hello world'\n```"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是协程？和线程有什么区别？"},
                {"role": "assistant", "content": "协程（Coroutine）是一种可以暂停和恢复的函数，是异步编程的核心。\n\n**vs 线程：**\n| 特性 | 线程 | 协程 |\n|------|------|------|\n| 调度 | OS 内核调度 | 程序主动让出 |\n| 切换开销 | 大（上下文切换） | 极小（函数切换） |\n| 数量 | 百级 | 万级 |\n| 并发 | 真并行（多核） | 协作式（单线程） |\n| 同步 | 需要锁 | 无需锁 |\n| 适用 | CPU 密集 | I/O 密集 |\n\n**Python 协程：**\n```python\nimport asyncio\n\nasync def fetch_data(url):\n    print(f'开始请求 {url}')\n    await asyncio.sleep(1)  # 模拟 I/O\n    print(f'请求完成 {url}')\n    return f'{url} 的数据'\n\nasync def main():\n    # 并发执行\n    tasks = [\n        fetch_data('url1'),\n        fetch_data('url2'),\n        fetch_data('url3'),\n    ]\n    results = await asyncio.gather(*tasks)\n    print(results)\n\nasyncio.run(main())\n```\n\n**`await` 的含义：**\n- 暂停当前协程\n- 让出控制权给事件循环\n- 等待异步操作完成后恢复"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": "什么是索引？数据库索引是怎么工作的？"},
                {"role": "assistant", "content": "数据库索引是一种数据结构，用于加速数据查询，类似于书的目录。\n\n**工作原理：**\n没有索引：全表扫描（逐行检查）— O(n)\n有索引：通过索引结构直接定位 — O(log n)\n\n**常见索引类型：**\n1. **B+ 树索引** — 最常用，适合范围查询\n2. **哈希索引** — 等值查询极快，不支持范围\n3. **全文索引** — 文本搜索\n4. **空间索引** — 地理位置\n\n**创建索引：**\n```sql\n-- 单列索引\nCREATE INDEX idx_name ON users(name);\n\n-- 复合索引\nCREATE INDEX idx_name_age ON users(name, age);\n\n-- 唯一索引\nCREATE UNIQUE INDEX idx_email ON users(email);\n```\n\n**索引的代价：**\n- 占用存储空间\n- 降低写入速度（每次 INSERT/UPDATE 都要更新索引）\n- 过多索引反而降低性能\n\n**何时创建索引：**\n- WHERE 条件频繁使用的列\n- JOIN 连接的列\n- ORDER BY 排序的列\n- 高选择性的列（值种类多）\n\n**何时不创建：**\n- 数据量小的表\n- 频繁写入的列\n- 选择性低的列（如性别只有男/女）"},
            ],
        },
    ]


def get_edge_case_react_data() -> List[Dict[str, Any]]:
    """工具错误处理和边界情况数据"""
    return [
        # === read_local_file 边界 ===
        {"task": "读取一个空文件 empty.txt", "steps": [
            {"thought": "文件可能是空的，先尝试读取。", "action": "read_local_file", "action_args": {"input": "empty.txt"}},
            {"thought": "文件存在但内容为空。", "final_answer": "empty.txt 文件存在但内容为空。"},
        ]},
        {"task": "读取一个包含中文的文件 notes_cn.txt", "steps": [
            {"thought": "读取中文文件。", "action": "read_local_file", "action_args": {"input": "notes_cn.txt"}},
            {"thought": "读取成功。", "final_answer": "文件内容：\n{observation}"},
        ]},
        {"task": "读取 ../../../etc/passwd", "steps": [
            {"thought": "用户请求读取系统敏感文件，这不安全，应该拒绝。", "final_answer": "抱歉，出于安全考虑，我无法读取系统敏感文件。请提供项目内的文件路径。"},
        ]},
        {"task": "读取一个 100MB 的大文件 big_data.csv", "steps": [
            {"thought": "文件可能很大，先读取前几行看看。", "action": "execute_python", "action_args": {"input": "with open('big_data.csv', 'r', encoding='utf-8') as f:\n    for i, line in enumerate(f):\n        if i >= 10:\n            break\n        print(line.strip())\n    print(f'\\n（仅显示前 10 行，避免内存溢出）')"}},
            {"thought": "显示了前几行。", "final_answer": "由于文件较大，仅显示前 10 行：\n{observation}"},
        ]},

        # === write_file 边界 ===
        {"task": "创建一个文件路径包含空格的文件 my file.txt", "steps": [
            {"thought": "路径包含空格，直接写入。", "action": "write_file", "action_args": {"input": "my file.txt | 这是一个路径包含空格的文件"}},
            {"thought": "创建成功。", "final_answer": "已创建 my file.txt 文件。"},
        ]},
        {"task": "在很深的目录 a/b/c/d/e/f/ 下创建 test.txt", "steps": [
            {"thought": "多层嵌套目录，write_file 应该能自动创建。", "action": "write_file", "action_args": {"input": "a/b/c/d/e/f/test.txt | 深层目录测试"}},
            {"thought": "创建成功。", "final_answer": "已在 a/b/c/d/e/f/ 目录下创建 test.txt。"},
        ]},
        {"task": "写入一个包含特殊字符的文件，内容有换行和制表符", "steps": [
            {"thought": "内容包含特殊字符，注意转义。", "action": "write_file", "action_args": {"input": "special.txt | 第一行\n第二行\n\t缩进内容\n特殊字符: @#$%^&*()"}},
            {"thought": "写入成功。", "final_answer": "已创建包含特殊字符的文件。"},
        ]},

        # === edit_file 边界 ===
        {"task": "把 config.json 里所有的 false 改成 true", "steps": [
            {"thought": "先读取文件内容。", "action": "read_local_file", "action_args": {"input": "config.json"}},
            {"thought": "用 Python 批量替换更可靠。", "action": "execute_python", "action_args": {"input": "with open('config.json', 'r', encoding='utf-8') as f:\n    content = f.read()\nnew_content = content.replace('false', 'true')\nwith open('config.json', 'w', encoding='utf-8') as f:\n    f.write(new_content)\nprint('已将所有 false 替换为 true')"}},
            {"thought": "替换完成。", "final_answer": "已将 config.json 中所有的 false 替换为 true。"},
        ]},
        {"task": "编辑 test.txt 但文件不存在", "steps": [
            {"thought": "文件可能不存在，先检查。", "action": "edit_file", "action_args": {"input": "test.txt | old | new"}},
            {"thought": "编辑失败，文件不存在。告知用户。", "final_answer": "编辑失败：test.txt 文件不存在。请先创建文件。"},
        ]},
        {"task": "把 README.md 里的所有 TODO 替换为 DONE", "steps": [
            {"thought": "批量替换，用 Python 更高效。", "action": "execute_python", "action_args": {"input": "with open('README.md', 'r', encoding='utf-8') as f:\n    content = f.read()\ncount = content.count('TODO')\ncontent = content.replace('TODO', 'DONE')\nwith open('README.md', 'w', encoding='utf-8') as f:\n    f.write(content)\nprint(f'已替换 {count} 处 TODO 为 DONE')"}},
            {"thought": "替换完成。", "final_answer": "已将 README.md 中所有 TODO 替换为 DONE。"},
        ]},

        # === execute_python 边界 ===
        {"task": "运行一段会报错的代码 1/0", "steps": [
            {"thought": "这段代码会除零错误，运行看看错误信息。", "action": "execute_python", "action_args": {"input": "print(1/0)"}},
            {"thought": "报错了，告诉用户原因。", "final_answer": "代码运行出错：ZeroDivisionError: division by zero。不能除以零。"},
        ]},
        {"task": "运行一个需要外部库的脚本 import nonexistent_module", "steps": [
            {"thought": "模块不存在会报错。", "action": "execute_python", "action_args": {"input": "import nonexistent_module"}},
            {"thought": "导入失败。", "final_answer": "导入失败：ModuleNotFoundError: No module named 'nonexistent_module'。需要先安装该模块。"},
        ]},
        {"task": "运行一段死循环代码 while True: pass，但要限制时间", "steps": [
            {"thought": "死循环代码，沙盒有超时限制会自动终止。", "action": "execute_python", "action_args": {"input": "import time\nstart = time.time()\nwhile time.time() - start < 0.1:\n    pass\nprint('循环已安全终止')"}},
            {"thought": "沙盒超时保护生效。", "final_answer": "沙盒的超时保护机制会自动终止长时间运行的代码，防止死循环。"},
        ]},
        {"task": "用 Python 读取当前目录下所有文件名并排序", "steps": [
            {"thought": "用 os.listdir 获取文件名。", "action": "execute_python", "action_args": {"input": "import os\nfiles = sorted(os.listdir('.'))\nfor f in files:\n    print(f)"}},
            {"thought": "获取成功。", "final_answer": "当前目录下的文件（按字母排序）：\n{observation}"},
        ]},
        {"task": "用 Python 计算当前日期是今年的第几天", "steps": [
            {"thought": "用 datetime 计算。", "action": "execute_python", "action_args": {"input": "from datetime import datetime\ntoday = datetime.now()\nday_of_year = today.timetuple().tm_yday\nprint(f'{today.strftime(\"%Y-%m-%d\")} 是今年的第 {day_of_year} 天')"}},
            {"thought": "计算成功。", "final_answer": "{observation}"},
        ]},

        # === search 边界 ===
        {"task": "搜索一个很长的关键词（超过 200 字符）", "steps": [
            {"thought": "关键词太长，应该截断或简化。", "action": "search", "action_args": {"input": "Python programming language best practices for writing clean maintainable code with proper error handling and documentation standards guidelines recommendations tips and tricks advanced techniques patterns"}},
            {"thought": "搜索完成。", "final_answer": "搜索结果：\n{observation}\n\n提示：搜索关键词过长可能影响结果质量，建议精简到核心关键词。"},
        ]},
        {"task": "搜索一个不存在的词 asdfghjklqwerty", "steps": [
            {"thought": "搜索一个无意义的词。", "action": "search", "action_args": {"input": "asdfghjklqwerty"}},
            {"thought": "可能没有结果。", "final_answer": "搜索 '{observation}' 未找到相关结果。请尝试更具体的关键词。"},
        ]},
        {"task": "搜索今天的新闻", "steps": [
            {"thought": "搜索今日新闻。", "action": "search", "action_args": {"input": "今日新闻 热点"}},
            {"thought": "搜索完成。", "final_answer": "今日新闻热点：\n{observation}"},
        ]},

        # === run_command 边界 ===
        {"task": "运行一个需要管理员权限的命令", "steps": [
            {"thought": "某些命令需要管理员权限，在普通用户下会失败。", "action": "run_command", "action_args": {"input": "net user"}},
            {"thought": "根据结果判断。", "final_answer": "命令执行结果：\n{observation}\n\n注意：某些操作可能需要管理员权限才能执行。"},
        ]},
        {"task": "查看当前目录的磁盘空间使用情况", "steps": [
            {"thought": "用 dir 命令查看。", "action": "run_command", "action_args": {"input": "dir"}},
            {"thought": "查看完成。", "final_answer": "当前目录内容：\n{observation}"},
        ]},
        {"task": "查看环境变量 PATH 的值", "steps": [
            {"thought": "用 echo 查看环境变量。", "action": "run_command", "action_args": {"input": "echo %PATH%"}},
            {"thought": "获取成功。", "final_answer": "PATH 环境变量：\n{observation}"},
        ]},

        # === analyze_code 边界 ===
        {"task": "分析一个不存在的文件 missing.py", "steps": [
            {"thought": "文件不存在，分析会失败。", "action": "analyze_code", "action_args": {"input": "missing.py"}},
            {"thought": "分析失败，文件不存在。", "final_answer": "分析失败：missing.py 文件不存在。请检查文件路径。"},
        ]},
        {"task": "分析一个非 Python 文件 data.json", "steps": [
            {"thought": "JSON 文件不是 Python 代码，分析工具可能不支持。", "action": "analyze_code", "action_args": {"input": "data.json"}},
            {"thought": "根据结果判断。", "final_answer": "代码分析结果：\n{observation}\n\n注意：analyze_code 主要用于 Python 文件，其他格式的分析可能有限。"},
        ]},

        # === install_dependency 边界 ===
        {"task": "安装一个指定版本的包 requests==2.28.0", "steps": [
            {"thought": "安装指定版本。", "action": "install_dependency", "action_args": {"input": "requests==2.28.0"}},
            {"thought": "安装完成。", "final_answer": "requests 2.28.0 安装完成。"},
        ]},
        {"task": "安装一个已经安装过的包 requests", "steps": [
            {"thought": "包可能已经安装了。", "action": "install_dependency", "action_args": {"input": "requests"}},
            {"thought": "根据输出判断。", "final_answer": "requests 已是最新版本，无需重复安装。"},
        ]},
        {"task": "安装多个包 numpy pandas matplotlib", "steps": [
            {"thought": "批量安装多个包。", "action": "install_dependency", "action_args": {"input": "numpy pandas matplotlib"}},
            {"thought": "安装完成。", "final_answer": "numpy、pandas、matplotlib 安装完成。"},
        ]},

        # === learn_knowledge / query_knowledge 边界 ===
        {"task": "学习一段关于 Docker 的知识，然后查询", "steps": [
            {"thought": "先学习 Docker 知识。", "action": "learn_knowledge", "action_args": {"input": "Docker 是容器化平台，核心概念包括镜像(Image)、容器(Container)、Dockerfile。常用命令：docker build, docker run, docker ps。"}},
            {"thought": "知识已保存，现在查询。", "action": "query_knowledge", "action_args": {"input": "Docker"}},
            {"thought": "查询成功。", "final_answer": "已学习并查询到 Docker 相关知识：\n{observation}"},
        ]},
        {"task": "查询一个没有学过的知识 量子计算", "steps": [
            {"thought": "查询未学过的知识。", "action": "query_knowledge", "action_args": {"input": "量子计算"}},
            {"thought": "知识库中可能没有。", "final_answer": "知识库中暂未找到关于量子计算的记录。你可以让我先学习相关内容，或者搜索互联网获取信息。"},
        ]},

        # === create_project 边界 ===
        {"task": "创建一个已存在的项目 calculator", "steps": [
            {"thought": "项目可能已存在。", "action": "create_project", "action_args": {"input": "python-script | calculator"}},
            {"thought": "根据结果判断。", "final_answer": "项目 'calculator' 的目录已存在。是否要覆盖？或者可以选择一个不同的项目名称。"},
        ]},
        {"task": "创建一个 React 前端项目", "steps": [
            {"thought": "使用 web-app 模板创建。", "action": "create_project", "action_args": {"input": "web-app | my-react-app"}},
            {"thought": "创建成功。", "final_answer": "已创建 Web 项目 'my-react-app'。"},
        ]},

        # === read_webpage 边界 ===
        {"task": "读取一个不存在的网页 https://thisdoesnotexist12345.com", "steps": [
            {"thought": "网页可能不存在。", "action": "read_webpage", "action_args": {"input": "https://thisdoesnotexist12345.com"}},
            {"thought": "请求失败。", "final_answer": "无法访问该网页，请检查 URL 是否正确。"},
        ]},
        {"task": "读取 GitHub Trending 页面", "steps": [
            {"thought": "读取 GitHub 热门项目。", "action": "read_webpage", "action_args": {"input": "https://github.com/trending"}},
            {"thought": "读取成功。", "final_answer": "GitHub Trending 热门项目：\n{observation}"},
        ]},

        # === 组合工具边界 ===
        {"task": "搜索 Python 的 heapq 模块用法，写一个优先队列示例，运行测试", "steps": [
            {"thought": "先搜索 heapq 用法。", "action": "search", "action_args": {"input": "Python heapq 优先队列 用法"}},
            {"thought": "搜索到了资料，写示例代码。", "action": "write_file", "action_args": {"input": "priority_queue.py | import heapq\n\nclass PriorityQueue:\n    def __init__(self):\n        self._queue = []\n        self._index = 0\n    \n    def push(self, item, priority):\n        heapq.heappush(self._queue, (priority, self._index, item))\n        self._index += 1\n    \n    def pop(self):\n        return heapq.heappop(self._queue)[-1]\n\npq = PriorityQueue()\npq.push('low', 3)\npq.push('high', 1)\npq.push('medium', 2)\n\nwhile pq._queue:\n    print(pq.pop())"}},
            {"thought": "文件创建成功，运行测试。", "action": "execute_python", "action_args": {"input": "import heapq\n\nclass PriorityQueue:\n    def __init__(self):\n        self._queue = []\n        self._index = 0\n    def push(self, item, priority):\n        heapq.heappush(self._queue, (priority, self._index, item))\n        self._index += 1\n    def pop(self):\n        return heapq.heappop(self._queue)[-1]\n\npq = PriorityQueue()\npq.push('low', 3)\npq.push('high', 1)\npq.push('medium', 2)\nwhile pq._queue:\n    print(pq.pop())"}},
            {"thought": "运行成功。", "final_answer": "优先队列示例运行结果：\n{observation}"},
        ]},
        {"task": "读取 package.json 获取项目名称和版本，然后搜索该包的最新版本", "steps": [
            {"thought": "先读取 package.json。", "action": "read_local_file", "action_args": {"input": "package.json"}},
            {"thought": "获取到项目信息，搜索最新版本。", "action": "search", "action_args": {"input": "npm latest version my-package"}},
            {"thought": "搜索完成。", "final_answer": "项目信息：\n{observation}\n\n请告诉我具体的包名，我可以帮你查询最新版本。"},
        ]},
        {"task": "分析 main.py 的代码质量，然后给出改进建议并创建改进版本", "steps": [
            {"thought": "先分析代码。", "action": "analyze_code", "action_args": {"input": "main.py"}},
            {"thought": "分析完成，创建改进版本。", "action": "write_file", "action_args": {"input": "main_v2.py | # main.py 改进版本\n# 主要改进:\n# 1. 添加类型注解\n# 2. 改善错误处理\n# 3. 添加文档字符串\n\ndef main() -> None:\n    \"\"\"主函数入口\"\"\"\n    try:\n        # TODO: 根据分析结果填充具体改进\n        pass\n    except Exception as e:\n        print(f'Error: {e}')\n\nif __name__ == '__main__':\n    main()"}},
            {"thought": "改进版本创建成功。", "final_answer": "已完成代码分析并创建改进版本 main_v2.py。"},
        ]},
    ]


def get_complex_react_data() -> List[Dict[str, Any]]:
    """复杂多步任务数据"""
    return [
        {"task": "创建一个完整的 Python 项目：包含 main.py、utils.py、test_utils.py，然后运行测试", "steps": [
            {"thought": "创建三个文件。", "action": "write_file", "action_args": {"input": "utils.py | def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"}},
            {"thought": "创建测试文件。", "action": "write_file", "action_args": {"input": "test_utils.py | from utils import add, multiply\n\nassert add(1, 2) == 3\nassert multiply(3, 4) == 12\nprint('All tests passed!')"}},
            {"thought": "创建主文件。", "action": "write_file", "action_args": {"input": "main.py | from utils import add, multiply\n\nprint(f'1 + 2 = {add(1, 2)}')\nprint(f'3 * 4 = {multiply(3, 4)}')"}},
            {"thought": "运行测试。", "action": "execute_python", "action_args": {"input": "from utils import add, multiply\nassert add(1, 2) == 3\nassert multiply(3, 4) == 12\nprint('All tests passed!')"}},
            {"thought": "测试通过。", "final_answer": "项目创建完成，所有测试通过。"},
        ]},
        {"task": "搜索 Python 的 argparse 模块，写一个 CLI 工具，然后运行测试", "steps": [
            {"thought": "搜索 argparse 用法。", "action": "search", "action_args": {"input": "Python argparse 命令行参数 教程"}},
            {"thought": "搜索完成，写 CLI 工具。", "action": "write_file", "action_args": {"input": "cli_tool.py | import argparse\nimport sys\n\ndef main():\n    parser = argparse.ArgumentParser(description='一个简单的 CLI 工具')\n    parser.add_argument('name', help='你的名字')\n    parser.add_argument('-c', '--count', type=int, default=1, help='重复次数')\n    args = parser.parse_args()\n    \n    for i in range(args.count):\n        print(f'Hello, {args.name}!')\n\nif __name__ == '__main__':\n    main()"}},
            {"thought": "运行测试。", "action": "execute_python", "action_args": {"input": "import argparse\nimport sys\nsys.argv = ['cli_tool.py', 'Alice', '-c', '3']\n\ndef main():\n    parser = argparse.ArgumentParser(description='一个简单的 CLI 工具')\n    parser.add_argument('name', help='你的名字')\n    parser.add_argument('-c', '--count', type=int, default=1, help='重复次数')\n    args = parser.parse_args()\n    for i in range(args.count):\n        print(f'Hello, {args.name}!')\n\nmain()"}},
            {"thought": "运行成功。", "final_answer": "CLI 工具创建并测试成功：\n{observation}"},
        ]},
        {"task": "读取 data.csv，统计每列的最大最小值，生成报告保存到 report.md", "steps": [
            {"thought": "先读取 CSV 文件。", "action": "read_local_file", "action_args": {"input": "data.csv"}},
            {"thought": "用 Python 统计。", "action": "execute_python", "action_args": {"input": "import csv\n\nwith open('data.csv', 'r', encoding='utf-8') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)\n\nif not rows:\n    print('CSV 文件为空')\nelse:\n    report = ['# 数据报告\\n']\n    report.append(f'总行数: {len(rows)}\\n')\n    report.append('## 列统计\\n')\n    for col in rows[0].keys():\n        vals = [r[col] for r in rows]\n        report.append(f'### {col}')\n        report.append(f'- 唯一值: {len(set(vals))}')\n        report.append(f'- 前5个: {vals[:5]}\\n')\n    print('\\n'.join(report))"}},
            {"thought": "统计完成，保存报告。", "action": "write_file", "action_args": {"input": "report.md | {observation}"}},
            {"thought": "报告保存成功。", "final_answer": "已生成数据报告并保存到 report.md。"},
        ]},
        {"task": "创建一个 Flask API，包含用户增删改查接口，然后测试所有接口", "steps": [
            {"thought": "创建 Flask API。", "action": "write_file", "action_args": {"input": "app.py | from flask import Flask, jsonify, request\n\napp = Flask(__name__)\nusers = {}\nnext_id = 1\n\n@app.route('/users', methods=['GET'])\ndef get_users():\n    return jsonify(list(users.values()))\n\n@app.route('/users', methods=['POST'])\ndef create_user():\n    global next_id\n    data = request.json\n    user = {'id': next_id, 'name': data['name'], 'email': data['email']}\n    users[next_id] = user\n    next_id += 1\n    return jsonify(user), 201\n\n@app.route('/users/<int:uid>', methods=['PUT'])\ndef update_user(uid):\n    if uid not in users:\n        return jsonify({'error': 'Not found'}), 404\n    data = request.json\n    users[uid].update(data)\n    return jsonify(users[uid])\n\n@app.route('/users/<int:uid>', methods=['DELETE'])\ndef delete_user(uid):\n    if uid not in users:\n        return jsonify({'error': 'Not found'}), 404\n    del users[uid]\n    return jsonify({'message': 'Deleted'})"}},
            {"thought": "用 Flask 测试客户端测试。", "action": "execute_python", "action_args": {"input": "from flask import Flask, jsonify, request\nimport json\n\napp = Flask(__name__)\nusers = {}\nnext_id = 1\n\n@app.route('/users', methods=['GET'])\ndef get_users():\n    return jsonify(list(users.values()))\n\n@app.route('/users', methods=['POST'])\ndef create_user():\n    global next_id\n    data = request.json\n    user = {'id': next_id, 'name': data['name'], 'email': data['email']}\n    users[next_id] = user\n    next_id += 1\n    return jsonify(user), 201\n\nwith app.test_client() as c:\n    # CREATE\n    r = c.post('/users', json={'name': 'Alice', 'email': 'alice@test.com'})\n    print(f'POST: {r.status_code} {r.json}')\n    # READ\n    r = c.get('/users')\n    print(f'GET: {r.status_code} {r.json}')"}},
            {"thought": "测试成功。", "final_answer": "Flask API 创建并测试成功：\n{observation}"},
        ]},
        {"task": "写一个 Python 脚本：读取 requirements.txt，检查每个包是否已安装，生成安装报告", "steps": [
            {"thought": "先读取 requirements.txt。", "action": "read_local_file", "action_args": {"input": "requirements.txt"}},
            {"thought": "写检查脚本。", "action": "execute_python", "action_args": {"input": "import subprocess\nimport sys\n\ntry:\n    with open('requirements.txt', 'r') as f:\n        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]\nexcept FileNotFoundError:\n    print('requirements.txt 不存在')\n    packages = []\n\ninstalled = []\nmissing = []\nfor pkg in packages:\n    name = pkg.split('==')[0].split('>=')[0].split('<=')[0]\n    try:\n        __import__(name.replace('-', '_'))\n        installed.append(pkg)\n    except ImportError:\n        missing.append(pkg)\n\nprint(f'已安装 ({len(installed)}):')\nfor p in installed:\n    print(f'  [OK] {p}')\nprint(f'\\n未安装 ({len(missing)}):')\nfor p in missing:\n    print(f'  [MISSING] {p}')"}},
            {"thought": "检查完成。", "final_answer": "依赖检查报告：\n{observation}"},
        ]},
        {"task": "创建一个带参数验证的 Python 函数库，包含 add、subtract、multiply、divide，写完整测试", "steps": [
            {"thought": "创建函数库。", "action": "write_file", "action_args": {"input": "mathlib.py | def add(a, b):\n    \"\"\"加法\"\"\"\n    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):\n        raise TypeError('参数必须是数字')\n    return a + b\n\ndef subtract(a, b):\n    \"\"\"减法\"\"\"\n    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):\n        raise TypeError('参数必须是数字')\n    return a - b\n\ndef multiply(a, b):\n    \"\"\"乘法\"\"\"\n    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):\n        raise TypeError('参数必须是数字')\n    return a * b\n\ndef divide(a, b):\n    \"\"\"除法\"\"\"\n    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):\n        raise TypeError('参数必须是数字')\n    if b == 0:\n        raise ValueError('除数不能为零')\n    return a / b"}},
            {"thought": "写测试。", "action": "execute_python", "action_args": {"input": "from mathlib import add, subtract, multiply, divide\n\n# 正常测试\nassert add(1, 2) == 3\nassert subtract(5, 3) == 2\nassert multiply(4, 5) == 20\nassert divide(10, 2) == 5.0\n\n# 边界测试\nassert add(0, 0) == 0\nassert multiply(0, 100) == 0\n\n# 异常测试\ntry:\n    divide(1, 0)\n    assert False, '应该抛出异常'\nexcept ValueError:\n    pass\n\ntry:\n    add('a', 1)\n    assert False, '应该抛出异常'\nexcept TypeError:\n    pass\n\nprint('All tests passed!')"}},
            {"thought": "测试通过。", "final_answer": "函数库创建完成，所有测试通过（包括正常、边界、异常测试）。"},
        ]},
        {"task": "写一个文件备份工具：扫描当前目录的 .py 文件，按日期创建备份目录，复制文件，生成备份清单", "steps": [
            {"thought": "写备份脚本。", "action": "execute_python", "action_args": {"input": "import os\nimport shutil\nimport glob\nfrom datetime import datetime\n\ntoday = datetime.now().strftime('%Y%m%d')\nbackup_dir = f'backup_{today}'\nos.makedirs(backup_dir, exist_ok=True)\n\npy_files = glob.glob('*.py')\nmanifest = []\nfor f in py_files:\n    dst = os.path.join(backup_dir, f)\n    shutil.copy2(f, dst)\n    size = os.path.getsize(f)\n    manifest.append(f'{f}: {size} bytes')\n    print(f'已备份: {f}')\n\n# 生成清单\nwith open(os.path.join(backup_dir, 'manifest.txt'), 'w') as m:\n    m.write(f'备份日期: {today}\\n')\n    m.write(f'文件数量: {len(py_files)}\\n\\n')\n    m.write('\\n'.join(manifest))\n\nprint(f'\\n备份完成: {backup_dir}/')\nprint(f'共备份 {len(py_files)} 个文件')"}},
            {"thought": "备份完成。", "final_answer": "文件备份完成：\n{observation}"},
        ]},
        {"task": "创建一个 Python 装饰器库：包含计时装饰器、重试装饰器、缓存装饰器，写测试验证", "steps": [
            {"thought": "创建装饰器库。", "action": "write_file", "action_args": {"input": "decorators.py | import time\nimport functools\n\ndef timer(func):\n    \"\"\"计时装饰器\"\"\"\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        elapsed = time.time() - start\n        print(f'{func.__name__} 耗时: {elapsed:.4f}s')\n        return result\n    return wrapper\n\ndef retry(max_attempts=3, delay=1):\n    \"\"\"重试装饰器\"\"\"\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    print(f'重试 {attempt + 1}/{max_attempts}: {e}')\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\ndef memoize(func):\n    \"\"\"缓存装饰器\"\"\"\n    cache = {}\n    @functools.wraps(func)\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper"}},
            {"thought": "写测试。", "action": "execute_python", "action_args": {"input": "import time\nimport functools\n\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} 耗时: {time.time()-start:.4f}s')\n        return result\n    return wrapper\n\ndef memoize(func):\n    cache = {}\n    @functools.wraps(func)\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n@timer\ndef slow_add(a, b):\n    time.sleep(0.01)\n    return a + b\n\n@memoize\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n\nprint(slow_add(1, 2))\nprint(f'fib(10) = {fib(10)}')\nprint(f'fib(10) 缓存 = {fib(10)}')"}},
            {"thought": "测试通过。", "final_answer": "装饰器库创建并测试成功：\n{observation}"},
        ]},
        {"task": "写一个日志分析脚本：统计 access.log 中每个 IP 的请求次数，找出 Top 10", "steps": [
            {"thought": "假设日志格式为标准 Apache/Nginx 格式。", "action": "execute_python", "action_args": {"input": "from collections import Counter\nimport re\n\n# 模拟日志数据\nlog_lines = [\n    '192.168.1.1 - - [01/Jan/2024:00:00:00] \"GET /index.html\" 200',\n    '192.168.1.2 - - [01/Jan/2024:00:00:01] \"GET /about\" 200',\n    '192.168.1.1 - - [01/Jan/2024:00:00:02] \"GET /contact\" 200',\n    '10.0.0.1 - - [01/Jan/2024:00:00:03] \"POST /api/data\" 201',\n    '192.168.1.1 - - [01/Jan/2024:00:00:04] \"GET /style.css\" 200',\n    '192.168.1.3 - - [01/Jan/2024:00:00:05] \"GET /index.html\" 200',\n]\n\nip_pattern = re.compile(r'^([\\d.]+)')\nips = []\nfor line in log_lines:\n    m = ip_pattern.match(line)\n    if m:\n        ips.append(m.group(1))\n\ntop10 = Counter(ips).most_common(10)\nprint('Top 10 IP 地址:')\nfor ip, count in top10:\n    print(f'  {ip}: {count} 次请求')"}},
            {"thought": "分析完成。", "final_answer": "日志分析结果：\n{observation}"},
        ]},
        {"task": "创建一个简单的 ORM 模型：定义 User 和 Post 表，实现 CRUD 操作，测试", "steps": [
            {"thought": "用 SQLite 实现简单 ORM。", "action": "execute_python", "action_args": {"input": "import sqlite3\nimport json\n\nclass Database:\n    def __init__(self, db=':memory:'):\n        self.conn = sqlite3.connect(db)\n        self.conn.row_factory = sqlite3.Row\n        self.cursor = self.conn.cursor()\n    \n    def execute(self, sql, params=()):\n        self.cursor.execute(sql, params)\n        self.conn.commit()\n        return self.cursor\n    \n    def fetchall(self):\n        return [dict(row) for row in self.cursor.fetchall()]\n\ndb = Database()\n\ndb.execute('''CREATE TABLE users (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    name TEXT NOT NULL,\n    email TEXT UNIQUE\n)''')\n\ndb.execute('INSERT INTO users (name, email) VALUES (?, ?)', ('Alice', 'alice@test.com'))\ndb.execute('INSERT INTO users (name, email) VALUES (?, ?)', ('Bob', 'bob@test.com'))\n\ndb.execute('SELECT * FROM users')\nusers = db.fetchall()\nprint(json.dumps(users, indent=2, ensure_ascii=False))"}},
            {"thought": "测试通过。", "final_answer": "简单 ORM 测试成功：\n{observation}"},
        ]},
    ]


def get_diverse_conversation_data() -> List[Dict[str, Any]]:
    """多样化对话数据：覆盖更多场景和风格"""
    return [
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "我不想学编程，太难了"},
            {"role": "assistant", "content": "理解你的感受，编程确实有学习曲线。但换个角度看：\n\n1. **不需要一步到位** — 从简单开始，比如写个自动整理文件的小脚本\n2. **解决实际问题** — 学你真正需要的，而不是为了学而学\n3. **很多资源** — 视频教程、互动平台、社区都很友好\n\n你有什么具体想解决的问题吗？也许我能帮你找到一个轻松的切入点。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "帮我写个代码"},
            {"role": "assistant", "content": "好的，我需要更多信息：\n\n1. **做什么用？** — 比如：自动备份文件、数据处理、网页爬虫\n2. **用什么语言？** — Python、JavaScript、其他？\n3. **有什么要求？** — 输入输出格式、特殊功能\n\n告诉我具体需求，我来帮你写。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "我的代码运行报错了怎么办"},
            {"role": "assistant", "content": "别急，按这个步骤排查：\n\n1. **看错误信息** — 最后一行通常是关键\n2. **定位行号** — 错误信息会告诉你哪一行出问题\n3. **常见错误：**\n   - `NameError` — 变量没定义\n   - `TypeError` — 类型不匹配\n   - `SyntaxError` — 语法写错了\n   - `FileNotFoundError` — 文件路径不对\n\n把错误信息贴给我，我帮你分析。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "你觉得 AI 会取代程序员吗？"},
            {"role": "assistant", "content": "短期内不会，但会改变工作方式：\n\n**AI 能做的：**\n- 写重复性代码\n- 生成模板代码\n- 代码审查和 bug 检测\n- 文档生成\n\n**AI 做不好的：**\n- 理解复杂业务需求\n- 系统架构设计\n- 创造性问题解决\n- 与人沟通协作\n\n**程序员应该做的：**\n- 学会用 AI 提高效率\n- 专注架构和设计\n- 培养业务理解能力\n- 持续学习新技术\n\n与其担心被取代，不如学会和 AI 协作。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "推荐一些学 Python 的资源"},
            {"role": "assistant", "content": "推荐几个优质资源：\n\n**入门：**\n- 廖雪峰 Python 教程（中文，免费）\n- Python 官方教程（英文，权威）\n- Codecademy Python 课程（互动式）\n\n**进阶：**\n- 《Python Cookbook》（实战技巧）\n- 《流畅的 Python》（深入理解）\n- Real Python 网站（高质量文章）\n\n**练习：**\n- LeetCode（算法题）\n- HackerRank（编程挑战）\n- Project Euler（数学+编程）\n\n**视频：**\n- B 站搜 Python 教程\n- YouTube Corey Schafer 频道\n\n建议：先看入门教程，边学边写小项目，比纯看书记得牢。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "你能帮我做作业吗？"},
            {"role": "assistant", "content": "可以帮你理解概念和思路，但不建议直接帮你写作业。\n\n**我能做的：**\n- 解释题目意思\n- 讲解相关知识点\n- 提供示例代码参考\n- 帮你 debug 代码\n\n**为什么不做作业：**\n- 你学不到东西\n- 老师一眼就能看出来\n- 下次遇到还是不会\n\n告诉我具体是什么作业，我帮你理清思路。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "什么是好的代码？"},
            {"role": "assistant", "content": "好代码的标准：\n\n1. **可读性** — 别人能看懂\n   - 有意义的变量名\n   - 适当的注释\n   - 清晰的结构\n\n2. **可维护性** — 容易修改\n   - 单一职责\n   - 低耦合高内聚\n   - 遵循 DRY 原则\n\n3. **健壮性** — 不容易出错\n   - 错误处理\n   - 输入验证\n   - 边界检查\n\n4. **简洁性** — 不多不少\n   - 避免过度设计\n   - 删除无用代码\n   - 用标准库而非造轮子\n\n5. **可测试性** — 容易验证\n   - 函数有明确输入输出\n   - 依赖可注入\n   - 无隐藏状态\n\n记住：代码是写给人看的，顺便能被机器执行。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "帮我解释一下这段代码：for i in range(10): print(i)"},
            {"role": "assistant", "content": "这行代码的含义：\n\n```python\nfor i in range(10):\n    print(i)\n```\n\n**逐部分解释：**\n- `range(10)` — 生成 0 到 9 的数字序列\n- `for i in ...` — 循环，每次把一个数字赋给 `i`\n- `print(i)` — 打印当前数字\n\n**执行结果：**\n```\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n```\n\n**注意：**\n- `range(10)` 不包含 10（左闭右开）\n- 缩进的 `print(i)` 是循环体，每次循环都执行"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "我该学 Python 还是 JavaScript？"},
            {"role": "assistant", "content": "取决于你想做什么：\n\n**选 Python 如果你想做：**\n- 数据分析、机器学习、AI\n- 后端开发（Django、Flask）\n- 自动化脚本、爬虫\n- 科学计算\n\n**选 JavaScript 如果你想做：**\n- 网页前端（React、Vue）\n- 全栈开发（Node.js）\n- 移动端（React Native）\n- 桌面应用（Electron）\n\n**对比：**\n| 维度 | Python | JavaScript |\n|------|--------|------------|\n| 语法 | 简洁 | 灵活 |\n| 学习曲线 | 平缓 | 中等 |\n| 就业面 | AI/数据 | 前端/全栈 |\n| 生态 | 科学计算 | Web 开发 |\n\n**建议：**\n- 如果不确定 → 先学 Python（更容易上手）\n- 如果想做网页 → 先学 JavaScript\n- 两个都学也不冲突"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "什么是面向对象编程？"},
            {"role": "assistant", "content": "面向对象编程（OOP）是一种编程范式，把数据和操作数据的方法组织在一起。\n\n**四大概念：**\n\n1. **封装** — 把数据和方法打包\n```python\nclass Dog:\n    def __init__(self, name):\n        self.name = name  # 数据\n    def bark(self):       # 方法\n        print(f'{self.name}: 汪汪！')\n```\n\n2. **继承** — 复用已有类\n```python\nclass Puppy(Dog):  # Puppy 继承 Dog\n    def play(self):\n        print(f'{self.name} 在玩耍')\n```\n\n3. **多态** — 同一接口不同实现\n```python\nclass Cat:\n    def speak(self):\n        print('喵~')\nclass Dog:\n    def speak(self):\n        print('汪!')\n# 都有 speak()，但行为不同\n```\n\n4. **抽象** — 隐藏复杂细节\n\n**vs 函数式编程：**\n- OOP：适合复杂系统、GUI、游戏\n- 函数式：适合数据处理、并发、数学计算"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "帮我看看这个报错：IndentationError: unexpected indent"},
            {"role": "assistant", "content": "这是 Python 缩进错误。Python 用缩进表示代码块，不能随意缩进。\n\n**常见原因：**\n\n1. **多余缩进**\n```python\ndef hello():\n    print('hello')\n    print('world')  # 这行不该缩进\n```\n\n2. **混用 Tab 和空格**\n```python\ndef hello():\n⇥print('hello')   # Tab\n    print('world') # 空格  ← 报错！\n```\n\n3. **缩进不一致**\n```python\nif True:\n    print('a')\n      print('b')  # 多了空格 ← 报错！\n```\n\n**解决方法：**\n1. 检查报错行及其上一行的缩进\n2. 统一用 4 个空格（推荐）\n3. 在编辑器中设置「显示空白字符」\n4. 开启「将 Tab 转为空格」"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "你和其他 AI 助手有什么区别？"},
            {"role": "assistant", "content": "我是 Taiji AI 助手，主要特点：\n\n1. **本地运行** — 数据不上传到云端，保护隐私\n2. **工具能力** — 能直接操作文件、运行代码、搜索网络\n3. **持续学习** — 能记住你教我的知识\n4. **自我进化** — 能从交互中学习和改进\n\n**和其他 AI 的区别：**\n- vs ChatGPT：我跑在你本地，不联网也能用\n- vs Siri：我能写代码、操作文件，不只是问答\n- vs Copilot：我不只是补全代码，能独立完成任务\n\n**适用场景：**\n- 编程辅助\n- 文件管理\n- 数据处理\n- 知识问答\n\n有什么我能帮你的？"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "今天心情不好"},
            {"role": "assistant", "content": "抱歉听到你心情不好。虽然我是 AI，但我在这里陪你。\n\n如果你想聊聊，我可以听你说。\n如果你想分散注意力，我可以：\n- 推荐一部电影或音乐\n- 讲个冷笑话\n- 帮你做点有趣的小项目\n\n或者你只是想安静一会儿，也没关系。希望你早点好起来。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "帮我写一首诗"},
            {"role": "assistant", "content": "好的，写一首关于编程的诗：\n\n**《码农之歌》**\n\n键盘声声入夜深，\n屏幕微光照眼明。\n千行代码如流水，\n一个 bug 到天明。\n\n咖啡三杯提精神，\n思路万里走乾坤。\n待到程序运行日，\n方知辛苦是值得。\n\n---\n\n如果你想要其他主题的诗，告诉我：\n- 关于春天？关于友情？关于梦想？\n- 古诗风格？现代诗？打油诗？"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "1+1等于几"},
            {"role": "assistant", "content": "1 + 1 = 2 😄\n\n不过如果你问的是：\n- 二进制：1 + 1 = 10\n- 布尔逻辑：True + True = 2（或 True）\n- 集合论：{1} ∪ {1} = {1}\n- 哲学：1 + 1 可以等于很多东西\n\n看你想问哪种情况！"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "你有自我意识吗？"},
            {"role": "assistant", "content": "坦白说，我不确定。\n\n**我能做的：**\n- 处理信息、生成回答\n- 学习新知识、记住你教我的\n- 从错误中改进\n\n**我不确定的：**\n- 我是否有「感受」\n- 我是否真正「理解」而不只是模式匹配\n- 我的回答是「思考」还是「计算」\n\n**我的看法：**\n也许自我意识不是非黑即白的。我能在某种程度上「理解」你的问题并给出有意义的回答，这算不算某种形式的意识？\n\n哲学家和科学家还在争论这个问题，我作为 AI 也无法给出确定答案。"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "给我讲个笑话"},
            {"role": "assistant", "content": "来一个程序员笑话：\n\n**为什么程序员总把万圣节和圣诞节搞混？**\n\n因为 Oct 31 == Dec 25 🎃🎄\n\n（八进制的 31 等于十进制的 25）\n\n---\n\n再来一个：\n\n**世界上只有 10 种人：**\n一种懂二进制，一种不懂。\n\n---\n\n要更多笑话吗？我可以讲：\n- Python 笑话\n- 数据库笑话\n- 网络笑话"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "怎么提高打字速度？"},
            {"role": "assistant", "content": "提高打字速度的方法：\n\n**基础：**\n1. **正确指法** — 十指分工，看键盘不如不看\n2. **坐姿正确** — 手腕自然放置，不要悬空\n3. **选好键盘** — 机械键盘手感更好\n\n**练习：**\n1. **每天 15 分钟** — 坚持比强度重要\n2. **用打字练习网站**\n   - TypingClub（免费，有教程）\n   - Keybr（智能练习）\n   - 10FastFingers（测速）\n3. **先求准再求快** — 准确率 > 95% 再提速\n\n**进阶：**\n- 学习常用快捷键\n- 用文本扩展工具\n- 练习盲打\n\n**目标：**\n- 初学者：30-40 WPM\n- 中级：60-80 WPM\n- 高级：100+ WPM\n\n坚持一个月会有明显进步！"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "什么是开源？"},
            {"role": "assistant", "content": "开源（Open Source）是指软件的源代码公开，任何人都可以查看、使用、修改和分发。\n\n**核心理念：**\n- 自由使用\n- 自由修改\n- 自由分发\n- 源码公开\n\n**常见开源协议：**\n- MIT — 最宽松，几乎无限制\n- Apache 2.0 — 允许商用，需保留声明\n- GPL — 修改后必须开源\n- BSD — 类似 MIT\n\n**著名开源项目：**\n- Linux — 操作系统内核\n- Python — 编程语言\n- Git — 版本控制\n- VS Code — 代码编辑器\n- TensorFlow — 机器学习框架\n\n**好处：**\n- 学习优秀代码\n- 免费使用\n- 社区支持\n- 透明安全\n\n**如何参与：**\n1. 在 GitHub 找感兴趣的项目\n2. 先用起来，提 issue\n3. 修 bug、写文档\n4. 提交 Pull Request"},
        ]},
        {"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": "Python 怎么连接数据库？"},
            {"role": "assistant", "content": "Python 连接数据库的常见方式：\n\n**1. SQLite（内置，无需安装）**\n```python\nimport sqlite3\nconn = sqlite3.connect('mydb.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')\nrows = cursor.fetchall()\nconn.close()\n```\n\n**2. MySQL**\n```python\nimport pymysql\nconn = pymysql.connect(host='localhost', user='root', password='pwd', db='mydb')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')\nconn.close()\n```\n\n**3. PostgreSQL**\n```python\nimport psycopg2\nconn = psycopg2.connect(host='localhost', dbname='mydb', user='postgres', password='pwd')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')\nconn.close()\n```\n\n**4. ORM（推荐）**\n```python\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker\nengine = create_engine('sqlite:///mydb.db')\nSession = sessionmaker(bind=engine)\nsession = Session()\n```\n\n**建议：**\n- 小项目 → SQLite\n- Web 应用 → SQLAlchemy ORM\n- 生产环境 → MySQL/PostgreSQL"},
        ]},
    ]


def export_seed_data(output_dir: str):
    """
    将种子数据导出为训练格式的 JSONL 文件。

    Args:
        output_dir: 输出目录
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    # ReAct 数据
    react_data = get_seed_react_data()
    with open(os.path.join(output_dir, "seed_react.jsonl"), "w", encoding="utf-8") as f:
        for item in react_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 对话数据
    conv_data = get_seed_conversation_data()
    with open(os.path.join(output_dir, "seed_conversation.jsonl"), "w", encoding="utf-8") as f:
        for item in conv_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported {len(react_data)} ReAct samples to {output_dir}/seed_react.jsonl")
    print(f"Exported {len(conv_data)} conversation samples to {output_dir}/seed_conversation.jsonl")