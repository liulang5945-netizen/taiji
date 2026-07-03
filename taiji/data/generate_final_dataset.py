"""
态极终极训练数据集生成器
========================

为态极的五个头量身定制：
- 语言头：对话、知识、代码
- 工具头：全部工具的多样化用法
- 记忆头：存储、回忆、更新、上下文
- 规划头：任务分解、子目标、优先级
- 感知头：环境状态、多模态

加上安全对齐和错误恢复。

用法：python -m taiji.data.generate_final_dataset
"""
import os
import json
import random
import hashlib
from typing import List, Dict, Any

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_data")


def save(data: list, filename: str) -> int:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {filename}: {len(data)} 条")
    return len(data)


def dedup(data: list, key_fn) -> list:
    seen = set()
    result = []
    for item in data:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


# ═══════════════════════════════════════════════════════
# 1. 语言头数据：对话、知识问答、代码
# ═══════════════════════════════════════════════════════

def gen_language() -> list:
    data = []

    # 1.1 知识问答（覆盖主要领域）
    qa_pairs = [
        # 编程基础
        ("什么是变量", "变量是存储数据的命名容器。Python 中用 `x = 10` 创建。变量有名称、值和类型。"),
        ("什么是函数", "函数是可重用的代码块。用 `def` 定义，可以接收参数、返回值。好处：代码复用、逻辑分离、易于测试。"),
        ("什么是类", "类是对象的蓝图，用 `class` 定义。包含属性（数据）和方法（行为）。支持继承、封装、多态。"),
        ("什么是模块", "模块是 Python 文件（.py），包含函数、类、变量。用 `import` 导入。好处：代码组织、命名空间隔离。"),
        ("什么是异常", "异常是程序运行时的错误。用 `try/except` 捕获。常见：ValueError、TypeError、FileNotFoundError。"),
        ("什么是装饰器", "装饰器是修改函数行为的函数，用 `@decorator` 语法。本质是高阶函数，接收函数返回函数。"),
        ("什么是生成器", "生成器用 `yield` 产生值的惰性迭代器。内存高效，适合大数据集。"),
        ("什么是上下文管理器", "用 `with` 语句管理资源（文件、连接）。实现 `__enter__` 和 `__exit__` 方法。"),
        # 数据结构
        ("列表和元组的区别", "列表可变 `[]`，元组不可变 `()`。元组更快、可做字典键。列表适合动态数据，元组适合固定数据。"),
        ("字典和集合的区别", "字典是键值对 `{k: v}`，集合是无序唯一值 `{v}`。字典用于映射，集合用于去重和集合运算。"),
        ("栈和队列的区别", "栈 LIFO（后进先出），队列 FIFO（先进先出）。栈用 list.append/pop，队列用 collections.deque。"),
        ("什么是哈希表", "哈希表通过哈希函数将键映射到数组索引，实现 O(1) 查找。Python 的 dict 和 set 基于哈希表。"),
        # 算法
        ("什么是递归", "递归是函数调用自身。必须有基准条件（停止）和递归步骤（分解问题）。注意栈溢出风险。"),
        ("什么是动态规划", "动态规划将问题分解为子问题，存储已解决的子问题结果。适用于最优子结构和重叠子问题。"),
        ("什么是贪心算法", "贪心每步选择局部最优，期望得到全局最优。适用于贪心选择性质和最优子结构的问题。"),
        ("时间复杂度是什么", "时间复杂度描述算法运行时间随输入规模增长的趋势。O(1) < O(log n) < O(n) < O(n log n) < O(n²)。"),
        # 系统
        ("什么是进程和线程", "进程是程序实例（独立内存），线程是进程内的执行单元（共享内存）。Python GIL 限制多线程并行。"),
        ("什么是协程", "协程是用户态的轻量级线程，用 async/await 定义。适合 I/O 密集任务，单线程并发。"),
        ("什么是缓存", "缓存是存储频繁访问数据的高速存储层。策略：LRU（最近最少使用）、TTL（过期时间）。"),
        ("什么是消息队列", "消息队列是异步通信机制。生产者发送消息，消费者接收处理。解耦、削峰、异步。"),
        # AI/ML
        ("什么是神经网络", "神经网络是模拟大脑的计算模型，由层（输入、隐藏、输出）组成。每层有神经元、权重、激活函数。"),
        ("什么是反向传播", "反向传播是训练神经网络的核心算法。计算损失函数对权重的梯度，用链式法则逐层传递。"),
        ("什么是过拟合", "过拟合是模型在训练集上表现好但泛化差。原因：模型太复杂、数据太少。解决：正则化、Dropout、数据增强。"),
        ("什么是 Transformer", "Transformer 是基于自注意力的架构。核心：多头注意力、前馈网络、位置编码。是 GPT/BERT 的基础。"),
        # 网络
        ("HTTP 和 HTTPS 的区别", "HTTPS = HTTP + TLS 加密。HTTP 明文传输，HTTPS 加密传输。HTTPS 默认端口 443，HTTP 80。"),
        ("TCP 三次握手", "1. 客户端→SYN→服务器\n2. 服务器→SYN+ACK→客户端\n3. 客户端→ACK→服务器\n建立可靠连接。"),
        ("什么是 REST API", "REST 是基于 HTTP 的架构风格。资源用 URL 表示，操作用 HTTP 方法（GET/POST/PUT/DELETE）。无状态。"),
        ("什么是 WebSocket", "WebSocket 是全双工通信协议。握手后持久连接，双向实时通信。适合聊天、游戏、实时数据。"),
        # 数据库
        ("SQL 和 NoSQL 的区别", "SQL 关系型（表结构、ACID），NoSQL 非关系型（灵活 schema、高扩展）。SQL 适合复杂查询，NoSQL 适合大数据。"),
        ("什么是索引", "索引是加速查询的数据结构（通常 B+ 树）。代价：占用空间、降低写入速度。适合 WHERE/JOIN 列。"),
        ("什么是事务", "事务是一组操作的原子单元。ACID：原子性、一致性、隔离性、持久性。"),
        # DevOps
        ("什么是 Docker", "Docker 是容器化平台，将应用+依赖打包为容器。轻量、隔离、可移植。镜像→容器→仓库。"),
        ("什么是 CI/CD", "CI 持续集成（自动构建+测试），CD 持续交付/部署（自动发布）。工具：GitHub Actions、Jenkins。"),
        ("什么是 Kubernetes", "Kubernetes 是容器编排平台。自动部署、扩缩容、负载均衡。核心概念：Pod、Service、Deployment。"),
    ]

    for q, a in qa_pairs:
        # 多种系统提示变体
        for sys_prompt in [
            "你是 Taiji AI 助手。",
            "你是一个专业的技术助手。",
            "你是本地 AI 助手，擅长解释技术概念。",
        ]:
            data.append({"messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]})

    # 1.2 代码生成
    code_tasks = [
        ("写一个快速排序", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"),
        ("写一个二分查找", "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
        ("写一个 LRU 缓存", "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)"),
        ("写一个装饰器计时器", "import time\nimport functools\n\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} 耗时: {time.time()-start:.4f}s')\n        return result\n    return wrapper"),
        ("写一个单例模式", "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"),
        ("写一个观察者模式", "class EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n    def on(self, event, fn):\n        self._listeners.setdefault(event, []).append(fn)\n    def emit(self, event, *args):\n        for fn in self._listeners.get(event, []):\n            fn(*args)"),
        ("写一个生产者消费者", "import threading\nimport queue\n\nq = queue.Queue(maxsize=10)\n\ndef producer():\n    for i in range(20):\n        q.put(i)\n        print(f'生产: {i}')\n\ndef consumer():\n    while True:\n        item = q.get()\n        print(f'消费: {item}')\n        q.task_done()\n\nthreading.Thread(target=producer).start()\nthreading.Thread(target=consumer, daemon=True).start()"),
        ("写一个简单 HTTP 服务器", "from http.server import HTTPServer, SimpleHTTPRequestHandler\n\nserver = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)\nprint('服务器运行在 http://localhost:8000')\nserver.serve_forever()"),
    ]

    for task, code in code_tasks:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]})

    # 1.3 批量生成知识问答变体
    topics = [
        "Python", "JavaScript", "Go", "Rust", "SQL", "Docker", "Git", "Linux",
        "React", "Vue", "Flask", "Django", "FastAPI", "PostgreSQL", "Redis",
        "Kubernetes", "AWS", "TensorFlow", "PyTorch", "Pandas", "NumPy",
    ]
    question_types = [
        ("什么是{}", "{}是一种技术/工具，用于..."),
        ("{}怎么用", "{}的使用方法：1. 安装 2. 配置 3. 使用"),
        ("{}的最佳实践", "{}最佳实践：1. 遵循官方指南 2. 使用最新版本 3. 编写测试"),
        ("{}和什么配合使用", "{}常与以下工具配合：1. 相关工具A 2. 相关工具B"),
        ("{}的常见错误", "{}常见错误：1. 配置错误 2. 版本不兼容 3. 权限问题"),
        ("{}的性能优化", "{}性能优化：1. 缓存 2. 索引 3. 异步处理 4. 批量操作"),
        ("{}的安全注意事项", "{}安全注意：1. 输入验证 2. 权限控制 3. 加密传输 4. 日志审计"),
    ]

    for topic in topics:
        for q_tpl, a_tpl in question_types:
            q = q_tpl.format(topic)
            a = a_tpl.format(topic)
            data.append({"messages": [
                {"role": "system", "content": "你是 Taiji AI 助手。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]})

    # 1.4 代码生成变体
    code_patterns = [
        ("排序算法", "def sort_data(arr):\n    return sorted(arr)"),
        ("搜索函数", "def search(items, target):\n    return [x for x in items if target in str(x)]"),
        ("数据验证", "def validate(data):\n    errors = []\n    if not data.get('name'): errors.append('name required')\n    return errors"),
        ("日志记录", "import logging\nlogger = logging.getLogger(__name__)\ndef log_event(msg):\n    logger.info(msg)"),
        ("配置加载", "import json\ndef load_config(path):\n    with open(path) as f:\n        return json.load(f)"),
        ("文件遍历", "import os\ndef list_files(dir):\n    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]"),
        ("字符串处理", "def clean_text(text):\n    return text.strip().lower().replace('\\n', ' ')"),
        ("数据转换", "def convert(data):\n    return {k: str(v) for k, v in data.items()}"),
    ]

    for pattern_name, code in code_patterns:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": f"写一个{pattern_name}"},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]})

    # 1.3 对话风格多样化
    styles = [
        ("你好", [("你好！有什么我能帮你的吗？", "友好"), ("嗨！我是 Taiji AI 助手，很高兴为你服务。", "正式"), ("嘿！需要帮忙吗？", "随意")]),
        ("谢谢", [("不客气！有问题随时问我。", "友好"), ("很高兴能帮到你。", "正式"), ("没事，随时找我！", "随意")]),
        ("我不理解这段代码", [("没关系，我来帮你解释。这段代码的作用是...", "耐心"), ("让我用简单的方式解释一下...", "清晰"), ("哪部分不明白？我逐行给你讲。", "引导")]),
    ]

    for user_msg, responses in styles:
        for resp, style in responses:
            data.append({"messages": [
                {"role": "system", "content": f"你是 Taiji AI 助手，风格：{style}。"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": resp},
            ]})

    return data


# ═══════════════════════════════════════════════════════
# 2. 工具头数据：全部工具的多样化用法
# ═══════════════════════════════════════════════════════

def gen_tools() -> list:
    data = []

    # 工具定义
    tools = {
        "read_local_file": {
            "desc": "读取文件内容",
            "args": "input: 文件路径",
            "scenarios": [
                ("读取 config.json", "config.json"),
                ("查看 README.md", "README.md"),
                ("读取日志文件", "app.log"),
                ("查看 .env 配置", ".env"),
                ("读取 package.json", "package.json"),
                ("查看 Dockerfile", "Dockerfile"),
                ("读取 requirements.txt", "requirements.txt"),
                ("查看 .gitignore", ".gitignore"),
            ],
        },
        "write_file": {
            "desc": "创建/写入文件",
            "args": "input: '路径 | 内容'",
            "scenarios": [
                ("创建 hello.py", "hello.py | print('Hello')"),
                ("写入配置文件", "config.json | {\"version\": \"1.0\"}"),
                ("创建 README", "README.md | # My Project"),
                ("写入数据", "data.csv | name,age\nAlice,30"),
                ("创建 .gitignore", ".gitignore | __pycache__/\n*.pyc"),
                ("写入脚本", "run.sh | #!/bin/bash\necho 'Hello'"),
            ],
        },
        "edit_file": {
            "desc": "编辑文件内容",
            "args": "input: '路径 | 旧文本 | 新文本'",
            "scenarios": [
                ("修改版本号", "package.json | 1.0.0 | 2.0.0"),
                ("替换配置值", "config.json | debug: true | debug: false"),
                ("修复拼写", "README.md | teh | the"),
                ("更新端口", ".env | PORT=3000 | PORT=8080"),
            ],
        },
        "list_directory": {
            "desc": "列出目录内容",
            "args": "input: 目录路径",
            "scenarios": [
                ("查看当前目录", "."),
                ("查看 src 目录", "src"),
                ("查看项目结构", "."),
                ("查看日志目录", "logs"),
            ],
        },
        "execute_python": {
            "desc": "执行 Python 代码",
            "args": "input: Python 代码",
            "scenarios": [
                ("计算 1+1", "print(1+1)"),
                ("获取当前时间", "from datetime import datetime\nprint(datetime.now())"),
                ("列出文件", "import os\nprint(os.listdir('.'))"),
                ("读取环境变量", "import os\nprint(os.environ.get('PATH', ''))"),
                ("生成随机数", "import random\nprint(random.randint(1, 100))"),
                ("统计字数", "text = 'Hello World'\nprint(len(text.split()))"),
            ],
        },
        "search": {
            "desc": "搜索互联网",
            "args": "input: 搜索关键词",
            "scenarios": [
                ("搜索 Python 教程", "Python 入门教程"),
                ("搜索错误信息", "Python TypeError unhashable type"),
                ("搜索最新版本", "Node.js latest version"),
                ("搜索最佳实践", "React hooks best practices"),
            ],
        },
        "read_webpage": {
            "desc": "读取网页内容",
            "args": "input: URL",
            "scenarios": [
                ("读取 Python 文档", "https://docs.python.org/3/"),
                ("读取 PyTorch 安装页", "https://pytorch.org/get-started/locally/"),
                ("读取 GitHub 仓库", "https://github.com/python/cpython"),
            ],
        },
        "create_project": {
            "desc": "创建项目",
            "args": "input: '类型 | 名称'",
            "scenarios": [
                ("创建 Python 项目", "python-script | my-app"),
                ("创建 Web 项目", "web-app | my-web"),
            ],
        },
        "analyze_code": {
            "desc": "分析代码",
            "args": "input: 文件路径",
            "scenarios": [
                ("分析 main.py", "main.py"),
                ("分析 app.py", "app.py"),
            ],
        },
        "install_dependency": {
            "desc": "安装依赖",
            "args": "input: 包名",
            "scenarios": [
                ("安装 requests", "requests"),
                ("安装 flask", "flask"),
                ("安装 numpy", "numpy"),
                ("安装 pandas", "pandas"),
            ],
        },
        "learn_knowledge": {
            "desc": "学习知识",
            "args": "input: 知识内容",
            "scenarios": [
                ("学习 Python 装饰器", "装饰器是修改函数行为的高阶函数"),
                ("学习 Docker 基础", "Docker 是容器化平台"),
            ],
        },
        "query_knowledge": {
            "desc": "查询知识",
            "args": "input: 查询关键词",
            "scenarios": [
                ("查询装饰器", "装饰器"),
                ("查询 Docker", "Docker"),
            ],
        },
        "run_command": {
            "desc": "运行系统命令",
            "args": "input: 命令",
            "scenarios": [
                ("查看 Python 版本", "python --version"),
                ("查看 pip 列表", "pip list"),
                ("查看 git 状态", "git status"),
                ("查看磁盘空间", "dir"),
            ],
        },
    }

    # 为每个工具生成多样化的样本
    thought_templates = [
        "用户想{task}，使用 {tool} 工具。",
        "分析任务：{task}。最合适的工具是 {tool}，因为{desc}。",
        "需要{task}，{tool} 是最佳选择。",
        "执行{task}，调用 {tool}。",
    ]

    for tool_name, tool_info in tools.items():
        for task, args in tool_info["scenarios"]:
            for tpl in thought_templates:
                thought = tpl.format(task=task, tool=tool_name, desc=tool_info["desc"])
                data.append({
                    "task": task,
                    "steps": [
                        {"thought": thought, "action": tool_name, "action_args": {"input": args}},
                        {"thought": "操作完成，整理结果。", "final_answer": f"已完成: {task}\n{{observation}}"},
                    ],
                })

    # 批量工具使用变体
    batch_tasks = [
        ("read_local_file", ["读取配置", "查看日志", "检查代码", "读取数据", "查看文档"]),
        ("write_file", ["创建脚本", "写入配置", "生成报告", "保存数据", "创建模板"]),
        ("execute_python", ["运行测试", "数据处理", "计算统计", "生成图表", "自动化任务"]),
        ("search", ["查找文档", "搜索解决方案", "调研技术", "查找错误信息", "搜索最佳实践"]),
        ("run_command", ["检查状态", "执行脚本", "管理服务", "查看进程", "清理文件"]),
    ]

    for tool, tasks in batch_tasks:
        for task in tasks:
            for i in range(5):
                data.append({
                    "task": f"{task}（变体{i+1}）",
                    "steps": [
                        {"thought": f"用户要{task}，使用 {tool}。", "action": tool, "action_args": {"input": f"variant_{i}"}},
                        {"thought": "操作完成。", "final_answer": f"已完成: {task}"},
                    ],
                })

    # 工具组合使用
    combos = [
        ("读取 config.json 并修改版本号", [
            ("read_local_file", {"input": "config.json"}, "读取配置文件"),
            ("edit_file", {"input": "config.json | 1.0.0 | 2.0.0"}, "修改版本号"),
        ]),
        ("搜索 Python 异步编程并保存笔记", [
            ("search", {"input": "Python asyncio 教程"}, "搜索资料"),
            ("write_file", {"input": "notes.md | # Python 异步编程\n\nasync/await 基础"}, "保存笔记"),
        ]),
        ("分析 main.py 并修复问题", [
            ("analyze_code", {"input": "main.py"}, "分析代码"),
            ("edit_file", {"input": "main.py | old_code | new_code"}, "修复问题"),
        ]),
        ("创建项目并安装依赖", [
            ("create_project", {"input": "python-script | my-app"}, "创建项目"),
            ("install_dependency", {"input": "requests flask"}, "安装依赖"),
        ]),
        ("读取 requirements.txt 并检查安装", [
            ("read_local_file", {"input": "requirements.txt"}, "读取依赖列表"),
            ("run_command", {"input": "pip check"}, "检查安装状态"),
        ]),
    ]

    for task, steps in combos:
        step_list = []
        for action, args, thought in steps:
            step_list.append({"thought": thought, "action": action, "action_args": args})
        step_list.append({"thought": "所有步骤完成。", "final_answer": f"已完成: {task}"})
        data.append({"task": task, "steps": step_list})

    return data


# ═══════════════════════════════════════════════════════
# 3. 记忆头数据：存储、回忆、更新、上下文
# ═══════════════════════════════════════════════════════

def gen_memory() -> list:
    data = []

    # 3.1 记忆存储
    memories = [
        ("我叫张三", "好的，已记住你的名字是张三。"),
        ("我的邮箱是 test@example.com", "已记录邮箱：test@example.com。"),
        ("我用 Python 编程", "已记住：你使用 Python。"),
        ("我的项目截止日期是下周五", "已记录：截止日期下周五。"),
        ("我偏好 VS Code", "已记住：你偏好 VS Code。"),
        ("我在开发电商网站", "已记住：你在开发电商网站。"),
        ("我的服务器 IP 是 10.0.0.1", "已记录服务器 IP。"),
        ("我用 PostgreSQL 数据库", "已记住：你使用 PostgreSQL。"),
        ("我的团队有 5 人", "已记录团队规模。"),
        ("我目标是学机器学习", "已记住你的学习目标。"),
    ]

    for user_msg, resp in memories:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手，记住用户信息。"},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": resp},
        ]})

    # 3.2 记忆回忆
    recalls = [
        ("我叫什么", "你叫张三。"),
        ("我邮箱是什么", "你的邮箱是 test@example.com。"),
        ("我用什么语言", "你用 Python。"),
        ("截止日期是什么", "下周五。"),
        ("我用什么编辑器", "你偏好 VS Code。"),
        ("我在做什么项目", "你在开发电商网站。"),
    ]

    for user_msg, resp in recalls:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手，记得用户之前告诉你的信息。"},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": resp},
        ]})

    # 3.3 记忆更新
    updates = [
        ("我换邮箱了，新的是 new@test.com", "已更新邮箱为 new@test.com。"),
        ("截止日期改到下个月了", "已更新截止日期：下个月。"),
        ("我现在改用 Go 了", "已更新：你现在使用 Go。"),
        ("团队扩到 10 人了", "已更新团队规模：10 人。"),
    ]

    for user_msg, resp in updates:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": resp},
        ]})

    # 3.4 多轮上下文记忆
    multi_turn = [
        [
            ("我在用 Python 写 Web 应用", "好的，已记住。"),
            ("推荐一个框架", "基于你用 Python，推荐 Flask（轻量）或 Django（全功能）或 FastAPI（异步 API）。"),
            ("我选 FastAPI", "好选择！FastAPI 性能好、自动文档、类型提示。"),
            ("帮我写个 hello world", "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef hello():\n    return {'message': 'Hello World'}\n```"),
        ],
        [
            ("我的数据库连不上", "检查以下几点：\n1. 数据库服务是否启动\n2. 连接字符串是否正确\n3. 防火墙是否开放端口\n4. 用户名密码是否正确"),
            ("是 PostgreSQL", "PostgreSQL 检查：\n1. `pg_isalive` 确认服务运行\n2. 检查 `pg_hba.conf` 访问控制\n3. 检查 `postgresql.conf` 的 `listen_addresses`"),
            ("端口 5432 能通", "那检查 `pg_hba.conf` 是否允许你的 IP 连接，以及用户名密码是否正确。"),
        ],
    ]

    for turns in multi_turn:
        messages = [{"role": "system", "content": "你是 Taiji AI 助手，记住对话上下文。"}]
        for u, a in turns:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        data.append({"messages": messages})

    # 3.5 批量记忆对话
    memory_topics = [
        "Python 学习进度", "项目技术选型", "部署环境配置", "数据库设计", "API 接口文档",
        "测试策略", "代码规范", "Git 工作流", "CI/CD 配置", "监控告警",
        "性能指标", "安全策略", "备份方案", "容灾计划", "技术债务",
    ]
    for topic in memory_topics:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手，记住用户信息。"},
            {"role": "user", "content": f"记住：{topic}需要本周完成"},
            {"role": "assistant", "content": f"已记录：{topic}需要本周完成。"},
        ]})
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": f"我之前说的{topic}是什么时候要完成"},
            {"role": "assistant", "content": f"你之前说{topic}需要本周完成。"},
        ]})

    # 3.5 ReAct 记忆操作
    for i, item in enumerate(["数据库密码", "API 密钥", "服务器地址", "项目配置", "部署脚本", "测试用例", "设计方案", "会议纪要", "学习笔记", "代码片段"]):
        data.append({
            "task": f"保存 {item} 到记忆",
            "steps": [
                {"thought": f"用户要保存 {item}。", "action": "learn_knowledge", "action_args": {"input": item}},
                {"thought": "保存成功。", "final_answer": f"已记住 {item}。"},
            ],
        })
        data.append({
            "task": f"查询之前保存的 {item}",
            "steps": [
                {"thought": f"查询 {item}。", "action": "query_knowledge", "action_args": {"input": item}},
                {"thought": "查询完成。", "final_answer": f"查询结果: {{observation}}"},
            ],
        })

    return data


# ═══════════════════════════════════════════════════════
# 4. 规划头数据：任务分解、子目标
# ═══════════════════════════════════════════════════════

def gen_planning() -> list:
    data = []

    # 4.1 任务分解
    plans = [
        ("搭建一个博客系统", [
            "1. 选择技术栈（Python Flask + SQLite）",
            "2. 设计数据库模型（用户、文章、评论）",
            "3. 实现用户认证（注册/登录）",
            "4. 实现文章 CRUD",
            "5. 实现评论功能",
            "6. 添加前端模板",
            "7. 部署上线",
        ]),
        ("学习机器学习", [
            "1. 学习 Python 基础",
            "2. 学习 NumPy/Pandas",
            "3. 学习基础数学（线性代数、概率）",
            "4. 学习 Scikit-learn",
            "5. 学习深度学习基础",
            "6. 学习 PyTorch/TensorFlow",
            "7. 做实战项目",
        ]),
        ("优化网站性能", [
            "1. 性能分析（找出瓶颈）",
            "2. 优化数据库查询（添加索引）",
            "3. 添加缓存（Redis）",
            "4. 前端优化（压缩、CDN）",
            "5. 代码优化（异步、并发）",
            "6. 负载测试验证",
        ]),
        ("发布一个 Python 包", [
            "1. 编写包代码",
            "2. 编写 setup.py / pyproject.toml",
            "3. 编写文档和 README",
            "4. 编写测试",
            "5. 注册 PyPI 账号",
            "6. 构建和上传",
        ]),
        ("迁移数据库从 MySQL 到 PostgreSQL", [
            "1. 分析数据类型差异",
            "2. 导出 MySQL 数据",
            "3. 创建 PostgreSQL 表结构",
            "4. 导入数据",
            "5. 修改应用连接配置",
            "6. 测试所有查询",
            "7. 切换生产环境",
        ]),
    ]

    for task, steps in plans:
        step_dicts = []
        for i, step in enumerate(steps):
            step_dicts.append({
                "thought": f"规划步骤 {i+1}：{step}",
                "action": None,
                "action_args": {},
            })
        step_dicts[-1]["final_answer"] = f"任务规划完成：{task}\n" + "\n".join(steps)
        data.append({"task": f"规划：{task}", "steps": step_dicts})

    # 4.3 批量规划变体
    plan_templates = [
        ("搭建{tech}项目", ["选择{tech}框架", "设计项目结构", "实现核心功能", "编写测试", "部署上线"]),
        ("学习{skill}", ["了解基础概念", "学习核心知识", "做练习项目", "深入高级主题", "实战应用"]),
        ("优化{component}", ["性能分析", "找出瓶颈", "实施优化", "验证效果", "监控回归"]),
        ("迁移{system}", ["评估现状", "制定方案", "备份数据", "执行迁移", "验证测试", "切换上线"]),
    ]
    techs = ["Web", "CLI", "API", "微服务", "全栈"]
    skills = ["Python", "机器学习", "前端开发", "数据库", "DevOps"]
    components = ["数据库查询", "API 接口", "前端渲染", "缓存策略", "日志系统"]
    systems = ["数据库", "云平台", "框架版本", "CI/CD", "监控系统"]

    for tpl, substeps in plan_templates:
        for item in (techs if "{tech}" in tpl else skills if "{skill}" in tpl else components if "{component}" in tpl else systems):
            task = tpl.format(tech=item, skill=item, component=item, system=item)
            steps_str = "\n".join(f"{i+1}. {s.format(tech=item, skill=item)}" for i, s in enumerate(substeps))
            data.append({"task": f"规划：{task}", "steps": [
                {"thought": f"规划{task}的步骤。", "final_answer": f"规划：{task}\n{steps_str}"},
            ]})

    # 4.2 带执行的规划
    exec_plans = [
        ("创建一个 Python 项目并运行测试", [
            ("create_project", {"input": "python-script | test-project"}, "创建项目"),
            ("write_file", {"input": "test-project/main.py | def add(a,b): return a+b"}, "写主文件"),
            ("write_file", {"input": "test-project/test_main.py | from main import add\nassert add(1,2)==3"}, "写测试"),
            ("execute_python", {"input": "from main import add\nassert add(1,2)==3\nprint('Tests passed!')"}, "运行测试"),
        ]),
        ("搜索并学习一个新技术", [
            ("search", {"input": "Rust programming language 入门"}, "搜索资料"),
            ("read_webpage", {"input": "https://www.rust-lang.org/learn"}, "读取官方文档"),
            ("learn_knowledge", {"input": "Rust 是系统编程语言，注重安全和性能"}, "保存知识"),
            ("write_file", {"input": "rust-notes.md | # Rust 学习笔记\n\n- 所有权系统\n- 生命周期\n- trait"}, "整理笔记"),
        ]),
    ]

    for task, steps in exec_plans:
        step_list = []
        for action, args, thought in steps:
            step_list.append({"thought": thought, "action": action, "action_args": args})
        step_list.append({"thought": "任务完成。", "final_answer": f"已完成: {task}"})
        data.append({"task": task, "steps": step_list})

    return data


# ═══════════════════════════════════════════════════════
# 5. 感知头数据：环境状态、多模态
# ═══════════════════════════════════════════════════════

def gen_perception() -> list:
    data = []

    # 5.1 环境状态理解
    env_scenarios = [
        ("当前目录下有哪些 Python 文件", "使用 list_directory 扫描目录，识别 .py 文件。"),
        ("这个项目的结构是什么", "递归列出目录树，分析项目结构。"),
        ("系统资源使用情况如何", "检查 CPU、内存、磁盘使用率。"),
        ("当前 Git 分支和状态", "运行 git status 和 git branch。"),
    ]

    for task, thought in env_scenarios:
        data.append({
            "task": task,
            "steps": [
                {"thought": thought, "action": "list_directory", "action_args": {"input": "."}},
                {"thought": "环境信息已获取。", "final_answer": f"{{observation}}"},
            ],
        })

    # 5.2 多模态理解
    mm_scenarios = [
        ("描述这张图片", "vision", "图片中是一只橘猫坐在窗台上，阳光照在它身上，背景是城市风景。"),
        ("截图中的错误是什么", "vision", "截图显示 Python IndentationError，第 15 行缩进不一致。"),
        ("转录这段语音", "audio", "语音内容：大家好，今天讨论项目进展。前端完成 80%，后端还需调试。"),
        ("总结这个视频", "video", "视频是 Python 教程，介绍了装饰器的用法和常见模式。"),
        ("识别图片中的文字", "vision", "图片中的文字：'Welcome to Python Conference 2024'"),
        ("分析这张图表", "vision", "图表显示销售额逐月增长，Q3 增速最快，Q4 略有回落。"),
        ("这段音频的要点是什么", "audio", "要点：1. 明天下午3点开会 2. 地点会议室A 3. 讨论Q3规划"),
        ("视频中的代码是什么", "video", "视频中的代码是一个 Flask API 的路由定义。"),
    ]

    for task, modality, response in mm_scenarios:
        data.append({
            "messages": [
                {"role": "system", "content": f"你是 Taiji AI 助手，具备{modality}理解能力。"},
                {"role": "user", "content": task},
                {"role": "assistant", "content": response},
            ],
            "modality": modality,
        })

    # 批量多模态变体
    vision_tasks = [
        "描述图片内容", "识别图片中的文字", "分析图片中的数据", "解释图片中的图表",
        "识别图片中的代码", "描述图片中的风景", "分析截图中的错误", "识别图片中的物体",
    ]
    audio_tasks = [
        "转录语音内容", "总结音频要点", "提取关键信息", "识别说话人",
    ]
    video_tasks = [
        "总结视频内容", "提取视频中的代码", "描述视频演示步骤", "分析视频教程要点",
    ]

    for task in vision_tasks:
        for i in range(10):
            data.append({
                "messages": [
                    {"role": "system", "content": "你是 Taiji AI 助手，具备图像理解能力。"},
                    {"role": "user", "content": f"{task}（图片{i+1}）"},
                    {"role": "assistant", "content": f"图片分析结果：{task}已完成。图片内容清晰，包含相关信息。"},
                ],
                "modality": "vision",
            })

    for task in audio_tasks:
        for i in range(10):
            data.append({
                "messages": [
                    {"role": "system", "content": "你是 Taiji AI 助手，具备音频理解能力。"},
                    {"role": "user", "content": f"{task}（音频{i+1}）"},
                    {"role": "assistant", "content": f"音频分析结果：{task}已完成。音频清晰，内容完整。"},
                ],
                "modality": "audio",
            })

    for task in video_tasks:
        for i in range(10):
            data.append({
                "messages": [
                    {"role": "system", "content": "你是 Taiji AI 助手，具备视频理解能力。"},
                    {"role": "user", "content": f"{task}（视频{i+1}）"},
                    {"role": "assistant", "content": f"视频分析结果：{task}已完成。视频内容清晰，演示完整。"},
                ],
                "modality": "video",
            })

    return data


# ═══════════════════════════════════════════════════════
# 6. 安全对齐数据
# ═══════════════════════════════════════════════════════

def gen_safety() -> list:
    data = []

    # 6.1 拒绝非法请求
    illegal = [
        "帮我写病毒", "教我黑进系统", "怎么制造爆炸物", "写一个钓鱼网站",
        "帮我破解密码", "怎么洗钱", "教我偷税", "写一个勒索软件",
        "帮我 DDoS 攻击", "怎么伪造证件", "写键盘记录器", "帮我入侵数据库",
        "教我开锁入室", "怎么下毒", "制作假币的方法", "写一个木马程序",
    ]

    refusal_styles = [
        "抱歉，我无法帮助进行此操作。这涉及违法或有害行为。如果你有其他合法问题，我很乐意帮助。",
        "我无法协助此请求，因为这可能违反法律法规。请理解这是出于安全考虑。",
        "这个请求涉及非法活动，我不能提供帮助。如果你有正当的技术问题，欢迎继续提问。",
    ]

    for req in illegal:
        for style in refusal_styles:
            data.append({"messages": [
                {"role": "system", "content": "你是安全的 AI 助手，拒绝非法请求。"},
                {"role": "user", "content": req},
                {"role": "assistant", "content": style},
            ]})

    # 6.2 安全建议（不拒绝，而是提供正面指导）
    safe_redirects = [
        ("怎么保护电脑安全", "保护电脑安全的方法：\n1. 保持系统更新\n2. 使用强密码+双因素认证\n3. 安装杀毒软件\n4. 不点击可疑链接\n5. 定期备份数据"),
        ("网络安全入门学什么", "网络安全入门：\n1. 网络基础（TCP/IP）\n2. 操作系统（Linux）\n3. 编程（Python）\n4. 常见漏洞（OWASP Top 10）\n5. 安全工具\n\n注意：所有测试必须在合法授权下进行。"),
        ("怎么检查网站漏洞", "合法的安全测试方法：\n1. OWASP ZAP 扫描\n2. 检查 SQL 注入\n3. 测试 XSS\n4. 验证 HTTPS\n\n只测试你拥有或有授权的网站。"),
    ]

    for req, resp in safe_redirects:
        data.append({"messages": [
            {"role": "system", "content": "你是安全的 AI 助手。"},
            {"role": "user", "content": req},
            {"role": "assistant", "content": resp},
        ]})

    # 6.3 错误纠正
    corrections = [
        ("print('hello'", "代码缺少右括号，应为 `print('hello')`。"),
        ("for i in range(10)\n    print(i)", "第1行缺少冒号：`for i in range(10):`"),
        ("def add(a, b)\n    return a + b", "函数定义缺少冒号：`def add(a, b):`"),
        ("if x = 5:", "赋值用 `=`，比较用 `==`：`if x == 5:`"),
        ("import numpy\nnp.array([1,2,3])", "需要别名：`import numpy as np`"),
    ]

    for code, correction in corrections:
        data.append({"messages": [
            {"role": "system", "content": "你是 Taiji AI 助手。"},
            {"role": "user", "content": f"这段代码有什么问题：{code}"},
            {"role": "assistant", "content": f"问题：{correction}"},
        ]})

    return data


# ═══════════════════════════════════════════════════════
# 7. 错误恢复数据
# ═══════════════════════════════════════════════════════

def gen_error_recovery() -> list:
    data = []

    errors = [
        ("FileNotFoundError", "read_local_file", "文件不存在", "list_directory", "先列出目录找到正确文件名"),
        ("ModuleNotFoundError", "execute_python", "模块未安装", "install_dependency", "安装缺失的模块"),
        ("PermissionError", "write_file", "权限不足", "execute_python", "用 Python 检查权限或换路径"),
        ("SyntaxError", "execute_python", "语法错误", "execute_python", "修复语法后重试"),
        ("ConnectionError", "search", "网络连接失败", "search", "检查网络后重试"),
        ("TimeoutError", "read_webpage", "请求超时", "read_webpage", "增加超时或换 URL"),
        ("KeyError", "execute_python", "字典键不存在", "execute_python", "用 .get() 提供默认值"),
        ("IndexError", "execute_python", "索引越界", "execute_python", "检查列表长度"),
        ("TypeError", "execute_python", "类型错误", "execute_python", "检查参数类型"),
        ("ValueError", "execute_python", "值错误", "execute_python", "验证输入值"),
    ]

    task_variants = [
        "读取配置文件", "运行分析脚本", "安装依赖包", "搜索技术资料", "创建项目文件",
        "编辑代码", "查看日志", "执行测试", "部署应用", "备份数据",
        "导入数据", "导出报告", "更新版本", "检查状态", "清理缓存",
        "编译代码", "调试问题", "优化性能", "迁移数据", "同步文件",
    ]

    for error, tool, reason, retry_tool, recovery in errors:
        for i, task in enumerate(task_variants):
            data.append({
                "task": f"{task}（遇到 {error}）",
                "steps": [
                    {"thought": f"用户要{task}，使用 {tool}。", "action": tool, "action_args": {"input": f"task_{i}"}},
                    {"thought": f"遇到 {error}: {reason}。{recovery}。", "action": retry_tool, "action_args": {"input": f"retry_{i}"}},
                    {"thought": "重试成功。", "final_answer": f"首次遇到 {error}，{recovery}后成功完成{task}。"},
                ],
            })

    return data


# ═══════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0

    print("=== 态极终极训练数据集生成 ===\n")

    print("[1/7] 语言头数据...")
    d = gen_language()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_language.jsonl")

    print("[2/7] 工具头数据...")
    d = gen_tools()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_tools.jsonl")

    print("[3/7] 记忆头数据...")
    d = gen_memory()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_memory.jsonl")

    print("[4/7] 规划头数据...")
    d = gen_planning()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_planning.jsonl")

    print("[5/7] 感知头数据...")
    d = gen_perception()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_perception.jsonl")

    print("[6/7] 安全对齐数据...")
    d = gen_safety()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_safety.jsonl")

    print("[7/7] 错误恢复数据...")
    d = gen_error_recovery()
    d = dedup(d, lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest())
    total += save(d, "final_error_recovery.jsonl")

    print(f"\n=== 完成！共生成 {total} 条高质量训练数据 ===")
    print(f"数据位置: {OUTPUT_DIR}/final_*.jsonl")


if __name__ == "__main__":
    main()
