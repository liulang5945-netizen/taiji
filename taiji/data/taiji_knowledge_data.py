"""
态极知识训练数据生成器
=====================

为态极生成高质量训练数据，覆盖两大核心能力：
1. 对话能力（~2000条）：自然中文问答、多轮对话、知识问答
2. 工具调用能力（~3000条）：ReAct 格式的完整工具调用链路

使用方式：
    python -m taiji.taiji_knowledge_data --output taiji/training_data --conv 2000 --react 3000

生成的数据可直接用于 `taiji/train/finetune_taiji.py` 的微调流程。
"""
import json
import os
import random
import logging
import argparse
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Taiji.KnowledgeData")


# ============================================================
# Part 1: 对话能力训练数据
# ============================================================

# 知识问答对 — 覆盖广泛领域的基础知识
KNOWLEDGE_QA = [
    # === 编程基础 ===
    {"q": "什么是Python？", "a": "Python 是一种高级、解释型、通用的编程语言，由 Guido van Rossum 于 1991 年发布。它的设计哲学强调代码的可读性和简洁性。Python 支持多种编程范式，包括面向对象、函数式和过程式编程。它拥有丰富的标准库和第三方生态系统，广泛应用于 Web 开发、数据科学、人工智能、自动化脚本等领域。"},
    {"q": "什么是递归？", "a": "递归是一种编程技巧，指函数直接或间接调用自身。递归函数通常包含两个部分：基准条件（base case，即停止递归的条件）和递归步骤（将问题分解为更小的子问题）。经典的递归例子包括计算阶乘、斐波那契数列和遍历树结构。使用递归时需要注意避免无限递归和栈溢出。"},
    {"q": "什么是API？", "a": "API（Application Programming Interface，应用程序编程接口）是一组定义了软件组件之间交互方式的规则和协议。简单来说，API 就像是一个餐厅的服务员，它接收你的请求（点菜），然后将请求传达给厨房（服务器），最后把结果（食物）返回给你。常见的 API 类型包括 REST API、GraphQL 和 RPC。"},
    {"q": "什么是Git？", "a": "Git 是一个分布式版本控制系统，由 Linus Torvalds 于 2005 年开发，最初用于管理 Linux 内核的源代码。Git 能够追踪文件的每一次修改，支持多人协作开发，允许创建分支进行并行开发，合并代码，以及回退到历史版本。常用命令包括 git clone、git add、git commit、git push、git pull 和 git branch。"},
    {"q": "什么是Docker？", "a": "Docker 是一个开源的容器化平台，它可以将应用程序及其所有依赖项打包成一个标准化的单元（称为容器），确保应用在任何环境中都能一致运行。与虚拟机不同，容器共享宿主机的操作系统内核，因此更加轻量级。Docker 的核心概念包括镜像（Image，只读模板）、容器（Container，运行实例）和 Dockerfile（构建镜像的脚本）。"},
    {"q": "什么是SQL？", "a": "SQL（Structured Query Language，结构化查询语言）是用于管理关系型数据库的标准语言。它可以用来创建、查询、更新和删除数据库中的数据。基本操作包括：SELECT（查询数据）、INSERT（插入数据）、UPDATE（更新数据）、DELETE（删除数据）、CREATE TABLE（创建表）等。常见的关系型数据库包括 MySQL、PostgreSQL、SQLite 和 SQL Server。"},
    {"q": "什么是正则表达式？", "a": "正则表达式（Regular Expression，简称 regex 或 regexp）是一种用于匹配字符串模式的强大工具。它使用特殊字符来定义搜索模式，常用于文本搜索、替换和数据验证。常用符号包括：.（匹配任意字符）、*（零次或多次）、+（一次或多次）、?（零次或一次）、\\d（数字）、\\w（字母数字）、[]（字符集）、^（行首）、$（行尾）。"},

    # === 机器学习/AI ===
    {"q": "什么是机器学习？", "a": "机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习规律和模式，而不需要被显式编程。机器学习主要分为三类：监督学习（从标注数据中学习输入到输出的映射）、无监督学习（发现数据中的隐藏结构）和强化学习（通过与环境交互来学习最优策略）。常见算法包括线性回归、决策树、随机森林、支持向量机和神经网络。"},
    {"q": "什么是深度学习？", "a": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。与传统机器学习不同，深度学习能够自动提取特征，无需人工设计。深度学习的核心架构包括：卷积神经网络（CNN，擅长图像处理）、循环神经网络（RNN，擅长序列数据）、Transformer（擅长并行处理，是 GPT 和 BERT 的基础）。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性成果。"},
    {"q": "什么是神经网络？", "a": "神经网络是一种受生物神经系统启发的计算模型，由大量相互连接的节点（称为神经元）组成。每个神经元接收输入信号，经过加权求和和激活函数处理后产生输出。神经网络通常包含输入层、隐藏层和输出层。通过反向传播算法和梯度下降优化，网络可以学习数据中的模式。深度学习就是使用多层隐藏层的神经网络。"},
    {"q": "什么是Transformer？", "a": "Transformer 是一种深度学习架构，由 Google 在 2017 年的论文《Attention Is All You Need》中提出。它的核心创新是自注意力机制（Self-Attention），能够并行处理序列中的所有位置，捕捉长距离依赖关系。Transformer 由编码器和解码器组成，每层包含多头注意力和前馈网络。它是 GPT、BERT、T5 等现代大语言模型的基础架构。"},
    {"q": "什么是大语言模型？", "a": "大语言模型（Large Language Model，LLM）是一种基于 Transformer 架构的大规模预训练语言模型，通过在海量文本数据上训练来学习语言的统计规律。代表性模型包括 GPT-4、Claude、LLaMA、Qwen 等。LLM 能够进行文本生成、问答、翻译、代码编写等多种任务。它们通常通过预训练+微调+人类反馈强化学习（RLHF）的流程来对齐人类偏好。"},
    {"q": "什么是LoRA？", "a": "LoRA（Low-Rank Adaptation，低秩适应）是一种高效的模型微调技术，由微软在 2021 年提出。它的核心思想是：在冻结原始模型参数的同时，向每一层注入可训练的低秩分解矩阵。这样只需训练极少量参数（通常不到原模型的 1%），就能达到接近全参数微调的效果。LoRA 特别适合在消费级硬件上微调大语言模型。"},

    # === 系统/运维 ===
    {"q": "什么是Linux？", "a": "Linux 是一个免费的开源操作系统内核，由 Linus Torvalds 于 1991 年首次发布。它是类 Unix 操作系统的核心组件，通常与 GNU 工具结合使用（称为 GNU/Linux）。Linux 以稳定、安全和高度可定制著称，广泛应用于服务器、嵌入式设备、超级计算机和 Android 手机。常见的 Linux 发行版包括 Ubuntu、CentOS、Debian、Fedora 和 Arch Linux。"},
    {"q": "什么是容器编排？", "a": "容器编排是指自动化管理、调度和协调容器化应用的过程。当应用由多个微服务容器组成时，手动管理变得不切实际。Kubernetes（K8s）是最流行的容器编排平台，它提供自动部署、负载均衡、自动扩缩容、服务发现、滚动更新和自愈能力。Docker Swarm 是另一个较简单的编排方案。"},

    # === 数学/算法 ===
    {"q": "什么是时间复杂度？", "a": "时间复杂度是衡量算法执行效率的指标，表示算法运行时间随输入规模增长的变化趋势，通常用大 O 表示法。常见的时间复杂度从低到高排列：O(1) 常数时间、O(log n) 对数时间、O(n) 线性时间、O(n log n) 线性对数时间、O(n²) 平方时间、O(2^n) 指数时间。选择合适的数据结构和算法可以显著降低时间复杂度。"},
    {"q": "什么是快速排序？", "a": "快速排序（Quick Sort）是一种高效的排序算法，采用分治策略。基本思想是：选择一个基准元素（pivot），将数组分为两部分——小于基准的元素放左边，大于基准的元素放右边，然后对两部分递归排序。平均时间复杂度为 O(n log n)，最坏情况为 O(n²)。快速排序是实际应用中最常用的排序算法之一，因为它的常数因子较小，缓存命中率高。"},

    # === 日常对话 ===
    {"q": "你好，你是谁？", "a": "你好！我是态极，一个本地运行的 AI 助手。我可以帮你回答问题、搜索信息、读写文件、执行代码、学习新知识等。有什么我可以帮你的吗？"},
    {"q": "你能做什么？", "a": "我可以帮你做很多事情！主要包括：\n\n1. **知识问答**：回答各种领域的问题\n2. **文件操作**：读取、创建、编辑、删除文件\n3. **代码执行**：运行 Python 代码并返回结果\n4. **网络搜索**：搜索互联网获取最新信息\n5. **知识学习**：自主学习新领域知识并存储\n6. **网页阅读**：读取网页内容并总结\n7. **项目创建**：帮你搭建项目脚手架\n\n有什么需要帮忙的，尽管说！"},
    {"q": "谢谢你的帮助", "a": "不客气！很高兴能帮到你。如果以后有任何问题，随时可以找我。祝你一切顺利！"},
    {"q": "今天天气怎么样？", "a": "抱歉，我是一个本地 AI 助手，无法直接获取实时天气信息。不过你可以告诉我你所在的城市，我可以帮你搜索一下天气预报。"},
    {"q": "推荐一本编程入门书", "a": "推荐几本经典的编程入门书籍：\n\n1. **《Python编程：从入门到实践》**：非常适合零基础，项目驱动学习\n2. **《Head First Java》**：图文并茂，适合 Java 入门\n3. **《C Primer Plus》**：C 语言经典教材\n4. **《JavaScript高级程序设计》**：前端开发必读\n5. **《算法图解》**：用图解方式讲解算法，通俗易懂\n\n如果你是完全的新手，建议从 Python 开始，因为它语法简洁，上手快，应用范围广。"},
    {"q": "解释一下面向对象编程", "a": "面向对象编程（OOP）是一种编程范式，它将数据和操作数据的方法组织成'对象'。OOP 有四大核心概念：\n\n1. **封装**：将数据和方法包装在一起，隐藏内部实现细节\n2. **继承**：子类可以继承父类的属性和方法，实现代码复用\n3. **多态**：同一个接口可以有不同的实现方式\n4. **抽象**：提取共同特征，忽略不重要的细节\n\nOOP 的好处是代码更易维护、可复用、易扩展。Python、Java、C++ 都支持面向对象编程。"},
    {"q": "什么是REST API？", "a": "REST（Representational State Transfer）是一种 API 设计风格，基于 HTTP 协议。它的核心原则包括：\n\n1. 使用 URL 表示资源，如 `/users/123` 表示 ID 为 123 的用户\n2. 使用 HTTP 方法表示操作：GET（获取）、POST（创建）、PUT（更新）、DELETE（删除）\n3. 无状态：每个请求包含所有必要信息\n4. 使用 JSON 或 XML 格式传输数据\n\nREST API 简单、灵活、易于理解，是目前最流行的 Web API 设计方式。"},
    {"q": "什么是WebSocket？", "a": "WebSocket 是一种在单个 TCP 连接上进行全双工通信的协议。与传统的 HTTP 请求-响应模式不同，WebSocket 允许服务器和客户端在建立连接后随时互相发送数据，无需反复建立连接。这使得 WebSocket 非常适合实时应用，如在线聊天、实时股票行情、多人游戏、协作编辑等场景。"},
    {"q": "解释一下HTTPS和HTTP的区别", "a": "HTTPS 是 HTTP 的安全版本，主要区别在于：\n\n1. **安全性**：HTTPS 使用 TLS/SSL 加密传输数据，HTTP 明文传输\n2. **端口**：HTTP 默认端口 80，HTTPS 默认端口 443\n3. **证书**：HTTPS 需要 SSL 证书\n4. **性能**：HTTPS 因为加密会有轻微的性能开销\n5. **SEO**：搜索引擎优先收录 HTTPS 网站\n\n现在几乎所有网站都应该使用 HTTPS，浏览器也会对 HTTP 网站标记为不安全。"},
]

# 多轮对话模板
MULTI_TURN_TEMPLATES = [
    {
        "turns": [
            {"user": "帮我写一个Python函数，计算两个数的最大公约数", "assistant": "好的，这是一个计算最大公约数的 Python 函数，使用欧几里得算法（辗转相除法）：\n\n```python\ndef gcd(a, b):\n    \"\"\"计算两个数的最大公约数\"\"\"\n    while b:\n        a, b = b, a % b\n    return a\n\n# 测试\nprint(gcd(12, 8))  # 输出: 4\nprint(gcd(100, 75))  # 输出: 25\n```\n\n这个算法的时间复杂度是 O(log(min(a, b)))，非常高效。"},
            {"user": "如果要计算多个数的最大公约数呢？", "assistant": "可以利用 gcd(a, b, c) = gcd(gcd(a, b), c) 的性质，对多个数依次求最大公约数：\n\n```python\nfrom functools import reduce\nfrom math import gcd\n\ndef gcd_multiple(*numbers):\n    \"\"\"计算多个数的最大公约数\"\"\"\n    return reduce(gcd, numbers)\n\n# 测试\nprint(gcd_multiple(12, 8, 6))  # 输出: 2\nprint(gcd_multiple(100, 75, 50))  # 输出: 25\nprint(gcd_multiple(24, 36, 48, 60))  # 输出: 12\n```\n\n这里用 `functools.reduce` 将 gcd 函数依次应用到列表中的每个数上，效果等同于 `gcd(gcd(gcd(12, 8), 6))`。"},
            {"user": "谢谢！这个方法很优雅", "assistant": "不客气！Python 的 `functools.reduce` 和标准库 `math.gcd` 配合使用确实很简洁优雅。如果你还有其他问题，随时问我！"},
        ]
    },
    {
        "turns": [
            {"user": "什么是虚拟环境？为什么要用它？", "assistant": "虚拟环境（Virtual Environment）是 Python 的一个工具，用于创建独立的 Python 运行环境。\n\n**为什么需要虚拟环境？**\n\n假设你有两个项目：项目 A 需要 Django 3.2，项目 B 需要 Django 4.1。如果没有虚拟环境，它们会共用同一个全局的 Django 版本，必然冲突。虚拟环境让每个项目拥有独立的包目录，互不影响。\n\n**创建和使用：**\n```bash\n# 创建虚拟环境\npython -m venv myenv\n\n# 激活（Windows）\nmyenv\\Scripts\\activate\n\n# 激活（Linux/Mac）\nsource myenv/bin/activate\n\n# 安装包（只在这个环境中生效）\npip install django==4.1\n\n# 退出虚拟环境\ndeactivate\n```"},
            {"user": "conda 和 venv 有什么区别？", "assistant": "两者都是 Python 环境管理工具，但有一些重要区别：\n\n**venv：**\n- Python 内置，无需额外安装\n- 只管理 Python 包\n- 轻量级，创建速度快\n- 使用 pip 安装包\n\n**conda：**\n- 需要安装 Anaconda 或 Miniconda\n- 可以管理 Python 版本本身（比如在 Python 3.8 和 3.11 之间切换）\n- 可以管理非 Python 依赖（如 C 库）\n- 有自己的包管理器，也可以用 pip\n- 更适合数据科学场景\n\n**选择建议：**\n- 纯 Python 项目 → 用 venv\n- 数据科学/需要特定 Python 版本 → 用 conda"},
        ]
    },
    {
        "turns": [
            {"user": "我想学机器学习，应该怎么开始？", "assistant": "学习机器学习的建议路线：\n\n**第一步：打好基础**\n- Python 编程基础\n- 线性代数（矩阵运算）\n- 概率统计基础\n- 微积分（求导、梯度）\n\n**第二步：入门机器学习**\n- 学习 Scikit-learn 库\n- 理解常见算法：线性回归、决策树、SVM、KNN\n- 做经典项目：手写数字识别、房价预测\n\n**第三步：深入深度学习**\n- 学习 PyTorch 或 TensorFlow\n- 理解 CNN、RNN、Transformer\n- 做实战项目\n\n**推荐资源：**\n- 吴恩达的 Coursera 课程\n- 《机器学习实战》\n- Kaggle 竞赛平台"},
            {"user": "需要很强的数学基础吗？", "assistant": "入门阶段不需要很强的数学基础，但需要了解一些核心概念：\n\n**必须掌握的：**\n- 线性代数：矩阵乘法、转置、逆矩阵\n- 微积分：导数、偏导数、链式法则\n- 概率统计：均值、方差、贝叶斯定理\n\n**好消息是：**\n1. 可以边学边补，不必先学完所有数学\n2. 很多框架（如 PyTorch）会自动处理微分\n3. 实际编程中大部分数学已被封装好\n\n**建议：**\n先入门机器学习编程，遇到不理解的数学概念再针对性学习。3Blue1Brown 的 YouTube 系列视频对理解线性代数和微积分非常有帮助。"},
        ]
    },
]


def generate_conversation_sample(knowledge_qa: Dict) -> Dict:
    """将知识问答对转化为训练样本"""
    system_prompt = random.choice([
        "你是态极，一个智能的本地 AI 助手。你会用自然、友好的方式回答用户的问题。",
        "你是态极 AI 助手，擅长用通俗易懂的语言解释复杂概念。",
        "你是态极，一个乐于助人的 AI 助手，知识渊博且表达清晰。",
    ])
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": knowledge_qa["q"]},
            {"role": "assistant", "content": knowledge_qa["a"]},
        ]
    }


def generate_multi_turn_sample(template: Dict) -> Dict:
    """将多轮对话模板转化为训练样本"""
    system_prompt = random.choice([
        "你是态极，一个智能的本地 AI 助手。你会用自然、友好的方式回答用户的问题。",
        "你是态极 AI 助手，擅长用通俗易懂的语言解释复杂概念。",
    ])
    messages = [{"role": "system", "content": system_prompt}]
    for turn in template["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {"messages": messages}


def generate_all_conversations(target_count: int = 2000) -> List[Dict]:
    """生成所有对话训练数据"""
    samples = []

    # 1. 知识问答（每条可重复使用不同 system prompt 增加多样性）
    for qa in KNOWLEDGE_QA:
        samples.append(generate_conversation_sample(qa))

    # 2. 多轮对话
    for template in MULTI_TURN_TEMPLATES:
        samples.append(generate_multi_turn_sample(template))

    # 3. 补充通用问答变体（增加数据量）
    generic_qa = [
        {"q": "用简单的语言解释一下什么是面向切面编程（AOP）", "a": "面向切面编程（AOP）是一种编程思想，用于处理那些分散在代码各处但不属于核心业务逻辑的功能，比如日志记录、权限检查、事务管理等。\n\n想象一下：你写了很多函数，每个函数都需要记录日志。如果没有 AOP，你得在每个函数里都加上日志代码。有了 AOP，你可以把'记录日志'这个功能抽出来，声明式地应用到所有需要的地方，核心代码保持干净。"},
        {"q": "什么是微服务架构？", "a": "微服务架构是一种将应用拆分为一组小型、独立服务的架构风格。每个微服务：\n\n1. 负责一个特定的业务功能\n2. 可以独立开发、测试、部署\n3. 使用自己的数据库\n4. 通过 API 与其他服务通信\n\n优点是灵活性高、技术栈自由、可独立扩缩容。缺点是分布式系统复杂度增加、调试困难、需要服务治理。"},
        {"q": "Git 的 rebase 和 merge 有什么区别？", "a": "**merge**：将两个分支的修改合并，创建一个新的合并提交。保留完整的分支历史。\n\n```\nA-B-C (main)\n   \\\n    D-E (feature)\n     \\\n      F (merge commit)\n```\n\n**rebase**：将 feature 分支的提交'移动'到 main 分支的最新提交之后，创建线性历史。\n\n```\nA-B-C-D'-E' (main + feature)\n```\n\n**选择建议：**\n- 团队协作中，公共分支用 merge，个人分支用 rebase\n- rebase 保持历史整洁，但会改写提交历史\n- 永远不要 rebase 已经推送到远程的公共分支"},
        {"q": "什么是设计模式？", "a": "设计模式是软件开发中经过验证的、解决常见问题的通用方案。它不是具体的代码，而是一种思想和模板。\n\n最常用的 23 种设计模式分为三类：\n\n**创建型**（如何创建对象）：\n- 单例模式：确保只有一个实例\n- 工厂模式：统一创建对象的接口\n- 建造者模式：分步构建复杂对象\n\n**结构型**（如何组合对象）：\n- 适配器模式：让不兼容的接口协同工作\n- 代理模式：控制对象的访问\n- 装饰器模式：动态添加功能\n\n**行为型**（对象间如何通信）：\n- 观察者模式：一对多的通知机制\n- 策略模式：可互换的算法\n- 模板方法模式：定义算法骨架"},
        {"q": "前端和后端有什么区别？", "a": "**前端**是用户直接看到和交互的部分，包括网页的布局、样式、动画和交互逻辑。主要技术是 HTML、CSS、JavaScript，常用框架有 React、Vue、Angular。\n\n**后端**是服务器端的逻辑，处理数据存储、业务规则、API 接口等。常用语言有 Python、Java、Node.js、Go，常用框架有 Django、Spring Boot、Express。\n\n简单比喻：前端是餐厅的装修和菜单，后端是厨房和仓库。用户在前端点菜（操作界面），后端处理订单并返回结果（数据和逻辑）。"},
        {"q": "什么是NoSQL数据库？", "a": "NoSQL（Not Only SQL）是非关系型数据库的统称，与传统的关系型数据库（如 MySQL）不同，它不使用固定的表结构。\n\n常见类型：\n1. **文档型**（MongoDB）：存储 JSON 格式的文档\n2. **键值型**（Redis）：简单的 key-value 存储，速度极快\n3. **列族型**（Cassandra）：适合大规模数据分析\n4. **图数据库**（Neo4j）：存储实体间的关系\n\n适用场景：数据结构灵活多变、需要高并发读写、海量数据存储。"},
        {"q": "什么是负载均衡？", "a": "负载均衡是将网络流量分配到多个服务器的技术，目的是：\n\n1. **提高可用性**：一台服务器挂了，流量自动转到其他服务器\n2. **提高性能**：多台服务器分担压力\n3. **便于扩展**：增加服务器即可提升处理能力\n\n常见算法：轮询（依次分配）、最少连接（分配给最空闲的服务器）、IP 哈希（同一 IP 始终分配到同一服务器）。\n\n常用工具：Nginx、HAProxy、AWS ELB。"},
        {"q": "什么是消息队列？", "a": "消息队列（Message Queue，MQ）是一种异步通信机制，允许应用程序通过发送和接收消息来通信，而不需要直接调用对方。\n\n工作原理：发送者将消息放入队列，接收者从队列中取出消息处理。\n\n典型应用场景：\n1. **异步处理**：用户下单后，不等库存扣减和通知完成就返回\n2. **流量削峰**：秒杀场景下，先将请求放入队列，后台慢慢处理\n3. **服务解耦**：生产者和消费者互不依赖\n\n常见消息队列：RabbitMQ、Kafka、RocketMQ、Redis Stream。"},
        {"q": "什么是CI/CD？", "a": "CI/CD 是持续集成/持续部署的缩写：\n\n**CI（持续集成）**：开发者频繁地将代码合并到主分支，每次合并后自动运行测试，尽早发现问题。\n\n**CD（持续交付/持续部署）**：\n- 持续交付：代码通过测试后，自动准备好部署，但需要手动审批发布\n- 持续部署：代码通过测试后，自动部署到生产环境\n\n常用工具：GitHub Actions、GitLab CI、Jenkins、CircleCI。\n\n好处：减少人工错误、加快发布速度、提高代码质量。"},
        {"q": "什么是依赖注入？", "a": "依赖注入（Dependency Injection，DI）是一种设计模式，核心思想是：对象不自己创建依赖，而是由外部传入。\n\n```python\n# 不使用依赖注入\nclass UserService:\n    def __init__(self):\n        self.db = MySQLDatabase()  # 硬编码依赖\n\n# 使用依赖注入\nclass UserService:\n    def __init__(self, db):  # 依赖从外部注入\n        self.db = db\n\n# 使用时\ndb = MySQLDatabase()\nuser_service = UserService(db)\n```\n\n好处：方便测试（可以注入 Mock 对象）、降低耦合、提高灵活性。"},
        {"q": "解释一下区块链", "a": "区块链是一种分布式账本技术，核心特点是：\n\n1. **去中心化**：没有中央机构控制，数据分布在所有参与者节点上\n2. **不可篡改**：每个区块包含前一个区块的哈希值，修改一个区块需要修改后面所有区块\n3. **透明可验证**：所有交易记录公开可查\n4. **共识机制**：通过工作量证明（PoW）或权益证明（PoS）等机制达成一致\n\n应用：加密货币（比特币、以太坊）、智能合约、供应链追溯、数字身份验证。"},
        {"q": "什么是函数式编程？", "a": "函数式编程（FP）是一种编程范式，核心理念是：用纯函数来构建程序，避免可变状态和副作用。\n\n关键概念：\n1. **纯函数**：相同输入总是产生相同输出，不修改外部状态\n2. **不可变性**：数据一旦创建就不能修改\n3. **高阶函数**：函数可以作为参数传递，也可以作为返回值\n4. **函数组合**：将小函数组合成复杂功能\n\nPython 支持函数式编程：map、filter、reduce、lambda、列表推导式都是 FP 特性。"},
        {"q": "什么是OAuth？", "a": "OAuth（Open Authorization）是一种开放标准的授权协议，允许用户在不泄露密码的情况下，授权第三方应用访问自己的资源。\n\n**生活中的例子**：你用微信登录某个 App，这个 App 获得了你的头像和昵称，但拿不到你的微信密码。\n\n**OAuth 2.0 流程**：\n1. 用户在第三方应用点击'用微信登录'\n2. 跳转到微信授权页面\n3. 用户同意授权\n4. 微信返回授权码给第三方应用\n5. 第三方应用用授权码换取访问令牌\n6. 用访问令牌获取用户信息"},
    ]
    for qa in generic_qa:
        samples.append(generate_conversation_sample(qa))

    # 如果不够目标数量，重复采样
    while len(samples) < target_count:
        qa = random.choice(KNOWLEDGE_QA + generic_qa)
        samples.append(generate_conversation_sample(qa))

    random.shuffle(samples)
    return samples[:target_count]


# ============================================================
# Part 2: 工具调用能力训练数据（ReAct 格式）
# ============================================================

# 所有可用工具及其使用说明
TOOL_DEFINITIONS = {
    "read_local_file": {
        "desc": "读取工作台文件内容",
        "args_desc": "文件路径，如 data.txt",
        "example_args": "README.md",
        "thought": "用户想查看文件内容，我需要使用 read_local_file 工具。",
        "sample_result": "# 项目文档\n\n这是一个测试项目。",
    },
    "write_file": {
        "desc": "创建或覆盖写入文件",
        "args_desc": "格式: 文件路径 | 文件内容",
        "example_args": "hello.py | print('Hello World')",
        "thought": "用户想创建一个新文件，我使用 write_file 工具来写入内容。",
        "sample_result": "文件 hello.py 已创建。",
    },
    "edit_file": {
        "desc": "精确编辑文件内容",
        "args_desc": "格式: 文件路径 | 旧文本 | 新文本",
        "example_args": "config.json | \"debug\": true | \"debug\": false",
        "thought": "需要修改文件中的特定内容，使用 edit_file 进行精确替换。",
        "sample_result": "文件已更新，替换了 1 处内容。",
    },
    "delete_file": {
        "desc": "删除文件或空目录",
        "args_desc": "文件路径",
        "example_args": "temp.txt",
        "thought": "用户要求删除文件，使用 delete_file 工具。",
        "sample_result": "文件 temp.txt 已删除。",
    },
    "list_directory": {
        "desc": "列出目录内容",
        "args_desc": "目录路径（可选）",
        "example_args": "src",
        "thought": "用户想查看目录结构，使用 list_directory 工具。",
        "sample_result": "src/\n  main.py\n  utils.py\n  config.json",
    },
    "create_directory": {
        "desc": "创建目录",
        "args_desc": "目录路径",
        "example_args": "src/components",
        "thought": "需要创建目录结构，使用 create_directory。",
        "sample_result": "目录 src/components 已创建。",
    },
    "create_project": {
        "desc": "创建项目脚手架",
        "args_desc": "格式: 项目类型 | 项目名",
        "example_args": "web-app | my-blog",
        "thought": "用户想创建新项目，使用 create_project 工具快速搭建脚手架。",
        "sample_result": "项目 my-blog 已创建，包含 index.html, style.css, main.js",
    },
    "install_dependency": {
        "desc": "安装 Python 依赖包",
        "args_desc": "包名",
        "example_args": "requests",
        "thought": "需要安装依赖包，使用 install_dependency 工具。",
        "sample_result": "已安装 requests 2.31.0",
    },
    "analyze_code": {
        "desc": "分析代码文件语法",
        "args_desc": "文件路径",
        "example_args": "main.py",
        "thought": "用户想了解代码结构，使用 analyze_code 工具分析。",
        "sample_result": "文件分析完成：\n- 函数: 5 个\n- 类: 2 个\n- 导入: 3 个\n- 行数: 120 行\n- 语法错误: 无",
    },
    "execute_python": {
        "desc": "在安全沙箱中执行 Python 代码",
        "args_desc": "Python 代码",
        "example_args": "print(sum(range(1, 101)))",
        "thought": "用户想执行 Python 代码，使用 execute_python 工具在沙箱中运行。",
        "sample_result": "5050",
    },
    "search": {
        "desc": "在互联网上搜索信息",
        "args_desc": "搜索关键词",
        "example_args": "Python 异步编程最佳实践",
        "thought": "用户需要最新的信息，使用 search 工具搜索互联网。",
        "sample_result": "搜索结果：\n1. Python asyncio 官方文档\n2. 异步编程实战指南\n3. FastAPI 异步最佳实践",
    },
    "read_webpage": {
        "desc": "读取网页正文内容",
        "args_desc": "完整 URL",
        "example_args": "https://docs.python.org/3/",
        "thought": "用户想了解某个网页的内容，使用 read_webpage 工具获取。",
        "sample_result": "Python 3 官方文档 - 本教程介绍了 Python 的基本概念和用法...",
    },
    "browse_web": {
        "desc": "用浏览器访问动态网页（支持 JS 渲染）",
        "args_desc": "完整 URL",
        "example_args": "https://example.com",
        "thought": "这个网页是动态渲染的，需要使用 browse_web 工具。",
        "sample_result": "页面内容：\n标题: Example Domain\n正文: This domain is for use in illustrative examples...",
    },
    "smart_fetch": {
        "desc": "智能抓取网页正文（Markdown 格式）",
        "args_desc": "URL 地址",
        "example_args": "https://blog.example.com/article",
        "thought": "需要抓取文章正文，使用 smart_fetch 工具获取 Markdown 格式内容。",
        "sample_result": "# 文章标题\n\n正文内容...\n\n## 第一节\n\n详细内容...",
    },
    "learn_knowledge": {
        "desc": "启动领域知识自学习",
        "args_desc": "格式: 领域名 | 来源URL(可选) | 深度(可选)",
        "example_args": "Python异步编程 | https://docs.python.org | deep",
        "thought": "用户想学习某个领域的知识，使用 learn_knowledge 工具启动自学习。",
        "sample_result": "✅ 学习完成 [Python异步编程]\n采集: 23 条\n新增: 18 | 更新: 3 | 跳过: 2\n验证得分: 85%",
    },
    "query_knowledge": {
        "desc": "查询已学习的知识库",
        "args_desc": "查询问题（可选 | 领域名）",
        "example_args": "什么是协程",
        "thought": "用户想查询之前学过的知识，使用 query_knowledge 工具检索知识库。",
        "sample_result": "关于'协程'的知识：\n协程是 Python 异步编程的核心概念，它是一种可以暂停和恢复的函数...",
    },
    "run_command": {
        "desc": "运行本地命令行命令",
        "args_desc": "命令字符串",
        "example_args": "ls -la",
        "thought": "需要执行系统命令，使用 run_command 工具。",
        "sample_result": "total 48\ndrwxr-xr-x 6 user user 4096 Jun 3 10:00 .\ndrwxr-xr-x 3 user user 4096 Jun 3 09:00 ..\n-rw-r--r-- 1 user user  128 Jun 3 10:00 README.md",
    },
}

# ReAct 任务模板
REACT_TASK_TEMPLATES = [
    # 单工具任务
    {"task": "读取 {file} 文件的内容", "tools": ["read_local_file"], "params": {"file": ["README.md", "config.json", "main.py", "requirements.txt"]}},
    {"task": "列出 {dir} 目录下的所有文件", "tools": ["list_directory"], "params": {"dir": [".", "src", "docs", "agent", "api"]}},
    {"task": "用 Python 计算 {expr}", "tools": ["execute_python"], "params": {"expr": ["1到100的和", "斐波那契数列前20项", "17是否是素数", "当前日期和时间"]}},
    {"task": "搜索 {query}", "tools": ["search"], "params": {"query": ["Python异步编程", "Docker入门教程", "机器学习最新进展", "PyTorch安装指南"]}},
    {"task": "创建 {file} 文件，写入 {content}", "tools": ["write_file"], "params": {"file": ["hello.py", "README.md", "config.json", "test.txt"], "content": ["print('Hello World')", "# 项目说明\n\n这是一个新项目。", '{"debug": true}', "测试内容"]}},
    {"task": "分析 {file} 的代码结构", "tools": ["analyze_code"], "params": {"file": ["main.py", "app.py", "utils.py"]}},
    {"task": "安装 Python 包 {pkg}", "tools": ["install_dependency"], "params": {"pkg": ["requests", "flask", "numpy", "pandas"]}},
    {"task": "读取网页 {url}", "tools": ["read_webpage"], "params": {"url": ["https://docs.python.org/3/", "https://pytorch.org", "https://fastapi.tiangolo.com"]}},
    {"task": "查看当前已具备的能力", "tools": ["my_capabilities"], "params": {}},
    {"task": "查询关于 {topic} 的已学知识", "tools": ["query_knowledge"], "params": {"topic": ["Python装饰器", "机器学习基础", "Docker容器化"]}},
    {"task": "运行命令 {cmd}", "tools": ["run_command"], "params": {"cmd": ["ls -la", "python --version", "pip list"]}},
    {"task": "删除文件 {file}", "tools": ["delete_file"], "params": {"file": ["temp.txt", "old_config.json", "debug.log"]}},

    # 多工具组合任务
    {"task": "读取 {file} 文件，然后把 {old} 改成 {new}", "tools": ["read_local_file", "edit_file"], "params": {"file": ["config.json", "README.md"], "old": ["debug\": true", "v1.0.0"], "new": ["debug\": false", "v2.0.0"]}},
    {"task": "先搜索 {query}，然后把搜索结果保存到 {file}", "tools": ["search", "write_file"], "params": {"query": ["Python最佳实践", "Docker常用命令", "Git工作流"] , "file": ["notes.md", "学习笔记.txt"]}},
    {"task": "创建 {file} 文件写入 {content}，然后读取验证", "tools": ["write_file", "read_local_file"], "params": {"file": ["test.py", "data.json"], "content": ["print('测试成功')", '{"name": "态极", "version": "1.0"}']}},
    {"task": "搜索 {query}，然后用 learn_knowledge 保存知识", "tools": ["search", "learn_knowledge"], "params": {"query": ["深度学习框架对比", "Web开发趋势2024", "Rust编程语言特点"]}},
    {"task": "创建项目 {name}，然后安装依赖 {pkg}", "tools": ["create_project", "install_dependency"], "params": {"name": ["my-api", "todo-app"], "pkg": ["flask", "requests"]}},
    {"task": "列出 {dir} 目录，找到 Python 文件后分析代码", "tools": ["list_directory", "analyze_code"], "params": {"dir": ["src", ".", "agent"]}},
    {"task": "用 Python 写一个 {desc}，然后运行测试", "tools": ["execute_python"], "params": {"desc": ["计算器程序", "密码生成器", "文件统计工具"]}},
]


def generate_react_sample(template: Dict) -> Dict:
    """根据模板生成一条 ReAct 训练样本"""
    task = template["task"]
    tools = template["tools"]
    params = template.get("params", {})

    # 随机填充参数
    for key, values in params.items():
        if values:
            task = task.replace(f"{{{key}}}", random.choice(values))

    # 构建步骤
    steps = []
    for i, tool_name in enumerate(tools):
        tool_def = TOOL_DEFINITIONS.get(tool_name)
        if not tool_def:
            continue

        # 生成思考
        thought = tool_def["thought"]

        # 生成参数
        example_args = tool_def["example_args"]
        # 尝试从任务中提取参数
        action_args = {"input": example_args}

        is_last = (i == len(tools) - 1)

        step = {
            "thought": thought,
            "action": tool_name,
            "action_args": action_args,
        }

        if is_last:
            final_answers = [
                "任务已完成，结果如下。",
                "所有操作执行完毕，以下是结果。",
                "操作成功完成。",
            ]
            step["final_answer"] = random.choice(final_answers)

        steps.append(step)

    if not steps:
        return None

    return {"task": task, "steps": steps}


def generate_all_react(target_count: int = 3000) -> List[Dict]:
    """生成所有 ReAct 训练数据"""
    samples = []

    # 从模板生成
    for template in REACT_TASK_TEMPLATES:
        sample = generate_react_sample(template)
        if sample:
            samples.append(sample)

    # 扩充：对每个模板随机生成多个变体
    while len(samples) < target_count:
        template = random.choice(REACT_TASK_TEMPLATES)
        sample = generate_react_sample(template)
        if sample:
            samples.append(sample)

    random.shuffle(samples)
    return samples[:target_count]


# ============================================================
# Part 3: 导出
# ============================================================

def export_training_data(
    output_dir: str = "taiji/training_data",
    num_conv: int = 2000,
    num_react: int = 3000,
):
    """生成并导出所有训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 对话数据
    conv_samples = generate_all_conversations(num_conv)
    conv_path = os.path.join(output_dir, "taiji_conversation_data.jsonl")
    with open(conv_path, "w", encoding="utf-8") as f:
        for item in conv_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"对话训练数据: {len(conv_samples)} 条 -> {conv_path}")

    # ReAct 数据
    react_samples = generate_all_react(num_react)
    react_path = os.path.join(output_dir, "taiji_react_data.jsonl")
    with open(react_path, "w", encoding="utf-8") as f:
        for item in react_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"ReAct 训练数据: {len(react_samples)} 条 -> {react_path}")

    return conv_path, react_path


def main():
    parser = argparse.ArgumentParser(description="态极知识训练数据生成器")
    parser.add_argument("--output", type=str, default="taiji/training_data", help="输出目录")
    parser.add_argument("--conv", type=int, default=2000, help="对话样本数量")
    parser.add_argument("--react", type=int, default=3000, help="ReAct 样本数量")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    conv_path, react_path = export_training_data(args.output, args.conv, args.react)
    print(f"\n✅ 训练数据生成完成！")
    print(f"  对话数据: {conv_path} ({args.conv} 条)")
    print(f"  ReAct 数据: {react_path} ({args.react} 条)")
    print(f"\n使用方式:")
    print(f"  python taiji/train/finetune_taiji.py")


if __name__ == "__main__":
    main()
