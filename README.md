# 态极 (TaijiCore)

> 本地部署的 AI 生命体系统 — 自主学习、进化、感知的智能伙伴

## 项目结构

```text
taiji/
├── taiji/               # 态极生命系统
│   ├── core/            # 核心推理引擎
│   ├── life/            # 生命系统（躯干、行动、代谢、感知）
│   ├── agent/           # Agent 系统
│   ├── agent_ext/       # Agent 扩展（MCP、工作流、沙箱）
│   ├── train/           # 训练系统
│   ├── multimodal/      # 多模态能力
│   ├── body/            # 身体系统
│   ├── brain/           # 大脑系统
│   ├── infra/           # 基础设施（胚胎、事件、自评估）
│   ├── safety/          # 安全与审核
│   └── tools/           # 工具（文件解析、RAG）
├── api/                 # FastAPI 后端服务
│   ├── routes_*/        # API 路由
│   ├── middleware/       # 中间件
│   └── training/        # 训练接口
├── frontend/            # Vue 3 Web 前端
├── tests/               # 测试套件
├── docs/                # 文档
├── scripts/             # 数据构建与工具脚本
├── taiji_data/          # 训练数据与生命数据
│   ├── training_data/   # 训练数据集
│   ├── feed_data/       # 喂养数据
│   ├── sleep_data/      # 睡眠数据
│   ├── life_data/       # 生命事件数据
│   └── evolution_data/  # 进化数据
├── knowledge_store/     # 知识库存储
├── security/            # JWT 密钥与安全
├── user_data/           # 用户聊天记录
├── .github/             # CI/CD 配置
├── requirements.txt     # Python 核心依赖
├── requirements-optional.txt # 可选依赖
├── version.json         # 版本信息
├── app_settings.json    # 应用设置
├── dev.bat              # 开发启动脚本
├── icon.ico             # 应用图标
├── docker-compose.yml   # Docker 编排
├── Dockerfile           # Docker 构建
├── prometheus.yml       # 监控配置
├── pyproject.toml       # 项目元数据
├── README.md            # 本文件
├── CHANGELOG.md         # 变更日志
└── LICENSE              # 许可证
```

## 快速开始

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..

# 启动开发模式
dev.bat
```

## 态极身体系统

态极拥有原生的身体系统，包含三个核心模块：

### 行动系统 (limbs)

- 工具调用、文件操作、代码执行

### 代谢系统 (metabolism)

- 硬件感知、资源管理、设备优化

### 感知系统 (senses)

- API 输入、终端输入、前端输入

```python
from taiji.life.body import BodyCore

body = BodyCore()

# 访问身体模块
body.limbs        # 行动系统
body.metabolism   # 代谢系统
body.senses       # 感知系统

# 状态检查
status = body.get_status()
```

## 生命系统

态极的核心是生命系统：

- **大脑 (TaijiCore)** — 核心推理引擎
- **灵魂 (Soul)** — 人格、情感、价值观
- **躯体 (Body)** — 资源管理、身体模块
- **记忆 (Memory)** — 情景、语义、压缩记忆
- **生命调度 (LifeScheduler)** — 自动安排学习、休息、探索

## 许可证

本项目为私有项目，保留所有权利。
