# 态极部署指南

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.10+ | 3.11 |
| Node.js | 18+ | 20+ |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 10GB | 50GB+（含模型和数据） |
| GPU | 可选（CPU 可运行） | NVIDIA 8GB+ VRAM |

## 快速启动（3 步）

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装前端依赖
cd frontend && npm install && cd ..

# 3. 一键启动
dev.bat
```

启动后访问：**http://localhost:5173**

## 详细部署步骤

### Step 1: 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 确认 Python 版本
python --version  # 应 >= 3.10
```

### Step 2: 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 可选：GPU 加速
pip install -r requirements-optional.txt

# 可选：语义检索（首次会下载 ~80MB 模型）
pip install sentence-transformers

# 前端依赖
cd frontend
npm install
cd ..
```

### Step 3: 模型准备

态极需要一个模型才能运行。有两种方式：

**方式 A：使用已有模型**
```bash
# 如果 taiji_data/final/ 目录下已有 model.pt 和 tokenizer/
# 直接跳到 Step 4
ls taiji_data/final/
# 应该看到: config.json  model.pt  tokenizer/
```

**方式 B：下载预训练模型**
```bash
# 启动后在设置页面下载模型
# 或使用命令行
python -c "
from taiji.model_ext.model_downloader import download_model
download_model('taiji-125m')
"
```

### Step 4: 启动服务

**一键启动（推荐）：**
```bash
dev.bat
```

**手动启动：**
```bash
# 终端 1：后端
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：前端
cd frontend && npm run dev
```

**Docker 启动：**
```bash
docker-compose up -d
```

### Step 5: 访问

打开浏览器访问 **http://localhost:5173**

## 功能验证

### 1. 聊天测试
- 在聊天框输入 "你好"
- 选择引擎：🧠 态极思维（对话）或 ⚡ 态极行动（推理+工具）
- 观察响应

### 2. 生命系统测试
- 访问左侧 sidebar 的 ❤️ 生命状态
- 点击 🍚 喂养、💤 睡眠、🎮 玩耍
- 观察需求变化

### 3. 工具调用测试
- 选择 ⚡ 态极行动 模式
- 输入 "搜索 Python 3.12 新特性"
- 观察工具调用卡片

### 4. 知识库测试
- 访问 📚 知识库 页面
- 上传一个 PDF 或文本文件
- 在聊天中提问相关问题

### 5. 训练测试
- 访问 🧬 训练 页面
- 上传训练数据
- 点击 "开始态极微调"

## 常见问题

### Q: 模型加载失败
```
检查 taiji_data/final/ 目录是否有 model.pt
如果没有，需要先下载或训练模型
```

### Q: 前端无法连接后端
```
确认后端已启动：curl http://127.0.0.1:8000/api/health
检查端口 8000 是否被占用
```

### Q: 内存不足
```
使用 CPU 模式（默认）
减小模型大小：选择 125M 参数的模型
关闭其他占用内存的程序
```

### Q: 工具调用不工作
```
检查网络连接（搜索需要联网）
确认 duckduckgo-search 已安装：pip install duckduckgo-search
```

### Q: 语义检索不工作
```
安装 sentence-transformers：pip install sentence-transformers
首次使用会下载模型（~80MB），需要联网
```

## 配置文件

| 文件 | 用途 |
|------|------|
| `app_settings.json` | 应用设置（主题、模型、搜索引擎） |
| `frontend/.env` | 前端环境变量 |
| `docker-compose.yml` | Docker 编排配置 |

## 目录结构

```
taiji_data/
├── final/              # 模型文件
│   ├── config.json     # 模型配置
│   ├── model.pt        # 模型权重
│   └── tokenizer/      # 分词器
├── training_data/      # 训练数据
├── feed_data/          # 喂养数据
├── sleep_data/         # 睡眠数据
├── life_data/          # 生命事件
├── evolution_data/     # 进化数据
├── memory/             # 持久化记忆
├── semantic_memory/    # 语义向量索引
├── web_cache/          # 网页缓存
└── explore_data/       # 探索数据

agent_workspace/        # Agent 工作目录
knowledge_store/        # RAG 知识库
user_data/              # 用户数据
security/               # JWT 密钥
```

## API 端点

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/chat/stream` | POST | 流式聊天（SSE） |
| `/api/life/status` | GET | 生命状态 |
| `/api/rag/search` | POST | 知识库搜索 |
| `/api/train/start` | POST | 开始训练 |
| `/api/settings` | GET/PUT | 设置管理 |

## 生产部署

### 使用 Docker
```bash
# 构建
docker build -t taiji .

# 运行
docker run -p 8000:8000 -p 5173:5173 -v ./taiji_data:/app/taiji_data taiji
```

### 使用 Docker Compose（推荐）
```bash
# 启动所有服务（含 Redis、Prometheus、Grafana）
docker-compose up -d

# 访问
# 态极: http://localhost:5173
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Nginx 反向代理
```nginx
server {
    listen 80;
    server_name taiji.example.com;

    location / {
        proxy_pass http://127.0.0.1:5173;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
