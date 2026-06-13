# TaijiCore 安装指南

## 系统要求

- **操作系统**: Windows 10+, macOS 10.13+, Linux (Ubuntu 18.04+)
- **Python 版本**: 3.11（推荐）/ 3.10+
- **内存**: 8GB RAM 最少（推荐 16GB+）
- **磁盘**: 20GB+ 可用空间（用于模型与数据）

## Windows 用户特别说明

1. **安装 Python 3.11**
   - 下载：https://www.python.org/downloads/
   - 安装时勾选 "Add Python 3.11 to PATH"
   - 验证：`python --version` 应显示 3.11.x

2. **GPU 支持（可选，但推荐）**
   - NVIDIA GPU: 安装 CUDA 12.1 + cuDNN 8.7
   - AMD GPU: 安装 ROCm 5.7+
   - 无 GPU: 使用 CPU 模式（速度会慢 10-50 倍）

3. **常见问题**
   - `pip install torch` 失败：尝试使用阿里镜像源
   - Visual C++ 编译错误：安装 Visual Studio Build Tools
   - 内存不足：减少 batch_size 或使用量化模型

## 快速开始（3 步）

### 1. 克隆项目
```bash
git clone <repository-url>
cd taiji
```

### 2. 创建虚拟环境并激活
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
# 基础依赖（必需）
pip install -r requirements.txt

# 可选依赖（推荐，包含语音、RAG 等）
pip install -r requirements-optional.txt

# GPU 加速（NVIDIA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 配置镜像源（加速下载）

创建或编辑 `~/.pip/pip.conf`:
```ini
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host = mirrors.aliyun.com
```

或在命令行指定：
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 启动应用

### 开发模式（推荐）
```bash
python start_taiji.py
```
浏览器自动打开 http://localhost:8000

### 纯后端 API 模式
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 前端独立开发
```bash
cd frontend
npm install
npm run dev
```
访问 http://localhost:5173

## 验证安装

```bash
# 运行测试
python -m pytest -q

# 检查语法
python check_syntax.py

# 健康检查
curl http://localhost:8000/api/health
```

## GPU 配置详解

### NVIDIA CUDA 设置
```bash
# 验证 CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 如显示 False，重装 torch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 强制使用 CPU（如遇 GPU 问题）
```bash
# 编辑 taiji/config.py
# 将 device = 'cuda' 改为 device = 'cpu'
```

## 常见问题解决

| 问题 | 症状 | 解决方案 |
|-----|------|---------|
| `ModuleNotFoundError: No module named 'torch'` | 导入失败 | `pip install torch` |
| `CUDA out of memory` | 运行时显存满 | 减小 batch_size 或使用 CPU |
| `pip install` 超时 | 下载中断 | 使用国内镜像源 |
| 服务启动无反应 | 长时间等待 | 检查是否在下载模型，查看 server.log |
| WebSocket 连接失败 | 前端无法连接后端 | 检查防火墙，确保后端运行在 127.0.0.1:8000 |

## 依赖版本一览

### 核心依赖
- PyTorch >= 2.0.0（深度学习）
- FastAPI >= 0.110.0（后端 API）
- Vue 3 + Vite（前端）
- Transformers >= 4.40.0（模型加载）

### 可选依赖
- PyQt6（启动器 GUI）
- sentence-transformers（向量检索）
- langchain（Agent 框架）
- PyPDF2, python-docx（文档解析）

详见 `requirements.txt` 和 `requirements-optional.txt`

## 下一步

- 启动后访问 http://localhost:8000 查看 Web 界面
- 查看 docs/TAIJI_DESIGN.md 了解架构设计
- 查看 README.md 了解功能特性
- 运行 `python -m pytest -q` 验证完整性

## 支持与反馈

- 提交 Issue: GitHub Issues
- 查看日志: `server.log` 或 `app.log`
- 联系作者: [项目主页]