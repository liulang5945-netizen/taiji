# 态极 · 入口文件清单

> 每个入口的职责边界，避免多处写启动逻辑。

## 产品入口

| 文件 | 职责 | 启动方式 |
|------|------|---------|
| `desktop/main.py` | **桌面客户端（产品入口）** | `python desktop/main.py` |
| `api/run_app.py` | **打包桌面客户端（PyInstaller）** | PyInstaller 打包后由桌面壳启动 |

## 后端核心

| 文件 | 职责 | 说明 |
|------|------|------|
| `api/app.py` | **FastAPI 应用定义（唯一）** | 定义 `app` 实例、CORS、中间件、lifespan、18 个路由模块。不包含 `uvicorn.run`，纯粹是 app 工厂。 |
| `api/api_server.py` | 兼容垫片（deprecated） | 仅 re-export `api.app`，供旧测试代码使用。新代码应直接 `from api.app import app`。 |

## 独立服务

| 文件 | 职责 | 说明 |
|------|------|------|
| `start_taiji.py` | WebSocket 服务器启动 | 启动 `taiji.core.websocket_server`（端口 8765）。被 `desktop/main.py` 通过子进程调用，也可独立运行。 |

## 前端

| 文件 | 职责 | 说明 |
|------|------|------|
| `frontend/vite.config.js` | Vite 构建/开发配置 | 开发模式：端口 5173，代理 `/api` → `localhost:8000`。生产模式：构建到 `dist/`，由 `api/app.py` 托管。 |

## 启动链路

### 产品模式（桌面客户端）
```
desktop/main.py
  ├─ subprocess: uvicorn api.app:app  (端口 8000)
  ├─ subprocess: start_taiji.py       (端口 8765)
  └─ QWebEngineView → http://127.0.0.1:8000
```

### 打包模式（PyInstaller）
```
api/run_app.py
  ├─ QThread: uvicorn.run(app)        (端口 8000, 进程内)
  └─ QWebEngineView → http://127.0.0.1:8000
```

### 开发模式
```
终端1: uvicorn api.app:app --reload   (端口 8000)
终端2: npm run dev                     (端口 5173, 代理 → 8000)
可选:  python start_taiji.py           (端口 8765)
```

## 规则

1. **`api/app.py` 是唯一的 FastAPI app 定义点**。不要在其他文件里重复创建 app 实例。
2. **不要在入口文件里写业务逻辑**。入口只负责：导入 app、启动 uvicorn、管理进程生命周期。
3. **新入口文件必须在本文档登记**。
4. **删除 `api/api_server.py` 前**，先迁移所有测试引用。
