@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title Taiji 开发模式

echo ============================================
echo   Taiji 一键开发启动 (Python 3.14)
echo   后端: http://127.0.0.1:8000
echo   前端: http://localhost:5173
echo ============================================
echo.

echo [1/2] 安装/更新依赖...
python -m pip install -r requirements.txt -q 2>nul

echo [2/2] 启动后端 (uvicorn)...
start "Taiji-后端" python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload --log-level info

echo [3/3] 启动前端 (Vite HMR)...
start "Taiji-前端" cmd /c "cd /d frontend && npm run dev"

echo.
echo ✅ 两个服务已启动，等待就绪后访问 http://localhost:5173
echo.
echo 关闭方式: 分别关闭两个控制台窗口
echo ============================================
pause