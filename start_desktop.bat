@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title 态极桌面客户端

echo ============================================
echo   态极桌面客户端
echo   版本: 1.6.0
echo ============================================
echo.

echo [1/3] 检查依赖...
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo 安装 PyQt6...
    pip install PyQt6 PyQt6-WebEngine -q
)

echo [2/3] 检查前端构建...
if not exist "frontend\dist\index.html" (
    echo 构建前端...
    cd frontend
    call npm install -q
    call npm run build
    cd ..
)

echo [3/3] 启动态极桌面客户端...
python desktop\main.py

pause
