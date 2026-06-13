@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title 态极打包

echo ============================================
echo   态极桌面客户端打包
echo ============================================
echo.

echo [1/3] 安装打包工具...
pip install pyinstaller -q

echo [2/3] 安装桌面依赖...
pip install PyQt6 PyQt6-WebEngine -q

echo [3/3] 开始打包...
python desktop\build.py

echo.
echo 打包完成！输出目录: dist\Taiji\
echo 运行: dist\Taiji\Taiji.exe
echo.
pause
