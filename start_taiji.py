"""
态极 WebSocket 服务器启动脚本
=============================

启动 taiji.core.websocket_server（端口 8765）。
可独立运行，也可被 desktop/main.py 通过子进程调用。

注意：这不是 HTTP API 服务器。HTTP API 由 api/app.py + uvicorn 提供（端口 8000）。
详见 docs/ENTRYPOINTS.md
"""
import asyncio
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger("Taiji")


async def main():
    """主函数"""
    logger.info("正在启动态极...")

    from taiji.core.websocket_server import start_server
    await start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n态极已停止")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
