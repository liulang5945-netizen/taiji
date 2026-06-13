"""
态极启动脚本
===========

启动态极 WebSocket 服务器。
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
