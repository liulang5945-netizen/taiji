import os

path = r'e:\taiji\api\routes_chat.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_s = '''@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天端点，支持 SSE 推送"""
    # 根据引擎类型选择数据收集器'''

new_s = '''@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天端点，支持 SSE 推送"""
    # 触发用户指令，中断当前生命活动
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        get_life_scheduler().handle_user_directive()
    except Exception as e:
        logger.warning(f"Failed to trigger user directive: {e}")

    # 根据引擎类型选择数据收集器'''

if old_s in content:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.replace(old_s, new_s))
    print('Replaced successfully')
else:
    print('Failed to find old string')