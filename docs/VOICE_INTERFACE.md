# 态极语音交互

## 功能概述

态极支持语音交互，可以与用户进行语音对话。

### 语音识别 (STT)

- 使用 `speech_recognition` 库
- 支持中文和英文
- 实时麦克风输入
- 音频文件识别

### 语音合成 (TTS)

- 使用 `pyttsx3` 库
- 支持中文语音
- 可调节语速和音量
- 自动清理 Markdown 格式

## 使用方法

### 前端界面

1. **语音输入** — 点击麦克风按钮 🎤
2. **自动识别** — 态极会识别你说的话
3. **语音回复** — 态极会用语音回复你

### 代码调用

```python
from taiji.multimodal.voice_interface import get_voice_interface

# 获取语音接口
voice = get_voice_interface()

# 语音识别
text = voice.listen(timeout=5)
print(f"识别结果: {text}")

# 语音合成
voice.speak("你好，我是态极")

# 语音对话
def process_input(text):
    return f"你说的是: {text}"

voice.voice_conversation(process_input, max_turns=5)
```

## 配置

```python
from taiji.multimodal.voice_interface import VoiceConfig, VoiceInterface

config = VoiceConfig(
    language="zh-CN",    # 语言
    rate=150,            # 语速（字/分钟）
    volume=0.8,          # 音量 0-1
    auto_listen=False,   # 自动监听
)

voice = VoiceInterface(config)
```

## 依赖

```bash
pip install speech_recognition pyttsx3
```

## 状态检查

```python
from taiji.multimodal.voice_interface import get_voice_interface

voice = get_voice_interface()
status = voice.get_status()

print(f"STT 可用: {status['stt_available']}")
print(f"TTS 可用: {status['tts_available']}")
```

## 注意事项

1. **麦克风权限** — 需要授权麦克风访问
2. **网络连接** — STT 需要网络连接（使用 Google 语音识别）
3. **音频设备** — 需要可用的音频输入/输出设备
4. **语言支持** — 默认支持中文，可配置其他语言
