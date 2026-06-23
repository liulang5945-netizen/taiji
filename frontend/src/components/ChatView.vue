<template>
  <main class="chat-workbench">
    <header class="chat-topbar">
      <div class="chat-title-block">
        <div class="title-row">
          <Brain :size="18" />
          <h1>{{ chatStore.currentSessionName || '态极对话' }}</h1>
        </div>
        <div class="life-strip">
          <span class="life-chip" :class="runtimeStore.life.life_state || 'idle'">
            <Activity :size="12" />
            {{ lifeStateText }}
          </span>
          <span class="metric-chip">精力 {{ energyPercent }}%</span>
          <span class="metric-chip">饱腹 {{ satietyPercent }}%</span>
          <span class="metric-chip">好奇 {{ curiosityPercent }}%</span>
        </div>
      </div>
      <div class="topbar-actions">
        <button class="icon-btn" @click="chatStore.clearCurrentChat()" :title="t('clear_history')">
          <RotateCcw :size="15" />
        </button>
      </div>
    </header>

    <section class="conversation-shell">
      <div class="messages-area" ref="messagesArea">
        <div class="chat-thread">
          <div v-if="runtimeNotice" class="runtime-notice" :class="runtimeStore.connectionClass">
            <span class="runtime-notice-dot"></span>
            <div>
              <strong>{{ runtimeNotice.title }}</strong>
              <p>{{ runtimeNotice.message }}</p>
            </div>
          </div>

          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <div class="empty-glow-ring">
              <img src="/logo.svg" alt="态极" class="empty-logo" />
            </div>
            <h2 class="empty-title">态 极</h2>
            <p class="empty-desc">把问题、文件或任务交给态极</p>
            <div class="empty-hints">
              <div class="hint-card" v-for="hint in quickHints" :key="hint.text" @click="chatStore.chatInput = hint.text">
                <component :is="hint.icon" :size="15" />
                <span>{{ hint.text }}</span>
              </div>
            </div>
          </div>

          <article v-for="(msg, index) in chatStore.messages" :key="msg.id"
            :class="['message-row', msg.role]">
            <div class="message-avatar">
              <User v-if="msg.role === 'user'" :size="15" />
              <Bot v-else :size="15" />
            </div>
            <div class="message-main">
              <div class="message-meta">
                <span>{{ msg.role === 'user' ? '你' : '态极' }}</span>
                <span class="mode-label">态极回应</span>
              </div>
              <div class="bubble">
                <div v-if="msg.role === 'user'" class="text-content">{{ msg.content }}</div>
                <div v-else class="markdown-body" v-html="renderMarkdown(msg.content)" />
              </div>
              <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
                <button class="msg-action-btn" @click="copyMsg(msg.content)" title="复制"><Copy :size="13" /></button>
                <button class="msg-action-btn" @click="chatStore.regenerateMessage(msg.id)" title="重新生成"><RotateCcw :size="13" /></button>
              </div>
            </div>
          </article>

          <article v-if="chatStore.isLoading" class="message-row assistant thinking-row">
            <div class="message-avatar breathing"><Bot :size="15" /></div>
            <div class="message-main">
              <div class="message-meta"><span>态极</span><span class="mode-label">{{ chatStore.isReceiving ? '正在回应' : '正在启动' }}</span></div>
              <div v-if="!chatStore.isReceiving" class="bubble loading-bubble">
                <span class="thinking-animation"><span class="think-dot"></span><span class="think-dot"></span><span class="think-dot"></span></span>
              </div>
            </div>
          </article>
        </div>
      </div>

      <footer class="composer-area">
        <div class="stop-container" v-if="chatStore.isReceiving">
          <button class="stop-btn" @click="chatStore.stopGeneration()">
            <Square :size="13" fill="currentColor" /> 中断执行
          </button>
        </div>

        <!-- 多模态工具栏 -->
        <div class="multimodal-toolbar">
          <button class="mm-btn" :class="{ active: isRecording }" @click="toggleVoice" title="语音输入">
            <Mic :size="16" />
            <span class="mm-label">{{ isRecording ? '停止' : '语音' }}</span>
          </button>
          <label class="mm-btn" title="上传图片">
            <ImageIcon :size="16" />
            <span class="mm-label">图片</span>
            <input type="file" accept="image/*" @change="onImageSelect" style="display:none" />
          </label>
          <label class="mm-btn" title="上传视频">
            <Video :size="16" />
            <span class="mm-label">视频</span>
            <input type="file" accept="video/*" @change="onVideoSelect" style="display:none" />
          </label>
          <label class="mm-btn" title="上传文件">
            <FileText :size="16" />
            <span class="mm-label">文件</span>
            <input type="file" multiple @change="onFileSelect" style="display:none" />
          </label>
          <button class="mm-btn" @click="toggleCamera" title="拍照/录像">
            <Camera :size="16" />
            <span class="mm-label">拍照</span>
          </button>
        </div>

        <div class="input-container">
          <textarea ref="inputRef" v-model="chatStore.chatInput"
            :placeholder="inputPlaceholder"
            rows="1" @keydown="onKeydown" />
          <button class="send-btn" :class="{ unavailable: !canSend }" :disabled="!canSend" @click="handleSend" title="发送">
            <Send :size="16" />
          </button>
        </div>
        <div class="composer-hint">Enter 发送 · Shift+Enter 换行 · 语音/图片/视频/文件</div>
      </footer>
    </section>
  </main>
</template>

<script setup>
defineOptions({ name: 'ChatView' })
import { ref, computed, watch, nextTick, onMounted, inject } from 'vue'
import { Activity, User, Bot, RotateCcw, Copy, Square, Send, Brain, Lightbulb, Code, BookOpen, Mic, Image as ImageIcon, Video, FileText, Camera } from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const { renderMarkdown } = useMarkdown()
const toast = inject('toast', () => {})
const t = (key) => appStore.t(key)

const messagesArea = ref(null)
const inputRef = ref(null)
const engineModel = ref('agent')  // 统一使用 ReAct 引擎

const energyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.fatigue || 0)).toFixed(0))
const satietyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.hunger || 0)).toFixed(0))
const curiosityPercent = computed(() => Math.max(0, runtimeStore.life.needs?.curiosity || 0).toFixed(0))
const lifeStateText = computed(() => ({ idle: '清醒', sleeping: '睡眠', feeding: '吸收', playing: '探索', working: '执行' }[runtimeStore.life.life_state || 'idle'] || ''))
const runtimeNotice = computed(() => runtimeStore.runtimeNotice)

const quickHints = [
  { icon: Lightbulb, text: '解释一个概念' },
  { icon: Code, text: '帮我写一段代码' },
  { icon: BookOpen, text: '总结一篇文档' },
]

const canSend = computed(() =>
  !!chatStore.chatInput.trim() && !chatStore.isLoading && runtimeStore.health.state === 'connected' && runtimeStore.health.modelLoaded
)

function scrollToBottom() { nextTick(() => { if (messagesArea.value) messagesArea.value.scrollTop = messagesArea.value.scrollHeight }) }
watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isReceiving, scrollToBottom)

const isRecording = ref(false)
const inputPlaceholder = computed(() => {
  if (isRecording.value) return '正在录音...'
  return '输入消息，Enter 发送，Shift+Enter 换行'
})

function handleSend() {
  if (!canSend.value) return
  chatStore.sendMessage(engineModel.value)
  scrollToBottom()
}
function onKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }

// 语音输入
async function toggleVoice() {
  if (isRecording.value) {
    isRecording.value = false
    // TODO: 停止录音并发送音频
    toast('语音功能开发中', 'info')
  } else {
    isRecording.value = true
    // TODO: 开始录音
    toast('语音功能开发中', 'info')
  }
}

// 图片上传
async function onImageSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await fetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[图片: ${data.filename}] `
      toast('图片已上传', 'success')
    } else {
      toast('图片上传失败', 'error')
    }
  } catch {
    toast('图片上传失败', 'error')
  }
  e.target.value = ''
}

// 视频上传
async function onVideoSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await fetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[视频: ${data.filename}] `
      toast('视频已上传', 'success')
    } else {
      toast('视频上传失败', 'error')
    }
  } catch {
    toast('视频上传失败', 'error')
  }
  e.target.value = ''
}

// 文件上传
async function onFileSelect(e) {
  const files = e.target.files
  if (!files.length) return
  for (const file of files) {
    const formData = new FormData()
    formData.append('file', file)
    try {
      const resp = await fetch('/api/taiji/upload', { method: 'POST', body: formData })
      if (resp.ok) {
        const data = await resp.json()
        chatStore.chatInput += `[文件: ${data.filename}] `
      }
    } catch {}
  }
  toast(`已上传 ${files.length} 个文件`, 'success')
  e.target.value = ''
}

// 拍照/录像
async function toggleCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    // TODO: 实现拍照/录像逻辑
    toast('摄像头功能开发中', 'info')
    stream.getTracks().forEach(t => t.stop())
  } catch {
    toast('无法访问摄像头', 'error')
  }
}

async function copyMsg(content) { try { await navigator.clipboard.writeText(content); toast('已复制', 'success') } catch { toast('复制失败', 'error') } }

onMounted(scrollToBottom)
</script>

<style scoped>
.chat-workbench { height: 100%; display: flex; flex-direction: column; background: var(--bg); color: var(--text); }

/* 顶栏 */
.chat-topbar {
  min-height: 52px; display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 10px 24px; border-bottom: 1px solid var(--glass-border);
  background: var(--glass-bg); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);
  flex-shrink: 0; position: relative; z-index: 5;
}
.chat-topbar::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent);
}
.chat-title-block { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.title-row { display: flex; align-items: center; gap: 8px; color: var(--text); }
.title-row h1 { margin: 0; font-size: 17px; font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.life-strip { display: flex; gap: 6px; }
.life-chip, .metric-chip {
  display: inline-flex; align-items: center; gap: 4px; height: 22px; padding: 0 8px;
  border-radius: var(--radius-full); font-size: 11px; color: var(--text-muted);
  border: 1px solid var(--border); background: var(--bg-muted);
}
.life-chip { color: var(--primary); background: var(--primary-subtle); border-color: var(--primary-light); font-weight: 600; }

.topbar-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.mode-switch {
  display: flex; padding: 3px; border-radius: var(--radius-md); background: var(--bg-muted);
  border: 1px solid var(--border);
}
.mode-switch button, .icon-btn, .msg-action-btn, .send-btn, .stop-btn { font: inherit; }
.mode-switch button {
  height: 30px; padding: 0 12px; display: inline-flex; align-items: center; justify-content: center; gap: 5px;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-secondary); cursor: pointer; font-size: 12px;
}
.mode-switch button.active { background: var(--bg-card); color: var(--primary); box-shadow: var(--shadow-sm); }
.icon-btn {
  width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); border-radius: var(--radius-md); color: var(--text-secondary);
  background: var(--glass-bg); cursor: pointer; transition: var(--transition);
}
.icon-btn:hover { color: var(--text); background: var(--bg-hover); transform: scale(1.05); }

/* 对话区 */
.conversation-shell { min-height: 0; flex: 1; display: flex; flex-direction: column; }
.messages-area { flex: 1; min-height: 0; overflow-y: auto; padding: 20px; }
.chat-thread { width: min(100%, 760px); margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }

/* 运行状态 */
.runtime-notice {
  display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px;
  border-radius: var(--radius-lg); background: var(--glass-bg); border: 1px solid var(--glass-border);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); box-shadow: var(--shadow-sm);
}
.runtime-notice-dot { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.runtime-notice.loading .runtime-notice-dot, .runtime-notice.connecting .runtime-notice-dot { background: var(--warning); }
.runtime-notice.error .runtime-notice-dot { background: var(--danger); }
.runtime-notice.connected .runtime-notice-dot { background: var(--success); }
.runtime-notice strong { display: block; color: var(--text); font-size: 13px; font-weight: 650; }
.runtime-notice p { margin: 3px 0 0; color: var(--text-muted); font-size: 12px; line-height: 1.5; }

/* 空状态 */
.empty-state {
  min-height: 340px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; text-align: center;
}
.empty-glow-ring {
  display: inline-flex; align-items: center; justify-content: center;
  width: 80px; height: 80px; border-radius: 50%;
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
  border: 2px solid rgba(99,102,241,0.3); margin-bottom: 8px;
  box-shadow: 0 0 40px rgba(99,102,241,0.2), 0 0 80px rgba(99,102,241,0.08);
  animation: ringPulse 3s ease-in-out infinite, floatSlow 6s ease-in-out infinite;
}
@keyframes ringPulse {
  0%, 100% { box-shadow: 0 0 40px rgba(99,102,241,0.2), 0 0 80px rgba(99,102,241,0.08); }
  50% { box-shadow: 0 0 50px rgba(99,102,241,0.3), 0 0 100px rgba(99,102,241,0.12); }
}
@keyframes floatSlow { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
.empty-brain { color: var(--primary); }
.empty-logo { width: 48px; height: 48px; }
.empty-title {
  margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: 0.12em;
  background: linear-gradient(135deg, var(--text), var(--primary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.empty-desc { margin: 0; font-size: 0.85rem; color: var(--text-secondary); }

.empty-hints { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 16px; }
.hint-card {
  display: flex; align-items: center; gap: 7px; padding: 10px 18px; border-radius: var(--radius-lg);
  background: var(--glass-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border); color: var(--text-secondary); font-size: 0.85rem;
  cursor: pointer; transition: var(--transition); box-shadow: var(--shadow-sm);
}
.hint-card:hover {
  color: var(--primary); border-color: rgba(99,102,241,0.25); background: rgba(99,102,241,0.08);
  box-shadow: 0 0 24px rgba(99,102,241,0.15), 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

/* 消息 */
.message-row { display: flex; gap: 12px; align-items: flex-start; }
.message-row.user { flex-direction: row-reverse; }
.message-row.user .message-main { align-items: flex-end; }
.message-avatar {
  width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius-md); flex-shrink: 0; margin-top: 2px;
}
.message-row.assistant .message-avatar { background: var(--primary-subtle); color: var(--primary); border: 1px solid var(--primary-light); }
.message-row.user .message-avatar { background: var(--primary-gradient); color: white; border: 1px solid transparent; }

.message-main { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 4px; }
.message-meta { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; color: var(--text); }
.message-row.user .message-meta { flex-direction: row-reverse; }
.mode-label { font-weight: 400; color: var(--text-muted); font-size: 11px; }

.bubble {
  max-width: min(100%, 720px); padding: 14px 16px; border-radius: var(--radius-lg);
  background: var(--glass-bg); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border); box-shadow: var(--shadow-sm); line-height: 1.7;
}
.message-row.user .bubble { color: white; background: var(--primary-gradient); border-color: transparent; }
.message-row.assistant .bubble { border-left: 3px solid var(--primary); }
.text-content { white-space: pre-wrap; word-break: break-word; }

.msg-actions { display: flex; gap: 4px; opacity: 0; transition: opacity 0.15s; margin-top: 4px; }
.message-row.user .msg-actions { justify-content: flex-end; }
.message-main:hover .msg-actions { opacity: 1; }
.msg-action-btn {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-muted);
  background: var(--glass-bg); cursor: pointer; transition: var(--transition);
}
.msg-action-btn:hover { color: var(--primary); background: var(--primary-subtle); }

/* 思考动画 */
.thinking-row .bubble { width: fit-content; }
.breathing { animation: breathe 2s ease-in-out infinite; }
@keyframes breathe { 0%, 100% { box-shadow: 0 0 0 rgba(99,102,241,0); } 50% { box-shadow: 0 0 12px rgba(99,102,241,0.2); } }
.thinking-animation { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.think-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); animation: dotBounce 1.2s ease-in-out infinite; }
.think-dot:nth-child(2) { animation-delay: 0.15s; }
.think-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes dotBounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.3; } 40% { transform: translateY(-5px); opacity: 1; } }

/* 输入区 */
.composer-area {
  flex-shrink: 0; padding: 12px 20px 16px; border-top: 1px solid var(--glass-border);
  background: var(--glass-bg); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);
  position: relative; z-index: 5;
}
.composer-area::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent);
}
.composer-area > * { width: min(100%, 760px); margin-left: auto; margin-right: auto; }

.composer-hint { text-align: center; font-size: 11px; color: var(--text-muted); margin-top: 6px; }

/* 多模态工具栏 */
.multimodal-toolbar {
  display: flex; align-items: center; justify-content: center; gap: 6px;
  margin-bottom: 8px; padding: 6px 0;
}
.mm-btn {
  display: inline-flex; align-items: center; gap: 4px; height: 32px; padding: 0 12px;
  border: 1px solid var(--border); border-radius: var(--radius-full);
  background: var(--bg-elevated); color: var(--text-secondary);
  cursor: pointer; font-size: 12px; transition: var(--transition);
}
.mm-btn:hover { color: var(--primary); border-color: var(--primary); background: var(--primary-subtle); }
.mm-btn.active { color: var(--danger); border-color: var(--danger); background: rgba(239,68,68,0.08); }
.mm-label { font-size: 11px; }

.stop-container { display: flex; justify-content: center; margin-bottom: 8px; }
.stop-btn {
  display: inline-flex; align-items: center; gap: 5px; height: 32px; padding: 0 16px;
  border: 1px solid var(--danger); border-radius: var(--radius-full); color: var(--danger);
  background: rgba(239,68,68,0.08); cursor: pointer; font-size: 12px; transition: var(--transition);
}
.stop-btn:hover { background: rgba(239,68,68,0.15); transform: scale(1.02); }

.input-container {
  display: grid; grid-template-columns: minmax(0, 1fr) 38px; align-items: end; gap: 8px;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: var(--radius-lg);
  background: var(--bg-input); transition: var(--transition);
}
.input-container:focus-within { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-subtle), 0 0 24px rgba(99,102,241,0.12); }

textarea {
  flex: 1; border: 0; outline: none; background: transparent; color: var(--text);
  font-family: var(--font); font-size: 14.5px; line-height: 1.7; resize: none;
  padding: 4px 0; min-height: 24px; max-height: 150px;
}
textarea::placeholder { color: var(--text-muted); }

.send-btn {
  width: 38px; height: 38px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-md); background: var(--primary-gradient); color: white;
  box-shadow: 0 2px 10px rgba(99,102,241,0.35); cursor: pointer; transition: var(--transition);
}
.send-btn:hover:not(:disabled) { box-shadow: 0 4px 20px rgba(99,102,241,0.5); transform: translateY(-1px) scale(1.05); }
.send-btn.unavailable { opacity: 0.4; }

/* Markdown */
.markdown-body { color: inherit; }
.markdown-body :deep(pre) {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 12px; overflow-x: auto; font-family: var(--font-mono); font-size: 13px; line-height: 1.5;
}
.markdown-body :deep(code) { font-family: var(--font-mono); font-size: 13.5px; background: var(--bg-elevated); padding: 2px 6px; border-radius: 6px; }
.markdown-body :deep(pre code) { background: transparent; padding: 0; }
.markdown-body :deep(p) { margin: 0 0 8px; }
.markdown-body :deep(p:last-child) { margin-bottom: 0; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; }
.markdown-body :deep(blockquote) { border-left: 3px solid var(--primary-light); padding-left: 12px; color: var(--text-secondary); }
.markdown-body :deep(img) {
  max-width: 100%; height: auto; border-radius: var(--radius-sm);
  margin: 8px 0; display: block; border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.markdown-body :deep(a) { color: var(--primary); text-decoration: none; }
.markdown-body :deep(a:hover) { text-decoration: underline; }
.markdown-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; }
.markdown-body :deep(th), .markdown-body :deep(td) { border: 1px solid var(--border); padding: 6px 10px; font-size: 13px; }
.markdown-body :deep(th) { background: var(--bg-muted); font-weight: 600; }
.markdown-body :deep(hr) { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

@media (max-width: 760px) {
  .chat-topbar { padding: 10px 14px; flex-direction: column; align-items: stretch; }
  .messages-area { padding: 14px 10px; }
  .composer-area { padding: 10px 12px 14px; }
}
</style>
