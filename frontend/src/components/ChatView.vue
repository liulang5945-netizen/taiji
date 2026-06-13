<template>
  <main class="chat-workbench">
    <header class="chat-topbar">
      <div class="chat-title-block">
        <div class="title-row">
          <Brain :size="20" />
          <h1>{{ chatStore.currentSessionName || '态极对话' }}</h1>
        </div>
        <div class="life-strip">
          <span class="life-chip" :class="runtimeStore.life.life_state || 'idle'">
            <Activity :size="13" />
            {{ lifeStateText }}
          </span>
          <span class="metric-chip">精力 {{ energyPercent }}%</span>
          <span class="metric-chip">饱腹 {{ satietyPercent }}%</span>
          <span class="metric-chip">好奇 {{ curiosityPercent }}%</span>
        </div>
      </div>

      <div class="topbar-actions">
        <div class="mode-switch" role="group" aria-label="对话模式">
          <button :class="{ active: engineModel === 'taiji' }" @click="setEngine('taiji')">
            <Brain :size="14" />
            思维
          </button>
          <button :class="{ active: engineModel === 'agent' }" @click="setEngine('agent')">
            <Zap :size="14" />
            行动
          </button>
        </div>
        <button class="icon-btn" @click="chatStore.clearCurrentChat()" :title="t('clear_history')">
          <RotateCcw :size="16" />
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
              <div v-if="runtimeNotice.canDo" class="runtime-can-do">
                <span v-for="action in runtimeNotice.canDo" :key="action" class="can-do-tag">{{ action }}</span>
              </div>
              <div v-if="runtimeStore.memory" class="runtime-meta-row">
                <span>{{ runtimeStore.memory.message }}</span>
              </div>
            </div>
          </div>

          <!-- 态极生命表达 -->
          <div v-if="runtimeStore.lifeExpressions.length" class="life-expressions">
            <div v-for="(expr, i) in runtimeStore.lifeExpressions.slice(0, 2)" :key="i"
                 class="life-expression" :class="'life-' + expr.type">
              <span class="life-expr-emoji">{{ expr.emoji }}</span>
              <span class="life-expr-text">{{ expr.text }}</span>
            </div>
          </div>

          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <div class="empty-symbol"><Bot :size="34" /></div>
            <h2>开始一次对话</h2>
            <p>{{ emptyStateCopy }}</p>
          </div>

          <div v-if="chatStore.lifeNeeds && chatStore.isLoading" class="life-needs-indicator">
            <span>饥饿 {{ Math.round(chatStore.lifeNeeds.hunger || 0) }}</span>
            <span>疲劳 {{ Math.round(chatStore.lifeNeeds.fatigue || 0) }}</span>
            <span>好奇 {{ Math.round(chatStore.lifeNeeds.curiosity || 0) }}</span>
            <span>压力 {{ Math.round(chatStore.lifeNeeds.stress || 0) }}</span>
          </div>

          <article
            v-for="(msg, index) in chatStore.messages"
            :key="index"
            :class="['message-row', msg.role]"
          >
            <div class="message-avatar">
              <User v-if="msg.role === 'user'" :size="16" />
              <Bot v-else :size="16" />
            </div>

            <div class="message-main">
              <div class="message-meta">
                <span>{{ msg.role === 'user' ? '你' : '态极' }}</span>
                <span v-if="msg.role === 'assistant'" class="mode-label">
                  {{ engineModel === 'agent' ? '行动回馈' : '思维回馈' }}
                </span>
              </div>

              <div class="bubble">
                <div v-if="msg.role === 'user'">
                  <div class="text-content">{{ msg.content }}</div>
                  <div v-if="msg.attachments?.length" class="message-media-grid">
                    <div v-for="(att, ai) in msg.attachments" :key="ai" class="message-media-card" :class="att.modality">
                      <img v-if="mediaAttachmentUrl(att) && att.modality === 'image'" :src="mediaAttachmentUrl(att)" :alt="att.name" />
                      <audio v-else-if="mediaAttachmentUrl(att) && att.modality === 'audio'" :src="mediaAttachmentUrl(att)" controls />
                      <video v-else-if="mediaAttachmentUrl(att) && att.modality === 'video'" :src="mediaAttachmentUrl(att)" controls />
                      <span v-else>{{ att.name }}</span>
                    </div>
                  </div>
                </div>
                <div v-else>
                  <div v-if="parsedMessages[index]?.reasoning" class="reasoning-block">
                    <button class="reasoning-toggle" @click="toggleReasoning(index)">
                      <ChevronDown v-if="expandedReasonings[index]" :size="14" />
                      <ChevronRight v-else :size="14" />
                      <span>思考过程</span>
                    </button>
                    <div
                      v-show="expandedReasonings[index]"
                      class="reasoning-content markdown-body"
                      v-html="renderMarkdown(parsedMessages[index].reasoning)"
                    />
                  </div>
                  <div class="markdown-body" v-html="renderMarkdown(parsedMessages[index] ? parsedMessages[index].content : msg.content)" />
                  <div v-if="mediaOutputs(index).length" class="message-media-grid output-media-grid">
                    <div v-for="(media, mi) in mediaOutputs(index)" :key="mi" class="message-media-card" :class="media.modality">
                      <img v-if="media.modality === 'image'" :src="media.url" :alt="media.label" />
                      <audio v-else-if="media.modality === 'audio'" :src="media.url" controls />
                      <video v-else-if="media.modality === 'video'" :src="media.url" controls />
                      <a v-else :href="media.url" target="_blank" rel="noreferrer">{{ media.label }}</a>
                    </div>
                  </div>

                  <div v-if="msg.toolCalls?.length" class="tool-cards">
                    <div v-for="(tc, ti) in msg.toolCalls" :key="ti" class="tool-card" :class="tc.status">
                      <div class="tool-card-header">
                        <Zap :size="14" />
                        <span class="tool-name">{{ tc.tool }}</span>
                        <span class="tool-status" :class="tc.status">
                          {{ tc.status === 'running' ? '执行中' : '完成' }}
                        </span>
                      </div>
                      <div class="tool-trace">
                        <div v-if="tc.args && Object.keys(tc.args).length" class="trace-step">
                          <span class="trace-label">行动</span>
                          <code>{{ JSON.stringify(tc.args, null, 2) }}</code>
                        </div>
                        <div v-if="tc.result" class="trace-step">
                          <span class="trace-label">观察</span>
                          <pre>{{ formatToolResult(tc.result) }}</pre>
                        </div>
                        <div v-else-if="tc.status === 'running'" class="trace-step muted">
                          <span class="trace-label">观察</span>
                          <span>等待工具返回...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
                <button class="msg-action-btn" @click="copyMsg(msg.content)" title="复制">
                  <Copy :size="14" />
                </button>
                <button class="msg-action-btn" @click="chatStore.regenerateMessage(index)" title="重新执行">
                  <RotateCcw :size="14" />
                </button>
              </div>
            </div>
          </article>

          <article v-if="chatStore.isLoading" class="message-row assistant thinking-row">
            <div class="message-avatar breathing"><Bot :size="16" /></div>
            <div class="message-main">
              <div class="message-meta">
                <span>态极</span>
                <span class="mode-label">{{ chatStore.isReceiving ? '正在回应' : '正在启动' }}</span>
              </div>
              <div v-if="!chatStore.isReceiving" class="bubble loading-bubble">
                <span class="thinking-animation">
                  <span class="think-dot"></span>
                  <span class="think-dot"></span>
                  <span class="think-dot"></span>
                </span>
              </div>
            </div>
          </article>
        </div>
      </div>

      <footer class="composer-area">
        <div class="stop-container" v-if="chatStore.isReceiving">
          <button class="stop-btn" @click="chatStore.stopGeneration()">
            <Square :size="14" fill="currentColor" />
            中断执行
          </button>
        </div>

        <div class="media-rail">
          <label class="media-entry">
            <input type="file" accept="image/*" multiple @change="onFileSelect" style="display:none" />
            <ImageUp :size="16" />
            <span>图片</span>
          </label>
          <label class="media-entry">
            <input type="file" accept="audio/*" multiple @change="onFileSelect" style="display:none" />
            <AudioLines :size="16" />
            <span>语音</span>
          </label>
          <label class="media-entry">
            <input type="file" accept="video/*" multiple @change="onFileSelect" style="display:none" />
            <Video :size="16" />
            <span>视频</span>
          </label>
          <button class="media-entry record-entry" :class="{ recording: isRecording }" @click="toggleRecording" :disabled="!recordingSupported">
            <Mic :size="16" />
            <span>{{ isRecording ? '停止录音' : '语音交流' }}</span>
          </button>
        </div>

        <div v-if="chatAttachments.length" class="chat-attachments-row">
          <div v-for="(att, idx) in chatAttachments" :key="idx" class="chat-attachment-card" :class="att.modality">
            <div class="attachment-preview" v-if="att.previewUrl && att.modality === 'image'">
              <img :src="att.previewUrl" :alt="att.name" />
            </div>
            <div v-else-if="att.previewUrl && att.modality === 'audio'" class="attachment-preview audio-preview">
              <audio :src="att.previewUrl" controls />
            </div>
            <div v-else-if="att.previewUrl && att.modality === 'video'" class="attachment-preview video-preview">
              <video :src="att.previewUrl" controls />
            </div>
            <div class="chat-attachment-meta">
              <span class="att-name">{{ att.name }}</span>
              <small>{{ att.modality === 'image' ? '图片' : att.modality === 'audio' ? '语音' : att.modality === 'video' ? '视频' : '文件' }}</small>
              <span v-if="att.uploading" class="att-state">上传中...</span>
            </div>
            <button class="att-remove" @click="removeAttachment(idx)"><X :size="12" /></button>
          </div>
        </div>

        <div class="input-container">
          <label class="attach-btn" title="投喂文件/代码">
            <Paperclip :size="18" />
            <input type="file" multiple @change="onFileSelect" style="display:none" />
          </label>
          <n-input
            ref="inputRef"
            v-model:value="chatStore.chatInput"
            type="textarea"
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            :rows="1"
            :autosize="{ minRows: 1, maxRows: 6 }"
            :theme-overrides="{
              borderRadius: '12px',
              color: 'transparent',
              colorFocus: 'transparent',
              border: 'none',
              borderHover: 'none',
              borderFocus: 'none',
              boxShadowFocus: 'none',
            }"
            @keydown="onKeydown"
            @paste="onPaste"
          />
          <button
            class="send-btn"
            :disabled="!canSend"
            @click="handleSend"
            title="发送"
          >
            <Send :size="18" />
          </button>
        </div>
      </footer>
    </section>
  </main>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted, inject } from 'vue'
import {
  Activity, User, Bot, RotateCcw, Copy, ChevronDown, ChevronRight,
  Square, Paperclip, X, Send, Brain, Zap, ImageUp, AudioLines, Video, Mic
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'
import { useChatUpload } from '@/composables/useChatUpload.js'
import { useWorkspaceBridge } from '@/composables/useWorkspaceBridge.js'

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const { renderMarkdown } = useMarkdown()
const workspaceBridge = useWorkspaceBridge()
const toast = inject('toast', () => {})
const t = (key, params) => appStore.t(key, params)

// 注册聊天回调，让终端输出可以回流到聊天
workspaceBridge.registerChatCallback((event) => {
  if (event.type === 'error') {
    chatStore.messages.push({ role: 'system', content: `🔧 ${event.content}`, timestamp: event.timestamp })
  } else if (event.type === 'explanation') {
    chatStore.messages.push({ role: 'assistant', content: event.content, timestamp: event.timestamp })
  } else if (event.type === 'terminal') {
    chatStore.messages.push({ role: 'system', content: `📟 ${event.content}`, timestamp: event.timestamp })
  }
})

const messagesArea = ref(null)
const inputRef = ref(null)
const isRecording = ref(false)
const recordingSupported = computed(() => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined')
let mediaRecorder = null
let recordingChunks = []

const engineModel = ref(localStorage.getItem('taiji_engine') || 'taiji')
function setEngine(val) {
  engineModel.value = val
  localStorage.setItem('taiji_engine', val)
}

const energyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.fatigue || 0)).toFixed(0))
const satietyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.hunger || 0)).toFixed(0))
const curiosityPercent = computed(() => Math.max(0, runtimeStore.life.needs?.curiosity || 0).toFixed(0))

const lifeStateText = computed(() => {
  const s = runtimeStore.life.life_state || 'idle'
  const map = {
    idle: '清醒',
    sleeping: '睡眠',
    feeding: '吸收',
    playing: '探索',
    working: '执行',
  }
  return map[s] || s
})

const runtimeNotice = computed(() => runtimeStore.runtimeNotice)

const emptyStateCopy = computed(() => {
  if (runtimeNotice.value) return '你可以先整理任务、挂载资料或检查工具能力。'
  return '把问题、文件或任务交给态极。'
})

const canSend = computed(() => {
  return (!!chatStore.chatInput.trim() || chatAttachments.some(att => !att.uploading))
    && !chatStore.isLoading
    && runtimeStore.health.state === 'connected'
    && runtimeStore.health.modelLoaded
})

const parsedMessages = ref({})
const expandedReasonings = ref({})

function parseMessageContent(content) {
  if (!content) return { content: '', reasoning: '' }
  const thinkMatch = content.match(/<(?:think|thinking)>([\s\S]*?)<\/(?:think|thinking)>/)
  if (thinkMatch) {
    const reasoning = thinkMatch[1].trim()
    const rest = content.replace(thinkMatch[0], '').trim()
    return { content: rest, reasoning }
  }
  return { content, reasoning: '' }
}

function toMediaUrl(raw) {
  if (!raw) return ''
  const text = String(raw).trim().replace(/^["'`]+|["'`]+$/g, '')
  if (/^(https?:|data:|blob:)/i.test(text)) return text
  if (/^[a-zA-Z]:[\\/]/.test(text)) {
    const name = text.split(/[\\/]/).pop()
    if (text.includes('multimodal_uploads') && name) return `/multimodal_media/${encodeURIComponent(name)}`
    const workspaceUrl = workspaceMediaUrlFromPath(text)
    if (workspaceUrl) return workspaceUrl
    return ''
  }
  return text
}

function workspaceMediaUrlFromPath(text) {
  const normalized = String(text || '').replaceAll('\\', '/')
  const marker = 'agent_workspace'
  const idx = normalized.toLowerCase().indexOf(marker)
  if (idx === -1) return ''
  const rel = normalized.slice(idx + marker.length).replace(/^\/+/, '')
  return rel ? `/workspace_data/${rel.split('/').map(encodeURIComponent).join('/')}` : ''
}

function mediaAttachmentUrl(att) {
  if (!att) return ''
  return att.previewUrl || att.publicUrl || toMediaUrl(att.savedPath)
}

function inferOutputModality(url) {
  const clean = String(url || '').split('?')[0].toLowerCase()
  if (/\.(png|jpe?g|gif|webp|bmp|svg)$/.test(clean) || clean.startsWith('data:image/')) return 'image'
  if (/\.(mp3|wav|ogg|flac|m4a|aac|webm)$/.test(clean) || clean.startsWith('data:audio/')) return 'audio'
  if (/\.(mp4|mov|mkv|avi|webm)$/.test(clean) || clean.startsWith('data:video/')) return 'video'
  return 'file'
}

function extractMediaOutputs(text) {
  const result = []
  const source = String(text || '')
  const patterns = [
    /(?:生成|输出|保存|文件|路径|path|url)[：:\s]+([a-zA-Z]:[\\/][^\s"'`<>]+\.(?:png|jpe?g|gif|webp|bmp|svg|mp3|wav|ogg|flac|m4a|aac|mp4|mov|mkv|avi|webm))/gi,
    /((?:https?:\/\/|data:(?:image|audio|video)\/)[^\s"'`<>]+\.(?:png|jpe?g|gif|webp|bmp|svg|mp3|wav|ogg|flac|m4a|aac|mp4|mov|mkv|avi|webm)?[^\s"'`<>]*)/gi,
  ]
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) {
      const raw = match[1]
      const url = toMediaUrl(raw)
      const modality = inferOutputModality(url)
      if (url && modality !== 'file' && !result.some(item => item.url === url)) {
        result.push({ url, modality, label: raw.split(/[\\/]/).pop() || modality })
      }
    }
  }
  return result
}

function mediaOutputs(index) {
  const parsed = parsedMessages.value[index]
  return extractMediaOutputs(parsed ? parsed.content : '')
}

function toggleReasoning(index) {
  expandedReasonings.value[index] = !expandedReasonings.value[index]
}

watch(() => chatStore.messages, (msgs) => {
  const parsed = {}
  msgs.forEach((msg, i) => {
    if (msg.role === 'assistant') parsed[i] = parseMessageContent(msg.content)
  })
  parsedMessages.value = parsed
}, { deep: true, immediate: true })

const { chatAttachments, onChatFileSelect: onFileSelect, onPaste, uploadChatFiles, removeChatAttachment: removeAttachment, detachAttachments } = useChatUpload()

async function toggleRecording() {
  if (!recordingSupported.value) {
    toast('当前环境不支持浏览器录音', 'warning')
    return
  }
  if (isRecording.value && mediaRecorder) {
    mediaRecorder.stop()
    return
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recordingChunks = []
    mediaRecorder = new MediaRecorder(stream)
    mediaRecorder.ondataavailable = event => {
      if (event.data?.size) recordingChunks.push(event.data)
    }
    mediaRecorder.onstop = async () => {
      isRecording.value = false
      stream.getTracks().forEach(track => track.stop())
      const blob = new Blob(recordingChunks, { type: mediaRecorder.mimeType || 'audio/webm' })
      const file = new File([blob], `voice_${Date.now()}.webm`, { type: blob.type })
      await uploadChatFiles([file])
      mediaRecorder = null
      recordingChunks = []
    }
    mediaRecorder.start()
    isRecording.value = true
  } catch (err) {
    isRecording.value = false
    toast(`无法开始录音: ${err.message}`, 'error')
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesArea.value) messagesArea.value.scrollTop = messagesArea.value.scrollHeight
  })
}

watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isReceiving, scrollToBottom)

function handleSend() {
  if (!canSend.value) return
  if ((runtimeStore.life.needs?.fatigue || 0) > 85) {
    toast('疲劳较高，正在强撑执行', 'warning')
  }
  chatStore.sendMessage(engineModel.value, chatAttachments)
  detachAttachments()
  setTimeout(() => runtimeStore.refreshLife(), 500)
  scrollToBottom()
}

function formatToolResult(result) {
  const text = typeof result === 'string' ? result : JSON.stringify(result, null, 2)
  return text.length > 700 ? `${text.slice(0, 700)}...` : text
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

async function copyMsg(content) {
  try {
    await navigator.clipboard.writeText(content)
    toast(`${t('copy')} ✓`, 'success')
  } catch {
    toast('复制失败', 'error')
  }
}

onMounted(() => {
  runtimeStore.refreshLife()
  scrollToBottom()
})

onUnmounted(() => {
  if (mediaRecorder && isRecording.value) mediaRecorder.stop()
})
</script>

<style scoped>
.chat-workbench {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  color: var(--text);
}

.chat-topbar {
  min-height: 76px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 14px 22px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  flex-shrink: 0;
}

.chat-title-block {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  color: var(--text);
}

.title-row h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 650;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  letter-spacing: 0;
}

.life-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.life-chip,
.metric-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 24px;
  padding: 0 9px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-muted);
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1;
}

.life-chip {
  color: var(--primary);
  background: var(--primary-subtle);
  border-color: var(--primary-light);
  font-weight: 600;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.mode-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(72px, 1fr));
  padding: 3px;
  border-radius: 10px;
  background: var(--bg-muted);
  border: 1px solid var(--border);
}

.mode-switch button,
.icon-btn,
.msg-action-btn,
.send-btn,
.stop-btn,
.reasoning-toggle,
.att-remove {
  font: inherit;
}

.mode-switch button {
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
}

.mode-switch button.active {
  background: var(--bg-card);
  color: var(--primary);
  box-shadow: var(--shadow-sm);
}

.icon-btn {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text-secondary);
  background: var(--bg-card);
  cursor: pointer;
}

.icon-btn:hover {
  color: var(--text);
  background: var(--bg-hover);
}

.conversation-shell {
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.messages-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 28px 24px;
}

.chat-thread {
  width: min(100%, 900px);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.runtime-notice {
  display: grid;
  grid-template-columns: 9px minmax(0, 1fr);
  align-items: start;
  gap: 11px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.runtime-notice-dot {
  width: 9px;
  height: 9px;
  margin-top: 6px;
  border-radius: 999px;
  background: var(--text-muted);
}

.runtime-notice.loading .runtime-notice-dot,
.runtime-notice.connecting .runtime-notice-dot {
  background: var(--warning);
}

.runtime-notice.error .runtime-notice-dot {
  background: var(--danger);
}

.runtime-notice.connected .runtime-notice-dot {
  background: var(--success);
}

.runtime-notice strong {
  display: block;
  color: var(--text);
  font-size: 13px;
  font-weight: 650;
}

.runtime-notice p {
  margin: 3px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.runtime-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 7px;
}

.runtime-meta-row span {
  padding: 3px 7px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-muted);
  color: var(--text-muted);
  font-size: 11px;
}

.runtime-can-do {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.can-do-tag {
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--primary-light);
  color: var(--primary);
  font-size: 11px;
  font-weight: 500;
}

/* 态极生命表达 */
.life-expressions {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
  padding: 0 4px;
}

.life-expression {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  font-size: 13px;
  animation: fadeIn 0.5s ease;
}

.life-expression.life-fatigue { border-left: 3px solid var(--info); }
.life-expression.life-hunger { border-left: 3px solid var(--warning); }
.life-expression.life-curiosity { border-left: 3px solid var(--success); }
.life-expression.life-stress { border-left: 3px solid var(--danger); }
.life-expression.life-boredom { border-left: 3px solid var(--purple); }

.life-expr-emoji { font-size: 16px; }
.life-expr-text { color: var(--text-secondary); font-style: italic; }

.empty-state {
  min-height: 360px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--text-muted);
  text-align: center;
}

.empty-symbol {
  width: 72px;
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 18px;
  background: var(--bg-muted);
  color: var(--primary);
  border: 1px solid var(--border);
}

.empty-state h2 {
  margin: 10px 0 0;
  color: var(--text);
  font-size: 22px;
}

.empty-state p {
  margin: 0;
}

.message-row {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
}

.message-row.user {
  grid-template-columns: minmax(0, 1fr) 34px;
}

.message-row.user .message-avatar {
  grid-column: 2;
}

.message-row.user .message-main {
  grid-column: 1;
  grid-row: 1;
  align-items: flex-end;
}

.message-avatar {
  width: 34px;
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 11px;
  background: var(--bg-muted);
  color: var(--primary);
  border: 1px solid var(--border);
}

.message-row.user .message-avatar {
  color: #fff;
  border-color: transparent;
  background: var(--primary-gradient);
}

.message-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.message-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.mode-label {
  color: var(--text-muted);
}

.bubble {
  max-width: min(100%, 780px);
  padding: 14px 16px;
  border-radius: 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  line-height: 1.72;
}

.message-row.user .bubble {
  color: #fff;
  background: var(--primary-gradient);
  border-color: transparent;
}

.message-row.assistant .bubble {
  border-left: 3px solid var(--primary);
}

.text-content {
  white-space: pre-wrap;
  word-break: break-word;
}

.message-media-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 240px));
  gap: 10px;
  margin-top: 12px;
}

.message-media-card {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-muted);
}

.message-media-card img,
.message-media-card video {
  display: block;
  width: 100%;
  max-height: 240px;
  object-fit: contain;
  background: #000;
}

.message-media-card audio {
  width: 100%;
  min-width: 160px;
  display: block;
}

.message-row.user .message-media-card {
  border-color: rgba(255,255,255,0.22);
  background: rgba(255,255,255,0.12);
}

.reasoning-block {
  margin-bottom: 12px;
}

.reasoning-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 30px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-secondary);
  background: var(--bg-muted);
  cursor: pointer;
}

.reasoning-content {
  margin-top: 8px;
  padding: 12px;
  border-radius: 8px;
  border-left: 3px solid var(--border);
  background: var(--bg-muted);
  color: var(--text-secondary);
  max-height: 320px;
  overflow-y: auto;
}

.tool-cards {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
}

.tool-card {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-muted);
}

.tool-card.running {
  border-color: var(--warning);
}

.tool-card.done {
  border-color: var(--success);
}

.tool-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-name {
  font-family: Consolas, 'Courier New', monospace;
  font-weight: 650;
}

.tool-status {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-muted);
}

.tool-status.running {
  color: var(--warning);
}

.tool-status.done {
  color: var(--success);
}

.tool-trace {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
}

.trace-step {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}

.trace-step.muted {
  color: var(--text-muted);
  font-size: 12px;
}

.trace-label {
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: var(--bg-card);
  color: var(--text-muted);
  font-size: 11px;
  border: 1px solid var(--border);
}

.trace-step code,
.trace-step pre {
  display: block;
  margin: 0;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 180px;
  overflow: auto;
}

.msg-actions {
  display: flex;
  gap: 6px;
  opacity: 0;
  transition: opacity 0.18s ease;
}

.message-main:hover .msg-actions {
  opacity: 1;
}

.msg-action-btn {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-muted);
  background: var(--bg-card);
  cursor: pointer;
}

.msg-action-btn:hover {
  color: var(--primary);
  background: var(--primary-subtle);
}

.thinking-row .bubble {
  width: fit-content;
}

.breathing {
  animation: breathe 2s ease-in-out infinite;
}

@keyframes breathe {
  0%, 100% { box-shadow: 0 0 0 rgba(99,102,241,0); transform: scale(1); }
  50% { box-shadow: 0 0 16px rgba(99,102,241,0.28); transform: scale(1.04); }
}

.thinking-animation {
  display: flex;
  gap: 6px;
  align-items: center;
  padding: 4px 2px;
}

.think-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--primary);
  animation: think-bounce 1.25s ease-in-out infinite;
}

.think-dot:nth-child(2) { animation-delay: 0.18s; }
.think-dot:nth-child(3) { animation-delay: 0.36s; }

@keyframes think-bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.38; }
  40% { transform: translateY(-6px); opacity: 1; }
}

.life-needs-indicator {
  align-self: center;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-card);
  color: var(--text-muted);
  font-size: 12px;
}

.composer-area {
  flex-shrink: 0;
  padding: 14px 24px 18px;
  border-top: 1px solid var(--border);
  background: var(--bg-card);
}

.composer-area > * {
  width: min(100%, 900px);
  margin-left: auto;
  margin-right: auto;
}

.stop-container {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}

.stop-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 14px;
  border: 1px solid var(--danger);
  border-radius: 999px;
  color: var(--danger);
  background: var(--danger-light);
  cursor: pointer;
}

.chat-attachments-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}

.media-rail {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.media-entry {
  height: 30px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-muted);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.media-entry:hover {
  color: var(--primary);
  background: var(--primary-subtle);
  border-color: var(--primary-light);
}

.media-entry.recording {
  color: var(--danger);
  background: var(--danger-light);
  border-color: var(--danger);
}

.media-entry:disabled {
  opacity: 0.48;
  cursor: not-allowed;
}

.chat-attachment-card {
  position: relative;
  min-width: 0;
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr) 22px;
  gap: 9px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-muted);
  color: var(--text-secondary);
  font-size: 12px;
}

.chat-attachment-card.audio {
  grid-template-columns: minmax(0, 1fr) 22px;
}

.attachment-preview {
  width: 72px;
  height: 54px;
  overflow: hidden;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
}

.attachment-preview img,
.attachment-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.audio-preview {
  width: 100%;
  grid-column: 1;
}

.audio-preview audio {
  width: 100%;
}

.chat-attachment-meta {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chat-attachment-meta small,
.att-state {
  color: var(--text-muted);
}

.att-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.att-remove {
  width: 18px;
  height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
}

.att-remove:hover {
  color: var(--danger);
  background: var(--danger-light);
}

.input-container {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr) 38px;
  align-items: end;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--bg);
  box-shadow: var(--shadow-sm);
}

.input-container:focus-within {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-subtle);
}

.attach-btn,
.send-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  cursor: pointer;
}

.attach-btn {
  color: var(--text-muted);
}

.attach-btn:hover {
  color: var(--primary);
  background: var(--primary-subtle);
}

.send-btn {
  border: 0;
  color: #fff;
  background: var(--primary);
}

.send-btn:hover:not(:disabled) {
  background: var(--primary-hover);
}

.send-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.markdown-body {
  color: inherit;
}

@media (max-width: 760px) {
  .chat-topbar {
    align-items: stretch;
    flex-direction: column;
    padding: 12px 14px;
  }

  .topbar-actions {
    justify-content: space-between;
  }

  .messages-area {
    padding: 18px 12px;
  }

  .composer-area {
    padding: 12px;
  }

  .message-row,
  .message-row.user {
    grid-template-columns: 30px minmax(0, 1fr);
  }

  .message-row.user .message-avatar {
    grid-column: 1;
  }

  .message-row.user .message-main {
    grid-column: 2;
    align-items: flex-start;
  }

  .bubble {
    max-width: 100%;
  }
}
</style>
