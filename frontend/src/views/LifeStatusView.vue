<template>
  <div class="life-status-view">
    <header class="view-header">
      <div class="header-left">
        <n-button quaternary size="small" @click="goHome">
          ← 回到对话
        </n-button>
        <h1>生命状态</h1>
      </div>
      <n-tag :type="connected ? 'success' : 'error'" size="small">
        {{ connected ? '已连接' : '未连接' }}
      </n-tag>
    </header>

    <div class="life-content">
      <!-- 态极形象 -->
      <div class="taiji-avatar-section">
        <div class="avatar-ring" :class="{ active: taijiStatus?.has_model }">
          <div class="avatar-inner">
            <span class="avatar-emoji">{{ currentEmoji }}</span>
          </div>
        </div>
        <h2>态极</h2>
        <p class="status-text">
          {{ taijiStatus?.has_model ? '我已醒来，准备与你交流' : '我还在沉睡中...' }}
        </p>
        <!-- 当前生命活动状态 -->
        <p v-if="currentActivity" class="activity-text">
          {{ currentActivity }}
        </p>
      </div>

      <!-- 需求状态 -->
      <div class="needs-section" v-if="taijiStatus?.needs">
        <h3>内在需求</h3>
        <div class="needs-grid">
          <div class="need-item">
            <span class="need-label">🍚 饥饿</span>
            <div class="need-bar">
              <div class="need-fill" :style="{ width: taijiStatus.needs.hunger + '%' }"
                   :class="{ critical: taijiStatus.needs.hunger > 70 }"></div>
            </div>
            <span class="need-value">{{ Math.round(taijiStatus.needs.hunger) }}</span>
          </div>
          <div class="need-item">
            <span class="need-label">😴 疲劳</span>
            <div class="need-bar">
              <div class="need-fill" :style="{ width: taijiStatus.needs.fatigue + '%' }"
                   :class="{ critical: taijiStatus.needs.fatigue > 80 }"></div>
            </div>
            <span class="need-value">{{ Math.round(taijiStatus.needs.fatigue) }}</span>
          </div>
          <div class="need-item">
            <span class="need-label">😐 无聊</span>
            <div class="need-bar">
              <div class="need-fill" :style="{ width: taijiStatus.needs.boredom + '%' }"
                   :class="{ critical: taijiStatus.needs.boredom > 60 }"></div>
            </div>
            <span class="need-value">{{ Math.round(taijiStatus.needs.boredom) }}</span>
          </div>
          <div class="need-item">
            <span class="need-label">😰 压力</span>
            <div class="need-bar">
              <div class="need-fill" :style="{ width: taijiStatus.needs.stress + '%' }"
                   :class="{ critical: taijiStatus.needs.stress > 70 }"></div>
            </div>
            <span class="need-value">{{ Math.round(taijiStatus.needs.stress) }}</span>
          </div>
          <div class="need-item">
            <span class="need-label">🔍 好奇</span>
            <div class="need-bar">
              <div class="need-fill curiosity" :style="{ width: taijiStatus.needs.curiosity + '%' }"></div>
            </div>
            <span class="need-value">{{ Math.round(taijiStatus.needs.curiosity) }}</span>
          </div>
        </div>
      </div>

      <!-- 生命状态 -->
      <div class="status-grid">
        <div class="status-card">
          <div class="status-icon">💪</div>
          <div class="status-info">
            <span class="status-label">身体状态</span>
            <span class="status-value" :class="{ healthy: taijiStatus?.body_status?.healthy }">
              {{ taijiStatus?.body_status?.healthy ? '健康' : '需要关注' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon">🦾</div>
          <div class="status-info">
            <span class="status-label">行动能力</span>
            <span class="status-value" :class="{ available: taijiStatus?.body_status?.limbs_available }">
              {{ taijiStatus?.body_status?.limbs_available ? '可用' : '不可用' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon">⚡</div>
          <div class="status-info">
            <span class="status-label">代谢系统</span>
            <span class="status-value" :class="{ available: taijiStatus?.body_status?.metabolism_available }">
              {{ taijiStatus?.body_status?.metabolism_available ? '正常' : '异常' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon">👁️</div>
          <div class="status-info">
            <span class="status-label">感知能力</span>
            <span class="status-value" :class="{ available: taijiStatus?.body_status?.senses_available }">
              {{ taijiStatus?.body_status?.senses_available ? '灵敏' : '迟钝' }}
            </span>
          </div>
        </div>
      </div>

      <!-- 互动按钮 -->
      <div class="action-buttons">
        <n-button strong secondary type="error" size="large" class="life-action-btn" @click="feedTaiji">
          <span class="btn-icon">🍎</span>
          <span class="btn-text">喂养</span>
        </n-button>
        <n-button strong secondary type="info" size="large" class="life-action-btn" @click="sleepTaiji">
          <span class="btn-icon">😴</span>
          <span class="btn-text">睡眠</span>
        </n-button>
        <n-button strong secondary type="success" size="large" class="life-action-btn" @click="playTaiji">
          <span class="btn-icon">🎮</span>
          <span class="btn-text">玩耍</span>
        </n-button>
        <n-button strong secondary type="warning" size="large" class="life-action-btn" @click="trainTaiji">
          <span class="btn-icon">🧠</span>
          <span class="btn-text">训练</span>
        </n-button>
      </div>

      <!-- 操作结果 -->
      <div v-if="actionResult" class="action-result">
        <p>{{ actionResult }}</p>
      </div>

      <!-- 生命活动日志 -->
      <div class="activity-log" v-if="activityLog.length">
        <h3>📡 生命活动日志</h3>
        <div class="log-list">
          <div v-for="(log, i) in activityLog" :key="i" class="log-entry"
               :class="'log-' + log.type">
            <span class="log-time">{{ log.time }}</span>
            <span class="log-emoji">{{ log.emoji }}</span>
            <span class="log-message">{{ log.message }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useWebSocket } from '@/composables/useWebSocket.js'

const router = useRouter()
function goHome() { router.push('/').catch(() => {}) }

const {
  connected,
  taijiStatus,
  feed,
  sleep,
  play,
  train,
  getStatus,
  on,
} = useWebSocket()

const actionResult = ref('')
const activityLog = ref([])
const currentActivity = ref('')

// 当前表情（根据生命状态变化）
const currentEmoji = computed(() => {
  if (!taijiStatus.value?.has_model) return '😴'
  const state = taijiStatus.value?.life_state || 'idle'
  const emojiMap = {
    'idle': '😊',
    'feeding': '🍚',
    'sleeping': '💤',
    'playing': '🎮',
    'working': '🏃',
  }
  return emojiMap[state] || '😊'
})

onMounted(() => {
  getStatus()
})

// 添加日志
function addLog(type, emoji, message) {
  activityLog.value.unshift({
    time: new Date().toLocaleTimeString(),
    type,
    emoji,
    message,
  })
  if (activityLog.value.length > 30) {
    activityLog.value.pop()
  }
}

// 监听生命事件（来自 EventBus 广播）
on('life_event', (data) => {
  const { event_type, data: eventData } = data
  const emojiMap = {
    'feed_complete': '🍚',
    'sleep_complete': '💤',
    'play_complete': '🎮',
    'life_started': '🌱',
    'life_stopped': '⏸️',
    'interaction_success': '✅',
    'interaction_failure': '❌',
  }
  const emoji = emojiMap[event_type] || '📌'
  const messageMap = {
    'feed_complete': '吃饭完成',
    'sleep_complete': '睡醒了',
    'play_complete': '玩耍完成',
    'life_started': '生命启动',
    'life_stopped': '生命暂停',
    'interaction_success': '交互成功',
    'interaction_failure': '交互失败',
  }
  const message = messageMap[event_type] || event_type
  addLog(event_type, emoji, message)

  // 刷新状态
  getStatus()
})

// 监听响应
on('feed_response', (data) => {
  const success = data.result?.success
  actionResult.value = success ? data.result.message : `喂养失败: ${data.result?.message}`
  addLog('feed', '🍚', success ? '手动喂养完成' : '喂养失败')
  getStatus()
})

on('sleep_response', (data) => {
  const success = data.result?.success
  actionResult.value = success ? data.result.message : `睡眠失败: ${data.result?.message}`
  addLog('sleep', '💤', success ? '手动睡眠完成' : '睡眠失败')
  getStatus()
})

on('play_response', (data) => {
  const success = data.result?.success
  actionResult.value = success ? data.result.message : `玩耍失败: ${data.result?.message}`
  addLog('play', '🎮', success ? '手动玩耍完成' : '玩耍失败')
  getStatus()
})

on('training_start', () => {
  actionResult.value = '开始训练...'
  currentActivity.value = '🧠 正在训练...'
  addLog('train', '🧠', '开始训练')
})

on('training_complete', (data) => {
  const success = data.result?.success
  actionResult.value = success ? data.result.message : `训练失败: ${data.result?.message}`
  currentActivity.value = ''
  addLog('train', '🧠', success ? '训练完成' : '训练失败')
  getStatus()
})

function feedTaiji() {
  currentActivity.value = '🍚 正在吃饭...'
  feed('用户喂养')
}

function sleepTaiji() {
  currentActivity.value = '💤 正在睡觉...'
  sleep()
}

function playTaiji() {
  currentActivity.value = '🎮 正在玩耍...'
  play()
}

function trainTaiji() {
  train(3, 5e-5)
}
</script>

<style scoped>
.life-status-view {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.view-header h1 {
  color: var(--primary);
  font-size: 28px;
  font-weight: 600;
  margin: 0;
}

.life-content {
  max-width: 800px;
  margin: 0 auto;
}

/* 态极头像 */
.taiji-avatar-section {
  text-align: center;
  margin-bottom: 40px;
}

.avatar-ring {
  width: 120px;
  height: 120px;
  margin: 0 auto 16px;
  border-radius: 50%;
  background: var(--primary-gradient);
  padding: 4px;
  transition: var(--transition);
}

.avatar-ring.active {
  background: linear-gradient(135deg, var(--info) 0%, #0099cc 100%);
  box-shadow: var(--shadow-glow);
}

.avatar-inner {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: var(--bg-card);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(12px);
}

.avatar-emoji {
  font-size: 48px;
}

.taiji-avatar-section h2 {
  color: var(--text);
  font-size: 28px;
  margin-bottom: 8px;
}

.status-text {
  color: var(--text-secondary);
  font-size: 16px;
}

.activity-text {
  color: var(--info);
  font-size: 14px;
  margin-top: 8px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 需求状态 */
.needs-section {
  margin-bottom: 32px;
  padding: 20px;
  background: var(--bg-card);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  backdrop-filter: blur(12px);
}

.needs-section h3 {
  color: var(--text);
  font-size: 18px;
  margin-bottom: 16px;
}

.needs-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.need-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.need-label {
  color: var(--text-secondary);
  font-size: 14px;
  min-width: 70px;
}

.need-bar {
  flex: 1;
  height: 8px;
  background: var(--bg-muted);
  border-radius: 4px;
  overflow: hidden;
}

.need-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--success), #40c057);
  border-radius: 4px;
  transition: width 0.5s ease;
}

.need-fill.critical {
  background: linear-gradient(90deg, var(--danger), #ee5a24);
  animation: bar-pulse 1s infinite;
}

.need-fill.curiosity {
  background: linear-gradient(90deg, var(--warning), #fab005);
}

@keyframes bar-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.need-value {
  color: var(--text-muted);
  font-size: 12px;
  min-width: 30px;
  text-align: right;
}

/* 状态网格 */
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.status-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: var(--bg-card);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  backdrop-filter: blur(12px);
  transition: var(--transition);
}

.status-card:hover {
  border-color: var(--primary);
  box-shadow: var(--shadow-sm);
}

.status-icon {
  font-size: 28px;
}

.status-info {
  display: flex;
  flex-direction: column;
}

.status-label {
  color: var(--text-muted);
  font-size: 12px;
}

.status-value {
  color: var(--danger);
  font-size: 14px;
  font-weight: 600;
}

.status-value.healthy,
.status-value.available {
  color: var(--success);
}

.action-buttons {
  display: grid;
  grid-template-columns: repeat(4, minmax(96px, 1fr));
  gap: 12px;
  margin: 0 auto 24px;
  max-width: 640px;
}

.action-buttons :deep(.n-button) {
  width: 100%;
  min-width: 0;
  min-height: 48px;
  padding: 0 10px;
}

.life-action-btn :deep(.n-button__content) {
  width: 100%;
  min-width: 0;
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.btn-icon,
.btn-text {
  min-width: 0;
  line-height: 1;
}

.btn-icon {
  font-size: 18px;
  text-align: center;
}

.btn-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 操作结果 */
.action-result {
  padding: 16px;
  background: var(--info-light);
  border-radius: var(--radius-sm);
  border: 1px solid rgba(6, 182, 212, 0.2);
  margin-bottom: 24px;
}

.action-result p {
  color: var(--info);
  font-size: 14px;
  margin: 0;
}

/* 生命活动日志 */
.activity-log {
  margin-top: 24px;
}

.activity-log h3 {
  color: var(--text);
  font-size: 18px;
  margin-bottom: 16px;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.log-entry {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-card);
  border-radius: 8px;
  border-left: 3px solid var(--border);
  animation: slide-in 0.3s ease;
  backdrop-filter: blur(8px);
}

.log-feed { border-left-color: var(--danger); }
.log-sleep { border-left-color: var(--purple); }
.log-play { border-left-color: var(--success); }
.log-train { border-left-color: var(--warning); }
.log-life_started { border-left-color: var(--success); }
.log-life_stopped { border-left-color: var(--danger); }
.log-interaction_success { border-left-color: var(--success); }
.log-interaction_failure { border-left-color: var(--danger); }

@keyframes slide-in {
  from { opacity: 0; transform: translateX(-10px); }
  to { opacity: 1; transform: translateX(0); }
}

.log-time {
  color: var(--text-muted);
  font-size: 11px;
  min-width: 70px;
}

.log-emoji {
  font-size: 16px;
}

.log-message {
  color: var(--text-secondary);
  font-size: 13px;
}

/* 响应式 */
@media (max-width: 640px) {
  .life-status-view { padding: 16px; }
  .view-header h1 { font-size: 22px; }
  .status-grid { grid-template-columns: 1fr; }
  .action-buttons { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .action-btn { min-width: 80px; padding: 14px 10px; }
  .btn-icon { font-size: 22px; }
  .btn-text { font-size: 12px; }
  .avatar-ring { width: 80px; height: 80px; }
  .avatar-emoji { font-size: 36px; }
}
</style>
