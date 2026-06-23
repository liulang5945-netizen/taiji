<template>
  <Transition name="splash-fade">
    <div v-if="visible" class="splash-wrapper">
      <div class="splash-bg"></div>
      <div class="splash-content">
        <div class="splash-logo">
          <Brain :size="36" />
        </div>
        <h1 class="splash-title">态 极</h1>
        <p class="splash-status">{{ statusText }}</p>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { Brain } from 'lucide-vue-next'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const runtimeStore = useRuntimeStore()
const visible = ref(true)
const statusText = ref('正在连接...')
let checkInterval = null

onMounted(() => {
  checkInterval = setInterval(() => {
    const state = runtimeStore.health?.state
    if (state === 'connected') {
      statusText.value = '就绪'
      setTimeout(() => { visible.value = false }, 300)
      clearInterval(checkInterval)
    } else if (state === 'loading') {
      statusText.value = '模型加载中...'
    } else if (state === 'downloading') {
      statusText.value = '模型下载中...'
    } else {
      statusText.value = '正在连接...'
    }
  }, 500)

  setTimeout(() => {
    visible.value = false
    clearInterval(checkInterval)
  }, 8000)
})

onUnmounted(() => {
  if (checkInterval) clearInterval(checkInterval)
})
</script>

<style scoped>
.splash-wrapper {
  position: fixed;
  inset: 0;
  z-index: 99999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.splash-bg {
  position: absolute;
  inset: 0;
  background: var(--bg-base, #1a1b26);
}

.splash-bg::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at 30% 40%, rgba(124,138,255,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 70% 60%, rgba(167,139,250,0.04) 0%, transparent 50%);
}

.splash-content {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  animation: contentIn 0.6s var(--ease, cubic-bezier(0.16, 1, 0.3, 1));
}

@keyframes contentIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Logo */
.splash-logo {
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 16px;
  background: var(--primary-gradient, linear-gradient(135deg, #7c8aff, #a78bfa));
  color: white;
  box-shadow: 0 8px 32px rgba(124,138,255,0.25);
  animation: breathe 3s ease-in-out infinite;
}

@keyframes breathe {
  0%, 100% { transform: scale(1); box-shadow: 0 8px 32px rgba(124,138,255,0.25); }
  50% { transform: scale(1.03); box-shadow: 0 12px 40px rgba(124,138,255,0.35); }
}

/* 标题 */
.splash-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  color: var(--text, #c8cedd);
}

/* 状态文字 */
.splash-status {
  margin: 0;
  font-size: 0.8rem;
  color: var(--text-muted, #565c74);
  letter-spacing: 0.02em;
}

/* 淡出 */
.splash-fade-leave-active {
  transition: opacity 0.4s ease;
}
.splash-fade-leave-to {
  opacity: 0;
}
</style>
