<template>
  <n-config-provider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <n-loading-bar-provider>
      <n-dialog-provider>
        <n-notification-provider>
          <n-message-provider>
            <div class="app-wrapper" @dragenter="onDragEnter" @dragleave="onDragLeave" @dragover="onDragOver" @drop="onDrop">
              <SplashScreen />
              <ToastManager ref="toastRef" />
              <ConfirmDialog ref="confirmRef" />
              <RuntimeExceptionCenter />

              <!-- === Sidebar === -->
              <AppSidebar />

              <!-- === Router View === -->
              <div class="router-wrapper">
                <RouteErrorView v-if="routeError" :message="routeError" />
                <router-view v-else v-slot="{ Component, route }">
                  <keep-alive include="ChatView">
                    <component :is="Component" :key="route.path" />
                  </keep-alive>
                </router-view>
              </div>

              <div
                v-if="dragOver"
                class="global-drag-overlay"
                @click="clearDragState"
                @dragenter.stop
                @dragover.prevent
                @dragleave="onDragLeave"
                @drop.prevent.stop="onDrop"
              >
                <div style="text-align:center;color:white;">
                  <div style="font-size:3rem;margin-bottom:12px;">📥</div>
                  <p style="font-size:1.2rem;color:rgba(255,255,255,0.9);">{{ appStore.t('drop_release') }}</p>
                </div>
              </div>
            </div>
          </n-message-provider>
        </n-notification-provider>
      </n-dialog-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>

<script setup>
import { ref, computed, onErrorCaptured, onMounted, onUnmounted, provide } from 'vue'
import { darkTheme, lightTheme } from 'naive-ui'
import ToastManager from './components/ToastManager.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import RuntimeExceptionCenter from './components/RuntimeExceptionCenter.vue'
import AppSidebar from './components/AppSidebar.vue'
import RouteErrorView from './components/RouteErrorView.vue'
import SplashScreen from './components/SplashScreen.vue'
import { useAppStore } from './stores/appStore.js'
import { useChatStore } from './stores/chatStore.js'
import { useApi } from './composables/useApi.js'
import { API_BASE, authFetch } from './composables/apiClient.js'
import { loadCheckpoints, trainAbortController } from './composables/useTraining.js'
import router from './router'

const appStore = useAppStore()
const chatStore = useChatStore()
const routeError = ref('')

// Naive UI 主题
const naiveTheme = computed(() => {
  return appStore.resolvedTheme === 'light' ? lightTheme : darkTheme
})

const themeOverrides = computed(() => {
  const accent = appStore.currentAccent || '#1d93ab'
  return {
    common: {
      primaryColor: accent,
      primaryColorHover: accent + 'cc',
      primaryColorPressed: accent + 'aa',
      primaryColorSuppl: accent + '88',
      borderRadius: '10px',
      borderRadiusSmall: '6px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif',
    },
    Button: {
      borderRadiusMedium: '10px',
      borderRadiusSmall: '6px',
    },
    Input: {
      borderRadius: '10px',
    },
    Card: {
      borderRadius: '16px',
    },
    Dialog: {
      borderRadius: '16px',
    },
    Notification: {
      borderRadius: '12px',
    },
  }
})

// Toast & Confirm
const toastRef = ref(null)
const confirmRef = ref(null)
const toast = (msg, type = 'info') => { if (toastRef.value) toastRef.value.showToast(msg, type) }
const $confirm = (options) => confirmRef.value ? confirmRef.value.show(options) : Promise.resolve(false)
provide('toast', toast)
provide('$confirm', $confirm)

// API connection
const { startHealthCheck, stopHealthCheck } = useApi()

function onRouteError(error) {
  routeError.value = error?.message || '页面加载失败'
}

router.onError((error) => {
  onRouteError(error)
})

router.beforeEach(() => {
  routeError.value = ''
  clearDragState()
})

router.afterEach(() => {
  routeError.value = ''
})

onErrorCaptured((error) => {
  onRouteError(error)
  return false
})

// Drag
const dragOver = ref(false)
let dragCounter = 0
let dragResetTimer = null
function isFileDrag(event) {
  return Array.from(event.dataTransfer?.types || []).includes('Files')
}
function clearDragState() {
  dragCounter = 0
  dragOver.value = false
  if (dragResetTimer) {
    clearTimeout(dragResetTimer)
    dragResetTimer = null
  }
}
function scheduleDragReset() {
  if (dragResetTimer) clearTimeout(dragResetTimer)
  dragResetTimer = setTimeout(clearDragState, 1200)
}
const onDragEnter = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragCounter++
  dragOver.value = true
  scheduleDragReset()
}
const onDragOver = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragOver.value = true
  scheduleDragReset()
}
const onDragLeave = (event) => {
  if (!isFileDrag(event)) return
  dragCounter--
  if (dragCounter <= 0) clearDragState()
}
const onDrop = (event) => {
  if (isFileDrag(event)) event.preventDefault()
  clearDragState()
}

// Lifecycle
onMounted(async () => {
  window.addEventListener('blur', clearDragState)
  window.addEventListener('keyup', onGlobalKeyup)
  try {
    const r = await authFetch(`${API_BASE}/api/settings`);
    if (r.ok) {
      const saved = await r.json();
      if (saved && typeof saved === 'object') {
        for (const [key, value] of Object.entries(saved)) {
          const storageKey = `taiji_${key}`;
          if (value !== undefined && value !== null) {
            localStorage.setItem(storageKey, typeof value === 'string' ? value : JSON.stringify(value));
          }
        }
        appStore.restoreUISettings(saved);
      }
    }
  } catch (e) { /* 静默处理 */ }

  await chatStore.loadSessions()
  if (chatStore.sessions.length === 0) {
    chatStore.createNewSession()
  }

  startHealthCheck()
  loadCheckpoints()
})

onUnmounted(() => {
  window.removeEventListener('blur', clearDragState)
  window.removeEventListener('keyup', onGlobalKeyup)
  clearDragState()
  stopHealthCheck()
  if (trainAbortController) trainAbortController.abort()
})

function onGlobalKeyup(event) {
  if (event.key === 'Escape') clearDragState()
}
</script>
<style>@import './assets/styles/index.css';</style>
