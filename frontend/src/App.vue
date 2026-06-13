<template>
  <n-config-provider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <n-loading-bar-provider>
      <n-dialog-provider>
        <n-notification-provider>
          <n-message-provider>
            <div class="app-wrapper" @dragenter="onDragEnter" @dragleave="onDragLeave" @dragover.prevent @drop.prevent="onDrop">
              <ToastManager ref="toastRef" />
              <ConfirmDialog ref="confirmRef" />
              <RuntimeExceptionCenter />

              <!-- === Sidebar === -->
              <AppSidebar />

              <!-- === Router View === -->
              <div class="router-wrapper">
                <router-view v-slot="{ Component, route }">
                  <keep-alive>
                    <component :is="Component" :key="route.path" />
                  </keep-alive>
                </router-view>
              </div>

              <div v-if="dragOver" class="global-drag-overlay">
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
import { ref, computed, onMounted, onUnmounted, provide } from 'vue'
import { darkTheme, lightTheme } from 'naive-ui'
import ToastManager from './components/ToastManager.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import RuntimeExceptionCenter from './components/RuntimeExceptionCenter.vue'
import AppSidebar from './components/AppSidebar.vue'
import { useAppStore } from './stores/appStore.js'
import { useChatStore } from './stores/chatStore.js'
import { useApi } from './composables/useApi.js'
import { loadCheckpoints, trainAbortController } from './composables/useTraining.js'

const appStore = useAppStore()
const chatStore = useChatStore()

// Naive UI 主题
const naiveTheme = computed(() => {
  return appStore.resolvedTheme === 'light' ? lightTheme : darkTheme
})

const themeOverrides = computed(() => {
  const accent = appStore.currentAccent || '#6366f1'
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
const { startHealthCheck } = useApi()

// Drag
const dragOver = ref(false)
let dragCounter = 0
const onDragEnter = () => { dragCounter++; dragOver.value = true }
const onDragLeave = () => { dragCounter--; if (dragCounter <= 0) { dragCounter = 0; dragOver.value = false } }
const onDrop = () => { dragCounter = 0; dragOver.value = false }

// Lifecycle
onMounted(async () => {
  try {
    const r = await fetch(`${import.meta.env.DEV ? '' : `${window.location.protocol}//${window.location.hostname}:8000`}/api/settings`);
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
  if (trainAbortController) trainAbortController.abort()
})
</script>
<style>@import './assets/styles/index.css';</style>
