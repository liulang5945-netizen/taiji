<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <Brain class="logo-icon-svg" :size="22" />
        </div>
        <div class="brand-copy">
          <h2>{{ t('title') }}</h2>
          <span>{{ runtimeStore.health.modelLoaded ? '智能体工作台' : '等待模型装载' }}</span>
        </div>
        <MemoryStatusBar class="sidebar-memory-ring" @memory-warning="onMemoryWarning" />
      </div>
      <div class="runtime-card" :class="runtimeStore.connectionClass">
        <span class="runtime-dot"></span>
        <div class="runtime-copy">
          <strong>{{ runtimeStore.connectionStatus }}</strong>
          <small>{{ runtimeStore.modelLifecycle.message }}</small>
        </div>
      </div>
      <div v-if="runtimeStore.issues.length" class="runtime-issues">
        <div v-for="issue in runtimeStore.issues.slice(0, 2)" :key="issue.title" class="runtime-issue" :class="issue.level">
          <strong>{{ issue.title }}</strong>
          <span>{{ issue.message }}</span>
        </div>
      </div>
    </div>

    <button class="new-chat-btn" @click="handleNewChat">
      <Plus :size="16" />
      {{ t('new_chat') }}
    </button>

    <div class="session-list">
      <div class="section-label">对话</div>
      <div v-for="session in chatStore.sessions" :key="session.id"
        :class="['session-item', { active: chatStore.currentSessionId === session.id }]"
        @click="openSession(session.id)">
        <span class="session-name">
          <MessageSquare :size="14" class="session-icon" />
          {{ session.name }}
        </span>
        <button class="session-del-btn" @click.stop="chatStore.deleteSession(session.id)" title="删除">
          <X :size="14" />
        </button>
      </div>
    </div>

    <div class="sidebar-footer">
      <div v-for="group in navGroups" :key="group.title" class="nav-group">
        <div class="section-label">{{ group.title }}</div>
        <button v-for="item in group.items" :key="item.path"
          class="settings-btn" :class="{ active: isActiveRoute(item.path) }"
          @click="navigateTo(item.path)">
          <span class="nav-icon-wrap">
            <component :is="item.icon" :size="14" class="nav-icon" />
          </span>
          <span class="nav-label">{{ item.label }}</span>
        </button>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed, inject } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  Brain, Plus, MessageSquare, X,
  BookOpen, Zap, Cpu, Layout, Settings, Heart
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import MemoryStatusBar from './MemoryStatusBar.vue'

const toast = inject('toast', () => {})
let _memoryWarningTimeout = null

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const router = useRouter()
const route = useRoute()

const t = (key, params) => appStore.t(key, params)

function isActiveRoute(path) {
  return route.path === path
}

function navigateTo(path) {
  if (route.path !== path) {
    router.push(path).catch(() => {})
  }
}

const navGroups = computed(() => [
  {
    title: '工作台',
    items: [
      { path: '/workspace', icon: Layout, label: 'IDE' },
    ],
  },
  {
    title: '能力',
    items: [
      { path: '/agent', icon: Cpu, label: t('agent_config') },
      { path: '/kb', icon: BookOpen, label: t('kb_management') },
      { path: '/train', icon: Zap, label: t('fine_tuning') },
      { path: '/life', icon: Heart, label: '生命状态' },
    ],
  },
  {
    title: '系统',
    items: [
      { path: '/settings', icon: Settings, label: t('sys_settings') },
    ],
  },
])

function handleNewChat() {
  chatStore.createNewSession()
  router.push('/').catch(() => {})
}

function openSession(id) {
  chatStore.switchSession(id)
  router.push('/').catch(() => {})
}

function onMemoryWarning(data) {
  if (_memoryWarningTimeout) return
  toast(
    `\u26A0\uFE0F 系统内存告急！可用仅 ${(data.available_pct * 100).toFixed(0)}% (${data.available_gb.toFixed(1)}GB)。请关闭其他应用或中断当前操作。`,
    'warning'
  )
  _memoryWarningTimeout = setTimeout(() => { _memoryWarningTimeout = null }, 30000)
}
</script>

<style scoped>
.sidebar-header {
  padding: 18px 14px 12px;
}

.sidebar-logo {
  align-items: center;
}

.logo-icon-wrap {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-gradient);
  border-radius: 12px;
  color: white;
  box-shadow: 0 4px 14px rgba(99,102,241,0.3);
  flex-shrink: 0;
  transition: var(--transition);
}

.brand-copy {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.brand-copy h2 {
  margin: 0;
  font-size: 17px;
  line-height: 1.2;
}

.brand-copy span {
  color: var(--text-muted);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.runtime-card {
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr);
  align-items: start;
  gap: 10px;
  margin-top: 14px;
  padding: 10px 11px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.runtime-dot {
  width: 8px;
  height: 8px;
  margin-top: 4px;
  border-radius: 999px;
  background: var(--text-muted);
}

.runtime-card.connected .runtime-dot {
  background: var(--success);
}

.runtime-card.loading .runtime-dot,
.runtime-card.connecting .runtime-dot {
  background: var(--warning);
}

.runtime-card.error .runtime-dot {
  background: var(--danger);
}

.runtime-copy {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.runtime-copy strong {
  color: var(--text);
  font-size: 12px;
  font-weight: 650;
}

.runtime-copy small {
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.35;
}

.runtime-issues {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}

.runtime-issue {
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--bg-muted);
}

.runtime-issue strong,
.runtime-issue span {
  display: block;
}

.runtime-issue strong {
  color: var(--text);
  font-size: 11px;
}

.runtime-issue span {
  margin-top: 2px;
  color: var(--text-muted);
  font-size: 10px;
  line-height: 1.35;
}

.runtime-issue.warning {
  border-color: color-mix(in srgb, var(--warning) 36%, transparent);
  background: var(--warning-light);
}

.runtime-issue.danger {
  border-color: color-mix(in srgb, var(--danger) 36%, transparent);
  background: var(--danger-light);
}

.nav-icon-wrap {
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  flex-shrink: 0;
  transition: var(--transition-fast);
  background: var(--bg-muted);
  color: var(--text-muted);
}

.settings-btn.active .nav-icon-wrap {
  background: var(--primary);
  color: white;
  box-shadow: 0 3px 10px rgba(99,102,241,0.25);
}

.settings-btn:hover .nav-icon-wrap {
  background: var(--primary-light);
  color: var(--primary);
  transform: scale(1.08);
}

.settings-btn.active:hover .nav-icon-wrap {
  background: var(--primary);
  color: white;
}

.nav-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.section-label {
  margin: 10px 8px 6px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
}

.session-list {
  padding: 0 10px 12px;
}

.sidebar-footer {
  gap: 8px;
  padding-top: 10px;
}

.nav-group {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

:global(.theme-light) .sidebar {
  background: rgba(255,255,255,0.94);
  border-right-color: rgba(15,23,42,0.08);
  box-shadow: 1px 0 0 rgba(15,23,42,0.04);
  -webkit-backdrop-filter: blur(14px);
  backdrop-filter: blur(14px);
}

:global(.theme-light) .sidebar::before {
  display: none;
}

:global(.theme-light) .new-chat-btn {
  background: #eef2ff;
  border-color: #dbeafe;
  color: #1e293b;
}

:global(.theme-light) .new-chat-btn:hover {
  background: #e0f2fe;
  border-color: #bae6fd;
}

:global(.theme-light) .session-item.active,
:global(.theme-light) .settings-btn.active {
  background: #ecfeff;
  border-left-color: #0f766e;
  color: #0f766e;
  box-shadow: none;
}

:global(.theme-light) .settings-btn:hover,
:global(.theme-light) .session-item:hover {
  background: #f1f5f9;
  transform: none;
}

:global(.theme-light) .nav-icon-wrap {
  background: #f1f5f9;
  color: #64748b;
}

:global(.theme-light) .settings-btn.active .nav-icon-wrap {
  background: #0f766e;
  color: #fff;
  box-shadow: none;
}
</style>
