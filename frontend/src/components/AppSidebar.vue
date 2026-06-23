<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <img src="/logo.svg" alt="态极" class="logo-icon-svg" />
        </div>
        <div class="brand-copy">
          <h2>{{ t('title') }}</h2>
          <span>{{ runtimeStore.health.modelLoaded ? '智能体工作台' : '等待模型装载' }}</span>
        </div>
      </div>
      <div class="runtime-card" :class="runtimeStore.connectionClass">
        <span class="runtime-dot"></span>
        <span class="runtime-label">{{ runtimeStore.connectionStatus }}</span>
        <MemoryStatusBar class="memory-badge" />
      </div>
    </div>

    <button class="new-chat-btn" @click="handleNewChat">
      <Plus :size="14" />
      {{ t('new_chat') }}
    </button>

    <div class="session-list">
      <div class="section-label">对话</div>
      <div v-for="session in chatStore.sessions" :key="session.id"
        :class="['session-item', { active: chatStore.currentSessionId === session.id }]"
        @click="openSession(session.id)">
        <span class="session-name">
          <MessageSquare :size="13" class="session-icon" />
          {{ session.name }}
        </span>
        <button class="session-del-btn" @click.stop="chatStore.deleteSession(session.id)">
          <X :size="13" />
        </button>
      </div>
    </div>

    <div class="sidebar-footer">
      <div v-for="group in navGroups" :key="group.title" class="nav-group">
        <div class="section-label">{{ group.title }}</div>
        <RouterLink v-for="item in group.items" :key="item.path"
          class="settings-btn" :class="{ active: isActiveRoute(item.path) }" :to="item.path">
          <component :is="item.icon" :size="13" class="nav-icon" />
          <span class="nav-label">{{ item.label }}</span>
        </RouterLink>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { RouterLink, useRouter, useRoute } from 'vue-router'
import { Plus, MessageSquare, X, BookOpen, Zap, Cpu, Layout, Settings, Heart } from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import MemoryStatusBar from './MemoryStatusBar.vue'

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const router = useRouter()
const route = useRoute()
const t = (key) => appStore.t(key)

function isActiveRoute(path) { return route.path === path }
function handleNewChat() { chatStore.createNewSession(); router.push('/').catch(() => {}) }
function openSession(id) { chatStore.switchSession(id); router.push('/').catch(() => {}) }

const navGroups = computed(() => [
  { title: '工作台', items: [{ path: '/workspace', icon: Layout, label: 'IDE' }] },
  { title: '能力', items: [
    { path: '/agent', icon: Cpu, label: t('agent_config') },
    { path: '/kb', icon: BookOpen, label: t('kb_management') },
    { path: '/train', icon: Zap, label: t('fine_tuning') },
    { path: '/life', icon: Heart, label: '生命状态' },
  ]},
  { title: '系统', items: [{ path: '/settings', icon: Settings, label: t('sys_settings') }] },
])
</script>

<style scoped>
.sidebar-header { padding: 18px 16px 12px; }
.sidebar-logo { display: flex; align-items: center; gap: 10px; }
.logo-icon-wrap {
  width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius-md); background: var(--primary-gradient); color: white;
  box-shadow: 0 4px 16px rgba(99,102,241,0.35), 0 0 0 2px rgba(99,102,241,0.1);
  flex-shrink: 0; transition: transform 0.3s var(--ease);
}
.logo-icon-wrap:hover { transform: scale(1.05); }
.logo-icon-svg { width: 28px; height: 28px; }
.brand-copy { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 2px; }
.brand-copy h2 { margin: 0; font-size: 16px; font-weight: 650; color: var(--text); }
.brand-copy span { color: var(--text-muted); font-size: 11px; }

.runtime-card {
  display: flex; align-items: center; gap: 8px; margin-top: 12px;
  padding: 8px 12px; border-radius: var(--radius-md); background: var(--bg-card);
  border: 1px solid var(--border); transition: var(--transition);
}
.runtime-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.runtime-card.connected .runtime-dot { background: var(--success); box-shadow: 0 0 6px rgba(74,222,128,0.4); }
.runtime-card.loading .runtime-dot, .runtime-card.connecting .runtime-dot { background: var(--warning); animation: pulse 1.5s infinite; }
.runtime-card.error .runtime-dot { background: var(--danger); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.runtime-label { flex: 1; color: var(--text-secondary); font-size: 12px; font-weight: 500; }
.memory-badge { flex-shrink: 0; }

.new-chat-btn {
  margin: 6px 12px 8px; height: 38px; display: flex; align-items: center; justify-content: center; gap: 6px;
  border: 1px solid var(--primary-light); border-radius: var(--radius-md);
  background: var(--primary-subtle); color: var(--primary); font-size: 13px; font-weight: 500;
  cursor: pointer; transition: var(--transition);
}
.new-chat-btn:hover { background: var(--primary-light); box-shadow: 0 2px 12px rgba(99,102,241,0.2); transform: translateY(-1px); }

.session-list { flex: 1; min-height: 0; overflow-y: auto; padding: 0 10px 10px; }
.section-label { padding: 8px 8px 4px; font-size: 11px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
.session-item {
  display: flex; align-items: center; justify-content: space-between; padding: 8px 12px;
  border-radius: var(--radius-sm); cursor: pointer; transition: var(--transition);
}
.session-item:hover { background: var(--bg-hover); }
.session-item.active { background: rgba(99,102,241,0.08); }
.session-name {
  display: flex; align-items: center; gap: 7px; min-width: 0; font-size: 13px;
  color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.session-item.active .session-name { color: var(--text); font-weight: 500; }
.session-icon { flex-shrink: 0; color: var(--text-muted); }
.session-del-btn {
  width: 22px; height: 22px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-muted);
  cursor: pointer; opacity: 0; flex-shrink: 0; transition: var(--transition);
}
.session-item:hover .session-del-btn { opacity: 0.5; }
.session-del-btn:hover { opacity: 1 !important; color: var(--danger); background: rgba(239,68,68,0.1); }

.sidebar-footer {
  padding: 10px 12px 14px; border-top: 1px solid var(--border); max-height: 40vh; overflow-y: auto;
}
.nav-group { margin-bottom: 4px; }
.nav-group .section-label { padding: 6px 6px 2px; }
.settings-btn {
  display: flex; align-items: center; gap: 8px; padding: 7px 10px;
  border-radius: var(--radius-sm); color: var(--text-secondary); font-size: 13px;
  text-decoration: none; transition: var(--transition);
}
.settings-btn:hover { background: var(--bg-hover); color: var(--text); }
.settings-btn.active { background: rgba(99,102,241,0.1); color: var(--primary); }

</style>

<style>
/* 布局样式 — 需要全局生效，因为 .sidebar 定义在 shell.css */
.sidebar { height: 100%; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }
</style>
