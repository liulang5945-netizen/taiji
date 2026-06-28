<template>
  <aside class="sidebar" :style="{ width: width + 'px' }">
    <div class="sidebar-resize-handle" 
         :class="{ active: isResizing }"
         @mousedown="$emit('resize-start', $event)">
    </div>
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <TaijiLogo :size="38" />
        </div>
        <div class="brand-copy">
          <h2>{{ t('title') }}</h2>
          <span>{{ runtimeStore.health.modelLoaded ? '在线' : '等待模型' }}</span>
        </div>
      </div>
      <div class="runtime-card" :class="runtimeStore.connectionClass" role="status" :aria-label="`运行状态: ${runtimeStore.connectionStatus}`">
        <div class="runtime-state">
          <span class="runtime-dot" aria-hidden="true"></span>
          <span class="runtime-label">{{ runtimeStore.connectionStatus }}</span>
        </div>
        <MemoryStatusBar class="memory-badge" />
      </div>
    </div>

    <button class="new-chat-btn" @click="handleNewChat" aria-label="新建对话">
      <Plus :size="15" />
      <span>{{ t('new_chat') }}</span>
    </button>

    <div class="session-list" role="list" aria-label="会话列表">
      <div class="section-label">对话</div>
      <div v-for="session in chatStore.sessions" :key="session.id"
        role="listitem"
        :class="['session-item', { active: chatStore.currentSessionId === session.id }]"
        @click="openSession(session.id)"
        tabindex="0"
        @keydown.enter="openSession(session.id)">
        <span class="session-name">
          <MessageSquare :size="14" class="session-icon" aria-hidden="true" />
          {{ session.name }}
        </span>
        <button class="session-del-btn" @click.stop="chatStore.deleteSession(session.id)"
          :aria-label="`删除会话 ${session.name}`">
          <X :size="13" />
        </button>
      </div>
    </div>

    <div class="sidebar-footer">
      <div v-for="group in navGroups" :key="group.title" class="nav-group">
        <div class="section-label">{{ group.title }}</div>
        <RouterLink v-for="item in group.items" :key="item.path"
          class="settings-btn" :class="{ active: isActiveRoute(item.path) }" :to="item.path">
          <TaijiLogo v-if="item.icon === 'TaijiLogo'" :size="15" class="nav-icon" />
          <component v-else :is="item.icon" :size="15" class="nav-icon" aria-hidden="true" />
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
import TaijiLogo from './TaijiLogo.vue'

const props = defineProps({
  width: { type: Number, default: 260 },
  isResizing: { type: Boolean, default: false },
})

defineEmits(['resize-start'])

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
.sidebar-header { padding: 16px 14px 10px; }
.sidebar-logo { display: flex; align-items: center; gap: 12px; min-width: 0; }
.logo-icon-wrap {
  width: 44px; height: 44px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 14px;
  flex-shrink: 0;
  transition: background-color 0.2s var(--ease), border-color 0.2s var(--ease);
}
.logo-icon-wrap:hover { background: var(--bg-hover); }
.brand-copy { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 2px; }
.brand-copy h2 {
  margin: 0; font-size: 15px; font-weight: 700; color: var(--text); line-height: 1.2;
  font-family: var(--font-display);
  letter-spacing: 0.02em;
}
.brand-copy span { color: var(--text-muted); font-size: 11px; line-height: 1.35; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Runtime card */
.runtime-card {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-top: 12px; padding: 9px 10px;
  border-radius: var(--radius-lg);
  background: var(--bg-muted);
  border: 1px solid var(--border-subtle);
}
.runtime-state { display: flex; align-items: center; gap: 7px; min-width: 0; }
.runtime-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.runtime-card.connected .runtime-dot { background: var(--success); box-shadow: 0 0 0 3px var(--success-light); }
.runtime-card.loading .runtime-dot, .runtime-card.connecting .runtime-dot { background: var(--warning); animation: pulse 1.5s infinite; }
.runtime-card.error .runtime-dot { background: var(--danger); }
.runtime-label { color: var(--text-secondary); font-size: 12px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.memory-badge { flex-shrink: 0; }

/* New chat button */
.new-chat-btn {
  margin: 6px 12px 10px; height: 34px;
  display: flex; align-items: center; justify-content: center; gap: 7px;
  border: 1px solid var(--primary-light);
  border-radius: var(--radius-lg);
  background: var(--primary-subtle);
  color: var(--primary-hover);
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: background-color 0.2s var(--ease), border-color 0.2s var(--ease), color 0.2s var(--ease);
}
.new-chat-btn:hover { background: var(--primary-light); border-color: var(--primary); color: var(--text); }

/* Session list */
.session-list { flex: 1; min-height: 0; overflow-y: auto; padding: 0 8px 10px; }
.section-label {
  padding: 9px 8px 5px; font-size: 10.5px; color: var(--text-muted);
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
}
.session-item {
  display: flex; align-items: center; justify-content: space-between;
  gap: 6px; min-height: 32px; padding: 6px 9px;
  border-radius: var(--radius-md);
  border: 1px solid transparent; cursor: pointer;
  transition: background-color 0.15s var(--ease), border-color 0.15s var(--ease);
}
.session-item:hover { background: var(--bg-hover); border-color: var(--border-subtle); }
.session-item.active { background: var(--primary-subtle); border-color: var(--primary-light); }
.session-item:focus-visible { outline: 2px solid var(--primary); outline-offset: -1px; }
.session-name {
  display: flex; align-items: center; gap: 7px; min-width: 0; font-size: 13px;
  color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.session-item.active .session-name { color: var(--text); font-weight: 600; }
.session-icon { flex-shrink: 0; color: var(--text-muted); }
.session-del-btn {
  width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-muted);
  cursor: pointer; opacity: 0; flex-shrink: 0;
  transition: opacity 0.15s var(--ease), color 0.15s var(--ease);
}
.session-item:hover .session-del-btn { opacity: 0.5; }
.session-del-btn:hover { opacity: 1 !important; color: var(--danger); background: var(--danger-light); }

/* Footer nav */
.sidebar-footer {
  padding: 9px 8px 12px;
  border-top: 1px solid var(--border);
  max-height: 42vh; overflow-y: auto;
}
.nav-group { margin-bottom: 4px; }
.nav-group .section-label { padding: 8px 8px 4px; }
.settings-btn {
  display: flex; align-items: center; gap: 8px;
  min-height: 32px; padding: 6px 9px;
  border: 1px solid transparent; border-radius: var(--radius-md);
  color: var(--text-secondary); font-size: 13px; text-decoration: none;
  transition: background-color 0.15s var(--ease), border-color 0.15s var(--ease), color 0.15s var(--ease);
}
.settings-btn:hover { background: var(--bg-hover); color: var(--text); border-color: var(--border-subtle); }
.settings-btn:focus-visible { outline: 2px solid var(--primary); outline-offset: -1px; }
.settings-btn.active { background: var(--primary-subtle); border-color: var(--primary-light); color: var(--primary-hover); font-weight: 600; }
.nav-icon { flex-shrink: 0; }
.nav-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>

<style>
/* 全局布局 */
.sidebar { height: 100%; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }

@media (max-width: 768px) {
  .sidebar { width: 56px !important; min-width: 56px !important; }
  .sidebar-header { padding: 14px 8px 8px !important; }
  .sidebar-logo { justify-content: center; }
  .brand-copy, .runtime-label, .memory-badge, .section-label,
  .session-name, .nav-label, .runtime-card { display: none !important; }
  .new-chat-btn { width: 36px; height: 32px; padding: 0 !important; margin: 10px auto !important; font-size: 0 !important; }
  .session-list { padding: 0 6px 8px !important; }
  .session-item { width: 36px; height: 32px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
  .session-del-btn { display: none !important; }
  .sidebar-footer { padding: 8px 6px 10px !important; }
  .settings-btn { width: 36px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
}
</style>
