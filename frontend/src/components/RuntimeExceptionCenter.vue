/**
 * 运行时异常中心组件
 * ===================
 *
 * 显示运行时异常，告诉用户：
 * - 发生了什么
 * - 影响什么
 * - 怎么恢复
 */
<template>
  <div v-if="runtimeStore.issues.length" class="exception-center">
    <transition-group name="exception" tag="div" class="exception-list">
      <div
        v-for="issue in runtimeStore.issues.slice(0, 3)"
        :key="issue.title"
        class="exception-card"
        :class="'exception-' + issue.level"
      >
        <div class="exception-icon">
          <span v-if="issue.level === 'danger'">🔴</span>
          <span v-else-if="issue.level === 'warning'">🟡</span>
          <span v-else>ℹ️</span>
        </div>
        <div class="exception-body">
          <div class="exception-title">{{ issue.title }}</div>
          <div class="exception-message">{{ issue.message }}</div>
        </div>
        <button class="exception-dismiss" @click="dismissIssue(issue)">×</button>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const runtimeStore = useRuntimeStore()

function dismissIssue(issue) {
  // runtimeStore.issues 是 computed，不能直接修改
  // 这里通过清除对应的源状态来 dismiss
  if (issue.title === '运行时断开') {
    runtimeStore.syncHealth('connecting')
  } else if (issue.title === '内存告急') {
    // 内存问题不能 dismiss，只能等待
  } else if (issue.title === '终端不可用') {
    runtimeStore.syncTerminal('disconnected', '')
  } else if (issue.title === '需要重新登录') {
    runtimeStore.auth.error = ''
  }
}
</script>

<style scoped>
.exception-center {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9990;
  max-width: 380px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.exception-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.exception-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  border-radius: 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(12px);
  animation: slideIn 0.3s ease;
}

.exception-card.exception-danger {
  border-left: 3px solid var(--danger);
}

.exception-card.exception-warning {
  border-left: 3px solid var(--warning);
}

.exception-card.exception-info {
  border-left: 3px solid var(--info);
}

.exception-icon {
  font-size: 16px;
  flex-shrink: 0;
  margin-top: 2px;
}

.exception-body {
  flex: 1;
  min-width: 0;
}

.exception-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}

.exception-message {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.exception-dismiss {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
  flex-shrink: 0;
}

.exception-dismiss:hover {
  color: var(--text);
}

@keyframes slideIn {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}

/* 过渡动画 */
.exception-enter-active { animation: slideIn 0.3s ease; }
.exception-leave-active { animation: slideIn 0.3s ease reverse; }
</style>
