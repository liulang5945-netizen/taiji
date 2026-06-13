/**
 * API 连接与健康检查 composable
 * 态极专属 — 无需模型切换/下载等功能
 */
import { ref, computed } from 'vue';
import { useAppStore } from '../stores/appStore.js';
import { useRuntimeStore } from '../stores/runtimeStore.js';
import { API_BASE, authFetch } from './apiClient.js';

/**
 * 动态 API 基地址：
 * - 开发环境：Vite proxy 拦截 /api，使用空字符串（相对路径）
 * - 生产环境：自动使用当前页面的 host
 */
export { API_BASE, authFetch };

// === 模块级共享状态 ===
const connectionState = ref('unknown');
const connectionErrorMsg = ref('');
const retryCountdown = ref(0);
const downloadProgress = ref(null);
const currentLang = ref('zh');
const taijiAvailable = ref(true);
const modelLoaded = ref(false);
let healthCheckTimer = null;
let retryTimer = null;
let consecutiveFailures = 0;
let isChatReceiving = false;

/**
 * 模块级函数：设置聊天接收状态
 */
export function setChatReceiving(receiving) {
  isChatReceiving = receiving;
}

export function useApi() {
  const appStore = useAppStore();
  const runtimeStore = useRuntimeStore();
  const t = (key, params = {}) => appStore.t(key, params);

  const connectionClass = computed(() => {
    if (connectionState.value === 'connected') return 'connected';
    if (connectionState.value === 'downloading') return 'downloading';
    if (connectionState.value === 'loading') return 'loading';
    if (connectionState.value === 'connecting') return 'connecting';
    return 'error';
  });

  const connectionStatus = computed(() => {
    if (connectionState.value === 'connected') return modelLoaded.value ? t('status_connected') : t('status_connected_no_model');
    if (connectionState.value === 'connecting') return t('status_connecting');
    if (connectionState.value === 'loading') {
      if (retryCountdown.value > 0) return t('retry', { n: retryCountdown.value });
      return t('status_model_loading');
    }
    return t('status_error');
  });

  function syncAppState(state, msg = '') {
    connectionState.value = state;
    connectionErrorMsg.value = msg;
    appStore.setConnectionState(state, msg, modelLoaded.value);
    runtimeStore.syncHealth(state, msg, modelLoaded.value);
  }

  async function checkHealth() {
    try {
      const controller = new AbortController();
      const timeout = 15000;
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      const resp = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!resp.ok) {
        syncAppState('error', `后端返回错误 (HTTP ${resp.status})`);
        return false;
      }

      const ctype = resp.headers.get('content-type') || '';
      if (!ctype.includes('application/json')) {
        syncAppState('error', '后端返回了非JSON响应，可能正在启动中...');
        return false;
      }

      const data = await resp.json();
      consecutiveFailures = 0;
      if (data.status === 'ok') {
        modelLoaded.value = !!data.model_loaded;
        syncAppState('connected');
        downloadProgress.value = null;
        retryCountdown.value = 0;
        taijiAvailable.value = true;
        clearRetryTimer();
        return true;
      } else if (data.status === 'downloading') {
        modelLoaded.value = false;
        downloadProgress.value = data;
        syncAppState('downloading', data.message || '模型正在下载...');
        return false;
      } else if (data.status === 'loading') {
        consecutiveFailures = 0;
        if (connectionState.value !== 'connected') {
          syncAppState('loading', data.message || '模型正在加载中...');
        }
        return false;
      } else {
        syncAppState('error', data.message || '后端报告错误');
        return false;
      }
    } catch (err) {
      if (connectionState.value === 'loading') {
        consecutiveFailures = 0;
        return false;
      }
      if (connectionState.value === 'unknown' || connectionState.value === 'connecting') {
        syncAppState('connecting', '正在连接后端服务...');
        consecutiveFailures = 0;
        return false;
      }
      if (isChatReceiving) return false;
      consecutiveFailures++;
      if (consecutiveFailures >= 5) {
        syncAppState('error', err.message);
      }
      return false;
    }
  }

  function startHealthCheck() {
    stopHealthCheck();
    consecutiveFailures = 0;
    runtimeStore.refreshAll().catch(() => {});
    checkHealth().catch(() => {});
    healthCheckTimer = setInterval(async () => {
      try {
        const connected = await checkHealth();
        if (connected) {
          clearInterval(healthCheckTimer);
          healthCheckTimer = setInterval(() => {
            checkHealth().catch(() => {});
            runtimeStore.refreshAll().catch(() => {});
          }, 10000);
        }
      } catch (e) {
        console.warn('[healthCheck] 健康检查异常:', e);
      }
    }, 2000);
  }

  function stopHealthCheck() {
    if (healthCheckTimer) {
      clearInterval(healthCheckTimer);
      healthCheckTimer = null;
    }
    clearRetryTimer();
  }

  function clearRetryTimer() {
    if (retryTimer) { clearInterval(retryTimer); retryTimer = null; }
  }

  return {
    API_BASE,
    connectionState, connectionErrorMsg, retryCountdown, downloadProgress,
    connectionClass, connectionStatus, currentLang,
    t, checkHealth, startHealthCheck, stopHealthCheck,
    setChatReceiving,
    taijiAvailable, modelLoaded,
  };
}
