/**
 * 态极 WebSocket 客户端
 * ====================
 *
 * 前端与态极核心的实时通信。
 */
import { ref, onMounted, onUnmounted } from 'vue'

// 动态 WebSocket URL：开发环境用 localhost，生产环境用当前 host
const WS_URL = import.meta.env.DEV
  ? 'ws://localhost:8765'
  : `ws://${window.location.hostname}:8765`

export function useWebSocket() {
  const connected = ref(false)
  const taijiStatus = ref(null)
  const lastMessage = ref(null)

  let ws = null
  let reconnectTimer = null
  const listeners = new Map()

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return

    ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      connected.value = true
      reconnectAttempt = 0
      console.log('已连接到态极')
      clearReconnect()
    }

    ws.onclose = () => {
      connected.value = false
      console.log('与态极断开连接')
      scheduleReconnect()
    }

    ws.onerror = (error) => {
      console.error('WebSocket 错误:', error)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        lastMessage.value = data
        handleMessage(data)
      } catch (e) {
        console.error('解析消息失败:', e)
      }
    }
  }

  function disconnect() {
    clearReconnect()
    if (ws) {
      ws.close()
      ws = null
    }
  }

  let reconnectAttempt = 0
  const MAX_RECONNECT_ATTEMPTS = 10

  function scheduleReconnect() {
    clearReconnect()
    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('WebSocket 重连次数超限，停止重连')
      return
    }
    const delay = Math.min(3000 * Math.pow(2, reconnectAttempt), 30000)
    reconnectAttempt++
    reconnectTimer = setTimeout(() => {
      console.log(`尝试重新连接... (第${reconnectAttempt}次)`)
      connect()
    }, delay)
  }

  function clearReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  function handleMessage(data) {
    const { type } = data

    // 触发监听器
    const typeListeners = listeners.get(type)
    if (typeListeners) {
      typeListeners.forEach(callback => callback(data))
    }

    // 更新状态
    if (type === 'status_response') {
      taijiStatus.value = data.status
    }
  }

  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    } else {
      console.warn('未连接到态极')
    }
  }

  function on(type, callback) {
    if (!listeners.has(type)) {
      listeners.set(type, new Set())
    }
    listeners.get(type).add(callback)

    // 返回取消监听函数
    return () => {
      const typeListeners = listeners.get(type)
      if (typeListeners) {
        typeListeners.delete(callback)
      }
    }
  }

  // 聊天
  function chat(message) {
    send({ type: 'chat', message })
  }

  // 喂养
  function feed(content, contentType = 'text') {
    send({ type: 'feed', content, content_type: contentType })
  }

  // 训练
  function train(epochs = 3, learningRate = 5e-5) {
    send({ type: 'train', epochs, learning_rate: learningRate })
  }

  // 睡眠
  function sleep() {
    send({ type: 'sleep' })
  }

  // 玩耍
  function play() {
    send({ type: 'play' })
  }

  // 语音识别
  function listenVoice() {
    send({ type: 'voice', action: 'listen' })
  }

  // 语音合成
  function speakText(text) {
    send({ type: 'voice', action: 'speak', text })
  }

  // 获取状态
  function getStatus() {
    send({ type: 'status' })
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    connected,
    taijiStatus,
    lastMessage,
    connect,
    disconnect,
    send,
    on,
    chat,
    feed,
    train,
    sleep,
    play,
    listenVoice,
    speakText,
    getStatus,
  }
}
