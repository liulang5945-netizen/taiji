import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { locales } from '@/locales/index.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'

// 防抖保存 UI 设置到后端
let _uiSaveTimer = null
function _debouncedSaveUI(data) {
  if (_uiSaveTimer) clearTimeout(_uiSaveTimer)
  _uiSaveTimer = setTimeout(async () => {
    try {
      await authFetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
    } catch (e) { /* silent fail */ }
  }, 1000)
}

export const useAppStore = defineStore('app', () => {
  // === State ===
  const currentTheme = ref(localStorage.getItem('taiji_theme') || 'dark')
  const currentAccent = ref(localStorage.getItem('taiji_accent') || '')
  const currentBgImage = ref(localStorage.getItem('taiji_bg_image') || '')
  const currentLang = ref('zh')
  const showWorkspace = ref(false)

  // 连接状态
  const connectionState = ref('unknown')
  const connectionErrorMsg = ref('')
  const retryCountdown = ref(0)
  const modelLoaded = ref(false)

  // === Getters ===
  const connectionClass = computed(() => {
    if (connectionState.value === 'connected') return 'connected'
    if (connectionState.value === 'loading') return 'loading'
    if (connectionState.value === 'connecting') return 'connecting'
    return 'error'
  })

  const connectionStatus = computed(() => {
    if (connectionState.value === 'connected') return modelLoaded.value ? t('status_connected') : t('status_connected_no_model')
    if (connectionState.value === 'connecting') return t('status_connecting')
    if (connectionState.value === 'loading') {
      if (retryCountdown.value > 0) return t('retry', { n: retryCountdown.value })
      return t('status_model_loading')
    }
    return t('status_error')
  })

  const resolvedTheme = computed(() => {
    if (currentTheme.value === 'auto') {
      if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
      }
      return 'dark'
    }
    return currentTheme.value
  })

  // === Helpers ===
  function t(key, params = {}) {
    let text = locales[currentLang.value][key] || locales['zh'][key] || key
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, v)
    }
    return text
  }

  // === Actions ===
  function toggleWorkspace() {
    showWorkspace.value = !showWorkspace.value
  }

  function setConnectionState(state, errorMsg = '', loaded = false) {
    connectionState.value = state
    connectionErrorMsg.value = errorMsg
    modelLoaded.value = loaded
  }

  // 预设主题色
  const accentPresets = [
    { name: '默认灰蓝', color: '#5b7a8a' },
    { name: '靛蓝', color: '#4f46e5' },
    { name: '翠绿', color: '#059669' },
    { name: '琥珀', color: '#d97706' },
    { name: '玫瑰', color: '#e11d48' },
    { name: '紫色', color: '#7c3aed' },
    { name: '青色', color: '#0891b2' },
    { name: '石墨', color: '#475569' },
  ]

  function applyTheme() {
    const r = document.documentElement
    r.classList.remove('theme-dark', 'theme-light')
    if (resolvedTheme.value === 'dark') {
      r.classList.add('theme-dark')
      r.setAttribute('data-theme', 'dark')
    } else if (resolvedTheme.value === 'light') {
      r.classList.add('theme-light')
      r.setAttribute('data-theme', 'light')
    } else {
      r.removeAttribute('data-theme')
    }
    applyAccent()
    applyBgImage()
  }

  function applyAccent() {
    const hex = currentAccent.value
    if (!hex) return
    const r = document.documentElement
    const rgb = hexToRgb(hex)
    if (!rgb) return
    r.style.setProperty('--primary', hex)
    r.style.setProperty('--primary-hover', darken(hex, 15))
    r.style.setProperty('--primary-light', `rgba(${rgb.r},${rgb.g},${rgb.b},0.08)`)
    r.style.setProperty('--primary-subtle', `rgba(${rgb.r},${rgb.g},${rgb.b},0.04)`)
    r.style.setProperty('--primary-gradient', `linear-gradient(135deg, ${hex} 0%, ${lighten(hex, 20)} 100%)`)
  }

  function applyBgImage() {
    const wrapper = document.querySelector('.app-wrapper')
    if (!wrapper) return
    if (currentBgImage.value) {
      wrapper.style.backgroundImage = `url(${currentBgImage.value})`
      wrapper.style.backgroundSize = 'cover'
      wrapper.style.backgroundPosition = 'center'
      wrapper.style.backgroundAttachment = 'fixed'
    } else {
      wrapper.style.backgroundImage = ''
    }
  }

  function hexToRgb(hex) {
    const m = hex.replace('#', '').match(/^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)
    return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : null
  }

  function darken(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = 1 - pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c * f).toString(16).padStart(2, '0')).join('')
  }

  function lighten(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c + (255 - c) * f).toString(16).padStart(2, '0')).join('')
  }

  function setTheme(theme) {
    currentTheme.value = theme
    localStorage.setItem('taiji_theme', theme)
    applyTheme()
    _debouncedSaveUI({ theme })
  }

  function setAccent(color) {
    currentAccent.value = color
    localStorage.setItem('taiji_accent', color)
    applyAccent()
    _debouncedSaveUI({ accent: color })
  }

  function setBgImage(dataUrl) {
    currentBgImage.value = dataUrl
    if (dataUrl) {
      localStorage.setItem('taiji_bg_image', dataUrl)
    } else {
      localStorage.removeItem('taiji_bg_image')
    }
    applyBgImage()
  }

  function restoreUISettings(serverSettings) {
    if (!serverSettings || typeof serverSettings !== 'object') return
    let needsApplyTheme = false

    if (serverSettings.accent !== undefined) {
      currentAccent.value = serverSettings.accent
      localStorage.setItem('taiji_accent', serverSettings.accent)
      needsApplyTheme = true
    }
    if (serverSettings.theme !== undefined) {
      currentTheme.value = serverSettings.theme
      localStorage.setItem('taiji_theme', serverSettings.theme)
      needsApplyTheme = true
    }
    if (serverSettings.lang !== undefined) {
      currentLang.value = serverSettings.lang
      localStorage.setItem('taiji_lang', serverSettings.lang)
    }
    if (needsApplyTheme) {
      applyTheme()
    }
  }

  // 初始化主题
  applyTheme()
  if (typeof window !== 'undefined' && window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener?.('change', () => {
      if (currentTheme.value === 'auto') applyTheme()
    })
  }

  return {
    // State
    currentTheme,
    currentAccent,
    currentBgImage,
    currentLang,
    showWorkspace,
    connectionState,
    connectionErrorMsg,
    retryCountdown,
    modelLoaded,
    // Getters
    connectionClass,
    connectionStatus,
    resolvedTheme,
    // Actions
    t,
    toggleWorkspace,
    setConnectionState,
    applyTheme,
    setTheme,
    setAccent,
    setBgImage,
    accentPresets,
    restoreUISettings,
  }
})
