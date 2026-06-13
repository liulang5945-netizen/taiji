<template>
  <section class="agent-view">
    <header class="agent-header">
      <div>
        <p class="eyebrow">能力面板</p>
        <h1><Bot :size="24" /> Agent</h1>
      </div>
      <button class="icon-btn" @click="refreshAgentRuntime" :disabled="runtimeStore.toolsLoading || mcpLoading" title="刷新能力">
        <RefreshCw :size="16" :class="{ spin: runtimeStore.toolsLoading || mcpLoading }" />
      </button>
    </header>

    <div v-if="runtimeStore.issues.length" class="runtime-alerts">
      <div v-for="issue in runtimeStore.issues" :key="issue.title" class="runtime-alert" :class="issue.level">
        <strong>{{ issue.title }}</strong>
        <span>{{ issue.message }}</span>
      </div>
    </div>

    <div class="agent-layout">
      <aside class="agent-side">
        <div class="runtime-panel">
          <div class="runtime-head">
            <span class="runtime-dot" :class="runtimeStore.connectionClass"></span>
            <div>
              <strong>{{ runtimeStore.modelLifecycle.title }}</strong>
              <small>{{ runtimeStore.modelLifecycle.message }}</small>
            </div>
          </div>
          <div class="runtime-facts">
            <span>工具 {{ runtimeStore.tools.length }}</span>
            <span>MCP {{ installedServers.length }}</span>
            <span v-if="runtimeStore.memoryAvailableGb !== null">内存 {{ runtimeStore.memoryAvailableGb.toFixed(1) }}GB</span>
          </div>
        </div>

        <div class="panel-section compact">
          <div class="panel-title">
            <Sliders :size="17" />
            <span>运行参数</span>
          </div>
          <label class="param-item">
            <span>最大迭代</span>
            <input v-model.number="maxIterations" type="number" min="1" max="50" class="form-input" @change="saveAgentPrefs" />
          </label>
          <label class="param-item">
            <span>温度</span>
            <input v-model.number="temperature" type="number" min="0" max="2" step="0.1" class="form-input" @change="saveAgentPrefs" />
          </label>
          <p class="panel-note">这些参数会在对话中的“行动”模式生效。</p>
        </div>
      </aside>

      <main class="agent-main">
        <div class="capability-grid">
          <div v-for="item in capabilityModules" :key="item.key" class="capability-card" :class="{ active: item.count > 0 }">
            <component :is="item.icon" :size="20" />
            <div>
              <strong>{{ item.title }}</strong>
              <span>{{ item.count ? `${item.count} 个能力已装载` : item.empty }}</span>
            </div>
          </div>
        </div>

        <div class="agent-tabs">
          <button :class="{ active: activeTab === 'tools' }" @click="activeTab = 'tools'">
            <Wrench :size="16" />
            工具注册表
          </button>
          <button :class="{ active: activeTab === 'installed' }" @click="activeTab = 'installed'; loadInstalled()">
            <PackageIcon :size="16" />
            MCP 运行
          </button>
          <button :class="{ active: activeTab === 'marketplace' }" @click="activeTab = 'marketplace'; loadMarketplace()">
            <Puzzle :size="16" />
            MCP 市场
          </button>
        </div>

        <section v-if="activeTab === 'tools'" class="panel-section">
          <div class="panel-title">
            <Wrench :size="17" />
            <span>真实装载工具</span>
            <small>{{ runtimeStore.toolsLoading ? '同步中...' : `${filteredTools.length}/${runtimeStore.tools.length}` }}</small>
          </div>
          <div class="toolbar-row">
            <input v-model="toolQuery" class="form-input search-input" placeholder="筛选工具、来源或描述" />
            <select v-model="toolSource" class="form-input source-select">
              <option value="">全部来源</option>
              <option v-for="source in toolSources" :key="source" :value="source">{{ source }}</option>
            </select>
          </div>

          <div v-if="filteredTools.length" class="tools-grid">
            <article v-for="tool in filteredTools" :key="tool.name" class="tool-card">
              <div class="tool-card-head">
                <span class="tool-kind" :class="tool.source || 'local'">{{ sourceLabel(tool.source) }}</span>
                <span class="tool-state" :class="{ enabled: tool.enabled !== false }">
                  {{ tool.enabled === false ? '停用' : '可用' }}
                </span>
              </div>
              <h3>{{ tool.name }}</h3>
              <p>{{ tool.description || '暂无描述' }}</p>
              <div class="tool-meta">
                <span>{{ categoryLabel(tool) }}</span>
                <span>{{ permissionLabel(tool) }}</span>
                <span v-if="tool.source_id">{{ tool.source_id }}</span>
              </div>
            </article>
          </div>
          <div v-else class="empty-panel">
            {{ runtimeStore.toolsLoading ? '正在同步工具注册表...' : '当前没有匹配的工具。' }}
          </div>
        </section>

        <section v-if="activeTab === 'installed'" class="panel-section">
          <div class="panel-title">
            <PackageIcon :size="17" />
            <span>MCP 运行状态</span>
            <small>{{ installedServers.length }} 个服务</small>
          </div>
          <div v-if="installedServers.length" class="mcp-grid">
            <article v-for="server in installedServers" :key="server.id" class="mcp-card">
              <div class="mcp-card-head">
                <Puzzle :size="18" />
                <div>
                  <h3>{{ server.name || server.id }}</h3>
                  <small>{{ server.npm_package || server.id }}</small>
                </div>
                <span class="mcp-status" :class="{ running: server.running }">
                  {{ server.running ? '运行中' : '已停止' }}
                </span>
              </div>
              <p>{{ server.description || '暂无描述' }}</p>
              <div v-if="server.runtime_info?.tools?.length" class="mcp-tools">
                <span v-for="tool in server.runtime_info.tools.slice(0, 6)" :key="tool.name">{{ tool.name }}</span>
                <span v-if="server.runtime_info.tools.length > 6">+{{ server.runtime_info.tools.length - 6 }}</span>
              </div>
              <div class="mcp-actions">
                <button v-if="!server.running" class="btn-success" @click="startServer(server.id)" :disabled="server._starting">
                  <Play :size="13" /> 启动
                </button>
                <button v-else class="btn-warning" @click="stopServer(server.id)">
                  <Square :size="13" /> 停止
                </button>
                <button v-if="server.running" class="btn-secondary" @click="restartServer(server.id)">
                  <RefreshCw :size="13" /> 重启
                </button>
                <button class="btn-danger" @click="uninstallServer(server.id)">
                  <Trash2 :size="13" /> 卸载
                </button>
              </div>
            </article>
          </div>
          <div v-else class="empty-panel">暂无已安装 MCP 服务。</div>
        </section>

        <section v-if="activeTab === 'marketplace'" class="panel-section">
          <div class="panel-title">
            <Puzzle :size="17" />
            <span>MCP 市场</span>
            <small>{{ marketplaceServers.length }} 个结果</small>
          </div>
          <div class="toolbar-row">
            <input v-model="mcpSearch" class="form-input search-input" placeholder="搜索 MCP 服务" @input="debounceSearch" />
            <select v-model="mcpCategory" class="form-input source-select" @change="loadMarketplace">
              <option value="">全部分类</option>
              <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
            </select>
            <button class="btn-secondary" @click="refreshMarketplace" :disabled="refreshing">
              <RefreshCw :size="13" :class="{ spin: refreshing }" />
              同步市场
            </button>
          </div>
          <div v-if="marketplaceServers.length" class="mcp-grid">
            <article v-for="server in marketplaceServers" :key="server.id" class="mcp-card">
              <div class="mcp-card-head">
                <Puzzle :size="18" />
                <div>
                  <h3>{{ server.name }}</h3>
                  <small>{{ server.category || '未分类' }}</small>
                </div>
                <span v-if="server.running" class="mcp-status running">运行中</span>
                <span v-else-if="server.installed" class="mcp-status">已安装</span>
              </div>
              <p>{{ server.description || '暂无描述' }}</p>
              <div class="tool-meta">
                <span>{{ server.npm_package }}</span>
                <span>{{ server.tools_count || '?' }} 个工具</span>
                <span v-if="server.rating">评分 {{ server.rating }}</span>
              </div>
              <div class="mcp-actions">
                <button v-if="!server.installed" class="btn-primary" @click="installServer(server.id)" :disabled="server._installing">
                  <Download :size="13" /> 安装
                </button>
                <template v-else>
                  <button v-if="!server.running" class="btn-success" @click="startServer(server.id)" :disabled="server._starting">
                    <Play :size="13" /> 启动
                  </button>
                  <button v-else class="btn-warning" @click="stopServer(server.id)">
                    <Square :size="13" /> 停止
                  </button>
                  <button class="btn-danger" @click="uninstallServer(server.id)">
                    <Trash2 :size="13" /> 卸载
                  </button>
                </template>
              </div>
            </article>
          </div>
          <div v-else class="empty-panel">
            {{ mcpLoading ? '正在加载 MCP 市场...' : '暂无匹配结果。' }}
          </div>
        </section>
      </main>
    </div>
  </section>
</template>

<script setup>
import { computed, inject, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  Bot, Wrench, Puzzle, Package as PackageIcon, Sliders, Globe, Eye, FileText,
  TerminalSquare, Image as ImageIcon, RefreshCw, Play, Square, Trash2, Download
} from 'lucide-vue-next'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'
import { useSettings } from '@/composables/useSettings.js'

const { saveSettingsToServer } = useSettings()
const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})
const $confirm = inject('$confirm', () => Promise.resolve(false))

const activeTab = ref('tools')
const toolQuery = ref('')
const toolSource = ref('')
const maxIterations = ref(runtimeStore.agentPrefs.maxIterations)
const temperature = ref(runtimeStore.agentPrefs.temperature)

const marketplaceServers = ref([])
const installedServers = ref([])
const categories = ref([])
const mcpSearch = ref('')
const mcpCategory = ref('')
const mcpLoading = ref(false)
const refreshing = ref(false)
let searchTimer = null

const toolSources = computed(() => {
  const values = new Set(runtimeStore.tools.map(tool => tool.source || 'local'))
  return Array.from(values).sort()
})

const filteredTools = computed(() => {
  const q = toolQuery.value.trim().toLowerCase()
  return runtimeStore.normalizedTools.filter(tool => {
    const source = tool.source || 'local'
    if (toolSource.value && source !== toolSource.value) return false
    if (!q) return true
    return tool.searchText.includes(q)
  })
})

const capabilityModules = computed(() => [
  { key: 'network', title: '联网搜索', count: runtimeStore.toolGroups.network.length, empty: '等待联网工具', icon: Globe },
  { key: 'browser', title: '网页浏览', count: runtimeStore.toolGroups.browsing.length, empty: '等待浏览工具', icon: Eye },
  { key: 'knowledge', title: '知识与文件', count: runtimeStore.toolGroups.knowledge.length, empty: '等待知识工具', icon: FileText },
  { key: 'action', title: '代码与终端', count: runtimeStore.toolGroups.action.length, empty: '等待执行工具', icon: TerminalSquare },
  { key: 'multimodal', title: '多模态', count: runtimeStore.toolGroups.multimodal.length, empty: '等待图像/语音/视频工具', icon: ImageIcon },
])

watch(() => runtimeStore.agentPrefs, (prefs) => {
  maxIterations.value = prefs.maxIterations
  temperature.value = prefs.temperature
}, { deep: true })

function sourceLabel(source) {
  if (!source || source === 'local') return '本地'
  if (source === 'mcp') return 'MCP'
  if (source === 'plugin') return '插件'
  return source
}

function categoryLabel(tool) {
  return tool.category || (tool.source === 'mcp' ? 'MCP' : '通用')
}

function permissionLabel(tool) {
  const text = tool.searchText || ''
  if (/shell|terminal|execute|delete|write|终端|执行|删除|写入/.test(text)) return '高权限'
  if (/web|search|browser|http|联网|搜索|浏览/.test(text)) return '联网'
  if (/file|document|kb|文件|文档|知识/.test(text)) return '文件/知识'
  return '只读/低风险'
}

function saveAgentPrefs() {
  runtimeStore.setAgentPrefs({
    maxIterations: maxIterations.value,
    temperature: temperature.value,
  })
  saveSettingsToServer({
    agent_max_iterations: runtimeStore.agentPrefs.maxIterations,
    agent_temperature: runtimeStore.agentPrefs.temperature,
  }).catch(() => {})
}

async function refreshAgentRuntime() {
  await Promise.allSettled([
    runtimeStore.refreshAll(),
    loadInstalled(),
    activeTab.value === 'marketplace' ? loadMarketplace() : Promise.resolve(),
  ])
}

function debounceSearch() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => loadMarketplace(), 300)
}

async function loadMarketplace() {
  mcpLoading.value = true
  try {
    const params = new URLSearchParams()
    if (mcpSearch.value) params.set('keyword', mcpSearch.value)
    if (mcpCategory.value) params.set('category', mcpCategory.value)
    const r = await authFetch(`${API_BASE}/api/mcp/marketplace?${params}`)
    if (r.ok) {
      const data = await r.json()
      marketplaceServers.value = (data.servers || []).map(s => ({ ...s, _installing: false, _starting: false }))
      categories.value = data.categories || []
    }
  } catch (e) {
    toast(`MCP 市场加载失败: ${e.message}`, 'error')
  } finally {
    mcpLoading.value = false
  }
}

async function refreshMarketplace() {
  refreshing.value = true
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/marketplace/refresh`, { method: 'POST' })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    toast('正在同步 MCP 市场', 'info')
    setTimeout(loadMarketplace, 1800)
  } catch (e) {
    toast(`同步失败: ${e.message}`, 'error')
  } finally {
    refreshing.value = false
  }
}

async function loadInstalled() {
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/installed`)
    if (r.ok) {
      const data = await r.json()
      installedServers.value = (data.servers || []).map(s => ({ ...s, _starting: false }))
    }
  } catch (e) {
    toast(`MCP 状态加载失败: ${e.message}`, 'error')
  }
}

async function installServer(id) {
  const server = marketplaceServers.value.find(s => s.id === id)
  if (server) server._installing = true
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/install`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    await loadMarketplace()
    await loadInstalled()
    await runtimeStore.refreshTools()
  } catch (e) {
    toast(`安装失败: ${e.message}`, 'error')
  } finally {
    if (server) server._installing = false
  }
}

async function startServer(id) {
  const server = marketplaceServers.value.find(s => s.id === id) || installedServers.value.find(s => s.id === id)
  if (server) server._starting = true
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    await refreshAgentRuntime()
  } catch (e) {
    toast(`启动失败: ${e.message}`, 'error')
  } finally {
    if (server) server._starting = false
  }
}

async function stopServer(id) {
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    await refreshAgentRuntime()
  } catch (e) {
    toast(`停止失败: ${e.message}`, 'error')
  }
}

async function restartServer(id) {
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/restart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    await refreshAgentRuntime()
  } catch (e) {
    toast(`重启失败: ${e.message}`, 'error')
  }
}

async function uninstallServer(id) {
  const ok = await $confirm({ title: '卸载确认', message: `确定卸载 MCP 服务 "${id}"？`, type: 'danger' })
  if (!ok) return
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/uninstall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    await refreshAgentRuntime()
  } catch (e) {
    toast(`卸载失败: ${e.message}`, 'error')
  }
}

onMounted(() => {
  refreshAgentRuntime()
})

onUnmounted(() => {
  if (searchTimer) {
    clearTimeout(searchTimer)
    searchTimer = null
  }
})
</script>

<style scoped>
.agent-view {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  color: var(--text);
  overflow: hidden;
}

.agent-header {
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 22px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--text-muted);
  font-size: 12px;
}

.agent-header h1 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 22px;
  letter-spacing: 0;
}

.icon-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border);
  border-radius: 9px;
  color: var(--text-secondary);
  background: var(--bg-card);
  cursor: pointer;
}

.runtime-alerts {
  display: grid;
  gap: 8px;
  padding: 12px 22px 0;
}

.runtime-alert {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--text-secondary);
  font-size: 12px;
}

.runtime-alert strong {
  color: var(--text);
}

.runtime-alert.warning {
  background: var(--warning-light);
  border-color: color-mix(in srgb, var(--warning) 35%, transparent);
}

.runtime-alert.danger {
  background: var(--danger-light);
  border-color: color-mix(in srgb, var(--danger) 35%, transparent);
}

.agent-layout {
  min-height: 0;
  flex: 1;
  display: grid;
  grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
  gap: 18px;
  padding: 18px 22px 22px;
  overflow: hidden;
}

.agent-side,
.agent-main {
  min-height: 0;
  overflow-y: auto;
}

.agent-side {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.runtime-panel,
.panel-section {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.runtime-panel {
  padding: 14px;
}

.runtime-head {
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr);
  gap: 10px;
  align-items: start;
}

.runtime-dot {
  width: 10px;
  height: 10px;
  margin-top: 5px;
  border-radius: 999px;
  background: var(--text-muted);
}

.runtime-dot.connected { background: var(--success); }
.runtime-dot.loading,
.runtime-dot.downloading,
.runtime-dot.connecting { background: var(--warning); }
.runtime-dot.error { background: var(--danger); }

.runtime-head strong,
.panel-title span {
  color: var(--text);
  font-weight: 650;
}

.runtime-head small {
  display: block;
  margin-top: 4px;
  color: var(--text-muted);
  line-height: 1.45;
}

.runtime-facts,
.tool-meta,
.mcp-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.runtime-facts {
  margin-top: 12px;
}

.runtime-facts span,
.tool-meta span,
.mcp-tools span {
  padding: 3px 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-muted);
  color: var(--text-muted);
  font-size: 11px;
}

.panel-section {
  padding: 14px;
}

.panel-section.compact {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.panel-title small {
  margin-left: auto;
  color: var(--text-muted);
  font-size: 12px;
}

.param-item {
  display: grid;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
}

.form-input {
  width: 100%;
  min-width: 0;
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  outline: none;
}

.form-input:focus {
  border-color: var(--primary);
}

.panel-note {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.agent-main {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.capability-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
}

.capability-card {
  display: flex;
  gap: 10px;
  min-height: 72px;
  align-items: center;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-card);
  color: var(--text-muted);
}

.capability-card.active {
  color: var(--primary);
  background: var(--primary-subtle);
  border-color: color-mix(in srgb, var(--primary) 30%, transparent);
}

.capability-card strong,
.capability-card span {
  display: block;
}

.capability-card strong {
  color: var(--text);
  font-size: 13px;
}

.capability-card span {
  margin-top: 4px;
  color: var(--text-muted);
  font-size: 12px;
}

.agent-tabs,
.toolbar-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.agent-tabs button,
.btn-primary,
.btn-secondary,
.btn-success,
.btn-warning,
.btn-danger {
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--text-secondary);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.agent-tabs button.active {
  color: var(--primary);
  background: var(--primary-subtle);
  border-color: var(--primary-light);
}

.search-input {
  flex: 1;
  min-width: 220px;
}

.source-select {
  width: 150px;
}

.tools-grid,
.mcp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 12px;
}

.toolbar-row + .tools-grid,
.toolbar-row + .mcp-grid,
.toolbar-row + .empty-panel {
  margin-top: 12px;
}

.tool-card,
.mcp-card {
  min-width: 0;
  padding: 13px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--bg-muted);
}

.tool-card-head,
.mcp-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-card-head {
  justify-content: space-between;
}

.tool-kind,
.tool-state,
.mcp-status {
  padding: 3px 7px;
  border-radius: 999px;
  background: var(--bg-card);
  color: var(--text-muted);
  font-size: 11px;
  border: 1px solid var(--border);
}

.tool-state.enabled,
.mcp-status.running {
  color: var(--success);
  background: var(--success-light);
  border-color: color-mix(in srgb, var(--success) 35%, transparent);
}

.tool-card h3,
.mcp-card h3 {
  margin: 10px 0 6px;
  color: var(--text);
  font-size: 14px;
  word-break: break-word;
}

.mcp-card h3 {
  margin: 0;
}

.mcp-card-head > div {
  min-width: 0;
  flex: 1;
}

.mcp-card-head small {
  display: block;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tool-card p,
.mcp-card p {
  min-height: 40px;
  margin: 0 0 10px;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.55;
}

.mcp-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 12px;
}

.btn-primary {
  color: #fff;
  background: var(--primary);
  border-color: transparent;
}

.btn-success {
  color: #fff;
  background: var(--success);
  border-color: transparent;
}

.btn-warning {
  color: #fff;
  background: var(--warning);
  border-color: transparent;
}

.btn-danger {
  color: #fff;
  background: var(--danger);
  border-color: transparent;
}

.empty-panel {
  padding: 32px 16px;
  border: 1px dashed var(--border);
  border-radius: 9px;
  color: var(--text-muted);
  text-align: center;
}

.spin {
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

:global(.theme-light) .agent-view {
  background: #f8fafc;
}

:global(.theme-light) .agent-header,
:global(.theme-light) .runtime-panel,
:global(.theme-light) .panel-section,
:global(.theme-light) .capability-card {
  background: #fff;
  border-color: rgba(15,23,42,0.08);
  box-shadow: 0 8px 24px rgba(15,23,42,0.04);
}

:global(.theme-light) .tool-card,
:global(.theme-light) .mcp-card {
  background: #f8fafc;
  border-color: rgba(15,23,42,0.08);
}

:global(.theme-light) .form-input {
  background: #fff;
  border-color: rgba(15,23,42,0.12);
}

@media (max-width: 880px) {
  .agent-layout {
    grid-template-columns: 1fr;
    overflow-y: auto;
  }

  .agent-side,
  .agent-main {
    overflow: visible;
  }
}
</style>
