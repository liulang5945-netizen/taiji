<template>
  <section class="dedicated-view">
    <div class="view-header"><h2><BookOpen :size="24" /> {{ t('kb_management') }}</h2></div>
    <div class="view-body">
      <!-- 上传队列 -->
      <n-card title="" size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Upload :size="18" /> {{ t('upload_queue') }}</div></template>
        <FileUploadQueue ref="kbUploadRef" upload-endpoint="/api/rag/upload"
          accept=".txt,.pdf,.docx,.doc,.pptx,.xlsx,.xls,.md,.csv,.json,.jsonl,.html,.htm,.epub,.rtf,.xml,.log,.ini,.cfg,.yaml,.yml,.py,.js,.ts,.css,.java,.c,.cpp,.h,.hpp,.sh,.bat,.ps1,.sql,.r,.go,.rs,.swift,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
          icon="FileText" title="知识库上传队列" upload-icon="Library" :drop-text="t('drop_upload')" :accept-hint="t('kb_support_formats')"
          success-text="✅ 上传成功，正在向量化建库" @all-uploaded="loadKBStats" />
      </n-card>

      <!-- 统计 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><BarChart3 :size="18" /> {{ t('kb_stats') }}</div></template>
        <div v-if="kbStats" class="kb-stats-row">
          <n-statistic label="" :value="kbStats.doc_count || 0">
            <template #prefix><span class="kb-stat-label">{{ t('kb_docs') }}</span></template>
          </n-statistic>
          <n-statistic label="" :value="kbStats.chunk_count || 0">
            <template #prefix><span class="kb-stat-label">{{ t('kb_chunks') }}</span></template>
          </n-statistic>
        </div>
        <n-empty v-else :description="t('kb_empty')" />
      </n-card>

      <!-- 搜索测试 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Search :size="18" /> {{ t('kb_search_test') }}</div></template>
        <n-space>
          <n-input v-model:value="kbSearchQuery" :placeholder="t('kb_search_placeholder')" style="flex:1;" @keydown.enter="searchKB" />
          <n-button type="primary" :loading="kbSearching" @click="searchKB">{{ t('search') }}</n-button>
        </n-space>
        <div v-if="kbResults.length" class="kb-results" style="margin-top:12px;">
          <n-card v-for="(r, i) in kbResults" :key="i" size="small" :bordered="true" style="margin-bottom:8px;">
            <p class="kb-result-text">{{ r.content || r.text || r }}</p>
            <template #footer v-if="r.score != null">
              <n-tag size="small" type="info">Score: {{ Number(r.score).toFixed(4) }}</n-tag>
            </template>
          </n-card>
        </div>
        <n-empty v-else-if="kbSearched" :description="t('kb_no_results')" style="margin-top:12px;" />
      </n-card>

      <!-- 检索策略配置 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><SettingsIcon :size="18" /> 检索策略配置</div></template>
        <n-space vertical>
          <n-checkbox v-model:checked="ragConfig.enable_hybrid" @update:checked="saveRagConfig">
            <n-text strong>混合检索 (Dense + BM25)</n-text>
            <br /><n-text depth="3" style="font-size:12px;">融合语义和关键词两种检索，提升召回率</n-text>
          </n-checkbox>
          <n-checkbox v-model:checked="ragConfig.enable_reranker" @update:checked="saveRagConfig">
            <n-text strong>Cross-Encoder 重排序</n-text>
            <br /><n-text depth="3" style="font-size:12px;">对候选结果精细打分，提升排序精度</n-text>
          </n-checkbox>
          <n-checkbox v-model:checked="ragConfig.enable_query_rewrite" @update:checked="saveRagConfig">
            <n-text strong>查询改写</n-text>
            <br /><n-text depth="3" style="font-size:12px;">用 LLM 改写用户问题以提升检索效果</n-text>
          </n-checkbox>
          <n-space align="center">
            <n-text>候选数量</n-text>
            <n-input-number v-model:value="ragConfig.candidate_k" :min="5" :max="50" size="small" style="width:100px;" @update:value="saveRagConfig" />
          </n-space>
        </n-space>
        <n-space v-if="ragStatus" style="margin-top:12px;" wrap>
          <n-tag size="small">文档: {{ ragStatus.doc_count }}</n-tag>
          <n-tag size="small">段落: {{ ragStatus.chunk_count }}</n-tag>
          <n-tag size="small" :type="ragStatus.has_embeddings ? 'success' : 'error'">Dense: {{ ragStatus.has_embeddings ? '✓' : '✗' }}</n-tag>
          <n-tag size="small" :type="ragStatus.has_bm25 ? 'success' : 'error'">BM25: {{ ragStatus.has_bm25 ? '✓' : '✗' }}</n-tag>
          <n-tag size="small">维度: {{ ragStatus.embed_dim }}</n-tag>
        </n-space>
      </n-card>

      <!-- 已挂载文件 -->
      <n-card size="small" :bordered="false">
        <template #header>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:8px;"><FolderOpen :size="18" /> {{ t('kb_mounted') }}</div>
            <n-button type="error" size="small" @click="clearKB">{{ t('clear_all') }}</n-button>
          </div>
        </template>
        <n-list v-if="kbFiles.length" bordered hoverable>
          <n-list-item v-for="f in kbFiles" :key="f">
            <template #prefix><FileText :size="16" /></template>
            {{ f }}
            <template #suffix>
              <n-button type="error" size="tiny" quaternary @click="deleteKBFile(f)">
                <template #icon><Trash2 :size="14" /></template>
              </n-button>
            </template>
          </n-list-item>
        </n-list>
        <n-empty v-else :description="t('kb_empty')" />
      </n-card>
    </div>
  </section>
</template>
<script setup>
import { ref } from 'vue';
import { BookOpen, Upload, BarChart3, Search, FolderOpen, Settings as SettingsIcon, Trash2, FileText } from 'lucide-vue-next';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi } from '../composables/useApi.js';
import { API_BASE, authFetch } from '../composables/apiClient.js';
const { t } = useApi();
const kbUploadRef = ref(null);
const kbStats = ref(null);
const kbSearchQuery = ref('');
const kbResults = ref([]);
const kbSearched = ref(false);
const kbSearching = ref(false);
const kbFiles = ref([]);
const ragConfig = ref({ enable_hybrid: true, enable_reranker: true, enable_query_rewrite: false, candidate_k: 20 });
const ragStatus = ref(null);
const loadRagConfig = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/config`); if (r.ok) { const d = await r.json(); if (d.config) ragConfig.value = { ...ragConfig.value, ...d.config }; } } catch (e) {} };
const loadRagStatus = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/status`); if (r.ok) { const d = await r.json(); if (d.status === 'ok') ragStatus.value = d; } } catch (e) {} };
const saveRagConfig = async () => { try { await authFetch(`${API_BASE}/api/rag/config`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragConfig.value) }); } catch (e) {} };
loadRagConfig(); loadRagStatus();
const loadKBStats = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/stats`); if (r.ok) kbStats.value = await r.json(); } catch (e) {} };
const loadKBFiles = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/files`); if (r.ok) { const d = await r.json(); kbFiles.value = d.files || []; } } catch (e) {} };
const searchKB = async () => { if (!kbSearchQuery.value.trim()) return; kbSearched.value = true; kbSearching.value = true; try { const r = await authFetch(`${API_BASE}/api/rag/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: kbSearchQuery.value, top_k: 5 }) }); const d = await r.json(); kbResults.value = d.results || []; } catch (e) { kbResults.value = []; } finally { kbSearching.value = false; } };
const clearKB = async () => { try { await authFetch(`${API_BASE}/api/rag/clear`, { method: 'DELETE' }); kbStats.value = null; kbFiles.value = []; } catch (e) {} };
const deleteKBFile = async (filename) => { try { await authFetch(`${API_BASE}/api/rag/file/${encodeURIComponent(filename)}`, { method: 'DELETE' }); loadKBFiles(); loadKBStats(); loadRagStatus(); } catch (e) {} };
loadKBStats(); loadKBFiles();
</script>
<style scoped>
.kb-stats-row {
  display: flex;
  gap: 32px;
}
.kb-stat-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-right: 8px;
}
.kb-result-text {
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  margin: 0;
}
</style>
