<template>
  <section class="dedicated-view">
    <div class="view-header">
      <h2><Zap :size="24" /> 🧬 态极 · 生命中心</h2>
      <n-tag v-if="taijiModelInfo.size" type="info" size="small">
        {{ taijiModelInfo.size }} · {{ taijiModelInfo.config?.num_hidden_layers }}层 · {{ taijiModelInfo.config?.hidden_size }}维
      </n-tag>
    </div>
    <div class="view-body">
      <!-- 数据上传 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Upload :size="18" /> {{ t('train_upload') }}</div></template>
        <FileUploadQueue
          ref="trainUploadRef"
          upload-endpoint="/api/train/upload_dataset"
          accept=".jsonl,.json,.txt,.csv,.md,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.html,.htm,.epub,.rtf,.xml,.log,.py,.js,.ts,.css,.java,.c,.cpp,.sh,.sql,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
          icon="BarChart2" title="训练数据上传" upload-icon="Download"
          :drop-text="t('train_upload')" :accept-hint="t('train_support')"
          success-text="✅ 数据集上传成功"
          @all-uploaded="loadTrainDatasets"
        />
      </n-card>

      <!-- 数据集列表 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> {{ t('train_datasets') }}</div>
            <n-space>
              <n-button size="small" @click="loadTrainDatasets"><template #icon><RefreshCw :size="14" /></template>{{ t('reload') }}</n-button>
              <n-button v-if="selectedDatasets.length > 0" size="small" type="error" @click="deleteSelectedDatasets(toast)">
                <template #icon><Trash2 :size="14" /></template>删除选中 ({{ selectedDatasets.length }})
              </n-button>
            </n-space>
          </div>
        </template>
        <div v-if="trainFiles.length">
          <n-space style="margin-bottom:8px;">
            <n-checkbox :checked="isAllSelected()" @update:checked="toggleSelectAll">{{ t('select_all') }}</n-checkbox>
            <n-text depth="3" style="font-size:12px;">已选 {{ selectedDatasets.length }}/{{ trainFiles.length }}</n-text>
          </n-space>
          <n-list bordered hoverable>
            <n-list-item v-for="f in trainFiles" :key="f">
              <n-checkbox :checked="selectedDatasets.includes(f)" @update:checked="toggleDataset(f)" style="margin-right:8px;" />
              <n-ellipsis style="max-width:400px;">{{ f }}</n-ellipsis>
              <template #suffix>
                <n-space size="small">
                  <n-button size="tiny" @click="previewDataset(f)">{{ t('preview') }}</n-button>
                  <n-button size="tiny" type="error" quaternary @click="deleteTrainFile(f)"><template #icon><Trash2 :size="14" /></template></n-button>
                </n-space>
              </template>
            </n-list-item>
          </n-list>
        </div>
        <n-empty v-else :description="t('train_no_data')" />
        <!-- 预览 -->
        <n-card v-if="trainPreview" size="small" :bordered="true" style="margin-top:12px;">
          <template #header>{{ t('dataset_preview') }} ({{ trainPreview.count || 0 }} {{ t('samples') }})</template>
          <div v-for="(s, i) in (trainPreview.samples || [])" :key="i" style="margin-bottom:12px;">
            <n-text strong>{{ t('instruction') }}:</n-text>
            <n-text>{{ s.instruction }}</n-text>
            <br /><n-text strong>{{ t('output') }}:</n-text>
            <n-text>{{ s.output }}</n-text>
          </div>
        </n-card>
      </n-card>

      <!-- 训练参数 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Brain :size="18" style="color:var(--primary);" /> 态极训练参数</div></template>
        <n-grid :cols="3" :x-gap="12" :y-gap="12">
          <n-gi><n-form-item label="Epochs"><n-input-number v-model:value="taijiTrainParams.num_epochs" :min="1" :max="100" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Batch Size"><n-input-number v-model:value="taijiTrainParams.batch_size" :min="1" :max="32" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Learning Rate"><n-input-number v-model:value="taijiTrainParams.learning_rate" :min="0.00001" :step="0.00001" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Max Length"><n-input-number v-model:value="taijiTrainParams.max_length" :min="64" :max="2048" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Save Steps"><n-input-number v-model:value="taijiTrainParams.save_steps" :min="10" :max="1000" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Keep Checkpoints"><n-input-number v-model:value="taijiTrainParams.keep_checkpoints" :min="1" :max="10" /></n-form-item></n-gi>
        </n-grid>
      </n-card>

      <!-- 控制 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Gamepad2 :size="18" /> 控制</div></template>
        <n-space>
          <n-button v-if="trainState === 'idle' || trainState === 'completed'" type="primary" @click="startTaijiTraining(toast)"><template #icon><Brain :size="14" /></template>🧬 开始态极微调</n-button>
          <n-button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" type="info" @click="resumeFromCheckpoint(toast, $confirm)"><template #icon><RefreshCw :size="14" /></template>恢复训练 ({{ pendingCheckpoints.length }})</n-button>
          <n-button v-if="trainState === 'running'" type="warning" @click="pauseTraining(toast)"><template #icon><Pause :size="14" /></template>{{ t('pause_training') }}</n-button>
          <n-button v-if="trainState === 'paused'" type="primary" @click="resumeTraining(toast)"><template #icon><Play :size="14" /></template>{{ t('resume_training') }}</n-button>
          <n-button v-if="trainState === 'running' || trainState === 'paused'" type="error" @click="stopTraining(toast)"><template #icon><StopCircle :size="14" /></template>{{ t('stop_training') }}</n-button>
          <n-button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" type="info" @click="forcePublish(toast, $confirm)"><template #icon><PackageIcon :size="14" /></template>强制发布</n-button>
        </n-space>
      </n-card>

      <!-- 训练进度 -->
      <div v-if="trainState === 'running' || trainState === 'paused'">
        <n-alert v-if="trainDevice.message" type="info" style="margin-bottom:12px;">
          <template #icon><Monitor :size="16" /></template>
          {{ trainDevice.message }}
        </n-alert>
        <n-card size="small" :bordered="false" style="margin-bottom:16px;">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><TrendingUp :size="18" /> 训练进度</div><n-tag :type="trainState === 'paused' ? 'warning' : 'info'">{{ trainProgress }}%</n-tag></div></template>
          <n-progress type="line" :percentage="trainProgress" :status="trainState === 'paused' ? 'warning' : 'info'" :processing="trainState === 'running'" />
          <n-text depth="3" style="margin-top:8px;display:block;">{{ trainProgressDesc }}</n-text>
        </n-card>
        <!-- 指标 -->
        <n-grid :cols="3" :x-gap="12" :y-gap="12" style="margin-bottom:16px;">
          <n-gi><n-card size="small"><n-statistic label="已用时间"><template #prefix><Clock :size="16" /></template>{{ fmtTime(trainMetrics.elapsed) }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="预计剩余"><template #prefix><Hourglass :size="16" /></template>{{ fmtTime(trainMetrics.eta) }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="当前 Loss"><template #prefix><Activity :size="16" /></template>{{ trainMetrics.current_loss != null ? trainMetrics.current_loss.toFixed(4) : '--' }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="吞吐量"><template #prefix><Zap :size="16" /></template>{{ trainMetrics.samples_per_sec >= 0.005 ? (trainMetrics.samples_per_sec < 0.1 ? trainMetrics.samples_per_sec.toFixed(2) : trainMetrics.samples_per_sec.toFixed(1)) + '/s' : '--' }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="Epoch"><template #prefix><BookOpen :size="16" /></template>{{ trainMetrics.epoch }}/{{ trainMetrics.total_epochs }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="学习率"><template #prefix><GraduationCap :size="16" /></template>{{ trainMetrics.lr != null ? trainMetrics.lr.toExponential(2) : '--' }}</n-statistic></n-card></n-gi>
        </n-grid>
        <!-- Loss 曲线 -->
        <n-card v-if="trainLoss.length >= 2" size="small" :bordered="false" style="margin-bottom:16px;">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><TrendingUp :size="18" /> Loss 曲线</div><n-tag size="small">最新: {{ trainLoss[trainLoss.length-1]?.loss?.toFixed(4) || '--' }}</n-tag></div></template>
          <canvas ref="lossCanvasRef" class="loss-canvas" width="600" height="160"></canvas>
        </n-card>
        <!-- 日志 -->
        <n-card v-if="trainLog" size="small" :bordered="false">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <div style="display:flex;align-items:center;gap:8px;"><FileTextIcon :size="18" /> 训练日志</div>
              <n-button size="tiny" quaternary @click="clearTrainLog"><template #icon><Trash2 :size="14" /></template>清空</n-button>
            </div>
          </template>
          <pre class="train-log-content" ref="trainLogRef">{{ trainLog }}</pre>
        </n-card>
      </div>

      <!-- 发布进度 -->
      <n-card v-if="publishingState === 'publishing'" size="small" :bordered="false" style="margin-top:12px;">
        <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> 发布进度</div><n-tag>{{ trainProgress }}%</n-tag></div></template>
        <n-progress type="line" :percentage="trainProgress" :processing="true" />
        <n-text depth="3" style="margin-top:8px;display:block;">{{ trainProgressDesc }}</n-text>
        <n-button type="error" style="margin-top:8px;" @click="cancelPublish()"><template #icon><Square :size="14" /></template>取消发布</n-button>
      </n-card>

      <!-- 发布与导出 -->
      <n-card v-if="trainState === 'completed' || publishingState !== 'idle'" size="small" :bordered="false" style="margin-top:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> 发布与导出</div></template>
        <n-text depth="3" style="display:block;margin-bottom:12px;">{{ t('publish_desc') }}</n-text>
        <n-space>
          <n-button type="primary" :disabled="publishingState !== 'idle'" @click="publishModel(toast)">
            {{ publishingState === 'publishing' ? '发布中...' : t('publish_model_btn') }}
          </n-button>
          <n-button type="info" :disabled="publishingState !== 'idle'" @click="exportModelToGGUF(toast, $confirm)">
            {{ publishingState === 'publishing' ? '导出中...' : t('export_gguf_btn') }}
          </n-button>
        </n-space>
      </n-card>
    </div>
  </section>
</template>

<script setup>
import { Monitor, Clock, Hourglass, Activity, Zap, BookOpen, GraduationCap, Trash2, Upload, Package as PackageIcon, RefreshCw, Gamepad2, Play, Pause, Square, StopCircle, TrendingUp, FileText as FileTextIcon, Brain } from 'lucide-vue-next';

import { inject, watch, nextTick } from 'vue';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi } from '../composables/useApi.js';
import {
  trainState, trainLog, trainLoss, trainFiles,
  selectedDatasets, trainPreview, trainParams,
  publishingState, trainProgress, trainProgressDesc,
  pendingCheckpoints, trainMetrics, trainDevice,
  lossCanvasRef, trainLogRef, fmtTime,
  clearTrainLog,
  loadTrainDatasets, previewDataset, deleteTrainFile, deleteSelectedDatasets,
  toggleSelectAll, toggleDataset, isAllSelected,
  pauseTraining, resumeTraining, stopTraining,
  loadCheckpoints, resumeFromCheckpoint,
  publishModel, forcePublish, exportModelToGGUF,
  cancelPublish, drawLossChart,
  taijiModelInfo, taijiTrainParams,
  startTaijiTraining,
} from '../composables/useTraining.js';

const toast = inject('toast');
const $confirm = inject('$confirm');
const { t } = useApi();

watch(() => trainLoss.value.length, () => {
  nextTick(() => drawLossChart());
});

loadTrainDatasets();
loadCheckpoints();
</script>

<style scoped>
.train-log-content {
  background: var(--bg-card);
  color: var(--text);
  padding: 12px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.6;
  max-height: 300px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.loss-canvas {
  width: 100%;
  height: 160px;
  border-radius: 8px;
}
</style>
