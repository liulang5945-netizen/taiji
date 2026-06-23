<template>
  <svg :width="size" :height="size" viewBox="0 0 128 128" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient :id="gradientId" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" :stop-color="colors[0]"/>
        <stop offset="100%" :stop-color="colors[1]"/>
      </linearGradient>
    </defs>
    <!-- 外圆环 -->
    <circle cx="64" cy="64" r="58" :stroke="`url(#${gradientId})`" stroke-width="2.5" fill="none" opacity="0.3"/>
    <!-- 太极主体轮廓 -->
    <path d="M64 6 C97.1 6 124 32.9 124 66 C124 82.5 117.3 97.4 106.3 108.3 C97.4 117.3 82.5 124 66 124 C32.9 124 6 97.1 6 64 C6 47.5 12.7 32.6 23.7 21.7"
      :stroke="`url(#${gradientId})`" stroke-width="2.5" fill="none"/>
    <!-- S 曲线 -->
    <path d="M64 6 C64 35.5 39.5 60 39.5 89.5 C39.5 106.3 53.7 120 64 124"
      :stroke="`url(#${gradientId})`" stroke-width="2.5" fill="none"/>
    <path d="M64 124 C64 94.5 88.5 70 88.5 40.5 C88.5 23.7 74.3 10 64 6"
      :stroke="`url(#${gradientId})`" stroke-width="2.5" fill="none"/>
    <!-- 阳区域 -->
    <path d="M64 6 C97.1 6 124 32.9 124 66 C124 82.5 117.3 97.4 106.3 108.3 C97.4 117.3 82.5 124 66 124 C53.7 120 39.5 106.3 39.5 89.5 C39.5 60 64 35.5 64 6Z"
      :fill="`url(#${gradientId})`" opacity="0.12"/>
    <!-- 阳眼 -->
    <circle cx="64" cy="36" r="7" :fill="`url(#${gradientId})`"/>
    <!-- 阴眼 -->
    <circle cx="64" cy="92" r="7" :fill="bgColor" :stroke="`url(#${gradientId})`" stroke-width="1.5"/>
    <!-- 中心连接线 -->
    <line x1="64" y1="43" x2="64" y2="61" :stroke="`url(#${gradientId})`" stroke-width="0.8" opacity="0.25"/>
    <line x1="64" y1="67" x2="64" y2="85" :stroke="`url(#${gradientId})`" stroke-width="0.8" opacity="0.25"/>
    <circle cx="64" cy="64" r="2.5" :fill="`url(#${gradientId})`" opacity="0.5"/>
  </svg>
</template>

<script setup>
import { computed } from 'vue'
import { useAppStore } from '@/stores/appStore.js'

const props = defineProps({
  size: { type: [Number, String], default: 40 },
})

const appStore = useAppStore()

const gradientId = computed(() => 'taiji-grad-' + Math.random().toString(36).slice(2, 8))

const colors = computed(() =>
  appStore.currentTheme === 'light'
    ? ['#4f46e5', '#7c3aed']
    : ['#7c8aff', '#a78bfa']
)

const bgColor = computed(() =>
  appStore.currentTheme === 'light' ? '#ffffff' : '#1a1b26'
)
</script>
