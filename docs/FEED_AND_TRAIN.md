# 态极喂养与训练

## 喂养系统

### 工作原理

```
用户喂养 → FeedEngine 收集数据 → 评估质量 → 生成训练样本
```

### 喂养方式

1. **文本喂养** — 直接输入文字
2. **文件喂养** — 喂养单个文件
3. **目录喂养** — 喂养整个目录

### 喂养流程

```python
from taiji.core.api import get_core

core = get_core()
core.initialize()

# 文本喂养
result = core.feed("这是一段知识", content_type="text")
print(result["message"])

# 文件喂养
result = core.feed("/path/to/file.py", content_type="file")
print(result["message"])

# 目录喂养
result = core.feed("/path/to/directory", content_type="directory")
print(result["message"])
```

### 质量评估

态极会自动评估喂养内容的质量：

- **营养评分** — 0~1 分，越高越好
- **分类** — code, knowledge, conversation, creative
- **去重** — 自动过滤重复内容

## 训练系统

### 工作原理

```
训练样本 → Trainer 训练模型 → 模型进化
```

### 训练方式

1. **手动训练** — 点击训练按钮
2. **自动训练** — 睡眠时自动训练

### 训练流程

```python
from taiji.core.api import get_core

core = get_core()
core.initialize()

# 手动训练
result = core.train(epochs=3, learning_rate=5e-5)
print(result["message"])

# 自动训练（睡眠）
result = core.sleep()
print(result["message"])
```

### 训练参数

- **epochs** — 训练轮数（默认 3）
- **learning_rate** — 学习率（默认 5e-5）

## 生命节律

态极的生命节律：

```
☀️ 醒来 → 🍚 吃饭（收集新知识）→ 🏃 活动（服务用户）
→ 🍚 吃饭（收集交互数据）→ 💤 睡觉（消化训练）→ ☀️ 醒来
```

### 喂养按钮 🍎

- 收集用户交互数据
- 评估数据质量
- 生成训练样本

### 睡眠按钮 😴

- 自动训练模型
- 消化收集的数据
- 提升态极能力

### 训练按钮 🧠

- 手动触发训练
- 指定训练参数
- 查看训练结果

## 使用建议

1. **定期喂养** — 每天喂养一些新知识
2. **充足睡眠** — 让态极有时间消化训练
3. **适度训练** — 不要过度训练，避免过拟合
4. **观察状态** — 关注态极的生命状态

## 注意事项

1. **数据质量** — 喂养高质量的内容
2. **训练时间** — 训练需要一定时间
3. **资源消耗** — 训练会消耗计算资源
4. **模型保存** — 训练后自动保存模型
