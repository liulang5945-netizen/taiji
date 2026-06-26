# 态极 AutoDL 紧急开训说明

这个目录是给当前 AutoDL 远端环境准备的补丁包。

适用场景:

- 远端 `/root/autodl-tmp/taiji` 不是完整仓库
- `git clone` 失败
- JupyterLab 里只有一部分 tokenizer 文件，缺少训练脚本

## 你要上传的文件

- `apply_taiji_autodl_patch.py`
- `taiji_autodl_patch_bundle.zip`

二选一即可；已经有 zip 时，优先上传 zip。

## 远端最短执行步骤

```bash
cd /root/autodl-tmp
mkdir -p taiji_patch
cd taiji_patch
unzip -o taiji_autodl_patch_bundle.zip
python3 apply_taiji_autodl_patch.py /root/autodl-tmp/taiji
cd /root/autodl-tmp/taiji
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements-autodl.txt
export TAIJI_DATA_PRESET=stage0_smoke
bash scripts/training/autodl_native_v2.sh data
export TAIJI_DATA_PRESET=english_boost_mirror
bash scripts/training/autodl_native_v2.sh data
bash scripts/training/autodl_native_v2.sh audit
bash scripts/training/autodl_native_v2.sh tokenizer
bash scripts/training/autodl_native_v2.sh smoke
```

## smoke 正常后启动正式训练

```bash
cd /root/autodl-tmp/taiji
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/training/autodl_native_v2.sh train
```

## 说明

- `stage0_smoke` 先拉最小可用集，确认链路正常。
- `english_boost_mirror` 再补英文主料，避免 1B 基座英文过弱。
- 当前脚本默认会重建 tokenizer，以匹配新的预训练语料。
- 如果你明确要保留现有 tokenizer，可先执行:

```bash
export TAIJI_SKIP_TOKENIZER_REBUILD=1
```

## 当前补丁包含

- mirror-aware 预训练数据下载脚本
- 1B 数据审计脚本
- native-v2 tokenizer 构建与验证脚本
- 1B 单卡预训练脚本
- AutoDL 4090D 训练启动脚本
- `requirements-autodl.txt`

