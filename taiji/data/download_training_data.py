"""
态极大规模训练数据下载器
========================
从 HuggingFace 下载开源数据集，转换为态极训练格式。

目标：下载 ~50 亿 token 的中英文数据
来源：
  - Cerebras/SlimPajama-627B（英文网页文本，清洗过的）
  - togethercomputer/RedPajama-Data-1T（英文多来源）
  - json 文件直接下载

用法：
  python taiji/data/download_training_data.py
"""
import os
import sys
import json
import hashlib
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("DataDownloader")

# 下载目标目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "taiji", "training_data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def download_slimpajama(max_samples=500_000):
    """
    下载 SlimPajama 数据集（清洗过的英文网页文本）
    每条约 500-2000 token，50万条约 5-10 亿 token
    """
    output_file = os.path.join(DATA_DIR, "slimpajama.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"SlimPajama 已存在: {count} 条 ({output_file})")
        return output_file

    logger.info(f"开始下载 SlimPajama（最多 {max_samples:,} 条）...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        start_time = time.time()
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                if count >= max_samples:
                    break
                text = item.get("text", "").strip()
                if len(text) < 100:  # 跳过太短的
                    continue
                # 转换为态极对话格式
                entry = {
                    "messages": [
                        {"role": "system", "content": "你是态极AI助手，本地运行，数据不出本机。"},
                        {"role": "user", "content": f"请阅读并理解以下内容：\n\n{text[:2000]}"},
                        {"role": "assistant", "content": text[:2000]}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    logger.info(f"  已下载 {count:,} 条 ({speed:.0f} 条/秒)")

        logger.info(f"SlimPajama 下载完成: {count:,} 条 -> {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"SlimPajama 下载失败: {e}")
        return None


def download_redpajama_code(max_samples=200_000):
    """
    下载 RedPajama 代码数据（GitHub 代码）
    让态极学会编程
    """
    output_file = os.path.join(DATA_DIR, "redpajama_code.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"RedPajama Code 已存在: {count} 条 ({output_file})")
        return output_file

    logger.info(f"开始下载 RedPajama Code（最多 {max_samples:,} 条）...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T",
            "github",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        start_time = time.time()
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                if count >= max_samples:
                    break
                text = item.get("text", "").strip()
                if len(text) < 200:
                    continue
                # 提取代码并生成问答对
                lines = text.split("\n")
                lang = "unknown"
                for line in lines[:5]:
                    if line.startswith("#!"):
                        if "python" in line: lang = "python"
                        elif "bash" in line or "sh" in line: lang = "shell"
                        elif "node" in line: lang = "javascript"
                        break
                    elif line.startswith("//") or line.startswith("/*"):
                        lang = "javascript"
                    elif line.startswith("def ") or line.startswith("import ") or line.startswith("class "):
                        lang = "python"
                    elif line.startswith("fn ") or line.startswith("use "):
                        lang = "rust"

                code = text[:3000]
                entry = {
                    "messages": [
                        {"role": "system", "content": "你是态极AI助手，擅长编程和技术问题。"},
                        {"role": "user", "content": f"请解释以下 {lang} 代码的功能：\n```{lang}\n{code[:1500]}\n```"},
                        {"role": "assistant", "content": f"这段代码的功能分析：\n\n```{lang}\n{code[:2000]}\n```\n\n以上是代码的主要逻辑。"}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed if elapsed > 0 else 0
                    logger.info(f"  已下载 {count:,} 条 ({speed:.0f} 条/秒)")

        logger.info(f"RedPajama Code 下载完成: {count:,} 条 -> {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"RedPajama Code 下载失败: {e}")
        return None


def download_alpaca_style(max_samples=100_000):
    """
    下载 Alpaca 风格的指令数据（英文指令跟随）
    """
    output_file = os.path.join(DATA_DIR, "alpaca_instructions.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"Alpaca 已存在: {count} 条 ({output_file})")
        return output_file

    logger.info(f"开始下载 Alpaca 指令数据（最多 {max_samples:,} 条）...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "tatsu-lab/alpaca",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                if count >= max_samples:
                    break
                instruction = item.get("instruction", "").strip()
                input_text = item.get("input", "").strip()
                output_text = item.get("output", "").strip()
                if not instruction or not output_text:
                    continue
                if len(output_text) < 20:
                    continue

                user_msg = instruction
                if input_text:
                    user_msg += f"\n\n输入: {input_text}"

                entry = {
                    "messages": [
                        {"role": "system", "content": "你是态极AI助手，本地运行，数据不出本机。"},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": output_text}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    logger.info(f"  已下载 {count:,} 条")

        logger.info(f"Alpaca 下载完成: {count:,} 条 -> {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Alpaca 下载失败: {e}")
        return None


def download_sharegpt(max_samples=100_000):
    """
    下载 ShareGPT 多轮对话数据
    """
    output_file = os.path.join(DATA_DIR, "sharegpt_conversations.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"ShareGPT 已存在: {count} 条 ({output_file})")
        return output_file

    logger.info(f"开始下载 ShareGPT 对话数据（最多 {max_samples:,} 条）...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                if count >= max_samples:
                    break
                conversations = item.get("conversations", [])
                if not conversations or len(conversations) < 2:
                    continue

                messages = [{"role": "system", "content": "你是态极AI助手，本地运行，数据不出本机。"}]
                for turn in conversations:
                    role = turn.get("from", "")
                    value = turn.get("value", "").strip()
                    if not value:
                        continue
                    if role in ("human", "user"):
                        messages.append({"role": "user", "content": value})
                    elif role in ("gpt", "assistant"):
                        messages.append({"role": "assistant", "content": value})

                if len(messages) < 3:
                    continue

                entry = {"messages": messages}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    logger.info(f"  已下载 {count:,} 条")

        logger.info(f"ShareGPT 下载完成: {count:,} 条 -> {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"ShareGPT 下载失败: {e}")
        return None


def merge_all_data():
    """合并所有下载的数据到统一训练文件"""
    output_file = os.path.join(os.path.dirname(DATA_DIR), "pretrain_all.jsonl")
    total = 0
    seen_hashes = set()

    with open(output_file, "w", encoding="utf-8") as out:
        for fname in os.listdir(DATA_DIR):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(DATA_DIR, fname)
            count = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 简单去重
                    h = hashlib.md5(line.encode()).hexdigest()[:16]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    out.write(line + "\n")
                    count += 1
                    total += 1
            logger.info(f"  {fname}: {count:,} 条")
            # 下载完的原始文件可以删除节省空间
            # os.remove(fpath)

    logger.info(f"合并完成: {total:,} 条 -> {output_file}")

    # 统计总 token 数
    token_count = 0
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            token_count += len(line) // 3  # 粗略估算：3字符≈1token
    logger.info(f"估算总 token 数: ~{token_count:,}")

    return output_file


def main():
    logger.info("=" * 60)
    logger.info("态极大规模训练数据下载")
    logger.info("=" * 60)
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"磁盘可用: {__import__('shutil').disk_usage('.').free / 1024/1024/1024:.0f}GB")
    logger.info("")

    # 下载各数据集（按优先级）
    # 总目标：~50 亿 token（约 15-20GB 文本）
    results = []

    # 1. SlimPajama（英文网页，最大量）
    r = download_slimpajama(max_samples=500_000)
    if r: results.append(r)

    # 2. 代码数据（让态极会编程）
    r = download_redpajama_code(max_samples=200_000)
    if r: results.append(r)

    # 3. 指令数据（让态极会听指令）
    r = download_alpaca_style(max_samples=100_000)
    if r: results.append(r)

    # 4. 多轮对话（让态极会聊天）
    r = download_sharegpt(max_samples=100_000)
    if r: results.append(r)

    logger.info("")
    logger.info("下载完成，开始合并...")

    # 合并去重
    merged = merge_all_data()

    logger.info("")
    logger.info("=" * 60)
    logger.info("数据准备完成！")
    logger.info(f"  合并文件: {merged}")
    logger.info(f"  下一步: 用态极训练器训练模型")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
