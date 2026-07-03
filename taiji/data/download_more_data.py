"""
态极训练数据补充下载器
======================
下载公开可用的大规模数据集
"""
import os, sys, json, time, logging, hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("DataDownloader")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "taiji", "training_data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def download_fineweb(max_samples=300_000):
    """HuggingFace FineWeb（英文高质量网页，10B token 样本）"""
    output_file = os.path.join(DATA_DIR, "fineweb.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"FineWeb 已存在: {count} 条")
        return count

    logger.info(f"下载 FineWeb（最多 {max_samples:,} 条）...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)

    count = 0
    start = time.time()
    with open(output_file, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break
            text = item.get("text", "").strip()
            if len(text) < 200:
                continue
            entry = {"messages": [
                {"role": "system", "content": "You are Taiji, a helpful AI assistant."},
                {"role": "user", "content": f"Read and understand this text:\n\n{text[:2000]}"},
                {"role": "assistant", "content": text[:2000]}
            ]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - start
                logger.info(f"  FineWeb: {count:,} 条 ({count/elapsed:.0f}/s)")

    logger.info(f"FineWeb 完成: {count:,} 条")
    return count


def download_openwebtext(max_samples=200_000):
    """OpenWebText（GPT-2 训练数据来源）"""
    output_file = os.path.join(DATA_DIR, "openwebtext.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"OpenWebText 已存在: {count} 条")
        return count

    logger.info(f"下载 OpenWebText（最多 {max_samples:,} 条）...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    count = 0
    start = time.time()
    with open(output_file, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break
            text = item.get("text", "").strip()
            if len(text) < 200:
                continue
            entry = {"messages": [
                {"role": "system", "content": "You are Taiji, a helpful AI assistant."},
                {"role": "user", "content": f"Summarize this text:\n\n{text[:1500]}"},
                {"role": "assistant", "content": text[:1500]}
            ]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - start
                logger.info(f"  OpenWebText: {count:,} 条 ({count/elapsed:.0f}/s)")

    logger.info(f"OpenWebText 完成: {count:,} 条")
    return count


def download_wikipedia(max_samples=100_000):
    """英文维基百科"""
    output_file = os.path.join(DATA_DIR, "wikipedia.jsonl")
    if os.path.exists(output_file):
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        logger.info(f"Wikipedia 已存在: {count} 条")
        return count

    logger.info(f"下载 Wikipedia（最多 {max_samples:,} 条）...")
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break
            title = item.get("title", "")
            text = item.get("text", "").strip()
            if len(text) < 300:
                continue
            entry = {"messages": [
                {"role": "system", "content": "You are Taiji, a knowledgeable AI assistant."},
                {"role": "user", "content": f"Tell me about {title}"},
                {"role": "assistant", "content": text[:2000]}
            ]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                logger.info(f"  Wikipedia: {count:,} 条")

    logger.info(f"Wikipedia 完成: {count:,} 条")
    return count


def merge_all():
    """合并所有数据（含之前下载的）"""
    output_file = os.path.join(os.path.dirname(DATA_DIR), "pretrain_all.jsonl")
    total = 0
    seen = set()

    with open(output_file, "w", encoding="utf-8") as out:
        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(DATA_DIR, fname)
            count = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    h = hashlib.md5(line.encode()).hexdigest()[:16]
                    if h in seen:
                        continue
                    seen.add(h)
                    out.write(line + "\n")
                    count += 1
                    total += 1
            logger.info(f"  {fname}: {count:,} 条")
            os.remove(fpath)  # 删除原始文件节省空间

    # 统计
    token_count = 0
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            token_count += len(line) // 3

    logger.info(f"")
    logger.info(f"合并完成: {total:,} 条, ~{token_count:,} token")
    logger.info(f"文件: {output_file}")
    return total


def main():
    logger.info("=" * 50)
    logger.info("补充下载训练数据")
    logger.info("=" * 50)

    # FineWeb: 大量英文网页（最重要的数据源）
    download_fineweb(max_samples=300_000)

    # OpenWebText: 高质量英文
    download_openwebtext(max_samples=200_000)

    # Wikipedia: 知识性文本
    download_wikipedia(max_samples=100_000)

    logger.info("")
    logger.info("合并所有数据...")
    merge_all()


if __name__ == "__main__":
    main()
