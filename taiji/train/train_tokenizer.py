"""
训练 SentencePiece 分词器

用全量训练数据训练一个 32000 词表的分词器，覆盖中英文。
词表大小 32000，与 config.py 中的 SPECIAL_TOKENS 兼容。
"""
import os
import sys
import json
import logging
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("TrainTokenizer")


def create_training_corpus(output_path: str, max_lines: int = 500000):
    """
    从全量训练数据中创建分词器训练语料。

    优先使用 pretrain_all.jsonl（最大、最多样的数据集），
    补充毕业对话数据和种子数据。
    """
    training_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji_data", "training_data"
    )

    lines = []

    # 1. 从 pretrain_all.jsonl 提取（最大的数据源，2GB，700K行）
    pretrain_file = os.path.join(training_dir, "pretrain_all.jsonl")
    if os.path.exists(pretrain_file):
        logger.info(f"Reading {pretrain_file}...")
        with open(pretrain_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    d = json.loads(line)
                    if "text" in d:
                        lines.append(d["text"])
                    elif "messages" in d:
                        for msg in d.get("messages", []):
                            content = msg.get("content", "").strip()
                            if content:
                                lines.append(content)
                except json.JSONDecodeError:
                    continue
        logger.info(f"  Extracted {len(lines)} lines from pretrain_all.jsonl")

    # 2. 补充毕业对话数据（中文为主）
    for fname in [
        "taiji_graduation_v2_conversation.jsonl",
        "taiji_graduation_v3_conversation.jsonl",
        "taiji_graduation_conversation.jsonl",
    ]:
        fpath = os.path.join(training_dir, fname)
        if os.path.exists(fpath):
            count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        for msg in d.get("messages", []):
                            content = msg.get("content", "").strip()
                            if content:
                                lines.append(content)
                                count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Added {count} lines from {fname}")

    # 3. 补充 ReAct 数据（工具调用格式）
    for fname in [
        "taiji_graduation_v2_react.jsonl",
        "taiji_graduation_v3_react.jsonl",
    ]:
        fpath = os.path.join(training_dir, fname)
        if os.path.exists(fpath):
            count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "task" in d:
                            lines.append(d["task"])
                            count += 1
                        for step in d.get("steps", []):
                            for key in ("thought", "final_answer"):
                                val = step.get(key, "").strip()
                                if val:
                                    lines.append(val)
                                    count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Added {count} lines from {fname}")

    # 4. 补充种子数据
    try:
        from taiji.data.seed_data import get_seed_react_data, get_seed_conversation_data
        seed_count = 0
        for item in get_seed_react_data():
            lines.append(item["task"])
            seed_count += 1
            for step in item.get("steps", []):
                for key in ("thought", "final_answer"):
                    val = step.get(key, "").strip()
                    if val:
                        lines.append(val)
                        seed_count += 1
        for item in get_seed_conversation_data():
            for msg in item.get("messages", []):
                content = msg.get("content", "").strip()
                if content:
                    lines.append(content)
                    seed_count += 1
        logger.info(f"  Added {seed_count} lines from seed_data")
    except Exception as e:
        logger.warning(f"  Could not load seed_data: {e}")

    # 5. 补充通用代码和技术文本
    general_texts = [
        "你好，我是态极AI助手。",
        "我可以帮你完成文件操作、代码执行、信息搜索等任务。",
        "Python 是一种高级编程语言，广泛应用于数据科学和人工智能。",
        "机器学习是人工智能的一个分支，让计算机从数据中学习。",
        "深度学习使用神经网络进行学习，是机器学习的子领域。",
        "快速排序是一种高效的排序算法，平均时间复杂度为 O(n log n)。",
        "递归是一种编程技巧，指函数调用自身。",
        "装饰器是 Python 的一种设计模式，使用 @ 语法糖。",
        "Hello, I am Taiji AI assistant.",
        "I can help you with file operations, code execution, and information search.",
        "def hello_world():\n    print('Hello')",
        "import os\nimport sys\nfrom typing import List, Dict",
        "class MyClass:\n    def __init__(self):\n        self.value = 0",
        "for i in range(10):\n    print(i)",
        "if __name__ == '__main__':\n    main()",
        "try:\n    result = do_something()\nexcept Exception as e:\n    print(f'Error: {e}')",
        "with open('file.txt', 'r') as f:\n    content = f.read()",
        "json.dumps({'key': 'value'})",
        "torch.tensor([1, 2, 3])",
        "SELECT * FROM users WHERE id = 1",
        "CREATE INDEX idx_name ON users(name)",
        "docker build -t myapp .",
        "kubectl get pods -n production",
        "git commit -m 'feat: add new feature'",
        "npm install react react-dom",
        "pip install torch transformers",
    ]
    lines.extend(general_texts)
    logger.info(f"  Added {len(general_texts)} general text lines")

    # 去重
    unique_lines = list(set(lines))
    logger.info(f"Total unique lines: {len(unique_lines)} (from {len(lines)} raw)")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in unique_lines:
            line = line.strip()
            if line:
                f.write(line + "\n")

    logger.info(f"Corpus saved: {output_path} ({len(unique_lines)} lines)")
    return len(unique_lines)


def train_sentencepiece(input_path: str, output_dir: str, vocab_size: int = 32000):
    """训练 SentencePiece 模型"""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "sentencepiece")

    logger.info(f"Training SentencePiece model (vocab_size={vocab_size})...")
    logger.info(f"Input corpus: {input_path}")

    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    model_path = model_prefix + ".model"
    logger.info(f"SentencePiece model saved to: {model_path}")

    # 验证
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    logger.info(f"Vocab size: {sp.GetPieceSize()}")

    # 测试中文
    test_cn = "你好世界，我是态极AI助手。"
    ids_cn = sp.EncodeAsIds(test_cn)
    pieces_cn = [sp.IdToPiece(i) for i in ids_cn]
    logger.info(f"CN test: {test_cn}")
    logger.info(f"  IDs: {ids_cn}")
    logger.info(f"  Pieces: {pieces_cn}")

    # 测试英文
    test_en = "Hello, I am Taiji AI assistant."
    ids_en = sp.EncodeAsIds(test_en)
    logger.info(f"EN test: {test_en}")
    logger.info(f"  IDs: {ids_en}")

    return model_path


if __name__ == "__main__":
    print("=" * 60)
    print("Training Taiji SentencePiece Tokenizer")
    print("=" * 60)
    print()

    # 1. 创建训练语料
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji_data", "tokenizer_corpus.txt"
    )
    num_lines = create_training_corpus(corpus_path)

    # 2. 训练 SentencePiece (32000 词表)
    tokenizer_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "taiji", "tokenizer"
    )
    model_path = train_sentencepiece(corpus_path, tokenizer_dir, vocab_size=32000)

    # 3. 测试 ModelSelfTokenizer
    print("\n--- Testing ModelSelfTokenizer ---\n")
    try:
        from taiji.tokenizer import ModelSelfTokenizer
        tokenizer = ModelSelfTokenizer(sp_model_path=model_path)

        test_texts = [
            "你好，我是态极AI助手。",
            "def hello_world():\n    print('Hello')",
            "请搜索 Python 教程",
            "I can help you with file operations.",
            "态极的终极目标是演化到AI的最终形态。",
        ]

        for text in test_texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            print(f"  原文: {text[:50]}")
            print(f"  IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
            print(f"  解码: {decoded[:50]}")
            print()
    except Exception as e:
        print(f"  ModelSelfTokenizer test failed: {e}")

    print("=" * 60)
    print("Tokenizer training completed!")
    print(f"Model: {model_path}")
    print("=" * 60)
