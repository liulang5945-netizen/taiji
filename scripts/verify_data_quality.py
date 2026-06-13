"""
验证数据质量
============
检查新预训练数据的质量
"""
import json

FILE = "taiji_data/training_data/pretrain_all_v2.jsonl"

def main():
    print("=" * 60)
    print("验证数据质量")
    print("=" * 60)

    total_lines = 0
    valid_lines = 0
    format_errors = 0
    empty_content = 0
    echo_patterns = 0

    # 统计字段
    has_system = 0
    has_user = 0
    has_assistant = 0
    multi_turn = 0

    with open(FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines += 1

            if not line.strip():
                continue

            try:
                item = json.loads(line)
                valid_lines += 1
            except json.JSONDecodeError:
                format_errors += 1
                continue

            messages = item.get("messages", [])
            if not messages:
                format_errors += 1
                continue

            # 检查消息结构
            roles = [m.get("role") for m in messages]
            if "system" in roles:
                has_system += 1
            if "user" in roles:
                has_user += 1
            if "assistant" in roles:
                has_assistant += 1
            if roles.count("user") > 1:
                multi_turn += 1

            # 检查内容质量
            for msg in messages:
                content = msg.get("content", "")
                if not content:
                    empty_content += 1
                    break

                # 检查是否是复制模式
                if msg.get("role") == "assistant":
                    # 检查是否和用户输入完全一样
                    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
                    if content == user_msg:
                        echo_patterns += 1
                        break

            if (i + 1) % 50000 == 0:
                print(f"  已检查 {i + 1:,} 条...")

    print()
    print("=" * 60)
    print("质量报告")
    print("=" * 60)
    print(f"  总行数: {total_lines:,}")
    print(f"  有效行: {valid_lines:,}")
    print(f"  格式错误: {format_errors:,}")
    print()
    print("内容统计:")
    print(f"  有系统提示: {has_system:,} ({has_system/valid_lines*100:.1f}%)")
    print(f"  有用户输入: {has_user:,} ({has_user/valid_lines*100:.1f}%)")
    print(f"  有助手回复: {has_assistant:,} ({has_assistant/valid_lines*100:.1f}%)")
    print(f"  多轮对话: {multi_turn:,} ({multi_turn/valid_lines*100:.1f}%)")
    print()
    print("质量问题:")
    print(f"  空内容: {empty_content:,}")
    print(f"  复制模式: {echo_patterns:,}")
    print()

    if echo_patterns < 100 and empty_content < 100:
        print("✅ 数据质量良好！")
    else:
        print("⚠️ 存在质量问题，建议进一步清洗")

    # 显示几个样本
    print()
    print("=" * 60)
    print("数据样本 (前3条)")
    print("=" * 60)

    with open(FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break

            item = json.loads(line)
            messages = item.get("messages", [])

            print(f"\n样本 {i + 1}:")
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # 截断显示
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{role}]: {content}")

if __name__ == "__main__":
    main()
