#!/usr/bin/env python3
"""Generate Taiji special-format text for tokenizer corpus balancing."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


TOOLS = [
    "search",
    "read_file",
    "write_file",
    "run_command",
    "inspect_image",
    "query_memory",
    "save_memory",
    "plan_task",
]

GOALS = [
    "检查训练数据质量",
    "修复脚本中的参数错误",
    "确认 tokenizer contract 没有被修改",
    "分析用户的长期目标",
    "整理下一步训练计划",
    "验证远端日志是否正常",
    "生成一份简洁报告",
    "比较中英文数据比例",
]

CAUSES = [
    "输入路径不存在",
    "数据类别配比不均衡",
    "工具返回为空",
    "网络镜像响应超时",
    "checkpoint 缺少必要文件",
    "用户目标发生变化",
    "样本重复率过高",
    "命令缺少必要参数",
]

CORRECTIONS = [
    "先检查文件树再执行命令",
    "按 zh/en/code/math/tech/taiji_special 重新采样",
    "降低 shard 数量并记录缺口",
    "切换到镜像友好的数据源",
    "重新验证 model.pt、config.json 和 tokenizer 文件",
    "复述当前目标并更新计划",
    "先去重再写入训练语料",
    "补齐参数后再重试",
]

OBSERVATIONS = [
    "终端正在输出 SentencePiece BPE 训练日志。",
    "报告显示代码和数学配额已经填满。",
    "词表验证通过，total_vocab 仍然是 256000。",
    "当前数据盘空间不足，需要清理 raw shard。",
    "用户希望优先保证 1B 的质量。",
    "英文数据已经落盘，但技术文档仍然偏少。",
    "多模态 token 已预留，但 alignment 尚未开始。",
    "当前任务不应该打断正在运行的训练进程。",
]

MEMORY_TYPES = ["short_term", "long_term", "episodic", "semantic"]
MODALITIES = ["image", "audio", "video", "screen"]


def build_records(limit: int) -> list[str]:
    records: list[str] = []

    for tool, goal in product(TOOLS, GOALS):
        payload = {"name": tool, "args": {"goal": goal, "dry_run": True}}
        records.append(f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>")
        records.append(f"<tool_result>工具 {tool} 已完成：{goal}。</tool_result>")

    for goal in GOALS:
        records.append(
            f"<plan><goal>{goal}</goal><step>理解目标</step><step>执行检查</step>"
            f"<step>验证输出</step><plan_done></plan_done></plan>"
        )
        records.append(f"<think>我需要围绕“{goal}”选择最少但可靠的操作。</think>")
        records.append(f"<final_answer>{goal} 已处理，并保留了可复查的结果。</final_answer>")

    for cause, correction in product(CAUSES, CORRECTIONS):
        records.append(f"<reflect><cause>{cause}</cause><correct>{correction}</correct></reflect>")

    for memory_type, goal in product(MEMORY_TYPES, GOALS):
        records.append(f"<mem_read><{memory_type}>读取与“{goal}”相关的记忆。</{memory_type}></mem_read>")
        records.append(f"<mem_write><{memory_type}>记录：{goal} 的执行结果已经验证。</{memory_type}></mem_write>")

    for modality, observation in product(MODALITIES, OBSERVATIONS):
        records.append(f"<{modality}>{observation}</{modality}>")
        records.append(f"<observe><{modality}>{observation}</{modality}></observe>")

    for observation in OBSERVATIONS:
        records.append(f"<observe>{observation}</observe>")
        records.append(f"<inner_voice>保持耐心，先验证事实：{observation}</inner_voice>")
        records.append(f"<context>{observation}</context>")

    deduped = list(dict.fromkeys(records))
    if len(deduped) >= limit:
        return deduped[:limit]

    expanded: list[str] = []
    round_idx = 0
    while len(expanded) < limit:
        for record in deduped:
            if len(expanded) >= limit:
                break
            expanded.append(f"{record} <state>variant={round_idx}</state>")
        round_idx += 1
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Taiji special-format tokenizer text")
    parser.add_argument(
        "--output",
        default="taiji_data/training_data/pretrain_12b/tokenizer_sample/taiji_special_vocab.jsonl",
    )
    parser.add_argument("--records", type=int, default=20_000)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(args.records)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for text in records:
            handle.write(json.dumps({"text": text, "source": "taiji_special_vocab"}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
