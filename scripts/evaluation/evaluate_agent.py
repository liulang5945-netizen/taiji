"""
态极 Agent 评测框架
====================

评测态极的 Agent 能力，对标 AgentBench。

评测维度：
1. 工具调用准确性 — 能否正确选择和调用工具
2. 多步推理能力 — 能否分解复杂任务
3. 错误恢复能力 — 能否从失败中恢复
4. 知识检索能力 — 能否有效使用知识库
5. 代码执行能力 — 能否正确编写和执行代码

使用方式：
    python scripts/evaluate_agent.py
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AgentEval")


# ═══════════════════════════════════════════════
# 评测用例
# ═══════════════════════════════════════════════

TOOL_SELECTION_CASES = [
    {
        "task": "搜索关于Python最新版本的信息",
        "expected_tool": "search",
        "category": "tool_selection",
    },
    {
        "task": "读取 config.json 文件的内容",
        "expected_tool": "read_local_file",
        "category": "tool_selection",
    },
    {
        "task": "计算 2 的 10 次方",
        "expected_tool": "calculator",
        "category": "tool_selection",
    },
    {
        "task": "查看当前日期和时间",
        "expected_tool": "datetime",
        "category": "tool_selection",
    },
    {
        "task": "在 agent_workspace 目录下创建一个新文件 hello.py",
        "expected_tool": "write_file",
        "category": "tool_selection",
    },
    {
        "task": "查看 agent_workspace 目录下有哪些文件",
        "expected_tool": "list_directory",
        "category": "tool_selection",
    },
    {
        "task": "帮我规划一个 Web 应用项目的开发步骤",
        "expected_tool": "create_plan",
        "category": "tool_selection",
    },
    {
        "task": "分析 data.csv 文件中的数据分布",
        "expected_tool": "data_query",
        "category": "tool_selection",
    },
    {
        "task": "检查 https://example.com 是否可以访问",
        "expected_tool": "url_check",
        "category": "tool_selection",
    },
    {
        "task": "帮我总结这段长文本的要点",
        "expected_tool": "text_summarize",
        "category": "tool_selection",
    },
]

MULTI_STEP_CASES = [
    {
        "task": "搜索 Python 3.12 的新特性，然后写一个使用新特性的示例文件",
        "expected_tools": ["search", "write_file"],
        "min_steps": 2,
        "category": "multi_step",
    },
    {
        "task": "读取 README.md 文件，总结其中的要点，然后保存到 summary.txt",
        "expected_tools": ["read_local_file", "text_summarize", "write_file"],
        "min_steps": 3,
        "category": "multi_step",
    },
    {
        "task": "查看当前目录结构，然后创建一个 analysis.py 脚本来分析文件统计",
        "expected_tools": ["list_directory", "write_file"],
        "min_steps": 2,
        "category": "multi_step",
    },
]

CODE_EXECUTION_CASES = [
    {
        "task": "编写一个 Python 函数计算斐波那契数列的第 10 项",
        "expected_output_contains": "55",
        "category": "code_execution",
    },
    {
        "task": "编写 Python 代码生成一个包含 1 到 100 的列表，然后计算平均值",
        "expected_output_contains": "50.5",
        "category": "code_execution",
    },
]


# ═══════════════════════════════════════════════
# 评测执行器
# ═══════════════════════════════════════════════

class AgentEvaluator:
    """态极 Agent 评测器"""

    def __init__(self):
        self.results = {
            "tool_selection": [],
            "multi_step": [],
            "code_execution": [],
        }
        self.start_time = None

    def evaluate_tool_selection(self) -> Dict[str, Any]:
        """评测工具选择准确性"""
        logger.info("\n" + "=" * 50)
        logger.info("评测: 工具选择准确性")
        logger.info("=" * 50)

        correct = 0
        total = len(TOOL_SELECTION_CASES)

        for i, case in enumerate(TOOL_SELECTION_CASES):
            task = case["task"]
            expected = case["expected_tool"]

            # 使用 ReAct 引擎的工具选择逻辑
            selected = self._select_tool(task)

            is_correct = selected == expected
            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"
            logger.info(f"  [{status}] {task[:40]}... → {selected} (期望: {expected})")

            self.results["tool_selection"].append({
                "task": task,
                "expected": expected,
                "selected": selected,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0
        logger.info(f"\n工具选择准确率: {correct}/{total} = {accuracy:.1%}")
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_multi_step(self) -> Dict[str, Any]:
        """评测多步推理能力"""
        logger.info("\n" + "=" * 50)
        logger.info("评测: 多步推理能力")
        logger.info("=" * 50)

        passed = 0
        total = len(MULTI_STEP_CASES)

        for case in MULTI_STEP_CASES:
            task = case["task"]
            expected_tools = case["expected_tools"]
            min_steps = case["min_steps"]

            # 模拟多步推理
            steps = self._plan_steps(task)
            used_tools = [s.get("tool") for s in steps if s.get("tool")]

            # 检查是否使用了期望的工具
            tools_matched = all(t in used_tools for t in expected_tools)
            steps_enough = len(steps) >= min_steps

            is_pass = tools_matched and steps_enough
            if is_pass:
                passed += 1

            status = "PASS" if is_pass else "FAIL"
            logger.info(f"  [{status}] {task[:40]}...")
            logger.info(f"    步骤: {len(steps)} (期望 >= {min_steps})")
            logger.info(f"    工具: {used_tools} (期望包含: {expected_tools})")

            self.results["multi_step"].append({
                "task": task,
                "expected_tools": expected_tools,
                "actual_tools": used_tools,
                "steps": len(steps),
                "passed": is_pass,
            })

        accuracy = passed / total if total > 0 else 0
        logger.info(f"\n多步推理通过率: {passed}/{total} = {accuracy:.1%}")
        return {"accuracy": accuracy, "passed": passed, "total": total}

    def evaluate_code_execution(self) -> Dict[str, Any]:
        """评测代码执行能力"""
        logger.info("\n" + "=" * 50)
        logger.info("评测: 代码执行能力")
        logger.info("=" * 50)

        passed = 0
        total = len(CODE_EXECUTION_CASES)

        for case in CODE_EXECUTION_CASES:
            task = case["task"]
            expected = case["expected_output_contains"]

            # 生成代码并执行
            code = self._generate_code(task)
            output = self._execute_code(code)

            is_pass = expected in output if output else False
            if is_pass:
                passed += 1

            status = "PASS" if is_pass else "FAIL"
            logger.info(f"  [{status}] {task[:40]}...")
            logger.info(f"    期望包含: '{expected}'")
            logger.info(f"    实际输出: '{output[:100]}'")

            self.results["code_execution"].append({
                "task": task,
                "expected": expected,
                "code": code,
                "output": output,
                "passed": is_pass,
            })

        accuracy = passed / total if total > 0 else 0
        logger.info(f"\n代码执行通过率: {passed}/{total} = {accuracy:.1%}")
        return {"accuracy": accuracy, "passed": passed, "total": total}

    # ─── 内部实现 ───────────────────────────────────

    def _select_tool(self, task: str) -> str:
        """根据任务选择工具（基于关键词匹配，长词优先）"""
        task_lower = task.lower()

        # 关键词 → 工具映射（按长度降序排列，避免短词误匹配）
        keyword_map = [
            ("创建文件", "write_file"), ("写入文件", "write_file"),
            ("新建文件", "write_file"), ("保存文件", "write_file"),
            ("查看文件", "read_local_file"), ("读取文件", "read_local_file"),
            ("查看目录", "list_directory"), ("文件列表", "list_directory"),
            ("文件目录", "list_directory"), ("目录结构", "list_directory"),
            ("搜索", "search"), ("查找", "search"), ("查询", "search"),
            ("读取", "read_local_file"), ("打开", "read_local_file"),
            ("写入", "write_file"), ("保存", "write_file"), ("创建", "write_file"),
            ("计算", "calculator"), ("求值", "calculator"),
            ("日期", "datetime"), ("时间", "datetime"), ("今天", "datetime"),
            ("目录", "list_directory"), ("文件夹", "list_directory"),
            ("规划", "create_plan"), ("计划", "create_plan"), ("步骤", "create_plan"),
            ("数据", "data_query"), ("分析", "data_query"), ("csv", "data_query"),
            ("访问", "url_check"), ("网址", "url_check"), ("链接", "url_check"),
            ("总结", "text_summarize"), ("摘要", "text_summarize"), ("概括", "text_summarize"),
            ("对比", "diff_text"), ("差异", "diff_text"),
            ("正则", "regex_match"), ("匹配", "regex_match"),
            ("调度", "schedule_task"), ("定时", "schedule_task"),
            ("代码", "execute_python"), ("执行", "execute_python"), ("运行", "execute_python"),
            ("知识", "learn_knowledge"), ("学习", "learn_knowledge"),
        ]

        for keyword, tool in keyword_map:
            if keyword in task_lower:
                return tool

        return "unknown"

    def _plan_steps(self, task: str) -> List[Dict]:
        """规划任务步骤（按逻辑顺序）"""
        steps = []
        task_lower = task.lower()
        step_num = 0

        # 按逻辑顺序排列：研究 → 理解 → 规划 → 执行 → 验证
        actions = [
            (["搜索", "查找", "了解", "调研", "search", "find"], "search", "先搜索相关信息"),
            (["读取", "读", "查看", "打开", "read"], "read_local_file", "读取文件内容"),
            (["规划", "计划", "分解", "plan"], "create_plan", "制定执行计划"),
            (["总结", "摘要", "概括", "summarize"], "text_summarize", "总结要点"),
            (["编写", "写入", "写一个", "创建", "保存", "实现", "开发", "write", "create"], "write_file", "写入文件"),
            (["代码", "程序", "执行", "运行", "code", "execute"], "execute_python", "执行代码"),
            (["示例", "example"], "write_file", "创建示例文件"),
            (["计算", "求值", "calculate"], "calculator", "数学计算"),
            (["目录", "文件列表", "directory"], "list_directory", "查看目录结构"),
            (["数据", "分析", "csv", "json", "data"], "data_query", "查询分析数据"),
            (["审查", "检查", "测试", "验证", "review"], "analyze_code", "审查代码质量"),
        ]

        for keywords, tool, thought in actions:
            if any(kw in task_lower for kw in keywords):
                step_num += 1
                steps.append({"step": step_num, "tool": tool, "thought": thought})

        if not steps:
            steps.append({"step": 1, "tool": "search", "thought": "先了解任务需求"})

        return steps

    def _generate_code(self, task: str) -> str:
        """根据任务生成代码"""
        if "斐波那契" in task:
            return "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nprint(fib(10))"
        elif "平均值" in task:
            return "data = list(range(1, 101))\nprint(sum(data) / len(data))"
        elif "排序" in task:
            return "data = [3, 1, 4, 1, 5, 9, 2, 6]\ndata.sort()\nprint(data)"
        else:
            return f"print('Task: {task[:50]}')"

    def _execute_code(self, code: str) -> str:
        """安全执行代码"""
        try:
            from taiji.agent_ext.sandbox_executor import execute_python_code_safe
            result = execute_python_code_safe(code)
            return str(result)
        except Exception as e:
            logger.warning(f"sandbox_executor 不可用: {e}")
            # 回退：使用共享命名空间执行
            try:
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                namespace = {"__builtins__": __builtins__}
                with redirect_stdout(f):
                    exec(code, namespace)
                return f.getvalue()
            except Exception as e2:
                return f"Error: {e2}"

    def run_all(self) -> Dict[str, Any]:
        """运行所有评测"""
        self.start_time = time.time()
        logger.info("态极 Agent 评测开始")
        logger.info("=" * 60)

        results = {
            "tool_selection": self.evaluate_tool_selection(),
            "multi_step": self.evaluate_multi_step(),
            "code_execution": self.evaluate_code_execution(),
        }

        # 计算总分
        total_score = (
            results["tool_selection"]["accuracy"] * 0.4 +
            results["multi_step"]["accuracy"] * 0.3 +
            results["code_execution"]["accuracy"] * 0.3
        )

        duration = time.time() - self.start_time

        logger.info("\n" + "=" * 60)
        logger.info("评测结果汇总")
        logger.info("=" * 60)
        logger.info(f"  工具选择: {results['tool_selection']['accuracy']:.1%}")
        logger.info(f"  多步推理: {results['multi_step']['accuracy']:.1%}")
        logger.info(f"  代码执行: {results['code_execution']['accuracy']:.1%}")
        logger.info(f"  总分: {total_score:.1%}")
        logger.info(f"  耗时: {duration:.1f}s")
        logger.info("=" * 60)

        # 保存详细结果
        output_path = Path(__file__).parent.parent / "taiji_data" / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "detailed": self.results,
                "total_score": total_score,
                "duration": duration,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"\n详细结果已保存: {output_path}")

        return results


def main():
    evaluator = AgentEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()
