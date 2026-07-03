"""
态极 (Taiji) Agent 轨迹训练数据体系
血液 — 为原生 Agent 模型设计的训练数据格式

不同于简单的问答对，Agent 轨迹包含完整的感知→思考→行动→观察→反思循环。
"""
import json
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("Taiji.AgentData")


class TrajectoryPhase(str, Enum):
    """轨迹阶段"""
    PERCEPTION = "perception"
    THINKING = "thinking"
    PLANNING = "planning"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    MEMORY = "memory"
    FINAL = "final"


@dataclass
class TrajectoryStep:
    """轨迹中的一步"""
    phase: TrajectoryPhase
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    reflection_type: Optional[str] = None
    memory_slot: Optional[int] = None

    def to_training_text(self) -> str:
        """转换为训练文本格式"""
        if self.phase == TrajectoryPhase.PERCEPTION:
            return f"<observe>{self.content}</observe>"
        elif self.phase == TrajectoryPhase.THINKING:
            return f"<think>{self.content}</think>"
        elif self.phase == TrajectoryPhase.PLANNING:
            return f"<plan>{self.content}</plan>"
        elif self.phase == TrajectoryPhase.ACTION:
            args_str = json.dumps(self.tool_args or {}, ensure_ascii=False)
            return f"<tool_call>{self.tool_name} {args_str}</tool_call>"
        elif self.phase == TrajectoryPhase.OBSERVATION:
            return f"<tool_result>{self.tool_result}</tool_result>"
        elif self.phase == TrajectoryPhase.REFLECTION:
            tag = self.reflection_type or "confirm"
            return f"<reflect><{tag}>{self.content}</{tag}></reflect>"
        elif self.phase == TrajectoryPhase.MEMORY:
            slot = self.memory_slot if self.memory_slot is not None else 0
            return f'<mem_write slot="{slot}">{self.content}</mem_write>'
        elif self.phase == TrajectoryPhase.FINAL:
            return f"<final_answer>{self.content}"
        return self.content


@dataclass
class AgentTrajectory:
    """完整的 Agent 任务轨迹"""
    task: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    environment: Optional[Dict] = None
    success: bool = True

    def to_training_sample(self) -> Dict[str, Any]:
        perception_parts = []
        if self.environment:
            if "workspace_files" in self.environment:
                files = "\n".join(self.environment["workspace_files"])
                perception_parts.append(f"<observe><tree>{files}</tree></observe>")
            if "system_info" in self.environment:
                info = "\n".join(f"{k}={v}" for k, v in self.environment["system_info"].items())
                perception_parts.append(f"<observe><state>{info}</state></observe>")
        trajectory_text = "\n".join(step.to_training_text() for step in self.steps)
        return {
            "task": self.task,
            "perception": "\n".join(perception_parts),
            "trajectory": trajectory_text,
            "success": self.success,
            "num_steps": len(self.steps),
        }


def get_agent_training_data() -> List[Dict[str, Any]]:
    """获取 Agent 轨迹训练数据（完整感知→思考→行动→观察→反思循环）"""
    return [
        {
            "task": "创建一个 Python 计算器项目",
            "environment": {"workspace_files": ["agent_workspace/"], "system_info": {"os": "windows", "python": "3.11"}},
            "steps": [
                {"phase": "perception", "content": "agent_workspace/ (空目录)"},
                {"phase": "thinking", "content": "用户想创建一个 Python 计算器项目。需要先规划步骤，然后逐步执行。"},
                {"phase": "planning", "content": '<plan task="创建 Python 计算器"><step id="1" status="active">创建 main.py</step><step id="2" status="pending">创建 README.md</step><step id="3" status="pending">测试运行</step></plan>'},
                {"phase": "action", "tool_name": "write_file", "tool_args": {"input": "calculator/main.py | def add(a, b): return a + b\ndef subtract(a, b): return a - b\ndef multiply(a, b): return a * b\ndef divide(a, b): return a / b if b != 0 else 'Error'\n\nprint(add(3, 5))"}},
                {"phase": "observation", "tool_result": "文件创建成功"},
                {"phase": "reflection", "content": "main.py 创建成功", "reflection_type": "confirm"},
                {"phase": "memory", "content": "已创建 calculator/main.py，包含加减乘除四个函数"},
                {"phase": "action", "tool_name": "execute_python", "tool_args": {"input": "def add(a, b): return a + b\nprint(add(3, 5))"}},
                {"phase": "observation", "tool_result": "8"},
                {"phase": "reflection", "content": "代码执行成功，输出正确", "reflection_type": "confirm"},
                {"phase": "final", "content": "已创建 Python 计算器项目，包含 calculator/main.py，支持加减乘除四种运算。测试运行正常。"},
            ],
            "success": True,
        },
        {
            "task": "读取 requirements.txt 并安装缺失的包",
            "environment": {"workspace_files": ["requirements.txt", "main.py"]},
            "steps": [
                {"phase": "perception", "content": "工作台有 requirements.txt 和 main.py"},
                {"phase": "thinking", "content": "需要先读取 requirements.txt 查看依赖列表，然后逐个检查并安装。"},
                {"phase": "action", "tool_name": "read_local_file", "tool_args": {"input": "requirements.txt"}},
                {"phase": "observation", "tool_result": "flask>=2.0\nrequests\nbeautifulsoup4"},
                {"phase": "memory", "content": "requirements.txt 包含 flask, requests, beautifulsoup4"},
                {"phase": "action", "tool_name": "install_dependency", "tool_args": {"input": "flask requests beautifulsoup4"}},
                {"phase": "observation", "tool_result": "Successfully installed flask-3.0 requests-2.31 beautifulsoup4-4.12"},
                {"phase": "reflection", "content": "所有依赖安装成功", "reflection_type": "confirm"},
                {"phase": "final", "content": "已读取 requirements.txt 并安装了 flask、requests、beautifulsoup4 三个依赖包。"},
            ],
            "success": True,
        },
        {
            "task": "运行 test.py 并修复错误",
            "environment": {"workspace_files": ["test.py"]},
            "steps": [
                {"phase": "perception", "content": "工作台有 test.py"},
                {"phase": "thinking", "content": "用户想运行测试并修复错误。先运行看看有什么问题。"},
                {"phase": "action", "tool_name": "execute_python", "tool_args": {"input": "print(1/0)"}},
                {"phase": "observation", "tool_result": "ZeroDivisionError: division by zero"},
                {"phase": "reflection", "content": "除零错误", "reflection_type": "detect"},
                {"phase": "reflection", "content": "代码中存在除以零的操作", "reflection_type": "cause"},
                {"phase": "reflection", "content": "添加零值检查", "reflection_type": "correct"},
                {"phase": "action", "tool_name": "edit_file", "tool_args": {"input": "test.py | print(1/0) | x = 1; y = 0; print(x/y if y != 0 else 'Error: division by zero')"}},
                {"phase": "observation", "tool_result": "编辑成功"},
                {"phase": "action", "tool_name": "execute_python", "tool_args": {"input": "x = 1; y = 0; print(x/y if y != 0 else 'Error: division by zero')"}},
                {"phase": "observation", "tool_result": "Error: division by zero"},
                {"phase": "reflection", "content": "修复成功，程序不再崩溃", "reflection_type": "confirm"},
                {"phase": "final", "content": "已修复 test.py 中的除零错误，添加了零值检查。"},
            ],
            "success": True,
        },
        {
            "task": "用态极识别一张图片的内容",
            "environment": {"workspace_files": ["photo.jpg"]},
            "steps": [
                {"phase": "perception", "content": "工作台有 photo.jpg"},
                {"phase": "thinking", "content": "用户想识别图片内容，使用 describe_image 工具。"},
                {"phase": "action", "tool_name": "describe_image", "tool_args": {"input": "photo.jpg"}},
                {"phase": "observation", "tool_result": "这是一张海滩照片，有蓝色的天空和白色的沙滩"},
                {"phase": "reflection", "content": "图像识别成功", "reflection_type": "confirm"},
                {"phase": "memory", "content": "photo.jpg 是海滩照片"},
                {"phase": "final", "content": "这张图片是一张海滩照片，画面中有蓝色的天空和白色的沙滩。"},
            ],
            "success": True,
        },
        {
            "task": "把一段语音转成文字",
            "environment": {"workspace_files": ["meeting.mp3"]},
            "steps": [
                {"phase": "perception", "content": "工作台有 meeting.mp3"},
                {"phase": "thinking", "content": "用户想把语音转成文字，使用 transcribe_audio 工具。"},
                {"phase": "action", "tool_name": "transcribe_audio", "tool_args": {"input": "meeting.mp3"}},
                {"phase": "observation", "tool_result": "今天下午三点开会，讨论项目进度"},
                {"phase": "reflection", "content": "语音识别成功", "reflection_type": "confirm"},
                {"phase": "final", "content": "语音内容转写结果：今天下午三点开会，讨论项目进度"},
            ],
            "success": True,
        },
    ]


class AgentTrajectoryDataset:
    """Agent 轨迹数据集（PyTorch Dataset）"""
    def __init__(self, tokenizer, trajectories: List[Dict], max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for traj_data in trajectories:
            traj = AgentTrajectory(
                task=traj_data["task"],
                environment=traj_data.get("environment"),
                success=traj_data.get("success", True),
            )
            for step_data in traj_data.get("steps", []):
                traj.steps.append(TrajectoryStep(
                    phase=TrajectoryPhase(step_data["phase"]),
                    content=step_data.get("content", ""),
                    tool_name=step_data.get("tool_name"),
                    tool_args=step_data.get("tool_args"),
                    tool_result=step_data.get("tool_result"),
                    reflection_type=step_data.get("reflection_type"),
                ))
            self.samples.append(traj.to_training_sample())
        logger.info(f"Agent trajectory dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]
        input_text = f"[用户] {sample['task']}\n[助手] "
        input_ids = self.tokenizer._encode(input_text)
        target_ids = self.tokenizer._encode(sample['trajectory'])
        max_half = self.max_length // 2
        input_ids = input_ids[:max_half]
        target_ids = target_ids[:max_half]
        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids
        pad_len = self.max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tool_target": torch.tensor(-100, dtype=torch.long),
        }


def build_agent_dataset(tokenizer, extra_data=None, max_length=1024):
    """构建 Agent 轨迹训练数据集"""
    data = get_agent_training_data()
    if extra_data:
        data.extend(extra_data)
    return AgentTrajectoryDataset(tokenizer, data, max_length)
