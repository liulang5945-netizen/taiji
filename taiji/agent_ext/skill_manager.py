"""
技能管理器
管理 Agent 的技能学习和激活。
"""
import json
import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("Taiji.SkillManager")


@dataclass
class Skill:
    """技能定义"""
    id: str
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    examples: List[Dict] = field(default_factory=list)
    success_count: int = 0
    fail_count: int = 0


class SkillManager:
    """技能管理器"""

    def __init__(self, data_dir: str = None):
        self._skills: Dict[str, Skill] = {}
        self._active_skill: Optional[Skill] = None
        self._data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "taiji_data", "skills"
        )
        os.makedirs(self._data_dir, exist_ok=True)
        self._load_skills()

    def _load_skills(self):
        """从磁盘加载技能"""
        skills_file = os.path.join(self._data_dir, "skills.json")
        if os.path.exists(skills_file):
            try:
                with open(skills_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data:
                    skill = Skill(**s)
                    self._skills[skill.id] = skill
                logger.info(f"Loaded {len(self._skills)} skills")
            except Exception as e:
                logger.warning(f"Failed to load skills: {e}")

    def _save_skills(self):
        """保存技能到磁盘"""
        skills_file = os.path.join(self._data_dir, "skills.json")
        try:
            data = [
                {
                    "id": s.id, "name": s.name, "description": s.description,
                    "tools": s.tools, "system_prompt": s.system_prompt,
                    "examples": s.examples, "success_count": s.success_count,
                    "fail_count": s.fail_count,
                }
                for s in self._skills.values()
            ]
            with open(skills_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save skills: {e}")

    def get_skill_system_prompt(self) -> str:
        """获取当前激活技能的系统提示"""
        if self._active_skill and self._active_skill.system_prompt:
            return self._active_skill.system_prompt
        return ""

    def activate_skill(self, skill_id: str) -> Optional[Skill]:
        """激活一个技能"""
        if skill_id in self._skills:
            self._active_skill = self._skills[skill_id]
            logger.info(f"Activated skill: {skill_id}")
            return self._active_skill
        logger.warning(f"Skill not found: {skill_id}")
        return None

    def deactivate_skill(self):
        """停用当前技能"""
        self._active_skill = None

    def learn_from_task(self, task: str, steps: List[Dict], answer: str):
        """从成功的任务中学习，创建或更新技能"""
        if not task or not answer:
            return

        # 简单实现：基于任务内容生成技能 ID
        skill_id = f"learned_{hash(task) % 10000:04d}"

        if skill_id in self._skills:
            skill = self._skills[skill_id]
            skill.success_count += 1
            # 添加新示例
            if len(skill.examples) < 5:
                skill.examples.append({
                    "task": task,
                    "steps": steps[:3],  # 只保留前3步
                    "answer": answer[:200],
                })
        else:
            # 创建新技能
            skill = Skill(
                id=skill_id,
                name=task[:50],
                description=f"从任务中学到: {task[:100]}",
                tools=[s.get("action", "").split("(")[0] for s in steps if s.get("action")],
                examples=[{
                    "task": task,
                    "steps": steps[:3],
                    "answer": answer[:200],
                }],
                success_count=1,
            )
            self._skills[skill_id] = skill
            logger.info(f"Learned new skill: {skill_id} from task: {task[:50]}")

        self._save_skills()

    def list_skills(self) -> List[Skill]:
        """列出所有技能"""
        return list(self._skills.values())

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """获取指定技能"""
        return self._skills.get(skill_id)

    def delete_skill(self, skill_id: str) -> bool:
        """删除技能"""
        if skill_id in self._skills:
            del self._skills[skill_id]
            self._save_skills()
            return True
        return False


# 全局实例
skill_manager = SkillManager()
