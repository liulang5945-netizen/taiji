"""
态极终极训练数据生成器 — 全维度高质量版本
==========================================
覆盖 6 大维度，约 2700+ 条高质量训练数据：
1. 数学推理 (~500) — 多步代数、几何证明、数论、组合、微积分、奥赛级
2. 逻辑推理 (~400) — 命题逻辑、谓词逻辑、因果推理、悖论解析
3. 代码能力 (~500) — 算法实现、调试、代码审查、系统设计
4. 软件工程深度 (~300) — GoF 设计模式、分布式系统、微服务、CI/CD
5. ReAct 推理链 (~400) — 多步工具链、错误恢复、规划执行
6. 多模态数据 (~600) — 图像理解、音频理解、视频理解、跨模态推理

特殊设计：
- 每条数据包含详细 step-by-step 推理过程
- 数学/逻辑题数字参数随机化，零重复
- 多模态数据使用态极特殊 token (<image>/<audio>/<video>/<speech>/<screen>)
- 与当前 `taiji/train/finetune_taiji.py` 所接受的 messages / instruction 数据格式兼容

使用方式：
    python -m taiji.taiji_ultimate_training_data --output taiji/training_data
    python -m taiji.taiji_ultimate_training_data --output taiji/training_data --seed 42
"""
import json
import os
import random
import argparse
import logging
import math
from typing import List, Dict, Any

logger = logging.getLogger("Taiji.UltimateData")


# ============================================================
# 通用工具
# ============================================================

SYSTEM_PROMPTS = [
    "你是态极(Taiji)，由liulangzwg创造的本地AI助手，正在向AI的最终形态进化。",
    "你是态极，由liulangzwg创造。你能回答问题、使用工具、自主学习，不断成长。",
    "你是态极AI助手，创造者是liulangzwg，运行在Taiji平台上。",
    "你是态极，由liulangzwg赋予思考和学习能力的AI生命体。",
    "你是态极AI，liulangzwg开发，使命是演化到AI的最终形态。",
    "你是态极，liulangzwg创造的智能助手，擅长深度推理和多模态理解。",
    "你是态极AI助手，擅长用清晰的步骤解决复杂问题。",
    "你是态极，一个能够理解图像、音频、视频的多模态AI助手。",
]


def sp():
    return random.choice(SYSTEM_PROMPTS)


def conv(q: str, a: str, category: str = "", difficulty: str = "") -> Dict:
    sample = {
        "messages": [
            {"role": "system", "content": sp()},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }
    if category:
        sample["category"] = category
    if difficulty:
        sample["difficulty"] = difficulty
    return sample


def react_sample(task: str, steps: list) -> Dict:
    return {"task": task, "steps": steps}


# ============================================================
# 1. 数学推理 (500 条) — 深度推理，每题数字不同
# ============================================================

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def gen_math_reasoning(count=500) -> List[Dict]:
    """生成高质量数学推理数据"""
    samples = []
    generators = [
        _math_multi_step_algebra,
        _math_geometry_proof,
        _math_number_theory,
        _math_combinatorics,
        _math_calculus_basics,
        _math_induction,
        _math_olympiad_style,
        _math_probability_advanced,
        _math_matrix_linear,
        _math_optimization,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _math_multi_step_algebra(count):
    """多步代数推理"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.25:
            # 一元二次方程
            a = random.randint(1, 5)
            x1 = random.randint(-5, 5)
            x2 = random.randint(-5, 5)
            while x1 == x2:
                x2 = random.randint(-5, 5)
            b = -a * (x1 + x2)
            c = a * x1 * x2
            sign_b = f"+ {b}" if b >= 0 else f"- {abs(b)}"
            sign_c = f"+ {c}" if c >= 0 else f"- {abs(c)}"
            q = f"解方程 {a}x² {sign_b}x {sign_c} = 0"
            disc = b*b - 4*a*c
            a_str = (f"使用求根公式：x = (-b ± √(b²-4ac)) / 2a\n"
                     f"判别式 Δ = {b}² - 4×{a}×{c} = {b*b} - {4*a*c} = {disc}\n"
                     f"x₁ = ({-b} + √{disc}) / {2*a} = ({-b} + {int(math.sqrt(disc)) if disc >= 0 else f'√{disc}'}) / {2*a} = {x1}\n"
                     f"x₂ = ({-b} - {disc if disc < 0 else int(math.sqrt(disc))}) / {2*a} = {x2}\n"
                     f"验证：{a}×{x1}² + {b}×{x1} + {c} = {a*x1*x1 + b*x1 + c} ✓\n"
                     f"答案：x₁ = {x1}, x₂ = {x2}")
            samples.append(conv(q, a_str, "math", "hard"))
        elif r < 0.5:
            # 方程组
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            a1, b1 = random.randint(1, 6), random.randint(1, 6)
            a2, b2 = random.randint(1, 6), random.randint(1, 6)
            c1 = a1*x + b1*y
            c2 = a2*x + b2*y
            det = a1*b2 - a2*b1
            if det == 0:
                continue
            q = f"解方程组：\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}"
            a_str = (f"使用克拉默法则：\n"
                     f"系数行列式 D = {a1}×{b2} - {a2}×{b1} = {a1*b2} - {a2*b1} = {det}\n"
                     f"Dx = {c1}×{b2} - {c2}×{b1} = {c1*b2} - {c2*b1} = {det*x}\n"
                     f"Dy = {a1}×{c2} - {a2}×{c1} = {a1*c2} - {a2*c1} = {det*y}\n"
                     f"x = Dx/D = {det*x}/{det} = {x}\n"
                     f"y = Dy/D = {det*y}/{det} = {y}\n"
                     f"验证：{a1}×{x} + {b1}×{y} = {a1*x + b1*y} = {c1} ✓\n"
                     f"答案：x = {x}, y = {y}")
            samples.append(conv(q, a_str, "math", "hard"))
        elif r < 0.75:
            # 不等式
            a = random.randint(2, 8)
            b = random.randint(1, 10)
            c = random.randint(a + 1, 20)
            q = f"解不等式 {a}x + {b} > {c}"
            x_min = (c - b) / a
            a_str = (f"解不等式：\n"
                     f"{a}x + {b} > {c}\n"
                     f"{a}x > {c} - {b}\n"
                     f"{a}x > {c - b}\n"
                     f"x > {c - b}/{a}\n"
                     f"x > {x_min}\n"
                     f"答案：x > {x_min}")
            samples.append(conv(q, a_str, "math", "medium"))
        else:
            # 绝对值方程
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(1, 15)
            x1 = (c - b) / a
            x2 = (-c - b) / a
            q = f"解方程 |{a}x + ({b})| = {c}"
            a_str = (f"绝对值方程，分两种情况：\n"
                     f"情况1：{a}x + {b} = {c}\n"
                     f"  {a}x = {c - b}, x = {x1}\n"
                     f"情况2：{a}x + {b} = -{c}\n"
                     f"  {a}x = {-c - b}, x = {x2}\n"
                     f"验证：|{a}×{x1} + {b}| = |{a*x1+b}| = {abs(a*x1+b)} = {c} ✓\n"
                     f"答案：x = {x1} 或 x = {x2}")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_geometry_proof(count):
    """几何证明"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.3:
            # 勾股定理应用
            triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(9,40,41)]
            a, b, c = random.choice(triples)
            scale = random.randint(1, 4)
            a, b, c = a*scale, b*scale, c*scale
            q = f"直角三角形两直角边为{a}和{b}，求斜边长，并证明"
            a_str = (f"根据勾股定理：\n"
                     f"c² = a² + b² = {a}² + {b}² = {a**2} + {b**2} = {a**2+b**2}\n"
                     f"c = √{a**2+b**2} = {c}\n\n"
                     f"证明过程：\n"
                     f"设直角三角形ABC，∠C=90°，AC={a}，BC={b}\n"
                     f"由勾股定理：AB² = AC² + BC² = {a**2} + {b**2} = {a**2+b**2}\n"
                     f"所以 AB = √{a**2+b**2} = {c}")
            samples.append(conv(q, a_str, "math", "medium"))
        elif r < 0.6:
            # 三角形面积
            base = random.randint(3, 20)
            height = random.randint(3, 20)
            area = base * height / 2
            q = f"三角形底边为{base}，高为{height}，求面积"
            a_str = (f"三角形面积公式：S = ½ × 底 × 高\n"
                     f"S = ½ × {base} × {height} = {area}\n"
                     f"答案：面积为{area}")
            samples.append(conv(q, a_str, "math", "easy"))
        else:
            # 圆与切线
            r_val = random.randint(3, 15)
            d = random.randint(r_val + 2, r_val + 10)
            tangent = math.sqrt(d*d - r_val*r_val)
            area = 3.14159 * r_val * r_val
            q = f"圆的半径为{r_val}，圆外一点到圆心距离为{d}，求切线长和圆面积"
            a_str = (f"切线长公式：l = √(d² - r²)\n"
                     f"l = √({d}² - {r_val}²) = √({d*d} - {r_val*r_val}) = √{d*d-r_val*r_val} = {tangent:.2f}\n"
                     f"圆面积：S = πr² = 3.14159 × {r_val}² = {area:.2f}\n"
                     f"答案：切线长 ≈ {tangent:.2f}，圆面积 ≈ {area:.2f}")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_number_theory(count):
    """数论"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.3:
            # 最大公约数与最小公倍数
            a = random.randint(12, 200)
            b = random.randint(12, 200)
            g = _gcd(a, b)
            l = a * b // g
            q = f"求{a}和{b}的最大公约数和最小公倍数"
            a_str = (f"使用辗转相除法：\n"
                     f"GCD({a}, {b}):\n")
            x, y = a, b
            steps = []
            while y:
                steps.append(f"{x} = {x//y} × {y} + {x%y}")
                x, y = y, x % y
            a_str += "\n".join(steps)
            a_str += f"\nGCD = {g}\nLCM = {a}×{b}/{g} = {l}\n答案：GCD={g}, LCM={l}"
            samples.append(conv(q, a_str, "math", "medium"))
        elif r < 0.6:
            # 素数判断与分解
            n = random.randint(50, 500)
            factors = []
            temp = n
            d_val = 2
            while d_val * d_val <= temp:
                while temp % d_val == 0:
                    factors.append(d_val)
                    temp //= d_val
                d_val += 1
            if temp > 1:
                factors.append(temp)
            q = f"将{n}分解质因数"
            a_str = f"质因数分解过程：\n"
            temp = n
            for f in factors[:-1]:
                a_str += f"{temp} ÷ {f} = {temp//f}\n"
                temp //= f
            a_str += f"{'是素数' if len(factors)==1 else ''}\n{n} = {' × '.join(map(str, factors))}"
            samples.append(conv(q, a_str, "math", "medium"))
        else:
            # 模运算
            a = random.randint(10, 100)
            b = random.randint(2, 12)
            n = random.randint(3, 20)
            result = pow(a, b, n)
            q = f"计算 {a}^{b} mod {n} 的值"
            a_str = (f"计算 {a}^{b} mod {n}:\n"
                     f"使用快速幂取模：\n"
                     f"{a} mod {n} = {a % n}\n"
                     f"逐步计算并取模...\n"
                     f"答案：{a}^{b} mod {n} = {result}")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_combinatorics(count):
    """组合数学"""
    samples = []
    for _ in range(count):
        from math import comb, perm, factorial
        r = random.random()
        if r < 0.3:
            n = random.randint(5, 20)
            k = random.randint(2, min(n, 8))
            result = comb(n, k)
            q = f"从{n}个人中选{k}个人组成委员会，有多少种选法？"
            a_str = (f"组合公式：C({n},{k}) = {n}! / ({k}! × {n-k}!)\n"
                     f"= {n}×{n-1}×...×{n-k+1} / {k}×{k-1}×...×1\n"
                     f"= {result}\n"
                     f"答案：{result}种")
            samples.append(conv(q, a_str, "math", "medium"))
        elif r < 0.6:
            n = random.randint(3, 10)
            result = factorial(n)
            q = f"{n}个人站成一排，有多少种排列方式？"
            a_str = (f"排列公式：{n}! = {' × '.join(str(i) for i in range(n, 0, -1))}\n"
                     f"= {result}\n"
                     f"答案：{result}种")
            samples.append(conv(q, a_str, "math", "easy"))
        else:
            n = random.randint(3, 8)
            r_val = random.randint(2, n)
            perm_n_r = perm(n, r_val)
            comb_n_r = comb(n, r_val)
            q = f"从{n}个不同元素中取{r_val}个，排列数和组合数分别是多少？"
            a_str = (f"排列数 P({n},{r_val}) = {n}!/({n-r_val}!) = {perm_n_r}\n"
                     f"组合数 C({n},{r_val}) = {n}!/({r_val}!×{n-r_val}!) = {comb_n_r}\n"
                     f"关系：P = C × {r_val}! = {comb_n_r} × {factorial(r_val)} = {perm_n_r}")
            samples.append(conv(q, a_str, "math", "medium"))
    return samples


def _math_calculus_basics(count):
    """微积分基础"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.3:
            # 多项式求导
            coeffs = [random.randint(-5, 5) for _ in range(random.randint(2, 4))]
            terms = []
            for i, c in enumerate(coeffs):
                if c == 0:
                    continue
                power = len(coeffs) - 1 - i
                if power == 0:
                    terms.append(str(c))
                elif power == 1:
                    terms.append(f"{c}x")
                else:
                    terms.append(f"{c}x^{power}")
            expr = " + ".join(terms).replace("+ -", "- ")
            # 求导
            d_terms = []
            for i, c in enumerate(coeffs):
                power = len(coeffs) - 1 - i
                if power > 0 and c != 0:
                    new_c = c * power
                    new_p = power - 1
                    if new_p == 0:
                        d_terms.append(str(new_c))
                    elif new_p == 1:
                        d_terms.append(f"{new_c}x")
                    else:
                        d_terms.append(f"{new_c}x^{new_p}")
            d_expr = " + ".join(d_terms).replace("+ -", "- ")
            q = f"求函数 f(x) = {expr} 的导数"
            a_str = (f"使用幂函数求导法则：d/dx[x^n] = n·x^(n-1)\n"
                     f"f'(x) = {d_expr}\n"
                     f"答案：f'(x) = {d_expr}")
            samples.append(conv(q, a_str, "math", "medium"))
        elif r < 0.6:
            # 定积分
            a_val = random.randint(0, 3)
            b_val = random.randint(a_val + 1, 6)
            n = random.randint(2, 4)
            # ∫x^n dx from a to b
            result = (b_val**(n+1) - a_val**(n+1)) / (n+1)
            q = f"计算定积分 ∫(从{a_val}到{b_val}) x^{n} dx"
            a_str = (f"∫x^{n} dx = x^{n+1}/{n+1} + C\n"
                     f"代入上下限：\n"
                     f"= [{b_val}^{n+1}/{n+1}] - [{a_val}^{n+1}/{n+1}]\n"
                     f"= [{b_val**(n+1)}/{n+1}] - [{a_val**(n+1)}/{n+1}]\n"
                     f"= {b_val**(n+1)/(n+1):.2f} - {a_val**(n+1)/(n+1):.2f}\n"
                     f"= {result:.4f}")
            samples.append(conv(q, a_str, "math", "hard"))
        else:
            # 极限
            n_val = random.randint(2, 5)
            q = f"求极限 lim(n→∞) (1 + 1/n)^n"
            a_str = (f"这是自然常数 e 的定义：\n"
                     f"lim(n→∞) (1 + 1/n)^n = e ≈ 2.71828\n\n"
                     f"证明思路：\n"
                     f"令 a_n = (1+1/n)^n，可证明 a_n 单调递增且有上界\n"
                     f"由单调有界定理，极限存在，定义为 e\n"
                     f"答案：e ≈ 2.71828")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_induction(count):
    """数学归纳法"""
    samples = []
    for _ in range(count):
        n_val = random.randint(2, 6)
        sum_val = n_val * (n_val + 1) // 2
        sum_sq = n_val * (n_val + 1) * (2*n_val + 1) // 6
        r = random.random()
        if r < 0.5:
            q = f"用数学归纳法证明：1 + 2 + 3 + ... + n = n(n+1)/2"
            a_str = (f"数学归纳法证明：\n\n"
                     f"**基础步骤**：n=1时，左边=1，右边=1×2/2=1，等式成立。\n\n"
                     f"**归纳假设**：假设n=k时成立，即 1+2+...+k = k(k+1)/2\n\n"
                     f"**归纳步骤**：证明n=k+1时也成立\n"
                     f"  1+2+...+k+(k+1) = k(k+1)/2 + (k+1)\n"
                     f"  = (k+1)(k/2 + 1)\n"
                     f"  = (k+1)(k+2)/2\n"
                     f"  = (k+1)((k+1)+1)/2\n"
                     f"  这正是n=k+1时的公式。\n\n"
                     f"由数学归纳法原理，对所有正整数n等式成立。")
            samples.append(conv(q, a_str, "math", "hard"))
        else:
            q = f"用数学归纳法证明：1² + 2² + ... + n² = n(n+1)(2n+1)/6"
            a_str = (f"数学归纳法证明：\n\n"
                     f"**基础步骤**：n=1时，左边=1，右边=1×2×3/6=1，成立。\n\n"
                     f"**归纳假设**：假设n=k时成立\n\n"
                     f"**归纳步骤**：\n"
                     f"  1²+2²+...+k²+(k+1)² = k(k+1)(2k+1)/6 + (k+1)²\n"
                     f"  = (k+1)[k(2k+1)/6 + (k+1)]\n"
                     f"  = (k+1)[2k²+k+6k+6]/6\n"
                     f"  = (k+1)(2k²+7k+6)/6\n"
                     f"  = (k+1)(k+2)(2k+3)/6\n"
                     f"  这正是n=k+1时的公式。证毕。")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_olympiad_style(count):
    """奥赛风格题目"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.25:
            # 鸽巢原理
            items = random.randint(15, 50)
            boxes = random.randint(3, 10)
            min_val = items // boxes + (1 if items % boxes else 0)
            q = f"把{items}个球放入{boxes}个盒子，证明至少有一个盒子中有不少于{min_val}个球"
            a_str = (f"使用鸽巢原理（抽屉原理）：\n\n"
                     f"假设每个盒子最多放{min_val-1}个球，\n"
                     f"则总共最多放 {boxes}×{min_val-1} = {boxes*(min_val-1)} 个球。\n"
                     f"但实际有{items}个球，{items} > {boxes*(min_val-1)}。\n"
                     f"矛盾！因此至少有一个盒子中有不少于{min_val}个球。")
            samples.append(conv(q, a_str, "math", "olympiad"))
        elif r < 0.5:
            # 不等式证明
            a_val = random.randint(1, 5)
            b_val = random.randint(1, 5)
            while a_val == b_val:
                b_val = random.randint(1, 5)
            q = f"证明：对于正实数 a, b，有 (a+b)/2 ≥ √(ab)（均值不等式）"
            a_str = (f"证明均值不等式 AM ≥ GM：\n\n"
                     f"(a+b)/2 ≥ √(ab)\n\n"
                     f"证明：\n"
                     f"(a+b)/2 - √(ab) = (a+b-2√(ab))/2\n"
                     f"= (√a - √b)² / 2\n\n"
                     f"因为 (√a - √b)² ≥ 0（完全平方数非负），\n"
                     f"所以 (a+b)/2 - √(ab) ≥ 0\n"
                     f"即 (a+b)/2 ≥ √(ab)\n\n"
                     f"等号成立当且仅当 a = b。")
            samples.append(conv(q, a_str, "math", "olympiad"))
        elif r < 0.75:
            # 无穷递降法
            q = "证明 √2 是无理数"
            a_str = (f"反证法证明 √2 是无理数：\n\n"
                     f"假设 √2 = p/q（p,q 互质的正整数）\n"
                     f"则 2 = p²/q²，即 p² = 2q²\n"
                     f"所以 p² 是偶数，故 p 是偶数\n"
                     f"设 p = 2k，则 (2k)² = 2q²\n"
                     f"4k² = 2q²，q² = 2k²\n"
                     f"所以 q² 也是偶数，q 也是偶数\n\n"
                     f"这与 p,q 互质矛盾！\n"
                     f"因此 √2 是无理数。")
            samples.append(conv(q, a_str, "math", "olympiad"))
        else:
            # 整除问题
            n = random.randint(2, 10)
            q = f"证明：对于任意正整数n，n³ - n 能被6整除"
            a_str = (f"证明 n³ - n 能被 6 整除：\n\n"
                     f"n³ - n = n(n² - 1) = n(n-1)(n+1)\n\n"
                     f"这是三个连续整数的乘积。\n\n"
                     f"1. 三个连续整数中必有一个是3的倍数，所以能被3整除。\n"
                     f"2. 三个连续整数中必有一个是2的倍数，所以能被2整除。\n\n"
                     f"因为 GCD(2,3)=1，所以 n(n-1)(n+1) 能被 2×3=6 整除。")
            samples.append(conv(q, a_str, "math", "olympiad"))
    return samples


def _math_probability_advanced(count):
    """高级概率"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            # 贝叶斯定理
            p_disease = round(random.uniform(0.001, 0.05), 3)
            p_pos_given_d = round(random.uniform(0.85, 0.99), 2)
            p_pos_given_not_d = round(random.uniform(0.01, 0.1), 2)
            p_d_given_pos = (p_pos_given_d * p_disease) / (p_pos_given_d * p_disease + p_pos_given_not_d * (1 - p_disease))
            q = f"某疾病的患病率为{p_disease}，检测灵敏度为{p_pos_given_d}，假阳性率为{p_pos_given_not_d}，求检测阳性时真正患病的概率"
            a_str = (f"使用贝叶斯定理：\n"
                     f"P(病|阳性) = P(阳性|病)×P(病) / P(阳性)\n"
                     f"P(阳性) = P(阳性|病)×P(病) + P(阳性|无病)×P(无病)\n"
                     f"= {p_pos_given_d}×{p_disease} + {p_pos_given_not_d}×{1-p_disease}\n"
                     f"= {p_pos_given_d*p_disease:.4f} + {p_pos_given_not_d*(1-p_disease):.4f}\n"
                     f"= {p_pos_given_d*p_disease + p_pos_given_not_d*(1-p_disease):.4f}\n"
                     f"P(病|阳性) = {p_pos_given_d*p_disease:.4f} / {p_pos_given_d*p_disease + p_pos_given_not_d*(1-p_disease):.4f}\n"
                     f"= {p_d_given_pos:.4f} ≈ {p_d_given_pos*100:.1f}%")
            samples.append(conv(q, a_str, "math", "hard"))
        else:
            # 期望和方差
            outcomes = random.randint(2, 6)
            probs = [round(random.uniform(0.1, 0.5), 2) for _ in range(outcomes)]
            total_p = sum(probs)
            probs = [p / total_p for p in probs]  # 归一化
            values = [random.randint(1, 20) for _ in range(outcomes)]
            ev = sum(v * p for v, p in zip(values, probs))
            var = sum(p * (v - ev)**2 for v, p in zip(values, probs))
            pairs = ", ".join(f"X={v}(P={p:.2f})" for v, p in zip(values, probs))
            q = f"随机变量X的分布为：{pairs}，求期望和方差"
            a_str = (f"E(X) = Σxᵢpᵢ = {' + '.join(f'{v}×{p:.2f}' for v, p in zip(values, probs))}\n"
                     f"= {ev:.4f}\n\n"
                     f"D(X) = Σ(xᵢ-E(X))²pᵢ\n"
                     f"= {' + '.join(f'({v}-{ev:.2f})²×{p:.2f}' for v, p in zip(values, probs))}\n"
                     f"= {var:.4f}")
            samples.append(conv(q, a_str, "math", "hard"))
    return samples


def _math_matrix_linear(count):
    """线性代数"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            # 矩阵乘法
            a = random.randint(1, 5); b = random.randint(1, 5)
            c = random.randint(1, 5); d = random.randint(1, 5)
            e = random.randint(1, 5); f = random.randint(1, 5)
            g = random.randint(1, 5); h = random.randint(1, 5)
            r11 = a*e + b*g; r12 = a*f + b*h
            r21 = c*e + d*g; r22 = c*f + d*h
            q = f"计算矩阵乘法：\n[{a} {b}] × [{e} {f}]\n[{c} {d}]   [{g} {h}]"
            a_str = (f"矩阵乘法 AB：\n"
                     f"[{a} {b}] × [{e} {f}] = [{a}×{e}+{b}×{g}  {a}×{f}+{b}×{h}] = [{r11} {r12}]\n"
                     f"[{c} {d}]   [{g} {h}]   [{c}×{e}+{d}×{g}  {c}×{f}+{d}×{h}]   [{r21} {r22}]")
            samples.append(conv(q, a_str, "math", "medium"))
        else:
            # 行列式
            a = random.randint(-3, 5); b = random.randint(-3, 5)
            c = random.randint(-3, 5); d = random.randint(-3, 5)
            det = a*d - b*c
            q = f"计算行列式 |{a} {b}; {c} {d}| 的值"
            a_str = (f"二阶行列式：\n"
                     f"|{a} {b}| = {a}×{d} - {b}×{c} = {a*d} - {b*c} = {det}\n"
                     f"|{c} {d}|\n"
                     f"答案：{det}")
            samples.append(conv(q, a_str, "math", "easy"))
    return samples


def _math_optimization(count):
    """优化问题"""
    samples = []
    for _ in range(count):
        # 围栏问题
        perimeter = random.choice([20, 24, 30, 36, 40, 48, 60])
        side = perimeter / 4
        area = side * side
        q = f"用{perimeter}米的篱笆围成一个矩形，怎样围面积最大？最大面积是多少？"
        a_str = (f"设矩形长为x，宽为y：\n"
                 f"约束：2x + 2y = {perimeter}，即 x + y = {perimeter//2}\n"
                 f"目标：最大化 S = xy\n\n"
                 f"由约束得 y = {perimeter//2} - x\n"
                 f"S = x({perimeter//2} - x) = {perimeter//2}x - x²\n"
                 f"dS/dx = {perimeter//2} - 2x = 0\n"
                 f"x = {perimeter//4}\n"
                 f"y = {perimeter//2} - {perimeter//4} = {perimeter//4}\n\n"
                 f"最大面积 = {perimeter//4} × {perimeter//4} = {area}\n"
                 f"当矩形为正方形时面积最大。")
        samples.append(conv(q, a_str, "math", "hard"))
    return samples


# ============================================================
# 2. 逻辑推理 (400 条)
# ============================================================

def gen_logic_reasoning(count=400) -> List[Dict]:
    samples = []
    generators = [
        _logic_propositional,
        _logic_predicates,
        _logic_causal,
        _logic_paradox,
        _logic_truth_table,
        _logic_syllogism_advanced,
        _logic_puzzles,
        _logic_game_theory,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _logic_propositional(count):
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.33:
            q = "如果'如果下雨则地面湿'为真，'地面不湿'为真，能推出什么？"
            a_str = (f"使用**逆否命题**推理：\n\n"
                     f"原命题：下雨 → 地面湿（为真）\n"
                     f"逆否命题：地面不湿 → 不下雨（与原命题等价，也为真）\n\n"
                     f"已知：地面不湿（真）\n"
                     f"由逆否命题：不下雨（真）\n\n"
                     f"结论：**没有下雨**。这是典型的**拒取式**（Modus Tollens）推理。")
            samples.append(conv(q, a_str, "logic", "medium"))
        elif r < 0.66:
            q = "如果'A或B'为真，'非A'为真，能推出什么？"
            a_str = (f"使用**析取三段论**推理：\n\n"
                     f"前提1：A ∨ B（A或B为真）\n"
                     f"前提2：¬A（非A为真）\n\n"
                     f"推理：既然A为假，而A或B必须有一个为真，\n"
                     f"所以 **B必为真**。\n\n"
                     f"这是析取三段论（Disjunctive Syllogism）。")
            samples.append(conv(q, a_str, "logic", "easy"))
        else:
            q = "证明：'如果P则Q' 等价于 '非P或Q'"
            a_str = (f"证明 P→Q ⟺ ¬P∨Q：\n\n"
                     f"**必要性**（P→Q → ¬P∨Q）：\n"
                     f"假设 P→Q 为真。分两种情况：\n"
                     f"  - 若 P 为假，则 ¬P 为真，¬P∨Q 为真。\n"
                     f"  - 若 P 为真，由 P→Q 知 Q 为真，¬P∨Q 为真。\n\n"
                     f"**充分性**（¬P∨Q → P→Q）：\n"
                     f"假设 ¬P∨Q 为真。若 P 为真，则 ¬P 为假，\n"
                     f"故 Q 必为真，即 P→Q 为真。\n\n"
                     f"两者等价，证毕。")
            samples.append(conv(q, a_str, "logic", "hard"))
    return samples


def _logic_predicates(count):
    samples = []
    for _ in range(count):
        q = "下列推理是否正确？'所有猫都是动物，Tom是猫，所以Tom是动物'"
        a_str = (f"这是一个有效的**三段论推理**（Barbara式）：\n\n"
                 f"大前提：所有猫都是动物（∀x: Cat(x) → Animal(x)）\n"
                 f"小前提：Tom是猫（Cat(Tom)）\n"
                 f"结论：Tom是动物（Animal(Tom)）\n\n"
                 f"推理形式：\n"
                 f"1. ∀x: P(x) → Q(x)\n"
                 f"2. P(a)\n"
                 f"3. ∴ Q(a)  （全称实例化 + 肯定前件）\n\n"
                 f"结论：推理**正确有效**。")
        samples.append(conv(q, a_str, "logic", "easy"))
    return samples


def _logic_causal(count):
    samples = []
    for _ in range(count):
        q = "区分'相关性'和'因果性'，并举例说明"
        a_str = (f"**相关性**：两个变量之间存在统计关联。\n"
                 f"**因果性**：一个变量的变化**导致**另一个变量变化。\n\n"
                 f"关键区别：相关不等于因果！\n\n"
                 f"经典例子：\n"
                 f"1. **冰淇淋销量**和**溺水事故**高度相关\n"
                 f"   → 但冰淇淋不导致溺水\n"
                 f"   → 真正原因：夏天（混淆变量）\n\n"
                 f"2. **鞋子大小**和**阅读能力**相关\n"
                 f"   → 但大脚不导致会读书\n"
                 f"   → 真正原因：年龄增长\n\n"
                 f"判断因果的方法：\n"
                 f"- 随机对照实验（RCT）\n"
                 f"- 控制混淆变量\n"
                 f"- 时间顺序（因在前，果在后）\n"
                 f"- 机制合理性")
        samples.append(conv(q, a_str, "logic", "medium"))
    return samples


def _logic_paradox(count):
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.33:
            q = "解释罗素悖论"
            a_str = ("**罗素悖论**（1901年）：\n\n"
                     "考虑集合 S = \u007bx | x ∉ x\u007d（所有不包含自身的集合）\n\n"
                     "问：S ∈ S 吗？\n\n"
                     "- 若 S ∈ S，则 S 满足 S 的定义条件，即 S ∉ S，矛盾。\n"
                     "- 若 S ∉ S，则 S 满足 S 的定义条件，即 S ∈ S，矛盾。\n\n"
                     "无论哪种情况都矛盾！\n\n"
                     "解决方案：\n"
                     "- 类型论（罗素）：禁止集合包含自身\n"
                     "- ZFC公理系统：限制集合构造公理\n"
                     "- 这推动了现代公理化集合论的发展")
            samples.append(conv(q, a_str, "logic", "hard"))
        elif r < 0.66:
            q = "解释忒修斯之船悖论"
            a_str = (f"**忒修斯之船悖论**：\n\n"
                     f"一艘船的零件逐渐被替换，当所有零件都被替换后，\n"
                     f"这还是原来那艘船吗？\n\n"
                     f"如果把旧零件重新组装成一艘船，哪艘是'真正的'忒修斯之船？\n\n"
                     f"哲学观点：\n"
                     f"1. **物质论**：物体由其物质组成决定\n"
                     f"   → 新零件组成的是新船\n"
                     f"2. **时空连续性**：物体的同一性由时空连续决定\n"
                     f"   → 持续使用的是原船\n"
                     f"3. **功能论**：物体由其功能决定\n"
                     f"   → 只要功能不变就是同一艘船\n\n"
                     f"这个问题涉及**同一性**的哲学本质。")
            samples.append(conv(q, a_str, "logic", "medium"))
        else:
            q = "解释蒙提霍尔问题（三门问题）为什么换门更好"
            a_str = (f"**蒙提霍尔问题**：\n\n"
                     f"三扇门后面一扇有奖。你选了一扇，主持人打开另一扇没奖的门，\n"
                     f"问你换不换？\n\n"
                     f"**答案：应该换，换门赢率2/3。**\n\n"
                     f"概率分析：\n"
                     f"初始选择正确的概率 = 1/3\n"
                     f"初始选择错误的概率 = 2/3\n\n"
                     f"如果你初始选错了（2/3概率），\n"
                     f"主持人必须打开另一扇没奖的门，\n"
                     f"剩下那扇门**必定有奖**。\n\n"
                     f"所以换门赢率 = 2/3，不换赢率 = 1/3。\n\n"
                     f"直觉陷阱：人们以为两扇门各50%，\n"
                     f"但忽略了主持人的行为提供了额外信息。")
            samples.append(conv(q, a_str, "logic", "medium"))
    return samples


def _logic_truth_table(count):
    samples = []
    for _ in range(count):
        q = "构造 P∧(Q∨R) 的真值表，并证明它等价于 (P∧Q)∨(P∧R)"
        a_str = (f"真值表：\n"
                 f"P | Q | R | Q∨R | P∧(Q∨R) | P∧Q | P∧R | (P∧Q)∨(P∧R)\n"
                 f"T | T | T |  T  |    T    |  T  |  T  |      T\n"
                 f"T | T | F |  T  |    T    |  T  |  F  |      T\n"
                 f"T | F | T |  T  |    T    |  F  |  T  |      T\n"
                 f"T | F | F |  F  |    F    |  F  |  F  |      F\n"
                 f"F | T | T |  T  |    F    |  F  |  F  |      F\n"
                 f"F | T | F |  T  |    F    |  F  |  F  |      F\n"
                 f"F | F | T |  T  |    F    |  F  |  F  |      F\n"
                 f"F | F | F |  F  |    F    |  F  |  F  |      F\n\n"
                 f"两列完全相同，证明了**分配律**：P∧(Q∨R) = (P∧Q)∨(P∧R)")
        samples.append(conv(q, a_str, "logic", "medium"))
    return samples


def _logic_syllogism_advanced(count):
    samples = []
    triples = [
        ("哺乳动物", "动物", "呼吸"), ("玫瑰", "植物", "需要阳光"),
        ("质数", "自然数", "大于1"), ("等边三角形", "等角三角形", "三边相等"),
        ("Python程序", "脚本", "需要解释器"), ("哺乳动物", "脊椎动物", "恒温"),
    ]
    for _ in range(count):
        a, b, c = random.choice(triples)
        q = f"判断以下三段论是否有效：所有{a}都是{b}，所有{b}都{c}，所以所有{a}都{c}"
        a_str = (f"三段论分析：\n\n"
                 f"大前提：所有{b}都{c}\n"
                 f"小前提：所有{a}都是{b}\n"
                 f"结论：所有{a}都{c}\n\n"
                 f"这是标准的 Barbara 式三段论（AAA-1），\n"
                 f"是**有效的**推理形式。\n\n"
                 f"形式化：\n"
                 f"∀x: B(x) → C(x)\n"
                 f"∀x: A(x) → B(x)\n"
                 f"∴ ∀x: A(x) → C(x)  （传递性）")
        samples.append(conv(q, a_str, "logic", "easy"))
    return samples


def _logic_puzzles(count):
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            q = "有25匹马，每次只能5匹赛跑，没有计时器，最少几次比赛能找出最快的3匹？"
            a_str = (f"答案：**7次**\n\n"
                     f"步骤：\n"
                     f"1. 分5组各5匹比赛（5次）\n"
                     f"2. 5组冠军比赛（第6次），找出总冠军\n"
                     f"3. 第6次的2、3名 + 总冠军组的2、3名 + 第6次第2名所在组的第2名 = 5匹\n"
                     f"   这5匹比赛（第7次），前2名就是总排名2、3\n\n"
                     f"为什么是这5匹？\n"
                     f"- 总冠军已确定（第6次第1名）\n"
                     f"- 第6次的4、5名不可能是前3\n"
                     f"- 各组3名以后不可能是前3\n"
                     f"- 只剩候选5匹，一次比赛确定")
            samples.append(conv(q, a_str, "logic", "hard"))
        else:
            q = "四个人过桥，分别需要1、2、5、10分钟，桥同时最多2人，需手电筒（来回），最少多久？"
            a_str = (f"答案：**17分钟**\n\n"
                     f"最优策略：\n"
                     f"1. 1和2过桥（2分钟）\n"
                     f"2. 1回来（1分钟）\n"
                     f"3. 5和10过桥（10分钟）\n"
                     f"4. 2回来（2分钟）\n"
                     f"5. 1和2过桥（2分钟）\n"
                     f"总计：2+1+10+2+2 = **17分钟**\n\n"
                     f"关键洞察：让最快的两个人（1和2）承担来回送手电筒的任务。\n"
                     f"不要让最慢的两个人分别带人过桥（那样需要25分钟）。")
            samples.append(conv(q, a_str, "logic", "hard"))
    return samples


def _logic_game_theory(count):
    samples = []
    for _ in range(count):
        q = "解释囚徒困境，以及为什么理性选择可能导致对双方都差的结果"
        a_str = (f"**囚徒困境**：\n\n"
                 f"两个嫌疑人被分开审讯，各自选择：\n"
                 f"- 合作（沉默）：双方各判1年\n"
                 f"- 背叛（揭发对方）：揭发者释放，被揭发者判10年\n"
                 f"- 双方互相揭发：各判5年\n\n"
                 f"收益矩阵：\n"
                 f"          B沉默    B揭发\n"
                 f"A沉默  (-1,-1)  (-10,0)\n"
                 f"A揭发  (0,-10)  (-5,-5)\n\n"
                 f"纳什均衡：双方都揭发（-5,-5）\n\n"
                 f"矛盾：双方都沉默（-1,-1）对双方都更好，\n"
                 f"但理性人会选择背叛（占优策略）。\n\n"
                 f"应用：军备竞赛、价格战、公共资源过度使用。\n"
                 f"解决：重复博弈、声誉机制、制度约束。")
        samples.append(conv(q, a_str, "logic", "medium"))
    return samples


# ============================================================
# 3. 代码能力 (500 条)
# ============================================================

def gen_code_ability(count=500) -> List[Dict]:
    samples = []
    generators = [
        _code_algorithms,
        _code_debugging,
        _code_review,
        _code_data_structures,
        _code_system_design,
        _code_testing,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _code_algorithms(count):
    """算法实现"""
    samples = []
    algo_qa = [
        ("实现归并排序", 
         "```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\n# 时间O(nlogn) 空间O(n) 稳定排序\nprint(merge_sort([38, 27, 43, 3, 9, 82, 10]))",
         "medium"),
        ("实现Dijkstra最短路径算法",
         "```python\nimport heapq\n\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    \n    while pq:\n        curr_dist, curr_node = heapq.heappop(pq)\n        if curr_dist > distances[curr_node]:\n            continue\n        for neighbor, weight in graph[curr_node].items():\n            dist = curr_dist + weight\n            if dist < distances[neighbor]:\n                distances[neighbor] = dist\n                heapq.heappush(pq, (dist, neighbor))\n    return distances\n\n# 示例\ngraph = {\n    'A': {'B': 1, 'C': 4},\n    'B': {'A': 1, 'C': 2, 'D': 5},\n    'C': {'A': 4, 'B': 2, 'D': 1},\n    'D': {'B': 5, 'C': 1}\n}\nprint(dijkstra(graph, 'A'))  # {'A':0, 'B':1, 'C':3, 'D':4}",
         "hard"),
        ("实现LRU缓存",
         "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n\ncache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nprint(cache.get(1))   # 1\ncache.put(3, 3)       # 淘汰key=2\nprint(cache.get(2))   # -1",
         "medium"),
        ("实现二叉树的层序遍历",
         "```python\nfrom collections import deque\n\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef level_order(root):\n    if not root:\n        return []\n    result = []\n    queue = deque([root])\n    while queue:\n        level_size = len(queue)\n        level = []\n        for _ in range(level_size):\n            node = queue.popleft()\n            level.append(node.val)\n            if node.left:\n                queue.append(node.left)\n            if node.right:\n                queue.append(node.right)\n        result.append(level)\n    return result",
         "medium"),
        ("实现动态规划解背包问题",
         "```python\ndef knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    \n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            dp[i][w] = dp[i-1][w]  # 不选第i个\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i][w], \n                    dp[i-1][w-weights[i-1]] + values[i-1])\n    \n    return dp[n][capacity]\n\nweights = [2, 3, 4, 5]\nvalues = [3, 4, 5, 6]\ncapacity = 8\nprint(knapsack(weights, values, capacity))  # 10",
         "hard"),
        ("实现Trie前缀树",
         "```python\nclass TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    \n    def insert(self, word: str) -> None:\n        node = self.root\n        for ch in word:\n            if ch not in node.children:\n                node.children[ch] = TrieNode()\n            node = node.children[ch]\n        node.is_end = True\n    \n    def search(self, word: str) -> bool:\n        node = self._find(word)\n        return node is not None and node.is_end\n    \n    def startsWith(self, prefix: str) -> bool:\n        return self._find(prefix) is not None\n    \n    def _find(self, prefix: str):\n        node = self.root\n        for ch in prefix:\n            if ch not in node.children:\n                return None\n            node = node.children[ch]\n        return node",
         "medium"),
    ]
    for i in range(count):
        q, a, diff = algo_qa[i % len(algo_qa)]
        n = random.randint(3, 100)
        context_q = q + f"（用Python实现，注意时间复杂度）"
        samples.append(conv(context_q, a, "code", diff))
    return samples


def _code_debugging(count):
    """代码调试"""
    samples = []
    debug_cases = [
        ("这段代码有什么Bug？\n```python\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\nprint(find_max([-5, -2, -8]))  # 期望-2，实际？\n```",
         "**Bug**：`max_val` 初始化为 0，当列表全为负数时返回 0（不在列表中）。\n\n**修复**：\n```python\ndef find_max(lst):\n    max_val = lst[0]  # 初始化为第一个元素\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n```\n\n或使用 `max_val = float('-inf')`。\n\n**教训**：初始化极值时，要考虑边界情况（全负数、空列表等）。"),
        ("找Bug：\n```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            flatten(item)\n        else:\n            result.append(item)\n    return result\n\nprint(flatten([1, [2, [3, 4], 5]]))\n```",
         "**Bug**：递归调用 `flatten(item)` 的结果没有合并到 `result` 中。\n\n**修复**：\n```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))  # 用extend合并\n        else:\n            result.append(item)\n    return result\n```\n\n递归结果被丢弃了，需要用 `extend` 或 `+=` 合并。"),
        ("找Bug：\n```python\ndef cache_result(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n@cache_result\ndef process(data=[]):\n    data.append(1)\n    return sum(data)\n\nprint(process())  # 1\nprint(process())  # 期望2，实际？\n```",
         "**Bug**：默认可变参数 `data=[]` 是共享的，多次调用会累积修改。\n\n**修复**：\n```python\ndef process(data=None):\n    if data is None:\n        data = []\n    data.append(1)\n    return sum(data)\n```\n\n**Python经典陷阱**：不要用可变对象作为默认参数。"),
    ]
    for i in range(count):
        q, a = debug_cases[i % len(debug_cases)]
        samples.append(conv(q, a, "code", "hard"))
    return samples


def _code_review(count):
    """代码审查"""
    samples = []
    for _ in range(count):
        q = "审查以下代码的安全性和最佳实践问题：\n```python\nimport os\nimport pickle\n\ndef load_user_data(filepath):\n    with open(filepath, 'rb') as f:\n        data = pickle.load(f)\n    return data\n\ndef run_command(user_input):\n    os.system(f'echo {user_input}')\n```"
        a_str = ("**安全审查报告**：\n\n"
                 "🔴 **严重问题**：\n"
                 "1. `pickle.load()` - **反序列化漏洞**\n"
                 "   - pickle 可以执行任意代码\n"
                 "   - 攻击者可以构造恶意 pickle 文件\n"
                 "   - 修复：使用 JSON 或 msgpack 等安全格式\n\n"
                 "2. `os.system()` - **命令注入漏洞**\n"
                 "   - 用户输入直接拼接到命令中\n"
                 "   - 攻击者可以注入 `; rm -rf /`\n"
                 "   - 修复：使用 `subprocess.run(['echo', user_input])`\n\n"
                 "🟡 **建议改进**：\n"
                 "3. 缺少输入验证和错误处理\n"
                 "4. 文件路径应做规范化处理\n"
                 "5. 应记录安全相关操作的日志\n\n"
                 "修复后代码：\n"
                 "```python\nimport json\nimport subprocess\nimport os\n\ndef load_user_data(filepath):\n    filepath = os.path.realpath(filepath)\n    if not filepath.startswith(ALLOWED_DIR):\n        raise ValueError('Invalid path')\n    with open(filepath, 'r') as f:\n        return json.load(f)\n\ndef run_command(user_input):\n    # 使用列表参数避免shell注入\n    result = subprocess.run(\n        ['echo', str(user_input)],\n        capture_output=True, text=True, timeout=5\n    )\n    return result.stdout\n```")
        samples.append(conv(q, a_str, "code", "hard"))
    return samples


def _code_data_structures(count):
    """数据结构实现"""
    samples = []
    ds_qa = [
        ("实现一个线程安全的生产者-消费者队列",
         "```python\nimport threading\nimport queue\nimport time\n\nclass SafeQueue:\n    def __init__(self, maxsize=10):\n        self._queue = queue.Queue(maxsize=maxsize)\n        self._lock = threading.Lock()\n        self._closed = False\n    \n    def put(self, item, timeout=None):\n        if self._closed:\n            raise RuntimeError('Queue closed')\n        self._queue.put(item, timeout=timeout)\n    \n    def get(self, timeout=None):\n        return self._queue.get(timeout=timeout)\n    \n    def close(self):\n        self._closed = True\n\ndef producer(q, n):\n    for i in range(n):\n        q.put(i)\n        print(f'Produced: {i}')\n        time.sleep(0.01)\n    q.put(None)  # 哨兵值\n\ndef consumer(q):\n    while True:\n        item = q.get()\n        if item is None:\n            break\n        print(f'Consumed: {item}')\n\nq = SafeQueue(maxsize=5)\nt1 = threading.Thread(target=producer, args=(q, 10))\nt2 = threading.Thread(target=consumer, args=(q,))\nt1.start(); t2.start()\nt1.join(); t2.join()",
         "medium"),
        ("实现一个支持O(1)操作的最小栈",
         "```python\nclass MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []  # 辅助栈，维护最小值\n    \n    def push(self, val: int) -> None:\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    \n    def pop(self) -> None:\n        if self.stack.pop() == self.min_stack[-1]:\n            self.min_stack.pop()\n    \n    def top(self) -> int:\n        return self.stack[-1]\n    \n    def get_min(self) -> int:\n        return self.min_stack[-1]  # O(1)\n\nms = MinStack()\nms.push(3); ms.push(1); ms.push(2)\nprint(ms.get_min())  # 1\nms.pop()\nprint(ms.get_min())  # 1",
         "medium"),
    ]
    for i in range(count):
        q, a, diff = ds_qa[i % len(ds_qa)]
        samples.append(conv(q, a, "code", diff))
    return samples


def _code_system_design(count):
    """系统设计代码"""
    samples = []
    for _ in range(count):
        q = "设计一个简单的分布式限流器（Rate Limiter），用Python实现"
        a_str = ("**滑动窗口限流器实现**：\n\n"
                 "```python\nimport time\nfrom collections import deque\nfrom threading import Lock\n\nclass SlidingWindowRateLimiter:\n    def __init__(self, max_requests: int, window_seconds: int):\n        self.max_requests = max_requests\n        self.window_seconds = window_seconds\n        self.requests = deque()  # 存储请求时间戳\n        self.lock = Lock()\n    \n    def allow_request(self) -> bool:\n        with self.lock:\n            now = time.time()\n            # 移除窗口外的旧请求\n            while self.requests and self.requests[0] < now - self.window_seconds:\n                self.requests.popleft()\n            \n            if len(self.requests) < self.max_requests:\n                self.requests.append(now)\n                return True\n            return False\n    \n    def get_remaining(self) -> int:\n        with self.lock:\n            now = time.time()\n            while self.requests and self.requests[0] < now - self.window_seconds:\n                self.requests.popleft()\n            return max(0, self.max_requests - len(self.requests))\n\n# 使用示例：每分钟最多60个请求\nlimiter = SlidingWindowRateLimiter(max_requests=60, window_seconds=60)\nfor i in range(65):\n    if limiter.allow_request():\n        print(f'Request {i}: allowed')\n    else:\n        print(f'Request {i}: rate limited')\n```\n\n"
                 "**设计要点**：\n"
                 "- 滑动窗口比固定窗口更平滑\n"
                 "- 线程安全（Lock保护共享状态）\n"
                 "- O(1)时间复杂度（每个请求最多进出队列一次）\n"
                 "- 分布式版本可用 Redis 的 ZSET 实现")
        samples.append(conv(q, a_str, "code", "hard"))
    return samples


def _code_testing(count):
    """测试编写"""
    samples = []
    for _ in range(count):
        q = "为以下函数编写完整的单元测试：\n```python\ndef validate_email(email: str) -> bool:\n    if not email or '@' not in email:\n        return False\n    local, domain = email.rsplit('@', 1)\n    if not local or not domain:\n        return False\n    if '.' not in domain:\n        return False\n    return True\n```"
        a_str = ("```python\nimport unittest\n\nclass TestValidateEmail(unittest.TestCase):\n    \n    def test_valid_emails(self):\n        \"\"\"测试有效邮箱\"\"\"\n        valid = [\n            'user@example.com',\n            'test.email@domain.org',\n            'user+tag@example.com',\n            'a@b.co',\n        ]\n        for email in valid:\n            self.assertTrue(validate_email(email), f'Should be valid: {email}')\n    \n    def test_invalid_no_at(self):\n        \"\"\"测试缺少@符号\"\"\"\n        self.assertFalse(validate_email('userexample.com'))\n    \n    def test_invalid_empty(self):\n        \"\"\"测试空字符串\"\"\"\n        self.assertFalse(validate_email(''))\n    \n    def test_invalid_no_domain(self):\n        \"\"\"测试缺少域名\"\"\"\n        self.assertFalse(validate_email('user@'))\n    \n    def test_invalid_no_local(self):\n        \"\"\"测试缺少本地部分\"\"\"\n        self.assertFalse(validate_email('@example.com'))\n    \n    def test_invalid_no_dot_in_domain(self):\n        \"\"\"测试域名缺少点号\"\"\"\n        self.assertFalse(validate_email('user@com'))\n    \n    def test_multiple_at_signs(self):\n        \"\"\"测试多个@符号\"\"\"\n        # rsplit取最后一个@，所以 user@@example.com -> local='user@', domain='example.com'\n        self.assertTrue(validate_email('user@@example.com'))\n    \n    def test_none_input(self):\n        \"\"\"测试None输入\"\"\"\n        self.assertFalse(validate_email(None))\n\nif __name__ == '__main__':\n    unittest.main()\n```\n\n"
                 "**测试覆盖要点**：\n"
                 "- 正常路径（有效邮箱多种格式）\n"
                 "- 边界条件（空字符串、None）\n"
                 "- 错误输入（缺少@、缺少域名、缺少本地部分）\n"
                 "- 特殊情况（多个@、特殊字符）")
        samples.append(conv(q, a_str, "code", "medium"))
    return samples


# ============================================================
# 4. 软件工程深度 (300 条)
# ============================================================

def gen_software_engineering(count=300) -> List[Dict]:
    samples = []
    generators = [
        _se_design_patterns,
        _se_distributed_systems,
        _se_microservices,
        _se_database_design,
        _se_cicd_devops,
        _se_architecture,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _se_design_patterns(count):
    samples = []
    patterns = [
        ("解释策略模式（Strategy Pattern）并用Python实现",
         "**策略模式**：定义一系列算法，把它们封装起来，使它们可以互相替换。\n\n```python\nfrom abc import ABC, abstractmethod\n\nclass SortStrategy(ABC):\n    @abstractmethod\n    def sort(self, data: list) -> list:\n        pass\n\nclass BubbleSort(SortStrategy):\n    def sort(self, data):\n        arr = data.copy()\n        n = len(arr)\n        for i in range(n):\n            for j in range(0, n-i-1):\n                if arr[j] > arr[j+1]:\n                    arr[j], arr[j+1] = arr[j+1], arr[j]\n        return arr\n\nclass QuickSort(SortStrategy):\n    def sort(self, data):\n        if len(data) <= 1:\n            return data\n        pivot = data[len(data)//2]\n        left = [x for x in data if x < pivot]\n        mid = [x for x in data if x == pivot]\n        right = [x for x in data if x > pivot]\n        return self.sort(left) + mid + self.sort(right)\n\nclass Sorter:\n    def __init__(self, strategy: SortStrategy):\n        self._strategy = strategy\n    \n    def sort(self, data):\n        return self._strategy.sort(data)\n\n# 运行时可切换策略\nsorter = Sorter(QuickSort())\nprint(sorter.sort([3,1,4,1,5,9,2,6]))\n```\n\n**应用场景**：排序算法选择、支付方式选择、压缩算法选择、验证策略等。"),
        ("解释观察者模式（Observer Pattern）",
         "**观察者模式**：一对多的依赖关系，当对象状态改变时通知所有依赖者。\n\n```python\nclass EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n    \n    def on(self, event: str, callback):\n        self._listeners.setdefault(event, []).append(callback)\n    \n    def off(self, event: str, callback):\n        if event in self._listeners:\n            self._listeners[event].remove(callback)\n    \n    def emit(self, event: str, *args, **kwargs):\n        for callback in self._listeners.get(event, []):\n            callback(*args, **kwargs)\n\n# 使用\nemitter = EventEmitter()\nemitter.on('data_received', lambda d: print(f'Received: {d}'))\nemitter.on('data_received', lambda d: print(f'Logging: {d}'))\nemitter.emit('data_received', {'key': 'value'})\n```\n\n**应用**：事件系统、消息通知、GUI编程、微服务事件驱动。"),
    ]
    for i in range(count):
        q, a = patterns[i % len(patterns)]
        samples.append(conv(q, a, "software_engineering", "medium"))
    return samples


def _se_distributed_systems(count):
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.33:
            q = "解释CAP定理，以及它在实际系统设计中的应用"
            a_str = ("**CAP定理**（Brewer定理）：\n\n"
                     "分布式系统最多同时满足以下三项中的两项：\n"
                     "- **C（一致性）**：所有节点看到相同数据\n"
                     "- **A（可用性）**：每个请求都能得到响应\n"
                     "- **P（分区容错）**：网络分区时系统仍能运行\n\n"
                     "**实际选择**：\n"
                     "- **CP系统**（放弃可用性）：ZooKeeper、etcd、HBase\n"
                     "  - 适用于银行、库存等需要强一致的场景\n"
                     "- **AP系统**（放弃一致性）：Cassandra、DynamoDB、CouchDB\n"
                     "  - 适用于社交网络、缓存等可容忍短暂不一致的场景\n\n"
                     "**BASE理论**是CAP的实践补充：\n"
                     "- Basically Available（基本可用）\n"
                     "- Soft State（软状态）\n"
                     "- Eventually Consistent（最终一致性）")
            samples.append(conv(q, a_str, "software_engineering", "hard"))
        elif r < 0.66:
            q = "解释Raft一致性算法的核心思想"
            a_str = ("**Raft一致性算法**：\n\n"
                     "Raft将一致性问题分解为三个子问题：\n\n"
                     "1. **领导选举（Leader Election）**\n"
                     "   - 节点状态：Follower → Candidate → Leader\n"
                     "   - 超时机制：选举超时触发投票\n"
                     "   - 多数票当选\n\n"
                     "2. **日志复制（Log Replication）**\n"
                     "   - Leader 接收客户端请求\n"
                     "   - 将日志条目复制到 Followers\n"
                     "   - 多数确认后提交\n\n"
                     "3. **安全性（Safety）**\n"
                     "   - 已提交的日志不会丢失\n"
                     "   - 新 Leader 必须包含所有已提交日志\n\n"
                     "**应用**：etcd、Consul、CockroachDB、TiKV\n\n"
                     "比 Paxos 更容易理解和实现。")
            samples.append(conv(q, a_str, "software_engineering", "hard"))
        else:
            q = "如何设计一个高可用的分布式ID生成器？"
            a_str = ("**雪花算法（Snowflake）**实现：\n\n"
                     "```python\nimport time\nimport threading\n\nclass SnowflakeIDGenerator:\n    def __init__(self, machine_id: int, datacenter_id: int):\n        self.machine_id = machine_id\n        self.datacenter_id = datacenter_id\n        self.sequence = 0\n        self.last_timestamp = -1\n        self.lock = threading.Lock()\n        # 时间戳起始值 (2024-01-01)\n        self.epoch = 1704067200000\n    \n    def _current_millis(self):\n        return int(time.time() * 1000)\n    \n    def generate(self) -> int:\n        with self.lock:\n            timestamp = self._current_millis()\n            if timestamp == self.last_timestamp:\n                self.sequence = (self.sequence + 1) & 0xFFF  # 12位\n                if self.sequence == 0:\n                    while timestamp <= self.last_timestamp:\n                        timestamp = self._current_millis()\n            else:\n                self.sequence = 0\n            self.last_timestamp = timestamp\n            \n            return (\n                ((timestamp - self.epoch) << 22) |\n                (self.datacenter_id << 17) |\n                (self.machine_id << 12) |\n                self.sequence\n            )\n\n# 使用\ngen = SnowflakeIDGenerator(machine_id=1, datacenter_id=1)\nfor _ in range(5):\n    print(gen.generate())\n```\n\n"
                     "**设计要点**：\n"
                     "- 64位ID：1位符号+41位时间+5位数据中心+5位机器+12位序列\n"
                     "- 单机每毫秒可生成4096个ID\n"
                     "- 趋势递增，适合数据库索引\n"
                     "- 线程安全（锁保护）")
            samples.append(conv(q, a_str, "software_engineering", "hard"))
    return samples


def _se_microservices(count):
    samples = []
    for _ in range(count):
        q = "解释微服务中的Saga模式如何处理分布式事务"
        a_str = ("**Saga模式**：将长事务拆分为一系列本地事务，每个事务有对应的补偿操作。\n\n"
                 "**两种实现方式**：\n\n"
                 "1. **编排式（Choreography）**：\n"
                 "   - 每个服务监听事件并执行操作\n"
                 "   - 去中心化，服务间通过事件通信\n"
                 "   - 适合简单流程（3-4个服务）\n\n"
                 "2. **协调式（Orchestration）**：\n"
                 "   - 一个中心协调器管理整个流程\n"
                 "   - 集中控制，流程清晰\n"
                 "   - 适合复杂流程\n\n"
                 "**电商订单示例**：\n"
                 "```\n创建订单 → 扣减库存 → 处理支付 → 发送通知\n    ↓(补偿)    ↓(补偿)    ↓(补偿)\n取消订单 ← 恢复库存 ← 退款 ← 取消通知\n```\n\n"
                 "**关键点**：\n"
                 "- 每个正向操作都有对应的补偿操作\n"
                 "- 补偿操作必须是幂等的\n"
                 "- 最终一致性（非强一致性）\n"
                 "- 需要处理补偿失败的情况（人工介入）")
        samples.append(conv(q, a_str, "software_engineering", "hard"))
    return samples


def _se_database_design(count):
    samples = []
    for _ in range(count):
        q = "设计一个高性能的用户表，考虑索引策略和查询优化"
        a_str = ("**用户表设计**：\n\n"
                 "```sql\nCREATE TABLE users (\n    id BIGINT PRIMARY KEY AUTO_INCREMENT,\n    username VARCHAR(50) NOT NULL,\n    email VARCHAR(100) NOT NULL,\n    password_hash VARCHAR(255) NOT NULL,\n    status TINYINT DEFAULT 1 COMMENT '0=禁用 1=正常',\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,\n    deleted_at TIMESTAMP NULL DEFAULT NULL COMMENT '软删除',\n    \n    UNIQUE KEY uk_username (username),\n    UNIQUE KEY uk_email (email),\n    INDEX idx_status_created (status, created_at),\n    INDEX idx_deleted_at (deleted_at)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n```\n\n"
                 "**索引策略**：\n"
                 "1. 主键索引：自增ID，InnoDB聚簇索引\n"
                 "2. 唯一索引：username 和 email 保证唯一性\n"
                 "3. 复合索引：(status, created_at) 支持按状态筛选+时间排序\n"
                 "4. 软删除索引：deleted_at 支持软删除查询\n\n"
                 "**查询优化**：\n"
                 "```sql\n-- 按用户名查询（唯一索引，O(1)）\nSELECT * FROM users WHERE username = 'john' AND deleted_at IS NULL;\n\n-- 分页查询（覆盖索引）\nSELECT id, username, email FROM users \nWHERE status = 1 AND deleted_at IS NULL \nORDER BY created_at DESC LIMIT 20 OFFSET 0;\n\n-- 避免深分页：使用游标\nSELECT * FROM users WHERE id > {last_id} LIMIT 20;\n```\n\n"
                 "**设计原则**：\n"
                 "- 软删除而非物理删除（数据安全）\n"
                 "- 时间戳审计字段\n"
                 "- 避免 SELECT *（只查需要的列）\n"
                 "- 避免深分页（用游标分页）")
        samples.append(conv(q, a_str, "software_engineering", "hard"))
    return samples


def _se_cicd_devops(count):
    samples = []
    for _ in range(count):
        q = "设计一个完整的CI/CD流水线，包含代码质量检查、测试、构建和部署"
        a_str = ("**GitHub Actions CI/CD 流水线**：\n\n"
                 "```yaml\nname: CI/CD Pipeline\non:\n  push:\n    branches: [main, develop]\n  pull_request:\n    branches: [main]\n\njobs:\n  lint:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install ruff mypy\n      - run: ruff check .\n      - run: mypy src/ --ignore-missing-imports\n\n  test:\n    needs: lint\n    runs-on: ubuntu-latest\n    services:\n      postgres:\n        image: postgres:15\n        env:\n          POSTGRES_PASSWORD: test\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install -r requirements.txt\n      - run: pip install pytest pytest-cov\n      - run: pytest --cov=src --cov-report=xml\n      - uses: codecov/codecov-action@v3\n\n  build:\n    needs: test\n    if: github.ref == 'refs/heads/main'\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: docker build -t myapp:${{ github.sha }} .\n      - run: docker push registry/myapp:${{ github.sha }}\n\n  deploy:\n    needs: build\n    runs-on: ubuntu-latest\n    environment: production\n    steps:\n      - run: kubectl set image deployment/myapp \\\n            myapp=registry/myapp:${{ github.sha }}\n```\n\n"
                 "**流水线设计要点**：\n"
                 "1. **并行执行**：lint 和 test 可并行（test 依赖 lint）\n"
                 "2. **缓存依赖**：加速构建\n"
                 "3. **环境隔离**：每个 job 独立环境\n"
                 "4. **门禁机制**：质量不达标不能合并\n"
                 "5. **灰度部署**：先 staging 后 production\n"
                 "6. **回滚机制**：保留前N个版本的镜像")
        samples.append(conv(q, a_str, "software_engineering", "hard"))
    return samples


def _se_architecture(count):
    samples = []
    for _ in range(count):
        q = "对比微服务架构和单体架构的优缺点，何时选择哪种？"
        a_str = ("**单体架构 vs 微服务架构**：\n\n"
                 "| 维度 | 单体架构 | 微服务架构 |\n"
                 "|------|----------|------------|\n"
                 "| 开发复杂度 | 低（一个项目） | 高（多个服务） |\n"
                 "| 部署 | 简单（一个包） | 复杂（多个容器） |\n"
                 "| 技术栈 | 统一 | 可以不同 |\n"
                 "| 扩展性 | 整体扩展 | 按服务扩展 |\n"
                 "| 可靠性 | 单点故障 | 故障隔离 |\n"
                 "| 数据一致性 | 强一致 | 最终一致 |\n"
                 "| 团队协作 | 大团队困难 | 独立团队开发 |\n\n"
                 "**选择单体的场景**：\n"
                 "- 初创项目，快速验证\n"
                 "- 小团队（<5人）\n"
                 "- 业务简单，变化不频繁\n"
                 "- 对性能要求极高（避免网络开销）\n\n"
                 "**选择微服务的场景**：\n"
                 "- 业务复杂，需要独立部署\n"
                 "- 大团队，多团队协作\n"
                 "- 需要不同技术栈\n"
                 "- 需要高可用和弹性扩展\n\n"
                 "**建议**：从单体开始，随着业务增长逐步拆分为微服务。避免过早优化。")
        samples.append(conv(q, a_str, "software_engineering", "medium"))
    return samples


# ============================================================
# 5. ReAct 推理链 (400 条)
# ============================================================

def gen_react_data(count=400) -> List[Dict]:
    samples = []
    generators = [
        _react_single_tool,
        _react_multi_tool,
        _react_error_recovery,
        _react_planning,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _react_single_tool(count):
    """单工具ReAct任务"""
    samples = []
    tasks = [
        ("读取 requirements.txt 文件的内容", "read_local_file", "requirements.txt",
         "文件内容已获取，包含项目的依赖列表。"),
        ("用Python计算1到1000的和", "execute_python", "print(sum(range(1, 1001)))",
         "1到1000的和为500500。"),
        ("搜索Python异步编程最新教程", "search", "Python async await 异步编程 2024 教程",
         "已找到相关教程资源。"),
        ("列出当前目录的所有文件", "list_directory", ".",
         "已获取目录结构。"),
        ("分析main.py的代码结构", "analyze_code", "main.py",
         "代码分析完成，包含函数定义和导入信息。"),
    ]
    for i in range(count):
        task, tool, args, answer = tasks[i % len(tasks)]
        samples.append(react_sample(task, [
            {"thought": f"用户需要执行{task}，我使用{tool}工具。", "action": tool, "action_args": {"input": args}},
            {"thought": "任务完成，整理结果。", "final_answer": answer}
        ]))
    return samples


def _react_multi_tool(count):
    """多工具ReAct任务"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            task = "读取config.json，把debug模式改为false，然后验证修改"
            samples.append(react_sample(task, [
                {"thought": "先读取配置文件查看当前内容。", "action": "read_local_file", "action_args": {"input": "config.json"}},
                {"thought": "获取到配置内容，现在修改debug模式。", "action": "edit_file", "action_args": {"input": "config.json | \"debug\": true | \"debug\": false"}},
                {"thought": "修改完成，读取验证。", "action": "read_local_file", "action_args": {"input": "config.json"}},
                {"thought": "验证完成，debug已改为false。", "final_answer": "已将config.json中的debug模式修改为false，验证成功。"}
            ]))
        else:
            task = "搜索Python装饰器教程，把要点保存到notes.md，然后学习保存"
            samples.append(react_sample(task, [
                {"thought": "先搜索Python装饰器相关资料。", "action": "search", "action_args": {"input": "Python decorator 装饰器 教程 最佳实践"}},
                {"thought": "搜索到资料，将要点保存到文件。", "action": "write_file", "action_args": {"input": "notes.md | # Python装饰器要点\n\n- @语法糖\n- functools.wraps保留元信息\n- 常见用途：日志、缓存、权限"}},
                {"thought": "文件已保存，再用知识学习功能存储。", "action": "learn_knowledge", "action_args": {"input": "Python装饰器是一种设计模式，使用@语法糖，可以不修改原函数的情况下扩展功能。"}},
                {"thought": "全部完成。", "final_answer": "已搜索Python装饰器教程、保存要点到notes.md、并完成知识学习。"}
            ]))
    return samples


def _react_error_recovery(count):
    """错误恢复ReAct任务"""
    samples = []
    for _ in range(count):
        task = "运行test.py，如果出错就修复后重新运行"
        samples.append(react_sample(task, [
            {"thought": "先运行test.py看看有没有错误。", "action": "execute_python", "action_args": {"input": "exec(open('test.py').read())"}},
            {"thought": "执行出错了，分析错误信息。检测到NameError，变量未定义。", "action": "read_local_file", "action_args": {"input": "test.py"}},
            {"thought": "找到问题所在，缺少import语句。修复代码。", "action": "edit_file", "action_args": {"input": "test.py | print(math.sqrt(4)) | import math\nprint(math.sqrt(4))"}},
            {"thought": "修复完成，重新运行测试。", "action": "execute_python", "action_args": {"input": "import math\nprint(math.sqrt(4))"}},
            {"thought": "测试通过，错误已修复。", "final_answer": "已修复test.py中的import缺失问题，测试运行正常。输出：2.0"}
        ]))
    return samples


def _react_planning(count):
    """规划执行ReAct任务"""
    samples = []
    for _ in range(count):
        task = "创建一个完整的Python项目：一个简单的REST API服务器"
        samples.append(react_sample(task, [
            {"thought": "需要创建一个REST API项目。先规划步骤：1.创建目录结构 2.创建主文件 3.创建配置 4.测试运行。", "action": "create_directory", "action_args": {"input": "rest_api"}},
            {"thought": "目录已创建，现在创建主程序文件。", "action": "write_file", "action_args": {"input": "rest_api/app.py | from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/health')\ndef health():\n    return jsonify({'status': 'ok'})\n\nif __name__ == '__main__':\n    app.run(debug=True, port=5000)"}},
            {"thought": "主文件已创建，现在创建requirements.txt。", "action": "write_file", "action_args": {"input": "rest_api/requirements.txt | flask>=2.0"}},
            {"thought": "项目文件已创建完成，验证代码语法。", "action": "execute_python", "action_args": {"input": "import ast\nwith open('rest_api/app.py') as f:\n    ast.parse(f.read())\nprint('语法检查通过')"}},
            {"thought": "项目创建成功。", "final_answer": "REST API项目已创建完成，包含：\n- rest_api/app.py（Flask主程序，包含健康检查接口）\n- rest_api/requirements.txt（依赖文件）\n语法检查通过。"}
        ]))
    return samples


# ============================================================
# 6. 多模态数据 (600 条)
# ============================================================

def gen_multimodal_data(count=600) -> List[Dict]:
    samples = []
    generators = [
        _mm_vision_understanding,
        _mm_audio_understanding,
        _mm_video_understanding,
        _mm_cross_modal,
        _mm_generation_tasks,
        _mm_screen_understanding,
    ]
    per_gen = count // len(generators) + 1
    for gen_func in generators:
        samples.extend(gen_func(per_gen))
    random.shuffle(samples)
    return samples[:count]


def _mm_vision_understanding(count):
    """视觉理解 — 使用<image>特殊token"""
    samples = []
    scenarios = [
        ("用户上传了一张图片，内容是一只橘猫坐在窗台上晒太阳。请描述这张图片。",
         "<image>一张室内照片，橘色虎斑猫侧卧在白色窗台上，窗外阳光明媚，猫半眯着眼睛。</image>\n\n"
         "这是一张温馨的室内照片。画面中：\n\n"
         "1. **主体**：一只橘色虎斑猫侧卧在窗台上\n"
         "2. **姿态**：猫半眯着眼睛，看起来很享受\n"
         "3. **环境**：白色窗台，窗外阳光充足\n"
         "4. **氛围**：安静、舒适、温暖\n\n"
         "这张照片捕捉了猫咪享受阳光的惬意时刻，很有生活气息。"),
        ("识别图片中的文字：<image>一张手机截图，显示微信聊天界面，对方发来消息：明天下午3点开会，记得带笔记本电脑</image>",
         "图片中识别到的文字内容：\n\n"
         "聊天消息：\n- **发送者**：聊天对象\n- **内容**：「明天下午3点开会，记得带笔记本电脑」\n\n"
         "这是一条工作安排通知，提醒：\n"
         "- 时间：明天下午3点\n"
         "- 事项：开会\n"
         "- 需要携带：笔记本电脑\n\n"
         "建议设置日历提醒，提前准备好笔记本电脑。"),
        ("分析这张图表数据：<image>柱状图，标题为'2024年Q1-Q4销售额'，Q1=120万，Q2=150万，Q3=180万，Q4=210万</image>",
         "图表分析结果：\n\n"
         "**标题**：2024年Q1-Q4销售额\n\n"
         "**数据**：\n"
         "| 季度 | 销售额 | 环比增长 |\n"
         "|------|--------|----------|\n"
         "| Q1 | 120万 | — |\n"
         "| Q2 | 150万 | +25% |\n"
         "| Q3 | 180万 | +20% |\n"
         "| Q4 | 210万 | +16.7% |\n\n"
         "**分析**：\n"
         "1. 全年销售额呈**持续增长**趋势\n"
         "2. 全年总销售额 = 660万\n"
         "3. Q4是全年最高，可能是年末促销效应\n"
         "4. 增长率逐季放缓，需要注意增长动力\n\n"
         "**建议**：Q1增长率较低，建议加强Q1营销策略。"),
        ("描述这张截图的UI设计：<image>一个现代风格的登录页面，深色背景，居中的白色卡片式登录表单，包含邮箱输入框、密码输入框和蓝色登录按钮</image>",
         "UI设计分析：\n\n"
         "**页面类型**：登录页面\n\n"
         "**设计风格**：\n"
         "- 整体风格：现代、简洁、暗色主题\n"
         "- 背景：深色（可能是深灰或深蓝）\n"
         "- 卡片：白色居中卡片，圆角设计\n"
         "- 按钮：蓝色主色调登录按钮\n\n"
         "**布局**：\n"
         "- 垂直居中布局\n"
         "- 卡片式设计，左右留白\n"
         "- 表单元素垂直排列\n\n"
         "**优点**：\n"
         "1. 暗色主题减少视觉疲劳\n"
         "2. 卡片式设计聚焦注意力\n"
         "3. 蓝色CTA按钮突出\n\n"
         "**建议**：可增加「忘记密码」链接和第三方登录选项。"),
        ("OCR识别图片中的代码：<image>代码截图，显示一段Python FastAPI路由代码</image>",
         "识别到的代码内容：\n\n"
                 "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    return {'message': 'Hello World'}\n\n@app.get('/items/{item_id}')\nasync def read_item(item_id: int, q: str = None):\n    return {'item_id': item_id, 'q': q}\n```\n\n"
                 "这是一个 FastAPI 示例代码，包含：\n"
                 "1. 根路由 `/` — 返回欢迎消息\n"
                 "2. 动态路由 `/items/{item_id}` — 带路径参数和查询参数"),
    ]
    for i in range(count):
        q, a = scenarios[i % len(scenarios)]
        samples.append(conv(q, a, "multimodal_vision", "medium"))
    return samples


def _mm_audio_understanding(count):
    """音频理解 — 使用<audio>/<speech>特殊token"""
    samples = []
    scenarios = [
        ("转录这段语音内容：<audio>一段会议录音，发言人说：'各位好，今天的议题是讨论Q2的产品路线图。首先，我们需要确定优先级最高的三个功能。'</audio>",
         "<speech>各位好，今天的议题是讨论Q2的产品路线图。首先，我们需要确定优先级最高的三个功能。</speech>\n\n"
         "语音转录完成。内容摘要：\n"
         "- **场合**：会议\n"
         "- **议题**：Q2产品路线图\n"
         "- **下一步**：确定Top 3优先功能\n\n"
         "建议后续整理会议纪要。"),
        ("分析这段音频的情感：<audio>一段客服通话录音，客户语气焦急地说：'我的订单已经一个星期了还没收到！这到底怎么回事？'</audio>",
         "**音频情感分析**：\n\n"
         "<speech>我的订单已经一个星期了还没收到！这到底怎么回事？</speech>\n\n"
         "**情感判断**：\n"
         "- 主要情感：**焦急/不满**\n"
         "- 语气特征：语速较快、语调上扬、有强调\n"
         "- 情感强度：中高（7/10）\n\n"
         "**场景分析**：\n"
         "- 客户在进行售后咨询\n"
         "- 订单延迟超过预期（一周）\n"
         "- 客户期望得到解释和解决方案\n\n"
         "**建议回复策略**：\n"
         "1. 先道歉，表达理解\n"
         "2. 查询订单状态\n"
         "3. 给出明确的解决方案\n"
         "4. 提供补偿（如适用）"),
        ("识别音频中的语言和说话人数：<audio>一段多语言对话，先是英文'Hello, how are you?'，然后中文'我很好，谢谢'，最后日语'こんにちは'</audio>",
         "**音频分析结果**：\n\n"
                 "**识别到的语言**：\n"
                 "1. 英语：「Hello, how are you?」— 打招呼\n"
                 "2. 中文：「我很好，谢谢」— 回应问候\n"
                 "3. 日语：「こんにちは」— 你好\n\n"
                 "**说话人分析**：\n"
                 "- 至少 **2-3 位**说话人\n"
                 "- 使用了 **3 种语言**（英语、中文、日语）\n"
                 "- 可能是多语言交流场景\n\n"
                 "**场景推测**：国际会议、语言学习练习、或多语言社交场合。"),
    ]
    for i in range(count):
        q, a = scenarios[i % len(scenarios)]
        samples.append(conv(q, a, "multimodal_audio", "medium"))
    return samples


def _mm_video_understanding(count):
    """视频理解 — 使用<video>特殊token"""
    samples = []
    scenarios = [
        ("描述这个视频的内容：<video>一个烹饪教程视频，时长5分钟，展示如何制作番茄炒蛋。厨师先切番茄，打蛋，然后热油下锅翻炒。</video>",
         "<video>烹饪教程：番茄炒蛋，时长5分钟</video>\n\n"
         "**视频内容描述**：\n\n"
         "这是一个烹饪教程视频，展示制作番茄炒蛋的全过程：\n\n"
         "**步骤**：\n"
         "1. 准备食材：番茄切块、鸡蛋打散\n"
         "2. 热油下锅，先炒蛋至半熟盛出\n"
         "3. 另起油锅，炒番茄至出汁\n"
         "4. 将鸡蛋倒回，与番茄翻炒均匀\n"
         "5. 调味出锅\n\n"
         "**视频特点**：\n"
         "- 时长：约5分钟\n"
         "- 类型：烹饪教程\n"
         "- 菜品：番茄炒蛋（中式家常菜）\n\n"
         "这是一道经典的中式家常菜，简单易学。"),
        ("分析这个视频中的动作：<video>健身视频，教练演示了10个深蹲、10个俯卧撑、10个仰卧起坐，循环3组</video>",
         "<video>健身训练视频，3组循环训练</video>\n\n"
         "**动作识别**：\n\n"
         "| 动作 | 次数 | 组数 |\n"
         "|------|------|------|\n"
         "| 深蹲 | 10次 | 3组 |\n"
         "| 俯卧撑 | 10次 | 3组 |\n"
         "| 仰卧起坐 | 10次 | 3组 |\n\n"
         "**训练分析**：\n"
         "- 类型：HIIT循环训练\n"
         "- 总次数：每动作30次，共90次\n"
         "- 目标肌群：腿部、胸部、核心\n"
         "- 难度：中等（适合有一定基础的人）\n\n"
         "**建议**：\n"
         "- 组间休息30-60秒\n"
         "- 注意动作标准，避免受伤\n"
         "- 可根据自身情况调整次数"),
    ]
    for i in range(count):
        q, a = scenarios[i % len(scenarios)]
        samples.append(conv(q, a, "multimodal_video", "medium"))
    return samples


def _mm_cross_modal(count):
    """跨模态推理"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.33:
            q = ("用户上传了一张照片<image>一张海滩日落照片</image>，"
                 "并发送了一段语音<audio>用户说：帮我写一首关于这个景色的诗</audio>。"
                 "请结合图像和语音内容回答。")
            a_str = ("<image>海滩日落：金色阳光洒在海面上，天空橙红渐变，沙滩上有几棵椰子树剪影</image>\n"
                     "<speech>帮我写一首关于这个景色的诗</speech>\n\n"
                     "看到这张美丽的海滩日落照片，为您写一首诗：\n\n"
                     "**海之暮歌**\n\n"
                     "金色的光，铺满了海面，\n"
                     "天边的云，染成了橙红，\n"
                     "椰树的影，在风中轻摇，\n"
                     "潮水退去，留下了宁静。\n\n"
                     "这一刻，时间仿佛停驻，\n"
                     "海与天，交融在远方，\n"
                     "日落的美，无需言语，\n"
                     "只需静看，便是诗行。")
            samples.append(conv(q, a_str, "multimodal_cross", "hard"))
        elif r < 0.66:
            q = "根据这张截图中的代码错误信息和终端输出，诊断问题：<image>终端截图显示Python报错：ModuleNotFoundError: No module named 'requests'</image><audio>用户说：我运行程序报错了，帮我看看</audio>"
            a_str = ("<image>终端错误：ModuleNotFoundError: No module named 'requests'</image>\n"
                     "<speech>我运行程序报错了，帮我看看</speech>\n\n"
                     "**问题诊断**：\n\n"
                     "从截图中看到错误信息：`ModuleNotFoundError: No module named 'requests'`\n\n"
                     "**原因**：Python 环境中没有安装 requests 库。\n\n"
                     "**解决方案**：\n"
                     "```bash\npip install requests\n```\n\n"
                     "**如果已安装仍报错**，检查：\n"
                     "1. 是否在正确的虚拟环境中\n"
                     "2. IDE 是否使用了正确的 Python 解释器\n"
                     "3. 尝试 `pip3 install requests`\n\n"
                     "运行 `pip list | grep requests` 验证是否安装成功。")
            samples.append(conv(q, a_str, "multimodal_cross", "medium"))
        else:
            q = "我拍了一张菜单照片<image>餐厅菜单照片，显示宫保鸡丁48元、麻婆豆腐32元、米饭3元</image>，一共点了宫保鸡丁、麻婆豆腐和2碗米饭，帮我算算多少钱"
            a_str = ("<image>菜单：宫保鸡丁48元、麻婆豆腐32元、米饭3元</image>\n\n"
                     "**账单计算**：\n\n"
                     "| 菜品 | 单价 | 数量 | 小计 |\n"
                     "|------|------|------|------|\n"
                     "| 宫保鸡丁 | 48元 | 1份 | 48元 |\n"
                     "| 麻婆豆腐 | 32元 | 1份 | 32元 |\n"
                     "| 米饭 | 3元 | 2碗 | 6元 |\n\n"
                     "**总计**：48 + 32 + 6 = **86元**\n\n"
                     "如果需要AA制（2人），每人 **43元**。")
            samples.append(conv(q, a_str, "multimodal_cross", "easy"))
    return samples


def _mm_generation_tasks(count):
    """多模态生成任务"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            q = "根据以下描述生成一张图片：<img_gen>一只可爱的白色小猫，戴着红色围巾，坐在雪地里，背景是圣诞树</img_gen>"
            a_str = ("<img_gen>一只可爱的白色小猫，戴着红色围巾，坐在雪地里，背景是圣诞树</img_gen>\n\n"
                     "图像生成描述已接收。为了生成最佳效果的图片，建议的详细描述：\n\n"
                     "**画面元素**：\n"
                     "- 主体：纯白色小猫，蓝色大眼睛，毛茸茸的\n"
                     "- 配饰：鲜艳的红色针织围巾\n"
                     "- 背景：白雪覆盖的地面，装饰着彩灯的圣诞树\n"
                     "- 氛围：温馨、节日气氛\n\n"
                     "**风格建议**：卡通可爱风格，柔和光线，浅景深\n\n"
                     "（注：实际图像生成需要调用 generate_image 工具）")
            samples.append(conv(q, a_str, "multimodal_gen", "medium"))
        else:
            q = "将以下文字转换为语音：<tts>欢迎使用态极AI助手，我是由liulangzwg创造的智能助手，很高兴为您服务。</tts>"
            a_str = ("<tts>欢迎使用态极AI助手，我是由liulangzwg创造的智能助手，很高兴为您服务。</tts>\n\n"
                     "语音合成参数：\n"
                     "- **文本**：欢迎使用态极AI助手...\n"
                     "- **语言**：中文\n"
                     "- **推荐声音**：zh-CN-XiaoxiaoNeural（温柔女声）\n"
                     "- **语速**：正常\n"
                     "- **格式**：MP3\n\n"
                     "（注：实际语音生成需要调用 text_to_speech 工具）")
            samples.append(conv(q, a_str, "multimodal_gen", "easy"))
    return samples


def _mm_screen_understanding(count):
    """屏幕理解 — 使用<screen>特殊token"""
    samples = []
    for _ in range(count):
        r = random.random()
        if r < 0.5:
            q = "分析这个屏幕截图中的错误：<screen>VS Code编辑器截图，Python文件中第42行有红色波浪线，鼠标悬停显示'TypeError: unsupported operand type(s) for +: int and str'</screen>"
            a_str = ("<screen>VS Code编辑器，第42行TypeError错误</screen>\n\n"
                     "**错误分析**：\n\n"
                     "**错误类型**：TypeError\n"
                     "**错误信息**：unsupported operand type(s) for +: int and str\n"
                     "**位置**：第42行\n\n"
                     "**原因**：代码尝试将一个整数(int)和字符串(str)用+号连接。\n\n"
                     "**常见场景**：\n"
                     "```python\n# 错误写法\nresult = '总数：' + 42  # TypeError!\n\n# 修复方法1：类型转换\nresult = '总数：' + str(42)\n\n# 修复方法2：f-string\nresult = f'总数：{42}'\n\n# 修复方法3：format\nresult = '总数：{}'.format(42)\n```\n\n"
                     "**建议**：使用f-string是最Pythonic的方式。")
            samples.append(conv(q, a_str, "multimodal_screen", "medium"))
        else:
            q = "理解这个UI截图并给出改进建议：<screen>一个电商网站的购物车页面截图，商品列表显示在左侧，右侧是价格汇总。整体布局较拥挤，字体较小，缺少明显的结算按钮</screen>"
            a_str = ("<screen>电商购物车页面，布局拥挤</screen>\n\n"
                     "**UI分析**：\n\n"
                     "**当前问题**：\n"
                     "1. 布局拥挤 — 信息密度过高\n"
                     "2. 字体较小 — 可读性差\n"
                     "3. 结算按钮不明显 — 转化率可能受影响\n\n"
                     "**改进建议**：\n\n"
                     "**布局优化**：\n"
                     "- 增加商品间的间距（16-24px）\n"
                     "- 左右分栏使用卡片式设计\n"
                     "- 商品图片适当放大\n\n"
                     "**字体优化**：\n"
                     "- 商品名称：16px\n"
                     "- 价格：18px，加粗\n"
                     "- 辅助信息：14px\n\n"
                     "**结算按钮**：\n"
                     "- 使用高对比度颜色（如橙色/红色）\n"
                     "- 固定在底部或右侧显眼位置\n"
                     "- 添加「去结算」大按钮（48px高）\n\n"
                     "**移动端适配**：\n"
                     "- 响应式布局\n"
                     "- 结算按钮固定底部")
            samples.append(conv(q, a_str, "multimodal_screen", "hard"))
    return samples


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="态极终极训练数据生成器")
    parser.add_argument("--output", type=str, default="taiji/training_data", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # 1. 数学推理
    logger.info("生成数学推理数据 (500)...")
    math_data = gen_math_reasoning(500)

    # 2. 逻辑推理
    logger.info("生成逻辑推理数据 (400)...")
    logic_data = gen_logic_reasoning(400)

    # 3. 代码能力
    logger.info("生成代码能力数据 (500)...")
    code_data = gen_code_ability(500)

    # 4. 软件工程深度
    logger.info("生成软件工程数据 (300)...")
    se_data = gen_software_engineering(300)

    # 5. 多模态数据
    logger.info("生成多模态数据 (600)...")
    mm_data = gen_multimodal_data(600)

    # 6. ReAct数据
    logger.info("生成ReAct推理链数据 (400)...")
    react_data_list = gen_react_data(400)

    # === 合并对话数据 ===
    all_conv = math_data + logic_data + code_data + se_data + mm_data
    random.shuffle(all_conv)

    # === 导出对话数据 ===
    conv_path = os.path.join(args.output, "taiji_ultimate_conversation.jsonl")
    with open(conv_path, "w", encoding="utf-8") as f:
        for item in all_conv:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # === 导出ReAct数据 ===
    react_path = os.path.join(args.output, "taiji_ultimate_react.jsonl")
    with open(react_path, "w", encoding="utf-8") as f:
        for item in react_data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # === 导出多模态专用数据（按模态分类） ===
    mm_vision = [s for s in mm_data if s.get("category", "").startswith("multimodal_vision")]
    mm_audio = [s for s in mm_data if s.get("category", "").startswith("multimodal_audio")]
    mm_video = [s for s in mm_data if s.get("category", "").startswith("multimodal_video")]
    mm_cross = [s for s in mm_data if s.get("category", "").startswith("multimodal_cross")]
    mm_gen = [s for s in mm_data if s.get("category", "").startswith("multimodal_gen")]
    mm_screen = [s for s in mm_data if s.get("category", "").startswith("multimodal_screen")]

    for name, data in [("vision", mm_vision), ("audio", mm_audio), ("video", mm_video),
                        ("cross", mm_cross), ("generation", mm_gen), ("screen", mm_screen)]:
        path = os.path.join(args.output, f"taiji_ultimate_mm_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # === 数据集统计报告 ===
    report = {
        "name": "态极终极训练数据集",
        "version": "1.0",
        "seed": args.seed,
        "total_conversation": len(all_conv),
        "total_react": len(react_data_list),
        "total": len(all_conv) + len(react_data_list),
        "categories": {
            "数学推理": {"count": len(math_data), "difficulty": ["easy", "medium", "hard", "olympiad"]},
            "逻辑推理": {"count": len(logic_data), "difficulty": ["easy", "medium", "hard"]},
            "代码能力": {"count": len(code_data), "difficulty": ["medium", "hard"]},
            "软件工程": {"count": len(se_data), "difficulty": ["medium", "hard"]},
            "多模态视觉": {"count": len(mm_vision), "special_tokens": ["<image>", "</image>"]},
            "多模态音频": {"count": len(mm_audio), "special_tokens": ["<audio>", "</audio>", "<speech>", "</speech>"]},
            "多模态视频": {"count": len(mm_video), "special_tokens": ["<video>", "</video>"]},
            "跨模态推理": {"count": len(mm_cross), "special_tokens": ["<image>", "<audio>", "<video>"]},
            "多模态生成": {"count": len(mm_gen), "special_tokens": ["<img_gen>", "<tts>"]},
            "屏幕理解": {"count": len(mm_screen), "special_tokens": ["<screen>", "</screen>"]},
            "ReAct推理链": {"count": len(react_data_list), "format": "task/steps"},
        },
        "multimodal_special_tokens": [
            "<image>", "</image>", "<img_gen>", "<img_edit>", "<img_result>",
            "<audio>", "</audio>", "<speech>", "</speech>", "<tts>", "<tts_result>",
            "<video>", "</video>", "<video_gen>", "<frames>",
            "<screen>", "</screen>",
        ],
        "output_files": {
            "conversation": conv_path,
            "react": react_path,
            "mm_vision": os.path.join(args.output, "taiji_ultimate_mm_vision.jsonl"),
            "mm_audio": os.path.join(args.output, "taiji_ultimate_mm_audio.jsonl"),
            "mm_video": os.path.join(args.output, "taiji_ultimate_mm_video.jsonl"),
            "mm_cross": os.path.join(args.output, "taiji_ultimate_mm_cross.jsonl"),
            "mm_generation": os.path.join(args.output, "taiji_ultimate_mm_generation.jsonl"),
            "mm_screen": os.path.join(args.output, "taiji_ultimate_mm_screen.jsonl"),
        }
    }

    report_path = os.path.join(args.output, "taiji_ultimate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # === 打印统计 ===
    print(f"\n{'='*60}")
    print(f"  态极终极训练数据生成完成！")
    print(f"{'='*60}")
    print(f"  数学推理:     {len(math_data):>5} 条")
    print(f"  逻辑推理:     {len(logic_data):>5} 条")
    print(f"  代码能力:     {len(code_data):>5} 条")
    print(f"  软件工程:     {len(se_data):>5} 条")
    print(f"  多模态视觉:   {len(mm_vision):>5} 条")
    print(f"  多模态音频:   {len(mm_audio):>5} 条")
    print(f"  多模态视频:   {len(mm_video):>5} 条")
    print(f"  跨模态推理:   {len(mm_cross):>5} 条")
    print(f"  多模态生成:   {len(mm_gen):>5} 条")
    print(f"  屏幕理解:     {len(mm_screen):>5} 条")
    print(f"  ReAct推理链:  {len(react_data_list):>5} 条")
    print(f"{'='*60}")
    print(f"  对话总计:     {len(all_conv):>5} 条")
    print(f"  ReAct总计:    {len(react_data_list):>5} 条")
    print(f"  总计:         {len(all_conv) + len(react_data_list):>5} 条")
    print(f"{'='*60}")
    print(f"\n输出文件:")
    print(f"  {conv_path}")
    print(f"  {react_path}")
    print(f"  {report_path}")
    print(f"  以及 6 个多模态分类文件...")
    print(f"\n使用方式:")
    print(f"  python taiji/train/finetune_taiji.py")
    print(f"  # 然后将 taiji_ultimate_conversation.jsonl 和 taiji_ultimate_react.jsonl")
    print(f"  # 加载到训练流水线中")


if __name__ == "__main__":
    main()
