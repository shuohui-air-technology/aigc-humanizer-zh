"""
AI 模式检测引擎
===============
基于 humanizer-zh-academic skill 的 16 种 AI 写作模式，提供规则化检测与风险评分。

设计原则：
  - 纯规则驱动（正则 + 关键词），无需 GPU / LLM 推理。
  - 每种模式返回匹配位置、匹配文本、严重程度和改写建议。
  - 硬约束（7 项）与软模式（16 项）分开评估，硬约束命中即判定高风险。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """单次模式命中结果。"""

    pattern_id: int
    pattern_name: str
    severity: str  # "high" | "medium" | "low"
    location_hint: str  # 命中位置描述，如 "段首" / "段末" / "段中"
    matched_text: str  # 截取的匹配片段（最多 80 字）
    suggestion: str  # 具体改写建议


@dataclass
class ParagraphRisk:
    """单段风险评分。"""

    index: int  # 段落序号（0-based）
    prefix: str  # 段落前 40 字，便于定位
    score: int = 0
    max_score: int = 16
    matches: list[PatternMatch] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        pct = self.score / max(self.max_score, 1)
        if pct >= 0.5:
            return "🔴 高风险"
        if pct >= 0.25:
            return "🟡 中风险"
        return "🟢 低风险"


@dataclass
class RiskReport:
    """全文风险报告。"""

    total_paragraphs: int
    paragraph_risks: list[ParagraphRisk]
    hard_violations: list[dict]  # 硬约束命中详情
    overall_risk: str  # "高风险" | "中风险" | "低风险"
    summary: str  # 一句话总结

    @property
    def total_score(self) -> int:
        return sum(p.score for p in self.paragraph_risks)


# ========================== 模式定义 ========================================

# 每条模式：(id, name, severity, 正则/关键词列表, 改写建议模板)
_PATTERN_DEFS: ClassVar = [
    # ---- 模式 1：理论起笔 ----
    (
        1,
        "理论起笔模式",
        "high",
        [
            # 段首出现 "依据/基于/根据...理论/框架/观点/原则"
            r"^(依据|基于|根据|按照|遵循)[^。\n]{0,40}(理论|框架|观点|原则|模型|视角|范式)",
        ],
        "将理论名称从段首移到段中，让现象描述在前、理论在需要解释时自然引入",
    ),
    # ---- 模式 2：段末套路 ----
    (
        2,
        "段末套路结尾",
        "high",
        [
            r"(此案例(印证|挑战|揭示|完美诠释)了)",
            r"(从中可以看出)",
            r"(这提示我们)",
            r"(该结论(印证|表明|揭示)了)",
        ],
        "删除「此案例XX了」的固定开头；将结论转化为从问题逻辑出发的自然推断",
    ),
    # ---- 模式 3：编号逻辑 ----
    (
        3,
        "整齐编号逻辑",
        "medium",
        [
            # 同时出现 首先...其次...再次/最后
            r"(首先|第一|其一).{0,60}(其次|第二|其二).{0,60}(再次|第三|最后|其三)",
        ],
        "用「最根本的是……此外……至于……」代替等长编号，让各条理由篇幅与重要性匹配",
    ),
    # ---- 模式 4：被动分析套话 ----
    (
        4,
        "被动分析套话",
        "high",
        [
            r"(该处理体现了)",
            r"(该设计基于)",
            r"(该决策反映了)",
            r"(这一做法展现了)",
            r"(上述选择印证了)",
            r"(该(方法|方案|策略)的(选择|设计)基于)",
        ],
        "把「该XX基于/体现了」改为说明「为什么这么做」的具体叙述，加入研究过程中的真实判断",
    ),
    # ---- 模式 5：模板化问题陈述 ----
    (
        5,
        "模板化问题陈述",
        "medium",
        [
            r"(面临的核心问题是)",
            r"(核心挑战在于)",
            r"(主要矛盾体现在)",
            r"(关键问题是如何)",
            r"(XX面临的核心问题是)",
        ],
        "用具体的矛盾情境代替抽象的「核心问题」声明；用反问或设问把问题演示出来",
    ),
    # ---- 模式 6：三元并列对称 ----
    (
        6,
        "高度对称三元并列",
        "medium",
        [
            # 检测 "XX上，...；XX上，...；XX上，..."  或 "理论上/实践上/方法上"
            r"([\u4e00-\u9fff]{1,6}(上|层面|维度))[，,].{5,60}[；;]\s*\1[，,].{5,60}[；;]",
        ],
        "主动打破三元对称；让各项表述长度与实际分量匹配，最重要的多说、次要的一笔带过",
    ),
    # ---- 模式 7：段末冗余总结 ----
    (
        7,
        "段末冗余总结句",
        "high",
        [
            r"(综上所述)",
            r"(由此可见)",
            r"(不难发现)",
            r"(可以看出)",
            r"(因此可以得出结论)",
            r"(总之[,，])",
            r"(总的来看)",
        ],
        "删除段末的总结句；如需衔接下段，用过渡提问或转折句代替",
    ),
    # ---- 模式 8：模糊归因 ----
    (
        8,
        "模糊归因",
        "high",
        [
            r"(专家认为|学者认为|业内普遍认为|有观点认为|一些学者指出|学界普遍认同)",
        ],
        "有具体来源则引用；无出处则将观点改写为本文自身的分析判断并说明依据",
    ),
    # ---- 模式 9：填充短语 ----
    (
        9,
        "填充短语与过度限定",
        "low",
        [
            r"(值得注意的是[,，])",
            r"(需要指出的是[,，])",
            r"(总体而言[,，])",
            r"(总的来说[,，])",
            r"(毋庸置疑[,，])",
        ],
        "直接删除填充引导语，让句子从信息开始",
    ),
    # ---- 模式 10：泛化结论 ----
    (
        10,
        "泛化结论与意义声明",
        "high",
        [
            r"(具有重要意义)",
            r"(前景广阔|未来可期|具有广阔的发展空间)",
            r"(意义深远|影响深刻|意义重大)",
            r"(提供了新思路|开辟了新方向|打开了新路径)",
            r"(具有重要的(理论|现实|实践|学术)意义)",
        ],
        "用「可检验的推论」或「具体的后续研究方向」代替泛化展望；有什么说什么",
    ),
    # ---- 模式 11：AI 高频词 ----
    (
        11,
        "AI 高频词汇",
        "medium",
        [
            # 优先处理（权重高）
            r"(深刻揭示了)",
            r"(不可或缺)",
            r"(综合运用)",
            # 次要处理
            r"(深入探讨)",
            r"(系统梳理)",
            r"(提供了理论支撑)",
            r"(有效解决了)",
            r"(完善了理论体系)",
            r"(充分说明)",
        ],
        "优先替换词表前4项（深刻揭示→说明/表明，不可或缺→离不开，等）；其他视上下文决定",
    ),
    # ---- 模式 12：回避「是」 ----
    (
        12,
        "回避系动词「是」",
        "medium",
        [
            r"(作为.{1,20}的.{1,10}(载体|角色|功能|桥梁))",
            r"(扮演着.{1,15}的角色)",
            r"(发挥着.{1,15}的作用)",
            r"(充当着.{1,15}的功能)",
            r"(起到了.{1,15}的作用)",
        ],
        "直接使用「是」代替「作为XX载体」「扮演着XX角色」等冗长搭配",
    ),
    # ---- 模式 13：过度排比 ----
    (
        13,
        "过度对仗排比",
        "medium",
        [
            # 四字短语连续出现 4 次以上（如 "突破范式，填补空白，创新视角，丰富方法"）
            r"([\u4e00-\u9fff]{4}[，,、]){4,}",
            # 五字对偶
            r"([\u4e00-\u9fff]{5}[，,、]){4,}",
        ],
        "打破工整排比，集中写真正重要的贡献点，次要的不写或用一句话带过",
    ),
    # ---- 模式 14：三步走结构 ----
    (
        14,
        "结构性三步走",
        "medium",
        [
            r"(从[\u4e00-\u9fff]{1,4}(角度|维度|层面)看).{0,60}(从[\u4e00-\u9fff]{1,4}(角度|维度|层面)看).{0,60}(从[\u4e00-\u9fff]{1,4}(角度|维度|层面)看)",
            r"(一方面).{0,60}(另一方面).{0,60}(此外)",
        ],
        "打破三维等重假设；让最重要的维度先说、多说，次要的简说，意外发现单独提出",
    ),
    # ---- 模式 15：破折号密度 ----
    (
        15,
        "破折号密度异常",
        "low",
        None,  # 特殊处理：按段统计 —— 数量
        "一段内 —— 超过 4 次时删减为句子切分；全文冒号连续 3 次以上时部分改为破折号",
    ),
    # ---- 模式 16：加粗滥用 ----
    (
        16,
        "正文加粗滥用",
        "low",
        None,  # 特殊处理：统计 **...** 数量
        "正文加粗控制在全文 ≤5 处（不含标题），超过时解除加粗并用句式变化替代强调",
    ),
]

# ---------------------------------------------------------------------------
# 硬约束定义（7 项，命中一条即高风险）
# ---------------------------------------------------------------------------
_HARD_CONSTRAINTS: ClassVar = [
    {
        "id": "HC-1",
        "name": "AI高频词密度过高",
        "check": "ai_word_density",
        "threshold": "每段 > 2 个即命中",
        "pattern_ids": [11],
    },
    {
        "id": "HC-2",
        "name": "段末总结套句过多",
        "check": "ending_cliche_count",
        "threshold": "全文 > 1 处即命中",
        "pattern_ids": [2, 7],
    },
    {
        "id": "HC-3",
        "name": "整齐三元并列过多",
        "check": "triple_parallel_count",
        "threshold": "每段 > 1 处即命中",
        "pattern_ids": [3, 6, 14],
    },
    {
        "id": "HC-4",
        "name": "理论起笔段落占比过高",
        "check": "theory_opening_ratio",
        "threshold": ">20% 段落数即命中",
        "pattern_ids": [1],
    },
    {
        "id": "HC-5",
        "name": "正文加粗过多",
        "check": "bold_count",
        "threshold": "全文 > 5 处即命中",
        "pattern_ids": [16],
    },
    {
        "id": "HC-6",
        "name": "泛化结尾",
        "check": "generic_ending",
        "threshold": "全文 0 处（命中即违规）",
        "pattern_ids": [10],
    },
    {
        "id": "HC-7",
        "name": "模糊归因",
        "check": "vague_attribution",
        "threshold": "全文 0 处（命中即违规）",
        "pattern_ids": [8],
    },
]

# ---------------------------------------------------------------------------
# 段落切分
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r"\n{1,3}")
_SENTENCE_SPLIT = re.compile(r"[。！？\n]+")
_CLAUSE_SPLIT = re.compile(r"[，,；;]+")


def _split_paragraphs(text: str) -> list[str]:
    """将文本切分为段落，支持自动降级。

    优先按空行切分；若结果过少（≤2 段且平均每段 >200 字），
    则降级为按句号切分，再降级为按逗号切分，确保各类文本
    都能被合理拆分，避免单段落导致模式检测失效。
    """
    text = text.strip()
    if not text:
        return []

    # 第一级：按空行/换行切
    raw = _PARAGRAPH_SPLIT.split(text)
    result = [p.strip() for p in raw if p.strip()]

    # 第二级：段太少且太长 → 按句号切
    avg_len = sum(len(p) for p in result) / len(result)
    if len(result) <= 2 and avg_len > 200:
        raw = _SENTENCE_SPLIT.split(text)
        result = [p.strip() for p in raw if p.strip()]

    # 第三级：还是太少且太长 → 按逗号切
    avg_len = sum(len(p) for p in result) / len(result)
    if len(result) <= 2 and avg_len > 200:
        raw = _CLAUSE_SPLIT.split(text)
        result = [p.strip() for p in raw if p.strip()]

    return result


# ---------------------------------------------------------------------------
# 模式检测核心
# ---------------------------------------------------------------------------


class PatternDetector:
    """16 种 AI 写作模式检测器。

    用法:
        detector = PatternDetector()
        report = detector.analyze(text)
        print(report.overall_risk)
        for pr in report.paragraph_risks:
            print(f"  段{pr.index}: {pr.risk_level} ({pr.score}分)")
    """

    def __init__(self) -> None:
        # 预编译所有正则
        self._compiled: list[tuple[int, str, str, list[re.Pattern] | None, str]] = []
        for pid, name, sev, patterns, suggestion in _PATTERN_DEFS:
            if patterns is None:
                self._compiled.append((pid, name, sev, None, suggestion))
            else:
                compiled = [re.compile(p) for p in patterns]
                self._compiled.append((pid, name, sev, compiled, suggestion))

        self._hc = _HARD_CONSTRAINTS

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> RiskReport:
        """对全文执行 16 种模式扫描，返回结构化风险报告。"""
        paragraphs = _split_paragraphs(text)
        para_risks: list[ParagraphRisk] = []

        for idx, para in enumerate(paragraphs):
            pr = self._analyze_paragraph(para, idx)
            para_risks.append(pr)

        # 硬约束评估
        hard_violations = self._evaluate_hard_constraints(paragraphs, para_risks)

        # 综合风险判定
        total = sum(p.score for p in para_risks)
        max_possible = sum(p.max_score for p in para_risks)
        ratio = total / max(max_possible, 1)

        if hard_violations:
            overall = "🔴 高风险"
        elif ratio >= 0.35:
            overall = "🟡 中风险"
        else:
            overall = "🟢 低风险"

        summary = self._build_summary(total, len(paragraphs), hard_violations, ratio)

        return RiskReport(
            total_paragraphs=len(paragraphs),
            paragraph_risks=para_risks,
            hard_violations=hard_violations,
            overall_risk=overall,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 段落级分析
    # ------------------------------------------------------------------

    def _analyze_paragraph(self, para: str, idx: int) -> ParagraphRisk:
        pr = ParagraphRisk(index=idx, prefix=para[:40])

        for pid, name, sev, compiled, suggestion in self._compiled:
            if compiled is None:
                # 特殊处理模式（15/16）
                matches = self._detect_special(pid, para)
            else:
                matches = []
                for cre in compiled:
                    for m in cre.finditer(para):
                        matched = m.group(0)
                        # 模式 1 只在段首生效
                        if pid == 1 and m.start() > 5:
                            continue
                        # 模式 2/7 优先在段末生效，但也检测段中
                        matches.append(matched)

            for matched_text in matches:
                loc = self._guess_location(pid, para, matched_text)
                pm = PatternMatch(
                    pattern_id=pid,
                    pattern_name=name,
                    severity=sev,
                    location_hint=loc,
                    matched_text=matched_text[:80],
                    suggestion=suggestion,
                )
                pr.matches.append(pm)
                pr.score += self._severity_weight(sev)

        return pr

    # ------------------------------------------------------------------
    # 特殊模式（15：破折号 / 16：加粗）
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_special(pid: int, para: str) -> list[str]:
        if pid == 15:
            # 破折号密度
            count = para.count("——") + para.count("--") + para.count("—")
            if count >= 4:
                return [f"本段含 {count} 处破折号（阈值 4）"]
        elif pid == 16:
            # 加粗（Markdown **...**）
            count = len(re.findall(r"\*\*[^*]+\*\*", para))
            if count >= 2:
                return [f"本段含 {count} 处加粗"]
        return []

    # ------------------------------------------------------------------
    # 硬约束评估
    # ------------------------------------------------------------------

    def _evaluate_hard_constraints(
        self, paragraphs: list[str], para_risks: list[ParagraphRisk]
    ) -> list[dict]:
        violations: list[dict] = []

        # HC-1: AI 高频词密度（每段 > 2）
        for pr in para_risks:
            p11_matches = [m for m in pr.matches if m.pattern_id == 11]
            if len(p11_matches) > 2:
                violations.append(
                    {
                        "constraint": "HC-1",
                        "name": "AI高频词密度过高",
                        "detail": f"第{pr.index+1}段命中 {len(p11_matches)} 个 AI 高频词（上限 2）",
                        "items": [m.matched_text for m in p11_matches],
                    }
                )

        # HC-2: 段末总结套句（全文 > 1）
        ending_cliches: list[dict] = []
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id in (2, 7):
                    ending_cliches.append(
                        {"para": pr.index + 1, "text": m.matched_text}
                    )
        if len(ending_cliches) > 1:
            violations.append(
                {
                    "constraint": "HC-2",
                    "name": "段末总结套句过多",
                    "detail": f"全文检测到 {len(ending_cliches)} 处段末套句（上限 1）",
                    "items": ending_cliches,
                }
            )

        # HC-3: 三元并列（每段 > 1）
        for pr in para_risks:
            triple = [m for m in pr.matches if m.pattern_id in (3, 6, 14)]
            if len(triple) > 1:
                violations.append(
                    {
                        "constraint": "HC-3",
                        "name": "整齐三元并列过多",
                        "detail": f"第{pr.index+1}段含 {len(triple)} 处三元并列（上限 1）",
                        "items": [m.matched_text for m in triple],
                    }
                )

        # HC-4: 理论起笔占比 > 20%
        theory_opening_count = sum(
            1 for pr in para_risks if any(m.pattern_id == 1 for m in pr.matches)
        )
        if paragraphs and theory_opening_count / len(paragraphs) > 0.2:
            violations.append(
                {
                    "constraint": "HC-4",
                    "name": "理论起笔段落占比过高",
                    "detail": f"{theory_opening_count}/{len(paragraphs)} 段以理论开头（上限 20%）",
                    "items": [],
                }
            )

        # HC-5: 加粗 > 5
        bold_total = sum(
            len(re.findall(r"\*\*[^*]+\*\*", p)) for p in paragraphs
        )
        if bold_total > 5:
            violations.append(
                {
                    "constraint": "HC-5",
                    "name": "正文加粗过多",
                    "detail": f"全文含 {bold_total} 处加粗（上限 5）",
                    "items": [],
                }
            )

        # HC-6: 泛化结尾（命中即违规）
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id == 10:
                    violations.append(
                        {
                            "constraint": "HC-6",
                            "name": "泛化结尾",
                            "detail": f"第{pr.index+1}段检测到泛化结论: 「{m.matched_text}」",
                            "items": [m.matched_text],
                        }
                    )

        # HC-7: 模糊归因（命中即违规）
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id == 8:
                    violations.append(
                        {
                            "constraint": "HC-7",
                            "name": "模糊归因",
                            "detail": f"第{pr.index+1}段检测到模糊归因: 「{m.matched_text}」",
                            "items": [m.matched_text],
                        }
                    )

        return violations

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_weight(sev: str) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(sev, 1)

    @staticmethod
    def _guess_location(pid: int, para: str, matched: str) -> str:
        idx = para.find(matched)
        if idx < 0:
            return "未知"
        ratio = idx / max(len(para), 1)
        if ratio < 0.15:
            return "段首"
        if ratio > 0.75:
            return "段末"
        return "段中"

    @staticmethod
    def _build_summary(
        total_score: int, n_paras: int, violations: list[dict], ratio: float
    ) -> str:
        parts: list[str] = []
        parts.append(f"共 {n_paras} 段，风险总分 {total_score}")
        if violations:
            parts.append(f"硬约束命中 {len(violations)} 项")
        if ratio >= 0.5:
            parts.append("→ 建议深度改写后重新检测")
        elif ratio >= 0.25:
            parts.append("→ 建议局部修补高风险段落")
        else:
            parts.append("→ 风险较低，可微调后提交")
        return "；".join(parts)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_default_detector = PatternDetector()


def quick_scan(text: str) -> RiskReport:
    """快捷扫描，使用默认检测器实例。"""
    return _default_detector.analyze(text)
