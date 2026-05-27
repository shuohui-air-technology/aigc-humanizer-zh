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

RULE_VERSION = "0.2"
DEFAULT_PARAGRAPH_MAX_SCORE = 16

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


@dataclass(frozen=True)
class PatternRule:
    """内部规则定义：正则、位置约束、权重和单段计分上限。"""

    pattern_id: int
    name: str
    severity: str
    patterns: list[str] | None
    suggestion: str
    location: str = "any"  # any | start | closing
    weight: int = 1
    score_cap: int = 1


@dataclass
class ParagraphRisk:
    """单段风险评分。"""

    index: int  # 段落序号（0-based）
    prefix: str  # 段落前 40 字，便于定位
    score: int = 0
    max_score: int = DEFAULT_PARAGRAPH_MAX_SCORE
    score_raw: int = 0
    score_capped: int = 0
    rule_version: str = RULE_VERSION
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
    rule_version: str = RULE_VERSION

    @property
    def total_score(self) -> int:
        return sum(p.score for p in self.paragraph_risks)


# ========================== 模式定义 ========================================

# 每条规则包含：正则/特殊检测、位置约束、计分权重和单段计分上限。
_PATTERN_DEFS: list[PatternRule] = [
    # ---- 模式 1：理论起笔 ----
    PatternRule(
        1,
        "理论起笔模式",
        "high",
        [
            # 段首出现 "依据/基于/根据...理论/框架/观点/原则"
            r"^(依据|基于|根据|按照|遵循)[^。\n]{0,40}(理论|框架|观点|原则|模型|视角|范式)",
        ],
        "将理论名称从段首移到段中，让现象描述在前、理论在需要解释时自然引入",
        location="start",
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 2：段末套路 ----
    PatternRule(
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
        location="closing",
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 3：编号逻辑 ----
    PatternRule(
        3,
        "整齐编号逻辑",
        "medium",
        [
            # 同时出现 首先...其次...再次/最后
            r"(首先|第一|其一).{0,60}(其次|第二|其二).{0,60}(再次|第三|最后|其三)",
        ],
        "用「最根本的是……此外……至于……」代替等长编号，让各条理由篇幅与重要性匹配",
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 4：被动分析套话 ----
    PatternRule(
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
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 5：模板化问题陈述 ----
    PatternRule(
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
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 6：三元并列对称 ----
    PatternRule(
        6,
        "高度对称三元并列",
        "medium",
        [
            # 检测 "理论上，...；实践上，...；方法上，..." 等三元维度结构。
            r"(?:[\u4e00-\u9fff]{1,6}(?:上|层面|维度|方面)[，,：:][^；;。]{2,80}[；;]\s*){2}[\u4e00-\u9fff]{1,6}(?:上|层面|维度|方面)[，,：:]",
        ],
        "主动打破三元对称；让各项表述长度与实际分量匹配，最重要的多说、次要的一笔带过",
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 7：段末冗余总结 ----
    PatternRule(
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
        location="closing",
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 8：模糊归因 ----
    PatternRule(
        8,
        "模糊归因",
        "high",
        [
            r"(专家认为|学者认为|研究表明|业内普遍认为|有观点认为|一些学者指出|学界普遍认同)",
        ],
        "有具体来源则引用；无出处则将观点改写为本文自身的分析判断并说明依据",
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 9：填充短语 ----
    PatternRule(
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
        weight=1,
        score_cap=1,
    ),
    # ---- 模式 10：泛化结论 ----
    PatternRule(
        10,
        "泛化结论与意义声明",
        "high",
        [
            r"(具有重要的(理论|现实|实践|学术)意义)",
            r"(具有重要意义)",
            r"(前景广阔|未来可期|具有广阔的发展空间)",
            r"(意义深远|影响深刻|意义重大)",
            r"(提供了新思路|开辟了新方向|打开了新路径)",
        ],
        "用「可检验的推论」或「具体的后续研究方向」代替泛化展望；有什么说什么",
        weight=3,
        score_cap=3,
    ),
    # ---- 模式 11：AI 高频词 ----
    PatternRule(
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
        weight=2,
        score_cap=4,
    ),
    # ---- 模式 12：回避「是」 ----
    PatternRule(
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
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 13：过度排比 ----
    PatternRule(
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
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 14：三步走结构 ----
    PatternRule(
        14,
        "结构性三步走",
        "medium",
        [
            r"(从[\u4e00-\u9fff]{1,6}(角度|维度|层面|方面)(看|而言)?).{0,80}(从[\u4e00-\u9fff]{1,6}(角度|维度|层面|方面)(看|而言)?).{0,80}(从[\u4e00-\u9fff]{1,6}(角度|维度|层面|方面)(看|而言)?)",
            r"(一方面).{0,60}(另一方面).{0,60}(此外)",
        ],
        "打破三维等重假设；让最重要的维度先说、多说，次要的简说，意外发现单独提出",
        weight=2,
        score_cap=2,
    ),
    # ---- 模式 15：破折号密度 ----
    PatternRule(
        15,
        "破折号密度异常",
        "low",
        None,  # 特殊处理：按段统计 —— 数量
        "一段内 —— 超过 4 次时删减为句子切分；全文冒号连续 3 次以上时部分改为破折号",
        weight=1,
        score_cap=1,
    ),
    # ---- 模式 16：加粗滥用 ----
    PatternRule(
        16,
        "正文加粗滥用",
        "low",
        None,  # 特殊处理：统计 **...** 数量
        "正文加粗控制在全文 ≤5 处（不含标题），超过时解除加粗并用句式变化替代强调",
        weight=1,
        score_cap=1,
    ),
]

# ---------------------------------------------------------------------------
# 硬约束定义（7 项，命中一条即高风险）
# ---------------------------------------------------------------------------
_HARD_CONSTRAINTS: list[dict] = [
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

_BLOCK_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])\s*")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[，,；;])\s*")


def _join_soft_lines(block: str) -> str:
    """将单换行视为软换行，保留空行作为段落边界。"""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return "".join(lines)


def _split_by_boundary(text: str, boundary: re.Pattern) -> list[str]:
    parts = boundary.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """将文本切分为段落，支持自动降级。

    优先按空行切分；若结果过少（≤2 段且平均每段 >200 字），
    则降级为按句号切分，再降级为按逗号切分，确保各类文本
    都能被合理拆分，避免单段落导致模式检测失效。
    """
    text = text.strip()
    if not text:
        return []

    # 第一级：只按空行切段；单换行默认是复制/排版产生的软换行。
    raw = _BLOCK_SPLIT.split(text)
    result = [_join_soft_lines(p) for p in raw if p.strip()]

    # 第二级：段太少且太长 → 按句号切
    avg_len = sum(len(p) for p in result) / len(result)
    if len(result) <= 2 and avg_len > 200:
        flattened = _join_soft_lines(text)
        result = _split_by_boundary(flattened, _SENTENCE_BOUNDARY)

    # 第三级：还是太少且太长 → 按逗号切
    avg_len = sum(len(p) for p in result) / len(result)
    if len(result) <= 2 and avg_len > 200:
        flattened = _join_soft_lines(text)
        result = _split_by_boundary(flattened, _CLAUSE_BOUNDARY)

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
        self._compiled: list[tuple[PatternRule, list[re.Pattern] | None]] = []
        for rule in _PATTERN_DEFS:
            if rule.patterns is None:
                self._compiled.append((rule, None))
            else:
                compiled = [re.compile(p) for p in rule.patterns]
                self._compiled.append((rule, compiled))

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
            rule_version=RULE_VERSION,
        )

    # ------------------------------------------------------------------
    # 段落级分析
    # ------------------------------------------------------------------

    def _analyze_paragraph(self, para: str, idx: int) -> ParagraphRisk:
        pr = ParagraphRisk(index=idx, prefix=para[:40])

        for rule, compiled in self._compiled:
            if compiled is None:
                # 特殊处理模式（15/16）
                matches = self._detect_special(rule.pattern_id, para)
            else:
                matches = []
                for cre in compiled:
                    for m in cre.finditer(para):
                        matched = m.group(0)
                        if not self._passes_location(rule.location, para, m.start(), matched):
                            continue
                        if not self._passes_context_filter(
                            rule.pattern_id, para, m.start(), matched
                        ):
                            continue
                        matches.append(matched)

            for matched_text in matches:
                loc = self._guess_location(rule.pattern_id, para, matched_text)
                pm = PatternMatch(
                    pattern_id=rule.pattern_id,
                    pattern_name=rule.name,
                    severity=rule.severity,
                    location_hint=loc,
                    matched_text=matched_text[:80],
                    suggestion=rule.suggestion,
                )
                pr.matches.append(pm)

            raw_score = len(matches) * rule.weight
            pr.score_raw += raw_score
            pr.score += min(raw_score, rule.score_cap)

        pr.score_capped = min(pr.score, pr.max_score)
        pr.score = pr.score_capped

        return pr

    # ------------------------------------------------------------------
    # 特殊模式（15：破折号 / 16：加粗）
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_special(pid: int, para: str) -> list[str]:
        if pid == 15:
            # 破折号密度
            count = len(re.findall(r"——|--|—", para))
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
                items = self._match_items(pr, p11_matches, "每段 AI 高频词超过 2 个")
                violations.append(
                    {
                        "constraint": "HC-1",
                        "name": "AI高频词密度过高",
                        "paragraph": pr.index + 1,
                        "pattern_ids": [11],
                        "detail": f"第{pr.index+1}段命中 {len(p11_matches)} 个 AI 高频词（上限 2）",
                        "reason": "每段 AI 高频词超过 2 个",
                        "items": items,
                    }
                )

        # HC-2: 段末总结套句（全文 > 1）
        ending_cliches: list[dict] = []
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id in (2, 7):
                    ending_cliches.extend(
                        self._match_items(pr, [m], "全文段末总结套句超过 1 处")
                    )
        if len(ending_cliches) > 1:
            violations.append(
                {
                    "constraint": "HC-2",
                    "name": "段末总结套句过多",
                    "paragraph": None,
                    "pattern_ids": [2, 7],
                    "detail": f"全文检测到 {len(ending_cliches)} 处段末套句（上限 1）",
                    "reason": "全文段末总结套句超过 1 处",
                    "items": ending_cliches,
                }
            )

        # HC-3: 三元并列（每段 > 1）
        for pr in para_risks:
            triple = [m for m in pr.matches if m.pattern_id in (3, 6, 14)]
            if len(triple) > 1:
                items = self._match_items(pr, triple, "每段三元并列结构超过 1 处")
                violations.append(
                    {
                        "constraint": "HC-3",
                        "name": "整齐三元并列过多",
                        "paragraph": pr.index + 1,
                        "pattern_ids": [3, 6, 14],
                        "detail": f"第{pr.index+1}段含 {len(triple)} 处三元并列（上限 1）",
                        "reason": "每段三元并列结构超过 1 处",
                        "items": items,
                    }
                )

        # HC-4: 理论起笔占比 > 20%
        theory_opening_count = sum(
            1 for pr in para_risks if any(m.pattern_id == 1 for m in pr.matches)
        )
        if paragraphs and theory_opening_count / len(paragraphs) > 0.2:
            items: list[dict] = []
            for pr in para_risks:
                theory_matches = [m for m in pr.matches if m.pattern_id == 1]
                items.extend(
                    self._match_items(pr, theory_matches, "理论起笔段落占比超过 20%")
                )
            violations.append(
                {
                    "constraint": "HC-4",
                    "name": "理论起笔段落占比过高",
                    "paragraph": None,
                    "pattern_ids": [1],
                    "detail": f"{theory_opening_count}/{len(paragraphs)} 段以理论开头（上限 20%）",
                    "reason": "理论起笔段落占比超过 20%",
                    "items": items,
                }
            )

        # HC-5: 加粗 > 5
        bold_items: list[dict] = []
        for idx, paragraph in enumerate(paragraphs):
            for matched in re.findall(r"\*\*[^*]+\*\*", paragraph):
                bold_items.append(
                    {
                        "para": idx + 1,
                        "paragraph": idx + 1,
                        "pattern_id": 16,
                        "pattern_name": "正文加粗滥用",
                        "text": matched,
                        "matched_text": matched,
                        "reason": "全文正文加粗超过 5 处",
                    }
                )
        bold_total = len(bold_items)
        if bold_total > 5:
            violations.append(
                {
                    "constraint": "HC-5",
                    "name": "正文加粗过多",
                    "paragraph": None,
                    "pattern_ids": [16],
                    "detail": f"全文含 {bold_total} 处加粗（上限 5）",
                    "reason": "全文正文加粗超过 5 处",
                    "items": bold_items,
                }
            )

        # HC-6: 泛化结尾（命中即违规）
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id == 10:
                    items = self._match_items(pr, [m], "泛化意义声明命中即违规")
                    violations.append(
                        {
                            "constraint": "HC-6",
                            "name": "泛化结尾",
                            "paragraph": pr.index + 1,
                            "pattern_ids": [10],
                            "detail": f"第{pr.index+1}段检测到泛化结论: 「{m.matched_text}」",
                            "reason": "泛化意义声明命中即违规",
                            "items": items,
                        }
                    )

        # HC-7: 模糊归因（命中即违规）
        for pr in para_risks:
            for m in pr.matches:
                if m.pattern_id == 8:
                    items = self._match_items(pr, [m], "无具体出处的模糊归因命中即违规")
                    violations.append(
                        {
                            "constraint": "HC-7",
                            "name": "模糊归因",
                            "paragraph": pr.index + 1,
                            "pattern_ids": [8],
                            "detail": f"第{pr.index+1}段检测到模糊归因: 「{m.matched_text}」",
                            "reason": "无具体出处的模糊归因命中即违规",
                            "items": items,
                        }
                    )

        return violations

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _match_items(
        pr: ParagraphRisk, matches: list[PatternMatch], reason: str
    ) -> list[dict]:
        return [
            {
                "para": pr.index + 1,
                "paragraph": pr.index + 1,
                "pattern_id": m.pattern_id,
                "pattern_name": m.pattern_name,
                "text": m.matched_text,
                "matched_text": m.matched_text,
                "reason": reason,
            }
            for m in matches
        ]

    @staticmethod
    def _severity_weight(sev: str) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(sev, 1)

    @staticmethod
    def _passes_location(
        location: str, para: str, start: int, matched: str
    ) -> bool:
        if location == "any":
            return True
        if location == "start":
            return start <= max(5, int(len(para) * 0.15))
        if location == "closing":
            last_sentence_start = max(
                para.rfind("。", 0, start),
                para.rfind("！", 0, start),
                para.rfind("？", 0, start),
            )
            if last_sentence_start >= 0:
                last_sentence_start += 1
            else:
                last_sentence_start = 0

            marker_is_at_sentence_start = start - last_sentence_start <= 3
            next_breaks = [
                pos
                for pos in (
                    para.find("。", start + len(matched)),
                    para.find("！", start + len(matched)),
                    para.find("？", start + len(matched)),
                )
                if pos >= 0
            ]
            next_break = min(next_breaks) if next_breaks else -1
            is_final_sentence = (
                next_break == -1 or not para[next_break + 1 :].strip()
            )
            return marker_is_at_sentence_start and is_final_sentence
        return True

    @staticmethod
    def _passes_context_filter(
        pid: int, para: str, start: int, matched: str
    ) -> bool:
        if pid != 8:
            return True

        if matched == "研究表明" and start > 0 and para[start - 1] == "本":
            return False

        lookback = para[max(0, start - 50) : start]
        citation_before = re.search(
            r"([A-Z][A-Za-z\-]+|[\u4e00-\u9fff]{2,8})[（(]\d{4}[）)]的?\s*$",
            lookback,
        )
        parenthetical_citation_before = re.search(
            r"[（(][^（）()]{0,40}\d{4}[^（）()]{0,20}[）)]的?\s*$",
            lookback,
        )
        return not (citation_before or parenthetical_citation_before)

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
