"""
文本质量评估器
==============
提供两个维度的评估：
  1. TtrEvaluator  — 基于 jieba 分词的 Type-Token Ratio 计算
  2. QualityAssessor — 6 维度 60 分制综合学术质量评分

均基于 humanizer-zh-academic skill 定义的标准。
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

import jieba

# ---------------------------------------------------------------------------
# 标点清理表
# ---------------------------------------------------------------------------

_PUNCTUATION_TABLE = str.maketrans(
    "",
    "",
    string.punctuation + "，。！？；：""''（）【】《》、…—·～‖「」『』",
)


# ========================== TTR 评估器 ======================================

# 违禁词库（必须拦截的高 AI 特征词）
_BANNED_PHRASES: list[str] = [
    "综上所述",
    "由此可见",
    "不可或缺",
    "具有重要意义",
    "此案例印证了",
]


@dataclass
class TtrResult:
    """TTR 评估结果。"""

    passed: bool
    ttr_score: float  # 0.0 ~ 1.0
    token_count: int
    unique_tokens: int
    banned_words_found: list[str]
    details: str


class TtrEvaluator:
    """Type-Token Ratio 词汇丰富度评估器。

    用法:
        evaluator = TtrEvaluator()
        result = evaluator.evaluate(text, target_threshold=0.45)
        print(f"TTR={result.ttr_score:.4f}, passed={result.passed}")
    """

    def __init__(self, target_threshold: float = 0.45) -> None:
        self.target_threshold = target_threshold

    def evaluate(self, text: str) -> TtrResult:
        """计算 TTR 并扫描违禁词。"""
        try:
            # 1. 清理标点
            cleaned = text.translate(_PUNCTUATION_TABLE)

            # 2. jieba 分词
            tokens_raw = jieba.lcut(cleaned)
            tokens = [t.strip() for t in tokens_raw if t.strip()]

            if not tokens:
                return TtrResult(
                    passed=False,
                    ttr_score=0.0,
                    token_count=0,
                    unique_tokens=0,
                    banned_words_found=[],
                    details="分词结果为空（文本仅含标点或空白）",
                )

            # 3. TTR
            unique = len(set(tokens))
            total = len(tokens)
            ttr = unique / total

            # 4. 违禁词
            found = [w for w in _BANNED_PHRASES if w in text]

            # 5. 判定
            reasons: list[str] = []
            if ttr < self.target_threshold:
                reasons.append(
                    f"TTR ({ttr:.4f}) 低于阈值 ({self.target_threshold})"
                )
            if found:
                reasons.append(f"违禁词: {', '.join(found)}")

            passed = len(reasons) == 0

            return TtrResult(
                passed=passed,
                ttr_score=round(ttr, 4),
                token_count=total,
                unique_tokens=unique,
                banned_words_found=found,
                details="; ".join(reasons) if reasons else "通过",
            )
        except Exception as exc:
            return TtrResult(
                passed=False,
                ttr_score=0.0,
                token_count=0,
                unique_tokens=0,
                banned_words_found=[],
                details=f"评估异常: {exc}",
            )


# ========================== 60 分制质量评估 =================================


@dataclass
class QualityScore:
    """6 维度质量评分。"""

    directness: int  # 直接性 /10
    rhythm: int  # 节奏变化 /10
    authenticity: int  # 真实性 /10
    information_density: int  # 信息密度 /10
    academic_norm: int  # 学术规范 /10
    anti_detection: int  # 抗检测性 /10
    total: int = 0  # /60
    grade: str = ""  # "优秀" | "良好" | "需修订"
    suggestions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.total = (
            self.directness
            + self.rhythm
            + self.authenticity
            + self.information_density
            + self.academic_norm
            + self.anti_detection
        )
        if self.total >= 54:
            self.grade = "优秀，可直接提交"
        elif self.total >= 42:
            self.grade = "良好，针对扣分项局部修补"
        else:
            self.grade = "需重新修订高风险段落"


class QualityAssessor:
    """6 维度 60 分制学术文本质量评估器。

    评分维度（每项 0-10）：
      1. 直接性 — 直接陈述 vs 绕圈宣告
      2. 节奏   — 句子长度变化，长短交替
      3. 真实性 — 是否像真实学者在说话
      4. 信息密度 — 每句话承载信息，无废话
      5. 学术规范 — 归因具体、限定合理、语域匹配
      6. 抗检测性 — 模式规律性是否被充分打破

    用法:
        assessor = QualityAssessor()
        score = assessor.assess(text)
        print(f"{score.total}/60 — {score.grade}")
    """

    # ── 各维度评分规则 ──────────────────────────────────────────────

    def assess(self, text: str) -> QualityScore:
        """对文本执行 6 维度评分。"""
        sentences = self._split_sentences(text)
        suggestions: list[str] = []

        directness = self._score_directness(text, suggestions)
        rhythm = self._score_rhythm(sentences, suggestions)
        authenticity = self._score_authenticity(text, suggestions)
        info_density = self._score_information_density(text, suggestions)
        academic_norm = self._score_academic_norm(text, suggestions)
        anti_detection = self._score_anti_detection(text, sentences, suggestions)

        return QualityScore(
            directness=directness,
            rhythm=rhythm,
            authenticity=authenticity,
            information_density=info_density,
            academic_norm=academic_norm,
            anti_detection=anti_detection,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # 维度 1：直接性（0-10）
    # ------------------------------------------------------------------

    def _score_directness(self, text: str, suggestions: list[str]) -> int:
        score = 8  # 起评分
        # 绕圈宣告模式
        indirect_patterns = [
            r"(值得注意的是[,，])",
            r"(需要指出的是[,，])",
            r"(总体而言[,，])",
            r"(作为.{1,20}的.{1,10}(载体|角色|桥梁))",
            r"(扮演着.{1,15}的角色)",
            r"(发挥着.{1,15}的作用)",
        ]
        import re

        for pat in indirect_patterns:
            count = len(re.findall(pat, text))
            if count > 0:
                score -= min(count, 3)
        if score < 10:
            suggestions.append(f"直接性: {score}/10 — 减少「值得注意的是」「作为XX载体」等绕圈表达")
        return max(score, 5)

    # ------------------------------------------------------------------
    # 维度 2：节奏（0-10）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rhythm(sentences: list[str], suggestions: list[str]) -> int:
        if len(sentences) < 3:
            return 7

        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)

        # 检查是否有长短交替
        var_count = sum(
            1 for i in range(1, len(lengths)) if abs(lengths[i] - lengths[i - 1]) > avg_len * 0.5
        )
        rhythm_ratio = var_count / max(len(lengths) - 1, 1)

        # 检查全长句段落
        all_long = sum(1 for l in lengths if l > avg_len * 1.5)
        long_ratio = all_long / max(len(lengths), 1)

        score = 8
        if rhythm_ratio < 0.2:
            score -= 3
        if long_ratio > 0.6:
            score -= 2

        if score < 10:
            suggestions.append(
                f"节奏: {score}/10 — 句长变化不足，建议穿插短句打破全长句段落"
            )
        return max(score, 4)

    # ------------------------------------------------------------------
    # 维度 3：真实性（0-10）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_authenticity(text: str, suggestions: list[str]) -> int:
        score = 6

        # 检测是否有研究者主体表达
        authentic_markers = [
            r"(笔者认为)",
            r"(出乎意料)",
            r"(坦率地说|老实说)",
            r"(笔者发现)",
            r"(本研究认为)",
            r"(本文认为)",
            r"(在笔者看来)",
        ]
        import re

        has_voice = any(re.search(m, text) for m in authentic_markers)
        if has_voice:
            score += 3

        # 扣分：过于中性的报道式语言
        report_markers = [
            r"(研究结果显示)",
            r"(数据表明)",
            r"(统计结果表明)",
        ]
        report_count = sum(len(re.findall(m, text)) for m in report_markers)
        if report_count >= 3 and not has_voice:
            score -= 3

        if score < 10:
            suggestions.append(
                f"真实性: {score}/10 — "
                + ("缺少研究者视角表达（如「笔者认为」「出乎意料」）" if not has_voice else "可进一步增强主观表达")
            )
        return max(score, 3)

    # ------------------------------------------------------------------
    # 维度 4：信息密度（0-10）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_information_density(text: str, suggestions: list[str]) -> int:
        score = 8
        import re

        # 填充短语
        filler_patterns = [
            r"(值得注意的是)",
            r"(需要指出的是)",
            r"(总体而言)",
            r"(毋庸置疑)",
        ]
        filler_count = sum(len(re.findall(p, text)) for p in filler_patterns)
        score -= min(filler_count, 4)

        # 过度限定
        overqualified = len(re.findall(r"(可能在一定程度上潜在地|某种程度上的)", text))
        score -= overqualified * 2

        if filler_count > 0 or overqualified > 0:
            suggestions.append(f"信息密度: {score}/10 — 删除填充短语和冗余限定词")
        return max(score, 3)

    # ------------------------------------------------------------------
    # 维度 5：学术规范（0-10）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_academic_norm(text: str, suggestions: list[str]) -> int:
        score = 8
        import re

        # 模糊归因扣分
        vague = len(re.findall(r"(专家认为|学者认为|业内普遍认为|一些学者指出)", text))
        score -= min(vague * 3, 6)

        # 具体引用加分
        has_citation = bool(re.search(r"[(（][A-Z][a-z]+.*?\d{4}[)）]", text))
        if has_citation:
            score += 1

        if vague > 0:
            suggestions.append(f"学术规范: {score}/10 — 检测到 {vague} 处模糊归因，需替换为具体引用或本文判断")
        return max(score, 2)

    # ------------------------------------------------------------------
    # 维度 6：抗检测性（0-10）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_anti_detection(
        text: str, sentences: list[str], suggestions: list[str]
    ) -> int:
        score = 7
        import re

        # 相同句式开头
        if len(sentences) >= 3:
            starters = [s[:6] for s in sentences if len(s) >= 6]
            unique_starters = len(set(starters))
            if unique_starters < len(starters) * 0.5:
                score -= 3

        # 三元并列对称
        triple_patterns = [
            r"(首先|第一).{0,60}(其次|第二).{0,60}(再次|第三|最后)",
            r"([\u4e00-\u9fff]{1,6}(上|层面|维度)).{5,60};.{5,60};",
        ]
        triple_count = sum(len(re.findall(p, text)) for p in triple_patterns)
        score -= min(triple_count * 2, 4)

        if score < 10:
            suggestions.append(f"抗检测性: {score}/10 — 句式结构模式化，建议打破对称性、增加随机波动")
        return max(score, 3)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """简单的句子切分（按 。！？ 分割）。"""
        import re

        parts = re.split(r"[。！？\n]+", text)
        return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_default_ttr = TtrEvaluator()
_default_qa = QualityAssessor()


def quick_ttr(text: str, threshold: float = 0.45) -> TtrResult:
    return _default_ttr.evaluate(text)


def quick_quality(text: str) -> QualityScore:
    return _default_qa.assess(text)
