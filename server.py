#!/usr/bin/env python3
"""
MCP Server: AIGC Humanizer ZH — 中文学术写作去 AI 味引擎
=========================================================
基于 humanizer-zh-academic skill（16 种 AI 模式、7 项硬约束、60 分制评估），
提供 6 个 MCP 工具供 agent（如 DeepSeek TUI）自动化调用。

工具列表:
  mask_latex           — LaTeX 公式保护（栈式扫描器）
  evaluate_ttr         — TTR 词汇丰富度 + 违禁词审查
  restore_latex        — LaTeX 公式还原
  analyze_ai_risk      — 16 种 AI 模式扫描 + 硬约束评估
  assess_quality       — 6 维度 60 分制质量评分
  generate_rewrite_plan — 基于风险报告生成结构化改写计划

典型工作流:
  1. mask_latex → 保护公式
  2. analyze_ai_risk → 扫描 AI 痕迹
  3. (agent 根据风险报告 + 改写计划执行润色)
  4. assess_quality → 验证效果
  5. restore_latex → 还原公式

用法:
  python server.py          # stdio 传输，由 MCP 客户端调用
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 确保 src 可导入
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from mcp.server.fastmcp import FastMCP

from src.scanner import LatexScanner
from src.patterns import PatternDetector
from src.evaluator import TtrEvaluator, QualityAssessor

# ---------------------------------------------------------------------------
# FastMCP 实例
# ---------------------------------------------------------------------------
mcp = FastMCP("aigc-humanizer-zh")

# ---------------------------------------------------------------------------
# 引擎实例（状态由各工具管理）
# ---------------------------------------------------------------------------
_scanner = LatexScanner()
_detector = PatternDetector()
_ttr = TtrEvaluator(target_threshold=0.45)
_qa = QualityAssessor()


# ======================== 工具 1: mask_latex ================================

@mcp.tool()
def mask_latex(text: str) -> dict[str, Any]:
    """提取文本中所有 LaTeX 公式/环境并用语义占位符替换。

    捕获三种模式：行内公式 $...$、区块公式 $$...$$、LaTeX 环境 \\begin{...}\\end{...}。
    采用栈式逐字符扫描，正确处理嵌套。

    Args:
        text: 包含 LaTeX 公式的中文学术文本。

    Returns:
        dict: masked_text (替换后文本), count (公式数), warnings (扫描警告).
    """
    try:
        masked, count = _scanner.mask(text)
        return {
            "masked_text": masked,
            "count": count,
            "warnings": _scanner.warnings,
            "status": "warning" if _scanner.warnings else "ok",
        }
    except Exception as exc:
        return {
            "masked_text": text,
            "count": 0,
            "warnings": [str(exc)],
            "status": "error",
        }


# ======================== 工具 2: evaluate_ttr ==============================

@mcp.tool()
def evaluate_ttr(text: str, target_threshold: float = 0.45) -> dict[str, Any]:
    """计算文本的 Type-Token Ratio (TTR) 词汇丰富度，并扫描违禁词。

    使用 jieba.lcut 分词，清理标点后计算 TTR = 唯一词元/总词元。
    违禁词库包含：「综上所述」「由此可见」「不可或缺」「具有重要意义」「此案例印证了」。

    Args:
        text: 待评估文本（建议先经 mask_latex 处理）。
        target_threshold: TTR 达标阈值，默认 0.45。

    Returns:
        dict: passed, ttr_score, token_count, unique_tokens, details.
    """
    try:
        _ttr.target_threshold = target_threshold
        result = _ttr.evaluate(text)
        return {
            "passed": result.passed,
            "ttr_score": result.ttr_score,
            "token_count": result.token_count,
            "unique_tokens": result.unique_tokens,
            "banned_words_found": result.banned_words_found,
            "details": result.details,
        }
    except Exception as exc:
        return {
            "passed": False,
            "ttr_score": 0.0,
            "token_count": 0,
            "unique_tokens": 0,
            "banned_words_found": [],
            "details": f"评估异常: {exc}",
        }


# ======================== 工具 3: restore_latex =============================

@mcp.tool()
def restore_latex(masked_text: str) -> str:
    """将 mask_latex 产生的占位符无损还原为原始 LaTeX 公式。

    占位符格式: [INLINE_MATH_xxx] / [BLOCK_MATH_xxx] / [ENV_xxx]。

    Args:
        masked_text: 包含占位符的骨架文本。

    Returns:
        str: 还原原始公式后的完整文本。
    """
    try:
        return _scanner.restore(masked_text)
    except Exception as exc:
        return f"[restore_latex 异常: {exc}]"


# ======================== 工具 4: analyze_ai_risk ===========================

@mcp.tool()
def analyze_ai_risk(text: str) -> dict[str, Any]:
    """扫描文本中的 16 种 AI 写作模式，评估 7 项硬约束，输出风险报告。

    基于 humanizer-zh-academic skill，检测：理论起笔、段末套路、编号逻辑、
    被动套话、模板化问题陈述、三元并列、冗余总结、模糊归因、填充短语、
    泛化结论、AI 高频词、回避「是」、过度排比、三步走、破折号密度、加粗滥用。

    Args:
        text: 中文学术文本。

    Returns:
        dict: overall_risk (高/中/低), hard_violations (硬约束命中), 
              paragraph_risks (每段风险详情), summary (一句话总结).
    """
    try:
        report = _detector.analyze(text)

        para_details = []
        for pr in report.paragraph_risks:
            para_details.append({
                "index": pr.index + 1,
                "prefix": pr.prefix,
                "score": pr.score,
                "risk_level": pr.risk_level,
                "patterns": [
                    {
                        "id": m.pattern_id,
                        "name": m.pattern_name,
                        "severity": m.severity,
                        "location": m.location_hint,
                        "matched": m.matched_text,
                        "suggestion": m.suggestion,
                    }
                    for m in pr.matches
                ],
            })

        return {
            "overall_risk": report.overall_risk,
            "total_paragraphs": report.total_paragraphs,
            "total_score": report.total_score,
            "hard_violations": report.hard_violations,
            "paragraph_risks": para_details,
            "summary": report.summary,
        }
    except Exception as exc:
        return {
            "overall_risk": "评估失败",
            "total_paragraphs": 0,
            "total_score": 0,
            "hard_violations": [],
            "paragraph_risks": [],
            "summary": f"分析异常: {exc}",
        }


# ======================== 工具 5: assess_quality ============================

@mcp.tool()
def assess_quality(text: str) -> dict[str, Any]:
    """对文本执行 6 维度 60 分制综合质量评分。

    维度：直接性、节奏变化、真实性、信息密度、学术规范、抗检测性（各 0-10 分）。

    Args:
        text: 改写后的中文学术文本。

    Returns:
        dict: total_score (/60), grade (优秀/良好/需修订), dimensions (各维度分), suggestions.
    """
    try:
        score = _qa.assess(text)
        return {
            "total_score": score.total,
            "max_score": 60,
            "grade": score.grade,
            "dimensions": {
                "directness": score.directness,
                "rhythm": score.rhythm,
                "authenticity": score.authenticity,
                "information_density": score.information_density,
                "academic_norm": score.academic_norm,
                "anti_detection": score.anti_detection,
            },
            "suggestions": score.suggestions,
        }
    except Exception as exc:
        return {
            "total_score": 0,
            "max_score": 60,
            "grade": "评估失败",
            "dimensions": {},
            "suggestions": [f"评估异常: {exc}"],
        }


# ======================== 工具 6: generate_rewrite_plan =====================

@mcp.tool()
def generate_rewrite_plan(text: str) -> dict[str, Any]:
    """基于 AI 风险扫描结果，生成按优先级排序的结构化改写计划。

    先执行 analyze_ai_risk，再将命中模式按 SKILL.md 的 SOP 组织为：
      1. 移位（理论起笔） 2. 砍尾（段末套句） 3. 破对称（并列/三步走）
      4. 换词（AI 高频词） 5. 去模糊（模糊归因/填充短语） 6. 注视角（学者个性）。

    Args:
        text: 中文学术文本。

    Returns:
        dict: rewrite_plan (按优先级排列的改写步骤), risk_summary (简要风险).
    """
    try:
        report = _detector.analyze(text)

        # 按 SOP 优先级归类
        plan: list[dict[str, Any]] = []

        categories = {
            "移位: 理论名称从段首移到段中": [1],
            "砍尾: 删除段末总结套句": [2, 7],
            "破对称: 打破并列句等长等重结构": [3, 6, 14],
            "换词: 替换 AI 高频词": [11],
            "去模糊: 消除模糊归因与填充短语": [8, 9],
            "注视角: 加入研究者判断与主观表达": [],  # 特殊：无具体模式 ID，由 agent 自主注入
        }

        for category_name, pattern_ids in categories.items():
            items: list[dict[str, Any]] = []
            for pr in report.paragraph_risks:
                for m in pr.matches:
                    if not pattern_ids or m.pattern_id in pattern_ids:
                        items.append({
                            "paragraph": pr.index + 1,
                            "pattern": m.pattern_name,
                            "matched": m.matched_text,
                            "suggestion": m.suggestion,
                        })

            if items or not pattern_ids:
                # 注视角总是添加
                if not pattern_ids:
                    items.append({
                        "paragraph": "全文",
                        "pattern": "写作气质",
                        "matched": "—",
                        "suggestion": "加入「笔者认为」「出乎意料的是」等主观表达；用短句打破长句节奏",
                    })

            if items:
                plan.append({
                    "priority": len(plan) + 1,
                    "category": category_name,
                    "item_count": len(items),
                    "items": items[:8],  # 每类最多展示 8 条
                })

        return {
            "rewrite_plan": plan,
            "risk_summary": {
                "overall_risk": report.overall_risk,
                "total_score": report.total_score,
                "hard_violations_count": len(report.hard_violations),
                "summary": report.summary,
            },
        }
    except Exception as exc:
        return {
            "rewrite_plan": [],
            "risk_summary": {
                "overall_risk": "生成失败",
                "total_score": 0,
                "hard_violations_count": 0,
                "summary": f"异常: {exc}",
            },
        }


# ======================== 工具 7: analyze_by_paragraph ======================

@mcp.tool()
def analyze_by_paragraph(text: str) -> dict[str, Any]:
    """逐段分析 AI 风险，返回适合交互式展示的结构化数据。

    将全文切分为段落，对每段执行 16 种模式扫描，计算归一化 AIGC 评分（0-100）。
    返回层级数据：全文概览 + 每段详情（含命中模式、改写建议、是否需要改写）。

    适合 agent 逐段展示给用户，用户决策后再执行改写。

    Args:
        text: 中文学术文本。

    Returns:
        dict:
            - overview: 全文概览（总段数、高风险段数、整体 AIGC 风险百分比）
            - paragraphs: 每段详情列表，每项包含:
                * index: 段号 (1-based)
                * text_preview: 段落前 80 字
                * aigc_score: 归一化风险评分 (0-100)
                * risk_level: 🔴/🟡/🟢
                * needs_rewrite: bool (aigc_score >= 25 建议改写)
                * pattern_count: 命中模式数
                * patterns: 命中模式详情
                * rewrite_priority: 改写优先级排序
    """
    try:
        report = _detector.analyze(text)
        paragraphs = _split_text_paragraphs(text)

        para_results: list[dict[str, Any]] = []
        high_risk_count = 0

        for pr in report.paragraph_risks:
            # 归一化 AIGC 评分 (0-100)
            max_score = pr.max_score if pr.max_score > 0 else 1
            aigc_score = round((pr.score / max_score) * 100)

            if aigc_score >= 50:
                high_risk_count += 1

            # 整理改写优先级
            rewrite_priority: list[dict[str, Any]] = []
            high_items = [m for m in pr.matches if m.severity == "high"]
            med_items = [m for m in pr.matches if m.severity == "medium"]
            low_items = [m for m in pr.matches if m.severity == "low"]
            for m in high_items + med_items + low_items:
                rewrite_priority.append({
                    "pattern_name": m.pattern_name,
                    "severity": m.severity,
                    "action": m.suggestion,
                })

            para_results.append({
                "index": pr.index + 1,
                "text_preview": (paragraphs[pr.index] if pr.index < len(paragraphs) else pr.prefix)[:80],
                "aigc_score": aigc_score,
                "risk_level": pr.risk_level,
                "needs_rewrite": aigc_score >= 25,
                "pattern_count": len(pr.matches),
                "patterns": [
                    {
                        "id": m.pattern_id,
                        "name": m.pattern_name,
                        "severity": m.severity,
                        "matched": m.matched_text,
                        "suggestion": m.suggestion,
                    }
                    for m in pr.matches
                ],
                "rewrite_priority": rewrite_priority,
            })

        total_paras = len(para_results)
        overall_pct = (
            round((report.total_score / max(total_paras * 16, 1)) * 100)
            if total_paras > 0
            else 0
        )

        return {
            "overview": {
                "total_paragraphs": total_paras,
                "high_risk_paragraphs": high_risk_count,
                "overall_aigc_risk_pct": overall_pct,
                "hard_violations": len(report.hard_violations),
                "summary": report.summary,
            },
            "paragraphs": para_results,
            "hard_violations": report.hard_violations,
        }
    except Exception as exc:
        return {
            "overview": {
                "total_paragraphs": 0,
                "high_risk_paragraphs": 0,
                "overall_aigc_risk_pct": 0,
                "hard_violations": 0,
                "summary": f"分析异常: {exc}",
            },
            "paragraphs": [],
            "hard_violations": [],
        }


# ======================== 工具 8: build_rewrite_prompt ======================

@mcp.tool()
def build_rewrite_prompt(
    paragraph: str,
    pattern_matches: list[dict[str, Any]],
    style: str = "毕业论文",
) -> str:
    """为单段文本生成可直接喂给 LLM 的改写 prompt。

    根据检测到的 AI 模式和文体要求，构建一个包含：原始段落、命中模式说明、
    具体改写约束、改写后输出格式的结构化 prompt。

    agent 可将此 prompt 发送给 LLM 执行改写，改写结果由用户确认后替换原文。

    Args:
        paragraph: 需要改写的单段原文。
        pattern_matches: analyze_by_paragraph 返回的该段 patterns 列表。
        style: 文体类型 — "期刊论文" | "毕业论文" | "研究报告" | "学术博客"。

    Returns:
        str: 可直接使用的改写 prompt。
    """
    # 文体约束映射
    style_guides = {
        "期刊论文": "使用「本文」代替「我」；语域正式但不过度僵硬；破折号密度适中偏少",
        "毕业论文": "可使用「笔者」；可适度口语化；破折号密度适中",
        "研究报告": "使用「本报告/本研究」；语域中低正式度；破折号少用",
        "学术博客": "自由使用「我」；可中高口语化；破折号密度适中",
    }
    style_rule = style_guides.get(style, style_guides["毕业论文"])

    # 构建命中模式摘要
    pattern_notes: list[str] = []
    for m in pattern_matches:
        pattern_notes.append(
            f"- [{m.get('severity', '?')}] {m.get('name', '?')}: "
            f"原文「{m.get('matched', '?')}」→ {m.get('suggestion', '?')}"
        )

    patterns_text = "\n".join(pattern_notes) if pattern_notes else "（未检测到明显 AI 模式）"

    prompt = f"""## 任务：改写以下中文学术段落，降低 AIGC 检测率

### 原始段落
{paragraph}

### 检测到的 AI 模式（需修复）
{patterns_text}

### 改写约束
1. **保留核心意思**：不改变论证逻辑与学术观点，不新增事实、案例、数据。
2. **文体要求**（{style}）：{style_rule}。
3. **打破模式**：
   - 砍掉「此案例印证了」「由此可见」等段末总结套句
   - 将理论名称从段首移到段中
   - 用自然的因果/转折代替整齐编号
   - 加入研究者视角：「笔者认为」「出乎意料的是」「坦率地说」等
   - 用短句打破长句节奏，句长参差有变化
4. **禁止新增**：不可添加原文没有的数据、引用、案例、人名。
5. **禁止过度均质化**：保留 1-2 处轻微口语化表达作为自然噪声。

### 输出格式
只输出改写后的段落文本，不要加任何解释、标记或前后缀。"""

    return prompt


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _split_text_paragraphs(text: str) -> list[str]:
    """将文本按空行切分为段落列表。"""
    import re
    raw = re.split(r"\n{1,3}", text.strip())
    return [p.strip() for p in raw if p.strip()]


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    mcp.run()
