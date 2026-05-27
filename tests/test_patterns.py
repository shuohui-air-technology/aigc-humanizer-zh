import pytest

from src.patterns import PatternDetector, _split_paragraphs


def pattern_ids(text: str) -> set[int]:
    report = PatternDetector().analyze(text)
    return {
        match.pattern_id
        for paragraph in report.paragraph_risks
        for match in paragraph.matches
    }


@pytest.mark.parametrize(
    ("text", "expected_id"),
    [
        ("依据框架理论，知识是在互动中形成的。", 1),
        ("译者最终选择意译。此案例印证了目的论的解释力。", 2),
        ("首先，数据有限；其次，标注成本高；再次，算法偏差尚未解决。", 3),
        ("该处理体现了研究设计对样本差异的回应。", 4),
        ("研究者面临的核心问题是如何保证跨文化样本可比。", 5),
        ("理论上，模型解释充分；实践上，数据可获得；方法上，路径清晰。", 6),
        ("前文已经说明机制。综上所述，非正式网络仍然关键。", 7),
        ("专家认为，平台机制会改变用户参与方式。", 8),
        ("值得注意的是，数据显示两组存在差异。", 9),
        ("本研究具有重要的理论意义。", 10),
        ("该结果深刻揭示了平台治理的复杂性。", 11),
        ("社交媒体作为政治动员的重要载体影响了信息流动。", 12),
        ("突破范式，填补空白，创新视角，丰富方法，完善体系，", 13),
        ("从经济维度看成本较高，从社会维度看扩散较快，从文化维度看阻力明显。", 14),
        ("这——不是——一个——普通——问题。", 15),
        ("本段 **重点** 与 **问题** 都被加粗。", 16),
    ],
)
def test_all_documented_patterns_have_positive_examples(
    text: str, expected_id: int
) -> None:
    assert expected_id in pattern_ids(text)


def test_soft_newlines_are_merged_before_blank_line_split() -> None:
    text = "第一行仍属同一段\n第二行只是排版换行\n\n第二段开始。"

    assert _split_paragraphs(text) == ["第一行仍属同一段第二行只是排版换行", "第二段开始。"]


def test_long_unstructured_text_falls_back_to_sentence_split() -> None:
    text = "这是一个较长的句子，用来模拟没有空行的论文复制文本。" * 16

    paragraphs = _split_paragraphs(text)

    assert len(paragraphs) > 2


def test_vague_attribution_ignores_specific_author_year_citation() -> None:
    ids = pattern_ids("Boulianne（2015）研究表明，社交媒体使用与政治参与相关。")

    assert 8 not in ids


def test_vague_attribution_ignores_self_attribution() -> None:
    ids = pattern_ids("本研究表明，平台机制会影响内容扩散。")

    assert 8 not in ids


def test_closing_cliche_must_be_in_final_sentence() -> None:
    ids = pattern_ids("可以看出，两组差异明显。接下来本文讨论样本限制。")

    assert 7 not in ids


def test_repeated_same_pattern_keeps_evidence_but_caps_score() -> None:
    text = "深刻揭示了，深刻揭示了，深刻揭示了，深刻揭示了，深刻揭示了。"
    report = PatternDetector().analyze(text)
    paragraph = report.paragraph_risks[0]

    assert len([m for m in paragraph.matches if m.pattern_id == 11]) == 5
    assert paragraph.score_raw > paragraph.score
    assert paragraph.score <= paragraph.max_score


def test_low_risk_human_sample_stays_low() -> None:
    report = PatternDetector().analyze(
        "访谈记录显示，两位受访者在具体流程上存在分歧。"
        "这个差异后来影响了编码结果，我们在复核时保留了原始标注。"
    )

    assert report.paragraph_risks[0].risk_level == "🟢 低风险"


def test_all_hard_constraints_emit_structured_details() -> None:
    text = (
        "依据框架理论，该结果深刻揭示了治理机制，也不可或缺，并综合运用多种方法。\n\n"
        "基于制度理论，样本差异可以解释为组织约束的结果。\n\n"
        "理论上，模型解释充分；实践上，数据可获得；方法上，路径清晰。"
        "首先，样本明确；其次，变量清楚；再次，路径稳定。\n\n"
        "材料显示差异明显。由此可见，平台机制仍然关键。\n\n"
        "访谈结果存在分歧。综上所述，非正式网络仍然关键。\n\n"
        "**一** **二** **三** **四** **五** **六**\n\n"
        "本研究具有重要意义。专家认为，该机制仍会持续。"
    )

    report = PatternDetector().analyze(text)
    constraints = {item["constraint"] for item in report.hard_violations}

    assert constraints == {"HC-1", "HC-2", "HC-3", "HC-4", "HC-5", "HC-6", "HC-7"}
    for violation in report.hard_violations:
        assert "paragraph" in violation
        assert "pattern_ids" in violation
        assert "reason" in violation
        assert violation["items"]
        for item in violation["items"]:
            assert "paragraph" in item
            assert "pattern_id" in item
            assert "matched_text" in item
            assert "reason" in item
