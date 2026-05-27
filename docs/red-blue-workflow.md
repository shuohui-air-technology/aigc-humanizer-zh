# Red-Blue Evaluation Workflow

This project uses a deterministic red-blue loop to improve rule accuracy:

1. Red team proposes hard examples.
2. Humans review and sanitize the examples.
3. Blue team adjusts rules, scoring, or thresholds.
4. The judge script blocks regressions.

The loop improves a rule engine. It is not model fine-tuning, and it does not
claim to bypass any specific AIGC detector.

## Fixture Format

Red-blue fixtures live in `tests/fixtures/red_blue/*.jsonl`. Each line is one
case:

```json
{
  "id": "p08_pos_01",
  "case_type": "positive",
  "domain": "media",
  "style": "期刊论文",
  "text": "专家认为，社交媒体平台会显著改变用户的政治参与方式。",
  "expected_patterns": [8],
  "expected_hard_constraints": ["HC-7"],
  "expected_risk_level": "high",
  "adversary_note": "无出处专家认为。"
}
```

Required fields:

- `id`: stable unique case id.
- `text`: anonymized synthetic or fully sanitized text.
- `expected_patterns`: pattern IDs expected to fire.
- `expected_hard_constraints`: hard constraints expected to fire.
- `expected_risk_level`: `low`, `medium`, or `high`.
- `case_type`: `positive`, `negative`, `near_miss`, or `adversarial`.

For negative and near-miss cases, add `negative_patterns` and
`negative_hard_constraints` when the case targets specific non-matches.

## Red Team Prompt

Use this prompt with any LLM provider. Do not paste private papers or personal
data. Generate candidates, then review them before committing fixtures.

```text
你是 aigc-humanizer-zh 的红队样例生成器。

目标：为一个中文学术 AI 写作模式检测器生成匿名合成测试样例。
不要使用真实论文、真实作者、真实机构或隐私数据。

请围绕以下目标模式生成 JSONL 候选：
- 目标模式 ID: {pattern_id}
- 模式名称: {pattern_name}
- 文体: 期刊论文 / 毕业论文 / 研究报告 / 学术博客
- 领域: 教育、传播、组织管理、公共政策、方法论等

每条候选必须包含：
id, case_type, domain, style, text,
expected_patterns, expected_hard_constraints, expected_risk_level,
adversary_note

请输出三类样例：
1. positive：应明确命中目标模式。
2. near_miss：表面相似但不应命中目标模式。
3. adversarial：故意伪装、组合或边界化的难例。

约束：
- 文本必须匿名合成。
- 不新增真实引用或真实数据。
- near_miss 必须解释为什么不应命中。
- 不要输出 Markdown，只输出 JSONL。
```

## Human Review Gate

Before a generated case enters `tests/fixtures/red_blue/`, check:

- The text is synthetic or fully sanitized.
- Expected labels match current project definitions.
- Positive examples are not just duplicates with synonym swaps.
- Near-miss examples are plausible and carry `negative_reason`.
- The fixture does not encode a rule implementation detail as the only reason
  it passes.

## Blue Team Rule

Run the judge before and after changing rules:

```bash
python scripts/evaluate_red_blue.py \
  --fixtures tests/fixtures/red_blue \
  --min-f1 0.70 \
  --max-fpr 0.15
```

Rule changes are acceptable when:

- Existing pytest tests pass.
- Red-blue evaluation passes.
- `negative_false_positive_rate` does not exceed the budget.
- Any intentional F1 drop is explained in the PR or change note.

## Current v0.3 Baseline

The initial synthetic suite covers:

- 16 patterns with at least 3 positive examples each.
- 7 hard constraints with at least 3 positive examples each.
- Negative and near-miss examples covering all patterns and hard constraints.

This is a baseline, not a final benchmark. Add real-world-inspired synthetic
cases whenever a false positive or false negative is discovered.
