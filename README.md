# AIGC Humanizer ZH

> 中文学术写作 AIGC 率降低工具 — 基于 16 种 AI 写作模式的**检测 → 报告 → 改写 → 验证**完整链路

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rule Version](https://img.shields.io/badge/rules-v0.2-orange.svg)](src/patterns.py)

---

## 这是什么？

`aigc-humanizer-zh` 帮助你在不改动学术观点的前提下，系统性地降低中文学术文本的 AIGC 检测率。它基于真实论文改写实验（AIGC 率从 >50% 降至 11%）归纳的规律，提供两种形态：

| | 完整版（MCP Server） | 轻量版（Skill 文件） |
|---|---|---|
| **文件** | `server.py` + `src/` | `humanizer-zh-light.md` |
| **依赖** | Python 3.10+ / jieba / mcp | 零依赖 |
| **部署** | 配置 MCP 客户端 | 复制到 skills 目录即用 |
| **AI 模式** | 16 种全覆盖，正则引擎自动扫描 | 10 种核心高频，agent 自行判断 |
| **TTR 词汇丰富度** | ✓ jieba 分词自动计算 | ✗ |
| **60 分制评分** | ✓ 6 维度自动打分 | ✗ |
| **LaTeX 保护** | ✓ 栈式扫描器（嵌套安全） | 手动占位符规则 |
| **规则版本** | v0.2（score_raw/capped 追踪） | — |
| **适用场景** | 批量处理、高精度需求 | 快速上手、临时使用 |

---

## Features

### 检测引擎

- **16 种 AI 写作模式** — 理论起笔、段末套路、编号逻辑、被动套话、三元并列、冗余总结、模糊归因、填充短语、泛化结论、AI 高频词、回避「是」、过度排比、三步走、破折号密度、加粗滥用。每种模式独立计分，带 `weight` 和 `score_cap` 上限
- **7 项硬约束** — 高频词密度、段末套句、三元并列、理论起笔占比、加粗过量、泛化结尾、模糊归因，命中即判定高风险
- **上下文感知过滤** — `_passes_location`（段首/段末位置约束）+ `_passes_context_filter`（模糊归因过滤 `本研究表明`/`Boulianne(2015)` 等具体引用）
- **P6/P14 规则分离** — P6 专注 `理论上/实践上/方法上` 对称三元，P14 专注 `从...维度看` 三步走，消除重叠命中
- **LaTeX 栈式扫描器** — 逐字符扫描，正确处理嵌套环境/行内/区块公式，支持 `mask → 润色 → restore` 无损往返
- **60 分制质量评估** — 6 维度评分（直接性、节奏、真实性、信息密度、学术规范、抗检测性），≥54 优秀 / ≥42 良好
- **TTR 词汇丰富度** — jieba 分词计算 Type-Token Ratio，内置违禁词拦截；jieba 不可用时自动回退

### MCP Server

- **8 个 MCP 工具** — `mask_latex` / `evaluate_ttr` / `restore_latex` / `analyze_ai_risk` / `assess_quality` / `generate_rewrite_plan` / `analyze_by_paragraph` / `build_rewrite_prompt`
- **v0.2 元数据** — 所有工具输出含 `rule_version`、`score_raw`、`score_capped`、`score_max`，计分过程完全透明
- **逐段交互式工作流** — 展示每段 AIGC 评分 (0-100)，用户逐段决策是否改写，改写后对比确认
- **结构化改写计划** — 按 SOP 6 步（移位→砍尾→破对称→换词→去模糊→注视角）自动生成优先级排序的改写指导

### 红蓝军评测闭环（v0.3 新增）

- **62 条匿名合成样例** — `tests/fixtures/red_blue/synthetic.jsonl`，覆盖 16 种 pattern 各 ≥3 正例、7 项硬约束各 ≥3 正例、negative/near_miss 全覆盖
- **确定性裁判脚本** — `scripts/evaluate_red_blue.py`，零网络零 LLM 依赖，输出 pattern/HC 级 F1、误报率、覆盖率、分数边界检查
- **回归测试套件** — 22 项 pytest 测试，含 schema 校验、v0.2 元数据字段、红蓝评测阈值（F1≥0.70 / FPR≤0.15）
- **红队 Prompt 模板** — 见 `docs/red-blue-workflow.md`，可直接用 LLM 生成正例/near_miss/adversarial 候选样例

### 轻量版 Skill

- **零依赖** — `humanizer-zh-light.md`，248 行纯文本，复制到 skills 目录即用
- **10 种高频模式** — 含触发词、改写示例、替换词表
- **6 步改写 SOP** — 移位→砍尾→破对称→换词→去模糊→注视角
- **学者视角注入** — 4 种去「机器稿」方法：承认局限、表达意外、留下判断、短句造节奏
- **噪声保留原则** — 每千字保留 2-3 处轻微 AI 特征，避免过度均质化

---

## 快速开始

### 完整版（MCP Server）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
.venv/bin/pytest                  # 单元测试
python scripts/evaluate_red_blue.py \
  --fixtures tests/fixtures/red_blue \
  --min-f1 0.70 --max-fpr 0.15    # 红蓝评测

# 3. 启动服务（stdio 传输，供 MCP 客户端调用）
python server.py
```

在 DeepSeek TUI 的 `~/.deepseek/config.toml` 中配置：

```toml
[mcp_servers.aigc-humanizer]
command = "python"
args = ["/path/to/aigc-humanizer-zh/server.py"]
```

### 轻量版（Skill 文件）

```
/skill https://raw.githubusercontent.com/shuohui-air-technology/aigc-humanizer-zh/main/humanizer-zh-light.md
```

加载后 agent 即获得「逐段扫描 AI 痕迹 → 交互式改写 → 自检输出」的能力。

---

## MCP 工具速查

| 工具 | 功能 | 关键输出 |
|---|---|---|
| `mask_latex` | LaTeX 公式 → 占位符（栈式扫描） | `masked_text`, `count`, `warnings` |
| `evaluate_ttr` | TTR 词汇丰富度 + 违禁词拦截 | `passed`, `ttr_score`, `banned_words_found` |
| `restore_latex` | 占位符 → 原始 LaTeX 公式 | 还原后完整文本 |
| `analyze_ai_risk` | 16 种模式 + 7 项硬约束 | `rule_version`, `score_raw/capped/max`, `hard_violations` |
| `assess_quality` | 6 维度 60 分制评分 | `total_score`, `grade`, `dimensions` |
| `generate_rewrite_plan` | SOP 6 步改写计划 | `rewrite_plan` (优先级排序) |
| `analyze_by_paragraph` | 逐段 AIGC 评分 (0-100) | `aigc_score`, `needs_rewrite`, `patterns` |
| `build_rewrite_prompt` | 单段 LLM 改写 prompt | 结构化 prompt（可直接喂给 LLM） |

---

## 典型工作流

```
原始文本
    │
    ▼ mask_latex()
骨架文本（公式→占位符）
    │
    ▼ analyze_by_paragraph()
逐段 AIGC 评分 (0-100)
    │
    ├── 段1: 12/100 🟢 → 跳过
    ├── 段2: 25/100 🟡 → 用户决定是否改写
    ├── 段3: 50/100 🔴 → build_rewrite_prompt() → LLM 改写 → 用户确认替换
    └── ...
    │
    ▼ assess_quality()
60 分制验证（≥54 优秀）
    │
    ▼ restore_latex()
最终文本（公式已还原）
```

---

## 16 种 AI 写作模式

| ID | 模式 | 严重度 | 触发示例 | 完整版 | 轻量版 |
|----|------|--------|----------|:---:|:---:|
| 1 | 理论起笔 | 🔴 高 | 「依据社会建构主义理论……」 | ✓ | ✓ |
| 2 | 段末套路结尾 | 🔴 高 | 「此案例印证了……」 | ✓ | ✓ |
| 3 | 整齐编号逻辑 | 🟡 中 | 「首先……其次……再次……」 | ✓ | ✓ |
| 4 | 被动分析套话 | 🔴 高 | 「该处理体现了……」 | ✓ | ✓ |
| 5 | 模板化问题陈述 | 🟡 中 | 「面临的核心问题是……」 | ✓ | — |
| 6 | 三元并列对称 | 🟡 中 | 「理论上……实践上……方法上……」 | ✓ | ✓ |
| 7 | 段末冗余总结 | 🔴 高 | 「综上所述……」 | ✓ | ✓ |
| 8 | 模糊归因 | 🔴 高 | 「专家认为……」（无出处） | ✓ | ✓ |
| 9 | 填充短语 | 🟢 低 | 「值得注意的是……」 | ✓ | ✓ |
| 10 | 泛化结论 | 🔴 高 | 「具有重要意义……」 | ✓ | ✓ |
| 11 | AI 高频词 | 🟡 中 | 「深刻揭示」「不可或缺」 | ✓ | ✓ |
| 12 | 回避系动词「是」 | 🟡 中 | 「作为……重要载体」 | ✓ | — |
| 13 | 过度排比 | 🟡 中 | 「突破范式，填补空白……」 | ✓ | — |
| 14 | 三步走结构 | 🟡 中 | 「从经济维度……社会维度……文化维度……」 | ✓ | — |
| 15 | 破折号密度异常 | 🟢 低 | 一段内 —— 超过 4 次 | ✓ | — |
| 16 | 正文加粗滥用 | 🟢 低 | 全文 ** 超过 5 处 | ✓ | — |

> P6（三元对称）和 P14（三步走）在 v0.2 中已分离，不再重叠命中。

---

## 7 项硬约束

| # | 约束 | 阈值 | 违规后果 |
|---|------|------|----------|
| HC-1 | AI 高频词密度 | 每段 > 2 个 | 高风险 |
| HC-2 | 段末总结套句 | 全文 > 1 处 | 高风险 |
| HC-3 | 整齐三元并列 | 每段 > 1 处 | 高风险 |
| HC-4 | 理论起笔占比 | > 20% 段落 | 高风险 |
| HC-5 | 正文加粗 | 全文 > 5 处 | 高风险 |
| HC-6 | 泛化结尾 | 全文 > 0 处 | 高风险 |
| HC-7 | 模糊归因 | 全文 > 0 处 | 高风险 |

---

## 红蓝军评测闭环

本项目使用确定性的红蓝对抗循环来持续提升规则精度：

```
红队生成样例 → 人工审核 → 蓝队调规则 → 裁判脚本阻断回归
```

```bash
# 运行红蓝评测（当前 baseline: F1 ≥ 0.70, 误报率 ≤ 0.15）
python scripts/evaluate_red_blue.py \
  --fixtures tests/fixtures/red_blue \
  --min-f1 0.70 --max-fpr 0.15
```

当前 v0.3 baseline：62 条合成样例，16 种 pattern 全覆盖，negative/near_miss 误报率 0.00。

详细工作流（含红队 LLM prompt 模板）见 [`docs/red-blue-workflow.md`](docs/red-blue-workflow.md)。

---

## 工程结构

```
aigc-humanizer-zh/
├── src/
│   ├── patterns.py          # 16 种 AI 模式 + 7 项硬约束（841 行，v0.2）
│   ├── scanner.py           # LaTeX 栈式扫描器（219 行）
│   └── evaluator.py         # TTR + 60 分制质量评估（404 行）
├── server.py                # MCP Server 入口，8 个 @mcp.tool()
├── scripts/
│   └── evaluate_red_blue.py # 红蓝军评测裁判脚本（344 行）
├── tests/
│   ├── test_patterns.py     # 16 模式全覆盖 + 精度过滤测试
│   ├── test_scanner.py      # LaTeX 扫描器测试
│   ├── test_evaluator.py    # TTR 评估器测试
│   ├── test_server_wrappers.py  # v0.2 元数据字段测试
│   ├── test_red_blue_evaluator.py  # 红蓝评测回归测试
│   └── fixtures/red_blue/
│       └── synthetic.jsonl  # 62 条匿名合成样例
├── docs/
│   └── red-blue-workflow.md # 红蓝工作流 + 红队 prompt 模板
├── humanizer-zh-light.md    # 轻量版 Skill 文件（248 行，零依赖）
├── SKILL.md                 # 完整版 Skill 参考（554 行）
├── WORKFLOW.md              # 交互式工作流 + Agent Prompt 模板
├── pyproject.toml
├── requirements.txt         # mcp + jieba
└── LICENSE                  # MIT
```

---

## 参考来源

- **真实论文对比实验** — 同一论文 AI 润色版（AIGC >50%）与人工二次修改版（AIGC 11%）的逐段差异分析
- **Wikipedia: Signs of AI writing** — WikiProject AI Cleanup 维护的英文 AI 写作模式分类框架
- **humanizer-zh-academic skill** — 16 种 AI 模式、7 项硬约束、60 分制评估
- **de-AI-writing skill (OUBIGFA)** — 硬约束数字化设计参考
- **Humanizer-zh (op7418)** — 模式分类与质量评分框架参考

> 思路很简单：AI 写出来的东西太「整齐」了。真正的人写论文，句子有长有短，有时候多说一句，有时候一笔带过，偶尔还会犹豫一下。这个项目要做的，就是帮你在不伤学术内容的前提下，把这些自然的波动找回来。

---

## 许可证

MIT
