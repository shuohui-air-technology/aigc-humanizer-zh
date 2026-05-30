# AIGC Humanizer ZH

> 中文学术写作 AIGC 率降低工具。基于 16 种 AI 写作模式，不改动学术观点，只打破机器写作的模式规律。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)](pyproject.toml)

---

## 这是什么？

真实论文改写实验中，同一篇文章 AI 润色版的 AIGC 检测率超过 50%，经过系统性人工去味处理后降至 11%。差异不在于内容——观点没变、论证没变、数据没变——只在于 AI 写作有高度可预测的模式规律，而人类写作更随机、更情境化。

本项目把这种「去味」经验归纳为 **16 种 AI 写作模式**和 **7 项硬约束**，提供两种使用方式：

---

## 方式一：MCP Server（完整版）

适合已配置 DeepSeek TUI / Claude Desktop MCP 的用户，需要自动检测和高精度分析。

### 安装与启动

```bash
pip install -r requirements.txt   # mcp + jieba
python server.py                  # stdio 传输，供 MCP 客户端调用
```

在 `~/.deepseek/config.toml` 中配置：

```toml
[mcp_servers.aigc-humanizer]
command = "python"
args = ["/path/to/aigc-humanizer-zh/server.py"]
```

### 8 个 MCP 工具

工具覆盖「保护公式 → 扫描痕迹 → 逐段决策 → 执行改写 → 验证质量 → 还原公式」的完整链路：

| 工具 | 做什么 | 输出什么 |
|---|---|---|
| `mask_latex` | 把 LaTeX 公式替换为占位符，防止润色时被篡改 | `masked_text`, `count`, `warnings` |
| `evaluate_ttr` | jieba 分词计算词汇丰富度 + 扫描违禁词 | `passed`, `ttr_score`, `banned_words_found` |
| `analyze_ai_risk` | 16 种模式 + 7 项硬约束一次扫描 | `rule_version`, `score_raw/capped/max`, `hard_violations` |
| `generate_rewrite_plan` | 按 SOP 6 步生成优先级排序的改写计划 | `rewrite_plan`（移位→砍尾→破对称→换词→去模糊→注视角） |
| `analyze_by_paragraph` | 逐段输出 AIGC 风险评分 (0-100) | `aigc_score`, `needs_rewrite`, 命中模式详情 |
| `build_rewrite_prompt` | 为单段生成可直接喂给 LLM 的改写 prompt | 结构化 prompt（含原文、模式、文体约束） |
| `assess_quality` | 6 维度 60 分制质量评分 | `total_score`, `grade`, 各维度分 |
| `restore_latex` | 把占位符还原为原始 LaTeX 公式 | 还原后完整文本 |

### 典型工作流

```
mask_latex → analyze_by_paragraph → 用户逐段决策 →
  ├── 低风险段: 跳过
  └── 高风险段: build_rewrite_prompt → LLM 改写 → 用户确认替换
→ assess_quality（≥54 优秀） → restore_latex → 完成
```

### 检测引擎特点

- **16 种模式**全部由正则引擎自动扫描，每种模式独立计分（`weight` + `score_cap` 上限），输出 `score_raw`（原始分）和 `score_capped`（封顶分），计分过程透明
- **7 项硬约束**由代码自动评估，命中即判定高风险
- **上下文感知过滤**：P8 模糊归因自动排除 `Boulianne(2015)研究表明`、`本研究表明` 等具体引用；P2/P7 段末套句只在段落末尾句生效
- **P6/P14 规则分离**：三元对称并列（`理论上/实践上/方法上`）和三步走结构（`从经济维度看/从社会维度看/从文化维度看`）不再重叠命中
- **红蓝军评测闭环**：62 条合成样例的回归测试套件，`scripts/evaluate_red_blue.py` 输出 pattern/HC 级 F1 和误报率

### 运行测试

```bash
.venv/bin/pytest                                           # 22 项单元测试
python scripts/evaluate_red_blue.py \
  --fixtures tests/fixtures/red_blue \
  --min-f1 0.70 --max-fpr 0.15                              # 红蓝评测
```

---

## 方式二：Skill 文件

如果你不想安装 Python 依赖，或者只是临时处理一两篇论文，可以直接加载 Skill 文件让 agent 按规则执行。

### 两种 Skill

| 文件 | 定位 | 行数 | 特点 |
|------|------|------|------|
| `SKILL.md` | **轻量版**（默认） | 410 | 16 种模式速查 + HC-1~HC-7 + 逐段改写流程，零依赖，加载即用 |
| `SKILL_full.md` | **完整版** | 623 | 轻量版全部内容 + LaTeX 公式保护/还原 + TTR 词汇丰富度自检 + 结构化逐段输出模板 + MCP 工具映射 |

轻量版已作为默认 Skill 文件，直接加载即可。如需完整版能力：

### 加载

```
/skill https://raw.githubusercontent.com/shuohui-air-technology/aigc-humanizer-zh/main/SKILL.md
```

加载后 agent 即获得：逐段扫描 → 交互式改写 → 最终自检 的完整能力。

如需完整版（含 LaTeX 处理 + TTR + 结构化模板）：

```
/skill https://raw.githubusercontent.com/shuohui-air-technology/aigc-humanizer-zh/main/SKILL_full.md
```

### 能做什么

- **识别全部 16 种 AI 写作模式** — 与 MCP Server 共享同一套模式编号和改写规则。其中 12 种通过阅读直接识别，4 种统计类模式（排比、三步走、破折号密度、加粗滥用）标注 ⚡ 需你自主判断
- **执行 7 项硬约束** — 改写后逐项核查，命中即修复
- **6 步 SOP 改写流程** — 移位→砍尾→破对称→换词→去模糊→注视角
- **注入学者视角** — 承认局限、表达意外、留下判断、短句造节奏
- **质量自评** — 6 维度自主判断（直接性、节奏、真实性、信息密度、学术规范、抗检测性）
- **噪声保留** — 每千字保留 2-3 处轻微 AI 特征，避免过度均质化

### 轻量版与完整版的差异

轻量版（`SKILL.md`）阉割了完整版（`SKILL_full.md`）的部分能力：TTR 词汇丰富度判断标准、LaTeX 公式手动保护/还原流程、结构化逐段输出模板。4 种统计类模式（P13-P16）需你逐段手动统计。但核心的**逐段交互式改写流程**和**16 种模式的改写规则**完整保留。

与 MCP Server 相比，两个 Skill 版本都无法自动执行 TTR 计算和正则引擎扫描，但完整版提供了更接近 MCP 工具链路的手动操作指南。

---

## 检测能力参考

### 16 种 AI 写作模式

| ID | 模式 | 严重度 | 典型触发 |
|----|------|--------|----------|
| 1 | 理论起笔 | 🔴 高 | 「依据社会建构主义理论……」 |
| 2 | 段末套路结尾 | 🔴 高 | 「此案例印证了……」 |
| 3 | 整齐编号逻辑 | 🟡 中 | 「首先……其次……再次……」 |
| 4 | 被动分析套话 | 🔴 高 | 「该处理体现了……」 |
| 5 | 模板化问题陈述 | 🟡 中 | 「面临的核心问题是……」 |
| 6 | 三元并列对称 | 🟡 中 | 「理论上……实践上……方法上……」 |
| 7 | 段末冗余总结 | 🔴 高 | 「综上所述……由此可见……」 |
| 8 | 模糊归因 | 🔴 高 | 「专家认为……」（无出处） |
| 9 | 填充短语与过度限定 | 🟢 低 | 「值得注意的是……」「可能在一定程度上……」 |
| 10 | 泛化结论与意义声明 | 🔴 高 | 「具有重要意义……前景广阔……」 |
| 11 | AI 高频词汇 | 🟡 中 | 「深刻揭示」「不可或缺」「综合运用」 |
| 12 | 回避系动词「是」 | 🟡 中 | 「作为……重要载体」「扮演着……角色」 |
| 13 | 过度对仗排比 | 🟡 中 | 「突破范式，填补空白，创新视角……」 |
| 14 | 结构性三步走 | 🟡 中 | 「从经济维度……社会维度……文化维度……」 |
| 15 | 破折号密度异常 | 🟢 低 | 一段内 —— 超过 4 次 |
| 16 | 正文加粗滥用 | 🟢 低 | 全文 ** 超过 5 处 |

### 7 项硬约束

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

项目使用确定性的红蓝对抗循环来持续提升规则精度：

```
红队生成样例 → 人工审核 → 蓝队调规则 → 裁判脚本阻断回归
```

```bash
python scripts/evaluate_red_blue.py \
  --fixtures tests/fixtures/red_blue \
  --min-f1 0.70 --max-fpr 0.15
```

当前 baseline：62 条匿名合成样例，16 种 pattern 全覆盖，negative/near_miss 误报率 0.00。

详细工作流（含红队 LLM prompt 模板）见 [`docs/red-blue-workflow.md`](docs/red-blue-workflow.md)。

---

## 工程结构

```
aigc-humanizer-zh/
├── src/
│   ├── patterns.py            # 16 种 AI 模式 + 7 项硬约束（841 行）
│   ├── scanner.py             # LaTeX 栈式扫描器（219 行）
│   └── evaluator.py           # TTR + 60 分制质量评估（404 行）
├── server.py                  # MCP Server 入口，8 个 @mcp.tool()
├── scripts/
│   └── evaluate_red_blue.py   # 红蓝军评测裁判脚本（344 行）
├── tests/                     # 6 个测试文件 + 62 条合成样例
├── docs/
│   └── red-blue-workflow.md   # 红蓝工作流文档
├── SKILL.md                   # 轻量版 Skill（410 行，默认加载）
├── SKILL_full.md              # 完整版 Skill（623 行，含 LaTeX/TTR/结构化模板）
├── WORKFLOW.md                # 交互式工作流 Agent Prompt 模板
├── pyproject.toml             # Python 工程配置
├── requirements.txt           # mcp + jieba
└── LICENSE                    # MIT
```

---

## 参考来源

- 真实论文 AI 润色版（AIGC >50%）与人工改写版（AIGC 11%）的逐段对比实验
- Wikipedia: Signs of AI writing（WikiProject AI Cleanup）
- de-AI-writing skill (OUBIGFA) — 硬约束数字化设计参考
- Humanizer-zh (op7418) — 模式分类与质量评分框架参考

---

## 注意事项与局限性

### 检测模式的适用范围

本项目的 16 种 AI 写作模式主要针对 **GPT-3.5/4 等早期模型的典型输出风格**，这些特征也是目前主流 AIGC 检测工具的核心判据。随着模型迭代，新一代 AI 的写作风格趋于多样化，在大部分非模板化的论文中可能不会有效触发这些检查机制。因此：

- **16 种模式适合作为保守的硬性检查机制**，但不应作为唯一的判断依据
- **实际使用中**，大多数场景下加载轻量版 Skill（`SKILL.md`）即可达成类似的改写效果，无需部署完整的 MCP Server 工具链

### 改写可能影响文本质量

改写过程本质上是对原文表达方式的干预。虽然本项目设计了 7 项硬约束来防止破坏性修改，但在以下情况下仍可能出现文本质量下降：

- 连续多次改写导致语言均质化，失去原文的节奏和个性
- LaTeX 公式保护不到位（尤其在 Skill 模式下需手动操作）
- 过度追求「去味」导致表达生硬、学术严谨性受损

**建议在改写前对论文原文进行完整备份**，每轮改写后对比原文审阅，确保学术内容无损。

### 使用声明

本项目**仅作学术交流与技术研究使用**。作者不支持、不鼓励将本工具应用于任何可能影响学术诚信的使用场景，包括但不限于：

- 将他人的 AI 生成文本伪装为人类撰写
- 规避学术机构对 AIGC 内容的合理检测
- 以「去 AI 味」为手段掩盖抄袭或造假行为

工具是中性的，使用者的意图决定其价值。请在学术诚信的边界内使用。

---

## 许可证

MIT
