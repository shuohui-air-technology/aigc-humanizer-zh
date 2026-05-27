"""
LaTeX 栈式扫描器
================
基于栈的轻量级 LaTeX 公式/环境提取器。不使用简单正则，逐字符扫描，
正确处理嵌套环境、行内公式、区块公式的交叉嵌套。

提供：
  - LatexScanner 类：单次扫描，生成占位符映射。
  - mask / restore 两步式工作流。
"""

from __future__ import annotations

import hashlib
from typing import Optional


class LatexScanner:
    """LaTeX 公式扫描与占位符管理。

    用法:
        scanner = LatexScanner()
        masked, count = scanner.mask(text)   # 公式 → 占位符
        # ... 对 masked 做润色 ...
        restored = scanner.restore(masked)   # 占位符 → 公式

    占位符格式:
        [INLINE_MATH_a1b2c3]   行内公式 $...$
        [BLOCK_MATH_d4e5f6]    区块公式 $$...$$
        [ENV_g7h8i9]           LaTeX 环境 \\begin{...}...\\end{...}
    """

    def __init__(self) -> None:
        self._vault: dict[str, str] = {}
        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def mask(self, text: str) -> tuple[str, int]:
        """提取文本中所有 LaTeX 公式/环境，替换为占位符。

        Returns:
            (masked_text, count): 替换后的文本和提取到的公式总数。
        """
        self._vault.clear()
        self._warnings.clear()

        try:
            masked, vault, warnings = self._scan(text)
            self._vault = vault
            self._warnings = warnings
            return masked, len(vault)
        except Exception:
            # 扫描异常时原样返回
            return text, 0

    def restore(self, masked_text: str) -> str:
        """将占位符替换回原始 LaTeX 公式。"""
        result = masked_text
        for placeholder, original in self._vault.items():
            result = result.replace(placeholder, original)
        return result

    @property
    def warnings(self) -> list[str]:
        """扫描过程中产生的警告（如未闭合标记）。"""
        return list(self._warnings)

    @property
    def vault_size(self) -> int:
        """当前 vault 中的占位符数量。"""
        return len(self._vault)

    # ------------------------------------------------------------------
    # 核心扫描器
    # ------------------------------------------------------------------

    def _scan(self, text: str) -> tuple[str, dict[str, str], list[str]]:
        """基于栈的逐字符 LaTeX 扫描器。

        栈帧: (type: str, start: int, env_name: str | None)
          type ∈ {'inline', 'block', 'env'}

        模式切换:
          - 栈空（普通文本）: $$/ $/ \\begin{name} 分别压入对应帧
          - inline 模式: 仅 $ 闭合
          - block  模式: 仅 $$ 闭合
          - env    模式: \\end{name} 闭合、\\begin{name} 嵌套、
                        $$/$ 开启环境内公式
        """
        n = len(text)
        vault: dict[str, str] = {}
        output: list[str] = []
        stack: list[tuple[str, int, Optional[str]]] = []
        warnings: list[str] = []

        i = 0
        text_start = 0

        while i < n:
            if not stack:
                # ---------- 普通文本模式 ----------
                if text[i : i + 2] == "$$":
                    if text_start < i:
                        output.append(text[text_start:i])
                    stack.append(("block", i, None))
                    text_start = i
                    i += 2
                elif text[i] == "$":
                    if text_start < i:
                        output.append(text[text_start:i])
                    stack.append(("inline", i, None))
                    text_start = i
                    i += 1
                elif text[i : i + 7] == "\\begin{":
                    try:
                        j = text.index("}", i + 7)
                    except ValueError:
                        i += 1
                        continue
                    env_name = text[i + 7 : j]
                    if text_start < i:
                        output.append(text[text_start:i])
                    stack.append(("env", i, env_name))
                    text_start = i
                    i = j + 1
                else:
                    i += 1

            elif stack[-1][0] == "inline":
                # ---------- 行内公式 ----------
                if text[i] == "$":
                    start = stack.pop()[1]
                    latex = text[start : i + 1]
                    ph = self._make_placeholder("INLINE_MATH", latex, vault)
                    output.append(ph)
                    text_start = i + 1
                i += 1

            elif stack[-1][0] == "block":
                # ---------- 区块公式 ----------
                if text[i : i + 2] == "$$":
                    start = stack.pop()[1]
                    latex = text[start : i + 2]
                    ph = self._make_placeholder("BLOCK_MATH", latex, vault)
                    output.append(ph)
                    text_start = i + 2
                    i += 2
                else:
                    i += 1

            else:  # stack[-1][0] == 'env'
                # ---------- LaTeX 环境 ----------
                if text[i : i + 5] == "\\end{":
                    try:
                        j = text.index("}", i + 5)
                    except ValueError:
                        i += 1
                        continue
                    env_name = text[i + 5 : j]
                    if env_name == stack[-1][2]:
                        start = stack.pop()[1]
                        latex = text[start : j + 1]
                        ph = self._make_placeholder("ENV", latex, vault)
                        output.append(ph)
                        text_start = j + 1
                        i = j + 1
                    else:
                        i += 1
                elif text[i : i + 7] == "\\begin{":
                    try:
                        j = text.index("}", i + 7)
                    except ValueError:
                        i += 1
                        continue
                    sub_env = text[i + 7 : j]
                    stack.append(("env", i, sub_env))
                    i = j + 1
                elif text[i : i + 2] == "$$":
                    stack.append(("block", i, None))
                    i += 2
                elif text[i] == "$":
                    stack.append(("inline", i, None))
                    i += 1
                else:
                    i += 1

        # 追加末尾普通文本
        if text_start < n:
            output.append(text[text_start:])

        # 检测未闭合标记
        while stack:
            frame = stack.pop()
            warnings.append(f"未闭合的 {frame[0]} 标记，起始位置 {frame[1]}")

        return "".join(output), vault, warnings

    # ------------------------------------------------------------------
    # 占位符生成
    # ------------------------------------------------------------------

    @staticmethod
    def _make_placeholder(
        prefix: str, content: str, vault: dict[str, str]
    ) -> str:
        h = hashlib.md5(content.encode("utf-8")).hexdigest()[:6]
        placeholder = f"[{prefix}_{h}]"
        if placeholder in vault and vault[placeholder] != content:
            seq = 2
            while True:
                placeholder = f"[{prefix}_{h}_{seq}]"
                if placeholder not in vault or vault[placeholder] == content:
                    break
                seq += 1
        vault[placeholder] = content
        return placeholder
