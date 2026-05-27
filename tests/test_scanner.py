from src.scanner import LatexScanner


def test_mask_and_restore_latex_variants() -> None:
    scanner = LatexScanner()
    text = (
        "行内 $x+1$，块公式 $$y=2$$，"
        "\\begin{equation}z=3\\end{equation}"
    )

    masked, count = scanner.mask(text)

    assert count == 3
    assert "[INLINE_MATH_" in masked
    assert "[BLOCK_MATH_" in masked
    assert "[ENV_" in masked
    assert scanner.restore(masked) == text


def test_unclosed_latex_warning_keeps_text() -> None:
    scanner = LatexScanner()

    masked, count = scanner.mask("这里有未闭合公式 $x+1")

    assert count == 0
    assert masked == "这里有未闭合公式 $x+1"
    assert scanner.warnings
