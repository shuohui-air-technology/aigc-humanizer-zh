from src.evaluator import quick_ttr


def test_quick_ttr_uses_threshold_argument() -> None:
    text = "甲甲甲甲"

    assert quick_ttr(text, threshold=0.9).passed is False
    assert quick_ttr(text, threshold=0.1).passed is True
