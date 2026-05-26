"""aigc-humanizer-zh — 中文学术写作 AIGC 率降低引擎."""

from src.patterns import PatternDetector, RiskReport
from src.scanner import LatexScanner
from src.evaluator import TtrEvaluator, QualityAssessor

__all__ = [
    "PatternDetector",
    "RiskReport",
    "LatexScanner",
    "TtrEvaluator",
    "QualityAssessor",
]
__version__ = "0.1.0"
