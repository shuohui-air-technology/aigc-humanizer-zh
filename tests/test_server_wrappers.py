import importlib
import sys
import types


class DummyFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name

    def tool(self):
        def decorator(func):
            return func

        return decorator

    def run(self) -> None:
        return None


def import_server_with_mcp_stub(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = DummyFastMCP

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def test_analyze_ai_risk_keeps_existing_fields_and_adds_v02_metadata(monkeypatch) -> None:
    server = import_server_with_mcp_stub(monkeypatch)

    result = server.analyze_ai_risk("专家认为，本研究具有重要意义。")

    assert result["rule_version"] == "0.2"
    assert "overall_risk" in result
    assert "paragraph_risks" in result
    paragraph = result["paragraph_risks"][0]
    assert "score" in paragraph
    assert "score_raw" in paragraph
    assert "score_capped" in paragraph
    assert "score_max" in paragraph


def test_analyze_by_paragraph_scores_are_bounded_and_compatible(monkeypatch) -> None:
    server = import_server_with_mcp_stub(monkeypatch)
    text = "深刻揭示了，深刻揭示了，深刻揭示了，深刻揭示了，深刻揭示了。"

    result = server.analyze_by_paragraph(text)

    assert result["overview"]["rule_version"] == "0.2"
    paragraph = result["paragraphs"][0]
    assert 0 <= paragraph["aigc_score"] <= 100
    assert "needs_rewrite" in paragraph
    assert "patterns" in paragraph
    assert "score_raw" in paragraph
    assert "score_capped" in paragraph
    assert "score_max" in paragraph
