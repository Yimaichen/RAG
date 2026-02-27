"""Minimal smoke tests - verify package imports work correctly."""


def test_import_arag():
    import arag
    assert hasattr(arag, "__version__")


def test_import_core():
    from arag.core.config import Config
    from arag.core.context import AgentContext
    from arag.core.llm import LLMClient

    assert Config is not None
    assert AgentContext is not None
    assert LLMClient is not None


def test_import_tools():
    from arag.tools.base import BaseTool
    from arag.tools.registry import ToolRegistry

    assert BaseTool is not None
    assert ToolRegistry is not None


def test_import_agent():
    from arag.agent.base import BaseAgent
    assert BaseAgent is not None


def test_context_basic():
    from arag.core.context import AgentContext

    ctx = AgentContext()
    assert ctx.total_retrieved_tokens == 0

    ctx.mark_chunk_as_read("42")
    assert ctx.is_chunk_read("42")
    assert not ctx.is_chunk_read("99")

    ctx.add_retrieval_log("test_tool", tokens=100)
    assert ctx.total_retrieved_tokens == 100

    summary = ctx.get_summary()
    assert summary["chunks_read_count"] == 1


def test_tool_registry():
    from arag.tools.registry import ToolRegistry

    registry = ToolRegistry()
    assert registry.list_tools() == []

    result, log = registry.execute("nonexistent", None)
    assert "Error" in result


def test_config_basic():
    from arag.core.config import Config

    cfg = Config({"llm": {"model": "test", "temperature": 0.5}})
    assert cfg.get("llm.model") == "test"
    assert cfg.get("llm.temperature") == 0.5
    assert cfg.get("llm.missing", "default") == "default"
