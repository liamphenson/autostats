from autostats.core.tools.registry import REGISTRY


def test_every_registered_tool_produces_a_valid_openai_schema():
    tools = REGISTRY.to_openai_tools()
    assert len(tools) >= 20
    for schema in tools:
        assert schema["type"] == "function"
        assert schema["name"]
        assert "parameters" in schema


def test_dispatch_raises_on_unknown_tool(tool_ctx):
    import pytest

    with pytest.raises(KeyError):
        REGISTRY.dispatch(tool_ctx, "not_a_real_tool", {})
