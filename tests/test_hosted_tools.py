from llmkit.hosted_tools import (
    HostedTool,
    WebSearch,
    hosted_tools_for_anthropic,
    hosted_tools_for_gemini,
    hosted_tools_for_openai,
)


def test_web_search_is_hosted_tool():
    ws = WebSearch()
    assert isinstance(ws, HostedTool)


def test_hosted_tools_for_openai():
    result = hosted_tools_for_openai([WebSearch()])
    assert result == [{"type": "web_search_preview"}]


def test_hosted_tools_for_anthropic():
    result = hosted_tools_for_anthropic([WebSearch()])
    assert result == [{"type": "web_search_20250305", "name": "web_search"}]


def test_hosted_tools_for_gemini():
    result = hosted_tools_for_gemini([WebSearch()])
    assert result == [{"google_search": {}}]


def test_empty_list():
    assert hosted_tools_for_openai([]) == []
    assert hosted_tools_for_anthropic([]) == []
    assert hosted_tools_for_gemini([]) == []
