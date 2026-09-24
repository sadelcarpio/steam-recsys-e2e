from conftest import body, event

from steam_serving import handler


def test_handler_builds_the_app_once(tables, monkeypatch):
    handler.app.cache_clear()
    monkeypatch.setenv("DEFAULT_LIMIT", "3")
    first = handler.handler(event("/popular"), None)
    assert first["statusCode"] == 200 and len(body(first)["recommendations"]) == 3
    assert handler.app() is handler.app()
    handler.app.cache_clear()
