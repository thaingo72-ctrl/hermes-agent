from __future__ import annotations

import threading

import tui_gateway.methods_tools as methods_tools
import tui_gateway.server as srv


MANAGEMENT_METHODS = {
    "browser.manage",
    "plugins.list",
    "plugins.manage",
    "config.show",
    "tools.list",
    "tools.show",
    "tools.configure",
    "toolsets.list",
    "agents.list",
    "cron.manage",
    "learning.frames",
    "learning.detail",
    "learning.delete",
    "learning.edit",
    "skills.manage",
    "skills.reload",
}


def test_management_handlers_are_registered_from_canonical_module():
    for name in MANAGEMENT_METHODS:
        assert srv._methods[name].__module__ == "tui_gateway.methods_management"


def test_extracted_management_handlers_are_not_registered_by_legacy_tools_module():
    legacy_names = {name for name, _handler in methods_tools._registry._pending}

    assert MANAGEMENT_METHODS.isdisjoint(legacy_names)
    assert "shell.exec" not in legacy_names


def test_concurrent_management_requests_keep_response_state_isolated(monkeypatch):
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    responses: dict[str, dict] = {}

    def load_cfg():
        entered.wait(timeout=2)
        release.wait(timeout=2)
        return {"agent": {"max_turns": 120}, "enabled_toolsets": [], "verbose": False}

    def browser_status():
        entered.wait(timeout=2)
        release.wait(timeout=2)
        return "http://127.0.0.1:9222"

    monkeypatch.setattr(srv, "_load_cfg", load_cfg)
    monkeypatch.setattr(srv, "_resolve_model", lambda: "isolated-model")
    monkeypatch.setattr(srv, "_resolve_browser_cdp_url", browser_status)

    def run(name: str, method: str, params: dict) -> None:
        responses[name] = srv.handle_request(
            {"jsonrpc": "2.0", "id": name, "method": method, "params": params}
        )

    threads = [
        threading.Thread(target=run, args=("config", "config.show", {})),
        threading.Thread(
            target=run,
            args=("browser", "browser.manage", {"action": "status"}),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert responses["config"]["id"] == "config"
    assert responses["browser"]["id"] == "browser"
    assert responses["config"]["result"]["sections"][0]["rows"][0] == [
        "Model",
        "isolated-model",
    ]
    assert responses["browser"]["result"] == {
        "connected": True,
        "url": "http://127.0.0.1:9222",
    }
    assert set(responses) == {"config", "browser"}
