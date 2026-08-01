"""Contract tests for the TUI config/project/settings JSON-RPC domain."""

from __future__ import annotations

import inspect
import threading
import types

import pytest

import tui_gateway.methods_config as methods_config
import tui_gateway.server as server


CONFIG_DOMAIN_METHODS = (
    "config.get",
    "config.set",
    "projects.discover_repos",
    "projects.record_repos",
    "projects.tree",
    "projects.project_sessions",
    "projects.list",
    "projects.get",
    "projects.create",
    "projects.update",
    "projects.add_folder",
    "projects.remove_folder",
    "projects.set_primary",
    "projects.archive",
    "projects.delete",
    "projects.set_active",
    "projects.for_cwd",
    "setup.status",
    "setup.runtime_check",
)


def _rpc(method: str, params: dict | None = None) -> dict:
    resp = server.handle_request(
        {"jsonrpc": "2.0", "id": method, "method": method, "params": params or {}}
    )
    assert resp is not None
    return resp


def test_config_domain_is_explicitly_owned_by_methods_config():
    assert tuple(methods_config.CONFIG_METHODS) == CONFIG_DOMAIN_METHODS

    for name in CONFIG_DOMAIN_METHODS:
        assert name in server._methods
        handler = server._methods[name]
        assert inspect.isfunction(handler)
        assert handler.__module__ == "tui_gateway.methods_config"
        assert handler is methods_config.REGISTERED_METHODS[name]


def test_config_domain_registration_does_not_use_globals_rebinding():
    source = inspect.getsource(methods_config)
    forbidden = ("types.FunctionType", "vars(server)", "__globals__")
    assert not any(token in source for token in forbidden)
    assert all(not isinstance(fn, types.FunctionType) or fn.__module__ == methods_config.__name__ for fn in methods_config.REGISTERED_METHODS.values())


def test_config_set_and_get_theme_roundtrip_through_handle_request(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"tui_theme": "dark"}})

    writes: list[tuple[str, str]] = []
    monkeypatch.setattr(server, "_write_config_key", lambda key, value: writes.append((key, value)))

    set_resp = _rpc("config.set", {"key": "theme", "value": "light"})
    assert set_resp["result"] == {"key": "theme", "value": "light"}
    assert writes == [("display.tui_theme", "light")]

    get_resp = _rpc("config.get", {"key": "theme"})
    assert get_resp["result"] == {"value": "dark"}


def test_concurrent_config_reads_and_writes_keep_request_payloads_local(monkeypatch):
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    writes: list[tuple[str, str]] = []
    lock = threading.Lock()

    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"tui_theme": "auto"}})

    def write_config_key(key: str, value: str) -> None:
        entered.wait(timeout=2)
        release.wait(timeout=2)
        with lock:
            writes.append((key, value))

    monkeypatch.setattr(server, "_write_config_key", write_config_key)

    results: dict[str, dict] = {}

    def run(label: str, value: str) -> None:
        results[label] = _rpc("config.set", {"key": "theme", "value": value})

    threads = [
        threading.Thread(target=run, args=("light", "light")),
        threading.Thread(target=run, args=("dark", "dark")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results == {
        "light": {
            "jsonrpc": "2.0",
            "id": "config.set",
            "result": {"key": "theme", "value": "light"},
        },
        "dark": {
            "jsonrpc": "2.0",
            "id": "config.set",
            "result": {"key": "theme", "value": "dark"},
        },
    }
    assert sorted(writes) == [
        ("display.tui_theme", "dark"),
        ("display.tui_theme", "light"),
    ]


@pytest.mark.parametrize("method", CONFIG_DOMAIN_METHODS)
def test_no_config_domain_handler_compat_aliases_remain_in_server(method: str):
    handler = server._methods[method]
    assert handler.__module__ == methods_config.__name__
