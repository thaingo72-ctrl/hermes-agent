from __future__ import annotations

import types

import tui_gateway.methods_complete as complete
import tui_gateway.server as server


def test_completion_handlers_are_registered_as_explicit_completion_callables():
    """Completion RPCs are owned directly by methods_complete, not rebound server globals."""
    expected = {
        "paste.collapse",
        "complete.path",
        "complete.slash",
        "model.options",
        "model.save_key",
        "model.disconnect",
    }

    for name in expected:
        handler = server._methods[name]
        assert isinstance(handler, types.FunctionType)
        assert handler.__module__ == "tui_gateway.methods_complete"
        assert handler.__globals__ is complete.__dict__


def test_completion_helpers_are_owned_by_methods_complete():
    assert complete._completion_cwd.__module__ == "tui_gateway.methods_complete"
    assert complete._rank_slash_completions.__module__ == "tui_gateway.methods_complete"
    assert complete._model_picker_context.__module__ == "tui_gateway.methods_complete"


def test_complete_path_registered_handler_uses_request_local_cwd(tmp_path):
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta", encoding="utf-8")

    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "c1",
            "method": "complete.path",
            "params": {"word": "a", "cwd": str(tmp_path)},
        }
    )

    assert resp["result"]["items"] == [
        {"text": "alpha.txt", "display": "alpha.txt", "meta": ""}
    ]


def test_complete_slash_registered_handler_preserves_rpc_behavior():
    resp = server.handle_request(
        {"jsonrpc": "2.0", "id": "c2", "method": "complete.slash", "params": {"text": "/det"}}
    )

    assert any(item["text"] == "/details" for item in resp["result"]["items"])
