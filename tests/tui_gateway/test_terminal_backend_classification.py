from tui_gateway import server


def test_config_selected_ssh_is_not_classified_as_local(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"terminal": {"backend": "ssh", "cwd": "/home/alice/project"}},
    )

    session = {"cwd": "/home/alice/project"}

    assert server._is_local_terminal_backend() is False
    assert server._display_session_cwd(session) == "/home/alice/project"
    assert session["cwd"] == "/home/alice/project"