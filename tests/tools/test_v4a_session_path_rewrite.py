import json

import pytest

import tools.file_tools as ft
import tools.terminal_tool as tt
from tools.file_operations import PatchResult


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})


class RecordingPatchOps:
    def __init__(self):
        self.patch = ""

    def patch_v4a(self, patch_content: str) -> PatchResult:
        self.patch = patch_content
        return PatchResult(success=True)


@pytest.fixture
def two_sessions(tmp_path, monkeypatch):
    wt_a = tmp_path / "wt-a"
    wt_b = tmp_path / "wt-b"
    for wt in (wt_a, wt_b):
        wt.mkdir()
        (wt / "update.txt").write_text("old\n")
        (wt / "delete.txt").write_text("bye\n")
        (wt / "move-src.txt").write_text("move\n")
    monkeypatch.chdir(tmp_path)
    tt.record_session_cwd("sess-a", str(wt_a))
    tt.record_session_cwd("sess-b", str(wt_b))
    monkeypatch.setattr(ft, "_check_sensitive_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ft, "_check_cross_profile_path", lambda *_args, **_kwargs: None)
    return wt_a, wt_b


@pytest.mark.parametrize(
    ("patch_text", "expected_headers"),
    [
        (
            """*** Begin Patch
*** Update File: update.txt
@@
-old
+new
*** End Patch
""",
            ("*** Update File: {root}/update.txt",),
        ),
        (
            """*** Begin Patch
*** Add File: add.txt
+hello
*** End Patch
""",
            ("*** Add File: {root}/add.txt",),
        ),
        (
            """*** Begin Patch
*** Delete File: delete.txt
*** End Patch
""",
            ("*** Delete File: {root}/delete.txt",),
        ),
        (
            """*** Begin Patch
*** Move File: move-src.txt -> move-dst.txt
*** End Patch
""",
            ("*** Move File: {root}/move-src.txt -> {root}/move-dst.txt",),
        ),
    ],
)
def test_v4a_headers_are_rewritten_per_session_before_apply(
    two_sessions, monkeypatch, patch_text, expected_headers
):
    wt_a, wt_b = two_sessions
    ops = RecordingPatchOps()
    monkeypatch.setattr(ft, "_get_file_ops", lambda _task_id: ops)

    result = json.loads(ft.patch_tool(mode="patch", patch=patch_text, task_id="sess-a"))

    assert result["success"] is True
    for header in expected_headers:
        assert header.format(root=wt_a) in ops.patch
        assert header.format(root=wt_b) not in ops.patch


def test_v4a_rewrite_keeps_two_shared_backend_sessions_in_their_worktrees(
    two_sessions, monkeypatch
):
    wt_a, wt_b = two_sessions
    patches: dict[str, str] = {}

    def get_ops(task_id: str):
        ops = RecordingPatchOps()
        original = ops.patch_v4a

        def record(patch_content: str) -> PatchResult:
            patches[task_id] = patch_content
            return original(patch_content)

        ops.patch_v4a = record
        return ops

    monkeypatch.setattr(ft, "_get_file_ops", get_ops)
    patch_text = """*** Begin Patch
*** Update File: update.txt
@@
-old
+new
*** Add File: add.txt
+hello
*** Delete File: delete.txt
*** Move File: move-src.txt -> move-dst.txt
*** End Patch
"""

    assert json.loads(ft.patch_tool(mode="patch", patch=patch_text, task_id="sess-a"))["success"]
    assert json.loads(ft.patch_tool(mode="patch", patch=patch_text, task_id="sess-b"))["success"]

    assert str(wt_a / "update.txt") in patches["sess-a"]
    assert str(wt_a / "add.txt") in patches["sess-a"]
    assert str(wt_a / "delete.txt") in patches["sess-a"]
    assert f"{wt_a / 'move-src.txt'} -> {wt_a / 'move-dst.txt'}" in patches["sess-a"]
    assert str(wt_b / "update.txt") in patches["sess-b"]
    assert str(wt_b / "add.txt") in patches["sess-b"]
    assert str(wt_b / "delete.txt") in patches["sess-b"]
    assert f"{wt_b / 'move-src.txt'} -> {wt_b / 'move-dst.txt'}" in patches["sess-b"]
    assert str(wt_b) not in patches["sess-a"]
    assert str(wt_a) not in patches["sess-b"]
