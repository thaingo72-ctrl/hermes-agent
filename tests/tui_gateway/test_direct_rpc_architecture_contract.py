"""Architecture contract for direct TUI JSON-RPC handler ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tui_gateway import server


ROOT = Path(__file__).resolve().parents[2]
TUI_GATEWAY = ROOT / "tui_gateway"
DIRECT_METHOD_MODULES = sorted(TUI_GATEWAY.glob("methods_*.py"))
FORBIDDEN_REBINDING_TOKENS = (
    "FunctionType",
    "HandlerRegistry",
    "__globals__",
    "method_ctx",
    "sys.modules[__name__]",
)


def _registered_rpc_assignments(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "methods"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                names.add(target.slice.value)
    return names


def test_direct_rpc_modules_have_unique_static_ownership():
    owners: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for path in DIRECT_METHOD_MODULES:
        for rpc_name in _registered_rpc_assignments(path):
            owner = path.name
            if rpc_name in owners:
                duplicates.setdefault(rpc_name, [owners[rpc_name]]).append(owner)
            else:
                owners[rpc_name] = owner

    assert not duplicates
    assert "session.interrupt" in owners
    assert owners["session.interrupt"] == "methods_session_state.py"


def test_no_legacy_rebinding_architecture_remains_in_tui_gateway():
    assert not (TUI_GATEWAY / "methods_session.py").exists()
    assert not (TUI_GATEWAY / "method_ctx.py").exists()

    offenders: list[tuple[str, str]] = []
    for path in sorted(TUI_GATEWAY.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_REBINDING_TOKENS:
            if token in source:
                offenders.append((path.name, token))

    assert offenders == []


@pytest.mark.parametrize(
    ("rpc_name", "module_name"),
    [
        ("billing.state", "tui_gateway.methods_billing"),
        ("config.get", "tui_gateway.methods_config"),
        ("complete.slash", "tui_gateway.methods_complete"),
        ("prompt.submit", "tui_gateway.methods_prompt"),
        ("session.create", "tui_gateway.methods_session_lifecycle"),
        ("session.usage", "tui_gateway.methods_session_meta"),
        ("session.status", "tui_gateway.methods_session_state"),
        ("commands.catalog", "tui_gateway.methods_system"),
        ("browser.manage", "tui_gateway.methods_management"),
        ("slash.exec", "tui_gateway.methods_operations"),
        ("pet.info", "tui_gateway.methods_pet"),
        ("delegation.status", "tui_gateway.methods_delegation"),
    ],
)
def test_aggregate_domains_are_direct_service_injected_once(rpc_name: str, module_name: str):
    handler = server._methods[rpc_name]
    target = getattr(handler, "func", handler)
    assert target.__module__ == module_name
