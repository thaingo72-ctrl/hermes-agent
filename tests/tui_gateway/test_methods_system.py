from __future__ import annotations

import inspect

from tui_gateway import methods_system, methods_tools, server


SYSTEM_METHODS = {
    "system.battery",
    "process.stop",
    "process.list",
    "process.kill",
    "reload.mcp",
    "reload.env",
    "commands.catalog",
    "cli.exec",
    "command.resolve",
    "command.dispatch",
}


def test_system_methods_are_registered_from_canonical_module():
    owners = {name: server._methods[name].__module__ for name in SYSTEM_METHODS}

    assert owners == {name: methods_system.__name__ for name in SYSTEM_METHODS}


def test_legacy_tools_module_no_longer_registers_system_methods():
    legacy_names = {name for name, _handler in methods_tools._registry._pending}

    assert SYSTEM_METHODS.isdisjoint(legacy_names)


def test_methods_system_has_no_legacy_rebinding_or_service_locator_patterns():
    source = inspect.getsource(methods_system)
    forbidden = [
        "ContextVar",
        "FunctionType",
        "__globals__",
        "vars(",
        "globals(",
        "sys.modules[__name__]",
    ]

    assert all(pattern not in source for pattern in forbidden)
