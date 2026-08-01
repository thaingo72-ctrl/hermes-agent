"""Ownership and state-isolation tests for canonical TUI pet RPCs."""

from __future__ import annotations

import threading
from pathlib import Path

from tui_gateway import methods_pet, methods_session, server


def test_pet_handlers_are_registered_from_pet_module():
    for name in methods_pet.PET_RPC_METHODS:
        handler = server._methods[name]
        target = getattr(handler, "func", handler)
        assert target.__module__ == "tui_gateway.methods_pet"


def test_legacy_session_module_no_longer_owns_pet_handlers():
    registered = {name for name, _handler in methods_session._registry._pending}
    assert not any(name.startswith("pet.") for name in registered)


def test_pet_register_profile_scopes_profile_backed_handlers():
    methods: dict[str, object] = {}
    scoped: list[str] = []

    def profile_scoped(handler):
        scoped.append(handler.pet_rpc_method)

        def wrapped(rid, params):
            return handler(rid, params)

        return wrapped

    services = methods_pet.PetServices(
        emit=lambda event, session_id, payload: None,
        load_config=lambda: {},
        resolve_active_pet=lambda slug: None,
        installed_pets=lambda: [],
        install_pet=lambda slug: None,
        remove_pet=lambda slug: False,
        export_pet=lambda slug: ("pet.zip", b""),
        rename_pet=lambda slug, name: slug,
        thumbnail_png=lambda slug, source_url="": b"",
        load_pet=lambda slug: None,
        unique_slug=lambda name: "pet",
        set_active=lambda slug: None,
        clear_active_if=lambda slug: None,
        rename_active_if=lambda old, new: None,
        set_enabled=lambda enabled: None,
        set_pet_scale=lambda scale: (1.0, None),
        fetch_manifest=lambda: [],
        prefetch_manifest=lambda: None,
        generate_base_drafts=lambda *args, **kwargs: [],
        hatch_pet=lambda **kwargs: None,
        resolve_provider=lambda **kwargs: None,
        list_sprite_providers=lambda: [],
        get_hermes_home=lambda: Path("/tmp/hermes"),
        new_token=lambda: "token",
        copyfile=lambda src, dest: None,
        generation_state=methods_pet.PetGenerationState(),
    )

    methods_pet.register(methods, services=services, profile_scoped=profile_scoped)

    assert set(methods) == set(methods_pet.PET_RPC_METHODS)
    assert set(scoped) == set(methods_pet.PROFILE_SCOPED_PET_RPC_METHODS)


def test_generation_state_keeps_concurrent_tokens_isolated():
    state = methods_pet.PetGenerationState()
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    results: dict[str, bool] = {}

    def worker(token: str, cancel: bool) -> None:
        state.arm(token)
        entered.wait(timeout=2)
        if cancel:
            state.request(token)
        release.wait(timeout=2)
        results[token] = state.is_cancelled(token)
        state.release(token)

    threads = [
        threading.Thread(target=worker, args=("generate-a", True)),
        threading.Thread(target=worker, args=("generate-b", False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results == {"generate-a": True, "generate-b": False}
    assert state.snapshot() == set()
