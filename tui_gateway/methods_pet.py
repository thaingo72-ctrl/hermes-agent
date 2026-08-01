"""Pet JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.pet import constants, render, store
from agent.pet.generate.imagegen import GenerationError

PET_RPC_METHODS = (
    "pet.info",
    "pet.info.meta",
    "pet.cells",
    "pet.gallery",
    "pet.select",
    "pet.remove",
    "pet.export",
    "pet.rename",
    "pet.thumb",
    "pet.disable",
    "pet.scale",
    "pet.cancel",
    "pet.generate.status",
    "pet.generate",
    "pet.hatch",
)

PROFILE_SCOPED_PET_RPC_METHODS = (
    "pet.info",
    "pet.info.meta",
    "pet.cells",
    "pet.gallery",
    "pet.select",
    "pet.remove",
    "pet.export",
    "pet.rename",
    "pet.thumb",
    "pet.disable",
    "pet.scale",
    "pet.generate.status",
    "pet.generate",
    "pet.hatch",
)

PET_REFERENCE_MIME_EXT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
    "gif": "gif",
}


def _reference_max_bytes() -> int:
    try:
        return max(
            1,
            int(os.environ.get("HERMES_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)),
        )
    except (TypeError, ValueError):
        return 16 * 1024 * 1024


@dataclass(frozen=True)
class PetGenerationState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: set[str] = field(default_factory=set)

    def arm(self, token: str) -> None:
        with self.lock:
            self.cancelled.discard(token)

    def request(self, token: str) -> None:
        with self.lock:
            self.cancelled.add(token)

    def is_cancelled(self, token: str) -> bool:
        with self.lock:
            return token in self.cancelled

    def release(self, token: str) -> None:
        with self.lock:
            self.cancelled.discard(token)

    def snapshot(self) -> set[str]:
        with self.lock:
            return set(self.cancelled)


@dataclass(frozen=True)
class PetPayloadCache:
    lock: threading.Lock = field(default_factory=threading.Lock)
    payloads: dict[tuple, dict] = field(default_factory=dict)
    max_entries: int = 8

    def get(self, key: tuple) -> dict | None:
        with self.lock:
            cached = self.payloads.get(key)
        return clone_pet_payload(cached) if cached is not None else None

    def put(self, key: tuple, payload: dict) -> None:
        with self.lock:
            self.payloads[key] = payload
            while len(self.payloads) > self.max_entries:
                self.payloads.pop(next(iter(self.payloads)))


@dataclass(frozen=True)
class PetServices:
    emit: Callable[[str, str, dict], None]
    load_config: Callable[[], dict]
    resolve_active_pet: Callable[[str], Any]
    installed_pets: Callable[[], list]
    install_pet: Callable[[str], Any]
    remove_pet: Callable[[str], bool]
    export_pet: Callable[[str], tuple[str, bytes]]
    rename_pet: Callable[[str, str], str]
    thumbnail_png: Callable[..., bytes]
    load_pet: Callable[[str], Any]
    unique_slug: Callable[[str], str]
    set_active: Callable[[str], None]
    clear_active_if: Callable[[str], None]
    rename_active_if: Callable[[str, str], None]
    set_enabled: Callable[[bool], None]
    set_pet_scale: Callable[[Any], tuple[float, str | None]]
    fetch_manifest: Callable[[], list]
    prefetch_manifest: Callable[[], None]
    generate_base_drafts: Callable[..., list]
    hatch_pet: Callable[..., Any]
    resolve_provider: Callable[..., Any]
    list_sprite_providers: Callable[[], list]
    get_hermes_home: Callable[[], Path]
    new_token: Callable[[], str]
    copyfile: Callable[[Any, Any], None]
    generation_state: PetGenerationState
    payload_cache: PetPayloadCache = field(default_factory=PetPayloadCache)


def default_pet_services(*, emit: Callable[[str, str, dict], None]) -> PetServices:
    """Build lazy service wrappers so runtime dependencies are explicit."""

    def load_config() -> dict:
        from hermes_cli.config import load_config

        return load_config()

    def set_active(slug: str) -> None:
        from hermes_cli.pets import _set_active

        _set_active(slug)

    def clear_active_if(slug: str) -> None:
        from hermes_cli.pets import _clear_active_if

        _clear_active_if(slug)

    def rename_active_if(old_slug: str, new_slug: str) -> None:
        from hermes_cli.pets import _rename_active_if

        _rename_active_if(old_slug, new_slug)

    def set_enabled(enabled: bool) -> None:
        from hermes_cli.pets import _set_enabled

        _set_enabled(enabled)

    def set_pet_scale(value) -> tuple[float, str | None]:
        from hermes_cli.pets import set_pet_scale

        return set_pet_scale(value)

    def fetch_manifest() -> list:
        from agent.pet.manifest import fetch_manifest

        return fetch_manifest()

    def prefetch_manifest() -> None:
        from agent.pet.manifest import prefetch

        prefetch()

    def generate_base_drafts(*args, **kwargs) -> list:
        from agent.pet.generate import generate_base_drafts

        return generate_base_drafts(*args, **kwargs)

    def hatch_pet(**kwargs):
        from agent.pet.generate import hatch_pet

        return hatch_pet(**kwargs)

    def resolve_provider(**kwargs):
        from agent.pet.generate.imagegen import resolve_provider

        return resolve_provider(**kwargs)

    def list_sprite_providers() -> list:
        from agent.pet.generate.imagegen import list_sprite_providers

        return list_sprite_providers()

    def get_hermes_home() -> Path:
        from hermes_constants import get_hermes_home

        return get_hermes_home()

    return PetServices(
        emit=emit,
        load_config=load_config,
        resolve_active_pet=store.resolve_active_pet,
        installed_pets=store.installed_pets,
        install_pet=store.install_pet,
        remove_pet=store.remove_pet,
        export_pet=store.export_pet,
        rename_pet=store.rename_pet,
        thumbnail_png=store.thumbnail_png,
        load_pet=store.load_pet,
        unique_slug=store.unique_slug,
        set_active=set_active,
        clear_active_if=clear_active_if,
        rename_active_if=rename_active_if,
        set_enabled=set_enabled,
        set_pet_scale=set_pet_scale,
        fetch_manifest=fetch_manifest,
        prefetch_manifest=prefetch_manifest,
        generate_base_drafts=generate_base_drafts,
        hatch_pet=hatch_pet,
        resolve_provider=resolve_provider,
        list_sprite_providers=list_sprite_providers,
        get_hermes_home=get_hermes_home,
        new_token=lambda: uuid.uuid4().hex[:12],
        copyfile=shutil.copyfile,
        generation_state=PetGenerationState(),
    )


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def pet_frame_counts(spritesheet) -> dict:
    try:
        return render.state_frame_counts(str(spritesheet))
    except Exception:
        return {}


def pet_sheet_revision(spritesheet) -> str:
    try:
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        return "0:0"


def pet_payload_cache_key(pet, *, scale: float) -> tuple | None:
    try:
        stat = pet.spritesheet.stat()
    except Exception:
        return None
    return (
        str(pet.spritesheet),
        stat.st_mtime_ns,
        stat.st_size,
        pet.slug,
        pet.display_name,
        round(scale, 4),
    )


def clone_pet_payload(payload: dict) -> dict:
    out = dict(payload)
    if isinstance(payload.get("framesByState"), dict):
        out["framesByState"] = dict(payload["framesByState"])
    if isinstance(payload.get("framesByRow"), dict):
        out["framesByRow"] = dict(payload["framesByRow"])
    if isinstance(payload.get("stateRows"), list):
        out["stateRows"] = list(payload["stateRows"])
    return out


def pet_row_frame_counts(spritesheet) -> dict:
    try:
        from PIL import Image

        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        cols = max(1, image.width // constants.FRAME_W)
        row_count = max(1, image.height // constants.FRAME_H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * constants.FRAME_H
            count = 0
            for col in range(cols):
                left = col * constants.FRAME_W
                frame = image.crop((left, top, left + constants.FRAME_W, top + constants.FRAME_H))
                if render._frame_is_blank(frame):
                    break
                count += 1
            out[name] = count
        return out
    except Exception:
        return {}


def pet_config(services: PetServices) -> dict:
    try:
        cfg = services.load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        return display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        return {}


def pet_config_scale(services: PetServices) -> float:
    try:
        return float(pet_config(services).get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    except Exception:
        return constants.DEFAULT_SCALE


def pet_sprite_payload(pet, *, scale: float, services: PetServices) -> dict:
    cache_key = pet_payload_cache_key(pet, scale=scale)
    if cache_key is not None:
        cached = services.payload_cache.get(cache_key)
        if cached is not None:
            return cached

    raw = pet.spritesheet.read_bytes()
    suffix = pet.spritesheet.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp"
    payload = {
        "slug": pet.slug,
        "displayName": pet.display_name,
        "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": pet_sheet_revision(pet.spritesheet),
        "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H,
        "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": pet_frame_counts(pet.spritesheet),
        "framesByRow": pet_row_frame_counts(pet.spritesheet),
        "loopMs": constants.LOOP_MS,
        "scale": scale,
        "stateRows": pet_state_rows(pet.spritesheet),
    }
    if cache_key is not None:
        services.payload_cache.put(cache_key, payload)
    return clone_pet_payload(payload)


def pet_active_selection(services: PetServices):
    pet_cfg = pet_config(services)
    enabled = bool(pet_cfg.get("enabled"))
    configured_slug = str(pet_cfg.get("slug", "") or "")
    pet = services.resolve_active_pet(configured_slug) if enabled else None
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return enabled, pet, scale


def pet_state_rows(spritesheet) -> list[str]:
    try:
        from PIL import Image

        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    except Exception:
        return list(constants.STATE_ROWS)


def pet_gen_root(services: PetServices) -> Path:
    root = services.get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def pet_gen_sweep(root: Path, *, max_age_s: float = 3600.0) -> None:
    try:
        now = time.time()
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
    except Exception:
        return


def pet_png_data_uri(path, *, max_px: int = 160) -> str:
    from PIL import Image

    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


def pet_reference_images_from_data_url(ref_raw: str, stage: Path) -> list[Path]:
    match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")

    mime = match.group(1).lower()
    ext = PET_REFERENCE_MIME_EXT.get(mime)
    if ext is None:
        raise ValueError("unsupported reference image type")

    max_bytes = _reference_max_bytes()
    payload = "".join(match.group(2).split())
    approx = (len(payload) * 3) // 4
    if approx > max_bytes:
        raise ValueError("reference image too large")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc

    if len(raw) > max_bytes:
        raise ValueError("reference image too large")

    ref_path = stage / f"reference.{ext}"
    ref_path.write_bytes(raw)
    return [ref_path]


def pet_info(rid, params: dict, services: PetServices) -> dict:
    try:
        enabled, pet, scale = pet_active_selection(services)
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(rid, {"enabled": True, **pet_sprite_payload(pet, scale=scale, services=services)})
    except Exception:
        return _ok(rid, {"enabled": False})


def pet_info_meta(rid, params: dict, services: PetServices) -> dict:
    try:
        enabled, pet, scale = pet_active_selection(services)
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "scale": scale,
                "spritesheetRevision": pet_sheet_revision(pet.spritesheet),
            },
        )
    except Exception:
        return _ok(rid, {"enabled": False})


def pet_cells(rid, params: dict, services: PetServices) -> dict:
    try:
        pet_cfg = pet_config(services)
        if not bool(pet_cfg.get("enabled")):
            return _ok(rid, {"enabled": False})

        pet = services.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
        if pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        state = str(params.get("state") or constants.PetState.IDLE.value)
        scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
        cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

        if params.get("graphics"):
            configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            graphics_mode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
            if graphics_mode == "kitty":
                image_id = render.kitty_image_id(pet.slug)
                payload = render.PetRenderer(str(pet.spritesheet), mode="kitty", scale=scale).kitty_payload(
                    state, image_id=image_id
                )
                if payload:
                    count = len(payload["frames"]) or 1
                    return _ok(
                        rid,
                        {
                            "enabled": True,
                            "slug": pet.slug,
                            "displayName": pet.display_name,
                            "state": state,
                            "graphics": "kitty",
                            "imageId": image_id,
                            "color": render.kitty_color_hex(image_id),
                            "cols": payload["cols"],
                            "rows": payload["rows"],
                            "placeholder": payload["placeholder"],
                            "frames": payload["frames"],
                            "frameMs": constants.LOOP_MS / max(1, count),
                            "scale": scale,
                        },
                    )

        renderer = render.PetRenderer(
            str(pet.spritesheet),
            mode="unicode",
            scale=scale,
            unicode_cols=cols,
        )
        count = renderer.frame_count(state) or 1
        frames = []
        for index in range(count):
            grid = renderer.cells(state, index, cols=cols)
            frames.append([[[*top, *bottom] for (top, bottom) in row] for row in grid])

        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "state": state,
                "cols": cols,
                "frameMs": constants.LOOP_MS / max(1, count),
                "frames": frames,
                "scale": scale,
            },
        )
    except Exception:
        return _ok(rid, {"enabled": False})


def pet_gallery(rid, params: dict, services: PetServices) -> dict:
    local_only = bool(params.get("localOnly"))
    try:
        pet_cfg = pet_config(services)
        installed = {pet.slug: pet for pet in services.installed_pets()}

        gallery: list[dict] = []
        seen: set[str] = set()
        try:
            if local_only:
                services.prefetch_manifest()

            for entry in [] if local_only else services.fetch_manifest():
                seen.add(entry.slug)
                gallery.append(
                    {
                        "slug": entry.slug,
                        "displayName": entry.display_name,
                        "installed": entry.slug in installed,
                        "spritesheetUrl": entry.spritesheet_url,
                        "curated": "/curated/" in entry.spritesheet_url,
                        "generated": entry.slug in installed and installed[entry.slug].generated,
                    }
                )
        except Exception:
            pass

        for slug, pet in installed.items():
            if slug not in seen:
                gallery.append(
                    {
                        "slug": slug,
                        "displayName": pet.display_name,
                        "installed": True,
                        "spritesheetUrl": "",
                        "generated": pet.generated,
                    }
                )

        return _ok(
            rid,
            {
                "enabled": bool(pet_cfg.get("enabled")),
                "active": str(pet_cfg.get("slug", "") or ""),
                "pets": gallery,
            },
        )
    except Exception:
        return _ok(rid, {"enabled": False, "active": "", "pets": []})


def pet_select(rid, params: dict, services: PetServices) -> dict:
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet.manifest import ManifestError

        try:
            pet = services.install_pet(slug)
        except (store.PetStoreError, ManifestError) as exc:
            return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
        services.set_active(slug)
        return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})
    except Exception as exc:
        return _err(rid, 5031, f"pet.select failed: {exc}")


def pet_remove(rid, params: dict, services: PetServices) -> dict:
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        removed = services.remove_pet(slug)
        try:
            services.clear_active_if(slug)
        except Exception:
            pass
        return _ok(rid, {"ok": removed, "slug": slug})
    except Exception as exc:
        return _err(rid, 5031, f"pet.remove failed: {exc}")


def pet_export(rid, params: dict, services: PetServices) -> dict:
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        filename, data = services.export_pet(slug)
        return _ok(
            rid,
            {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
        )
    except Exception as exc:
        return _err(rid, 5031, f"pet.export failed: {exc}")


def pet_rename(rid, params: dict, services: PetServices) -> dict:
    slug = str(params.get("slug") or "").strip()
    name = str(params.get("name") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        new_slug = services.rename_pet(slug, name)
        if not new_slug:
            return _err(rid, 5031, "pet.rename failed")
        if new_slug != slug:
            try:
                services.rename_active_if(slug, new_slug)
            except Exception:
                pass
        return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})
    except Exception as exc:
        return _err(rid, 5031, f"pet.rename failed: {exc}")


def pet_thumb(rid, params: dict, services: PetServices) -> dict:
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        data = services.thumbnail_png(slug, source_url=str(params.get("url") or ""))
        if not data:
            return _ok(rid, {"ok": False, "slug": slug})
        return _ok(
            rid,
            {
                "ok": True,
                "slug": slug,
                "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
            },
        )
    except Exception:
        return _ok(rid, {"ok": False, "slug": slug})


def pet_disable(rid, params: dict, services: PetServices) -> dict:
    try:
        services.set_enabled(False)
        return _ok(rid, {"ok": True})
    except Exception as exc:
        return _err(rid, 5031, f"pet.disable failed: {exc}")


def pet_scale(rid, params: dict, services: PetServices) -> dict:
    try:
        scale, err = services.set_pet_scale(params.get("scale"))
        if err:
            return _err(rid, 4004, err)
        return _ok(rid, {"ok": True, "scale": scale})
    except Exception as exc:
        return _err(rid, 5031, f"pet.scale failed: {exc}")


def pet_cancel(rid, params: dict, services: PetServices) -> dict:
    token = str(params.get("token") or "").strip()
    if token:
        services.generation_state.request(token)
    return _ok(rid, {"ok": True})


def pet_generate_status(rid, params: dict, services: PetServices) -> dict:
    try:
        try:
            services.resolve_provider(require_references=True)
            available = True
        except GenerationError:
            available = False
        try:
            providers = services.list_sprite_providers()
        except Exception:
            providers = []
        return _ok(rid, {"available": available, "providers": providers})
    except Exception:
        return _ok(rid, {"available": False, "providers": []})


def pet_generate(rid, params: dict, services: PetServices) -> dict:
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    try:
        root = pet_gen_root(services)
        pet_gen_sweep(root)

        token = services.new_token()
        services.generation_state.arm(token)
        stage = root / token
        stage.mkdir(parents=True, exist_ok=True)

        reference_images = None
        if ref_raw:
            try:
                reference_images = pet_reference_images_from_data_url(ref_raw, stage)
            except ValueError as exc:
                services.generation_state.release(token)
                return _err(rid, 4004, str(exc))

        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = services.resolve_provider(require_references=bool(reference_images), prefer=provider_name)
            except GenerationError as exc:
                services.generation_state.release(token)
                return _err(rid, 5031, str(exc))

        concept = prompt or "a pet based on the reference image"
        out: list[dict] = []

        try:
            services.emit("pet.generate.progress", "", {"token": token, "count": count})
        except Exception:
            pass

        def on_draft(index: int, src) -> None:
            dest = stage / f"draft-{index}.png"
            try:
                services.copyfile(src, dest)
                data_uri = pet_png_data_uri(dest)
            except Exception:
                return
            out.append({"index": index, "dataUri": data_uri})
            try:
                services.emit(
                    "pet.generate.progress",
                    "",
                    {"token": token, "index": index, "dataUri": data_uri, "count": count},
                )
            except Exception:
                pass

        try:
            services.generate_base_drafts(
                concept,
                n=count,
                style=style,
                reference_images=reference_images,
                provider=sprite,
                on_draft=on_draft,
                is_cancelled=lambda: services.generation_state.is_cancelled(token),
            )
        except GenerationError as exc:
            services.generation_state.release(token)
            return _err(rid, 5031, str(exc))

        cancelled = services.generation_state.is_cancelled(token)
        services.generation_state.release(token)
        if cancelled:
            return _err(rid, 5031, "generation cancelled")
        if not out:
            return _err(rid, 5031, "generation produced no usable drafts")
        out.sort(key=lambda draft: draft["index"])
        return _ok(rid, {"ok": True, "token": token, "drafts": out})
    except Exception as exc:
        return _err(rid, 5031, f"pet.generate failed: {exc}")


def pet_hatch(rid, params: dict, services: PetServices) -> dict:
    token = str(params.get("token") or "").strip()
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    index = params.get("index", 0)
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    try:
        base = pet_gen_root(services) / token / f"draft-{index}.png"
        if not base.is_file():
            return _err(rid, 4004, "draft expired — generate again")

        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = services.resolve_provider(require_references=True, prefer=provider_name)
            except GenerationError as exc:
                return _err(rid, 5031, str(exc))

        services.generation_state.arm(cancel_token)
        slug = services.unique_slug(name)

        def on_progress(event: str, detail: str) -> None:
            payload: dict = {"event": event, "detail": detail}
            if event == "row" and detail.count(":") == 2:
                state, done, total = detail.split(":")
                payload = {"event": "row", "state": state, "done": done, "total": total}
            try:
                services.emit("pet.hatch.progress", "", payload)
            except Exception:
                pass

        try:
            result = services.hatch_pet(
                base_image=base,
                slug=slug,
                display_name=name,
                description=str(params.get("description") or ""),
                concept=str(params.get("prompt") or name),
                style=str(params.get("style") or "auto").strip() or "auto",
                provider=sprite,
                on_progress=on_progress,
                is_cancelled=lambda: services.generation_state.is_cancelled(cancel_token),
            )
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))
        finally:
            services.generation_state.release(cancel_token)

        pet = services.load_pet(result.slug)
        payload = pet_sprite_payload(pet, scale=pet_config_scale(services), services=services) if pet else {}
        return _ok(
            rid,
            {
                "ok": True,
                "slug": result.slug,
                "displayName": result.display_name,
                "warnings": result.validation.get("warnings", []),
                "pet": payload,
            },
        )
    except Exception as exc:
        return _err(rid, 5031, f"pet.hatch failed: {exc}")


PET_HANDLERS: dict[str, Callable[[Any, dict, PetServices], dict]] = {
    "pet.info": pet_info,
    "pet.info.meta": pet_info_meta,
    "pet.cells": pet_cells,
    "pet.gallery": pet_gallery,
    "pet.select": pet_select,
    "pet.remove": pet_remove,
    "pet.export": pet_export,
    "pet.rename": pet_rename,
    "pet.thumb": pet_thumb,
    "pet.disable": pet_disable,
    "pet.scale": pet_scale,
    "pet.cancel": pet_cancel,
    "pet.generate.status": pet_generate_status,
    "pet.generate": pet_generate,
    "pet.hatch": pet_hatch,
}


def _bind(handler: Callable[[Any, dict, PetServices], dict], services: PetServices):
    def bound(rid, params):
        return handler(rid, params, services)

    bound.__name__ = handler.__name__
    bound.__module__ = handler.__module__
    return bound


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: PetServices,
    profile_scoped: Callable[[Callable[[Any, dict], dict]], Callable[[Any, dict], dict]],
) -> None:
    """Register pet callables under the existing JSON-RPC names."""

    for name, handler in PET_HANDLERS.items():
        bound = _bind(handler, services)
        bound.pet_rpc_method = name
        if name in PROFILE_SCOPED_PET_RPC_METHODS:
            bound = profile_scoped(bound)
            bound.__name__ = handler.__name__
            bound.__module__ = handler.__module__
            bound.pet_rpc_method = name
        methods[name] = bound
