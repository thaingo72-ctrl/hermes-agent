"""Billing and subscription JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BillingServices:
    build_billing_state: Callable[[], Any]
    build_usage_model: Callable[[], Any]
    build_subscription_state: Callable[[], Any]
    subscription_change_preview_from_payload: Callable[[dict], Any]
    post_subscription_preview: Callable[..., dict]
    put_subscription_pending_change: Callable[..., dict]
    delete_subscription_pending_change: Callable[[], dict]
    post_subscription_upgrade: Callable[..., dict]
    post_charge: Callable[..., dict]
    get_charge_status: Callable[[str], dict]
    patch_auto_top_up: Callable[..., dict]
    step_up_nous_billing_scope: Callable[..., bool]
    new_idempotency_key: Callable[[], str]
    emit: Callable[[str, str, dict], None]


def default_billing_services(
    *, emit: Callable[[str, str, dict], None]
) -> BillingServices:
    """Build lazy service wrappers so tests can monkeypatch provider modules."""

    def build_billing_state():
        from agent.billing_view import build_billing_state as service

        return service()

    def build_usage_model():
        from agent.billing_usage import build_usage_model as service

        return service()

    def build_subscription_state():
        from agent.subscription_view import build_subscription_state as service

        return service()

    def subscription_change_preview_from_payload(payload: dict):
        from agent.subscription_view import (
            subscription_change_preview_from_payload as service,
        )

        return service(payload)

    def post_subscription_preview(**kwargs):
        from hermes_cli.nous_billing import post_subscription_preview as service

        return service(**kwargs)

    def put_subscription_pending_change(**kwargs):
        from hermes_cli.nous_billing import put_subscription_pending_change as service

        return service(**kwargs)

    def delete_subscription_pending_change():
        from hermes_cli.nous_billing import delete_subscription_pending_change as service

        return service()

    def post_subscription_upgrade(**kwargs):
        from hermes_cli.nous_billing import post_subscription_upgrade as service

        return service(**kwargs)

    def post_charge(**kwargs):
        from hermes_cli.nous_billing import post_charge as service

        return service(**kwargs)

    def get_charge_status(charge_id: str):
        from hermes_cli.nous_billing import get_charge_status as service

        return service(charge_id)

    def patch_auto_top_up(**kwargs):
        from hermes_cli.nous_billing import patch_auto_top_up as service

        return service(**kwargs)

    def step_up_nous_billing_scope(**kwargs):
        from hermes_cli.auth import step_up_nous_billing_scope as service

        return service(**kwargs)

    def new_idempotency_key():
        from agent.billing_view import new_idempotency_key as service

        return service()

    return BillingServices(
        build_billing_state=build_billing_state,
        build_usage_model=build_usage_model,
        build_subscription_state=build_subscription_state,
        subscription_change_preview_from_payload=subscription_change_preview_from_payload,
        post_subscription_preview=post_subscription_preview,
        put_subscription_pending_change=put_subscription_pending_change,
        delete_subscription_pending_change=delete_subscription_pending_change,
        post_subscription_upgrade=post_subscription_upgrade,
        post_charge=post_charge,
        get_charge_status=get_charge_status,
        patch_auto_top_up=patch_auto_top_up,
        step_up_nous_billing_scope=step_up_nous_billing_scope,
        new_idempotency_key=new_idempotency_key,
        emit=emit,
    )


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRemoteSpendingRevoked,
        BillingScopeRequired,
        BillingSessionRevoked,
        BillingTransient,
    )

    kind = "error"
    if isinstance(exc, BillingRemoteSpendingRevoked):
        kind = "remote_spending_revoked"
    elif isinstance(exc, BillingSessionRevoked):
        kind = "session_revoked"
    elif isinstance(exc, BillingScopeRequired):
        kind = "insufficient_scope"
    elif isinstance(exc, BillingTransient):
        kind = str(exc.error) if getattr(exc, "error", None) else "rate_limited"
    elif getattr(exc, "error", None):
        kind = str(exc.error)
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
        "actor": getattr(exc, "actor", None),
        "code": getattr(exc, "code", None),
        "recovery": getattr(exc, "recovery", None),
    }


def _usage_payload(state, services: BillingServices) -> dict:
    if not getattr(state, "logged_in", False):
        return {"available": False}
    try:
        return _serialize_usage_model(services.build_usage_model())
    except Exception:
        return {"available": False}


def _serialize_billing_state(state, services: BillingServices) -> dict:
    """Serialize a BillingState for the wire (Decimals to strings, money-safe)."""
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    card = None
    if state.card is not None:
        card = {
            "brand": state.card.brand,
            "last4": state.card.last4,
            "masked": state.card.masked,
            "display": state.card.display,
            "resolved_via": state.card.resolved_via,
        }
    payment_method = None
    if state.payment_method is not None:
        pm = state.payment_method
        if pm.kind == "card":
            payment_method = {
                "kind": "card",
                "brand": pm.brand,
                "last4": pm.last4,
                "wallet": pm.wallet,
                "resolved_via": pm.resolved_via,
            }
        elif pm.kind == "link":
            payment_method = {
                "kind": "link",
                "email": pm.email,
                "resolved_via": pm.resolved_via,
            }
        else:
            payment_method = {
                "kind": "unknown",
                "raw_kind": pm.raw_kind,
                "resolved_via": pm.resolved_via,
            }
    monthly_cap = None
    if state.monthly_cap is not None:
        mc = state.monthly_cap
        monthly_cap = {
            "limit_usd": _s(mc.limit_usd),
            "limit_display": format_money(mc.limit_usd),
            "spent_this_month_usd": _s(mc.spent_this_month_usd),
            "spent_display": format_money(mc.spent_this_month_usd),
            "is_default_ceiling": mc.is_default_ceiling,
        }
    auto_reload = None
    if state.auto_reload is not None:
        ar = state.auto_reload
        card_out = None
        if ar.card is not None:
            if ar.card.kind == "distinct":
                card_out = {
                    "kind": "distinct",
                    "payment_method_id": ar.card.payment_method_id,
                    "brand": ar.card.brand,
                    "last4": ar.card.last4,
                }
            else:
                card_out = {"kind": ar.card.kind}
        auto_reload = {
            "enabled": ar.enabled,
            "threshold_usd": _s(ar.threshold_usd),
            "threshold_display": format_money(ar.threshold_usd),
            "reload_to_usd": _s(ar.reload_to_usd),
            "reload_to_display": format_money(ar.reload_to_usd),
            "card": card_out,
        }
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "org_name": state.org_name,
        "org_slug": state.org_slug,
        "role": state.role,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "can_charge": state.can_charge,
        "balance_usd": _s(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_s(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _s(state.min_usd),
        "max_usd": _s(state.max_usd),
        "card": card,
        "payment_method": payment_method,
        "monthly_cap": monthly_cap,
        "auto_reload": auto_reload,
        "portal_url": state.portal_url,
        "error": state.error,
        "usage": _usage_payload(state, services),
    }


def _serialize_usage_bar(bar) -> Optional[dict]:
    if bar is None:
        return None
    from agent.billing_usage import _fmt_usd

    return {
        "kind": bar.kind,
        "remaining_display": _fmt_usd(bar.remaining_usd),
        "total_display": _fmt_usd(bar.total_usd),
        "spent_display": _fmt_usd(bar.spent_usd),
        "pct_used": bar.pct_used,
        "fill_fraction": bar.fill_fraction,
    }


def _serialize_usage_model(model) -> dict:
    from agent.billing_usage import _fmt_usd, format_renews

    if model is None or not getattr(model, "available", False):
        return {"ok": True, "available": False}

    return {
        "ok": True,
        "available": True,
        "status": model.status,
        "plan_name": model.plan_name,
        "renews_at": model.renews_at,
        "renews_display": getattr(model, "renews_display", None)
        or format_renews(model.renews_at),
        "subscription_remaining_display": (
            None
            if model.subscription_remaining_usd is None
            else _fmt_usd(model.subscription_remaining_usd)
        ),
        "topup_remaining_display": (
            None
            if model.topup_remaining_usd is None
            else _fmt_usd(model.topup_remaining_usd)
        ),
        "total_spendable_display": (
            None
            if model.total_spendable_usd is None
            else _fmt_usd(model.total_spendable_usd)
        ),
        "has_topup": model.has_topup,
        "plan_bar": _serialize_usage_bar(model.plan_bar),
        "topup_bar": _serialize_usage_bar(model.topup_bar),
    }


def _serialize_subscription_state(state, services: BillingServices) -> dict:
    """Serialize a SubscriptionState for the wire (Decimals to strings)."""
    from agent.billing_usage import format_renews
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    current = None
    if state.current is not None:
        c = state.current
        current = {
            "tier_id": c.tier_id,
            "tier_name": c.tier_name,
            "monthly_credits": _s(c.monthly_credits),
            "credits_remaining": _s(c.credits_remaining),
            "cycle_ends_at": c.cycle_ends_at,
            "pending_downgrade_tier_name": c.pending_downgrade_tier_name,
            "pending_downgrade_at": c.pending_downgrade_at,
            "pending_downgrade_display": format_renews(c.pending_downgrade_at),
            "cancel_at_period_end": c.cancel_at_period_end,
            "cancellation_effective_at": c.cancellation_effective_at,
            "cancellation_effective_display": format_renews(
                c.cancellation_effective_at
            ),
        }
    tiers = [
        {
            "tier_id": t.tier_id,
            "name": t.name,
            "tier_order": t.tier_order,
            "dollars_per_month_display": format_money(t.dollars_per_month),
            "monthly_credits": _s(t.monthly_credits),
            "is_current": t.is_current,
            "is_enabled": t.is_enabled,
        }
        for t in state.tiers
    ]
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "org_name": state.org_name,
        "org_id": state.org_id,
        "role": state.role,
        "context": state.context,
        "current": current,
        "tiers": tiers,
        "portal_url": state.portal_url,
        "error": state.error,
        "usage": _usage_payload(state, services),
    }


def _serialize_subscription_preview(p) -> dict:
    return {
        "ok": True,
        "effect": p.effect,
        "reason": p.reason,
        "current_tier_id": p.current_tier_id,
        "current_tier_name": p.current_tier_name,
        "target_tier_id": p.target_tier_id,
        "target_tier_name": p.target_tier_name,
        "monthly_credits_delta": (
            None
            if p.monthly_credits_delta is None
            else str(p.monthly_credits_delta)
        ),
        "amount_due_now_cents": p.amount_due_now_cents,
        "effective_at": p.effective_at,
    }


def billing_state(rid, params: dict, services: BillingServices) -> dict:
    try:
        state = services.build_billing_state()
        return _ok(rid, _serialize_billing_state(state, services))
    except Exception:
        return _ok(
            rid,
            {"ok": True, "logged_in": False, "error": "could not load billing state"},
        )


def usage_bars(rid, params: dict, services: BillingServices) -> dict:
    try:
        return _ok(rid, _serialize_usage_model(services.build_usage_model()))
    except Exception:
        return _ok(rid, {"ok": True, "available": False})


def subscription_state(rid, params: dict, services: BillingServices) -> dict:
    try:
        state = services.build_subscription_state()
        return _ok(rid, _serialize_subscription_state(state, services))
    except Exception:
        return _ok(
            rid,
            {
                "ok": True,
                "logged_in": False,
                "error": "could not load subscription state",
            },
        )


def subscription_preview(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "invalid_request",
                "message": "subscription_type_id is required",
            },
        )
    try:
        preview = services.subscription_change_preview_from_payload(
            services.post_subscription_preview(subscription_type_id=tier_id)
        )
        return _ok(rid, _serialize_subscription_preview(preview))
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


def subscription_change(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    cancel = bool(params.get("cancel"))
    tier_id = params.get("subscription_type_id")
    if not cancel and not tier_id:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "invalid_request",
                "message": "subscription_type_id or cancel is required",
            },
        )
    try:
        result = services.put_subscription_pending_change(
            subscription_type_id=tier_id, cancel=cancel
        )
        return _ok(rid, {"ok": True, "message": result.get("message"), "payload": result})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


def subscription_resume(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    try:
        result = services.delete_subscription_pending_change()
        return _ok(rid, {"ok": True, "message": result.get("message"), "payload": result})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


def subscription_upgrade(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "invalid_request",
                "message": "subscription_type_id is required",
            },
        )
    key = params.get("idempotency_key") or services.new_idempotency_key()
    try:
        result = services.post_subscription_upgrade(
            subscription_type_id=tier_id, idempotency_key=key
        )
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "target_tier_name": result.get("targetTierName"),
                "recovery_url": result.get("recoveryUrl"),
                "reason": result.get("reason"),
                "idempotency_key": key,
            },
        )
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key
        return _ok(rid, env)
    except Exception as exc:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "error",
                "message": str(exc),
                "idempotency_key": key,
            },
        )


def billing_charge(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    amount = params.get("amount_usd")
    if amount is None:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "invalid_request",
                "message": "amount_usd is required",
            },
        )
    key = params.get("idempotency_key") or services.new_idempotency_key()
    try:
        result = services.post_charge(amount_usd=amount, idempotency_key=key)
        return _ok(
            rid,
            {
                "ok": True,
                "charge_id": result.get("chargeId"),
                "idempotency_key": key,
            },
        )
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key
        return _ok(rid, env)
    except Exception as exc:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "error",
                "message": str(exc),
                "idempotency_key": key,
            },
        )


def billing_charge_status(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    charge_id = params.get("charge_id")
    if not charge_id:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "invalid_charge_id",
                "message": "charge_id is required",
            },
        )
    try:
        result = services.get_charge_status(charge_id)
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "amount_usd": result.get("amountUsd"),
                "settled_at": result.get("settledAt"),
                "reason": result.get("reason"),
            },
        )
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


def billing_auto_reload(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    try:
        enabled = bool(params.get("enabled"))
        threshold = params.get("threshold")
        top_up_amount = params.get("top_up_amount")
        if threshold is None or top_up_amount is None:
            return _ok(
                rid,
                {
                    "ok": False,
                    "error": "invalid_request",
                    "message": "threshold and top_up_amount are required",
                },
            )
        services.patch_auto_top_up(
            enabled=enabled, threshold=threshold, top_up_amount=top_up_amount
        )
        return _ok(rid, {"ok": True})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


def billing_step_up(rid, params: dict, services: BillingServices) -> dict:
    from hermes_cli.nous_billing import BillingError

    sid = params.get("session_id") or ""
    try:

        def on_verification(url: str, code: str) -> None:
            services.emit(
                "billing.step_up.verification",
                sid,
                {"verification_url": url, "user_code": code},
            )

        granted = services.step_up_nous_billing_scope(
            open_browser=False, on_verification=on_verification
        )
        return _ok(rid, {"ok": True, "granted": bool(granted)})
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["granted"] = False
        return _ok(rid, env)
    except Exception as exc:
        return _ok(
            rid,
            {"ok": False, "error": "error", "message": str(exc), "granted": False},
        )


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: BillingServices,
) -> None:
    """Register ordinary billing callables under the existing JSON-RPC names."""

    def bind(handler: Callable[[Any, dict, BillingServices], dict]):
        return lambda rid, params: handler(rid, params, services)

    methods["billing.state"] = bind(billing_state)
    methods["usage.bars"] = bind(usage_bars)
    methods["subscription.state"] = bind(subscription_state)
    methods["subscription.preview"] = bind(subscription_preview)
    methods["subscription.change"] = bind(subscription_change)
    methods["subscription.resume"] = bind(subscription_resume)
    methods["subscription.upgrade"] = bind(subscription_upgrade)
    methods["billing.charge"] = bind(billing_charge)
    methods["billing.charge_status"] = bind(billing_charge_status)
    methods["billing.auto_reload"] = bind(billing_auto_reload)
    methods["billing.step_up"] = bind(billing_step_up)
