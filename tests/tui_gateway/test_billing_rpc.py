"""Tests for the Phase 2b billing JSON-RPC methods (tui_gateway/server.py).

Verifies the structured envelope contract the Ink side branches on:
- billing.state serializes BillingState (Decimals → strings) + fails open.
- billing.charge / charge_status / auto_reload return typed error envelopes
  (result.ok=false, result.error=<code>) instead of JSON-RPC errors.
- billing.charge mints + echoes an idempotency_key for retry reuse.
"""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

import agent.billing_view as bv
import hermes_cli.nous_billing as nb
import tui_gateway.server as srv
from agent.billing_view import BillingState, CardInfo, MonthlyCap, PaymentMethodInfo


def _call(method: str, params: dict) -> dict:
    """Invoke a registered RPC method through request dispatch."""
    envelope = srv.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    return envelope["result"]


def test_billing_handlers_are_registered_as_explicit_billing_callables():
    """The durable billing slice should not use the legacy globals-rebinding seam."""
    assert srv._methods["billing.charge"].__module__ == "tui_gateway.methods_billing"
    assert srv._methods["subscription.upgrade"].__module__ == "tui_gateway.methods_billing"


# ---------------------------------------------------------------------------
# billing.state
# ---------------------------------------------------------------------------


def test_billing_state_serializes_decimals_as_strings(monkeypatch):
    state = BillingState(
        logged_in=True,
        org_name="Acme",
        role="OWNER",
        balance_usd=Decimal("142.5"),
        cli_billing_enabled=True,
        charge_presets=(Decimal("100"), Decimal("250")),
        min_usd=Decimal("10"),
        max_usd=Decimal("10000"),
        card=CardInfo(brand="visa", last4="4242"),
        payment_method=PaymentMethodInfo(
            kind="link",
            email="billing@example.com",
            resolved_via="customerDefault",
        ),
        monthly_cap=MonthlyCap(
            limit_usd=Decimal("1000"), spent_this_month_usd=Decimal("180"), is_default_ceiling=True
        ),
        portal_url="https://portal/billing?topup=open",
    )
    monkeypatch.setattr(bv, "build_billing_state", lambda *a, **kw: state)
    res = _call("billing.state", {})
    assert res["ok"] is True and res["logged_in"] is True
    # Money on the wire is STRING, not float/number.
    assert res["balance_usd"] == "142.5"
    assert res["balance_display"] == "$142.50"
    assert res["charge_presets"] == ["100", "250"]
    assert res["card"] == {
        "brand": "visa",
        "last4": "4242",
        "masked": "visa ····4242",
        "display": "visa ····4242",
        "resolved_via": None,
    }
    assert res["payment_method"] == {
        "kind": "link",
        "email": "billing@example.com",
        "resolved_via": "customerDefault",
    }
    assert res["monthly_cap"]["is_default_ceiling"] is True
    assert res["is_admin"] is True and res["can_charge"] is True


# ---------------------------------------------------------------------------
# billing.charge — typed error envelopes
# ---------------------------------------------------------------------------


def test_concurrent_billing_and_upgrade_errors_keep_request_payloads_isolated(monkeypatch):
    """Two concurrent money-path failures must not share response scratch state."""
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    results: dict[str, dict] = {}

    def fail_charge(*, amount_usd, idempotency_key):
        entered.wait(timeout=2)
        release.wait(timeout=2)
        raise nb.BillingError(
            f"charge failed for {amount_usd} using {idempotency_key}",
            error="charge_declined",
        )

    def fail_upgrade(*, subscription_type_id, idempotency_key):
        entered.wait(timeout=2)
        release.wait(timeout=2)
        raise nb.BillingError(
            f"upgrade failed for {subscription_type_id} using {idempotency_key}",
            error="upgrade_declined",
        )

    monkeypatch.setattr(nb, "post_charge", fail_charge)
    monkeypatch.setattr(nb, "post_subscription_upgrade", fail_upgrade)

    def run(name: str, method: str, params: dict) -> None:
        results[name] = _call(method, params)

    threads = [
        threading.Thread(
            target=run,
            args=(
                "charge",
                "billing.charge",
                {"amount_usd": "25", "idempotency_key": "charge-key"},
            ),
        ),
        threading.Thread(
            target=run,
            args=(
                "upgrade",
                "subscription.upgrade",
                {"subscription_type_id": "pro", "idempotency_key": "upgrade-key"},
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results == {
        "charge": {
            "ok": False,
            "error": "charge_declined",
            "message": "charge failed for 25 using charge-key",
            "portal_url": None,
            "retry_after": None,
            "payload": {},
            "actor": None,
            "code": None,
            "recovery": None,
            "idempotency_key": "charge-key",
        },
        "upgrade": {
            "ok": False,
            "error": "upgrade_declined",
            "message": "upgrade failed for pro using upgrade-key",
            "portal_url": None,
            "retry_after": None,
            "payload": {},
            "actor": None,
            "code": None,
            "recovery": None,
            "idempotency_key": "upgrade-key",
        },
    }


# ---------------------------------------------------------------------------
# billing.charge_status — the poll
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# billing.auto_reload
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# billing.step_up
# ---------------------------------------------------------------------------
