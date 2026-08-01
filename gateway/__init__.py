"""
Hermes Gateway - Multi-platform messaging integration.

This module provides a unified gateway for connecting the Hermes agent
to various messaging platforms (Telegram, Discord, WhatsApp, Weixin, and more) with:
- Session management (persistent conversations with reset policies)
- Dynamic context injection (agent knows where messages come from)
- Delivery routing (cron job outputs to appropriate channels)
- Platform-specific toolsets (different capabilities per platform)
"""

from .config import GatewayConfig, PlatformConfig, HomeChannel, load_gateway_config


def __getattr__(name):
    if name in {
        "SessionContext",
        "SessionStore",
        "SessionResetPolicy",
        "build_session_context_prompt",
    }:
        from . import session

        return getattr(session, name)
    if name in {"DeliveryRouter", "DeliveryTarget"}:
        from . import delivery

        return getattr(delivery, name)
    raise AttributeError(name)

__all__ = [
    # Config
    "GatewayConfig",
    "PlatformConfig", 
    "HomeChannel",
    "load_gateway_config",
    # Session
    "SessionContext",
    "SessionStore",
    "SessionResetPolicy",
    "build_session_context_prompt",
    # Delivery
    "DeliveryRouter",
    "DeliveryTarget",
]
