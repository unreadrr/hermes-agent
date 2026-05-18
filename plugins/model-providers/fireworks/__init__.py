"""Fireworks AI provider profile.

Fireworks hosts OpenAI-compatible inference for a curated catalog of open
weights (GLM-5.1, DeepSeek V4, Kimi K2, gpt-oss-120b, …) at
``https://api.fireworks.ai/inference/v1``.

Quirks
------
Several of the thinking-capable Fireworks models — most notably GLM-5.1 —
default to **reasoning ON**, which burns ``max_tokens`` budget on a private
``reasoning_content`` stream before the visible ``content`` is emitted.  For
auxiliary tasks (titles, compression, classification) this either truncates
the answer to a stub or wastes quota.  The Fireworks OpenAI-compat shape
accepts a top-level ``reasoning_effort`` parameter
(``"none" | "low" | "medium" | "high"``) that callers can use to opt out.

This profile translates Hermes' generic ``reasoning_config`` dict into
Fireworks' native ``reasoning_effort``:

    {"enabled": False}                → reasoning_effort="none"
    {"effort": "none"}                → reasoning_effort="none"
    {"effort": "low"|"medium"|"high"} → reasoning_effort=<effort>
    None / anything else              → omit (server default)

Reference: https://fireworks.ai/models/fireworks/glm-5p1 (reasoning_effort
parameter, 2026-05 catalog).
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class FireworksProfile(ProviderProfile):
    """Fireworks AI — top-level ``reasoning_effort`` for thinking models."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        top_level: dict[str, Any] = {}

        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled")
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if enabled is False or effort == "none":
                top_level["reasoning_effort"] = "none"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort
            # Any other value (incl. unset) → leave alone so Fireworks
            # applies its own per-model server default.

        return {}, top_level


fireworks = FireworksProfile(
    name="fireworks",
    aliases=("fw", "fireworks-ai"),
    env_vars=("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"),
    display_name="Fireworks AI",
    description="Fireworks AI (hosted open models — GLM-5.1, DeepSeek V4, Kimi K2, gpt-oss, …)",
    signup_url="https://fireworks.ai/account/api-keys",
    fallback_models=(
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/kimi-k2p5",
        "accounts/fireworks/models/gpt-oss-120b",
    ),
    base_url="https://api.fireworks.ai/inference/v1",
    default_aux_model="accounts/fireworks/models/glm-5p1",
)

register_provider(fireworks)
