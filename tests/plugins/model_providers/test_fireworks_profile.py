"""Unit tests for the Fireworks AI provider profile.

Fireworks hosts OpenAI-compatible inference for GLM-5.1, DeepSeek V4, Kimi
K2, gpt-oss, and others.  Several of those models default to reasoning-on,
which burns the caller's ``max_tokens`` budget on a private reasoning
stream.  Fireworks exposes a top-level ``reasoning_effort`` parameter so
callers can opt out — these tests pin the profile's translation contract
without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fireworks_profile():
    """Resolve the registered Fireworks profile via the provider registry."""
    # Triggers plugin discovery, registers the Fireworks profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("fireworks")
    assert profile is not None, "fireworks provider profile must be registered"
    return profile


class TestFireworksReasoningEffortWireShape:
    """``build_api_kwargs_extras`` produces Fireworks' ``reasoning_effort`` shape."""

    def test_no_reasoning_config_omits_effort(self, fireworks_profile):
        """No reasoning_config → server default (omit reasoning_effort)."""
        extra_body, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config=None, model="accounts/fireworks/models/glm-5p1"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_disabled_maps_to_none(self, fireworks_profile):
        """``enabled=False`` → ``reasoning_effort="none"`` so GLM-5.1 doesn't
        burn tokens on hidden thinking on auxiliary calls (titles, compression).
        """
        extra_body, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "none"}

    def test_effort_none_maps_to_none(self, fireworks_profile):
        """``effort="none"`` is the canonical way to opt out per Fireworks docs."""
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_standard_efforts_pass_through(self, fireworks_profile, effort):
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level == {"reasoning_effort": effort}

    @pytest.mark.parametrize("effort", ["LOW", "  Medium  ", "High"])
    def test_efforts_normalized_lowercase_stripped(self, fireworks_profile, effort):
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level["reasoning_effort"] in {"low", "medium", "high"}

    def test_unknown_effort_omitted(self, fireworks_profile):
        """Garbage effort → omit reasoning_effort so the server applies default."""
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level == {}

    def test_empty_effort_omitted(self, fireworks_profile):
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level == {}

    def test_disabled_with_effort_field_still_off(self, fireworks_profile):
        """``enabled=False`` wins over any ``effort`` setting."""
        _, top_level = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="accounts/fireworks/models/glm-5p1",
        )
        assert top_level == {"reasoning_effort": "none"}


class TestFireworksRegistration:
    """Profile metadata is wired up correctly for the picker + auth registry."""

    def test_profile_name_and_aliases(self, fireworks_profile):
        assert fireworks_profile.name == "fireworks"
        assert "fw" in fireworks_profile.aliases
        assert "fireworks-ai" in fireworks_profile.aliases

    def test_base_url_pinned(self, fireworks_profile):
        assert fireworks_profile.base_url == "https://api.fireworks.ai/inference/v1"

    def test_env_vars_include_api_key(self, fireworks_profile):
        assert "FIREWORKS_API_KEY" in fireworks_profile.env_vars

    def test_glm_5p1_in_fallback_models(self, fireworks_profile):
        """GLM-5.1 must show up in the picker even when /v1/models is unreachable."""
        assert "accounts/fireworks/models/glm-5p1" in fireworks_profile.fallback_models

    def test_default_aux_model_is_glm_5p1(self, fireworks_profile):
        assert fireworks_profile.default_aux_model == "accounts/fireworks/models/glm-5p1"


class TestFireworksAuxModelConsumer:
    """The auxiliary client picks up the aux model from the profile."""

    def test_consumer_api_returns_glm_5p1(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert (
            _get_aux_model_for_provider("fireworks")
            == "accounts/fireworks/models/glm-5p1"
        )

    def test_consumer_api_returns_non_empty(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("fireworks") != ""


class TestFireworksAutoRegistration:
    """The plugin auto-extends PROVIDER_REGISTRY + CANONICAL_PROVIDERS so it
    surfaces in the model picker and the auth/credential resolver.
    """

    def test_provider_registry_includes_fireworks(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "fireworks" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["fireworks"]
        assert cfg.auth_type == "api_key"
        assert "FIREWORKS_API_KEY" in cfg.api_key_env_vars
        assert cfg.inference_base_url == "https://api.fireworks.ai/inference/v1"
        assert cfg.base_url_env_var == "FIREWORKS_BASE_URL"

    def test_canonical_providers_includes_fireworks(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = {p.slug for p in CANONICAL_PROVIDERS}
        assert "fireworks" in slugs

    def test_provider_model_ids_returns_glm_5p1_in_fallback(self, monkeypatch):
        """Without an API key, picker falls back to the profile's curated list."""
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
        from hermes_cli.models import provider_model_ids

        models = provider_model_ids("fireworks")
        assert "accounts/fireworks/models/glm-5p1" in models
