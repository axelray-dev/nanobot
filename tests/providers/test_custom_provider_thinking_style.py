"""Tests for custom provider thinking_style config option."""

from __future__ import annotations

import pytest

from nanobot.providers.registry import ProviderSpec, create_dynamic_spec


class TestCreateDynamicSpecThinkingStyle:
    """Verify create_dynamic_spec respects thinking_style parameter."""

    def test_default_thinking_style_is_empty(self) -> None:
        spec = create_dynamic_spec("my-custom")
        assert spec.thinking_style == ""

    def test_thinking_type_style(self) -> None:
        spec = create_dynamic_spec("volcengine", thinking_style="thinking_type")
        assert spec.thinking_style == "thinking_type"

    def test_enable_thinking_style(self) -> None:
        spec = create_dynamic_spec("dashscope", thinking_style="enable_thinking")
        assert spec.thinking_style == "enable_thinking"

    def test_reasoning_split_style(self) -> None:
        spec = create_dynamic_spec("minimax", thinking_style="reasoning_split")
        assert spec.thinking_style == "reasoning_split"

    def test_none_thinking_style_becomes_empty(self) -> None:
        spec = create_dynamic_spec("my-custom", thinking_style=None)
        assert spec.thinking_style == ""

    def test_empty_string_thinking_style(self) -> None:
        spec = create_dynamic_spec("my-custom", thinking_style="")
        assert spec.thinking_style == ""

    def test_spec_name_normalization_preserved(self) -> None:
        spec = create_dynamic_spec("My-Custom-Provider", thinking_style="thinking_type")
        assert spec.name == "my_custom_provider"
        assert spec.thinking_style == "thinking_type"

    def test_other_spec_fields_unchanged(self) -> None:
        spec = create_dynamic_spec("test", thinking_style="thinking_type")
        assert spec.backend == "openai_compat"
        assert spec.is_direct is True
        assert spec.env_key == ""
        assert spec.keywords == ()
