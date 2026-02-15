"""Tests for config placeholder resolution (YAML ${...} substitution)."""

import pytest

from src.utils.config import lookup_path, resolve_placeholders


def test_lookup_path_simple():
    cfg = {"a": {"b": "value"}}
    assert lookup_path(cfg, "a.b") == "value"
    assert lookup_path(cfg, "a") == {"b": "value"}
    assert lookup_path(cfg, "x") is None
    assert lookup_path(cfg, "a.x") is None


def test_lookup_path_nested():
    cfg = {
        "paths": {
            "checkpoints": {
                "age": "checkpoints/age.pt",
                "gender": "checkpoints/gender.pt",
            }
        }
    }
    assert lookup_path(cfg, "paths.checkpoints.age") == "checkpoints/age.pt"
    assert lookup_path(cfg, "paths.checkpoints") == {
        "age": "checkpoints/age.pt",
        "gender": "checkpoints/gender.pt",
    }


def test_resolve_placeholders_leaves_non_strings_unchanged():
    assert resolve_placeholders(42, {}) == 42
    assert resolve_placeholders(3.14, {}) == 3.14
    assert resolve_placeholders(True, {}) is True


def test_resolve_placeholders_string_without_placeholder():
    cfg = {}
    assert resolve_placeholders("hello", cfg) == "hello"
    assert resolve_placeholders("data/UTKFace", cfg) == "data/UTKFace"


def test_resolve_placeholders_single_placeholder():
    cfg = {"paths": {"data": {"utkface": "data/UTKFace"}}}
    out = resolve_placeholders("${paths.data.utkface}", cfg)
    assert out == "data/UTKFace"


def test_resolve_placeholders_missing_key_keeps_literal():
    cfg = {"paths": {}}
    out = resolve_placeholders("${paths.checkpoints.age}", cfg)
    assert out == "${paths.checkpoints.age}"


def test_resolve_placeholders_dict_recursive():
    cfg = {"paths": {"checkpoints": {"age": "ckpt/age.pt"}}}
    obj = {
        "data_dir": "${paths.checkpoints.age}",
        "nested": {"x": "${paths.checkpoints.age}"},
    }
    out = resolve_placeholders(obj, cfg)
    assert out["data_dir"] == "ckpt/age.pt"
    assert out["nested"]["x"] == "ckpt/age.pt"


def test_resolve_placeholders_list_recursive():
    cfg = {"x": "replaced"}
    obj = ["a", "${x}", "b"]
    out = resolve_placeholders(obj, cfg)
    assert out == ["a", "replaced", "b"]
