# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# ISO 26262 Traceability
# =============================================================================
# Trace ID:       UT-CFG-CLI-01
# Architecture:   ARCH-CFG-001
# Unit Design:    UD-CFG-CLI-01
# Intent:         --config/--set parsing, schema-driven type coercion, and the
#                 single-LayerContribution merge that feeds ConfigBundle.build.
# Preconditions:  None (no GPU, no TRT, no network).
# Postconditions: Parsing rejects malformed tokens; coercion follows schema
#                 type_tag; --set wins over --config within the session layer;
#                 unknown namespace/field is a fail-fast error.
# =============================================================================

"""Unit tests for ``tensorrt_model_connect.config.cli_support``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from tensorrt_model_connect.runtime_config import (
        ConfigBundle,
        ConfigField,
        Layer,
        LayerContribution,
        Schema,
        build_cli_contribution,
        clear_for_testing,
        coerce_scalar,
        load_layered_file,
        parse_set_token,
        parse_set_tokens,
        register_schema,
        resolve_cli_config,
        write_effective_config_next_to,
    )
except ImportError:  # pragma: no cover
    pytest.skip("tensorrt_model_connect.runtime_config not importable", allow_module_level=True)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    clear_for_testing()
    yield
    clear_for_testing()


def _register_demo_schema():
    """Standard demo schema used across several tests."""
    register_schema(Schema(
        namespace="triattention",
        fields=(
            ConfigField(
                name="kv_budget", type_tag="int32", default=6144,
                allowed_layers=frozenset({
                    Layer.SESSION_REQUEST,
                    Layer.PLATFORM_PROFILE,
                    Layer.BUNDLE_DEFAULT,
                }),
                validator=None,
            ),
            ConfigField(
                name="protect_prefill", type_tag="bool", default=True,
                allowed_layers=frozenset({
                    Layer.SESSION_REQUEST, Layer.BUNDLE_DEFAULT,
                }),
                validator=None,
            ),
            ConfigField(
                name="dump_scores_path", type_tag="string", default="",
                allowed_layers=frozenset({Layer.SESSION_REQUEST}),
                validator=None,
            ),
        ),
    ))


# ---- --set token parsing ---------------------------------------------------


def test_parse_set_token_basic():
    assert parse_set_token("ns.field=42") == ("ns", "field", "42")


def test_parse_set_token_allows_equals_in_value():
    # Value after the first '=' is preserved verbatim.
    assert parse_set_token("ns.field=a=b=c") == ("ns", "field", "a=b=c")


def test_parse_set_token_missing_equals():
    with pytest.raises(ValueError, match="missing '='"):
        parse_set_token("ns.field42")


def test_parse_set_token_missing_dot():
    with pytest.raises(ValueError, match=r"missing '\.'"):
        parse_set_token("nsfield=42")


def test_parse_set_token_empty_parts():
    with pytest.raises(ValueError, match="empty"):
        parse_set_token(".field=42")
    with pytest.raises(ValueError, match="empty"):
        parse_set_token("ns.=42")


def test_parse_set_tokens_last_write_wins():
    result = parse_set_tokens([
        "ns.field=first",
        "ns.field=second",
        "ns.field=third",
    ])
    assert result == {"ns": {"field": "third"}}


def test_parse_set_tokens_groups_by_namespace():
    result = parse_set_tokens([
        "ns_a.f1=1", "ns_a.f2=2", "ns_b.f3=3",
    ])
    assert result == {"ns_a": {"f1": "1", "f2": "2"}, "ns_b": {"f3": "3"}}


# ---- type coercion ---------------------------------------------------------


def test_coerce_int():
    assert coerce_scalar("42", "int32", "x.y") == 42
    assert coerce_scalar("-7", "int64", "x.y") == -7


def test_coerce_int_rejects_float_text():
    with pytest.raises(ValueError, match="expected integer"):
        coerce_scalar("3.14", "int32", "x.y")


def test_coerce_float():
    assert coerce_scalar("3.14", "float", "x.y") == pytest.approx(3.14)


def test_coerce_bool_true_vocab():
    for token in ("true", "True", "TRUE", "1", "yes", "on"):
        assert coerce_scalar(token, "bool", "x.y") is True


def test_coerce_bool_false_vocab():
    for token in ("false", "False", "FALSE", "0", "no", "off"):
        assert coerce_scalar(token, "bool", "x.y") is False


def test_coerce_bool_rejects_unknown():
    with pytest.raises(ValueError, match="expected bool"):
        coerce_scalar("maybe", "bool", "x.y")


def test_coerce_string_identity():
    assert coerce_scalar("hello", "string", "x.y") == "hello"
    assert coerce_scalar("/tmp/foo", "path", "x.y") == "/tmp/foo"


def test_coerce_unknown_type_tag_raises():
    with pytest.raises(ValueError, match="unsupported type_tag"):
        coerce_scalar("anything", "list<int>", "x.y")


# ---- load_layered_file -----------------------------------------------------


def test_load_json_profile(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }))
    loaded = load_layered_file(path)
    assert loaded == {
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }


def test_load_yaml_profile_if_available(tmp_path: Path):
    pytest.importorskip("yaml")
    path = tmp_path / "profile.yaml"
    path.write_text("triattention:\n  kv_budget: 4096\n  protect_prefill: true\n")
    loaded = load_layered_file(path)
    assert loaded == {
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_layered_file(tmp_path / "nope.json")


def test_load_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "profile.toml"
    path.write_text("[triattention]\nkv_budget = 4096")
    with pytest.raises(ValueError, match="unsupported extension"):
        load_layered_file(path)


def test_load_rejects_non_mapping_top_level(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_layered_file(path)


def test_load_rejects_non_dict_namespace_body(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"triattention": 4096}))
    with pytest.raises(ValueError, match="must map to a dict"):
        load_layered_file(path)


# ---- build_cli_contribution merging + validation ---------------------------


def test_build_cli_contribution_from_config_only():
    _register_demo_schema()
    contrib = build_cli_contribution(
        config_file_values={
            "triattention": {"kv_budget": 4096, "protect_prefill": False},
        },
        set_tokens=None,
    )
    assert contrib.layer == Layer.SESSION_REQUEST
    assert contrib.values == {
        "triattention": {"kv_budget": 4096, "protect_prefill": False},
    }


def test_build_cli_contribution_coerces_set_tokens_per_schema():
    _register_demo_schema()
    contrib = build_cli_contribution(
        config_file_values=None,
        set_tokens=[
            "triattention.kv_budget=8192",
            "triattention.protect_prefill=false",
            "triattention.dump_scores_path=/tmp/x.pt",
        ],
    )
    assert contrib.values == {
        "triattention": {
            "kv_budget": 8192,
            "protect_prefill": False,
            "dump_scores_path": "/tmp/x.pt",
        },
    }


def test_set_overrides_config_within_session_layer():
    _register_demo_schema()
    contrib = build_cli_contribution(
        config_file_values={"triattention": {"kv_budget": 4096}},
        set_tokens=["triattention.kv_budget=8192"],
    )
    assert contrib.values["triattention"]["kv_budget"] == 8192


def test_build_cli_contribution_unknown_namespace_raises():
    _register_demo_schema()
    with pytest.raises(ValueError, match="unknown namespace 'missing'"):
        build_cli_contribution(
            config_file_values={"missing": {"x": 1}},
        )


def test_build_cli_contribution_unknown_field_raises():
    _register_demo_schema()
    with pytest.raises(ValueError, match="unknown field 'nope'"):
        build_cli_contribution(
            set_tokens=["triattention.nope=1"],
        )


def test_set_coercion_error_surfaces_field_name():
    _register_demo_schema()
    with pytest.raises(ValueError, match="triattention.kv_budget"):
        build_cli_contribution(
            set_tokens=["triattention.kv_budget=not_a_number"],
        )


# ---- resolve_cli_config (end-to-end) --------------------------------------


def test_resolve_cli_config_builds_bundle(tmp_path: Path):
    _register_demo_schema()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "triattention": {"kv_budget": 4096, "protect_prefill": False},
    }))
    bundle = resolve_cli_config(
        config_path=str(profile),
        set_tokens=["triattention.kv_budget=8192"],
    )
    assert bundle.get("triattention", "kv_budget") == 8192
    assert bundle.get("triattention", "protect_prefill") is False
    assert bundle.source_of("triattention", "kv_budget") == Layer.SESSION_REQUEST
    assert bundle.get("triattention", "dump_scores_path") == ""  # schema default


def test_resolve_cli_config_with_extra_platform_layer():
    _register_demo_schema()
    platform_contrib = LayerContribution(
        layer=Layer.PLATFORM_PROFILE,
        values={"triattention": {"kv_budget": 10240}},
    )
    bundle = resolve_cli_config(
        config_path=None, set_tokens=None,
        extra_contributions=[platform_contrib],
    )
    assert bundle.get("triattention", "kv_budget") == 10240
    assert bundle.source_of("triattention", "kv_budget") == Layer.PLATFORM_PROFILE


def test_resolve_cli_config_session_beats_platform():
    _register_demo_schema()
    platform_contrib = LayerContribution(
        layer=Layer.PLATFORM_PROFILE,
        values={"triattention": {"kv_budget": 10240}},
    )
    bundle = resolve_cli_config(
        config_path=None,
        set_tokens=["triattention.kv_budget=8192"],
        extra_contributions=[platform_contrib],
    )
    # Session layer (from --set) beats platform layer.
    assert bundle.get("triattention", "kv_budget") == 8192
    assert bundle.source_of("triattention", "kv_budget") == Layer.SESSION_REQUEST


# ---- write_effective_config_next_to ---------------------------------------


def test_write_effective_config_next_to_uses_suffix(tmp_path: Path):
    _register_demo_schema()
    bundle = resolve_cli_config(
        set_tokens=["triattention.kv_budget=8192"],
    )
    bundle_path = tmp_path / "some" / "bundle.bundle"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    written = write_effective_config_next_to(bundle, bundle_path)
    assert written == tmp_path / "some" / "bundle.effective_config.json"
    loaded = json.loads(written.read_text())
    assert loaded["triattention"]["kv_budget"] == {
        "value": 8192, "source": "session_request",
    }


# ---- bundle defaults: block -----------------------------------------------


def test_bundle_defaults_contribution_reads_block():
    from tensorrt_model_connect.runtime_config import bundle_defaults_contribution
    header = json.dumps({
        "model_id": "demo",
        "vocab_size": 100,
        "defaults": {
            "triattention": {"kv_budget": 4096, "protect_prefill": True},
        },
        "sections": {},
    })
    contrib = bundle_defaults_contribution(header)
    assert contrib.layer == Layer.BUNDLE_DEFAULT
    assert contrib.values == {
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }


def test_bundle_defaults_contribution_absent_block_is_empty():
    from tensorrt_model_connect.runtime_config import bundle_defaults_contribution
    header = json.dumps({"model_id": "demo", "sections": {}})
    contrib = bundle_defaults_contribution(header)
    assert contrib.layer == Layer.BUNDLE_DEFAULT
    assert contrib.values == {}


def test_bundle_defaults_contribution_accepts_mapping():
    from tensorrt_model_connect.runtime_config import bundle_defaults_contribution
    parsed = {"defaults": {"ns": {"field": 1}}, "sections": {}}
    contrib = bundle_defaults_contribution(parsed)
    assert contrib.values == {"ns": {"field": 1}}


def test_bundle_defaults_feeds_bundle_default_layer(tmp_path: Path):
    from tensorrt_model_connect.runtime_config import bundle_defaults_contribution
    _register_demo_schema()
    header = json.dumps({"defaults": {"triattention": {"kv_budget": 4096}}})
    defaults_contrib = bundle_defaults_contribution(header)
    # Session value beats bundle_default; bundle_default beats schema default.
    bundle = ConfigBundle.build(
        [defaults_contrib] + [LayerContribution(
            layer=Layer.SESSION_REQUEST,
            values={"triattention": {"kv_budget": 8192}},
        )]
    )
    assert bundle.get("triattention", "kv_budget") == 8192
    assert bundle.source_of("triattention", "kv_budget") == Layer.SESSION_REQUEST
    # Without session override, bundle default wins.
    bundle2 = ConfigBundle.build([defaults_contrib])
    assert bundle2.get("triattention", "kv_budget") == 4096
    assert bundle2.source_of("triattention", "kv_budget") == Layer.BUNDLE_DEFAULT


def test_bundle_writer_round_trip_with_defaults(tmp_path: Path):
    """End-to-end: Python builder writes a .bundle, reader gets defaults back."""
    pytest.importorskip("struct")
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, BUNDLE_MAGIC, write_bundle
    import struct

    path = tmp_path / "bundle.bundle"
    info = BundleInfo(
        model_id="demo", vocab_size=100, hidden_size=16, num_layers=1,
        defaults={"triattention": {"kv_budget": 4096, "protect_prefill": True}},
    )
    write_bundle(path, info, [BundleSection(name="dummy", data=b"\x00\x00")])

    # Re-parse the header JSON directly to confirm round-trip.
    raw = path.read_bytes()
    assert raw.startswith(BUNDLE_MAGIC)
    header_len = struct.unpack("<Q", raw[8:16])[0]
    header_text = raw[16:16 + header_len].decode("utf-8")
    header = json.loads(header_text)
    assert header["defaults"] == {
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }

    # Feed into the registry helper.
    from tensorrt_model_connect.runtime_config import bundle_defaults_contribution
    contrib = bundle_defaults_contribution(header_text)
    assert contrib.values == {
        "triattention": {"kv_budget": 4096, "protect_prefill": True},
    }


def test_bundle_writer_omits_defaults_when_empty(tmp_path: Path):
    """No defaults: block → old readers unaffected."""
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    import struct

    path = tmp_path / "bundle.bundle"
    info = BundleInfo(model_id="demo", vocab_size=100, hidden_size=16, num_layers=1)
    write_bundle(path, info, [BundleSection(name="dummy", data=b"\x00\x00")])

    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[8:16])[0]
    header = json.loads(raw[16:16 + header_len].decode("utf-8"))
    assert "defaults" not in header
