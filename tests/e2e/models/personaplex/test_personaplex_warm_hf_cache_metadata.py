"""PersonaPlex-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import family_hf_warm_dependencies


def test_personaplex_reference_dependencies_are_family_owned() -> None:
    deps = dict(family_hf_warm_dependencies("personaplex"))

    assert deps["personaplex-mimi-codec"] == "kyutai/mimi"
