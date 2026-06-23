"""Flux-owned HF cache metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import family_hf_warm_files


def test_flux_declares_clip_metric_warm_file() -> None:
    assert family_hf_warm_files("flux") == [
        (
            "flux-clip-metrics-openclip",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "open_clip_pytorch_model.bin",
        )
    ]
