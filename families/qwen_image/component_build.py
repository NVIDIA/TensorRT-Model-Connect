# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile existing edit components with separate TensorRT allocation lifetimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def _build_component(component: str, model: Path, output: Path, settings: dict) -> None:
    """Delegate to the original family graph and checkpoint loaders unchanged."""
    settings = dict(settings)
    if component == "text_encoder":
        from .qwen25_vl_text_encoder_builder import (
            build_qwen25vl_text_encoder_engine, load_qwen25vl_text_encoder_weights,
        )

        config, weights = load_qwen25vl_text_encoder_weights(model / "text_encoder",
            max_seq_len=settings.pop("max_seq_len"), apply_final_norm=settings.pop("apply_final_norm"))
        settings["image_grid_thw"] = tuple(settings["image_grid_thw"])
        build_qwen25vl_text_encoder_engine(config, weights, output, **settings)
    elif component == "denoiser":
        from .qwen_image_dit_builder import build_qwen_image_dit_engine, load_qwen_image_dit_weights

        config, weights = load_qwen_image_dit_weights(model / "transformer")
        settings["image_token_shapes"] = [tuple(shape) for shape in settings["image_token_shapes"]]
        build_qwen_image_dit_engine(config, weights, output, **settings)
    elif component == "vision":
        from .model import _load_qwen25vl_visual_weights
        from .qwen_vl_vision_builder import build_qwen_vl_vision_engine

        weights = _load_qwen25vl_visual_weights(str(model / "text_encoder"))
        output.write_bytes(build_qwen_vl_vision_engine(settings.pop("vision_config"), weights, **settings))
    elif component in {"vae_decoder", "vae_encoder"}:
        from .qwen_image_vae_builder import (
            build_qwen_image_vae_decoder_engine, build_qwen_image_vae_encoder_engine,
            load_qwen_image_vae_weights,
        )

        config, weights = load_qwen_image_vae_weights(model / "vae")
        if component == "vae_decoder":
            from .qwen_image_preprocessor import extract_preprocessor_source, pack_qwen_image_preprocessor_weights

            build_qwen_image_vae_decoder_engine(config, weights, output, **settings)
            output.with_suffix(".weights").write_bytes(
                pack_qwen_image_preprocessor_weights(extract_preprocessor_source(config)))
        else:
            build_qwen_image_vae_encoder_engine(config, weights, output, **settings)
    else:
        raise ValueError(f"unknown Qwen-Image component: {component}")


def build_edit_components(model: Path, config_json: bytes, settings: dict) -> dict:
    """Release each compiler's CUDA allocations before starting the next one.

    TensorRT build allocations can survive a Python builder's return and garbage
    collection. A process boundary also releases them before native validation or
    reference inference starts. This does not change any graph, precision or shape.
    """
    import torch

    device = torch.cuda.current_device()
    launcher = ("import json, runpy, sys; sys.path[:] = json.loads(sys.argv.pop(1)); "
                "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)")
    plans = {}
    with tempfile.TemporaryDirectory(prefix="qwen-image-edit-build-") as directory:
        root = Path(directory)
        for component in ("text_encoder", "denoiser", "vision", "vae_decoder", "vae_encoder"):
            output = root / f"{component}.plan"
            subprocess.run([sys.executable, "-c", launcher,
                json.dumps([str(path) for path in sys.path]), __spec__.name,
                component, str(model), str(output), str(device), json.dumps(settings[component])], check=True)
            plans[component] = output.read_bytes()
        preprocessor = (root / "vae_decoder.weights").read_bytes()
    return {"config_json": config_json, "text_encoders": [("qwen2_5_vl_lm", plans["text_encoder"])],
            "denoiser": plans["denoiser"], "vision_engine": plans["vision"],
            "vae_decoder": plans["vae_decoder"], "vae_encoder": plans["vae_encoder"],
            "preprocessor_weights": preprocessor}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("device", type=int)
    parser.add_argument("settings")
    args = parser.parse_args(argv)
    import torch

    torch.cuda.set_device(args.device)
    _build_component(args.component, args.model, args.output, json.loads(args.settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
