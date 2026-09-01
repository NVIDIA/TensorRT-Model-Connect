# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the exact training-revision LeRobot ACT PyTorch inference path."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

_CHECKPOINT_SHA256 = "42772891cb6eba1e7bc36ad8e12c0fa0723c61f036fa235c725ce6026e6e81df"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_draccus_import_shim() -> None:
    """Provide registration-only draccus APIs; parsing is performed explicitly below."""

    class ChoiceRegistry:
        @classmethod
        def register_subclass(cls, name):
            def decorate(subclass):
                subclass._trtmc_choice_name = name
                return subclass

            return decorate

        @classmethod
        def get_choice_name(cls, subclass):
            del cls
            return getattr(subclass, "_trtmc_choice_name", subclass.__name__.lower())

    @contextlib.contextmanager
    def config_type(*args, **kwargs):
        del args, kwargs
        yield

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("draccus parsing is disabled in the pinned ACT reference")

    module = types.ModuleType("draccus")
    module.ChoiceRegistry = ChoiceRegistry
    module.config_type = config_type
    module.parse = unavailable
    module.dump = unavailable
    sys.modules["draccus"] = module


def _load_config(config_path: Path):
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.pop("type", None) != "act":
        raise ValueError("pinned policy config is not ACT")
    payload.pop("device", None)
    payload.pop("use_amp", None)
    # The checkpoint replaces the complete backbone. Avoid a semantically dead
    # ImageNet download so the exact-source reference remains offline-capable.
    payload["pretrained_backbone_weights"] = None
    payload["normalization_mapping"] = {
        FeatureType[key]: NormalizationMode[value]
        for key, value in payload["normalization_mapping"].items()
    }
    for field in ("input_features", "output_features"):
        payload[field] = {
            key: PolicyFeature(
                type=FeatureType[value["type"]],
                shape=tuple(value["shape"]),
            )
            for key, value in payload[field].items()
        }
    config = ACTConfig(**payload)
    config.validate_features()
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    entrypoint = arguments.source_root / "lerobot/common/policies/act/modeling_act.py"
    if not entrypoint.is_file():
        raise FileNotFoundError("source root is not the pinned LeRobot checkout")
    checkpoint = arguments.checkpoint_dir / "model.safetensors"
    config_path = arguments.checkpoint_dir / "config.json"
    if _sha256(checkpoint) != _CHECKPOINT_SHA256:
        raise ValueError("ACT checkpoint SHA-256 does not match the qualified artifact")

    import torch
    from PIL import Image
    from safetensors.torch import load_model

    _install_draccus_import_shim()
    sys.path.insert(0, str(arguments.source_root.resolve()))
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    config = _load_config(config_path)
    policy = ACTPolicy(config)
    load_model(policy, str(checkpoint), strict=True, device="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    pixels = np.asarray(Image.open(arguments.image).convert("RGB"), dtype=np.float32)
    state = np.fromfile(arguments.state, dtype="<f4")
    if pixels.shape != (480, 640, 3) or state.shape != (14,):
        raise ValueError("recorded ACT observation does not have the qualified shape")
    batch = {
        "observation.images.top": torch.from_numpy(pixels / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device),
        "observation.state": torch.from_numpy(state.copy()).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        batch = policy.normalize_inputs(batch)
        batch["observation.images"] = torch.stack(
            [batch[key] for key in config.image_features], dim=-4
        )
        actions = policy.model(batch)[0][:, : config.n_action_steps]
        actions = policy.unnormalize_outputs({"action": actions})["action"]
    result = actions.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (100, 14) or not np.isfinite(result).all():
        raise ValueError(f"ACT reference returned invalid actions {result.shape}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, actions=result)
    print(json.dumps({"num_actions": 100, "action_dim": 14}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
