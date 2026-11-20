# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer tool: calibrate BEVFusion FP8 on reproducible synthetic BEV features.

Requires nvidia-modelopt and the declared reference inputs prepared by
families.bevfusion.tests.reference. These scales qualify the feature-input
performance workload, not dataset-level segmentation accuracy. Calibration
defaults to seeds 32..63; retained validation uses separate seeds 0..7.
"""

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np

from .model import INPUT_SHAPES, Weights
from .quantization import (
    CONVOLUTIONS,
    DEFAULT_EXCLUSIONS,
    FP8_MAX,
    MIN_SCALE,
)

FIRST_SEED = 32
SAMPLES = 32


def save_checkpoint(output, values, document):
    """Keep calibrated scales and their complete source weights in one artifact."""
    import torch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        torch.save({"state_dict": values, "bevfusion_quantization": document}, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate(model_dir, output, *, first_seed=FIRST_SEED, samples=SAMPLES, exclude=None):
    """Run ModelOpt once and save the exact calibrated BEV weights with their scales."""
    exclude = set(DEFAULT_EXCLUSIONS if exclude is None else exclude)
    if first_seed < 0 or samples < 1 or not exclude < set(CONVOLUTIONS):
        raise ValueError(
            "calibration requires nonnegative seeds, positive samples and known layers"
        )
    model_dir, output = Path(model_dir), Path(output)
    checkpoint = model_dir / "bevfusion-seg.pth"
    if checkpoint.resolve() == output.resolve():
        raise ValueError("calibration output must differ from the source checkpoint")
    import modelopt.torch.quantization as mtq
    import torch

    from .tests.reference import Reference, SOURCE_REVISION

    weights = Weights(checkpoint)
    config = {"quant_cfg": {"*": {"enable": False}}, "algorithm": "max"}
    for name in CONVOLUTIONS:
        if name in exclude:
            continue
        for suffix, axis in (("input_quantizer", None), ("weight_quantizer", 0)):
            config["quant_cfg"][f"{name}.{suffix}"] = {
                "enable": True,
                "num_bits": (4, 3),
                "axis": axis,
            }
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=[]):
            model = Reference(model_dir / "reference", state_dict=weights.values).cuda().eval()

        def forward_loop(calibration_model):
            with torch.inference_mode():
                for seed in range(first_seed, first_seed + samples):
                    rng = np.random.default_rng(seed)
                    inputs = [
                        torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)).cuda()
                        for shape in INPUT_SHAPES.values()
                    ]
                    calibration_model(*inputs)

        model = mtq.quantize(model, config, forward_loop)
        modules = dict(model.named_modules())
        scales = {}
        for name in CONVOLUTIONS:
            if name in exclude:
                scales[name] = None
                continue
            amax = modules[name].input_quantizer.amax
            if amax is None or not np.isfinite(float(amax)) or float(amax) <= 0:
                raise ValueError(f"ModelOpt did not calibrate {name}")
            scales[name] = float(max(float(amax) / FP8_MAX, MIN_SCALE))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    document = {
        "version": 1,
        "calibration": {
            "checkpoint_file": checkpoint.name,
            "architecture_revision": SOURCE_REVISION,
            "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
            "torch_version": str(torch.__version__),
            "algorithm": "max",
            "input_distribution": "numpy default_rng standard_normal float32",
            "input_shapes": INPUT_SHAPES,
            "first_seed": first_seed,
            "samples": samples,
            "excluded_layers": sorted(exclude),
        },
        "layers": scales,
    }
    save_checkpoint(output, weights.values, document)
    print(f"saved calibrated checkpoint with {len(scales) - len(exclude)} FP8 layers to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--exclude", action="append", choices=CONVOLUTIONS)
    args = parser.parse_args()
    calibrate(
        args.model_dir,
        args.output,
        first_seed=args.first_seed,
        samples=args.samples,
        exclude=args.exclude,
    )
