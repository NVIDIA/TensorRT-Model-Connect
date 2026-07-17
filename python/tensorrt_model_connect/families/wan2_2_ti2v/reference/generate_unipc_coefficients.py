# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit source-exact scalar coefficients for Wan2.2's fixed 50-step UniPC run.

This is a qualification utility, not part of the Model-Connect runtime.  In
particular, the order-two corrector coefficients are solved by the official
PyTorch implementation's CUDA path so their IEEE-754 encodings can be carried
into a Python-free CUDA implementation without recomputing them differently.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import torch


NUM_INFERENCE_STEPS = 50
NUM_TRAIN_TIMESTEPS = 1000
FLOW_SHIFT = 5.0
SOLVER_ORDER = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact CUDA scalar coefficients for Wan2.2 TI2V-5B UniPC."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--official-source",
        type=Path,
        default=Path(os.environ.get("WAN22_OFFICIAL_SOURCE", "/workspace/Wan2.2-official")),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--autocast-bf16",
        action="store_true",
        help="Generate coefficients under Wan2.2's outer BF16 autocast context",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _load_scheduler(source_root: Path):
    module_path = source_root / "wan" / "utils" / "fm_solvers_unipc.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official Wan2.2 UniPC source is missing: {module_path}")
    spec = importlib.util.spec_from_file_location("wan22_official_unipc_coefficients", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official UniPC module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler, module_path


def _float32(value: torch.Tensor | float) -> dict[str, int | float | str]:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected one float32 scalar, got shape {tuple(value.shape)}")
        scalar = value.detach().to(dtype=torch.float32).reshape(())
    else:
        scalar = torch.tensor(value, dtype=torch.float32)
    bits = int(scalar.view(torch.int32).item()) & 0xFFFFFFFF
    return {
        "float": float(scalar.item()),
        "uint32": bits,
        "uint32_hex": f"0x{bits:08x}",
    }


def _lambda_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    alpha, sigma = 1 - sigma, sigma
    return torch.log(alpha) - torch.log(sigma)


def _transition_scalars(
    sigma_t: torch.Tensor,
    sigma_s0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror the official CPU scalar operation boundaries for one transition."""
    alpha_t, sigma_t = 1 - sigma_t, sigma_t
    _, sigma_s0 = 1 - sigma_s0, sigma_s0
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(1 - sigma_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0
    hh = -h
    h_phi_1 = torch.expm1(hh)
    # The upstream source invokes expm1 a second time for BH2 rather than
    # aliasing h_phi_1.  Preserve that operation boundary here.
    b_h = torch.expm1(hh)
    ratio = sigma_t / sigma_s0
    model_coefficient = alpha_t * h_phi_1
    residual_coefficient = alpha_t * b_h
    return ratio, model_coefficient, residual_coefficient, h


def _rk_vector(
    sigmas: torch.Tensor,
    *,
    step_index: int,
    order: int,
    corrector: bool,
    h: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if corrector:
        sigma_s0_index = step_index - 1
    else:
        sigma_s0_index = step_index
    lambda_s0 = _lambda_from_sigma(sigmas[sigma_s0_index])

    rks: list[torch.Tensor | float] = []
    for history_offset in range(1, order):
        if corrector:
            history_index = step_index - (history_offset + 1)
        else:
            history_index = step_index - history_offset
        lambda_si = _lambda_from_sigma(sigmas[history_index])
        rks.append((lambda_si - lambda_s0) / h)
    rks.append(1.0)
    # This is intentionally torch.tensor(list, device=...) rather than stack:
    # it is the conversion used by the official scheduler.
    return torch.tensor(rks, device=device)


def _rho_vector(
    rks: torch.Tensor,
    *,
    h: torch.Tensor,
    order: int,
    corrector: bool,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Reproduce the official BH2 R/b construction and rho selection."""
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    b_h = torch.expm1(hh)
    factorial_i = 1
    rows: list[torch.Tensor] = []
    rhs_values: list[torch.Tensor] = []
    for index in range(1, order + 1):
        rows.append(torch.pow(rks, index - 1))
        rhs_values.append(h_phi_k * factorial_i / b_h)
        factorial_i *= index + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i

    matrix = torch.stack(rows)
    rhs = torch.tensor(rhs_values, device=device)
    if corrector:
        if order == 1:
            return torch.tensor([0.5], dtype=torch.float32, device=device), "simplified_0.5"
        # This must stay on CUDA.  The generated bits qualify the exact solve
        # implementation used by the official scheduler on this device.
        return (
            torch.linalg.solve(matrix, rhs).to(device).to(torch.float32),
            "torch.linalg.solve_cuda",
        )

    if order == 1:
        return torch.empty(0, dtype=torch.float32, device=device), "not_used"
    if order == 2:
        return torch.tensor([0.5], dtype=torch.float32, device=device), "simplified_0.5"
    return (
        torch.linalg.solve(matrix[:-1, :-1], rhs[:-1]).to(device).to(torch.float32),
        "torch.linalg.solve_cuda",
    )


def _coefficient_record(
    scheduler,
    *,
    step_index: int,
    order: int,
    corrector: bool,
    device: torch.device,
) -> dict[str, Any]:
    sigmas = scheduler.sigmas
    if corrector:
        sigma_t_index = step_index
        sigma_s0_index = step_index - 1
    else:
        sigma_t_index = step_index + 1
        sigma_s0_index = step_index
    sigma_t = sigmas[sigma_t_index]
    sigma_s0 = sigmas[sigma_s0_index]
    ratio, model_coefficient, residual_coefficient, h = _transition_scalars(sigma_t, sigma_s0)
    rks = _rk_vector(
        sigmas,
        step_index=step_index,
        order=order,
        corrector=corrector,
        h=h,
        device=device,
    )
    rhos, rho_source = _rho_vector(
        rks,
        h=h,
        order=order,
        corrector=corrector,
        device=device,
    )
    return {
        "order": order,
        "sigma_t_index": sigma_t_index,
        "sigma_s0_index": sigma_s0_index,
        "ratio": _float32(ratio),
        "coefficient": _float32(model_coefficient),
        "residual_coefficient": _float32(residual_coefficient),
        "rk": [_float32(value) for value in rks.unbind()],
        "rho": [_float32(value) for value in rhos.unbind()],
        "rho_source": rho_source,
    }


def _make_payload(
    scheduler,
    *,
    source_root: Path,
    source_file: Path,
    device: torch.device,
) -> dict[str, Any]:
    sigmas = scheduler.sigmas
    timesteps = scheduler.timesteps
    if sigmas.dtype != torch.float32 or tuple(sigmas.shape) != (NUM_INFERENCE_STEPS + 1,):
        raise RuntimeError(
            f"Unexpected official sigma contract: dtype={sigmas.dtype}, shape={tuple(sigmas.shape)}"
        )
    if timesteps.dtype != torch.int64 or tuple(timesteps.shape) != (NUM_INFERENCE_STEPS,):
        raise RuntimeError(
            "Unexpected official timestep contract: "
            f"dtype={timesteps.dtype}, shape={tuple(timesteps.shape)}"
        )

    lower_order_nums = 0
    previous_order = 0
    steps: list[dict[str, Any]] = []
    for step_index, timestep in enumerate(timesteps):
        corrector_record = None
        if step_index > 0:
            if previous_order <= 0:
                raise RuntimeError("Official corrector order tracking became invalid")
            corrector_record = _coefficient_record(
                scheduler,
                step_index=step_index,
                order=previous_order,
                corrector=True,
                device=device,
            )

        remaining = NUM_INFERENCE_STEPS - step_index
        available_order = min(SOLVER_ORDER, lower_order_nums + 1)
        predictor_order = min(remaining, available_order)
        predictor_record = _coefficient_record(
            scheduler,
            step_index=step_index,
            order=predictor_order,
            corrector=False,
            device=device,
        )
        timestep_int = int(timestep.item())
        steps.append(
            {
                "step_index": step_index,
                "timestep": {
                    "int64": timestep_int,
                    "as_float32": _float32(float(timestep_int)),
                },
                "conversion_sigma": _float32(sigmas[step_index]),
                "corrector": corrector_record,
                "predictor": predictor_record,
            }
        )
        previous_order = predictor_order
        lower_order_nums = min(SOLVER_ORDER, lower_order_nums + 1)

    properties = torch.cuda.get_device_properties(device)
    source_status = _git(source_root, "status", "--porcelain", "--untracked-files=no")
    return {
        "schema_version": 1,
        "kind": "wan2_2_ti2v_5b_unipc_50_step_cuda_coefficients",
        "contract": {
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
            "flow_shift": FLOW_SHIFT,
            "solver_order": SOLVER_ORDER,
            "predict_x0": True,
            "solver_type": "bh2",
            "lower_order_final": True,
            "prediction_type": "flow_prediction",
            "float_encoding": "IEEE-754 binary32, emitted as decimal and uint32 bits",
        },
        "provenance": {
            "official_source": str(source_root),
            "official_source_revision": _git(source_root, "rev-parse", "HEAD"),
            "official_source_tracked_dirty": bool(source_status),
            "official_scheduler_file": str(source_file),
            "official_scheduler_sha256": _sha256(source_file),
            "generator_file": str(Path(__file__).resolve()),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device),
            "cuda_device_index": torch.cuda.current_device()
            if device.index is None
            else device.index,
            "cuda_compute_capability": [properties.major, properties.minor],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "sigmas": [_float32(sigma) for sigma in sigmas.unbind()],
        "timesteps": [int(timestep.item()) for timestep in timesteps],
        "steps": steps,
    }


def _write_json_safely(output: Path, payload: dict[str, Any]) -> None:
    output = output.expanduser().absolute()
    if output.is_symlink():
        raise RuntimeError(f"Refusing to replace symlink output: {output}")
    if output.exists():
        if not output.is_file():
            raise RuntimeError(f"Output is not a regular file: {output}")
        if output.stat().st_size:
            raise RuntimeError(f"Refusing to overwrite nonempty output file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    source_root = args.official_source.expanduser().resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Exact order-two rho qualification requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.empty(1, dtype=torch.float32, device=device)

    scheduler_class, source_file = _load_scheduler(source_root)
    scheduler = scheduler_class(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=SOLVER_ORDER,
        prediction_type="flow_prediction",
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device, shift=FLOW_SHIFT)
    qualification_context = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.autocast_bf16 else nullcontext()
    )
    with qualification_context:
        payload = _make_payload(
            scheduler,
            source_root=source_root,
            source_file=source_file,
            device=device,
        )
    payload["contract"]["autocast_bf16"] = args.autocast_bf16
    torch.cuda.synchronize(device)
    output = args.output.expanduser().absolute()
    _write_json_safely(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "steps": len(payload["steps"]),
                "official_source_revision": payload["provenance"]["official_source_revision"],
                "torch_version": payload["provenance"]["torch_version"],
                "cuda_device": payload["provenance"]["cuda_device"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
