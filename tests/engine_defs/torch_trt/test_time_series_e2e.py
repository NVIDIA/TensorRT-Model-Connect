"""End-to-end Torch-TRT parity tests for numeric time-series models.

These tests exercise the real user flow:
  1. Build a local HF checkpoint into a `.trtfb` bundle via `trtmc build`
  2. Run C++ inference via `build/trtmc solve`
  3. Compare the output against the official Python reference implementation

They are intentionally serial and tiny so they can run in CI on a single GPU.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
import subprocess
import tempfile

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_trtmc_binary() -> Path | None:
    """Resolve the C++ trtmc binary, preferring an explicit override, then a
    local source build, then the trtmc installed on PATH.

    In CI there is no source-tree ``build/trtmc``; the wheel installs the native
    trtmc on PATH (``command -v trtmc``), so ``shutil.which`` finds it and the
    test runs against the packaged binary. Locally, ``build/trtmc`` (a fresh
    source build) takes precedence, and ``TRTMC_BINARY`` overrides everything.
    """
    for candidate in (
        os.environ.get("TRTMC_BINARY"),
        str(REPO_ROOT / "build" / "trtmc"),
        shutil.which("trtmc"),
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


TRTMC_BIN = _resolve_trtmc_binary()


def _require_trtmc_binary() -> Path:
    """Skip only when no trtmc binary can be found anywhere (a missing build
    artifact is an environment gap, gated like ``@requires_gpu``)."""
    if TRTMC_BIN is None:
        pytest.skip(
            "trtmc binary not found (set TRTMC_BINARY, build ./build/trtmc, "
            "or install the trtmc wheel so it is on PATH)"
        )
    return TRTMC_BIN


def _has_torchtrt() -> bool:
    try:
        import torch_tensorrt  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


requires_torchtrt = pytest.mark.skipif(
    not _has_torchtrt(), reason="torch_tensorrt not available"
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU not available"
)


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    result = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _parse_solve_stdout(stdout: str) -> torch.Tensor:
    for line in stdout.splitlines():
        if not line.startswith("Output ["):
            continue
        payload = line.split(":", 1)[1].strip()
        values = [float(x) for x in payload.split()] if payload else []
        return torch.tensor(values, dtype=torch.float32)
    raise AssertionError(f"Could not parse solve output:\n{stdout}")


def _build_bundle(model_dir: Path, bundle_path: Path, *, max_cache_length: int) -> None:
    _require_trtmc_binary()
    _run(
        [
            str(TRTMC_BIN),
            "build",
            str(model_dir),
            "-o",
            str(bundle_path),
            "--max-cache-length",
            str(max_cache_length),
            "--precision",
            "fp32",
        ]
    )
    assert bundle_path.is_file(), f"Expected bundle at {bundle_path}"


def _run_solve(bundle_path: Path, *, field_input: list[float] | None = None,
               branch_input: list[float] | None = None, trunk_input: list[float] | None = None) -> torch.Tensor:
    cmd = [str(TRTMC_BIN), "solve", str(bundle_path)]
    if field_input is not None:
        cmd.extend(["--field-input", ",".join(str(v) for v in field_input)])
    if branch_input is not None:
        cmd.extend(["--branch-input", ",".join(str(v) for v in branch_input)])
    if trunk_input is not None:
        cmd.extend(["--trunk-input", ",".join(str(v) for v in trunk_input)])
    result = _run(cmd)
    return _parse_solve_stdout(result.stdout)


@pytest.mark.e2e
@pytest.mark.trt
@pytest.mark.gpu
@pytest.mark.slow
@requires_torchtrt
@requires_gpu
def test_time_series_models_match_reference_end_to_end():
    _require_trtmc_binary()

    torch.manual_seed(0)

    artifacts_root = REPO_ROOT / "artifacts"
    artifacts_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trtmc_time_series_e2e_", dir=artifacts_root) as tmp:
        tmpdir = Path(tmp)

        # PatchTST classification: field_input -> logits
        patchtst_dir = tmpdir / "patchtst_model"
        patchtst_bundle = tmpdir / "patchtst.trtfb"
        patchtst_dir.mkdir()
        patchtst_model = transformers.PatchTSTForClassification(
            transformers.PatchTSTConfig(
                num_input_channels=2,
                num_targets=3,
                context_length=8,
                prediction_length=4,
                patch_length=2,
                stride=2,
                d_model=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                attention_dropout=0.0,
                ff_dropout=0.0,
                positional_dropout=0.0,
                path_dropout=0.0,
                head_dropout=0.0,
                use_cls_token=True,
            )
        ).eval()
        patchtst_model.save_pretrained(patchtst_dir)
        patchtst_values = torch.tensor(
            [
                [0.1, -0.2], [0.2, -0.1], [0.3, 0.0], [0.4, 0.1],
                [0.5, 0.2], [0.6, 0.3], [0.7, 0.4], [0.8, 0.5],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        patchtst_mask = torch.ones_like(patchtst_values, dtype=torch.bool)
        with torch.no_grad():
            patchtst_ref = patchtst_model(
                past_values=patchtst_values,
                past_observed_mask=patchtst_mask,
                return_dict=True,
            ).prediction_logits.reshape(-1).float()
        _build_bundle(patchtst_dir, patchtst_bundle, max_cache_length=8)
        patchtst_trt = _run_solve(
            patchtst_bundle,
            field_input=patchtst_values.reshape(-1).tolist(),
        )
        torch.testing.assert_close(patchtst_trt, patchtst_ref, rtol=0.0, atol=1e-6)

        # PatchTSMixer prediction: field_input -> forecast
        patchtsmixer_dir = tmpdir / "patchtsmixer_model"
        patchtsmixer_bundle = tmpdir / "patchtsmixer.trtfb"
        patchtsmixer_dir.mkdir()
        patchtsmixer_model = transformers.PatchTSMixerForPrediction(
            transformers.PatchTSMixerConfig(
                context_length=16,
                patch_length=8,
                patch_stride=8,
                prediction_length=4,
                num_input_channels=2,
                d_model=8,
                expansion_factor=2,
                num_layers=1,
                dropout=0.0,
                head_dropout=0.0,
            )
        ).eval()
        patchtsmixer_model.save_pretrained(patchtsmixer_dir)
        patchtsmixer_values = torch.tensor(
            [
                [0.1, -0.2], [0.2, -0.1], [0.3, 0.0], [0.4, 0.1],
                [0.5, 0.2], [0.6, 0.3], [0.7, 0.4], [0.8, 0.5],
                [0.9, 0.6], [1.0, 0.7], [1.1, 0.8], [1.2, 0.9],
                [1.3, 1.0], [1.4, 1.1], [1.5, 1.2], [1.6, 1.3],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        patchtsmixer_mask = torch.ones_like(patchtsmixer_values, dtype=torch.float32)
        with torch.no_grad():
            patchtsmixer_ref = patchtsmixer_model(
                past_values=patchtsmixer_values,
                observed_mask=patchtsmixer_mask,
                return_loss=False,
                return_dict=True,
            ).prediction_outputs.reshape(-1).float()
        _build_bundle(patchtsmixer_dir, patchtsmixer_bundle, max_cache_length=16)
        patchtsmixer_trt = _run_solve(
            patchtsmixer_bundle,
            field_input=patchtsmixer_values.reshape(-1).tolist(),
        )
        torch.testing.assert_close(patchtsmixer_trt, patchtsmixer_ref, rtol=0.0, atol=1e-6)

        # TimesFM forecasting: branch_input(series) + trunk_input(freq) -> mean forecast
        timesfm_dir = tmpdir / "timesfm_model"
        timesfm_bundle = tmpdir / "timesfm.trtfb"
        timesfm_dir.mkdir()
        timesfm_model = transformers.TimesFmModelForPrediction(
            transformers.TimesFmConfig(
                patch_length=2,
                context_length=8,
                horizon_length=4,
                freq_size=3,
                num_hidden_layers=1,
                hidden_size=16,
                intermediate_size=32,
                head_dim=8,
                num_attention_heads=2,
                quantiles=(0.1, 0.5, 0.9),
                attention_dropout=0.0,
            )
        ).eval()
        timesfm_model.save_pretrained(timesfm_dir)
        timesfm_series = torch.linspace(0.0, 1.0, steps=8, dtype=torch.float32)
        timesfm_freq = torch.tensor(2, dtype=torch.int64)
        with torch.no_grad():
            timesfm_ref = timesfm_model(
                past_values=[timesfm_series],
                freq=[timesfm_freq],
                return_dict=True,
            ).mean_predictions.reshape(-1).float()
        _build_bundle(timesfm_dir, timesfm_bundle, max_cache_length=8)
        timesfm_trt = _run_solve(
            timesfm_bundle,
            branch_input=timesfm_series.tolist(),
            trunk_input=[float(timesfm_freq.item())],
        )
        torch.testing.assert_close(timesfm_trt, timesfm_ref, rtol=0.0, atol=1e-6)

        if importlib.util.find_spec("chronos") is not None:
            import chronos

            chronos_dir = tmpdir / "chronos_bolt_model"
            chronos_bundle = tmpdir / "chronos_bolt.trtfb"
            chronos_dir.mkdir()
            chronos_config = transformers.T5Config(
                d_model=16,
                d_ff=32,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                dropout_rate=0.0,
                decoder_start_token_id=0,
                pad_token_id=0,
                eos_token_id=1,
            )
            chronos_config.architectures = ["ChronosBoltModelForForecasting"]
            chronos_config.chronos_config = {
                "context_length": 16,
                "prediction_length": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.1, 0.5, 0.9],
                "use_reg_token": True,
            }
            chronos_model = chronos.chronos_bolt.ChronosBoltModelForForecasting(
                chronos_config
            ).eval()
            chronos_model.save_pretrained(chronos_dir)
            chronos_series = torch.tensor(
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=torch.float32
            )
            with torch.no_grad():
                chronos_ref = chronos_model(
                    context=torch.cat(
                        [
                            torch.full((10,), float("nan"), dtype=torch.float32),
                            chronos_series,
                        ]
                    ).unsqueeze(0)
                ).quantile_preds.reshape(-1).float()
            _build_bundle(chronos_dir, chronos_bundle, max_cache_length=16)
            chronos_trt = _run_solve(
                chronos_bundle,
                branch_input=chronos_series.tolist(),
            )
            torch.testing.assert_close(chronos_trt, chronos_ref, rtol=0.0, atol=1e-6)
