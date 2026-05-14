"""Unit tests for HfDiffusersReference Qwen-Image dispatch.

Trace: ARCH-E2E-001, UD-FAM-QWEN-IMAGE-01, UT-QWEN-IMAGE-HF-REF-001
Intent: Validate that ``HfDiffusersReference._run_full_pipeline`` constructs
    a Python subprocess script that loads ``diffusers.QwenImagePipeline``
    and calls it with the manifest's prompt, negative_prompt, true_cfg_scale,
    num_inference_steps, height, width, and seed. Mocks the subprocess so
    no GPU/HF download is required.
Preconditions: The Qwen-Image manifest carries ``family == "qwen_image"`` and
    publishes ``cfg_scale`` (which maps to ``true_cfg_scale``), ``height``,
    ``width``, ``num_inference_steps``, ``negative_prompt``, and ``seed``.
Postconditions: The generated subprocess script contains the
    ``QwenImagePipeline.from_pretrained`` dispatch with bf16 dtype, the
    ``true_cfg_scale=<value>`` kwarg (NOT ``guidance_scale``), and the
    expected ``height`` / ``width`` / ``num_inference_steps`` values.
    The FLUX/Wan branches remain unaffected.
"""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.references import hf_diffusers


def _make_qwen_image_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="qwen-image-case",
        hf_id="Qwen/Qwen-Image-2512",
        family="qwen_image",
        runtime_strategy="diffusion_qwen_image",
        bundle="qwen-image-case.trtfb",
        inputs=inputs or {},
    )


def _make_flux_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="flux-case",
        hf_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        runtime_strategy="diffusion_flux",
        bundle="flux-case.trtfb",
        inputs=inputs or {},
    )


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def _capture_subprocess(monkeypatch):
    """Patch subprocess.run inside the hf_diffusers reference to capture argv."""
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Generated 1 frames\n", stderr="")

    monkeypatch.setattr(hf_diffusers.subprocess, "run", _fake_run)
    return captured


def _extract_script(cmd: list[str]) -> str:
    """Pull the inline Python script string from a ``python -c <script>`` cmd."""
    assert "-c" in cmd, f"expected python -c invocation, got {cmd!r}"
    idx = cmd.index("-c")
    return cmd[idx + 1]


def test_qwen_image_reference_uses_qwen_image_pipeline(monkeypatch, tmp_path):
    """Qwen-Image manifest must drive QwenImagePipeline with true_cfg_scale."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "A red apple on a wooden table",
            "negative_prompt": " ",
            "num_inference_steps": 20,
            "cfg_scale": 4.0,
            "height": 1024,
            "width": 1024,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])

    # Qwen-Image branch is selected.
    assert "QwenImagePipeline" in script
    assert 'family in ("qwen_image",)' in script

    # Uses true_cfg_scale, NOT guidance_scale (Qwen-Image-specific kwarg).
    assert "true_cfg_scale=qi_cfg_scale" in script
    # The numeric cfg value is bound into the script as the qi_cfg_scale literal.
    assert "qi_cfg_scale = 4.0" in script

    # bf16 dtype matches the PSNR-validated Python E2E gate.
    assert "torch.bfloat16" in script

    # Dimensions / steps / seed / negative prompt are all threaded through.
    assert "qi_height = 1024" in script
    assert "qi_width = 1024" in script
    assert "num_steps = 20" in script
    assert "seed = 42" in script
    assert "qi_negative_prompt = ' '" in script

    # Prompt is interpolated as a Python literal.
    assert "prompt = 'A red apple on a wooden table'" in script


def test_qwen_image_reference_falls_back_to_guidance_scale(monkeypatch, tmp_path):
    """If ``cfg_scale`` is absent, ``guidance_scale`` maps to true_cfg_scale."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "guidance_scale": 3.5,
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "true_cfg_scale=qi_cfg_scale" in script
    assert "qi_cfg_scale = 3.5" in script


def test_qwen_image_reference_image_height_width_alias(monkeypatch, tmp_path):
    """``image_height`` / ``image_width`` aliases should map to qi_height/qi_width."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "image_height": 768,
            "image_width": 512,
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    assert "qi_height = 768" in script
    assert "qi_width = 512" in script


def test_qwen_image_reference_writes_frames_dir(monkeypatch, tmp_path):
    """Reference must write to ``hf_frames/`` (matches comparator frame glob)."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "num_inference_steps": 4,
            "seed": 7,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    # The script saves frames as frame_NNNN.png so the comparator's
    # ``glob("frame_*.png")`` picks them up on the reference side too.
    assert 'frame_{i:04d}.png' in script or 'frame_{{i:04d}}.png' in script
    # frames_dir is placed under the per-case artifact dir as hf_frames/.
    assert "hf_frames" in script


def test_qwen_image_reference_forward_compat_edit_variants(monkeypatch, tmp_path):
    """The dispatch should try Edit variants first when ``image`` input is set."""
    # No image input -> should NOT prefer Edit variants.
    case_no_image = _make_qwen_image_case(
        inputs={"prompt": "scene", "num_inference_steps": 4},
    )
    ctx = _make_ctx(case_no_image, tmp_path)
    captured = _capture_subprocess(monkeypatch)
    hf_diffusers.HfDiffusersReference().run_stage(
        case_no_image, StageSpec(name="end_to_end"), ctx)
    script = _extract_script(captured["cmd"])
    # Script enumerates Edit + non-Edit classes but skips Edit when no image.
    assert "QwenImageEditPlusPipeline" in script
    assert "QwenImageEditPipeline" in script
    assert "QwenImagePipeline" in script
    # The runtime guard against picking Edit when no image is present.
    assert 'if "Edit" in cls_name and not False' in script

    # With image input -> Edit variants are eligible.
    case_with_image = _make_qwen_image_case(
        inputs={"prompt": "scene", "num_inference_steps": 4,
                "image": "/tmp/x.png"},
    )
    ctx2 = _make_ctx(case_with_image, tmp_path)
    captured2 = _capture_subprocess(monkeypatch)
    hf_diffusers.HfDiffusersReference().run_stage(
        case_with_image, StageSpec(name="end_to_end"), ctx2)
    script2 = _extract_script(captured2["cmd"])
    assert 'if "Edit" in cls_name and not True' in script2


def test_flux_branch_unaffected(monkeypatch, tmp_path):
    """The Qwen-Image branch must not regress the existing FLUX dispatch."""
    case = _make_flux_case(
        inputs={
            "prompt": "A cat in a meadow",
            "num_inference_steps": 4,
            "image_height": 1024,
            "image_width": 1024,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = _extract_script(captured["cmd"])
    # FLUX still selects FluxPipeline, not QwenImagePipeline.
    assert "FluxPipeline" in script
    assert 'family in ("flux",)' in script
    # The family literal bound into the script must be "flux" (not
    # "qwen_image"); the dispatcher then enters the FLUX branch at runtime.
    assert "family = 'flux'" in script
    # The Qwen-Image branch is structurally present in the script template
    # but is gated by ``family in ("qwen_image",)`` — it cannot fire when
    # the runtime family is FLUX.
