"""Unit tests for owned scheduler modules.

These tests target deterministic math and branch behavior only.

Trace: ARCH-PIP-DIFF-001, UD-DIFF-SCHED
Intent: Validate flow-match Euler scheduler timestep construction and shift behavior
Preconditions: Scheduler is instantiated with known train timesteps and shift parameters
Postconditions: Timestep arrays are decreasing float32 with correct schedule shape and sigma termination
"""

from __future__ import annotations

import runpy
import sys
import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


# Ensure imports resolve to this workspace's Python package and owner-local tools.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / "python"
for import_root in (_REPO_ROOT, _PKG_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DIFFUSION_SCHEDULER_OWNERS = ("flux", "pixart", "wan_t2v", "z_image")


@pytest.fixture(params=DIFFUSION_SCHEDULER_OWNERS)
def scheduler_modules(request: pytest.FixtureRequest):
    """Return the scheduler module pair owned by one diffusion family."""
    family = str(request.param)
    tools_package = _REPO_ROOT / "tools/families" / family / "schedulers"
    module_root = (
        f"tools.families.{family}.schedulers"
        if tools_package.is_dir()
        else f"tensorrt_model_connect.families.{family}.schedulers"
    )
    package = import_module(module_root)
    flow = import_module(f"{module_root}.flow_match_euler")
    base = import_module(f"{module_root}.base")
    return family, package, flow, base


@pytest.mark.unit
def test_flow_match_set_timesteps_builds_expected_schedule(scheduler_modules) -> None:
    """Intent: verify deterministic timestep construction for shift=1.

    Preconditions: Scheduler is created with default train timesteps and shift.
    Postconditions: Timesteps are decreasing float32 values and sigma appends 0.
    """
    _family, _package, flow, _base = scheduler_modules
    FlowMatchEulerScheduler = flow.FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(num_inference_steps=4)

    np.testing.assert_allclose(
        scheduler.timesteps,
        np.array([1000.0, 667.0, 334.0, 1.0], dtype=np.float32),
    )
    assert scheduler.timesteps.dtype == np.float32
    np.testing.assert_allclose(scheduler._sigmas[-1], 0.0)
    assert np.all(np.diff(scheduler.timesteps) < 0.0)


@pytest.mark.unit
def test_flow_match_set_timesteps_with_shift_changes_tail(scheduler_modules) -> None:
    """Intent: validate shifted schedule branch is used when shift != 1.

    Preconditions: Scheduler is created with non-default shift value.
    Postconditions: Final timestep is larger than unshifted schedule tail.
    """
    _family, _package, flow, _base = scheduler_modules
    FlowMatchEulerScheduler = flow.FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=2.0)
    scheduler.set_timesteps(num_inference_steps=4)

    assert scheduler.timesteps[0] == pytest.approx(1000.0, abs=1e-6)
    assert scheduler.timesteps[-1] > 1.0
    assert np.all(np.diff(scheduler.timesteps) < 0.0)


@pytest.mark.unit
def test_flow_match_step_uses_sigma_delta_and_returns_float32(scheduler_modules) -> None:
    """Intent: verify Euler update uses sigma_next - sigma.

    Preconditions: Internal sigma schedule is initialized with two values.
    Postconditions: Step output matches expected update and is float32.
    """
    _family, _package, flow, _base = scheduler_modules
    FlowMatchEulerScheduler = flow.FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler()
    scheduler._sigmas = np.array([1.0, 0.5], dtype=np.float64)

    sample = np.array([1.0, 1.0], dtype=np.float32)
    model_output = np.array([2.0, -2.0], dtype=np.float32)

    updated = scheduler.step(
        model_output=model_output,
        timestep=999.0,
        sample=sample,
        step_index=0,
    )

    np.testing.assert_allclose(updated, np.array([0.0, 2.0], dtype=np.float32))
    assert updated.dtype == np.float32


@pytest.mark.unit
def test_flow_match_add_noise_is_linear_interpolation(scheduler_modules) -> None:
    """Intent: verify add_noise follows z_t = (1-sigma)x + sigma*noise.

    Preconditions: Original sample, noise sample, and timestep are provided.
    Postconditions: Output equals the expected convex combination.
    """
    _family, _package, flow, _base = scheduler_modules
    FlowMatchEulerScheduler = flow.FlowMatchEulerScheduler

    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
    original = np.array([4.0, -4.0], dtype=np.float32)
    noise = np.array([0.0, 8.0], dtype=np.float32)

    mixed = scheduler.add_noise(original=original, noise=noise, timestep=250.0)

    np.testing.assert_allclose(mixed, np.array([3.0, -1.0], dtype=np.float32))
    assert mixed.dtype == np.float32


@pytest.mark.unit
def test_get_scheduler_factory_and_unknown_error(scheduler_modules) -> None:
    """Intent: validate scheduler factory dispatch and error branch.

    Preconditions: Scheduler name is valid once and invalid once.
    Postconditions: Valid name returns instance; invalid name raises ValueError.
    """
    _family, package, _flow, _base = scheduler_modules
    FlowMatchEulerScheduler = package.FlowMatchEulerScheduler
    get_scheduler = package.get_scheduler

    scheduler = get_scheduler("flow_match_euler", shift=1.5)
    assert isinstance(scheduler, FlowMatchEulerScheduler)
    assert scheduler.shift == pytest.approx(1.5)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        get_scheduler("does_not_exist")


@pytest.mark.unit
def test_scheduler_protocol_declares_required_methods(scheduler_modules) -> None:
    """Intent: smoke-test protocol surface for scheduler interface.

    Preconditions: Scheduler protocol is importable.
    Postconditions: Protocol exposes all required API method names.
    """
    _family, _package, _flow, base = scheduler_modules
    Scheduler = base.Scheduler

    assert hasattr(Scheduler, "timesteps")
    assert hasattr(Scheduler, "set_timesteps")
    assert hasattr(Scheduler, "step")
    assert hasattr(Scheduler, "add_noise")


@pytest.mark.unit
def test_package_main_module_invokes_build_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate `python -m tensorrt_model_connect` delegates to build_cli.main.

    Preconditions: A fake `tensorrt_model_connect.build_cli` module with callable `main` is injected.
    Postconditions: Importing/executing `tensorrt_model_connect.__main__` calls fake `main` once.
    """
    calls: list[str] = []

    fake_cli = types.ModuleType("tensorrt_model_connect.build_cli")

    def _fake_main() -> None:
        calls.append("called")

    fake_cli.main = _fake_main  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.build_cli", fake_cli)
    sys.modules.pop("tensorrt_model_connect.__main__", None)

    runpy.run_module("tensorrt_model_connect.__main__", run_name="__main__")

    assert calls == ["called"]


@pytest.mark.unit
def test_package_init_exports_expected_symbols() -> None:
    """Intent: verify top-level package re-exports expected public API.

    Preconditions: Local `tensorrt_model_connect` package is importable.
    Postconditions: Public symbols from `tensorrt_model_connect.__init__` exist and are usable.
    """
    import importlib

    pkg = importlib.import_module("tensorrt_model_connect")

    assert pkg.__version__ == "0.1.0"
    assert callable(pkg.build)
    assert callable(pkg.build_bundle)
    assert callable(pkg.write_bundle)
    assert hasattr(pkg, "ModelConfig")
    assert hasattr(pkg, "Pipeline")
