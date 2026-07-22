# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for E2E manifest schema validation.

Trace: ARCH-E2E-001, UD-E2E-MANIFEST
Intent: Validate E2E manifest schema enforcement for required fields, type checks, and skip semantics
Preconditions: e2e_harness manifest_loader is importable
Postconditions: Invalid manifests raise appropriate ValueError/TypeError; valid manifests pass validation
"""

import json
import os
import pytest
import warnings
from pathlib import Path

# Try to import manifest_loader
try:
    from tests.e2e_harness.manifest_loader import (
        _validate_manifest,
        find_manifest_path,
        iter_manifest_paths,
        load_all_manifests,
        load_manifest,
        load_model_manifest,
    )
    from tests.e2e_harness.registry import (
        activate_model_plugins,
        get_comparator,
        get_reference,
        get_runner,
        reset as reset_e2e_registry,
    )
except ImportError:
    pytest.skip("e2e_harness not available", allow_module_level=True)


EXAMPLE_FAMILY = "example_family"
EXAMPLE_MODEL_ID = "example-org/example-model"
EXAMPLE_RUNTIME_STRATEGY = "llama_decoder_kv_cache"


class TestManifestValidation:
    """Test manifest schema validation."""

    def _write_manifest(self, tmp_path, data):
        path = os.path.join(str(tmp_path), "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def _write_unified_manifest(self, tmp_path, data):
        model_fields = {
            "hf_id",
            "model_id",
            "bundle",
            "family",
            "runtime_strategy",
            "task_strategy",
            "max_cache_length",
            "precision",
            "fp32_layers",
            "quantization",
            "fp8_scales",
            "trust_remote_code",
            "build_args",
            "build_env",
            "e2e_parallel_resource",
            "e2e_size",
            "distributed_runtime",
        }
        model = {"name": data.get("name", "")}
        model.update({key: value for key, value in data.items() if key in model_fields})
        testcase = {key: value for key, value in data.items() if key not in model_fields}
        model["testcases"] = [testcase]
        return self._write_manifest(tmp_path, model)

    def test_missing_name_raises(self, tmp_path):
        """Manifest without 'name' should raise ValueError."""
        path = self._write_manifest(
            tmp_path,
            {"hf_id": EXAMPLE_MODEL_ID, "family": EXAMPLE_FAMILY},
        )
        with pytest.raises(ValueError, match="name"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_hf_id_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'hf_id' (and no skip) should raise ValueError."""
        path = self._write_manifest(
            tmp_path,
            {"name": "test-model", "family": EXAMPLE_FAMILY},
        )
        with pytest.raises(ValueError, match="hf_id"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_family_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'family' (and no skip) should raise ValueError."""
        path = self._write_manifest(tmp_path, {"name": "test-model", "hf_id": "org/model"})
        with pytest.raises(ValueError, match="family"):
            _validate_manifest(json.load(open(path)), path)

    def test_skipped_manifest_allows_missing_hf_id(self, tmp_path):
        """Skipped manifests don't need hf_id or family."""
        data = {"name": "test-model", "skip": "not available"}
        _validate_manifest(data, "test.json")  # Should not raise

    def test_wrong_type_max_new_tokens_string(self, tmp_path):
        """max_new_tokens must be int, not string."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "max_new_tokens": "20",
        }
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_new_tokens_float(self, tmp_path):
        """max_new_tokens must be int, not float."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "max_new_tokens": 20.5,
        }
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_cache_length(self, tmp_path):
        """max_cache_length must be int, not float."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "max_cache_length": 256.5,
        }
        with pytest.raises(TypeError, match="max_cache_length"):
            _validate_manifest(data, "test.json")

    def test_unknown_runtime_strategy_warns(self, tmp_path):
        """Unknown runtime_strategy should emit a warning."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": "bogus_strategy",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            assert any("bogus_strategy" in str(warning.message) for warning in w)

    def test_known_runtime_strategy_no_warning(self, tmp_path):
        """Known runtime_strategy should not emit a warning."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            strategy_warnings = [x for x in w if "runtime_strategy" in str(x.message)]
            assert len(strategy_warnings) == 0

    def test_valid_manifest_passes(self, tmp_path):
        """A fully valid manifest should pass without errors."""
        data = {
            "name": "example-test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "bundle": "example-test.trtfb",
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "max_cache_length": 256,
            "max_new_tokens": 20,
            "prompt": "Hello",
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_multi_gpu_case_must_use_the_multi_device_tier(self):
        data = {
            "name": "tp4-test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "build_args": {"parallel": {"mode": "tensor_parallel", "tp_size": 4}},
            "distributed_runtime": {"enabled": True, "world_size": 4},
            "ci_tier": "nightly_only",
        }

        with pytest.raises(ValueError, match="4-GPU testcase.*multi_device"):
            _validate_manifest(data, "test.json")

        data["ci_tier"] = "multi_device"
        _validate_manifest(data, "test.json")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("build_args", [], "build_args must be an object"),
            (
                "build_args",
                {"parallel": []},
                "build_args.parallel must be an object",
            ),
            (
                "distributed_runtime",
                [],
                "distributed_runtime must be an object",
            ),
            (
                "preflight_requirements",
                [{"kind": "gpu_count_min", "args": []}],
                "preflight gpu_count_min args must be an object",
            ),
        ],
    )
    def test_device_settings_must_be_objects(self, field, value, message):
        data = {
            "name": "device-validation-test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            field: value,
        }

        with pytest.raises(TypeError, match=message):
            _validate_manifest(data, "test.json")

    def test_model_owned_layout_is_discovered(self, tmp_path):
        """Nested tests/e2e/models/<family>/manifests layout is supported."""
        models_dir = tmp_path / "models"
        manifest_dir = models_dir / EXAMPLE_FAMILY / "manifests"
        manifest_dir.mkdir(parents=True)
        manifest_path = manifest_dir / "example-test.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "example-test",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "testcases": [
                        {
                            "name": "example-test",
                            "prompt": "Hello",
                            "max_new_tokens": 4,
                        },
                        {
                            "name": "example-test-probe01",
                            "prompt": "Probe",
                            "max_new_tokens": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        assert iter_manifest_paths(models_dir) == [manifest_path]
        assert find_manifest_path("example-test", models_dir) == manifest_path
        assert find_manifest_path("example-test-probe01", models_dir) == manifest_path
        cases = load_all_manifests(models_dir)
        assert [case.name for case in cases] == [
            "example-test",
            "example-test-probe01",
        ]
        family_cases = load_all_manifests(models_dir / EXAMPLE_FAMILY)
        assert [case.name for case in family_cases] == [
            "example-test",
            "example-test-probe01",
        ]

    def test_model_owned_threshold_sidecar_is_loaded(self, tmp_path):
        """Model-local thresholds/<case>.json sidecars feed E2E thresholds."""
        family_dir = tmp_path / "models" / EXAMPLE_FAMILY
        manifest_dir = family_dir / "manifests"
        threshold_dir = family_dir / "thresholds"
        manifest_dir.mkdir(parents=True)
        threshold_dir.mkdir()
        manifest_path = manifest_dir / "example-test.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "example-test",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "testcases": [
                        {
                            "name": "example-test",
                            "prompt": "Hello",
                            "max_new_tokens": 4,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (threshold_dir / "example-test.json").write_text(
            json.dumps(
                {
                    "logit_atol": 10.0,
                    "threshold_overrides": {
                        "logit_cosine_p5": 0.0,
                        "token_agreement_rate": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        case = load_manifest(manifest_path)
        assert case.threshold_overrides == {
            "logit_atol": 10.0,
            "logit_cosine_p5": 0.0,
            "token_agreement_rate": 0.0,
        }

    def test_repo_model_indexes_cover_all_nested_manifests(self):
        """Every repo E2E manifest is listed from its family MODEL.toml."""
        models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
        nested_manifests = set(models_dir.glob("*/manifests/*.json"))
        assert nested_manifests

        family_dirs = {path.parent.parent for path in nested_manifests}
        missing_indexes = [
            path.relative_to(models_dir).as_posix()
            for path in sorted(family_dirs)
            if not (path / "MODEL.toml").is_file()
        ]
        assert not missing_indexes

        assert set(iter_manifest_paths(models_dir)) == nested_manifests

    def test_repo_model_dirs_own_e2e_runner(self):
        """Each model E2E folder owns its pytest runner entrypoint."""
        models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
        family_dirs = sorted(
            path
            for path in models_dir.iterdir()
            if path.is_dir() and (path / "MODEL.toml").is_file()
        )
        assert family_dirs

        missing_runner = []
        missing_test = []
        missing_plugins = []
        central_imports = []
        for family_dir in family_dirs:
            runner = family_dir / "runner.py"
            tests = sorted(family_dir.glob("test_*_e2e.py"))
            if not runner.is_file():
                missing_runner.append(family_dir.name)
            if len(tests) != 1:
                missing_test.append(family_dir.name)
            for plugin_name in ("runner.py", "reference.py", "comparator.py"):
                if not (family_dir / "e2e_plugins" / plugin_name).is_file():
                    missing_plugins.append(
                        f"{family_dir.relative_to(models_dir).as_posix()}/e2e_plugins/{plugin_name}"
                    )
            for plugin_subdir in ("runners", "references", "comparators"):
                if not (family_dir / "e2e_plugins" / plugin_subdir).is_dir():
                    missing_plugins.append(
                        f"{family_dir.relative_to(models_dir).as_posix()}/"
                        f"e2e_plugins/{plugin_subdir}"
                    )

            local_files = [path for path in [runner, *tests] if path.is_file()]
            local_files.extend(sorted((family_dir / "e2e_plugins").rglob("*.py")))
            for path in local_files:
                text = path.read_text(encoding="utf-8")
                if "tests.test_e2e" in text:
                    central_imports.append(path.relative_to(models_dir).as_posix())
                if any(
                    forbidden in text
                    for forbidden in (
                        "tests.e2e_harness.runners",
                        "tests.e2e_harness.references",
                        "tests.e2e_harness.comparators",
                    )
                ):
                    central_imports.append(path.relative_to(models_dir).as_posix())

        assert not missing_runner
        assert not missing_test
        assert not missing_plugins
        assert not central_imports

    def test_repo_model_thresholds_are_sidecars(self):
        """Per-model threshold overrides live under model-owned thresholds/."""
        models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
        inline_fields = {
            "threshold_overrides",
            "logit_atol",
            "layer_atol",
            "min_pixel_agreement",
            "min_pixel_mean",
            "max_pixel_mean",
            "min_pixel_std",
            "reference_min_pixel_std_for_ratio",
            "min_reference_std_ratio",
            "speech_min_token_match",
            "speech_min_frame_exact",
            "speech_min_rms",
        }

        inline_thresholds = []
        for manifest_path in sorted(models_dir.glob("*/manifests/*.json")):
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = sorted(inline_fields & raw.keys())
            if keys:
                inline_thresholds.append((manifest_path.relative_to(models_dir).as_posix(), keys))

        sidecars = sorted(models_dir.glob("*/thresholds/*.json"))
        testcase_paths = {}
        for manifest_path in sorted(models_dir.glob("*/manifests/*.json")):
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            for testcase in raw["testcases"]:
                testcase_paths[(manifest_path.parent.parent, testcase["name"])] = manifest_path
        missing_sidecars = [
            f"{manifest.relative_to(models_dir).as_posix()}: {name}"
            for (family_dir, name), manifest in testcase_paths.items()
            if not (family_dir / "thresholds" / f"{name}.json").is_file()
        ]
        missing_manifests = [
            path.relative_to(models_dir).as_posix()
            for path in sidecars
            if (path.parent.parent, path.stem) not in testcase_paths
        ]

        assert sidecars
        assert not inline_thresholds
        assert not missing_sidecars
        assert not missing_manifests

    def test_repo_model_assets_are_local(self):
        """Model E2E manifests resolve data assets from their own folders."""
        models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
        asset_fields = {
            "test_image",
            "test_input_audio",
            "speech_reference_tokens",
            "golden_snapshot_path",
            "edit_condition_image",
            "fp8_scales",
        }
        global_refs = []
        missing_assets = []

        def iter_asset_values(value, key=""):
            if isinstance(value, dict):
                if "relative_to" in value and isinstance(value.get("path"), str):
                    yield value["path"]
                for item_key, item_value in value.items():
                    yield from iter_asset_values(item_value, item_key)
            elif isinstance(value, list):
                for item in value:
                    yield from iter_asset_values(item, key)
            elif isinstance(value, str) and key in asset_fields:
                yield value

        for manifest_path in sorted(models_dir.glob("*/manifests/*.json")):
            text = manifest_path.read_text(encoding="utf-8")
            if "tests/e2e/data" in text:
                global_refs.append(manifest_path.relative_to(models_dir).as_posix())

            raw = json.loads(text)
            family_dir = manifest_path.parent.parent
            for asset in iter_asset_values(raw):
                if asset.startswith("tests/e2e/data"):
                    global_refs.append(manifest_path.relative_to(models_dir).as_posix())
                    continue
                if asset.startswith("data/") and not (family_dir / asset).is_file():
                    missing_assets.append(
                        f"{manifest_path.relative_to(models_dir).as_posix()}: {asset}"
                    )

        assert not global_refs
        assert not missing_assets

    def test_repo_models_use_model_local_e2e_plugins(self):
        """Every manifest resolves runner/reference/comparator from its folder."""
        models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
        failures = []
        for family_dir in sorted(
            path
            for path in models_dir.iterdir()
            if path.is_dir() and (path / "MODEL.toml").is_file()
        ):
            family_prefix = f"tests.e2e.models.{family_dir.name}.e2e_plugins."
            try:
                activate_model_plugins(family_dir)
                for manifest_path in sorted((family_dir / "manifests").glob("*.json")):
                    model = load_model_manifest(manifest_path)
                    for case in model.testcases:
                        resolved = {
                            "runner": get_runner(case.task_strategy),
                            "reference": get_reference(case.reference_backend),
                            "comparator": get_comparator(case.task_strategy),
                        }
                        for kind, plugin in resolved.items():
                            module = type(plugin).__module__ if plugin is not None else ""
                            if not module.startswith(family_prefix):
                                failures.append(
                                    (
                                        manifest_path.relative_to(models_dir).as_posix(),
                                        case.name,
                                        kind,
                                        module,
                                    )
                                )
            finally:
                reset_e2e_registry()

        assert not failures

    def test_repo_runtime_models_use_model_local_helpers(self):
        """Runtime plugin helpers are model-owned when production code uses them."""
        repo_root = Path(__file__).resolve().parents[2]
        runtime_models_dir = repo_root / "src" / "runtime" / "models"
        shared_helpers_dir = repo_root / "src" / "runtime" / "plugins" / "shared"

        obsolete_shared_helpers = sorted(shared_helpers_dir.glob("*_helpers.*"))
        obsolete_shared_helpers.extend(sorted(shared_helpers_dir.glob("plugin_helpers.*")))
        assert not obsolete_shared_helpers

        cmake_text = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8")
        assert "src/runtime/plugins/shared" not in cmake_text

        missing_helpers = []
        shared_includes = []
        cross_model_includes = []
        for model_dir in sorted(path for path in runtime_models_dir.iterdir() if path.is_dir()):
            if not (model_dir / "MODEL.toml").is_file():
                continue
            sources = sorted(model_dir.glob("*.[ch]pp")) + sorted(model_dir.glob("*.h"))
            uses_plugin_helpers = any(
                path.name not in {"plugin_helpers.h", "plugin_helpers.cpp"}
                and "plugin_helpers.h" in path.read_text(encoding="utf-8")
                for path in sources
            )
            if uses_plugin_helpers:
                for helper in ("plugin_helpers.h", "plugin_helpers.cpp"):
                    if not (model_dir / helper).is_file():
                        missing_helpers.append(f"{model_dir.name}/{helper}")

            for path in sources:
                text = path.read_text(encoding="utf-8")
                if "runtime/plugins/shared" in text:
                    shared_includes.append(path.relative_to(repo_root).as_posix())
                for include_line in [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip().startswith('#include "runtime/models/')
                ]:
                    prefix = f'#include "runtime/models/{model_dir.name}/'
                    if not include_line.startswith(prefix):
                        cross_model_includes.append(
                            f"{path.relative_to(repo_root).as_posix()}: {include_line}"
                        )
                if '#include "audio_helpers.h"' in text:
                    for helper in ("audio_helpers.h", "audio_helpers.cpp"):
                        if not (model_dir / helper).is_file():
                            missing_helpers.append(f"{model_dir.name}/{helper}")
                if '#include "diffusion_helpers.h"' in text:
                    for helper in ("diffusion_helpers.h", "diffusion_helpers.cpp"):
                        if not (model_dir / helper).is_file():
                            missing_helpers.append(f"{model_dir.name}/{helper}")

        assert not missing_helpers
        assert not shared_includes
        assert not cross_model_includes

    def test_model_id_accepted_as_hf_id_alternative(self, tmp_path):
        """model_id is accepted as an alternative to hf_id."""
        data = {
            "name": "test-model",
            "model_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_load_manifest_calls_validation(self, tmp_path):
        """load_manifest should call _validate_manifest and raise on bad input."""
        path = self._write_manifest(
            tmp_path,
            {"hf_id": EXAMPLE_MODEL_ID, "family": EXAMPLE_FAMILY},
        )
        with pytest.raises(ValueError, match="name"):
            load_manifest(path)

    def test_gated_manifest_requires_auth_preflight(self, tmp_path):
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "gated-test",
                "hf_id": "org/gated-model",
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                "gated": True,
            },
        )
        case = load_manifest(path)
        matches = [req for req in case.preflight if req.kind == "hf_auth_token_present"]
        assert len(matches) == 1
        assert matches[0].gating is True
        assert matches[0].args == {"hf_id": "org/gated-model"}

    def test_remote_code_manifest_auth_preflight_is_diagnostic(self, tmp_path):
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "remote-code-test",
                "hf_id": "org/remote-code-model",
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": "embedding",
                "trust_remote_code": True,
            },
        )
        case = load_manifest(path)
        matches = [req for req in case.preflight if req.kind == "hf_auth_token_present"]
        assert len(matches) == 1
        assert matches[0].gating is False
        assert matches[0].args == {"hf_id": "org/remote-code-model"}

    def test_explicit_asset_preflight_path_is_model_local(self, tmp_path):
        model_dir = tmp_path / "image_family"
        manifests_dir = model_dir / "manifests"
        manifests_dir.mkdir(parents=True)
        asset = model_dir / "data" / "test_img.jpeg"
        asset.parent.mkdir()
        asset.write_bytes(b"image")
        manifest_path = manifests_dir / "image-model.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "image-model",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "testcases": [
                        {
                            "name": "image-model",
                            "test_image": "data/test_img.jpeg",
                            "preflight_requirements": [
                                {
                                    "kind": "asset_exists",
                                    "args": {"path": "data/test_img.jpeg"},
                                    "gating": True,
                                },
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        case = load_manifest(manifest_path)

        assert case.preflight[0].args["path"] == str(asset)

    def test_bool_not_accepted_as_int(self, tmp_path):
        """Boolean values should not pass the int type check."""
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "max_new_tokens": True,
        }
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_execution_profiles_must_be_object(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "execution_profiles": "example_profile",
        }
        with pytest.raises(TypeError, match="execution_profiles"):
            _validate_manifest(data, "test.json")

    def test_reference_precision_is_validated_and_propagated(self, tmp_path):
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "fp16-reference-test",
                "hf_id": EXAMPLE_MODEL_ID,
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                "precision": "fp16",
                "reference_precision": "fp32",
            },
        )

        case = load_manifest(path)
        assert case.metadata["precision"] == "fp16"
        assert case.metadata["reference_precision"] == "fp32"

        with pytest.raises(ValueError, match="reference_precision"):
            _validate_manifest(
                {
                    "name": "bad-reference-precision",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "reference_precision": "tf32",
                },
                "test.json",
            )

    def test_fp32_layers_are_validated_and_propagated(self, tmp_path):
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "mixed-precision-test",
                "hf_id": EXAMPLE_MODEL_ID,
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                "precision": "fp16",
                "reference_precision": "fp32",
                "fp32_layers": [2],
            },
        )

        case = load_manifest(path)
        assert case.metadata["fp32_layers"] == [2]

        with pytest.raises(TypeError, match="fp32_layers"):
            _validate_manifest(
                {
                    "name": "bad-fp32-layer",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "fp32_layers": [-1],
                },
                "test.json",
            )

        with pytest.raises(ValueError, match="duplicates"):
            _validate_manifest(
                {
                    "name": "duplicate-fp32-layer",
                    "hf_id": EXAMPLE_MODEL_ID,
                    "family": EXAMPLE_FAMILY,
                    "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                    "fp32_layers": [2, 2],
                },
                "test.json",
            )

    def test_execution_profiles_reject_unknown_phase(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": EXAMPLE_MODEL_ID,
            "family": EXAMPLE_FAMILY,
            "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
            "execution_profiles": {
                "build": "example_profile",
                "verify": "example_profile",
            },
        }
        with pytest.raises(ValueError, match="unsupported phase"):
            _validate_manifest(data, "test.json")

    def test_quantization_block_propagates_to_metadata(self, tmp_path):
        """Quantization manifests should preserve the generic quant block."""
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "example-test-fp8",
                "hf_id": EXAMPLE_MODEL_ID,
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": EXAMPLE_RUNTIME_STRATEGY,
                "precision": "bf16",
                "quantization": {
                    "format": "fp8",
                    "scale_source": "precomputed",
                    "scale_artifact": "scales/example-fp8.json",
                    "calibration_samples": 16,
                },
            },
        )
        case = load_manifest(path)
        assert case.metadata["precision"] == "bf16"
        assert case.metadata["quantization"]["format"] == "fp8"
        assert case.metadata["quantization"]["scale_artifact"] == "scales/example-fp8.json"

    def test_skip_comparison_populates_metadata(self, tmp_path):
        """skip_comparison should set skip_comparison_reason without setting skip_reason."""
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "rerank-test",
                "hf_id": "org/rerank",
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": "reranking",
                "skip_comparison": "reference shape mismatch",
            },
        )
        case = load_manifest(path)
        assert case.metadata["skip_comparison_reason"] == "reference shape mismatch"
        # Partial skip must NOT set skip_reason (that would trigger full pytest.skip)
        assert "skip_reason" not in case.metadata

    def test_skip_comparison_does_not_exempt_required_fields(self, tmp_path):
        """skip_comparison still requires hf_id + family (unlike skip)."""
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "rerank-test",
                "skip_comparison": "reference shape mismatch",
            },
        )
        with pytest.raises(ValueError, match="hf_id"):
            _validate_manifest(json.load(open(path)), path)

    def test_skip_comparison_bool_defaults_reason(self, tmp_path):
        """skip_comparison: true should produce a default reason string."""
        path = self._write_unified_manifest(
            tmp_path,
            {
                "name": "rerank-test",
                "hf_id": "org/rerank",
                "family": EXAMPLE_FAMILY,
                "runtime_strategy": "reranking",
                "skip_comparison": True,
            },
        )
        case = load_manifest(path)
        assert case.metadata["skip_comparison_reason"]

    def test_skip_and_skip_comparison_are_independent(self, tmp_path):
        """`skip` still takes precedence (full skip); `skip_comparison` alone is partial."""
        path_full = self._write_unified_manifest(
            tmp_path,
            {
                "name": "full-skip",
                "skip": "broken",
            },
        )
        case_full = load_manifest(path_full)
        assert case_full.metadata["skip_reason"] == "broken"
        assert "skip_comparison_reason" not in case_full.metadata
