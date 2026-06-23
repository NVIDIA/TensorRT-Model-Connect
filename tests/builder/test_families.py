"""Unit tests for all family plugins — match() logic and basic attributes.

Pure Python, no TRT/GPU needed. Verifies every plugin's match() returns
True for its model_types and False for others, and checks special attributes
like runtime_strategy and embed_input.

Trace: ARCH-FAM-001, UD-FAM-MATCH
Intent: Validate family plugin match() dispatch and attribute correctness for all registered plugins
Preconditions: All family plugins are discoverable via the auto-discovery registry
Postconditions: Each plugin matches its declared model_types and rejects foreign types
"""

from __future__ import annotations

import ast
import io
import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip(
    "tensorrt",
    reason="family plugin registry tests import TensorRT-backed plugin modules",
)

from tensorrt_model_connect.families import find_plugin, _ALL_PLUGINS
from tests.e2e_harness.manifest_loader import iter_manifest_paths


def test_load_plugin_by_id_does_not_scan_family_metadata(monkeypatch, tmp_path):
    import tensorrt_model_connect.families as families

    class DummyPlugin:
        name = "example_family"

    def fail_metadata_scan():
        raise AssertionError("metadata scan should not run for direct plugin id")

    family_dir = tmp_path / "families" / "example_family"
    family_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text(
        'id = "example_family"\nplugin = "example_family"\n',
        encoding="utf-8",
    )

    families._PLUGIN_CACHE.clear()
    monkeypatch.setattr(
        families,
        "__file__",
        str(tmp_path / "families" / "__init__.py"),
    )
    monkeypatch.setattr(families, "_load_family_metadata", fail_metadata_scan)
    monkeypatch.setattr(
        families,
        "_load_plugin_from_module",
        lambda module_name: DummyPlugin() if module_name == "example_family" else None,
    )

    plugin = families.load_plugin_by_id("example_family")

    assert plugin is not None
    assert plugin.name == "example_family"


def test_candidate_module_names_uses_cached_metadata_index(monkeypatch):
    import tensorrt_model_connect.families as families

    metadata = [
        families._FamilyMetadata(
            id="alpha",
            import_module="alpha",
            aliases=frozenset({"alpha"}),
            compact_aliases=frozenset({"alpha"}),
            prefixes=frozenset({"alpha"}),
            compact_prefixes=frozenset({"alpha"}),
            capabilities=frozenset(),
            architecture_patterns=frozenset(),
            diffusion_pipeline_classes=frozenset(),
            nemo_target_patterns=frozenset(),
            nemo_model_type="",
        ),
        families._FamilyMetadata(
            id="alpha_vl",
            import_module="alpha_vl",
            aliases=frozenset({"alpha_vl"}),
            compact_aliases=frozenset({"alphavl"}),
            prefixes=frozenset({"alpha_vl"}),
            compact_prefixes=frozenset({"alphavl"}),
            capabilities=frozenset(),
            architecture_patterns=frozenset(),
            diffusion_pipeline_classes=frozenset(),
            nemo_target_patterns=frozenset(),
            nemo_model_type="",
        ),
    ]
    monkeypatch.setattr(families, "_METADATA_CACHE", metadata)
    monkeypatch.setattr(families, "_METADATA_INDEX_CACHE", None)

    assert families._candidate_module_names("alpha3") == ["alpha"]
    assert families._candidate_module_names("alpha_vl") == ["alpha_vl", "alpha"]

    def fail_metadata_scan():
        raise AssertionError("candidate lookup should use the metadata index cache")

    monkeypatch.setattr(families, "_load_family_metadata", fail_metadata_scan)

    assert families._candidate_module_names("alpha3") == ["alpha"]
    assert families._candidate_module_names("alpha_vl") == ["alpha_vl", "alpha"]


def test_family_metadata_owns_builder_capabilities_and_nemo_resolution():
    import tensorrt_model_connect.families as families

    for meta in families._load_family_metadata():
        alias = next(iter(sorted(meta.aliases)), None)
        if alias is None:
            continue
        for capability in meta.capabilities:
            assert families.family_has_capability(alias, capability)
        for pattern in meta.nemo_target_patterns:
            assert families.resolve_nemo_model_type({
                "target": f"example.{pattern}.Model",
            }) == meta.nemo_model_type

    assert families.resolve_nemo_model_type({
        "model_type": "custom_nemo_model",
    }) == "custom_nemo_model"


def _write_nemo_config(path: Path, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    info = tarfile.TarInfo("model_config.yaml")
    info.size = len(data)
    with tarfile.open(path, "w") as tar:
        tar.addfile(info, io.BytesIO(data))


def test_nemo_archive_resolution_uses_family_owned_adapters(tmp_path, monkeypatch):
    import tensorrt_model_connect.families as families

    metadata = [
        families._FamilyMetadata(
            id="example_nemo",
            import_module="example_nemo",
            aliases=frozenset({"example_nemo"}),
            compact_aliases=frozenset({"examplenemo"}),
            prefixes=frozenset({"example_nemo"}),
            compact_prefixes=frozenset({"examplenemo"}),
            capabilities=frozenset(),
            architecture_patterns=frozenset(),
            diffusion_pipeline_classes=frozenset(),
            nemo_target_patterns=frozenset(),
            nemo_model_type="",
            nemo_archive_adapter="adapter.py|resolve",
        ),
    ]

    def fake_adapter(path: Path) -> Path:
        resolved_dir = tmp_path / "resolved"
        resolved_dir.mkdir()
        (resolved_dir / "config.json").write_text(json.dumps({
            "model_type": "example_nemo",
            "_nemo_archive_path": str(path),
        }))
        return resolved_dir

    monkeypatch.setattr(families, "_load_family_metadata", lambda: metadata)
    monkeypatch.setattr(
        families,
        "_load_metadata_callable_from_file",
        lambda _meta, _spec: fake_adapter,
    )

    nemo_path = tmp_path / "example.nemo"
    _write_nemo_config(nemo_path, {"target": "example.Target"})
    resolved = families.resolve_nemo_archive_model_dir(nemo_path)

    assert resolved is not None
    config = json.loads((Path(resolved) / "config.json").read_text())
    assert config == {
        "model_type": "example_nemo",
        "_nemo_archive_path": str(nemo_path),
    }


def test_unknown_nemo_archive_has_no_family_adapter(tmp_path):
    import tensorrt_model_connect.families as families

    nemo_path = tmp_path / "unknown.nemo"
    _write_nemo_config(nemo_path, {"target": "example.UnknownModel"})

    assert families.resolve_nemo_archive_model_dir(nemo_path) is None


def test_repo_family_builders_use_model_local_helpers():
    """Model family builders must not import broad shared builder helpers."""
    families_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "python"
        / "tensorrt_model_connect"
        / "families"
    )
    required_helpers = {
        "graph_ops.py",
        "graph_blocks.py",
        "checkpoint_mapper.py",
        "default_decoder.py",
        "default_dual_profile_decoder.py",
        "default_dual_profile_decoder_tp.py",
        "utils.py",
    }
    forbidden_imports = (
        "from ...checkpoint_mapper import",
        "from ...config import",
        "from ...graph_ops import",
        "from ...graph_blocks import",
        "from ...builders.",
        "from ... import graph_ops",
        "from ... import graph_blocks",
        "from tensorrt_model_connect.checkpoint_mapper import",
        "from tensorrt_model_connect.config import",
        "from tensorrt_model_connect.graph_ops import",
        "from tensorrt_model_connect.graph_blocks import",
        "from tensorrt_model_connect.builders.",
    )

    missing_helpers = []
    central_imports = []
    for family_dir in sorted(families_dir.iterdir()):
        if not family_dir.is_dir() or not (family_dir / "MODEL.toml").is_file():
            continue
        expected = set(required_helpers)
        if family_dir.name == "elf_flow":
            expected.add("model_config.py")
        else:
            expected.add("config.py")
        missing_helpers.extend(
            f"{family_dir.name}/{helper}"
            for helper in sorted(expected)
            if not (family_dir / helper).is_file()
        )

        for path in sorted(family_dir.glob("*.py")):
            if path.name == "MODEL.toml":
                continue
            text = path.read_text(encoding="utf-8")
            if any(item in text for item in forbidden_imports):
                central_imports.append(
                    path.relative_to(families_dir).as_posix()
                )

    assert not missing_helpers
    assert not central_imports


def _discover_plugin_names_from_filesystem() -> set[str]:
    """Scan family plugin .py files with AST to extract plugin name attrs.

    This works even when TRT/torch are not installed, because it only parses
    the Python source — it never imports the modules.  The convention is:

        class FooPlugin:
            name = "foo"
        ...
        plugin = FooPlugin()

    We find every class that has a ``name = "<literal>"`` assignment in its
    body and whose class name is referenced in a module-level
    ``plugin = ClassName()`` assignment.
    """
    names: set[str] = set()
    repo_root = (
        Path(__file__).resolve().parent.parent.parent
        / "python"
        / "tensorrt_model_connect"
    )
    plugin_dirs = [
        repo_root / "families",
    ]

    for families_dir in plugin_dirs:
        py_files = [
            path
            for path in families_dir.glob("*.py")
            if not path.name.startswith("_") and path.stem != "base"
        ]
        py_files.extend(
            path
            for path in families_dir.glob("*/plugin.py")
            if not path.parent.name.startswith("_")
        )
        for py_file in sorted(py_files):
            if py_file.name.startswith("_") or py_file.stem == "base":
                continue
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue

            # 1) Find the class name referenced in  ``plugin = ClassName(...)``
            plugin_class_name: str | None = None
            for node in ast.iter_child_nodes(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "plugin"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                ):
                    plugin_class_name = node.value.func.id
                    break
            if plugin_class_name is None:
                continue

            # 2) Find that class and extract its ``name`` string attribute.
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == plugin_class_name:
                    for item in node.body:
                        if (
                            isinstance(item, ast.Assign)
                            and len(item.targets) == 1
                            and isinstance(item.targets[0], ast.Name)
                            and item.targets[0].id == "name"
                            and isinstance(item.value, ast.Constant)
                            and isinstance(item.value.value, str)
                        ):
                            names.add(item.value.value)
                    break
    return names


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    """Verify plugin auto-discovery finds all expected families."""

    def test_plugin_count(self):
        assert len(_ALL_PLUGINS) >= 20, (
            f"Expected >= 20 plugins, found {len(_ALL_PLUGINS)}: "
            f"{[p.name for p in _ALL_PLUGINS]}")

    def test_all_have_name(self):
        for p in _ALL_PLUGINS:
            assert hasattr(p, "name"), f"Plugin {p} missing 'name' attribute"
            assert isinstance(p.name, str) and p.name, (
                f"Plugin {p} has empty or non-string name")

    def test_all_have_matches(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "matches", None)), (
                f"Plugin {p.name} missing callable 'matches'")

    def test_all_have_load_weights(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "load_weights", None)), (
                f"Plugin {p.name} missing callable 'load_weights'")

    def test_all_have_build_engine(self):
        for p in _ALL_PLUGINS:
            assert callable(getattr(p, "build_engine", None)), (
                f"Plugin {p.name} missing callable 'build_engine'")

    def test_unique_names(self):
        names = [p.name for p in _ALL_PLUGINS]
        assert len(names) == len(set(names)), (
            f"Duplicate plugin names: {names}")

    def test_all_plugins_reject_unknown_types_with_bool(self):
        """Every plugin should return a bool and reject unrelated model types."""
        nonsense_types = [
            "zzzz_nonexistent_model_xyz_42",
            "__bogus__",
            "this_model_does_not_exist_ever_12345",
        ]

        for plugin in _ALL_PLUGINS:
            for bad_type in nonsense_types:
                result = plugin.matches(bad_type)
                assert result is False, (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{result!r}, expected False")
                assert isinstance(result, bool), (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{type(result).__name__}, expected bool")

    def test_all_plugins_have_e2e_manifest(self):
        """Validate that every family plugin has at least one E2E test manifest.

        Uses AST-based filesystem scanning so this test works even without
        TRT/torch installed (pure Python, no GPU). When a developer adds a
        new family plugin, they must also add a JSON manifest in
        tests/e2e/models/ so the E2E test suite covers that model.
        """
        _EXEMPT_PLUGINS: set[str] = set()

        models_dir = Path(__file__).resolve().parent.parent / "e2e" / "models"
        families_in_manifests: set[str] = set()
        for manifest_path in iter_manifest_paths(models_dir):
            with open(manifest_path) as f:
                data = json.load(f)
            family = data.get("family")
            if family:
                families_in_manifests.add(family)

        plugin_names = _discover_plugin_names_from_filesystem()
        assert plugin_names, "No plugin names discovered — AST scan may be broken"
        uncovered = plugin_names - families_in_manifests - _EXEMPT_PLUGINS
        assert not uncovered, (
            f"Plugins without E2E manifest coverage: {uncovered}. "
            f"Add a JSON manifest in tests/e2e/models/ with 'family' matching "
            f"the plugin name, or add to _EXEMPT_PLUGINS if WIP.")

    def test_all_manifests_have_valid_family(self):
        """Validate that every E2E manifest references a family that exists as a plugin.

        Uses AST-based filesystem scanning so this test works without
        TRT/torch installed. A typo or stale reference in a manifest's
        "family" field is caught here rather than at E2E runtime.
        """
        plugin_names = _discover_plugin_names_from_filesystem()
        assert plugin_names, "No plugin names discovered — AST scan may be broken"

        models_dir = Path(__file__).resolve().parent.parent / "e2e" / "models"
        invalid: list[str] = []
        for manifest_path in iter_manifest_paths(models_dir):
            with open(manifest_path) as f:
                data = json.load(f)
            family = data.get("family", "")
            if family and family not in plugin_names:
                invalid.append(f"{manifest_path.name}: family={family!r}")

        assert not invalid, (
            "Manifests referencing unknown family plugins:\n"
            + "\n".join(f"  {entry}" for entry in invalid))


# ---------------------------------------------------------------------------
# match() negative cases
# ---------------------------------------------------------------------------

_NEGATIVE_MATCH_CASES = [
    "unknown_model",
    "clip",
    "",
]


class TestMatchNegative:
    """Unknown model_types should return None."""

    @pytest.mark.parametrize("model_type", _NEGATIVE_MATCH_CASES)
    def test_no_match(self, model_type):
        assert find_plugin(model_type) is None, (
            f"model_type={model_type!r} should not match any plugin")


class TestPluginAttributeShape:
    """Generic protocol checks for optional plugin attributes."""

    def test_runtime_strategy_attribute_is_string_when_declared(self):
        for plugin in _ALL_PLUGINS:
            runtime_strategy = getattr(plugin, "runtime_strategy", None)
            assert runtime_strategy is None or isinstance(runtime_strategy, str), (
                f"Plugin {plugin.name!r} runtime_strategy should be a string "
                f"or absent, got {runtime_strategy!r}")

    def test_embed_input_plugins_expose_vl_entrypoints(self):
        for plugin in _ALL_PLUGINS:
            if not getattr(plugin, "embed_input", False):
                continue
            assert callable(getattr(plugin, "build_vision_engine", None)), (
                f"Plugin {plugin.name!r} has embed_input but no vision builder")
            assert callable(getattr(plugin, "get_vl_config", None)), (
                f"Plugin {plugin.name!r} has embed_input but no VL config hook")
