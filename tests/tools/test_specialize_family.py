from __future__ import annotations

import ast
from pathlib import Path

from tools import specialize_family


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    family = (
        tmp_path
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(family / "MODEL.toml", 'id = "demo"\n')
    _write(family / "__init__.py", "from .plugin import plugin\n")
    _write(
        family / "plugin.py",
        "from .model.encoder_builder import build_encoder\n\n"
        "def build_parallel():\n"
        "    from .model.tp_builder import build_tp\n"
        "    return build_tp()\n\n"
        "plugin = build_encoder\n",
    )
    _write(family / "config.py", "class ModelConfig:\n    pass\n")
    _write(family / "weights/__init__.py", "class WeightDict(dict):\n    pass\n")
    _write(family / "model/__init__.py", '"""Model package."""\n')
    _write(
        family / "model/model.py",
        '"""Core graph."""\n\n'
        "from __future__ import annotations\n\n"
        "from .encoder_builder import build_encoder\n\n"
        "def core():\n"
        "    return 1\n",
    )
    _write(
        family / "model/encoder_builder.py",
        '"""Encoder."""\n\n'
        "from __future__ import annotations\n\n"
        "from . import model as graph_ops\n\n"
        "from .model import core as _core\n\n"
        "def build_encoder():\n"
        "    return graph_ops.core() + _core()\n",
    )
    _write(
        family / "model/tp_builder.py",
        '"""Parallel encoder."""\n\n'
        "from __future__ import annotations\n\n"
        "from . import model as graph_ops\n\n"
        "def build_tp():\n"
        "    return graph_ops.core()\n",
    )
    _write(
        tmp_path / "tests/test_demo.py",
        "from tensorrt_model_connect.families.demo.model.tp_builder "
        "import build_tp\n",
    )
    return tmp_path


def test_specialize_family_merges_model_and_renames_parallel(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    specialization = specialize_family.SpecializationLayout(
        model_modules=("encoder_builder.py",),
        parallel_modules=("tp_builder.py",),
        runtime_modules=(),
    )

    specialize_family.specialize_family(repo, "demo", specialization)

    assert not (family / "model/encoder_builder.py").exists()
    assert not (family / "model/tp_builder.py").exists()
    assert (family / "model/parallel.py").is_file()
    model_text = (family / "model/model.py").read_text(encoding="utf-8")
    assert "def core" in model_text
    assert "def build_encoder" in model_text
    assert "graph_ops." not in model_text
    assert "_core" not in model_text
    assert "from .model import" not in model_text
    ast.parse(model_text)
    plugin_text = (family / "plugin.py").read_text(encoding="utf-8")
    assert "from .model.model import build_encoder" in plugin_text
    assert "from .model.parallel import build_tp" in plugin_text
    test_text = (repo / "tests/test_demo.py").read_text(encoding="utf-8")
    assert "families.demo.model.parallel" in test_text


def test_specialize_family_is_idempotent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    specialization = specialize_family.SpecializationLayout(
        model_modules=("encoder_builder.py",),
        parallel_modules=("tp_builder.py",),
        runtime_modules=(),
    )

    specialize_family.specialize_family(repo, "demo", specialization)
    before = {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    }
    specialize_family.specialize_family(repo, "demo", specialization)
    after = {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_specialize_family_moves_runtime_components_and_profiles(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    _write(family / "model/debug_runner.py", "class Runner:\n    pass\n")
    _write(family / "model/vision_builder.py", "def build_vision():\n    return 1\n")
    _write(family / "model/python_profile_verify.py", "VERIFIED = True\n")
    _write(
        family / "model/python_profile_requirements/demo.lock.txt",
        "demo==1\n",
    )
    _write(
        family / "MODEL.toml",
        'id = "demo"\n'
        'debug_runner = "model/debug_runner.py|Runner"\n'
        'config_adapter = "model/vision_builder.py|build_vision"\n'
        'python_profile_specs = ["demo|families/demo/model/'
        'python_profile_requirements/demo.lock.txt|families/demo/model/'
        'python_profile_verify.py|true"]\n',
    )
    _write(
        repo / "tests/test_paths.py",
        "from tensorrt_model_connect.families.demo.model.debug_runner import Runner\n"
        "from tensorrt_model_connect.families.demo.model.vision_builder "
        "import build_vision\n",
    )
    specialization = specialize_family.SpecializationLayout(
        runtime_modules=("debug_runner.py",),
        component_modules=(("vision_builder.py", "vision.py"),),
        profile_paths=(
            ("python_profile_requirements", "requirements"),
            ("python_profile_verify.py", "verify.py"),
        ),
    )

    specialize_family.specialize_family(repo, "demo", specialization)

    assert (family / "model/runtime.py").is_file()
    assert (family / "model/components/vision.py").is_file()
    assert (family / "profiles/requirements/demo.lock.txt").is_file()
    assert (family / "profiles/verify.py").is_file()
    manifest = (family / "MODEL.toml").read_text(encoding="utf-8")
    assert "model/runtime.py|Runner" in manifest
    assert "model/components/vision.py|build_vision" in manifest
    assert "/profiles/requirements/" in manifest
    assert "/profiles/verify.py" in manifest
    imports = (repo / "tests/test_paths.py").read_text(encoding="utf-8")
    assert "model.runtime import Runner" in imports
    assert "model.components.vision import build_vision" in imports


def test_specialize_family_normalizes_profile_manifest_without_runtime(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    _write(family / "model/python_profile_verify.py", "VERIFIED = True\n")
    _write(
        family / "model/python_profile_requirements/demo.lock.txt",
        "demo==1\n",
    )
    _write(
        family / "MODEL.toml",
        'id = "demo"\n'
        'python_profile_specs = ["demo|families/demo/model/'
        'python_profile_requirements/demo.lock.txt|families/demo/model/'
        'python_profile_verify.py|true"]\n',
    )
    specialization = specialize_family.SpecializationLayout(
        profile_paths=(
            ("python_profile_requirements", "requirements"),
            ("python_profile_verify.py", "verify.py"),
        ),
    )

    specialize_family.specialize_family(repo, "demo", specialization)

    manifest = (family / "MODEL.toml").read_text(encoding="utf-8")
    assert "/profiles/requirements/" in manifest
    assert "/profiles/verify.py" in manifest
