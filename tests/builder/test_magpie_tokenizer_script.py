"""Tests for scripts/magpie_tokenizer.py — HF cache path parsing and NeMo resolution.

Trace: ARCH-TOK-001, UD-TOK-MAGPIE
Intent: Validate Magpie tokenizer script HF cache path parsing and NeMo file resolution
Preconditions: Fake HF cache directory structure and snapshot_download mock are available
Postconditions: Repo ID is correctly extracted from cache paths and NeMo file is resolved via fallback download
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_magpie_script():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "magpie_tokenizer.py"
    spec = importlib.util.spec_from_file_location("magpie_tokenizer_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_family_magpie_tokenizer():
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root
        / "tensorrt_model_connect"
        / "tensorrt_model_connect"
        / "families"
        / "magpie_tts"
        / "magpie_tokenizer.py"
    )
    spec = importlib.util.spec_from_file_location("family_magpie_tokenizer", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_id_from_hf_cache_path() -> None:
    mod = _load_magpie_script()
    p = Path("/root/.cache/huggingface/hub/models--nvidia--magpie_tts_multilingual_357m/blobs/abc")
    assert mod._repo_id_from_hf_cache_path(p) == "nvidia/magpie_tts_multilingual_357m"


def test_family_tokenizer_is_model_owned() -> None:
    mod = _load_family_magpie_tokenizer()

    assert callable(mod.load_tokenizer)


def test_magpie_plugin_uses_family_tokenizer_for_ipa_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    plugin_path = (
        root
        / "tensorrt_model_connect"
        / "tensorrt_model_connect"
        / "families"
        / "magpie_tts"
        / "plugin.py"
    )
    source = plugin_path.read_text(encoding="utf-8")

    assert "from . import magpie_tokenizer" in source
    assert "spec_from_file_location" not in source
    assert 'parent.parent.parent.parent / "scripts"' not in source


def test_resolve_nemo_path_falls_back_to_repo_download(monkeypatch, tmp_path: Path) -> None:
    mod = _load_magpie_script()

    nemo_file = tmp_path / "magpie.nemo"
    nemo_file.write_bytes(b"dummy")

    calls: list[tuple[str, bool]] = []

    def fake_snapshot_download(*, repo_id: str, allow_patterns, local_files_only: bool):
        assert allow_patterns == ["*.nemo"]
        calls.append((repo_id, local_files_only))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    resolved = mod._resolve_nemo_path(
        "/root/.cache/huggingface/hub/models--nvidia--magpie_tts_multilingual_357m/blobs/hash"
    )

    assert resolved == nemo_file
    assert calls
    assert calls[0][0] == "nvidia/magpie_tts_multilingual_357m"
