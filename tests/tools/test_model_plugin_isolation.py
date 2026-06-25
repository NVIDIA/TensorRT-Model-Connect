from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "model_plugin_isolation.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    manifests_dir = repo_root / "tests" / "e2e" / "models" / "decoder_family" / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "decoder-small.json").write_text(
        json.dumps({
            "name": "decoder-small",
            "family": "decoder_family",
            "runtime_strategy": "llama_decoder_kv_cache",
        }),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / "llama"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "MODEL.toml").write_text(
        'id = "llama"\n'
        'runtime_library = "libtrtmc_model_llama.so"\n'
        'runtime_strategies = ["llama_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    return repo_root


def test_targets_resolve_e2e_model_to_runtime_plugin_owner(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
    )

    assert result.stdout.splitlines() == ["trtmc_model_llama"]


@pytest.mark.parametrize(
    ("family", "strategy", "target"),
    [
        ("qwen", "qwen_decoder_kv_cache", "trtmc_model_qwen"),
        ("deepseek_v2", "deepseek_v2_decoder_kv_cache", "trtmc_model_deepseek_v2"),
        ("olmo2", "olmo2_decoder_kv_cache", "trtmc_model_olmo2"),
        ("mixtral", "mixtral_decoder_moe", "trtmc_model_mixtral"),
        ("gpt_oss", "gpt_oss_decoder_moe", "trtmc_model_gpt_oss"),
        ("qwen_moe", "qwen_moe_decoder_moe", "trtmc_model_qwen_moe"),
        (
            "nemotron_labs_diffusion",
            "nemotron_labs_diffusion",
            "trtmc_model_nemotron_labs_diffusion",
        ),
        ("bloom", "bloom_decoder_kv_cache", "trtmc_model_bloom"),
        ("codegen", "codegen_decoder_kv_cache", "trtmc_model_codegen"),
        ("falcon", "falcon_decoder_kv_cache", "trtmc_model_falcon"),
        ("gemma", "gemma_decoder_kv_cache", "trtmc_model_gemma"),
        ("glm", "glm_decoder_kv_cache", "trtmc_model_glm"),
        ("gpt2", "gpt2_decoder_kv_cache", "trtmc_model_gpt2"),
        ("gpt_neo", "gpt_neo_decoder_kv_cache", "trtmc_model_gpt_neo"),
        ("gpt_neox", "gpt_neox_decoder_kv_cache", "trtmc_model_gpt_neox"),
        ("granite", "granite_decoder_kv_cache", "trtmc_model_granite"),
        ("internlm", "internlm_decoder_kv_cache", "trtmc_model_internlm"),
        ("llama", "llama_decoder_kv_cache", "trtmc_model_llama"),
        ("mistral", "mistral_decoder_kv_cache", "trtmc_model_mistral"),
        ("nemotron", "nemotron_decoder_kv_cache", "trtmc_model_nemotron"),
        ("olmo", "olmo_decoder_kv_cache", "trtmc_model_olmo"),
        ("opt", "opt_decoder_kv_cache", "trtmc_model_opt"),
        ("phi", "phi_decoder_kv_cache", "trtmc_model_phi"),
        ("phi_moe", "phi_moe_decoder_kv_cache", "trtmc_model_phi_moe"),
        ("stablelm", "stablelm_decoder_kv_cache", "trtmc_model_stablelm"),
        ("starcoder2", "starcoder2_decoder_kv_cache", "trtmc_model_starcoder2"),
        ("xglm", "xglm_decoder_kv_cache", "trtmc_model_xglm"),
    ],
)
def test_targets_resolve_model_owned_runtime_plugin(
    tmp_path: Path,
    family: str,
    strategy: str,
    target: str,
) -> None:
    repo_root = _make_repo(tmp_path)
    model_name = f"{family}-case"
    manifests_dir = repo_root / "tests" / "e2e" / "models" / family / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{model_name}.json").write_text(
        json.dumps({
            "name": model_name,
            "family": family,
            "runtime_strategy": strategy,
        }),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / family
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "MODEL.toml").write_text(
        f'id = "{family}"\n'
        f'runtime_library = "libtrtmc_model_{family}.so"\n'
        f'runtime_strategies = ["{strategy}"]\n',
        encoding="utf-8",
    )

    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--model",
        model_name,
    )

    assert result.stdout.splitlines() == [target]


def test_targets_resolve_model_owned_node_id_from_tests_file(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]\n",
        encoding="utf-8",
    )

    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--tests-file",
        str(tests_file),
    )

    assert result.stdout.splitlines() == ["trtmc_model_llama"]


def test_prepare_copies_only_selected_runtime_plugin(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    build_dir = tmp_path / "build"
    source_dir = build_dir / "models" / "llama"
    source_dir.mkdir(parents=True)
    source = source_dir / "libtrtmc_model_llama.so"
    source.write_bytes(b"fake-so")

    output_dir = tmp_path / "only-selected"
    result = _run(
        "prepare",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--build-dir",
        str(build_dir),
        "--output-dir",
        str(output_dir),
    )

    copied = output_dir / "llama" / "libtrtmc_model_llama.so"
    assert copied.read_bytes() == b"fake-so"
    assert result.stdout.splitlines() == [f"trtmc_model_llama {copied}"]
