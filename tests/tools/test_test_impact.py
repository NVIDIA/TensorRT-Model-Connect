"""Unit tests for tools/test_impact.py -- zero-false-negative guarantee.

Tests use synthetic manifests and family plugins in tmp directories to
verify rule classification in isolation. The validate test uses the real
repo state.

Trace: ARCH-CI-001, UD-CI-TEST-IMPACT
Intent: Validate test impact analysis rule classification and zero-false-negative guarantee
Preconditions: Synthetic manifests and family plugin files are created in temp directories
Postconditions: Changed files are correctly classified to affected test sets with no false negatives
"""

import json
import sys
from pathlib import Path

import pytest

# Add tools/ to path so we can import test_impact
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import test_impact  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_family(families_dir: Path, name: str, imports: str) -> None:
    (families_dir / f"{name}.py").write_text(imports, encoding="utf-8")


def _write_family_package(families_dir: Path, name: str, files: dict[str, str]) -> None:
    family_dir = families_dir / name
    family_dir.mkdir()
    for rel_path, content in files.items():
        (family_dir / rel_path).write_text(content, encoding="utf-8")


@pytest.fixture
def mock_repo(tmp_path):
    """Create a minimal mock repo with manifests and family plugins."""
    models_dir = tmp_path / "tests" / "e2e" / "models"
    models_dir.mkdir(parents=True)
    families_dir = tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
    families_dir.mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "plugins" / "shared").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "pipelines").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "chronos_bolt").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "text_generation").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "vision_language").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "flux").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "pixart").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "core").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "domains" / "diffusion").mkdir(parents=True)
    (tmp_path / "include" / "trtmc").mkdir(parents=True)
    (tmp_path / "tests" / "e2e" / "data").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "runners").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "comparators").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "plugins").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "references").mkdir(parents=True)
    (tmp_path / "tests" / "e2e_harness" / "thresholds" / "defaults").mkdir(parents=True)
    (tmp_path / "tests" / "builder").mkdir(parents=True)
    (tmp_path / "tests" / "cpp").mkdir(parents=True)
    (tmp_path / "tests" / "tools").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)

    # Manifests
    manifests = [
        {"name": "qwen3-0.6b", "family": "qwen", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "Q/Q3", "core": True},
        {"name": "qwen3-4b", "family": "qwen", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "Q/Q3-4b"},
        {"name": "llama-7b", "family": "llama", "runtime_strategy": "decoder_kv_cache",
         "hf_id": "meta/llama-7b"},
        {"name": "bert-base", "family": "bert", "runtime_strategy": "encoder_only",
         "hf_id": "bert-base", "core": True},
        {"name": "whisper-tiny-fp16", "family": "whisper", "runtime_strategy": "speech_to_text",
         "hf_id": "openai/whisper-tiny", "precision": "fp16", "core": True,
         "test_input_audio": "tests/e2e/data/Recording.wav"},
        {"name": "flux-schnell", "family": "flux", "runtime_strategy": "diffusion_flux",
         "hf_id": "bf/FLUX", "core": True},
        {"name": "z-image-turbo", "family": "z_image", "runtime_strategy": "diffusion_zimage",
         "hf_id": "Tongyi-MAI/Z-Image-Turbo"},
        {"name": "flux-2-dev", "family": "flux", "runtime_strategy": "diffusion_flux",
         "hf_id": "bf/FLUX2"},
        {"name": "flux-2-dev-fp8", "family": "flux", "runtime_strategy": "diffusion_flux",
         "hf_id": "bf/FLUX2", "fp8_scales": "flux2-fp8-scales.json"},
        {"name": "mamba-130m", "family": "mamba", "runtime_strategy": "ssm_recurrent",
         "hf_id": "ss/mamba", "core": True},
        {"name": "qwen25vl-3b", "family": "qwen_vl", "runtime_strategy": "vision_language",
         "hf_id": "Q/Q25VL", "test_image": "data/test_img.jpeg", "core": True},
        {"name": "bark-small", "family": "bark", "runtime_strategy": "text_to_audio_bark",
         "hf_id": "suno/bark", "core": True},
        {"name": "sam-vit", "family": "sam", "runtime_strategy": "prompted_segmentation",
         "hf_id": "fb/sam", "core": True},
        {"name": "segformer-b0", "family": "segformer", "runtime_strategy": "segmentation",
         "hf_id": "nv/segformer", "core": True},
        {"name": "mixtral-15m", "family": "mixtral", "runtime_strategy": "decoder_moe",
         "hf_id": "mist/mixtral", "core": True},
        {"name": "chronos-bolt-small", "family": "chronos_bolt",
         "runtime_strategy": "chronos_bolt_torchtrt",
         "hf_id": "amazon/chronos-bolt-small", "core": True},
        {"name": "convbert-base", "family": "convbert", "runtime_strategy": "encoder_only",
         "hf_id": "YituTech/conv-bert-base"},
    ]
    for m in manifests:
        _write_json(models_dir / f"{m['name']}.json", m)

    # Family plugins
    (families_dir / "__init__.py").write_text("")
    (families_dir / "base.py").write_text("")
    _write_family_package(
        families_dir,
        "qwen",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "llama",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "bert",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .encoder_builder import build\nfrom ...config import C\n",
            "encoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family(families_dir, "whisper",
                  "from ..config import C\nfrom ..graph_ops import rope\n")
    _write_family(families_dir, "flux",
                  "from ..config import C\n")
    _write_family(families_dir, "mamba",
                  "from ..config import C\nfrom ..graph_ops import ssm\n")
    _write_family_package(
        families_dir,
        "qwen_vl",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "bark",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family(families_dir, "sam",
                  "from ..config import C\nfrom ..graph_ops import rope\n")
    _write_family(families_dir, "segformer",
                  "from ..config import C\nfrom ..graph_ops import conv\n")
    _write_family_package(
        families_dir,
        "mixtral",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "convbert",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .builder import build_convbert_encoder_engine\n",
            "builder.py": "from ... import graph_ops\n",
        },
    )

    # Placeholder source files
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "standard_decoder_builder.py").write_text("")
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "encoder_builder.py").write_text("")
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "config.py").write_text("")
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "checkpoint_mapper.py").write_text("")
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "graph_ops.py").write_text("")
    (tmp_path / "tensorrt_model_connect" / "tensorrt_model_connect" / "engine_defs" / "torch_trt" / "strategies").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "text_generation" / "MODEL.toml").write_text(
        'runtime_strategies = ["decoder_kv_cache", "decoder_moe"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "vision_language" / "MODEL.toml").write_text(
        'runtime_strategies = ["vision_language"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "flux" / "MODEL.toml").write_text(
        'runtime_strategies = ["diffusion_flux"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "flux" / "pipeline.cpp").write_text(
        '#include "runtime/core/gpu_matmul.h"\n'
        '#include "runtime/domains/diffusion/diffusion_denoising_step_seam.h"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "pixart" / "MODEL.toml").write_text(
        'runtime_strategies = ["diffusion_pixart"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "pixart" / "pipeline.cpp").write_text(
        '#include "runtime/domains/diffusion/diffusion_denoising_step_seam.h"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "chronos_bolt" / "MODEL.toml").write_text(
        'runtime_strategies = ["chronos_bolt_torchtrt"]\n'
        'task_strategy = "neural_operator"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "chronos_bolt" / "plugin.cpp").write_text(
        '#include "runtime/models/chronos_bolt/pipeline.h"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "chronos_bolt" / "pipeline.cpp").write_text(
        '#include "runtime/models/chronos_bolt/pipeline.h"\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e" / "data" / "flux2-fp8-scales.json").write_text(
        "{}",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def imap(mock_repo):
    return test_impact.build_impact_map(mock_repo)


# ---------------------------------------------------------------------------
# Declarative rule table tests
# ---------------------------------------------------------------------------


class TestDeclarativeClassificationRules:
    def test_rule_table_has_unique_priorities_and_declared_coverage(self):
        """Every classification rule declares order and test coverage."""
        priorities = [rule.priority for rule in test_impact.CLASSIFICATION_RULES]

        assert priorities == sorted(priorities)
        assert len(priorities) == len(set(priorities))
        assert test_impact.CLASSIFICATION_RULES[-1].name == "catch_all"
        assert all(rule.covered_by for rule in test_impact.CLASSIFICATION_RULES)

    def test_scoped_cpp_helper_precedes_generic_cpp_source(self):
        """Scoped C++ rules stay narrower than the generic C++ fallback."""
        priorities = {
            rule.name: rule.priority for rule in test_impact.CLASSIFICATION_RULES
        }

        assert priorities["cpp_scoped_helper"] < priorities["cpp_source"]

    @pytest.mark.parametrize(
        "path,rule_name",
        [
            (
                "tensorrt_model_connect/tensorrt_model_connect/families/whisper.py",
                "family_plugin",
            ),
            (
                "tensorrt_model_connect/tensorrt_model_connect/"
                "engine_defs/torch_trt/families/base.py",
                "torchtrt_family_base",
            ),
            (
                "tensorrt_model_connect/tensorrt_model_connect/"
                "engine_defs/torch_trt/strategies/custom.py",
                "torchtrt_strategy_unknown",
            ),
            ("src/runtime/models/custom_backend/plugin.cpp", "cpp_runtime_model_unknown"),
            ("src/runtime/plugins/flux_plugin.cpp", "cpp_plugin_flux_runtime"),
            ("src/runtime/plugins/decoder_plugin.cpp", "cpp_plugin"),
            ("src/runtime/plugins/custom_plugin.cpp", "cpp_plugin_unknown"),
            ("src/runtime/pipelines/flux_pipeline.cpp", "cpp_pipeline_flux_runtime"),
            ("src/runtime/pipelines/text_generation_pipeline.cpp", "cpp_pipeline"),
            ("src/runtime/pipelines/custom_pipeline.cpp", "cpp_pipeline_unknown"),
            ("src/runtime/plugins/shared/custom_helper.h", "cpp_shared_helper_unknown"),
            ("tests/e2e_harness/runners/__init__.py", "harness_runner_init"),
            ("tests/e2e_harness/runners/custom.py", "harness_runner_unknown"),
            ("tests/e2e_harness/comparators/__init__.py", "harness_comparator_init"),
            ("tests/e2e_harness/comparators/custom.py", "harness_comparator_unknown"),
            ("tests/e2e_harness/references/__init__.py", "harness_reference_init"),
            ("tests/e2e_harness/references/custom.py", "harness_reference_unknown"),
            ("tests/e2e_harness/plugins/__init__.py", "harness_plugin_init"),
            ("tests/e2e_harness/plugins/custom.py", "harness_plugin_unknown"),
            (
                "tests/e2e_harness/thresholds/defaults/custom.json",
                "harness_threshold_unknown",
            ),
            ("tests/e2e_harness/test_orchestrator_phases.py", "harness_unit_test"),
            ("tools/make_elf_replay_artifact.py", "elf_replay_tool"),
        ],
    )
    def test_representative_rule_paths(self, imap, path, rule_name):
        """Representative paths keep their existing rule names."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == rule_name

    def test_specialized_builder_rule(self, mock_repo):
        """Root builder imports still use the dynamic family import index."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        families_dir = (
            mock_repo
            / "tensorrt_model_connect"
            / "tensorrt_model_connect"
            / "families"
        )
        _write_json(
            models_dir / "custom-builder-model.json",
            {
                "name": "custom-builder-model",
                "family": "custom_builder_family",
                "runtime_strategy": "encoder_only",
                "hf_id": "custom/model",
            },
        )
        _write_family(
            families_dir,
            "custom_builder_family",
            "from ..custom_builder import build\n",
        )
        (
            mock_repo
            / "tensorrt_model_connect"
            / "tensorrt_model_connect"
            / "custom_builder.py"
        ).write_text("", encoding="utf-8")

        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/custom_builder.py",
            imap,
        )

        assert match.rule == "specialized_builder"
        assert match.models == ["custom-builder-model"]


# ---------------------------------------------------------------------------
# Family isolation tests
# ---------------------------------------------------------------------------


class TestFamilyPlugin:
    def test_family_only_change(self, imap):
        """families/qwen/plugin.py -> exactly qwen models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py", imap)
        assert match.rule == "family_package"
        assert sorted(match.models) == ["qwen3-0.6b", "qwen3-4b"]

    def test_family_isolation(self, imap):
        """families/qwen/plugin.py does NOT affect llama models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py", imap)
        assert "llama-7b" not in match.models

    def test_family_with_no_manifest(self, imap):
        """A family package with no manifest -> empty models, no crash."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/nonexistent_family/plugin.py", imap)
        assert match.rule == "family_package"
        assert match.models == []

    def test_internal_family_folder_is_not_model_owned(self, imap):
        """families/_internal files are not a model ownership boundary."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/_internal/helper.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_base_all_models(self, imap):
        """families/base.py -> ALL models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/base.py", imap)
        assert match.rule == "family_base"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_init_all_models(self, imap):
        """families/__init__.py -> ALL models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/__init__.py", imap)
        assert match.rule == "family_base"
        assert len(match.models) == len(imap.all_model_names)

    def test_family_package_file(self, imap):
        """families/convbert/builder.py -> exactly ConvBERT models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/convbert/builder.py", imap)
        assert match.rule == "family_package"
        assert match.models == ["convbert-base"]

    def test_family_package_plugin(self, imap):
        """families/convbert/plugin.py uses the package folder as its impact boundary."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/convbert/plugin.py", imap)
        assert match.rule == "family_package"
        assert match.models == ["convbert-base"]

    def test_torchtrt_family_only_change(self, mock_repo):
        """Torch-TRT family plugin change maps only to that family's manifests."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        _write_json(
            models_dir / "patchtst-granite-official.json",
            {
                "name": "patchtst-granite-official",
                "family": "patchtst",
                "runtime_strategy": "patchtst_torchtrt",
                "hf_id": "ibm-granite/granite-timeseries-patchtst",
            },
        )
        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/patchtst.py",
            imap,
        )
        assert match.rule == "torchtrt_family_plugin"
        assert match.models == ["patchtst-granite-official"]


# ---------------------------------------------------------------------------
# Shared module tests (broad impact)
# ---------------------------------------------------------------------------


class TestSharedModules:
    def test_shared_module_all_models(self, imap):
        """checkpoint_mapper.py -> all models (no escalation)."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_shared_module_with_cap(self, imap):
        """checkpoint_mapper.py + cap -> core models only."""
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py"], imap, cap=5)
        assert result.cap_applied
        assert sorted(result.e2e_models) == sorted(imap.core_models)

    def test_graph_ops_all_models(self, imap):
        """graph_ops.py -> all models (shared utility, not a builder)."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/graph_ops.py", imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)

    def test_config_all_models(self, imap):
        """config.py -> all models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/config.py", imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# Specialized builder tests
# ---------------------------------------------------------------------------


class TestFamilyOwnedBuilder:
    def test_root_standard_decoder_builder_shim_is_broad(self, imap):
        """Root standard_decoder_builder.py is only a compatibility shim."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_local_standard_decoder_builder(self, imap):
        """families/qwen/standard_decoder_builder.py -> exactly qwen models."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/standard_decoder_builder.py",
            imap,
        )
        assert match.rule == "family_package"
        assert sorted(match.models) == ["qwen3-0.6b", "qwen3-4b"]

    def test_root_encoder_builder_shim_is_broad(self, imap):
        """Root encoder_builder.py is only a compatibility shim."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/encoder_builder.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_local_encoder_builder(self, imap):
        """families/bert/encoder_builder.py -> exactly bert family."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/bert/encoder_builder.py", imap)
        assert match.rule == "family_package"
        assert set(match.models) == {"bert-base"}


# ---------------------------------------------------------------------------
# C++ scope tests
# ---------------------------------------------------------------------------


class TestCppScope:
    def test_cpp_runtime_text_generation_scope(self, imap):
        """text_generation runtime model files -> only decoder models."""
        match = test_impact.classify_file(
            "src/runtime/models/text_generation/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert match.rebuild_cpp is True
        # Should include qwen, llama (kv_cache) and mixtral (moe)
        assert "qwen3-0.6b" in match.models
        assert "mixtral-15m" in match.models
        # Should NOT include non-decoder models
        assert "bert-base" not in match.models
        assert "flux-schnell" not in match.models

    def test_cpp_runtime_vision_language_scope(self, imap):
        """vision_language runtime model files -> only vision_language models."""
        match = test_impact.classify_file(
            "src/runtime/models/vision_language/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert set(match.models) == {"qwen25vl-3b"}

    def test_cpp_shared_audio(self, imap):
        """audio_helpers.h -> only audio pipeline models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/audio_helpers.h", imap)
        assert match.rule == "cpp_shared_helper"
        assert "whisper-tiny-fp16" in match.models
        assert "bark-small" in match.models
        assert "bert-base" not in match.models

    def test_cpp_shared_diffusion(self, imap):
        """diffusion_helpers.cpp -> only diffusion models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/diffusion_helpers.cpp", imap)
        assert match.rule == "cpp_shared_helper"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_cpp_shared_plugin_helpers(self, imap):
        """plugin_helpers.h -> ALL models."""
        match = test_impact.classify_file(
            "src/runtime/plugins/shared/plugin_helpers.h", imap)
        assert match.rule == "cpp_shared_plugin_helpers"
        assert len(match.models) == len(imap.all_model_names)

    def test_cpp_wildcard_all(self, imap):
        """trt_common.cpp -> all models (generic C++ source)."""
        match = test_impact.classify_file(
            "src/runtime/trt/trt_common.cpp", imap)
        assert match.rule == "cpp_source"
        assert len(match.models) == len(imap.all_model_names)

    def test_cpp_pipeline_scope(self, imap):
        """text_generation pipeline.cpp -> only decoder models."""
        match = test_impact.classify_file(
            "src/runtime/models/text_generation/pipeline.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert "qwen3-0.6b" in match.models
        assert "bert-base" not in match.models

    def test_flux_pipeline_runtime_scope_uses_non_fp8_l0_representative(self, imap):
        """flux pipeline.cpp is runtime-only, so FLUX.2 BF16 covers FP8 contract."""
        match = test_impact.classify_file(
            "src/runtime/models/flux/pipeline.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert "flux-2-dev" in match.models
        assert "flux-schnell" in match.models
        assert "flux-2-dev-fp8" not in match.models

    def test_flux_plugin_runtime_scope_uses_non_fp8_l0_representative(self, imap):
        """flux plugin.cpp is runtime-only, so it does not duplicate FP8 builder coverage."""
        match = test_impact.classify_file(
            "src/runtime/models/flux/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert "flux-2-dev" in match.models
        assert "flux-schnell" in match.models
        assert "flux-2-dev-fp8" not in match.models

    def test_cpp_runtime_model_scope(self, imap):
        """src/runtime/models/<strategy> files are scoped by MODEL.toml."""
        match = test_impact.classify_file(
            "src/runtime/models/chronos_bolt/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert match.rebuild_cpp is True
        assert match.models == ["chronos-bolt-small"]

    def test_cpp_runtime_model_manifest_scope(self, imap):
        """MODEL.toml itself is model-runtime scoped."""
        match = test_impact.classify_file(
            "src/runtime/models/chronos_bolt/MODEL.toml", imap)
        assert match.rule == "cpp_runtime_model"
        assert match.models == ["chronos-bolt-small"]

    def test_scoped_cpp_helper_gpu_matmul(self, imap):
        """gpu_matmul.cpp -> only the pipelines that reference it."""
        match = test_impact.classify_file(
            "src/runtime/core/gpu_matmul.cpp", imap)
        assert match.rule == "cpp_scoped_helper"
        assert "flux-schnell" in match.models
        assert "flux-2-dev" in match.models
        assert "flux-2-dev-fp8" not in match.models
        assert "qwen3-0.6b" not in match.models

    def test_scoped_cpp_helper_diffusion_seam(self, imap):
        """diffusion seam helper -> only diffusion pipelines that include it."""
        match = test_impact.classify_file(
            "src/runtime/domains/diffusion/diffusion_denoising_step_seam.h", imap)
        assert match.rule == "cpp_scoped_helper"
        assert "flux-schnell" in match.models
        assert "flux-2-dev" in match.models
        assert "flux-2-dev-fp8" not in match.models
        assert "qwen3-0.6b" not in match.models


# ---------------------------------------------------------------------------
# Safety net tests
# ---------------------------------------------------------------------------


class TestSafetyNet:
    def test_unknown_file_triggers_all(self, imap):
        """Unknown file -> ALL models (catch-all)."""
        match = test_impact.classify_file("some/new/directory/file.py", imap)
        assert match.rule == "catch_all"
        assert sorted(match.models) == sorted(imap.all_model_names)
        assert match.rebuild_cpp is True

    def test_manifest_self(self, imap):
        """Changing a manifest JSON -> only that one model."""
        match = test_impact.classify_file(
            "tests/e2e/models/qwen3-0.6b.json", imap)
        assert match.rule == "manifest"
        assert match.models == ["qwen3-0.6b"]

    def test_cmake_no_e2e_models(self, imap):
        """CMakeLists.txt -> no E2E models (build infra only) + rebuild flag."""
        match = test_impact.classify_file("CMakeLists.txt", imap)
        assert match.rule == "cmake"
        assert match.models == []
        assert match.rebuild_cpp is True

    def test_include_header(self, imap):
        """include/ header -> all models."""
        match = test_impact.classify_file(
            "include/trtmc/runtime/pipeline_factory.h", imap)
        assert match.rule == "cpp_source"
        assert len(match.models) == len(imap.all_model_names)


# ---------------------------------------------------------------------------
# No-impact tests
# ---------------------------------------------------------------------------


class TestNoImpact:
    def test_docs_no_impact(self, imap):
        """website/docs/ -> no E2E tests."""
        match = test_impact.classify_file("website/docs/wiki/Home.md", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_tools_no_impact(self, imap):
        """tools/diff_logits.py -> no E2E tests."""
        match = test_impact.classify_file("tools/diff_logits.py", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_e2e_runner_scripts_trigger_all_models(self, imap):
        """E2E runner changes must not skip E2E validation."""
        match = test_impact.classify_file("scripts/run_e2e_parallel.sh", imap)
        assert match.rule == "e2e_runner_script"
        assert match.models == imap.all_model_names

    def test_scripts_no_impact(self, imap):
        """scripts/ -> no E2E tests."""
        match = test_impact.classify_file("scripts/validate_family.sh", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_markdown_no_impact(self, imap):
        """*.md files -> no E2E tests."""
        match = test_impact.classify_file("AGENTS.md", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_github_ci_no_impact(self, imap):
        """.github workflows -> no E2E tests."""
        match = test_impact.classify_file(".github/workflows/trtmc-ci.yml", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    def test_gitignore_no_impact(self, imap):
        """.gitignore -> no E2E tests."""
        match = test_impact.classify_file(".gitignore", imap)
        assert match.rule == "no_impact"
        assert match.models == []

    @pytest.mark.parametrize(
        "path",
        [
            ".agents/plugins/marketplace.json",
            "plugins/trtmc-agent-skills/.codex-plugin/plugin.json",
            "plugins/trtmc-agent-skills/skills/fp16-trt-network/agents/openai.yaml",
        ],
    )
    def test_agent_plugin_metadata_no_impact(self, imap, path):
        """Codex agent/plugin metadata should not trigger E2E selection."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "no_impact"
        assert match.models == []
        assert match.unit_tiers == []
        assert match.rebuild_cpp is False

    def test_agent_plugin_metadata_aggregate_no_e2e(self, imap):
        """Agent-only plugin edits should aggregate to no selected E2E models."""
        result = test_impact.analyze_impact(
            [
                ".agents/plugins/marketplace.json",
                "plugins/trtmc-agent-skills/.codex-plugin/plugin.json",
                "plugins/trtmc-agent-skills/skills/debug-trt-mismatch/SKILL.md",
                "plugins/trtmc-agent-skills/skills/debug-trt-mismatch/agents/openai.yaml",
            ],
            imap,
        )
        assert result.e2e_models == []
        assert result.unit_tiers == []
        assert result.rebuild_cpp is False


class TestE2EDataFiles:
    def test_data_file_maps_to_manifest_users(self, imap):
        """Checked-in E2E data should map to manifests that reference it."""
        match = test_impact.classify_file(
            "tests/e2e/data/flux2-fp8-scales.json", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["flux-2-dev-fp8"]

    def test_repo_relative_data_file_maps_to_manifest_users(self, imap):
        """Manifest tests/e2e/data references should select only their users."""
        match = test_impact.classify_file(
            "tests/e2e/data/Recording.wav", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["whisper-tiny-fp16"]

    def test_manifest_relative_data_file_maps_to_manifest_users(self, imap):
        """Manifest data/ references resolve relative to tests/e2e/."""
        match = test_impact.classify_file(
            "tests/e2e/data/test_img.jpeg", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["qwen25vl-3b"]


# ---------------------------------------------------------------------------
# Unit tier tests
# ---------------------------------------------------------------------------


class TestUnitTiers:
    def test_unit_tier_builder(self, imap):
        """tests/builder/ -> unit tier 'builder', no E2E."""
        match = test_impact.classify_file(
            "tests/builder/test_config.py", imap)
        assert match.rule == "unit_builder"
        assert match.models == []
        assert "builder" in match.unit_tiers

    def test_unit_tier_cpp(self, imap):
        """tests/cpp/ -> unit tier 'cpp', no E2E."""
        match = test_impact.classify_file(
            "tests/cpp/test_bundle_format.cpp", imap)
        assert match.rule == "unit_cpp"
        assert match.models == []
        assert "cpp" in match.unit_tiers

    def test_unit_tier_tools(self, imap):
        """tests/tools/ -> unit tier 'tools', no E2E."""
        match = test_impact.classify_file(
            "tests/tools/test_diff_logits.py", imap)
        assert match.rule == "unit_tools"
        assert match.models == []
        assert "tools" in match.unit_tiers

    def test_elf_replay_tools_trigger_tools_tier(self, imap):
        """ELF helper tool edits run tools-tier tests without E2E."""
        for path in (
            "tools/make_elf_replay_artifact.py",
            "tools/prepare_elf_model_dir.py",
            "tools/validate_elf_replay_artifact.py",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "elf_replay_tool"
            assert match.models == []
            assert match.unit_tiers == ["tools"]

    def test_unit_tier_torchtrt_engine_defs(self, imap):
        """Torch-TRT engine-def tests run as builder tests without E2E."""
        match = test_impact.classify_file(
            "tests/engine_defs/torch_trt/test_config.py", imap)
        assert match.rule == "unit_torchtrt_builder"
        assert match.models == []
        assert "builder" in match.unit_tiers

    def test_source_implies_unit_tier(self, imap):
        """C++ source change implies 'cpp' unit tier alongside E2E."""
        match = test_impact.classify_file(
            "src/runtime/trt/trt_common.cpp", imap)
        assert "cpp" in match.unit_tiers
        assert len(match.models) > 0

    def test_builder_source_implies_unit_tier(self, imap):
        """Python builder source change implies 'builder' unit tier."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py", imap)
        assert "builder" in match.unit_tiers


# ---------------------------------------------------------------------------
# E2E harness tests
# ---------------------------------------------------------------------------


class TestHarness:
    def test_harness_runner(self, imap):
        """runners/text_generation.py -> text_generation_causal models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/runners/text_generation.py", imap)
        assert match.rule == "harness_runner"
        assert "qwen3-0.6b" in match.models
        assert "bert-base" not in match.models

    def test_harness_comparator(self, imap):
        """comparators/diffusion.py -> diffusion models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/comparators/diffusion.py", imap)
        assert match.rule == "harness_comparator"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_harness_plugin(self, imap):
        """plugins/diffusion.py -> diffusion models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/plugins/diffusion.py", imap)
        assert match.rule == "harness_plugin"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_harness_threshold_profile(self, imap):
        """Diffusion threshold profiles should stay scoped to diffusion models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/thresholds/defaults/diffusion_media_generation.json",
            imap)
        assert match.rule == "harness_threshold_profile"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_torchtrt_diffusion_strategy(self, imap):
        """Torch-TRT diffusion strategy changes should stay scoped to diffusion."""
        match = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/diffusion.py",
            imap)
        assert match.rule == "torchtrt_strategy"
        assert "flux-schnell" in match.models
        assert "qwen3-0.6b" not in match.models

    def test_harness_shared(self, imap):
        """e2e_harness/orchestrator.py -> ALL models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/orchestrator.py", imap)
        assert match.rule == "harness_shared"
        assert len(match.models) == len(imap.all_model_names)

    def test_harness_unit_test_file(self, imap):
        """e2e_harness/test_*.py -> direct tools-tier test only."""
        match = test_impact.classify_file(
            "tests/e2e_harness/test_orchestrator_phases.py", imap)
        assert match.rule == "harness_unit_test"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_torch_reference_includes_neural_operator_models(self, mock_repo):
        """torch_reference.py includes neural_operator-backed time-series manifests."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        _write_json(
            models_dir / "patchtst-granite-official.json",
            {
                "name": "patchtst-granite-official",
                "family": "patchtst",
                "runtime_strategy": "patchtst_torchtrt",
                "hf_id": "ibm-granite/granite-timeseries-patchtst",
            },
        )
        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file(
            "tests/e2e_harness/references/torch_reference.py",
            imap,
        )
        assert match.rule == "harness_reference"
        assert "patchtst-granite-official" in match.models

    def test_test_e2e_entrypoint(self, imap):
        """tests/test_e2e.py -> ALL models."""
        match = test_impact.classify_file("tests/test_e2e.py", imap)
        assert match.rule == "e2e_entrypoint"
        assert len(match.models) == len(imap.all_model_names)

    def test_conftest_entrypoint(self, imap):
        """tests/conftest.py -> ALL models."""
        match = test_impact.classify_file("tests/conftest.py", imap)
        assert match.rule == "e2e_entrypoint"
        assert len(match.models) == len(imap.all_model_names)

    def test_diff_refinement_rules_are_named_in_dispatch_order(self):
        """Diff refinement dispatch keeps named rules in reviewable order."""
        assert [rule.name for rule in test_impact.DIFF_REFINEMENT_RULES] == [
            "harness_shared_fp8_scales",
            "e2e_warm_hf_cache_diffusers_components",
            "shared_builder_fp8_scales_cli",
            "shared_builder_fp8_scales_engine",
            "shared_builder_diffusion_tokenizer",
            "torchtrt_compiler_tokenizer",
            "harness_manifest_diffusion_thresholds",
            "harness_reference_dpr_context_encoder",
            "harness_reference_vl_generated_only_decode",
            "e2e_waives_model_lines",
        ]
        assert all(callable(rule.matches) and callable(rule.refine)
                   for rule in test_impact.DIFF_REFINEMENT_RULES)

    def test_harness_shared_fp8_scales_rule_refines_orchestrator_diff(self, imap):
        """Diff-only fp8_scales plumbing narrows orchestrator scope."""
        diff_text = """
diff --git a/tests/e2e_harness/orchestrator.py b/tests/e2e_harness/orchestrator.py
@@ -1 +1 @@
-    CILane,
+    fp8_scales = case.metadata.get("fp8_scales")
+    if fp8_scales:
+        # Resolve relative to tests/e2e/data/
+        scales_path = Path(__file__).parent.parent / "e2e" / "data" / fp8_scales
+        if scales_path.is_file():
+            cmd.extend(["--fp8-scales", str(scales_path)])
"""
        broad = test_impact.classify_file("tests/e2e_harness/orchestrator.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/orchestrator.py", broad, diff_text, imap)
        assert refined.rule == "harness_shared_fp8_scales"
        assert refined.models == ["flux-2-dev-fp8"]

    def test_e2e_warm_hf_cache_diffusers_components_rule_refines_component_diff(self, imap):
        """Diffusers component-cache validation narrows to FP8 Diffusers coverage."""
        diff_text = """
diff --git a/scripts/warm_hf_cache.py b/scripts/warm_hf_cache.py
@@ -1 +1 @@
+_DIFFUSERS_WEIGHT_COMPONENTS = {"text_encoder", "text_encoder_2", "transformer", "vae"}
+    if (snapshot_dir / "model_index.json").is_file():
+        return has_entrypoint and has_weights and not _diffusers_missing_weight_components(snapshot_dir)
+def _diffusers_missing_weight_components(snapshot_dir: pathlib.Path) -> list[str]:
+    model_index = json.loads(model_index_path.read_text())
+    required_components = sorted(name for name, value in model_index.items())
+def _is_diffusers_component_enabled(value: object) -> bool:
+def _component_has_weight(snapshot_dir: pathlib.Path, component: str) -> bool:
+    component_dir = snapshot_dir / component
+        "entrypoint or required local weight artifact")
"""
        broad = test_impact.classify_file("scripts/warm_hf_cache.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "scripts/warm_hf_cache.py", broad, diff_text, imap)
        assert refined.rule == "e2e_warm_hf_cache_diffusers_components"
        assert refined.models == ["flux-2-dev-fp8"]

    def test_harness_manifest_diffusion_thresholds_rule_refines_manifest_loader_diff(self, imap):
        """Diffusion-only threshold plumbing in manifest_loader narrows scope."""
        diff_text = """
diff --git a/tests/e2e_harness/manifest_loader.py b/tests/e2e_harness/manifest_loader.py
@@ -1 +1 @@
+    if "reference_min_pixel_std_for_ratio" in manifest:
+        overrides["reference_min_pixel_std_for_ratio"] = manifest["reference_min_pixel_std_for_ratio"]
+    if "min_reference_std_ratio" in manifest:
+        overrides["min_reference_std_ratio"] = manifest["min_reference_std_ratio"]
"""
        broad = test_impact.classify_file("tests/e2e_harness/manifest_loader.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/manifest_loader.py", broad, diff_text, imap)
        assert refined.rule == "harness_manifest_diffusion_thresholds"
        assert "flux-schnell" in refined.models
        assert "qwen3-0.6b" not in refined.models

    def test_harness_reference_vl_generated_only_decode_rule_refines_hf_vl_diff(self, imap):
        """VL generated-only decode fallback is scoped to InternVL3-8B."""
        diff_text = """
diff --git a/tests/e2e_harness/references/hf_transformers.py b/tests/e2e_harness/references/hf_transformers.py
@@ -1 +1 @@
+def _decode_vl_generated_text(processor, generated_ids, input_len: int) -> str:
+    token_count = len(generated_ids)
+    def _decode_token_ids(token_ids) -> str:
+        return processor.decode(token_ids, skip_special_tokens=True).strip()
+    prompt_texts = (prompt, fallback_text, text_input)
+    if input_len > 0 and token_count > input_len:
+        text = _decode_token_ids(generated_ids[input_len:])
+            return text
+    if not text.strip():
+        raise RuntimeError("HF VL reference produced empty or prompt-only generated text")
+    return _decode_token_ids(generated_ids)
+            from tests.e2e_harness.references.hf_transformers import (
+                _decode_vl_generated_text,
+            )
+            text = _decode_vl_generated_text(
+                processor, generated_ids[0], input_len, prompt_texts)
"""
        broad = test_impact.classify_file(
            "tests/e2e_harness/references/hf_transformers.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/references/hf_transformers.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "harness_reference_vl_generated_only_decode"
        assert refined.models == ["internvl3-8b"]

    def test_harness_reference_dpr_context_encoder_rule_refines_hf_dpr_diff(self, imap):
        """DPR-only reference routing should not select every HF model."""
        diff_text = """
diff --git a/tests/e2e_harness/references/hf_transformers.py b/tests/e2e_harness/references/hf_transformers.py
@@ -1 +1 @@
-            tokenizer = AutoTokenizer.from_pretrained(
-                model_ref, trust_remote_code=trust_remote_code)
+                # AutoTokenizer/AutoModel route this context checkpoint through
+                # the DPR question classes in transformers 5.x. Use the
+                # context fast tokenizer so HF sees the same token ids as the
+                # tokenizer.json bundled into the TRT artifact.
+                from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
+                tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
+                model = _dpr.ctx_encoder.bert_model
+                tokenizer = AutoTokenizer.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
"""
        broad = test_impact.classify_file(
            "tests/e2e_harness/references/hf_transformers.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/references/hf_transformers.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "harness_reference_dpr_context_encoder"
        assert refined.models == ["dpr-ctx-encoder"]

    def test_e2e_waives_model_lines_rule_refines_named_model_diff(self, imap):
        """A waiver change for one known model should only re-run that model."""
        diff_text = """
diff --git a/tests/e2e/waives.txt b/tests/e2e/waives.txt
@@ -1 +1 @@
-flux-schnell XFAIL (old waiver)
"""
        broad = test_impact.classify_file("tests/e2e/waives.txt", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e/waives.txt", broad, diff_text, imap)
        assert refined.rule == "e2e_waives_model_lines"
        assert refined.models == ["flux-schnell"]

class TestDiffAwareBuilderRefinement:
    def test_shared_builder_fp8_scales_cli_rule_refines_cli_fp8_diff(self, imap):
        """CLI fp8-only plumbing narrows to fp8-scales manifests."""
        diff_text = """
diff --git a/tensorrt_model_connect/tensorrt_model_connect/cli.py b/tensorrt_model_connect/tensorrt_model_connect/cli.py
@@ -1 +1 @@
+    save_fp8_scales = getattr(args, 'save_fp8_scales', None)
+            save_fp8_scales=save_fp8_scales,
+    build_p.add_argument("--save-fp8-scales", default=None,
"""
        broad = test_impact.classify_file("tensorrt_model_connect/tensorrt_model_connect/cli.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tensorrt_model_connect/tensorrt_model_connect/cli.py", broad, diff_text, imap)
        assert refined.rule == "shared_builder_fp8_scales_cli"
        assert refined.models == ["flux-2-dev-fp8"]

    def test_shared_builder_fp8_scales_engine_rule_refines_engine_fp8_diff(self, imap):
        """Diffusion fp8-only engine_builder changes narrow to fp8-scales manifests."""
        diff_text = """
diff --git a/tensorrt_model_connect/tensorrt_model_connect/engine_builder.py b/tensorrt_model_connect/tensorrt_model_connect/engine_builder.py
@@ -1 +1 @@
+        save_fp8_scales = getattr(build_bundle, '_save_fp8_scales', None)
+            fp8_scales=fp8_scales, save_fp8_scales=save_fp8_scales)
+    save_fp8_scales: str | None = None,
+    if save_fp8_scales and isinstance(fp8_scales, dict):
+    _effective_precision = "bf16" if fp8_scales else precision
-        "precision": precision,
+        "precision": _effective_precision,
+        cfg_dict["quantization"] = {"format": "fp8"}
+    save_fp8_scales: str | None = None,
+        save_fp8_scales: Path to save calibrated FP8 scales JSON.
+    build_bundle._save_fp8_scales = save_fp8_scales
"""
        broad = test_impact.classify_file("tensorrt_model_connect/tensorrt_model_connect/engine_builder.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py", broad, diff_text, imap)
        assert refined.rule == "shared_builder_fp8_scales_engine"
        assert refined.models == ["flux-2-dev-fp8"]

    def test_shared_builder_diffusion_tokenizer_rule_refines_engine_tokenizer_diff(self, imap):
        """Diffusion tokenizer metadata plumbing should not select every model."""
        diff_text = """
diff --git a/tensorrt_model_connect/tensorrt_model_connect/engine_builder.py b/tensorrt_model_connect/tensorrt_model_connect/engine_builder.py
@@ -1 +1 @@
+def _detect_diffusion_tokenizer_add_special_tokens(model_dir: Path) -> bool:
+    for tok_subdir in ("tokenizer_2", "tokenizer"):
+        tok_dir = model_dir / tok_subdir
+            return _detect_tokenizer_add_special_tokens(tok_dir)
+    tokenizer_add_special_tokens = _detect_diffusion_tokenizer_add_special_tokens(model_dir_path)
+        "tokenizer_add_special_tokens": int(tokenizer_add_special_tokens),
"""
        broad = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "shared_builder_diffusion_tokenizer"
        assert "flux-schnell" in refined.models
        assert "qwen3-0.6b" not in refined.models

    def test_torchtrt_compiler_tokenizer_rule_refines_compiler_tokenizer_diff(self, mock_repo):
        """Torch-TRT tokenizer metadata changes narrow to Torch-TRT tokenizer users."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        _write_json(
            models_dir / "qwen2.5-0.5b-torchtrt.json",
            {
                "name": "qwen2.5-0.5b-torchtrt",
                "family": "qwen",
                "runtime_strategy": "torchtrt_decoder",
                "hf_id": "Q/Qwen2.5",
            },
        )
        _write_json(
            models_dir / "pixart-sigma-1024-torchtrt.json",
            {
                "name": "pixart-sigma-1024-torchtrt",
                "family": "pixart",
                "runtime_strategy": "diffusion_pixart_torchtrt",
                "hf_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            },
        )
        imap = test_impact.build_impact_map(mock_repo)
        diff_text = """
diff --git a/tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py b/tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py
@@ -1 +1 @@
+        from transformers import AutoTokenizer
+        ids_default = tok.encode("hello")
+        ids_without = tok.encode("hello", add_special_tokens=False)
+        return ids_default != ids_without
+            if bool(tok_cfg.get("add_eos_token", False)):
+                return True
+def _detect_diffusion_tokenizer_add_special_tokens(model_dir: Path) -> bool:
+    tokenizer_add_special_tokens = _detect_diffusion_tokenizer_add_special_tokens(model_dir_path)
"""
        broad = test_impact.classify_file(
            "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py",
            imap,
        )
        refined = test_impact.maybe_refine_match_with_diff(
            "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "torchtrt_compiler_tokenizer"
        assert set(refined.models) == {
            "pixart-sigma-1024-torchtrt",
            "qwen2.5-0.5b-torchtrt",
        }


# ---------------------------------------------------------------------------
# Aggregation / cap tests
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_multiple_families(self, imap):
        """Multiple family changes -> union of models."""
        result = test_impact.analyze_impact([
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py",
            "tensorrt_model_connect/tensorrt_model_connect/families/llama/plugin.py",
        ], imap)
        assert "qwen3-0.6b" in result.e2e_models
        assert "llama-7b" in result.e2e_models
        assert not result.cap_applied

    def test_cap_not_applied_when_under(self, imap):
        """Cap not applied when affected models <= cap."""
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap, cap=5)
        assert not result.cap_applied
        assert sorted(result.e2e_models) == ["qwen3-0.6b", "qwen3-4b"]

    def test_cap_applied_when_over(self, imap):
        """Cap applied when affected models > cap."""
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py"], imap, cap=5)
        assert result.cap_applied
        assert sorted(result.e2e_models) == sorted(imap.core_models)

    def test_no_changed_files(self, imap):
        """No files -> no impact."""
        result = test_impact.analyze_impact([], imap)
        assert result.e2e_models == []
        assert result.unit_tiers == []
        assert not result.rebuild_cpp

    def test_mixed_impact(self, imap):
        """Family plugin + unit test -> models + unit tier."""
        result = test_impact.analyze_impact([
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py",
            "tests/builder/test_config.py",
        ], imap)
        assert "qwen3-0.6b" in result.e2e_models
        assert "builder" in result.unit_tiers

    def test_l0_replaces_nightly_only_model(self, mock_repo):
        """PR L0 substitutes configured scale-only models with representatives."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        qwen4b = json.loads((models_dir / "qwen3-4b.json").read_text())
        qwen4b["ci_tier"] = "nightly_only"
        qwen4b["l0_replacement"] = "qwen3-0.6b"
        qwen4b["l0_replacement_reason"] = "scale-only coverage"
        _write_json(models_dir / "qwen3-4b.json", qwen4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap)

        assert result.e2e_models == ["qwen3-0.6b"]
        assert result.l0_replacements == [{
            "model": "qwen3-4b",
            "replacement": "qwen3-0.6b",
            "reason": "scale-only coverage",
        }]

    def test_nightly_keeps_exact_impacted_models(self, mock_repo):
        """Nightly policy does not apply PR L0 replacements."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        qwen4b = json.loads((models_dir / "qwen3-4b.json").read_text())
        qwen4b["ci_tier"] = "nightly_only"
        qwen4b["l0_replacement"] = "qwen3-0.6b"
        _write_json(models_dir / "qwen3-4b.json", qwen4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap,
            e2e_suite="nightly",
        )

        assert sorted(result.e2e_models) == ["qwen3-0.6b", "qwen3-4b"]
        assert result.l0_replacements == []

    def test_impact_excludes_multi_device_models_by_default(self, mock_repo):
        """Default impact selection matches current single-device CI capability."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        qwen4b = json.loads((models_dir / "qwen3-4b.json").read_text())
        qwen4b["ci_tier"] = "multi_device"
        _write_json(models_dir / "qwen3-4b.json", qwen4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"],
            imap,
        )

        assert result.e2e_models
        assert all(
            imap.model_metadata[model].get("ci_tier") != "multi_device"
            for model in result.e2e_models
        )

    def test_impact_can_include_multi_device_models_by_flag(self, mock_repo):
        """Manual multi-device selection opts in by clearing the default exclusion."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        qwen4b = json.loads((models_dir / "qwen3-4b.json").read_text())
        qwen4b["ci_tier"] = "multi_device"
        _write_json(models_dir / "qwen3-4b.json", qwen4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"],
            imap,
            exclude_ci_tiers=set(),
        )

        selected_ci_tiers = {
            str(imap.model_metadata[model].get("ci_tier", "") or "")
            for model in result.e2e_models
        }
        assert "" in selected_ci_tiers
        assert "multi_device" in selected_ci_tiers

    def test_manifest_change_uses_l0_replacement_for_nightly_only_model(
        self, mock_repo,
    ):
        """Direct nightly-only manifest edits still keep PR L0 at representative scale."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        qwen4b = json.loads((models_dir / "qwen3-4b.json").read_text())
        qwen4b["ci_tier"] = "nightly_only"
        qwen4b["l0_replacement"] = "qwen3-0.6b"
        qwen4b["l0_replacement_reason"] = "scale-only coverage"
        _write_json(models_dir / "qwen3-4b.json", qwen4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tests/e2e/models/qwen3-4b.json"], imap)

        assert result.e2e_models == ["qwen3-0.6b"]
        assert result.l0_replacements == [{
            "model": "qwen3-4b",
            "replacement": "qwen3-0.6b",
            "reason": "scale-only coverage",
        }]


# ---------------------------------------------------------------------------
# Validation test (uses real repo)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fallback_allowlist_accepts_reviewed_fallback_paths(
        self, imap, mock_repo, tmp_path,
    ):
        """Reviewed fallback classifications pass the fallback guardrail."""
        allowlist = tmp_path / "fallbacks.txt"
        tracked_paths = [
            "pyproject.toml",
            "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py",
            "tests/e2e_harness/contracts.py",
            "tools/diff_logits.py",
        ]
        allowlist.write_text(
            "\n".join([
                "catch_all pyproject.toml # conservative repo metadata fallback",
                "shared_builder_module "
                "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py "
                "# shared builder surface",
                "harness_shared tests/e2e_harness/contracts.py # shared harness surface",
                "no_impact tools/diff_logits.py # developer utility script",
            ]) + "\n",
            encoding="utf-8",
        )

        errors, warnings, fallbacks = test_impact.validate_fallback_allowlist(
            imap,
            mock_repo,
            tracked_paths=tracked_paths,
            allowlist_path=allowlist,
        )

        assert errors == []
        assert warnings == []
        assert {(entry["rule"], entry["path"]) for entry in fallbacks} == {
            ("catch_all", "pyproject.toml"),
            (
                "shared_builder_module",
                "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py",
            ),
            ("harness_shared", "tests/e2e_harness/contracts.py"),
            ("no_impact", "tools/diff_logits.py"),
        }

    def test_validate_rejects_unreviewed_fallback_path(self, imap, mock_repo, tmp_path):
        """A new tracked path classified by a broad fallback fails validation."""
        allowlist = tmp_path / "fallbacks.txt"
        allowlist.write_text(
            "shared_builder_module "
            "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py "
            "# existing reviewed shared builder surface\n",
            encoding="utf-8",
        )

        errors = test_impact.validate_map(
            imap,
            mock_repo,
            tracked_paths=[
                "tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py",
                "tensorrt_model_connect/tensorrt_model_connect/new_shared.py",
            ],
            fallback_allowlist_path=allowlist,
        )

        assert any(
            "Unreviewed broad fallback classification" in error
            and "new_shared.py -> shared_builder_module" in error
            for error in errors
        )
        assert not any("checkpoint_mapper.py" in error for error in errors)

    def test_validate_consistency(self):
        """Runs --validate on the real repo and checks it passes."""
        real_root = REPO_ROOT
        if not (real_root / "tests" / "e2e" / "models").is_dir():
            pytest.skip("Not in the project repo")
        imap = test_impact.build_impact_map(real_root)
        errors = test_impact.validate_map(imap, real_root)
        assert errors == [], f"Validation errors: {errors}"

    def test_real_repo_has_core_models(self):
        """Real repo has at least 5 core models."""
        real_root = REPO_ROOT
        if not (real_root / "tests" / "e2e" / "models").is_dir():
            pytest.skip("Not in the project repo")
        imap = test_impact.build_impact_map(real_root)
        assert len(imap.core_models) >= 5, (
            f"Expected at least 5 core models, got {len(imap.core_models)}"
        )


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestOutput:
    def test_human_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
        )
        output = test_impact.format_human(result)
        assert "qwen3-0.6b" in output
        assert "builder" in output
        assert "rebuild needed: no" in output

    def test_json_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b", "qwen3-4b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[{"file": "f.py", "rule": "family_plugin", "models": ["qwen3-0.6b"]}],
        )
        output = test_impact.format_json(result)
        data = json.loads(output)
        assert data["e2e_models"] == ["qwen3-0.6b", "qwen3-4b"]
        assert data["rebuild_cpp"] is False

    def test_json_cap_applied(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=sorted(imap.core_models),
            unit_tiers=[],
            rebuild_cpp=True,
            cap_applied=True,
            matched_rules=[],
        )
        output = test_impact.format_json(result)
        data = json.loads(output)
        assert data["cap_applied"] is True


# ---------------------------------------------------------------------------
# Coverage map integration tests
# ---------------------------------------------------------------------------


class TestCoverageMapIntegration:
    def test_impact_result_has_test_lists(self, imap):
        """ImpactResult with coverage map includes per-tier test lists."""
        coverage_map = {
            "tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py": [
                "tests/builder/test_engine_qwen.py::TestQwen::test_plugin",
            ],
        }
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap,
            coverage_map=coverage_map,
        )
        assert "tests/builder/test_engine_qwen.py::TestQwen::test_plugin" in result.builder_tests
        assert "builder" not in result.fallback_tiers

    def test_unknown_file_triggers_fallback(self, imap):
        """File not in coverage map triggers tier fallback."""
        coverage_map = {"tensorrt_model_connect/tensorrt_model_connect/config.py": ["tests/builder/test_config.py::test_a"]}
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap,
            coverage_map=coverage_map,
        )
        assert "builder" in result.fallback_tiers

    def test_no_coverage_map_no_test_lists(self, imap):
        """Without coverage map, test lists are empty and fallback_tiers empty."""
        result = test_impact.analyze_impact(
            ["tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py"], imap,
        )
        assert result.builder_tests == []
        assert result.cpp_tests == []
        assert result.fallback_tiers == []

    def test_changed_tools_test_selected_directly_without_coverage_map(self, imap):
        """A changed tools test file should run directly instead of all Python tests."""
        result = test_impact.analyze_impact(
            ["tests/tools/test_z_image_model_card_contract.py"], imap,
        )

        assert result.tools_tests == ["tests/tools/test_z_image_model_card_contract.py"]
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_changed_tools_test_suppresses_tools_fallback(self, imap):
        """Coverage-map fallback should not force full tools tier for the test file itself."""
        result = test_impact.analyze_impact(
            ["tests/tools/test_z_image_model_card_contract.py"], imap,
            coverage_map={},
        )

        assert result.tools_tests == ["tests/tools/test_z_image_model_card_contract.py"]
        assert "tools" not in result.fallback_tiers

    def test_changed_e2e_harness_unit_test_selected_directly(self, imap):
        """Changed e2e_harness test files run directly without broad E2E impact."""
        result = test_impact.analyze_impact(
            ["tests/e2e_harness/test_orchestrator_phases.py"], imap,
            coverage_map={},
        )

        assert result.e2e_models == []
        assert result.tools_tests == ["tests/e2e_harness/test_orchestrator_phases.py"]
        assert "tools" not in result.fallback_tiers

    def test_json_output_includes_test_lists(self, imap):
        """JSON output includes builder_tests, cpp_tests, fallback_tiers."""
        result = test_impact.ImpactResult(
            e2e_models=["qwen3-0.6b"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
            builder_tests=["tests/builder/test_config.py::test_a"],
            cpp_tests=[],
            tools_tests=[],
            fallback_tiers=[],
        )
        output = json.loads(test_impact.format_json(result))
        assert output["builder_tests"] == ["tests/builder/test_config.py::test_a"]
        assert output["cpp_tests"] == []
        assert output["fallback_tiers"] == []
