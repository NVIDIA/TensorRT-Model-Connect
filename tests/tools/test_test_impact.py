# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

_TASK_STRATEGY_BY_RUNTIME = {
    "decoder_family_decoder_kv_cache": "text_generation_causal",
    "decoder_peer_family_decoder_kv_cache": "text_generation_causal",
    "decoder_moe_family_decoder_moe": "text_generation_causal",
    "mamba_ssm_recurrent": "text_generation_causal",
    "vision_family_vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "text_to_audio_generic": "text_to_audio",
    "prompt_seg_family_prompted_segmentation": "prompted_segmentation",
    "prompt_seg_text_family_prompted_segmentation": "prompted_segmentation",
    "semantic_seg_family_segmentation": "segmentation",
    "encoder_family_encoder_only": "encoder_only_nlp",
    "context_embed_family_encoder_only": "encoder_only_nlp",
    "encoder_package_family_encoder_only": "encoder_only_nlp",
    "custom_builder_family_encoder_only": "encoder_only_nlp",
    "diffusion_media_primary": "diffusion_media_generation",
    "diffusion_media_secondary": "diffusion_media_generation",
    "registry_extension_family_decoder_kv_cache": "text_generation_causal",
    "sequence_point_runtime": "neural_operator",
    "sequence_mixer_runtime": "neural_operator",
    "sequence_global_runtime": "neural_operator",
    "sequence_quantile_runtime": "neural_operator",
    "translation_runtime": "text_generation_causal",
}


def _write_json(path: Path, data: dict) -> None:
    data = dict(data)
    runtime_strategy = data.get("runtime_strategy")
    if "task_strategy" not in data and isinstance(runtime_strategy, str):
        task_strategy = _TASK_STRATEGY_BY_RUNTIME.get(runtime_strategy)
        if task_strategy:
            data["task_strategy"] = task_strategy
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
    python_package_dir = tmp_path / "python" / "tensorrt_model_connect"
    families_dir = python_package_dir / "families"
    families_dir.mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "plugins" / "shared").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "pipelines").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "decoder_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "decoder_peer_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "decoder_moe_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "registry_extension_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "vision_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "prompt_seg_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "prompt_seg_text_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "semantic_seg_family").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "media_runtime").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "models" / "media_aux_runtime").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "core").mkdir(parents=True)
    (tmp_path / "src" / "runtime" / "domains" / "audio").mkdir(parents=True)
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

    (tmp_path / "tests" / "e2e_harness" / "manifest_loader.py").write_text(
        """
_DEFAULT_REFERENCE_BACKEND = {
    "diffusion_media_generation": "hf_diffusers",
    "neural_operator": "torch_reference",
    "text_generation_causal": "hf_transformers",
}
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "runners" / "text_generation.py").write_text(
        """
class TextGenerationRunner:
    @property
    def strategy_name(self):
        return "text_generation_causal"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "runners" / "diffusion.py").write_text(
        """
class DiffusionRunner:
    @property
    def strategy_name(self):
        return "diffusion_media_generation"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "comparators" / "diffusion.py").write_text(
        """
class DiffusionComparator:
    @property
    def task_strategy(self):
        return "diffusion_media_generation"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "comparators" / "segmentation.py").write_text(
        """
class SegmentationComparator:
    @property
    def task_strategy(self):
        return "segmentation"


class PromptedSegmentationComparator:
    @property
    def task_strategy(self):
        return "prompted_segmentation"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "plugins" / "diffusion.py").write_text(
        """
class DiffusionPlugin:
    reference_families = ["diffusers_image_gen"]
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "plugins" / "segmentation.py").write_text(
        """
class SegmentationPlugin:
    reference_families = ["semantic_segmentation"]
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "references" / "hf_diffusers.py").write_text(
        """
class HfDiffusersReference:
    @property
    def backend_name(self):
        return "hf_diffusers"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "references" / "hf_transformers.py").write_text(
        """
class HfTransformersReference:
    @property
    def backend_name(self):
        return "hf_transformers"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e_harness" / "references" / "torch_reference.py").write_text(
        """
class TorchReference:
    @property
    def backend_name(self):
        return "torch_reference"
""".lstrip(),
        encoding="utf-8",
    )
    _write_json(
        tmp_path
        / "tests"
        / "e2e_harness"
        / "thresholds"
        / "defaults"
        / "diffusion_media_generation.json",
        {},
    )

    # Manifests
    manifests = [
        {
            "name": "decoder-small",
            "family": "decoder_family",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "hf_id": "example/decoder-small",
            "core": True,
        },
        {
            "name": "decoder-large",
            "family": "decoder_family",
            "runtime_strategy": "decoder_family_decoder_kv_cache",
            "hf_id": "example/decoder-large",
        },
        {
            "name": "decoder-peer",
            "family": "decoder_peer_family",
            "runtime_strategy": "decoder_peer_family_decoder_kv_cache",
            "hf_id": "meta/decoder-peer",
        },
        {
            "name": "encoder-core",
            "family": "encoder_family",
            "runtime_strategy": "encoder_family_encoder_only",
            "hf_id": "encoder-core",
            "core": True,
        },
        {
            "name": "speech-core",
            "family": "speech_family",
            "runtime_strategy": "speech_family_speech_to_text",
            "hf_id": "example/speech-core",
            "precision": "fp16",
            "core": True,
            "test_input_audio": "tests/e2e/data/Recording.wav",
        },
        {
            "name": "canary-grouped",
            "family": "canary",
            "runtime_strategy": "speech_to_text",
            "hf_id": "example/canary-grouped",
            "core": True,
        },
        {
            "name": "nld-grouped",
            "family": "nemotron_labs_diffusion",
            "runtime_strategy": "nemotron_labs_diffusion",
            "task_strategy": "text_generation_causal",
            "hf_id": "example/nld-grouped",
        },
        {
            "name": "speech-streaming-case",
            "family": "nemotron_speech_streaming",
            "runtime_strategy": "speech_to_text",
            "hf_id": "example/speech-streaming-case",
        },
        {
            "name": "media-core",
            "family": "media_family",
            "runtime_strategy": "diffusion_media_primary",
            "hf_id": "example/media-core",
            "reference_family": "diffusers_image_gen",
            "core": True,
        },
        {
            "name": "media-alt",
            "family": "media_alt_family",
            "runtime_strategy": "diffusion_media_secondary",
            "hf_id": "example/media-alt",
        },
        {
            "name": "media-scale",
            "family": "media_family",
            "runtime_strategy": "diffusion_media_primary",
            "hf_id": "example/media-scale",
        },
        {
            "name": "media-scale-fp8",
            "family": "media_family",
            "runtime_strategy": "diffusion_media_primary",
            "hf_id": "example/media-scale",
            "fp8_scales": "media-scale-fp8-scales.json",
        },
        {
            "name": "recurrent-core",
            "family": "recurrent_family",
            "runtime_strategy": "mamba_ssm_recurrent",
            "hf_id": "example/recurrent-core",
            "core": True,
        },
        {
            "name": "vision-core",
            "family": "vision_family",
            "runtime_strategy": "vision_family_vision_language",
            "hf_id": "example/vision-core",
            "test_image": "data/test_img.jpeg",
            "core": True,
        },
        {
            "name": "audio-core",
            "family": "audio_family",
            "runtime_strategy": "text_to_audio_generic",
            "hf_id": "example/audio-core",
            "core": True,
        },
        {
            "name": "prompt-seg-core",
            "family": "prompt_seg_family",
            "runtime_strategy": "prompt_seg_family_prompted_segmentation",
            "hf_id": "example/prompt-seg-core",
            "reference_family": "prompted_segmentation_point",
            "core": True,
        },
        {
            "name": "prompt-seg-text",
            "family": "prompt_seg_text_family",
            "runtime_strategy": "prompt_seg_text_family_prompted_segmentation",
            "hf_id": "example/prompt-seg-text",
            "reference_family": "prompted_segmentation_text",
            "core": True,
        },
        {
            "name": "context-embed-model",
            "family": "context_embed_family",
            "runtime_strategy": "context_embed_family_encoder_only",
            "hf_id": "example/context-embed-model",
            "reference_family": "context_embed_reference",
        },
        {
            "name": "semantic-seg-core",
            "family": "semantic_seg_family",
            "runtime_strategy": "semantic_seg_family_segmentation",
            "hf_id": "example/semantic-seg-core",
            "reference_family": "semantic_segmentation",
            "core": True,
        },
        {
            "name": "decoder-moe-core",
            "family": "decoder_moe_family",
            "runtime_strategy": "decoder_moe_family_decoder_moe",
            "hf_id": "example/decoder-moe-core",
            "core": True,
        },
        {
            "name": "decoder-registry-case",
            "family": "registry_extension_family",
            "runtime_strategy": "registry_extension_family_decoder_kv_cache",
            "hf_id": "org/decoder-registry-case",
        },
        {
            "name": "encoder-package-core",
            "family": "encoder_package_family",
            "runtime_strategy": "encoder_package_family_encoder_only",
            "hf_id": "example/encoder-package",
        },
        {
            "name": "elf-flow-case",
            "family": "elf_flow",
            "runtime_strategy": "diffusion_text_generation",
            "hf_id": "example/elf-flow",
        },
        {
            "name": "sequence-point-core",
            "family": "sequence_point_family",
            "runtime_strategy": "sequence_point_runtime",
            "hf_id": "example/sequence-point-core",
        },
        {
            "name": "sequence-regression",
            "family": "sequence_point_family",
            "runtime_strategy": "sequence_point_runtime",
            "hf_id": "example/sequence-regression",
        },
        {
            "name": "sequence-mixer-core",
            "family": "sequence_mixer_family",
            "runtime_strategy": "sequence_mixer_runtime",
            "hf_id": "example/sequence-mixer-core",
        },
        {
            "name": "sequence-global-core",
            "family": "sequence_global_family",
            "runtime_strategy": "sequence_global_runtime",
            "hf_id": "example/sequence-global-core",
        },
        {
            "name": "sequence-quantile-core",
            "family": "sequence_quantile_family",
            "runtime_strategy": "sequence_quantile_runtime",
            "hf_id": "example/sequence-quantile-core",
        },
    ]
    for m in manifests:
        _write_json(models_dir / f"{m['name']}.json", m)

    prompt_seg_text_dir = models_dir / "prompt_seg_text_family"
    prompt_seg_text_dir.mkdir()
    (prompt_seg_text_dir / "impact_diff_rules.json").write_text(
        json.dumps(
            [
                {
                    "name": "prompt_seg_text_public_prompted_segmentation_api",
                    "path": "include/trtmc/pipeline.h",
                    "scope": {
                        "task_strategies": [
                            "segmentation",
                            "prompted_segmentation",
                            "object_detection",
                        ],
                    },
                    "allowed_tokens": [
                        "boxes",
                        "does_not_support_segment_prompted_text",
                        "image_height",
                        "image_pixels",
                        "image_width",
                        "pipeline_type",
                        "promptedsegmentationresult",
                        "segment_prompted_text",
                        "text_prompt",
                        "vector",
                    ],
                },
                {
                    "name": "prompt_seg_text_engine_builder_metadata",
                    "path": "python/tensorrt_model_connect/engine_builder.py",
                    "scope": {"owner_family": True},
                    "allowed_tokens": [
                        "getattr(plugin",
                        "preprocessor_config.json",
                        "processor_config.json",
                        "requires_tokenizer",
                        "runtime_strategy",
                    ],
                },
                {
                    "name": "prompt_seg_text_segment_prompted_cli_usage",
                    "path": "src/cli/args.cpp",
                    "scope": {
                        "task_strategies": [
                            "segmentation",
                            "prompted_segmentation",
                            "object_detection",
                        ],
                    },
                    "allowed_tokens": [
                        "background",
                        "hf_python",
                        "point_x",
                        "point_y",
                        "prompt",
                        "segment-prompted",
                    ],
                },
                {
                    "name": "prompt_seg_text_segment_prompted_cli_runtime",
                    "path": "src/cli/main.cpp",
                    "scope": {
                        "task_strategies": [
                            "segmentation",
                            "prompted_segmentation",
                            "object_detection",
                        ],
                    },
                    "allowed_tokens": [
                        ".txt",
                        "args.prompt",
                        "box",
                        "else",
                        "image.height",
                        "image.pixels",
                        "image.width",
                        "is_foreground",
                        "mask_idx",
                        "out_dir",
                        "point_x",
                        "point_y",
                        "promptedsegmentationresult",
                        "result.boxes",
                        "segment_prompted",
                        "segment_prompted_text",
                        "setfill",
                        "setprecision",
                        "static_cast",
                        "std::",
                    ],
                },
                {
                    "name": "prompt_seg_text_config",
                    "path": "src/runtime/models/prompt_seg_text_family/prompt_seg_text_config.h",
                    "scope": {"owner_family": True},
                    "allowed_tokens": [
                        "image_mean",
                        "image_size",
                        "image_std",
                        "class_map",
                        "low_res_mask_size",
                        "mask_threshold",
                        "masks",
                        "not_a_point_embed",
                        "num_mask_outputs",
                        "num_queries",
                        "point_embed_bg",
                        "point_embed_fg",
                        "promptsegtextconfig",
                        "score_threshold",
                        "shared_image_pe",
                        "text_max_position_embeddings",
                        "text_pad_token_id",
                        "text_projection_dim",
                        "vector",
                    ],
                },
                {
                    "name": "prompt_seg_text_bpe_end_of_word_suffix",
                    "path": "src/tokenizer/bpe_tokenizer.cpp",
                    "scope": {"owner_family": True},
                    "allowed_tokens": [
                        "byte_fallback",
                        "chars.back",
                        "end_of_word_suffix",
                        "get<std::string>",
                        'j["model"].value',
                        "is_string",
                        "model.find",
                        "model.value",
                        "mendofwordsuffix",
                        "optional_model_string",
                        "return_{}",
                        "string",
                    ],
                },
                {
                    "name": "prompt_seg_text_harness_contract",
                    "path": "tests/e2e_harness/contracts.py",
                    "scope": {"owner_family": True},
                    "allowed_tokens": [
                        "comparisonmode.mask_overlap",
                        "prompted_mask",
                        "prompted_segmentation_text",
                        "referencefamily.prompted_segmentation_text",
                        "prompt-seg-text",
                        "usercontract.prompted_mask",
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    context_embed_dir = models_dir / "context_embed_family"
    context_embed_dir.mkdir()
    (context_embed_dir / "impact_diff_rules.json").write_text(
        json.dumps(
            [
                {
                    "name": "context_embed_reference_backend",
                    "path": (
                        "tests/e2e/models/context_embed_family/e2e_plugins/"
                        "references/hf_transformers.py"
                    ),
                    "scope": {"models": ["context-embed-model"]},
                    "allowed_tokens": [
                        "autotokenizer.from_pretrained",
                        "automodel.from_pretrained",
                        "bert_model",
                        "context",
                        "ctx_encoder",
                        "contextencoder",
                        "contextencodertokenizerfast",
                        "questionencoder",
                        "model_ref",
                        "model_type",
                        "question_encoder",
                        "question_classes",
                        "same_token_ids",
                        "tokenizer",
                        "tokenizer.json",
                        "trust_remote_code",
                        "trt_artifact",
                        "wrong_class",
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    # Family plugins
    (families_dir / "__init__.py").write_text("")
    (families_dir / "base.py").write_text("")
    _write_family_package(
        families_dir,
        "decoder_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "decoder_peer_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "encoder_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .encoder_builder import build\nfrom ...config import C\n",
            "encoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family(
        families_dir, "speech_family", "from ..config import C\nfrom ..graph_ops import rope\n"
    )
    _write_family(families_dir, "media_family", "from ..config import C\n")
    _write_family(
        families_dir, "recurrent_family", "from ..config import C\nfrom ..graph_ops import ssm\n"
    )
    _write_family_package(
        families_dir,
        "vision_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "audio_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family(
        families_dir, "prompt_seg_family", "from ..config import C\nfrom ..graph_ops import rope\n"
    )
    _write_family(
        families_dir,
        "prompt_seg_text_family",
        "from ..config import C\nfrom ..graph_ops import rope\n",
    )
    _write_family(
        families_dir,
        "semantic_seg_family",
        "from ..config import C\nfrom ..graph_ops import conv\n",
    )
    _write_family_package(
        families_dir,
        "decoder_moe_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .standard_decoder_builder import build\nfrom ...config import C\n",
            "standard_decoder_builder.py": "def build():\n    pass\n",
        },
    )
    _write_family_package(
        families_dir,
        "encoder_package_family",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .builder import build_encoder_package_engine\n",
            "builder.py": "from ... import graph_ops\n",
        },
    )
    _write_family_package(
        families_dir,
        "elf_flow",
        {
            "__init__.py": "from .plugin import plugin\n",
            "plugin.py": "from .prepare_model_dir import prepare_model_dir\n",
            "prepare_model_dir.py": "def prepare_model_dir(args):\n    return {}\n",
        },
    )
    _write_family(families_dir, "sequence_point_family", "from ..config import C\n")
    _write_family(families_dir, "sequence_mixer_family", "from ..config import C\n")
    _write_family(families_dir, "sequence_global_family", "from ..config import C\n")
    _write_family(families_dir, "sequence_quantile_family", "from ..config import C\n")

    # Placeholder source files
    (python_package_dir / "standard_decoder_builder.py").write_text("")
    (python_package_dir / "encoder_builder.py").write_text("")
    (python_package_dir / "config.py").write_text("")
    (python_package_dir / "checkpoint_mapper.py").write_text("")
    (tmp_path / "src" / "runtime" / "models" / "decoder_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["decoder_family_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "decoder_peer_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["decoder_peer_family_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "decoder_moe_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["decoder_moe_family_decoder_moe"]\n',
        encoding="utf-8",
    )
    (
        tmp_path / "src" / "runtime" / "models" / "registry_extension_family" / "MODEL.toml"
    ).write_text(
        'runtime_strategies = ["registry_extension_family_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "vision_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["vision_family_vision_language"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "prompt_seg_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["prompt_seg_family_prompted_segmentation"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "prompt_seg_text_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["prompt_seg_text_family_prompted_segmentation"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "semantic_seg_family" / "MODEL.toml").write_text(
        'runtime_strategies = ["semantic_seg_family_segmentation"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "media_runtime" / "MODEL.toml").write_text(
        'runtime_strategies = ["diffusion_media_primary"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "media_runtime" / "pipeline.cpp").write_text(
        '#include "runtime/core/gpu_matmul.h"\n'
        '#include "runtime/models/media_runtime/diffusion_denoising_step_seam.h"\n',
        encoding="utf-8",
    )
    (
        tmp_path
        / "src"
        / "runtime"
        / "models"
        / "media_runtime"
        / "diffusion_denoising_step_seam.h"
    ).write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "media_aux_runtime" / "MODEL.toml").write_text(
        'runtime_strategies = ["diffusion_media_auxiliary"]\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "runtime" / "models" / "media_aux_runtime" / "pipeline.cpp").write_text(
        '#include "runtime/models/media_aux_runtime/diffusion_denoising_step_seam.h"\n',
        encoding="utf-8",
    )
    (
        tmp_path
        / "src"
        / "runtime"
        / "models"
        / "media_aux_runtime"
        / "diffusion_denoising_step_seam.h"
    ).write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "e2e" / "data" / "media-scale-fp8-scales.json").write_text(
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
        priorities = {rule.name: rule.priority for rule in test_impact.CLASSIFICATION_RULES}

        assert priorities["cpp_scoped_helper"] < priorities["cpp_source"]

    @pytest.mark.parametrize(
        "path,rule_name",
        [
            (
                "python/tensorrt_model_connect/families/speech_family.py",
                "family_plugin",
            ),
            (
                "python/tensorrt_model_connect/families/base.py",
                "family_base",
            ),
            ("src/runtime/models/custom_backend/plugin.cpp", "cpp_runtime_model_unknown"),
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
            (
                "python/tensorrt_model_connect/families/example_family/local_tool.py",
                "family_package",
            ),
        ],
    )
    def test_representative_rule_paths(self, imap, path, rule_name):
        """Representative paths keep their existing rule names."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == rule_name

    def test_specialized_builder_rule(self, mock_repo):
        """Root builder imports still use the dynamic family import index."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        families_dir = mock_repo / "python" / "tensorrt_model_connect" / "families"
        _write_json(
            models_dir / "custom-builder-model.json",
            {
                "name": "custom-builder-model",
                "family": "custom_builder_family",
                "runtime_strategy": "custom_builder_family_encoder_only",
                "hf_id": "custom/model",
            },
        )
        _write_family(
            families_dir,
            "custom_builder_family",
            "from ..custom_builder import build\n",
        )
        (mock_repo / "python" / "tensorrt_model_connect" / "custom_builder.py").write_text(
            "", encoding="utf-8"
        )

        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/custom_builder.py",
            imap,
        )

        assert match.rule == "specialized_builder"
        assert match.models == ["custom-builder-model"]


# ---------------------------------------------------------------------------
# Family isolation tests
# ---------------------------------------------------------------------------


class TestModelOwnedAdapterIsolation:
    @pytest.mark.parametrize(
        ("path", "expected_rule", "expected_tier", "rebuild"),
        (
            (
                "python/tensorrt_model_connect/families/decoder_family/optimized_adapter/adapter.py",
                "family_package",
                "builder",
                False,
            ),
            (
                "src/runtime/models/decoder_family/optimized_adapter/adapter.cpp",
                "cpp_runtime_model",
                "cpp",
                True,
            ),
            (
                "tests/e2e/models/decoder_family/optimized_adapter/test_contract.py",
                "e2e_model_owned_test",
                None,
                False,
            ),
        ),
    )
    def test_adapter_subtrees_use_existing_model_family_ownership(
        self,
        imap,
        path,
        expected_rule,
        expected_tier,
        rebuild,
    ):
        match = test_impact.classify_file(path, imap)

        assert match.rule == expected_rule
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert "decoder-peer" not in match.models
        assert match.rebuild_cpp is rebuild
        if expected_tier is not None:
            assert expected_tier in match.unit_tiers

class TestFamilyPlugin:
    def test_family_only_change(self, imap):
        """families/decoder_family/plugin.py -> exactly decoder_family models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/plugin.py", imap
        )
        assert match.rule == "family_package"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_family_isolation(self, imap):
        """families/decoder_family/plugin.py does NOT affect decoder_peer_family models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/plugin.py", imap
        )
        assert "decoder-peer" not in match.models

    def test_family_with_no_manifest(self, imap):
        """A family package with no manifest -> empty models, no crash."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/nonexistent_family/plugin.py", imap
        )
        assert match.rule == "family_package"
        assert match.models == []

    def test_internal_family_folder_is_not_model_owned(self, imap):
        """families/_internal files are not a model ownership boundary."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/_internal/helper.py", imap
        )
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_base_all_models(self, imap):
        """families/base.py -> ALL models."""
        match = test_impact.classify_file("python/tensorrt_model_connect/families/base.py", imap)
        assert match.rule == "family_base"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_init_all_models(self, imap):
        """families/__init__.py -> ALL models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/__init__.py", imap
        )
        assert match.rule == "family_base"
        assert len(match.models) == len(imap.all_model_names)

    def test_family_package_file(self, imap):
        """families/encoder_package_family/builder.py -> exactly encoder-package models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/encoder_package_family/builder.py", imap
        )
        assert match.rule == "family_package"
        assert match.models == ["encoder-package-core"]

    def test_family_package_plugin(self, imap):
        """families/encoder_package_family/plugin.py uses the package folder as its impact boundary."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/encoder_package_family/plugin.py", imap
        )
        assert match.rule == "family_package"
        assert match.models == ["encoder-package-core"]

    @pytest.mark.parametrize(
        "relative_path",
        [
            "native_plugins/CMakeLists.txt",
            "native_plugins/custom_plugin.cpp",
            "native_plugins/custom_plugin.cu",
            "native_plugins/custom_plugin.h",
            "assets/config.yaml",
        ],
    )
    def test_family_resource(self, imap, relative_path):
        """Every resource under a public family folder belongs only to that family."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/" + relative_path,
            imap,
        )

        assert match.rule == "family_package"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert "decoder-peer" not in match.models

    def test_family_development_tool(self, imap):
        """Family-owned development tools select only their owner models."""
        match = test_impact.classify_file("tools/families/decoder_family/debug_runner.py", imap)

        assert match.rule == "family_development_tool"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False


# ---------------------------------------------------------------------------
# Shared module tests (broad impact)
# ---------------------------------------------------------------------------


class TestSharedModules:
    def test_shared_module_all_models(self, imap):
        """checkpoint_mapper.py -> all models (no escalation)."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/checkpoint_mapper.py", imap
        )
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_shared_module_with_cap(self, imap):
        """checkpoint_mapper.py + cap -> core models only."""
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/checkpoint_mapper.py"], imap, cap=5
        )
        assert result.cap_applied
        assert sorted(result.e2e_models) == sorted(imap.core_models)

    @pytest.mark.parametrize(
        "path",
        (
            "python/tensorrt_model_connect/build_cli.py",
            "python/tensorrt_model_connect/engine_builder.py",
        ),
    )
    def test_public_build_dispatch_remains_an_all_model_boundary(self, imap, path):
        """Public dispatch changes cover delegated and native fallback builds."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)

    def test_bundle_writer_all_models(self, imap):
        """bundle_writer.py -> all models."""
        match = test_impact.classify_file("python/tensorrt_model_connect/bundle_writer.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_quantization_context_all_models(self, imap):
        """quantization/context.py -> all models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/quantization/context.py",
            imap,
        )
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_config_all_models(self, imap):
        """config.py -> all models."""
        match = test_impact.classify_file("python/tensorrt_model_connect/config.py", imap)
        assert match.rule == "shared_builder_module"
        assert len(match.models) == len(imap.all_model_names)

    def test_python_profile_requirements_scope(self, imap):
        """python profile locks affect only families that use that profile."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/sequence_quantile_family/profiles/requirements/sequence_profile.lock.txt",
            imap,
        )

        assert match.rule == "python_profile_requirements"
        assert match.models == ["sequence-quantile-core"]


# ---------------------------------------------------------------------------
# Specialized builder tests
# ---------------------------------------------------------------------------


class TestFamilyOwnedBuilder:
    def test_root_standard_decoder_builder_shim_is_broad(self, imap):
        """Root standard_decoder_builder.py is only a compatibility shim."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/standard_decoder_builder.py", imap
        )
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_local_model_implementation(self, imap):
        """families/decoder_family/model/model.py -> exactly decoder_family models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/model/model.py",
            imap,
        )
        assert match.rule == "family_package"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_family_model_index(self, imap):
        """families/decoder_family/MODEL.toml -> exactly decoder_family models."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/MODEL.toml",
            imap,
        )
        assert match.rule == "family_model_index"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_root_encoder_builder_shim_is_broad(self, imap):
        """Root encoder_builder.py is only a compatibility shim."""
        match = test_impact.classify_file("python/tensorrt_model_connect/encoder_builder.py", imap)
        assert match.rule == "shared_builder_module"
        assert sorted(match.models) == sorted(imap.all_model_names)

    def test_family_local_encoder_builder(self, imap):
        """families/encoder_family/encoder_builder.py -> exactly encoder family."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/encoder_family/encoder_builder.py", imap
        )
        assert match.rule == "family_package"
        assert set(match.models) == {"encoder-core"}


# ---------------------------------------------------------------------------
# C++ scope tests
# ---------------------------------------------------------------------------


class TestCppScope:
    def test_cpp_runtime_decoder_family_scope(self, imap):
        """decoder_family runtime model files -> only decoder_family models."""
        match = test_impact.classify_file("src/runtime/models/decoder_family/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert match.rebuild_cpp is True
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert "decoder-peer" not in match.models
        assert "decoder-moe-core" not in match.models
        assert "decoder-registry-case" not in match.models
        assert "encoder-core" not in match.models
        assert "media-core" not in match.models

    def test_cpp_runtime_vision_family_scope(self, imap):
        """vision_family runtime model files -> only vision_family models."""
        match = test_impact.classify_file("src/runtime/models/vision_family/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert set(match.models) == {"vision-core"}

    def test_cpp_model_local_plugin_helpers(self, imap):
        """Model-local plugin_helpers.h is scoped to its owning runtime model."""
        match = test_impact.classify_file("src/runtime/models/media_runtime/plugin_helpers.h", imap)
        assert match.rule == "cpp_runtime_model"
        assert "media-core" in match.models
        assert "decoder-small" not in match.models

    def test_cpp_wildcard_all(self, imap):
        """trt_common.cpp -> all models (generic C++ source)."""
        match = test_impact.classify_file("src/runtime/trt/trt_common.cpp", imap)
        assert match.rule == "cpp_source"
        assert len(match.models) == len(imap.all_model_names)

    def test_legacy_cpp_plugin_paths_are_broad_cpp_source(self, imap):
        """Legacy shared plugin/pipeline paths have no model-specific CI map."""
        for path in (
            "src/runtime/plugins/decoder_plugin.cpp",
            "src/runtime/plugins/media_runtime_plugin.cpp",
            "src/runtime/pipelines/legacy_decoder_pipeline.cpp",
            "src/runtime/pipelines/media_runtime_pipeline.cpp",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "cpp_source"
            assert len(match.models) == len(imap.all_model_names)

    def test_cpp_pipeline_scope(self, imap):
        """decoder_family pipeline.cpp -> only decoder_family models."""
        match = test_impact.classify_file("src/runtime/models/decoder_family/pipeline.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert "decoder-registry-case" not in match.models
        assert "encoder-core" not in match.models

    def test_media_family_pipeline_runtime_scope_uses_non_fp8_l0_representative(self, imap):
        """media_family pipeline.cpp is runtime-only, so media scale BF16 covers FP8 contract."""
        match = test_impact.classify_file("src/runtime/models/media_runtime/pipeline.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert "media-scale" in match.models
        assert "media-core" in match.models
        assert "media-scale-fp8" not in match.models

    def test_media_family_plugin_runtime_scope_uses_non_fp8_l0_representative(self, imap):
        """media_family plugin.cpp is runtime-only, so it does not duplicate FP8 builder coverage."""
        match = test_impact.classify_file("src/runtime/models/media_runtime/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert "media-scale" in match.models
        assert "media-core" in match.models
        assert "media-scale-fp8" not in match.models

    def test_cpp_runtime_model_scope(self, imap):
        """src/runtime/models/<strategy> files are scoped by MODEL.toml."""
        match = test_impact.classify_file("src/runtime/models/media_runtime/plugin.cpp", imap)
        assert match.rule == "cpp_runtime_model"
        assert match.rebuild_cpp is True
        assert sorted(match.models) == ["media-core", "media-scale"]

    def test_cpp_runtime_model_manifest_scope(self, imap):
        """MODEL.toml itself is model-runtime scoped."""
        match = test_impact.classify_file("src/runtime/models/media_runtime/MODEL.toml", imap)
        assert match.rule == "cpp_runtime_model"
        assert sorted(match.models) == ["media-core", "media-scale"]

    def test_scoped_cpp_helper_gpu_matmul(self, imap):
        """gpu_matmul.cpp -> only the pipelines that reference it."""
        match = test_impact.classify_file("src/runtime/core/gpu_matmul.cpp", imap)
        assert match.rule == "cpp_scoped_helper"
        assert "media-core" in match.models
        assert "media-scale" in match.models
        assert "media-scale-fp8" not in match.models
        assert "decoder-small" not in match.models

    def test_cpp_runtime_model_owned_diffusion_seam(self, imap):
        """model-owned diffusion seam helper -> owning runtime models only."""
        match = test_impact.classify_file(
            "src/runtime/models/media_runtime/diffusion_denoising_step_seam.h", imap
        )
        assert match.rule == "cpp_runtime_model"
        assert "media-core" in match.models
        assert "media-scale" in match.models
        assert "media-scale-fp8" not in match.models
        assert "decoder-small" not in match.models

    def test_third_party_stb_scopes_to_image_models(self, imap):
        """STB image headers affect image/video runtimes, not text-only models."""
        match = test_impact.classify_file("third_party/stb/stb_image.h", imap)
        assert match.rule == "third_party_stb_image"
        assert "vision-core" in match.models
        assert "media-core" in match.models
        assert "prompt-seg-core" in match.models
        assert "semantic-seg-core" in match.models
        assert "decoder-small" not in match.models
        assert match.unit_tiers == ["cpp"]
        assert match.rebuild_cpp is True


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
            "tests/e2e/models/decoder_family/manifests/decoder-small.json", imap
        )
        assert match.rule == "manifest"
        assert match.models == ["decoder-small"]

    def test_nested_manifest_uses_manifest_name_not_filename(self, mock_repo):
        """Nested manifest path lookup handles filename/name mismatches."""
        manifest_dir = mock_repo / "tests" / "e2e" / "models" / "translation_family" / "manifests"
        manifest_dir.mkdir(parents=True)
        _write_json(
            manifest_dir / "translation-case.json",
            {
                "name": "translation-case",
                "family": "translation_family",
                "runtime_strategy": "translation_runtime",
                "hf_id": "example/translation-case",
            },
        )
        imap = test_impact.build_impact_map(mock_repo)

        match = test_impact.classify_file(
            "tests/e2e/models/translation_family/manifests/translation-case.json", imap
        )

        assert match.rule == "manifest"
        assert match.models == ["translation-case"]

    def test_e2e_model_index_self(self, imap):
        """Changing a model E2E index -> only that family's models."""
        match = test_impact.classify_file("tests/e2e/models/decoder_family/MODEL.toml", imap)
        assert match.rule == "e2e_model_index"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_e2e_model_owned_test_self(self, imap):
        """Changing a model-owned E2E test -> only that family's models."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py",
            imap,
        )
        assert match.rule == "e2e_model_owned_test"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_e2e_model_owned_runner_self(self, imap):
        """Changing a model-owned E2E runner -> only that family's models."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/runner.py",
            imap,
        )
        assert match.rule == "e2e_model_owned_test"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_e2e_model_runner(self, imap):
        """Changing the uniform model runner selects every model."""
        match = test_impact.classify_file(
            "tests/e2e_harness/model_runner.py",
            imap,
        )
        assert match.rule == "harness_shared"
        assert match.models == imap.all_model_names

    def test_e2e_model_owned_waives_self(self, imap):
        """Changing a model-owned waive file -> only that family's models."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/waives.txt",
            imap,
        )
        assert match.rule == "e2e_model_owned_test"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_e2e_model_owned_impact_rules_self(self, imap):
        """Changing model-owned impact metadata -> only that family's models."""
        match = test_impact.classify_file(
            "tests/e2e/models/prompt_seg_text_family/impact_diff_rules.json",
            imap,
        )
        assert match.rule == "e2e_model_owned_test"
        assert match.models == ["prompt-seg-text"]

    def test_e2e_model_owned_root_json_self(self, imap):
        """Changing a model-owned root JSON sidecar -> only that family's models."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/perf_validation.json",
            imap,
        )
        assert match.rule == "e2e_model_owned_test"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_e2e_model_owned_threshold_self(self, imap):
        """Changing a model-owned threshold sidecar -> only that model."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/thresholds/decoder-small.json",
            imap,
        )
        assert match.rule == "e2e_model_threshold"
        assert match.models == ["decoder-small"]

    def test_e2e_model_owned_unknown_threshold_falls_back_to_family(self, imap):
        """Unknown threshold sidecars remain conservative."""
        match = test_impact.classify_file(
            "tests/e2e/models/decoder_family/thresholds/new-threshold-profile.json",
            imap,
        )
        assert match.rule == "e2e_model_threshold"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]

    def test_cmake_no_e2e_models(self, imap):
        """CMakeLists.txt -> no E2E models (build infra only) + rebuild flag."""
        match = test_impact.classify_file("CMakeLists.txt", imap)
        assert match.rule == "cmake"
        assert match.models == []
        assert match.rebuild_cpp is True

    def test_include_header(self, imap):
        """include/ header -> all models."""
        match = test_impact.classify_file("include/trtmc/runtime/pipeline_factory.h", imap)
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

    @pytest.mark.parametrize(
        "path",
        [
            "tools/ci/e2e_schedule.py",
            "tools/ci/e2e_scheduler.py",
            "scripts/schedule_e2e.py",
            "scripts/hf_cache_download_worker.py",
            "scripts/warm_hf_cache.py",
        ],
    )
    def test_e2e_runner_scripts_trigger_all_models(self, imap, path):
        """E2E runner changes must not skip E2E validation."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "e2e_runner_script"
        assert match.models == imap.all_model_names
        assert match.unit_tiers == ["tools"]

    def test_ci_orchestration_triggers_all_models(self, imap):
        """CI orchestration changes must validate every model-facing path."""
        match = test_impact.classify_file("tools/ci/pipeline.py", imap)
        assert match.rule == "ci_orchestration"
        assert match.models == imap.all_model_names
        assert match.unit_tiers == ["tools"]

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

    def test_model_plugin_evidence_report(self, imap):
        """Generated model-plugin evidence JSON is report-only."""
        match = test_impact.classify_file(
            "reports/model-plugin-encapsulation/e2e-parity-evidence-agent4.json",
            imap,
        )
        assert match.rule == "model_plugin_evidence_report"
        assert match.models == []
        assert match.unit_tiers == []

    def test_github_ci_config_triggers_tools_tier(self, imap):
        """.github workflows should validate CI tooling without selecting E2E."""
        match = test_impact.classify_file(".github/workflows/trtmc-ci.yml", imap)
        assert match.rule == "github_ci_config"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_test_impact_tool_triggers_tools_tier(self, imap):
        """Changing impact analysis should run tools-tier tests."""
        match = test_impact.classify_file("tools/test_impact.py", imap)
        assert match.rule == "test_impact_tool"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    @pytest.mark.parametrize(
        "path",
        [
            "CODEOWNERS",
            "hf_links_wave1.txt",
            "ruff.toml",
            "tests/__init__.py",
            "tests/assets/test_image.jpg",
            "tests/engine_defs/__init__.py",
            "verify_encoder.py",
        ],
    )
    def test_repo_metadata_and_test_assets_no_impact(self, imap, path):
        """Repo metadata and standalone test assets should not select E2E models."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "no_impact"
        assert match.models == []
        assert match.unit_tiers == []
        assert match.rebuild_cpp is False

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

    @pytest.mark.parametrize(
        "path",
        [
            "tests/e2e/timing_estimates.json",
            "tests/e2e_partition.py",
            "tests/runtime_strategy_matrix.yaml",
        ],
    )
    def test_e2e_schedule_metadata_tools_only(self, imap, path):
        """E2E scheduling/catalog metadata should run tools tests, not every model."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "e2e_schedule_metadata"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    @pytest.mark.parametrize(
        "path",
        [
            "tests/e2e/__init__.py",
            "tests/e2e/conftest.py",
            "tests/e2e/test_full_pipeline.py",
        ],
    )
    def test_legacy_e2e_tests_do_not_select_models(self, imap, path):
        """Legacy direct E2E tests are not the manifest-harness model selector."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "legacy_e2e_test_support"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    @pytest.mark.parametrize(
        "path",
        [
            "tests/e2e/models/decoder_family/run_decoder_fi.py",
            "tests/e2e/models/decoder_family/test_flashinfer_plugin.py",
            "tests/e2e/models/decoder_family/test_flashinfer_trt_attention.py",
            "tests/e2e/models/decoder_family/test_decoder_flashinfer.py",
            "tests/test_tvm_ffi_e2e.py",
        ],
    )
    def test_standalone_gpu_tests_do_not_select_models(self, imap, path):
        """Standalone GPU test files should not drive manifest-harness model scope."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "standalone_gpu_test_support"
        assert match.models == []
        assert match.unit_tiers == ["tools"]


class TestE2EDataFiles:
    def test_data_file_maps_to_manifest_users(self, imap):
        """Checked-in E2E data should map to manifests that reference it."""
        match = test_impact.classify_file("tests/e2e/data/media-scale-fp8-scales.json", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["media-scale-fp8"]

    def test_repo_relative_data_file_maps_to_manifest_users(self, imap):
        """Manifest tests/e2e/data references should select only their users."""
        match = test_impact.classify_file("tests/e2e/data/Recording.wav", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["speech-core"]

    def test_manifest_relative_data_file_maps_to_manifest_users(self, imap):
        """Manifest data/ references resolve relative to tests/e2e/."""
        match = test_impact.classify_file("tests/e2e/data/test_img.jpeg", imap)
        assert match.rule == "e2e_data_file"
        assert match.models == ["vision-core"]

    def test_declared_model_asset_maps_to_declaring_model(self, mock_repo):
        """Explicit model_assets entries select only the declaring model."""
        manifest = mock_repo / "tests/e2e/models/decoder_family/manifests/decoder-assets.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            manifest,
            {
                "name": "decoder-assets",
                "family": "decoder_family",
                "runtime_strategy": "decoder_family_decoder_kv_cache",
                "task_strategy": "text_generation_causal",
                "prompt_file": ("tests/e2e/models/decoder_family/assets/prompt.txt"),
                "model_assets": ["tests/e2e/models/decoder_family/assets/intrinsics.npy"],
            },
        )
        imap = test_impact.build_impact_map(mock_repo)

        for path in (
            "tests/e2e/models/decoder_family/assets/prompt.txt",
            "tests/e2e/models/decoder_family/assets/intrinsics.npy",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "e2e_data_file"
            assert match.models == ["decoder-assets"]

    @pytest.mark.parametrize(
        "path",
        [
            "tests/e2e/models/decoder_family/assets/new-prompt.txt",
            "tests/e2e/models/decoder_family/assets/new-intrinsics.npy",
        ],
    )
    def test_unlisted_family_asset_maps_to_family(self, imap, path):
        """Unlisted assets stay within their family instead of selecting all models."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == "e2e_model_owned_test"
        assert sorted(match.models) == ["decoder-large", "decoder-small"]
        assert "decoder-peer" not in match.models

    @pytest.mark.parametrize(
        "path",
        [
            "tests/e2e/models/speech_family/data/asr_probes/manifest.json",
            "tests/e2e/models/speech_family/data/asr_probes/generate_asr_probe_inputs.py",
        ],
    )
    def test_asr_probe_support_files_select_asr(self, imap, path):
        """ASR probe support files should stay scoped to their owning family."""
        match = test_impact.classify_file(path, imap)
        assert match.rule == "e2e_model_owned_test"
        assert "speech-core" in match.models
        assert "decoder-small" not in match.models


# ---------------------------------------------------------------------------
# Unit tier tests
# ---------------------------------------------------------------------------


class TestUnitTiers:
    def test_unit_tier_builder(self, imap):
        """tests/builder/ -> unit tier 'builder', no E2E."""
        match = test_impact.classify_file("tests/builder/test_config.py", imap)
        assert match.rule == "unit_builder"
        assert match.models == []
        assert "builder" in match.unit_tiers

    def test_family_unit_builder(self, imap):
        """families/<family>/tests/ -> unit tier 'builder', no E2E."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/tests/test_family.py", imap
        )
        assert match.rule == "family_unit_builder"
        assert match.models == []
        assert "builder" in match.unit_tiers

    def test_unit_tier_cpp(self, imap):
        """tests/cpp/ -> unit tier 'cpp', no E2E."""
        match = test_impact.classify_file("tests/cpp/test_bundle_format.cpp", imap)
        assert match.rule == "unit_cpp"
        assert match.models == []
        assert "cpp" in match.unit_tiers

    def test_unit_tier_tools(self, imap):
        """tests/tools/ -> unit tier 'tools', no E2E."""
        match = test_impact.classify_file("tests/tools/test_diff_logits.py", imap)
        assert match.rule == "unit_tools"
        assert match.models == []
        assert "tools" in match.unit_tiers

    def test_elf_flow_prepare_model_dir_is_family_owned(self, imap):
        """ELF model-dir preparation belongs to the elf_flow family boundary."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/elf_flow/prepare_model_dir.py",
            imap,
        )
        assert match.rule == "family_package"
        assert match.models == ["elf-flow-case"]

    @pytest.mark.parametrize(
        "path",
        [
            "tools/task_eval.py",
            "tools/elf_hf_reference.py",
            "tools/prepare_elf_task_eval_datasets.py",
            "tools/prepare_media_task_eval_datasets.py",
        ],
    )
    def test_task_eval_tool_triggers_tools_tier(self, imap, path):
        """task_eval tool edits run tools-tier tests without E2E."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == "task_eval_tool"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_task_eval_suite_config_triggers_tools_tier(self, imap):
        """task_eval suite config edits run tools-tier tests without E2E."""
        match = test_impact.classify_file("tests/task_eval/validation_suites.yaml", imap)

        assert match.rule == "task_eval_config"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_cpp_example_tool_triggers_cpp_tier(self, imap):
        """C++ example tools require C++ validation, not E2E model execution."""
        match = test_impact.classify_file("examples/trtmc_dataset_benchmark.cpp", imap)

        assert match.rule == "cpp_example_tool"
        assert match.models == []
        assert match.unit_tiers == ["cpp"]
        assert match.rebuild_cpp is True

    @pytest.mark.parametrize(
        "path",
        [
            "python/tensorrt_model_connect/benchmark/cli.py",
            "python/tensorrt_model_connect/benchmark/operations.py",
        ],
    )
    def test_benchmark_python_triggers_owned_units(self, imap, path):
        """Benchmark orchestration runs builder and tools units without model proofs."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == "benchmark_python"
        assert match.models == []
        assert match.unit_tiers == ["builder", "tools"]
        assert match.rebuild_cpp is False

    @pytest.mark.parametrize("path", ["examples/trtmc_bench.yaml", "scripts/trtmc-bench"])
    def test_benchmark_cli_assets_trigger_tools_tier(self, imap, path):
        """Benchmark CLI assets run their tools-tier contract tests."""
        match = test_impact.classify_file(path, imap)

        assert match.rule == "benchmark_cli_asset"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_e2e_selection_unit(self, imap):
        """E2E selection unit tests run tools-tier validation without E2E."""
        match = test_impact.classify_file("tests/test_e2e_selection.py", imap)
        assert match.rule == "e2e_selection_unit"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_model_plugin_validation_tools(self, imap):
        """Model-plugin validation tools run tools-tier validation without E2E."""
        for path in (
            "tools/e2e_origin_main_parity.py",
            "tools/model_plugin_isolation.py",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "model_plugin_validation_tool"
            assert match.models == []
            assert match.unit_tiers == ["tools"]
            assert match.rebuild_cpp is False

    def test_family_ownership_tools(self, imap):
        """Family migration and isolation tools run tools-tier validation."""
        for path in (
            "tools/families/__init__.py",
            "tools/family_source_isolation.py",
            "tools/family_specialization.py",
            "tools/migrate_family_layout.py",
            "tools/prune_family_helpers.py",
            "tools/relocate_family_development.py",
            "tools/specialize_family.py",
            "tools/specialize_family_switches.py",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "family_ownership_tool"
            assert match.models == []
            assert match.unit_tiers == ["tools"]
            assert match.rebuild_cpp is False

    def test_e2e_report_tools(self, imap):
        """E2E report code runs report tests without scheduling model E2E."""
        for path in (
            "scripts/generate_e2e_report.py",
            "scripts/generate_e2e_report_assets/e2e_report.css",
            "scripts/generate_e2e_report_assets/e2e_report.js",
            "scripts/reporting/__init__.py",
            "scripts/reporting/vlm_assessment.py",
        ):
            match = test_impact.classify_file(path, imap)
            assert match.rule == "e2e_report_tool"
            assert match.models == []
            assert match.unit_tiers == ["tools"]
            assert match.rebuild_cpp is False

    def test_model_ci_tool(self, imap):
        """Model selection/projection edits run tooling tests, not model E2E."""
        match = test_impact.classify_file("tools/model_ci.py", imap)

        assert match.rule == "model_ci_tool"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_nightly_artifact_selector_tool(self, imap):
        """Retry artifact selection runs its tooling contracts, not model E2E."""
        match = test_impact.classify_file("tools/select_latest_attempt_artifact.py", imap)

        assert match.rule == "nightly_artifact_selector_tool"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_nightly_issue_tracker_tool(self, imap):
        """Nightly issue tracking runs its tooling contracts, not model E2E."""
        match = test_impact.classify_file("tools/nightly_issue_tracker.py", imap)

        assert match.rule == "nightly_issue_tracker_tool"
        assert match.models == []
        assert match.unit_tiers == ["tools"]
        assert match.rebuild_cpp is False

    def test_source_implies_unit_tier(self, imap):
        """C++ source change implies 'cpp' unit tier alongside E2E."""
        match = test_impact.classify_file("src/runtime/trt/trt_common.cpp", imap)
        assert "cpp" in match.unit_tiers
        assert len(match.models) > 0

    def test_builder_source_implies_unit_tier(self, imap):
        """Python builder source change implies 'builder' unit tier."""
        match = test_impact.classify_file(
            "python/tensorrt_model_connect/families/decoder_family/plugin.py", imap
        )
        assert "builder" in match.unit_tiers


# ---------------------------------------------------------------------------
# E2E harness tests
# ---------------------------------------------------------------------------


class TestHarness:
    def test_harness_runner_placeholder(self, imap):
        """Shared runner placeholders run structural tools checks only."""
        match = test_impact.classify_file("tests/e2e_harness/runners/text_generation.py", imap)
        assert match.rule == "harness_runner_placeholder"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_harness_runner(self, mock_repo):
        """Non-placeholder shared runner routes stay task-scoped."""
        runner_path = mock_repo / "tests" / "e2e_harness" / "runners" / "custom_text.py"
        runner_path.write_text(
            """
class CustomTextRunner:
    @property
    def strategy_name(self):
        return "text_generation_causal"
""".lstrip(),
            encoding="utf-8",
        )
        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file("tests/e2e_harness/runners/custom_text.py", imap)
        assert match.rule == "harness_runner"
        assert "decoder-small" in match.models
        assert "encoder-core" not in match.models

    def test_harness_comparator_placeholder(self, imap):
        """Shared comparator placeholders run structural tools checks only."""
        match = test_impact.classify_file("tests/e2e_harness/comparators/diffusion.py", imap)
        assert match.rule == "harness_comparator_placeholder"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_harness_comparator(self, mock_repo):
        """Non-placeholder shared comparator routes stay task-scoped."""
        comparator_path = (
            mock_repo / "tests" / "e2e_harness" / "comparators" / "custom_diffusion.py"
        )
        comparator_path.write_text(
            """
class CustomDiffusionComparator:
    @property
    def task_strategy(self):
        return "diffusion_media_generation"
""".lstrip(),
            encoding="utf-8",
        )
        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file("tests/e2e_harness/comparators/custom_diffusion.py", imap)
        assert match.rule == "harness_comparator"
        assert "media-core" in match.models
        assert "decoder-small" not in match.models

    def test_harness_plugin(self, imap):
        """plugins/diffusion.py -> diffusion models."""
        match = test_impact.classify_file("tests/e2e_harness/plugins/diffusion.py", imap)
        assert match.rule == "harness_plugin"
        assert "media-core" in match.models
        assert "decoder-small" not in match.models

    def test_harness_segmentation_plugin(self, imap):
        """plugins/segmentation.py -> semantic segmentation contract users."""
        match = test_impact.classify_file("tests/e2e_harness/plugins/segmentation.py", imap)
        assert match.rule == "harness_plugin"
        assert "semantic-seg-core" in match.models
        assert "prompt-seg-text" not in match.models
        assert "prompt-seg-core" not in match.models
        assert "decoder-small" not in match.models

    def test_harness_threshold_profile(self, imap):
        """Diffusion threshold profiles should stay scoped to diffusion models."""
        match = test_impact.classify_file(
            "tests/e2e_harness/thresholds/defaults/diffusion_media_generation.json", imap
        )
        assert match.rule == "harness_threshold_profile"
        assert "media-core" in match.models
        assert "decoder-small" not in match.models

    def test_harness_shared(self, imap):
        """e2e_harness/orchestrator.py -> ALL models."""
        match = test_impact.classify_file("tests/e2e_harness/orchestrator.py", imap)
        assert match.rule == "harness_shared"
        assert len(match.models) == len(imap.all_model_names)

    def test_harness_unit_test_file(self, imap):
        """e2e_harness/test_*.py -> direct tools-tier test only."""
        match = test_impact.classify_file("tests/e2e_harness/test_orchestrator_phases.py", imap)
        assert match.rule == "harness_unit_test"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_harness_reference_placeholder(self, imap):
        """Shared reference placeholders run structural tools checks only."""
        match = test_impact.classify_file("tests/e2e_harness/references/torch_reference.py", imap)
        assert match.rule == "harness_reference_placeholder"
        assert match.models == []
        assert match.unit_tiers == ["tools"]

    def test_torch_reference_includes_neural_operator_models(self, mock_repo):
        """Non-placeholder shared references can still route by backend."""
        reference_path = (
            mock_repo / "tests" / "e2e_harness" / "references" / "custom_torch_reference.py"
        )
        reference_path.write_text(
            """
class CustomTorchReference:
    @property
    def backend_name(self):
        return "torch_reference"
""".lstrip(),
            encoding="utf-8",
        )
        models_dir = mock_repo / "tests" / "e2e" / "models"
        _write_json(
            models_dir / "neural-op-case.json",
            {
                "name": "neural-op-case",
                "family": "sequence_point_family",
                "runtime_strategy": "sequence_point_runtime",
                "hf_id": "example/neural-operator",
            },
        )
        imap = test_impact.build_impact_map(mock_repo)
        match = test_impact.classify_file(
            "tests/e2e_harness/references/custom_torch_reference.py",
            imap,
        )
        assert match.rule == "harness_reference"
        assert "neural-op-case" in match.models

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

    def test_diff_refinement_rules_are_named_in_dispatch_order(self, imap):
        """Diff refinement dispatch keeps named rules in reviewable order."""
        assert [rule.name for rule in test_impact.DIFF_REFINEMENT_RULES] == [
            "pyproject_task_eval_optional_dependencies",
            "harness_shared_fp8_scales",
            "e2e_timing_estimates_known_models",
            "runtime_strategy_matrix_known_strategies",
            "pyproject_known_profiles",
            "shared_builder_config_lookup_family_registry",
            "shared_builder_config_lookup_cli",
            "shared_builder_config_lookup_engine",
            "harness_shared_known_identifiers",
            "e2e_warm_hf_cache_diffusers_components",
            "shared_builder_fp8_scales_cli",
            "shared_builder_fp8_scales_engine",
            "shared_builder_diffusion_tokenizer",
            "harness_manifest_diffusion_thresholds",
            "harness_reference_vl_generated_only_decode",
            "harness_reference_known_identifiers",
            "e2e_waives_model_lines",
        ]
        assert all(
            callable(rule.matches) and callable(rule.refine)
            for rule in test_impact.DIFF_REFINEMENT_RULES
        )
        assert [spec.name for spec in imap.model_owned_diff_rules] == [
            "context_embed_reference_backend",
            "prompt_seg_text_public_prompted_segmentation_api",
            "prompt_seg_text_engine_builder_metadata",
            "prompt_seg_text_segment_prompted_cli_usage",
            "prompt_seg_text_segment_prompted_cli_runtime",
            "prompt_seg_text_config",
            "prompt_seg_text_bpe_end_of_word_suffix",
            "prompt_seg_text_harness_contract",
        ]
        effective_names = [
            rule.name for rule in test_impact._diff_refinement_rules_for_impact(imap)
        ]
        assert effective_names.index("prompt_seg_text_harness_contract") < effective_names.index(
            "harness_shared_known_identifiers"
        )

    def test_metadata_rules_refine_known_model_and_strategy_diffs(self, imap):
        """Registry additions for any known model/strategy avoid all-model E2E."""
        timing_diff = """
diff --git a/tests/e2e/timing_estimates.json b/tests/e2e/timing_estimates.json
@@ -1 +1 @@
+    "sequence-quantile-core": 45,
+    "decoder-small": 38,
"""
        broad_timing = test_impact.classify_file("tests/e2e/timing_estimates.json", imap)
        refined_timing = test_impact.maybe_refine_match_with_diff(
            "tests/e2e/timing_estimates.json", broad_timing, timing_diff, imap
        )
        assert refined_timing.rule == "e2e_timing_estimates_known_models"
        assert refined_timing.models == ["decoder-small", "sequence-quantile-core"]

        matrix_diff = """
diff --git a/tests/runtime_strategy_matrix.yaml b/tests/runtime_strategy_matrix.yaml
@@ -1 +1 @@
+    "decoder_moe_family_decoder_moe": {
+      "task_strategy": "text_generation_causal",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.decoder_moe_family.e2e_plugins.runners.text_generation.TextGenerationCausalRunner",
+      "comparator_class": "tests.e2e.models.decoder_moe_family.e2e_plugins.comparators.text.TextComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['decoder_moe_family_decoder_moe']."
+    },
+    "sequence_quantile_runtime": {
+      "task_strategy": "neural_operator",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.sequence_quantile_family.e2e_plugins.runners.neural_operator.NeuralOperatorRunner",
+      "comparator_class": "tests.e2e.models.sequence_quantile_family.e2e_plugins.comparators.neural_operator.NeuralOperatorComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['sequence_quantile_runtime']."
+    },
"""
        broad_matrix = test_impact.classify_file("tests/runtime_strategy_matrix.yaml", imap)
        refined_matrix = test_impact.maybe_refine_match_with_diff(
            "tests/runtime_strategy_matrix.yaml", broad_matrix, matrix_diff, imap
        )
        assert refined_matrix.rule == "runtime_strategy_matrix_known_strategies"
        assert refined_matrix.models == [
            "decoder-moe-core",
            "sequence-quantile-core",
        ]

    def test_shared_builder_rules_refine_config_lookup_to_candidate_models(self, imap):
        """Config-object plugin lookup is scoped to PR-level candidate models."""
        candidate_models = ["sequence-quantile-core"]
        init_diff = """
diff --git a/python/tensorrt_model_connect/families/__init__.py b/python/tensorrt_model_connect/families/__init__.py
@@ -1 +1 @@
-def find_plugin(model_type: str) -> "FamilyPlugin | None":
-    \"\"\"Find the first plugin that matches the given model_type.\"\"\"
+def find_plugin(model_type: object) -> "FamilyPlugin | None":
+    \"\"\"Find the first plugin that matches a model type or config object.\"\"\"
+    model_type_str = str(getattr(model_type, "model_type", model_type))
-        if p.matches(model_type):
+        matches_config = getattr(p, "matches_config", None)
+        if callable(matches_config) and matches_config(model_type):
+            return p
+        if p.matches(model_type_str):
"""
        broad_init = test_impact.classify_file(
            "python/tensorrt_model_connect/families/__init__.py", imap
        )
        refined_init = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/families/__init__.py",
            broad_init,
            init_diff,
            imap,
            candidate_models,
        )
        assert refined_init.rule == "shared_builder_config_lookup_family_registry"
        assert refined_init.models == ["sequence-quantile-core"]

        cli_diff = """
diff --git a/python/tensorrt_model_connect/build_cli.py b/python/tensorrt_model_connect/build_cli.py
@@ -1 +1 @@
-    plugin = find_plugin(config.model_type)
+    plugin = find_plugin(config)
-        raw_plugin = find_plugin(config.model_type)
+        raw_plugin = find_plugin(config)
"""
        broad_cli = test_impact.classify_file("python/tensorrt_model_connect/build_cli.py", imap)
        refined_cli = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/build_cli.py",
            broad_cli,
            cli_diff,
            imap,
            candidate_models,
        )
        assert refined_cli.rule == "shared_builder_config_lookup_cli"
        assert refined_cli.models == ["sequence-quantile-core"]

    def test_harness_rules_refine_registry_diffs_for_known_identifiers(self, imap):
        """Harness additions stay scoped to mentioned models and runtime strategies."""
        expected = [
            "sequence-global-core",
            "sequence-mixer-core",
            "sequence-point-core",
            "sequence-quantile-core",
            "sequence-regression",
        ]

        contracts_diff = """
diff --git a/tests/e2e_harness/contracts.py b/tests/e2e_harness/contracts.py
@@ -1 +1 @@
+    # 5.28 TIME_SERIES_POINT_FORECAST
+    "sequence-point-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    "sequence-mixer-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    "sequence-global-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    # 5.29 TIME_SERIES_QUANTILE_FORECAST
+    "sequence-quantile-core": ReferenceFamily.TIME_SERIES_QUANTILE_FORECAST.value,
+    # 5.30 TIME_SERIES_REGRESSION
+    "sequence-regression": ReferenceFamily.TIME_SERIES_REGRESSION.value,
+    "sequence_point_runtime": "neural_operator",
+    "sequence_mixer_runtime": "neural_operator",
+    "sequence_global_runtime": "neural_operator",
+    "sequence_quantile_runtime": "neural_operator",
"""
        broad_contracts = test_impact.classify_file("tests/e2e_harness/contracts.py", imap)
        refined_contracts = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/contracts.py", broad_contracts, contracts_diff, imap
        )
        assert refined_contracts.rule == "harness_shared_known_identifiers"
        assert refined_contracts.models == expected

        manifest_loader_diff = """
diff --git a/tests/e2e_harness/manifest_loader.py b/tests/e2e_harness/manifest_loader.py
@@ -1 +1 @@
+    if manifest.get("runtime_strategy") == "sequence_quantile_runtime":
+        reqs.append(PreflightRequirement(
+            kind="python_module_available",
+            args={"module": "sequence_profile", "phase": "build"},
+            gating=True,
+        ))
+    "sequence_point_runtime",
+    "sequence_mixer_runtime",
+    "sequence_global_runtime",
+    "sequence_quantile_runtime",
"""
        broad_loader = test_impact.classify_file("tests/e2e_harness/manifest_loader.py", imap)
        refined_loader = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/manifest_loader.py", broad_loader, manifest_loader_diff, imap
        )
        assert refined_loader.rule == "harness_shared_known_identifiers"
        assert refined_loader.models == expected

    def test_task_eval_optional_dependencies_do_not_select_e2e_models(
        self,
        imap,
        mock_repo,
        monkeypatch,
    ):
        """A task-eval-only optional extra must not trigger every model E2E."""
        diff = """
diff --git a/pyproject.toml b/pyproject.toml
index 8ab88813..8db12930 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -36,0 +37 @@ clip = ["open-clip-torch>=2.20", "Pillow>=9.0"]
+task-eval = ["rouge-score>=0.1.2", "sacrebleu>=2.4"]
"""
        monkeypatch.setattr(
            test_impact,
            "get_file_diff",
            lambda _base, _head, _repo_root, path: diff if path == "pyproject.toml" else "",
        )

        result = test_impact.analyze_impact(
            ["pyproject.toml"],
            imap,
            base="base",
            head="head",
            repo_root=mock_repo,
        )

        assert result.e2e_models == []
        assert result.e2e_test_ids == []
        assert result.unit_tiers == ["tools"]
        assert result.rebuild_cpp is False
        assert result.matched_rules == [
            {
                "file": "pyproject.toml",
                "rule": "pyproject_task_eval_optional_dependencies",
                "models": [],
            }
        ]

    def test_task_eval_extra_mixed_with_default_dependency_stays_conservative(self, imap):
        """Unrelated pyproject changes must not inherit the task-eval exemption."""
        diff = """
diff --git a/pyproject.toml b/pyproject.toml
@@ -20,0 +21 @@ dependencies = [
+    "new-runtime-dependency>=1",
@@ -36,0 +38 @@ clip = ["open-clip-torch>=2.20", "Pillow>=9.0"]
+task-eval = ["rouge-score>=0.1.2", "sacrebleu>=2.4"]
"""
        broad = test_impact.classify_file("pyproject.toml", imap)
        refined = test_impact.maybe_refine_match_with_diff("pyproject.toml", broad, diff, imap)

        assert refined.rule == "catch_all"
        assert refined.models == imap.all_model_names
        assert refined.rebuild_cpp is True

    def test_generic_shared_file_diff_selects_non_time_series_models(
        self,
        imap,
        mock_repo,
        monkeypatch,
    ):
        """Shared metadata diffs refine non-time-series models too."""
        diffs = {
            "pyproject.toml": """
diff --git a/pyproject.toml b/pyproject.toml
@@ -1 +1 @@
+decoder_family = ["decoder_family-builder>=1"]
""",
            "tests/e2e_harness/contracts.py": """
diff --git a/tests/e2e_harness/contracts.py b/tests/e2e_harness/contracts.py
@@ -1 +1 @@
+    "media-core": ReferenceFamily.DIFFUSERS_IMAGE_GEN.value,
""",
            "tests/runtime_strategy_matrix.yaml": """
diff --git a/tests/runtime_strategy_matrix.yaml b/tests/runtime_strategy_matrix.yaml
@@ -1 +1 @@
+    "decoder_moe_family_decoder_moe": {
+      "task_strategy": "text_generation_causal",
+      "cli_commands": ["run"],
+      "runner_class": "tests.e2e.models.decoder_moe_family.e2e_plugins.runners.text_generation.TextGenerationCausalRunner",
+      "comparator_class": "tests.e2e.models.decoder_moe_family.e2e_plugins.comparators.text.TextComparator",
+      "diff_framework_check_classes": []
+    },
""",
        }
        monkeypatch.setattr(
            test_impact,
            "get_file_diff",
            lambda _base, _head, _repo_root, path: diffs.get(path, ""),
        )

        result = test_impact.analyze_impact(
            sorted(diffs),
            imap,
            base="base",
            head="head",
            repo_root=mock_repo,
        )

        assert result.e2e_models == [
            "decoder-large",
            "decoder-moe-core",
            "decoder-small",
            "media-core",
        ]

    def test_time_series_pr_style_diff_selects_only_time_series(
        self,
        imap,
        mock_repo,
        monkeypatch,
    ):
        """Aggregate shared-file time-series plumbing does not trigger all-model E2E."""
        contracts_diff = """
diff --git a/tests/e2e_harness/contracts.py b/tests/e2e_harness/contracts.py
@@ -1 +1 @@
+    # 5.28 TIME_SERIES_POINT_FORECAST
+    "sequence-point-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    "sequence-mixer-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    "sequence-global-core": ReferenceFamily.TIME_SERIES_POINT_FORECAST.value,
+    # 5.29 TIME_SERIES_QUANTILE_FORECAST
+    "sequence-quantile-core": ReferenceFamily.TIME_SERIES_QUANTILE_FORECAST.value,
+    # 5.30 TIME_SERIES_REGRESSION
+    "sequence-regression": ReferenceFamily.TIME_SERIES_REGRESSION.value,
+    "sequence_point_runtime": "neural_operator",
+    "sequence_mixer_runtime": "neural_operator",
+    "sequence_global_runtime": "neural_operator",
+    "sequence_quantile_runtime": "neural_operator",
"""
        manifest_loader_diff = """
diff --git a/tests/e2e_harness/manifest_loader.py b/tests/e2e_harness/manifest_loader.py
@@ -1 +1 @@
+    if manifest.get("runtime_strategy") == "sequence_quantile_runtime":
+        reqs.append(PreflightRequirement(
+            kind="python_module_available",
+            args={"module": "sequence_profile", "phase": "build"},
+            gating=True,
+        ))
+    "sequence_point_runtime",
+    "sequence_mixer_runtime",
+    "sequence_global_runtime",
+    "sequence_quantile_runtime",
"""
        diffs = {
            "pyproject.toml": """
diff --git a/pyproject.toml b/pyproject.toml
@@ -1 +1 @@
+sequence_quantile_family = ["sequence-profile-runtime>=2.2.2"]
""",
            "python/tensorrt_model_connect/build_cli.py": """
diff --git a/python/tensorrt_model_connect/build_cli.py b/python/tensorrt_model_connect/build_cli.py
@@ -1 +1 @@
-    plugin = find_plugin(config.model_type)
+    plugin = find_plugin(config)
-        raw_plugin = find_plugin(config.model_type)
+        raw_plugin = find_plugin(config)
""",
            "python/tensorrt_model_connect/engine_builder.py": """
diff --git a/python/tensorrt_model_connect/engine_builder.py b/python/tensorrt_model_connect/engine_builder.py
@@ -1 +1 @@
-    plugin = find_plugin(config.model_type)
+    plugin = find_plugin(config)
""",
            "python/tensorrt_model_connect/families/__init__.py": """
diff --git a/python/tensorrt_model_connect/families/__init__.py b/python/tensorrt_model_connect/families/__init__.py
@@ -1 +1 @@
-def find_plugin(model_type: str) -> "FamilyPlugin | None":
+def find_plugin(model_type: object) -> "FamilyPlugin | None":
+    model_type_str = str(getattr(model_type, "model_type", model_type))
-        if p.matches(model_type):
+        matches_config = getattr(p, "matches_config", None)
+        if callable(matches_config) and matches_config(model_type):
+            return p
+        if p.matches(model_type_str):
""",
            "python/tensorrt_model_connect/families/sequence_quantile_family/MODEL.toml": """
diff --git a/python/tensorrt_model_connect/families/sequence_quantile_family/MODEL.toml b/python/tensorrt_model_connect/families/sequence_quantile_family/MODEL.toml
@@ -1 +1 @@
+python_profile_specs = [
+  "sequence_profile|families/sequence_quantile_family/profiles/requirements/sequence_profile.lock.txt|families/sequence_quantile_family/profiles/verify.py|true",
+]
+default_execution_profiles = [
+  "build|sequence_profile",
+  "reference|sequence_profile",
+]
""",
            "tests/e2e/timing_estimates.json": """
diff --git a/tests/e2e/timing_estimates.json b/tests/e2e/timing_estimates.json
@@ -1 +1 @@
+    "sequence-quantile-core": 45,
+    "sequence-mixer-core": 30,
+    "sequence-point-core": 38,
+    "sequence-regression": 32,
+    "sequence-global-core": 111,
""",
            "tests/e2e_harness/contracts.py": contracts_diff,
            "tests/e2e_harness/manifest_loader.py": manifest_loader_diff,
            "tests/e2e_harness/orchestrator.py": """
diff --git a/tests/e2e_harness/orchestrator.py b/tests/e2e_harness/orchestrator.py
@@ -1 +1 @@
+    "sequence_point_runtime",
+    "sequence_mixer_runtime",
+    "sequence_global_runtime",
+    "sequence_quantile_runtime",
""",
            "tests/e2e_harness/references/torch_reference.py": """
diff --git a/tests/e2e_harness/references/torch_reference.py b/tests/e2e_harness/references/torch_reference.py
@@ -1 +1 @@
+        if task == "neural_operator" and _is_supported_time_series_case(case):
+            return self._run_time_series_ref(case, stage, ctx)
+def _run_time_series_forward(case: E2ECase):
+    if case.family == "sequence_point_family":
+        return _run_sequence_point_family_forward(case)
+    if case.family == "sequence_mixer_family":
+        return _run_sequence_mixer_family_forward(case)
+    if case.family == "sequence_global_family":
+        return _run_sequence_global_family_forward(case)
+    if case.family == "sequence_quantile_family":
+        return _run_sequence_quantile_family_forward(case)
+def _run_sequence_point_family_forward(case: E2ECase):
+    import torch
+    import transformers
+    return torch.tensor([0.0]), "prediction_outputs"
+def _run_sequence_mixer_family_forward(case: E2ECase):
+    import torch
+    import transformers
+    return torch.tensor([0.0]), "prediction_outputs"
+def _run_sequence_global_family_forward(case: E2ECase):
+    import torch
+    import transformers
+    return torch.tensor([0.0]), "mean_predictions"
+def _run_sequence_quantile_family_forward(case: E2ECase):
+    import torch
+    import sequence_profile
+    return torch.tensor([0.0]), "quantile_preds"
""",
            "tests/runtime_strategy_matrix.yaml": """
diff --git a/tests/runtime_strategy_matrix.yaml b/tests/runtime_strategy_matrix.yaml
@@ -1 +1 @@
+    "sequence_point_runtime": {
+      "task_strategy": "neural_operator",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.sequence_point_family.e2e_plugins.runners.neural_operator.NeuralOperatorRunner",
+      "comparator_class": "tests.e2e.models.sequence_point_family.e2e_plugins.comparators.neural_operator.NeuralOperatorComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['sequence_point_runtime']."
+    },
+    "sequence_mixer_runtime": {
+      "task_strategy": "neural_operator",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.sequence_mixer_family.e2e_plugins.runners.neural_operator.NeuralOperatorRunner",
+      "comparator_class": "tests.e2e.models.sequence_mixer_family.e2e_plugins.comparators.neural_operator.NeuralOperatorComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['sequence_mixer_runtime']."
+    },
+    "sequence_global_runtime": {
+      "task_strategy": "neural_operator",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.sequence_global_family.e2e_plugins.runners.neural_operator.NeuralOperatorRunner",
+      "comparator_class": "tests.e2e.models.sequence_global_family.e2e_plugins.comparators.neural_operator.NeuralOperatorComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['sequence_global_runtime']."
+    },
+    "sequence_quantile_runtime": {
+      "task_strategy": "neural_operator",
+      "cli_commands": ["solve"],
+      "runner_class": "tests.e2e.models.sequence_quantile_family.e2e_plugins.runners.neural_operator.NeuralOperatorRunner",
+      "comparator_class": "tests.e2e.models.sequence_quantile_family.e2e_plugins.comparators.neural_operator.NeuralOperatorComparator",
+      "diff_framework_check_classes": [],
+      "diff_framework_exemption": "No diff_framework check currently registers runtime_strategies=['sequence_quantile_runtime']."
+    },
""",
        }
        for model_name in (
            "sequence-quantile-core",
            "sequence-mixer-core",
            "sequence-point-core",
            "sequence-regression",
            "sequence-global-core",
        ):
            diffs[f"tests/e2e/models/{model_name}.json"] = (
                f"diff --git a/tests/e2e/models/{model_name}.json "
                f"b/tests/e2e/models/{model_name}.json\n"
                "@@ -1 +1 @@\n"
                f'+{{"name": "{model_name}"}}\n'
            )

        monkeypatch.setattr(
            test_impact,
            "get_file_diff",
            lambda _base, _head, _repo_root, path: diffs.get(path, ""),
        )
        result = test_impact.analyze_impact(
            sorted(diffs),
            imap,
            base="base",
            head="head",
            repo_root=mock_repo,
        )

        assert result.e2e_models == [
            "sequence-global-core",
            "sequence-mixer-core",
            "sequence-point-core",
            "sequence-quantile-core",
            "sequence-regression",
        ]
        assert "decoder-small" not in result.e2e_models

    def test_prompt_seg_text_public_pipeline_rule_refines_prompted_segmentation_diff(self, imap):
        """text-prompt segmentation prompted-segmentation API additions stay scoped to segmentation."""
        diff_text = """
diff --git a/include/trtmc/pipeline.h b/include/trtmc/pipeline.h
@@ -1 +1 @@
+    std::vector<float> boxes;      // [num_masks, 4], xyxy absolute pixel coordinates
+    virtual PromptedSegmentationResult segment_prompted_text(const float* image_pixels,
+                                                             int32_t image_height,
+                                                             int32_t image_width,
+                                                             const std::string& text_prompt) {
+        (void)image_pixels;
+        (void)text_prompt;
+        throw std::runtime_error(std::string(pipeline_type()) +
+                                 " does not support segment_prompted_text()");
"""
        broad = test_impact.classify_file("include/trtmc/pipeline.h", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "include/trtmc/pipeline.h", broad, diff_text, imap
        )
        assert refined.rule == "prompt_seg_text_public_prompted_segmentation_api"
        assert "prompt-seg-text" in refined.models
        assert "prompt-seg-core" in refined.models
        assert "decoder-small" not in refined.models

    def test_prompt_seg_text_engine_builder_rule_refines_metadata_diff(self, imap):
        """text-prompt segmentation bundle metadata plumbing does not select every builder model."""
        diff_text = """
diff --git a/python/tensorrt_model_connect/engine_builder.py b/python/tensorrt_model_connect/engine_builder.py
@@ -1 +1 @@
+    "processor_config.json",
-    if runtime_strategy not in (
+    requires_tokenizer = bool(getattr(plugin, "requires_tokenizer", False))
+    if requires_tokenizer or runtime_strategy not in (
-                     "preprocessor_config.json"):
+                     "preprocessor_config.json", "processor_config.json"):
"""
        broad = test_impact.classify_file("python/tensorrt_model_connect/engine_builder.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/engine_builder.py", broad, diff_text, imap
        )
        assert refined.rule == "prompt_seg_text_engine_builder_metadata"
        assert refined.models == ["prompt-seg-text"]

    def test_prompt_seg_text_cli_rules_refine_segment_prompted_prompt_diff(self, imap):
        """text-prompt segmentation text-prompt CLI plumbing is segmentation-scoped."""
        args_diff = """
diff --git a/src/cli/args.cpp b/src/cli/args.cpp
@@ -1 +1 @@
-           "[--point-x F] [--point-y F] [--background] [--hf-python PATH]\\n"
+           "[--point-x F] [--point-y F] [--background] [--prompt TEXT] [--hf-python PATH]\\n"
"""
        broad_args = test_impact.classify_file("src/cli/args.cpp", imap)
        refined_args = test_impact.maybe_refine_match_with_diff(
            "src/cli/args.cpp", broad_args, args_diff, imap
        )
        assert refined_args.rule == "prompt_seg_text_segment_prompted_cli_usage"

        main_diff = """
diff --git a/src/cli/main.cpp b/src/cli/main.cpp
@@ -1 +1 @@
+    trtmc::PromptedSegmentationResult result;
+    if (!args.prompt.empty()) {
+        result = pipeline->segment_prompted_text(image.pixels.data(), image.height, image.width,
+                                                 args.prompt);
+    } else {
+        result = pipeline->segment_prompted(image.pixels.data(), image.height, image.width,
+                                            args.point_x, args.point_y, args.is_foreground);
+    }
+        const auto box_offset = static_cast<std::size_t>(mask_idx) * 4U;
+        if (result.boxes.size() >= box_offset + 4U) {
+            std::ostringstream box_path;
+            box_path << out_dir << "/box_" << std::setw(3) << std::setfill('0') << mask_idx
+                     << ".txt";
+            std::ofstream box_out(box_path.str());
+            box_out << std::fixed << std::setprecision(6) << result.boxes[box_offset] << ' '
+                    << result.boxes[box_offset + 1U] << '\\n';
+        }
"""
        broad_main = test_impact.classify_file("src/cli/main.cpp", imap)
        refined_main = test_impact.maybe_refine_match_with_diff(
            "src/cli/main.cpp", broad_main, main_diff, imap
        )
        assert refined_main.rule == "prompt_seg_text_segment_prompted_cli_runtime"
        assert "prompt-seg-text" in refined_main.models
        assert "semantic-seg-core" in refined_main.models
        assert "decoder-small" not in refined_main.models

    def test_prompt_seg_text_runtime_support_rules_refine_shared_cpp_diffs(self, imap):
        """text-prompt segmentation config and BPE suffix support avoid all-model fallback."""
        config_diff = """
diff --git a/src/runtime/models/prompt_seg_text_family/prompt_seg_text_config.h b/src/runtime/models/prompt_seg_text_family/prompt_seg_text_config.h
@@ -10 +10 @@ struct PromptSegConfig {
-    int32_t image_embedding_size{64};  // image_size / patch_size
+    int32_t image_embedding_size{64}; // image_size / patch_size
@@ -12 +12 @@ struct PromptSegConfig {
-    int32_t num_mask_outputs{4};       // num_multimask + 1
+    int32_t num_mask_outputs{4}; // num_multimask + 1
@@ -18,4 +18,4 @@ struct PromptSegConfig {
-    std::vector<float> point_embed_fg;           // foreground point [decoder_hidden_size]
-    std::vector<float> point_embed_bg;           // background point [decoder_hidden_size]
-    std::vector<float> not_a_point_embed;        // padding point [decoder_hidden_size]
-    std::vector<float> shared_image_pe;          // [2, num_pos_feats] flattened
+    std::vector<float> point_embed_fg;    // foreground point [decoder_hidden_size]
+    std::vector<float> point_embed_bg;    // background point [decoder_hidden_size]
+    std::vector<float> not_a_point_embed; // padding point [decoder_hidden_size]
+    std::vector<float> shared_image_pe;   // [2, num_pos_feats] flattened
@@ -25 +25 @@ struct PromptSegResult {
-    std::vector<float> masks;     // [num_masks, 256, 256]
+    std::vector<float> masks;      // [num_masks, 256, 256]
@@ -31,0 +32,13 @@ struct PromptSegResult {
+struct PromptSegTextConfig {
+    int32_t text_max_position_embeddings{32};
+    int32_t text_pad_token_id{1};
+    int32_t text_projection_dim{512};
+    int32_t image_size{1008};
+    int32_t low_res_mask_size{288};
+    int32_t num_queries{200};
+    float score_threshold{0.5F};
+    float mask_threshold{0.5F};
+    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
+    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
+};
+
@@ -43 +56 @@ struct SegmentationResult {
-    std::vector<int32_t> class_map;  // [H, W] class indices
+    std::vector<int32_t> class_map; // [H, W] class indices
"""
        broad_config = test_impact.classify_file(
            "src/runtime/models/prompt_seg_text_family/prompt_seg_text_config.h", imap
        )
        refined_config = test_impact.maybe_refine_match_with_diff(
            "src/runtime/models/prompt_seg_text_family/prompt_seg_text_config.h",
            broad_config,
            config_diff,
            imap,
        )
        assert refined_config.rule == "prompt_seg_text_config"
        assert refined_config.models == ["prompt-seg-text"]
        assert "decoder-small" not in refined_config.models

        tokenizer_diff = """
diff --git a/src/tokenizer/bpe_tokenizer.cpp b/src/tokenizer/bpe_tokenizer.cpp
@@ -1 +1 @@
+            if (!chars.empty() && !mEndOfWordSuffix.empty()) {
+                chars.back() += mEndOfWordSuffix;
+            }
+    static std::string optional_model_string(const nlohmann::json& model, const char* key) {
+        auto it = model.find(key);
+        if (it != model.end() && it->is_string())
+            return it->get<std::string>();
+        return {};
+    }
-        mByteFallback = j["model"].value("byte_fallback", false);
+        mByteFallback = model.value("byte_fallback", false);
+        mEndOfWordSuffix = optional_model_string(model, "end_of_word_suffix");
+    std::string mEndOfWordSuffix;
"""
        broad_tokenizer = test_impact.classify_file("src/tokenizer/bpe_tokenizer.cpp", imap)
        refined_tokenizer = test_impact.maybe_refine_match_with_diff(
            "src/tokenizer/bpe_tokenizer.cpp", broad_tokenizer, tokenizer_diff, imap
        )
        assert refined_tokenizer.rule == "prompt_seg_text_bpe_end_of_word_suffix"
        assert refined_tokenizer.models == ["prompt-seg-text"]

    def test_prompt_seg_text_harness_contract_rule_refines_contract_diff(self, imap):
        """text-prompt segmentation shared contract enum additions stay scoped to the text-prompt segmentation model."""
        contracts_diff = """
diff --git a/tests/e2e_harness/contracts.py b/tests/e2e_harness/contracts.py
@@ -1 +1 @@
+    PROMPTED_SEGMENTATION_TEXT = "prompted_segmentation_text"
+    "prompt-seg-text": ReferenceFamily.PROMPTED_SEGMENTATION_TEXT.value,
+    ReferenceFamily.PROMPTED_SEGMENTATION_TEXT.value: UserContract.PROMPTED_MASK.value,
+    ReferenceFamily.PROMPTED_SEGMENTATION_TEXT.value: ComparisonMode.MASK_OVERLAP.value,
"""
        broad_contracts = test_impact.classify_file("tests/e2e_harness/contracts.py", imap)
        refined_contracts = test_impact.maybe_refine_match_with_diff(
            "tests/e2e_harness/contracts.py", broad_contracts, contracts_diff, imap
        )
        assert refined_contracts.rule == "prompt_seg_text_harness_contract"
        assert refined_contracts.models == ["prompt-seg-text"]

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
            "tests/e2e_harness/orchestrator.py", broad, diff_text, imap
        )
        assert refined.rule == "harness_shared_fp8_scales"
        assert refined.models == ["media-scale-fp8"]

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
            "scripts/warm_hf_cache.py", broad, diff_text, imap
        )
        assert refined.rule == "e2e_warm_hf_cache_diffusers_components"
        assert refined.models == ["media-scale-fp8"]

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
            "tests/e2e_harness/manifest_loader.py", broad, diff_text, imap
        )
        assert refined.rule == "harness_manifest_diffusion_thresholds"
        assert "media-core" in refined.models
        assert "decoder-small" not in refined.models

    def test_harness_reference_vl_generated_only_decode_rule_refines_hf_vl_diff(self, imap):
        """InternVL-owned generated-only decode fallback is scoped to InternVL3-8B."""
        diff_text = """
diff --git a/tests/e2e/models/internvl/e2e_plugins/references/hf_transformers.py b/tests/e2e/models/internvl/e2e_plugins/references/hf_transformers.py
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
+            from tests.e2e.models.internvl.e2e_plugins.references.hf_transformers import (
+                _decode_vl_generated_text,
+            )
+            text = _decode_vl_generated_text(
+                processor, generated_ids[0], input_len, prompt_texts)
"""
        broad = test_impact.classify_file(
            "tests/e2e/models/internvl/e2e_plugins/references/hf_transformers.py",
            imap,
        )
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e/models/internvl/e2e_plugins/references/hf_transformers.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "harness_reference_vl_generated_only_decode"
        assert refined.models == ["internvl3-8b"]

    def test_model_owned_reference_rule_refines_hf_context_diff(self, imap):
        """A model-owned reference rule should not select every HF model."""
        diff_text = """
diff --git a/tests/e2e/models/context_embed_family/e2e_plugins/references/hf_transformers.py b/tests/e2e/models/context_embed_family/e2e_plugins/references/hf_transformers.py
@@ -1 +1 @@
-            tokenizer = AutoTokenizer.from_pretrained(
-                model_ref, trust_remote_code=trust_remote_code)
+                # AutoTokenizer/AutoModel route this context checkpoint through
+                # the question encoder classes in transformers. Use the
+                # context fast tokenizer so HF sees the same token ids as the
+                # tokenizer.json bundled into the TRT artifact.
+                from transformers import ContextEncoder, ContextEncoderTokenizerFast
+                tokenizer = ContextEncoderTokenizerFast.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
+                model = _context.ctx_encoder.bert_model
+                tokenizer = AutoTokenizer.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
"""
        broad = test_impact.classify_file(
            "tests/e2e/models/context_embed_family/e2e_plugins/references/hf_transformers.py",
            imap,
        )
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e/models/context_embed_family/e2e_plugins/references/hf_transformers.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "context_embed_reference_backend"
        assert refined.models == ["context-embed-model"]

    def test_e2e_waives_model_lines_rule_refines_named_model_diff(self, imap):
        """A waiver change for one known model should only re-run that model."""
        diff_text = """
diff --git a/tests/e2e/waives.txt b/tests/e2e/waives.txt
@@ -1 +1 @@
-media-core XFAIL (old waiver)
"""
        broad = test_impact.classify_file("tests/e2e/waives.txt", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "tests/e2e/waives.txt", broad, diff_text, imap
        )
        assert refined.rule == "e2e_waives_model_lines"
        assert refined.models == ["media-core"]


class TestDiffAwareBuilderRefinement:
    def test_shared_builder_fp8_scales_cli_rule_refines_cli_fp8_diff(self, imap):
        """CLI fp8-only plumbing narrows to fp8-scales manifests."""
        diff_text = """
diff --git a/python/tensorrt_model_connect/build_cli.py b/python/tensorrt_model_connect/build_cli.py
@@ -1 +1 @@
+    save_fp8_scales = getattr(args, 'save_fp8_scales', None)
+            save_fp8_scales=save_fp8_scales,
+    build_p.add_argument("--save-fp8-scales", default=None,
"""
        broad = test_impact.classify_file("python/tensorrt_model_connect/build_cli.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/build_cli.py", broad, diff_text, imap
        )
        assert refined.rule == "shared_builder_fp8_scales_cli"
        assert refined.models == ["media-scale-fp8"]

    def test_shared_builder_fp8_scales_engine_rule_refines_engine_fp8_diff(self, imap):
        """Diffusion fp8-only engine_builder changes narrow to fp8-scales manifests."""
        diff_text = """
diff --git a/python/tensorrt_model_connect/engine_builder.py b/python/tensorrt_model_connect/engine_builder.py
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
        broad = test_impact.classify_file("python/tensorrt_model_connect/engine_builder.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/engine_builder.py", broad, diff_text, imap
        )
        assert refined.rule == "shared_builder_fp8_scales_engine"
        assert refined.models == ["media-scale-fp8"]

    def test_shared_builder_diffusion_tokenizer_rule_refines_engine_tokenizer_diff(self, imap):
        """Diffusion tokenizer metadata plumbing should not select every model."""
        diff_text = """
diff --git a/python/tensorrt_model_connect/engine_builder.py b/python/tensorrt_model_connect/engine_builder.py
@@ -1 +1 @@
+def _diffusion_tokenizer_add_special_tokens_from_plugin(plugin, model_dir_path: Path) -> bool:
+    detector = getattr(plugin, "diffusion_tokenizer_add_special_tokens", None)
+    kwargs["detect_tokenizer_add_special_tokens"] = _detect_tokenizer_add_special_tokens
+    tokenizer_add_special_tokens = _diffusion_tokenizer_add_special_tokens_from_plugin(plugin, model_dir_path)
+        "tokenizer_add_special_tokens": int(tokenizer_add_special_tokens),
"""
        broad = test_impact.classify_file("python/tensorrt_model_connect/engine_builder.py", imap)
        refined = test_impact.maybe_refine_match_with_diff(
            "python/tensorrt_model_connect/engine_builder.py",
            broad,
            diff_text,
            imap,
        )
        assert refined.rule == "shared_builder_diffusion_tokenizer"
        assert "media-core" in refined.models
        assert "decoder-small" not in refined.models


# ---------------------------------------------------------------------------
# Aggregation / cap tests
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_multiple_families(self, imap):
        """Multiple family changes -> union of models."""
        result = test_impact.analyze_impact(
            [
                "python/tensorrt_model_connect/families/decoder_family/plugin.py",
                "python/tensorrt_model_connect/families/decoder_peer_family/plugin.py",
            ],
            imap,
        )
        assert "decoder-small" in result.e2e_models
        assert "decoder-peer" in result.e2e_models
        assert not result.cap_applied

    def test_cap_not_applied_when_under(self, imap):
        """Cap not applied when affected models <= cap."""
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"], imap, cap=5
        )
        assert not result.cap_applied
        assert sorted(result.e2e_models) == ["decoder-large", "decoder-small"]

    def test_cap_applied_when_over(self, imap):
        """Cap applied when affected models > cap."""
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/checkpoint_mapper.py"], imap, cap=5
        )
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
        result = test_impact.analyze_impact(
            [
                "python/tensorrt_model_connect/families/decoder_family/plugin.py",
                "tests/builder/test_config.py",
            ],
            imap,
        )
        assert "decoder-small" in result.e2e_models
        assert "builder" in result.unit_tiers

    def test_l0_replaces_nightly_only_model(self, mock_repo):
        """PR L0 substitutes configured scale-only models with representatives."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        decoder_family4b = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_family4b["ci_tier"] = "nightly_only"
        decoder_family4b["l0_replacement"] = "decoder-small"
        decoder_family4b["l0_replacement_reason"] = "scale-only coverage"
        _write_json(models_dir / "decoder-large.json", decoder_family4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"], imap
        )

        assert result.e2e_models == ["decoder-small"]
        assert result.l0_replacements == [
            {
                "model": "decoder-large",
                "replacement": "decoder-small",
                "reason": "scale-only coverage",
            }
        ]

    def test_waive_line_keeps_exact_model_despite_l0_replacement(self, mock_repo):
        """Waiver edits name exact configs, so L0 should not substitute them."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        decoder_large = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_large["l0_replacement"] = "decoder-small"
        decoder_large["l0_replacement_reason"] = "scale-only coverage"
        _write_json(models_dir / "decoder-large.json", decoder_large)

        imap = test_impact.build_impact_map(mock_repo)
        selected, replacements = test_impact._apply_l0_replacements(
            ["decoder-large"], imap, {"decoder-large"}
        )

        assert selected == ["decoder-large"]
        assert replacements == []

    def test_nightly_keeps_exact_impacted_models(self, mock_repo):
        """Nightly policy does not apply PR L0 replacements."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        decoder_family4b = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_family4b["ci_tier"] = "nightly_only"
        decoder_family4b["l0_replacement"] = "decoder-small"
        _write_json(models_dir / "decoder-large.json", decoder_family4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
            imap,
            e2e_suite="nightly",
        )

        assert sorted(result.e2e_models) == ["decoder-large", "decoder-small"]
        assert result.l0_replacements == []

    def test_impact_excludes_multi_device_models_by_default(self, mock_repo):
        """Default impact selection matches current single-device CI capability."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        decoder_family4b = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_family4b["ci_tier"] = "multi_device"
        _write_json(models_dir / "decoder-large.json", decoder_family4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
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
        decoder_family4b = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_family4b["ci_tier"] = "multi_device"
        _write_json(models_dir / "decoder-large.json", decoder_family4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
            imap,
            exclude_ci_tiers=set(),
        )

        selected_ci_tiers = {
            str(imap.model_metadata[model].get("ci_tier", "") or "") for model in result.e2e_models
        }
        assert "" in selected_ci_tiers
        assert "multi_device" in selected_ci_tiers

    def test_manifest_change_uses_l0_replacement_for_nightly_only_model(
        self,
        mock_repo,
    ):
        """Direct nightly-only manifest edits still keep PR L0 at representative scale."""
        models_dir = mock_repo / "tests" / "e2e" / "models"
        decoder_family4b = json.loads((models_dir / "decoder-large.json").read_text())
        decoder_family4b["ci_tier"] = "nightly_only"
        decoder_family4b["l0_replacement"] = "decoder-small"
        decoder_family4b["l0_replacement_reason"] = "scale-only coverage"
        _write_json(models_dir / "decoder-large.json", decoder_family4b)

        imap = test_impact.build_impact_map(mock_repo)
        result = test_impact.analyze_impact(
            ["tests/e2e/models/decoder_family/manifests/decoder-large.json"], imap
        )

        assert result.e2e_models == ["decoder-small"]
        assert result.l0_replacements == [
            {
                "model": "decoder-large",
                "replacement": "decoder-small",
                "reason": "scale-only coverage",
            }
        ]


# ---------------------------------------------------------------------------
# Validation test (uses real repo)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fallback_allowlist_accepts_reviewed_fallback_paths(
        self,
        imap,
        mock_repo,
        tmp_path,
    ):
        """Reviewed fallback classifications pass the fallback guardrail."""
        allowlist = tmp_path / "fallbacks.txt"
        tracked_paths = [
            "pyproject.toml",
            "python/tensorrt_model_connect/checkpoint_mapper.py",
            "tests/e2e_harness/contracts.py",
            "tools/diff_logits.py",
        ]
        allowlist.write_text(
            "\n".join(
                [
                    "catch_all pyproject.toml # conservative repo metadata fallback",
                    "shared_builder_module "
                    "python/tensorrt_model_connect/checkpoint_mapper.py "
                    "# shared builder surface",
                    "harness_shared tests/e2e_harness/contracts.py # shared harness surface",
                    "no_impact tools/diff_logits.py # developer utility script",
                ]
            )
            + "\n",
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
                "python/tensorrt_model_connect/checkpoint_mapper.py",
            ),
            ("harness_shared", "tests/e2e_harness/contracts.py"),
            ("no_impact", "tools/diff_logits.py"),
        }

    def test_validate_rejects_unreviewed_fallback_path(self, imap, mock_repo, tmp_path):
        """A new tracked path classified by a broad fallback fails validation."""
        allowlist = tmp_path / "fallbacks.txt"
        allowlist.write_text(
            "shared_builder_module "
            "python/tensorrt_model_connect/checkpoint_mapper.py "
            "# existing reviewed shared builder surface\n",
            encoding="utf-8",
        )

        errors = test_impact.validate_map(
            imap,
            mock_repo,
            tracked_paths=[
                "python/tensorrt_model_connect/checkpoint_mapper.py",
                "python/tensorrt_model_connect/new_shared.py",
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

    def test_flux_runtime_selects_batch2_detector(self):
        """FLUX runtime changes must execute the real-bundle batch contract."""
        real_root = REPO_ROOT
        if not (real_root / "tests" / "e2e" / "models").is_dir():
            pytest.skip("Not in the project repo")
        imap = test_impact.build_impact_map(real_root)

        result = test_impact.analyze_impact(
            ["src/runtime/models/flux/pipeline.cpp"], imap, e2e_suite="l0"
        )

        assert "flux-schnell-l0-batch2" in result.e2e_models
        assert (
            "tests/e2e/models/flux/test_flux_e2e.py::test_model_e2e[flux-schnell-l0-batch2]"
        ) in result.e2e_test_ids


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestOutput:
    def test_human_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["decoder-small"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
            e2e_test_ids=[
                "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]",
            ],
        )
        output = test_impact.format_human(result)
        assert (
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]"
            in output
        )
        assert "builder" in output
        assert "rebuild needed: no" in output

    def test_json_format(self, imap):
        result = test_impact.ImpactResult(
            e2e_models=["decoder-small", "decoder-large"],
            unit_tiers=["builder"],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[{"file": "f.py", "rule": "family_plugin", "models": ["decoder-small"]}],
        )
        output = test_impact.format_json(result)
        data = json.loads(output)
        assert data["e2e_models"] == ["decoder-small", "decoder-large"]
        assert data["e2e_test_ids"] == []
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
            "python/tensorrt_model_connect/families/decoder_family/plugin.py": [
                "tests/builder/test_engine_decoder_family.py::TestDecoderFamily::test_plugin",
            ],
        }
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
            imap,
            coverage_map=coverage_map,
        )
        assert (
            "tests/builder/test_engine_decoder_family.py::TestDecoderFamily::test_plugin"
            in result.builder_tests
        )
        assert "builder" not in result.fallback_tiers

    def test_shared_python_file_missing_coverage_triggers_fallback(self, imap):
        """Shared files missing from the coverage map keep full-tier fallback."""
        coverage_map = {
            "python/tensorrt_model_connect/config.py": ["tests/builder/test_config.py::test_a"]
        }
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/engine_builder.py"],
            imap,
            coverage_map=coverage_map,
        )
        assert "builder" in result.fallback_tiers

    def test_model_owned_python_missing_coverage_uses_family_tests(
        self,
        mock_repo,
    ):
        """Model-owned coverage misses do not fan out to the full builder tier."""
        family_dir = mock_repo / "tests" / "e2e" / "models" / "decoder_family"
        family_dir.mkdir()
        (family_dir / "test_decoder_family_builder.py").write_text(
            "def test_builder():\n    pass\n",
            encoding="utf-8",
        )
        (family_dir / "test_decoder_family_e2e.py").write_text(
            "def test_model_e2e():\n    pass\n",
            encoding="utf-8",
        )
        imap = test_impact.build_impact_map(mock_repo)

        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
            imap,
            coverage_map={},
            repo_root=mock_repo,
        )

        assert result.e2e_models == ["decoder-large", "decoder-small"]
        assert result.e2e_test_ids == [
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-large]",
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]",
        ]
        assert result.builder_tests == [
            "tests/e2e/models/decoder_family/test_decoder_family_builder.py",
        ]
        assert "builder" not in result.fallback_tiers

    def test_changed_model_owned_unit_test_runs_directly(self, mock_repo):
        """Changed model-owned non-E2E pytest files are direct builder targets."""
        family_dir = mock_repo / "tests" / "e2e" / "models" / "decoder_family"
        family_dir.mkdir()
        (family_dir / "test_decoder_family_builder.py").write_text(
            "def test_builder():\n    pass\n",
            encoding="utf-8",
        )
        imap = test_impact.build_impact_map(mock_repo)

        result = test_impact.analyze_impact(
            ["tests/e2e/models/decoder_family/test_decoder_family_builder.py"],
            imap,
            coverage_map={},
            repo_root=mock_repo,
        )

        assert result.builder_tests == [
            "tests/e2e/models/decoder_family/test_decoder_family_builder.py",
        ]
        assert "builder" not in result.fallback_tiers

    def test_changed_nested_model_owned_unit_test_runs_directly(self, mock_repo):
        """Nested adapter tests remain direct targets inside one family."""
        adapter_tests = (
            mock_repo / "tests" / "e2e" / "models" / "decoder_family" / "optimized_adapter"
        )
        adapter_tests.mkdir(parents=True)
        test_path = adapter_tests / "test_contract.py"
        test_path.write_text("def test_contract():\n    assert True\n", encoding="utf-8")
        imap = test_impact.build_impact_map(mock_repo)
        relative = test_path.relative_to(mock_repo).as_posix()

        result = test_impact.analyze_impact(
            [relative],
            imap,
            coverage_map={},
            repo_root=mock_repo,
        )

        assert result.builder_tests == [relative]
        assert sorted(result.e2e_models) == ["decoder-large", "decoder-small"]
        assert "builder" not in result.fallback_tiers

    def test_changed_nested_e2e_test_is_not_selected_as_a_unit_test(self, mock_repo):
        """The _e2e.py suffix remains an execution boundary at every depth."""
        test_path = (
            mock_repo
            / "tests"
            / "e2e"
            / "models"
            / "decoder_family"
            / "optimized_adapter"
            / "test_adapter_e2e.py"
        )
        test_path.parent.mkdir(parents=True)
        test_path.write_text("def test_adapter_e2e():\n    assert True\n", encoding="utf-8")
        imap = test_impact.build_impact_map(mock_repo)
        relative = test_path.relative_to(mock_repo).as_posix()

        result = test_impact.analyze_impact([relative], imap, repo_root=mock_repo)

        assert result.builder_tests == []
        assert sorted(result.e2e_models) == ["decoder-large", "decoder-small"]

    def test_no_coverage_map_no_test_lists(self, imap):
        """Without coverage map, test lists are empty and fallback_tiers empty."""
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/decoder_family/plugin.py"],
            imap,
        )
        assert result.builder_tests == []
        assert result.cpp_tests == []
        assert result.fallback_tiers == []

    def test_changed_tools_test_selected_directly_without_coverage_map(self, imap):
        """A changed tools test file should run directly instead of all Python tests."""
        result = test_impact.analyze_impact(
            ["tests/tools/test_media_alt_model_card_contract.py"],
            imap,
        )

        assert result.tools_tests == ["tests/tools/test_media_alt_model_card_contract.py"]
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_changed_tools_test_suppresses_tools_fallback(self, imap):
        """Coverage-map fallback should not force full tools tier for the test file itself."""
        result = test_impact.analyze_impact(
            ["tests/tools/test_media_alt_model_card_contract.py"],
            imap,
            coverage_map={},
        )

        assert result.tools_tests == ["tests/tools/test_media_alt_model_card_contract.py"]
        assert "tools" not in result.fallback_tiers

    def test_direct_builder_test_does_not_suppress_shared_source_fallback(self, imap):
        """Direct tests must not hide fallback required by changed shared builder code."""
        result = test_impact.analyze_impact(
            [
                "python/tensorrt_model_connect/engine_builder.py",
                "tests/builder/test_family_unit.py",
            ],
            imap,
            coverage_map={},
        )

        assert result.builder_tests == ["tests/builder/test_family_unit.py"]
        assert "builder" in result.fallback_tiers

    def test_changed_family_builder_test_selected_directly_without_coverage_map(self, imap):
        """Changed colocated family builder tests run directly without E2E impact."""
        result = test_impact.analyze_impact(
            ["python/tensorrt_model_connect/families/media_family/tests/test_family.py"],
            imap,
        )

        assert result.builder_tests == [
            "python/tensorrt_model_connect/families/media_family/tests/test_family.py"
        ]
        assert result.e2e_models == []
        assert result.fallback_tiers == []

    def test_changed_e2e_harness_unit_test_selected_directly(self, imap):
        """Changed e2e_harness test files run directly without broad E2E impact."""
        result = test_impact.analyze_impact(
            ["tests/e2e_harness/test_orchestrator_phases.py"],
            imap,
            coverage_map={},
        )

        assert result.e2e_models == []
        assert result.tools_tests == ["tests/e2e_harness/test_orchestrator_phases.py"]
        assert "tools" not in result.fallback_tiers

    @pytest.mark.parametrize("coverage_map", [None, {}])
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (
                "tools/ci/e2e_scheduler.py",
                [
                    "tests/tools/test_github_actions_ci.py",
                    "tests/tools/test_schedule_e2e.py",
                ],
            ),
            ("tools/ci/e2e_schedule.py", ["tests/tools/test_schedule_e2e.py"]),
        ],
    )
    def test_e2e_runner_selects_explicit_tools_tests(
        self, imap, coverage_map, path, expected
    ):
        """E2E scheduler edits select tests that coverage cannot discover."""
        result = test_impact.analyze_impact(
            [path],
            imap,
            coverage_map=coverage_map,
        )

        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == expected
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    @pytest.mark.parametrize(
        "path",
        ["scripts/hf_cache_download_worker.py", "scripts/warm_hf_cache.py"],
    )
    def test_hf_cache_scripts_select_focused_tools_tests(
        self,
        imap,
        coverage_map,
        path,
    ):
        result = test_impact.analyze_impact(
            [path],
            imap,
            coverage_map=coverage_map,
        )

        assert result.e2e_models == imap.all_model_names
        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == ["tests/tools/test_warm_hf_cache_static.py"]
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    def test_model_ci_selects_its_explicit_tools_tests(self, imap, coverage_map):
        """Projection/impact edits select their focused tooling suite."""
        result = test_impact.analyze_impact(
            ["tools/model_ci.py"],
            imap,
            coverage_map=coverage_map,
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == ["tests/tools/test_model_ci.py"]
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    def test_nightly_artifact_selector_selects_focused_tools_tests(self, imap, coverage_map):
        result = test_impact.analyze_impact(
            ["tools/select_latest_attempt_artifact.py"],
            imap,
            coverage_map=coverage_map,
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == [
            "tests/tools/test_github_actions_ci.py",
            "tests/tools/test_select_latest_attempt_artifact.py",
        ]
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    def test_nightly_issue_tracker_selects_focused_tools_tests(self, imap, coverage_map):
        result = test_impact.analyze_impact(
            ["tools/nightly_issue_tracker.py"],
            imap,
            coverage_map=coverage_map,
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == [
            "tests/tools/test_github_actions_ci.py",
            "tests/tools/test_nightly_issue_tracker.py",
        ]
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    @pytest.mark.parametrize(
        "path",
        [
            "scripts/generate_e2e_report.py",
            "scripts/generate_e2e_report_assets/e2e_report.css",
            "scripts/generate_e2e_report_assets/e2e_report.js",
            "scripts/reporting/__init__.py",
            "scripts/reporting/vlm_assessment.py",
        ],
    )
    def test_e2e_report_selects_its_explicit_tools_tests(
        self,
        imap,
        coverage_map,
        path,
    ):
        """Report edits select report tests without scheduling model E2E."""
        result = test_impact.analyze_impact(
            [path],
            imap,
            coverage_map=coverage_map,
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert result.tools_tests == ["tests/tools/test_generate_report.py"]
        assert result.fallback_tiers == []

    @pytest.mark.parametrize("coverage_map", [None, {}])
    def test_model_runner_selects_lifecycle_tests(self, imap, coverage_map):
        """Uniform runner edits select tests that Python coverage cannot infer."""
        result = test_impact.analyze_impact(
            ["tests/e2e_harness/model_runner.py"],
            imap,
            coverage_map=coverage_map,
        )

        assert result.unit_tiers == []
        assert result.tools_tests == ["tests/tools/test_model_e2e_runner.py"]
        assert result.fallback_tiers == []

    def test_github_ci_config_selects_tools_tier(self, imap):
        """CI config edits must not be classified as unit-test no-impact."""
        result = test_impact.analyze_impact(
            [".github/workflows/trtmc-ci.yml"],
            imap,
            coverage_map={},
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert "tools" in result.fallback_tiers

    def test_test_impact_tool_selects_tools_tier(self, imap):
        """Impact tool edits must force validation of tools tests."""
        result = test_impact.analyze_impact(
            ["tools/test_impact.py"],
            imap,
            coverage_map={},
        )

        assert result.e2e_models == []
        assert result.unit_tiers == ["tools"]
        assert "tools" in result.fallback_tiers

    def test_json_output_includes_test_lists(self, imap):
        """JSON output includes builder_tests, cpp_tests, fallback_tiers."""
        result = test_impact.ImpactResult(
            e2e_models=["decoder-small"],
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
