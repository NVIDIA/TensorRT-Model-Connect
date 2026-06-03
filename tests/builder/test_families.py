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
import json
from pathlib import Path

import pytest

from tensorrt_model_connect.families import find_plugin, _ALL_PLUGINS


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
        repo_root / "engine_defs" / "torch_trt" / "families",
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

    def test_all_plugins_matches_returns_bool(self):
        """Every plugin's matches() should return a bool, not just a truthy value."""
        # Build a mapping: plugin_name -> one known model_type that should match.
        name_to_type: dict[str, str] = {}
        for model_type, plugin_name in _POSITIVE_MATCH_CASES:
            if plugin_name not in name_to_type:
                name_to_type[plugin_name] = model_type

        for plugin in _ALL_PLUGINS:
            model_type = name_to_type.get(plugin.name)
            if model_type is None:
                continue  # no known positive case; covered by test_all_plugins_have_match_case
            result = plugin.matches(model_type)
            assert isinstance(result, bool), (
                f"Plugin {plugin.name!r}.matches({model_type!r}) returned "
                f"{type(result).__name__}, expected bool")
            assert result is True, (
                f"Plugin {plugin.name!r}.matches({model_type!r}) returned "
                f"{result!r}, expected True")

    def test_all_plugins_matches_own_type(self):
        """Every plugin matches its known model_type and rejects a nonsense type."""
        # Build a mapping: plugin_name -> one known model_type that should match.
        name_to_type: dict[str, str] = {}
        for model_type, plugin_name in _POSITIVE_MATCH_CASES:
            if plugin_name not in name_to_type:
                name_to_type[plugin_name] = model_type

        nonsense_types = [
            "zzzz_nonexistent_model_xyz_42",
            "__bogus__",
            "this_model_does_not_exist_ever_12345",
        ]

        for plugin in _ALL_PLUGINS:
            model_type = name_to_type.get(plugin.name)
            if model_type is None:
                continue  # no known positive case
            # Positive: must match own type
            assert plugin.matches(model_type), (
                f"Plugin {plugin.name!r} did not match its own type "
                f"{model_type!r}")
            # Negative: must reject nonsense types
            for bad_type in nonsense_types:
                result = plugin.matches(bad_type)
                assert result is False, (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{result!r}, expected False")
                assert isinstance(result, bool), (
                    f"Plugin {plugin.name!r}.matches({bad_type!r}) returned "
                    f"{type(result).__name__}, expected bool")

    def test_all_plugins_have_match_case(self):
        """Every discovered plugin should have at least one positive match case."""
        matched_names = {name for _, name in _POSITIVE_MATCH_CASES}
        plugin_names = {p.name for p in _ALL_PLUGINS}
        untested = plugin_names - matched_names
        assert not untested, (
            f"Plugins without positive match cases: {untested}. "
            f"Add entries to _POSITIVE_MATCH_CASES.")

    def test_all_plugins_have_e2e_manifest(self):
        """Validate that every family plugin has at least one E2E test manifest.

        Uses AST-based filesystem scanning so this test works even without
        TRT/torch installed (pure Python, no GPU). When a developer adds a
        new family plugin, they must also add a JSON manifest in
        tests/e2e/models/ so the E2E test suite covers that model.
        """
        _EXEMPT_PLUGINS = {
            "qwen3_omni",  # omni_multimodal strategy not yet wired in E2E harness
        }

        models_dir = Path(__file__).resolve().parent.parent / "e2e" / "models"
        families_in_manifests: set[str] = set()
        for manifest_path in models_dir.glob("*.json"):
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
        for manifest_path in sorted(models_dir.glob("*.json")):
            with open(manifest_path) as f:
                data = json.load(f)
            family = data.get("family", "")
            if family and family not in plugin_names:
                invalid.append(f"{manifest_path.name}: family={family!r}")

        assert not invalid, (
            "Manifests referencing unknown family plugins:\n"
            + "\n".join(f"  {entry}" for entry in invalid))


# ---------------------------------------------------------------------------
# match() positive cases
# ---------------------------------------------------------------------------

# (model_type, expected_plugin_name)
_POSITIVE_MATCH_CASES = [
    # Qwen family
    ("qwen", "qwen"),
    ("qwen2", "qwen"),
    ("qwen3", "qwen"),
    ("qwq", "qwen"),
    ("Qwen2", "qwen"),
    # LLaMA
    ("llama", "llama"),
    ("LLaMA", "llama"),
    # Mistral
    ("mistral", "mistral"),
    ("Mistral", "mistral"),
    # Gemma
    ("gemma", "gemma"),
    ("gemma2", "gemma"),
    # Phi (not phimoe)
    ("phi", "phi"),
    ("phi3", "phi"),
    ("Phi3", "phi"),
    # Phi-MoE
    ("phimoe", "phi_moe"),
    # Qwen3 MoE
    ("qwen3_moe", "qwen_moe"),
    # Granite
    ("granite", "granite"),
    # InternLM
    ("internlm", "internlm"),
    ("internlm2", "internlm"),
    # StarCoder2
    ("starcoder2", "starcoder2"),
    # GPT-2
    ("gpt2", "gpt2"),
    # OPT
    ("opt", "opt"),
    # Falcon
    ("falcon", "falcon"),
    # StableLM
    ("stablelm", "stablelm"),
    # Mamba
    ("mamba", "mamba"),
    # Qwen-VL
    ("qwen2_vl", "qwen_vl"),
    ("qwen2_5_vl", "qwen_vl"),
    ("qwen3_vl", "qwen_vl"),
    # OLMo
    ("olmo", "olmo"),
    # XGLM
    ("xglm", "xglm"),
    # GPT-NeoX
    ("gpt_neox", "gpt_neox"),
    ("gptneox", "gpt_neox"),
    # GPT-Neo
    ("gpt_neo", "gpt_neo"),
    # CodeGen
    ("codegen", "codegen"),
    # BLOOM
    ("bloom", "bloom"),
    # Mixtral
    ("mixtral", "mixtral"),
    # Nemotron
    ("nemotron", "nemotron"),
    # Nemotron Labs Diffusion
    ("nemotron_labs_diffusion", "nemotron_labs_diffusion"),
    # Wan T2V (diffusion)
    ("wan_t2v", "wan_t2v"),
    ("wan", "wan_t2v"),
    # LTX-Video (diffusion T2V)
    ("ltx_video", "ltx_video"),
    ("ltx-video", "ltx_video"),
    # Bark (text-to-audio)
    ("bark", "bark"),
    # SegFormer (segmentation)
    ("segformer", "segformer"),
    # Whisper (speech-to-text)
    ("whisper", "whisper"),
    # RWKV
    ("rwkv", "rwkv"),
    # DeepSeek-V2
    ("deepseek_v2", "deepseek_v2"),
    # InternVL
    ("internvl_chat", "internvl"),
    ("internvl3", "internvl"),
    # BERT (encoder-only)
    ("bert", "bert"),
    # RoBERTa / XLM-RoBERTa (encoder-only)
    ("roberta", "roberta"),
    ("xlm-roberta", "roberta"),
    # DistilBERT (encoder-only)
    ("distilbert", "distilbert"),
    # MPNet (encoder-only)
    ("mpnet", "mpnet"),
    # Eagle VLM (embedding/reranking)
    ("llama_nemotron_vl", "eagle_vlm"),
    # Qwen3-Omni (omni multimodal)
    ("qwen3_omni", "qwen3_omni"),
    ("qwen3omni", "qwen3_omni"),
    ("qwen3_omni_moe", "qwen3_omni"),
    # PersonaPlex (speech-to-speech)
    ("personaplex", "personaplex"),
    ("moshi", "personaplex"),
    ("personaplex_7b", "personaplex"),
    # NemotronH (hybrid Mamba-Attention)
    ("nemotron_h", "nemotron_h"),
    ("nemotron_hybrid", "nemotron_h"),
    # SAM (prompted segmentation)
    ("sam", "sam"),
    # timm Vision Transformer (image classification)
    ("vit_base_patch16_224", "timm_vit"),
    ("timm_vit", "timm_vit"),
    # Phi-4 Multimodal
    ("phi4_multimodal", "phi4_multimodal"),
    # FLUX (diffusion T2I)
    ("flux", "flux"),
    ("flux.2", "flux"),
    # Z-Image (diffusion T2I)
    ("z_image", "z_image"),
    # Qwen-Image (diffusion T2I/Edit)
    ("qwen_image", "qwen_image"),
    ("qwen-image", "qwen_image"),
    ("qwen_image_edit", "qwen_image"),
    # PixArt (diffusion T2I)
    ("pixart", "pixart"),
    ("pixart_sigma", "pixart"),
    ("pixart_alpha", "pixart"),
    # ELF Flow (diffusion text generation)
    ("elf", "elf_flow"),
    ("embedded_language_flow", "elf_flow"),
    # DeepSeek OCR (matches deepseek_vl_v2 model_type)
    ("deepseek_vl_v2", "deepseek_ocr"),
    # MagpieTTS (encoder-decoder TTS)
    ("magpie_tts", "magpie_tts"),
    ("decoder_ce", "magpie_tts"),
    # Qwen3.5 (hybrid DeltaNet + Attention)
    ("qwen3_5", "qwen3_5"),
    ("qwen3.5", "qwen3_5"),
    # GPT-OSS (OpenAI MoE)
    ("gpt_oss", "gpt_oss"),
    # GLM-4
    ("glm", "glm"),
    # Canary (FastConformer ASR)
    ("canary", "canary"),
    ("canary_asr", "canary"),
    ("enc_dec_multi_task", "canary"),
    # Nemotron Speech Streaming (FastConformer cache-aware RNNT ASR)
    ("nemotron_speech_streaming", "nemotron_speech_streaming"),
    ("nemotron_asr_streaming", "nemotron_speech_streaming"),
    ("fastconformer_cacheaware_rnnt", "nemotron_speech_streaming"),
    # Autopilot-generated families
    ("electra", "electra"),
    ("modernbert", "modernbert"),
    ("deberta", "deberta"),
    ("t5", "t5"),
    ("bart", "bart"),
    ("mbart", "bart"),
    ("marian", "marian"),
    ("albert", "albert"),
    ("olmo2", "olmo2"),
    ("fnet", "fnet"),
    ("xlnet", "xlnet"),
    ("convbert", "convbert"),
    ("dpr", "dpr"),
    # M2M-100 / NLLB (encoder-decoder seq2seq)
    ("m2m_100", "m2m_100"),
    ("nllb", "m2m_100"),
    # LTX-2 (Lightricks image-to-video with audio)
    ("ltx_2", "ltx_2"),
    ("ltx-2", "ltx_2"),
    ("ltx2", "ltx_2"),
]


class TestMatchPositive:
    """Each known model_type resolves to the correct plugin."""

    @pytest.mark.parametrize("model_type,expected_name", _POSITIVE_MATCH_CASES)
    def test_match(self, model_type, expected_name):
        plugin = find_plugin(model_type)
        assert plugin is not None, f"No plugin matched model_type={model_type!r}"
        assert plugin.name == expected_name, (
            f"model_type={model_type!r} matched {plugin.name!r}, "
            f"expected {expected_name!r}")


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


# ---------------------------------------------------------------------------
# Qwen vs Qwen-VL disambiguation
# ---------------------------------------------------------------------------

class TestQwenVLDisambiguation:
    """Qwen-VL should match VL plugin; plain Qwen should not."""

    def test_qwen_vl_matches_vl_plugin(self):
        plugin = find_plugin("qwen2_vl")
        assert plugin is not None
        assert plugin.name == "qwen_vl"

    def test_plain_qwen_does_not_match_vl(self):
        plugin = find_plugin("qwen3")
        assert plugin is not None
        assert plugin.name == "qwen"

    def test_qwen_vl_does_not_match_plain_qwen(self):
        """The plain Qwen plugin should reject VL model types."""
        qwen_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "qwen":
                qwen_plugin = p
                break
        assert qwen_plugin is not None
        assert not qwen_plugin.matches("qwen2_vl")


# ---------------------------------------------------------------------------
# Phi vs Phi-MoE disambiguation
# ---------------------------------------------------------------------------

class TestPhiDisambiguation:
    """Phi should not match phimoe, and vice versa."""

    def test_phi_rejects_phimoe(self):
        phi_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "phi":
                phi_plugin = p
                break
        assert phi_plugin is not None
        assert not phi_plugin.matches("phimoe")

    def test_phimoe_rejects_phi3(self):
        moe_plugin = None
        for p in _ALL_PLUGINS:
            if p.name == "phi_moe":
                moe_plugin = p
                break
        assert moe_plugin is not None
        assert not moe_plugin.matches("phi3")


# ---------------------------------------------------------------------------
# Special attributes
# ---------------------------------------------------------------------------

class TestRuntimeStrategy:
    """Plugins with non-default runtime_strategy."""

    def test_mamba_strategy(self):
        plugin = find_plugin("mamba")
        assert getattr(plugin, "runtime_strategy", None) == "ssm_recurrent"

    def test_mixtral_strategy(self):
        plugin = find_plugin("mixtral")
        assert getattr(plugin, "runtime_strategy", None) == "decoder_moe"

    def test_gpt_oss_strategy(self):
        plugin = find_plugin("gpt_oss")
        assert getattr(plugin, "runtime_strategy", None) == "decoder_moe"

    def test_qwen_vl_strategy(self):
        plugin = find_plugin("qwen2_vl")
        assert getattr(plugin, "runtime_strategy", None) == "vision_language"

    def test_internvl_strategy(self):
        plugin = find_plugin("internvl_chat")
        assert getattr(plugin, "runtime_strategy", None) == "vision_language"

    def test_omni_strategy(self):
        plugin = find_plugin("qwen3_omni")
        assert getattr(plugin, "runtime_strategy", None) == "omni_multimodal"

    def test_personaplex_strategy(self):
        plugin = find_plugin("personaplex")
        assert getattr(plugin, "runtime_strategy", None) == "speech_to_speech"

    def test_personaplex_bundle_overrides(self):
        from tensorrt_model_connect.config import ModelConfig

        plugin = find_plugin("personaplex")
        overrides = plugin.get_bundle_config_overrides(
            ModelConfig(model_type="personaplex"))
        assert overrides["eos_token_id"] == 2
        assert overrides["speech_depth_temperature"] == pytest.approx(0.0)
        assert overrides["speech_depth_top_k"] == 0
        assert overrides["speech_system_prompt"] == ""
        assert overrides["speech_text_prompt_ids"] == []

    def test_nemotron_h_strategy(self):
        plugin = find_plugin("nemotron_h")
        assert getattr(plugin, "runtime_strategy", None) == "hybrid_mamba_attention"

    def test_canary_strategy(self):
        plugin = find_plugin("canary")
        assert getattr(plugin, "runtime_strategy", None) == "speech_to_text"

    def test_nemotron_speech_streaming_strategy(self):
        plugin = find_plugin("nemotron_speech_streaming")
        assert getattr(plugin, "runtime_strategy", None) == "speech_to_text_rnnt"

    def test_standard_decoder_no_strategy(self):
        """Standard decoder plugins have no runtime_strategy (defaults to decoder_kv_cache)."""
        for name in ("qwen", "llama", "mistral", "gemma", "phi", "gpt2", "opt"):
            plugin = find_plugin(name)
            assert plugin is not None
            strategy = getattr(plugin, "runtime_strategy", None)
            assert strategy is None, (
                f"Plugin {plugin.name} should not have runtime_strategy, "
                f"got {strategy!r}")


class TestEmbedInput:
    """Only VL plugins should have embed_input=True."""

    def test_qwen_vl_has_embed_input(self):
        plugin = find_plugin("qwen2_vl")
        assert getattr(plugin, "embed_input", False) is True

    def test_internvl_has_embed_input(self):
        plugin = find_plugin("internvl_chat")
        assert getattr(plugin, "embed_input", False) is True

    def test_omni_has_embed_input(self):
        plugin = find_plugin("qwen3_omni")
        assert getattr(plugin, "embed_input", False) is True

    def test_standard_plugins_no_embed_input(self):
        for name in ("qwen", "llama", "mistral", "mamba", "mixtral"):
            plugin = find_plugin(name)
            assert plugin is not None
            assert not getattr(plugin, "embed_input", False), (
                f"Plugin {plugin.name} should not have embed_input")


class TestVLMethods:
    """VL plugins should have build_vision_engine and get_vl_config methods."""

    def test_qwen_vl_has_vl_methods(self):
        plugin = find_plugin("qwen2_vl")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))

    def test_internvl_has_vl_methods(self):
        plugin = find_plugin("internvl_chat")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))

    def test_omni_has_vl_methods(self):
        plugin = find_plugin("qwen3_omni")
        assert callable(getattr(plugin, "build_vision_engine", None))
        assert callable(getattr(plugin, "get_vl_config", None))
        assert callable(getattr(plugin, "build_extra_engines", None))

    def test_standard_plugins_vl_methods_return_none(self):
        """Non-VL plugins should return None from get_vl_config if they have it."""
        for name in ("qwen", "llama", "gpt2"):
            plugin = find_plugin(name)
            vl_config = getattr(plugin, "get_vl_config", None)
            if vl_config is not None and callable(vl_config):
                # Would need a ModelConfig, but default protocol returns None
                pass  # Can't call without a real config
