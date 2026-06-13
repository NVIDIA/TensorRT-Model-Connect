"""Cosmos3 plugin protocol + builder-wrapper tests."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.cosmos3.plugin import Cosmos3Plugin, plugin as cosmos3_plugin


class TestPluginProtocol:
    def test_singleton_exists(self):
        assert isinstance(cosmos3_plugin, Cosmos3Plugin)

    def test_plugin_name(self):
        assert Cosmos3Plugin.name == "cosmos3"

    def test_runtime_strategy(self):
        assert Cosmos3Plugin.runtime_strategy == "diffusion_cosmos3"

    def test_pipeline_classes(self):
        # The C++ runtime exposes Cosmos3Pipeline; an Omni variant is
        # mentioned in plugin.py for the full omni-loop case.
        assert "Cosmos3Pipeline" in Cosmos3Plugin.pipeline_classes


class TestPluginMatcher:
    @pytest.fixture
    def plugin(self):
        return Cosmos3Plugin()

    @pytest.mark.parametrize("model_type", [
        "cosmos3", "cosmos3_nano", "cosmos3_super",
        "Cosmos3", "COSMOS3", "cosmos-3", "cosmos_3",
        "cosmos3-super-reasoner",
    ])
    def test_matches_cosmos3_variants(self, plugin, model_type):
        assert plugin.matches(model_type)

    @pytest.mark.parametrize("model_type", [
        "cosmos", "cosmos2", "cosmos_predict", "cosmos_transfer",
        "cosmos_predict2", "qwen3_vl", "llama", "",
    ])
    def test_does_not_match_unrelated(self, plugin, model_type):
        assert not plugin.matches(model_type)


class TestPluginBehaviour:
    def test_build_components_raises_for_unimplemented_phases(self):
        plugin = Cosmos3Plugin()
        with pytest.raises(NotImplementedError) as exc:
            plugin.build_components(
                model_dir="/tmp/nope",
                config=None,
                weights={"_model_format": "diffusers"},
            )
        msg = str(exc.value)
        # Sanity: error names the remaining phases (Phase 4 DM + Phase 6
        # runtime) so the operator knows what's outstanding.
        assert "Phase 4" in msg
        assert "Phase 6" in msg
        assert "reasoner-only" in msg.lower()

    def test_get_diffusion_config_reports_omni_outputs(self):
        plugin = Cosmos3Plugin()
        cfg = plugin.get_diffusion_config(config=None)
        assert cfg["family"] == "cosmos3"
        # All five output modalities should be reported as supported.
        assert cfg["supports_image_output"]
        assert cfg["supports_video_output"]
        assert cfg["supports_audio_output"]
        assert cfg["supports_action_output"]
        assert cfg["supports_text_output"]
