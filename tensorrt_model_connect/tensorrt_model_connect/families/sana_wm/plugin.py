"""SANA-WM family plugin.

The public SANA-WM release is not a standard diffusers directory: it ships a
Sana-specific config.yaml plus DiT, LTX-2 VAE, and refiner weights. Building a
native TRT graph for this model family is separate work. This plugin creates a
TRTMC control bundle that routes runtime execution through the official
SANA-WM Python inference contract, while preserving the same inputs in e2e.
"""

from __future__ import annotations

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_HF_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
_DEFAULT_ACTION = "w-80,jw-40,w-40,lw-60,w-100"
_DEFAULT_TRANSLATION_SPEED = 0.055
_DEFAULT_ROTATION_SPEED_DEG = 1.2
_DEFAULT_NUM_FRAMES = 321
_DEFAULT_HEIGHT = 704
_DEFAULT_WIDTH = 1280


class SanaWmPlugin:
    name = "sana_wm"
    runtime_strategy = "diffusion_sana_wm"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in (
            "sana_wm",
            "sana-wm",
            "sanamsvideocamctrl_1600m_p1_d20",
        )

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        del model_dir
        weights = WeightDict()
        weights["_model_format"] = "sana_wm_yaml"
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del config, weights, max_cache_length, precision, quant_ctx, verbose
        # The runtime plugin ignores engine_plan. A small marker section keeps
        # the bundle shape compatible with the generic builder/writer path.
        return b"TRTMC_SANA_WM_PYTHON_BRIDGE\n"

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        raw = config.raw
        text_encoder = raw.get("text_encoder", {})
        scheduler = raw.get("scheduler", {})
        vae = raw.get("vae", {})
        if not isinstance(text_encoder, dict):
            text_encoder = {}
        if not isinstance(scheduler, dict):
            scheduler = {}
        if not isinstance(vae, dict):
            vae = {}

        video_height = int(raw.get("video_height", _DEFAULT_HEIGHT))
        video_width = int(raw.get("video_width", _DEFAULT_WIDTH))
        video_num_frames = int(raw.get("video_num_frames", _DEFAULT_NUM_FRAMES))

        return {
            "model_type": "sana_wm",
            "runtime_strategy": self.runtime_strategy,
            "engine_backend": "none",
            "sana_wm_hf_id": _HF_ID,
            "sana_wm_config_path": f"hf://{_HF_ID}/config.yaml",
            "sana_wm_model_path": f"hf://{_HF_ID}/dit/sana_wm_1600m_720p.safetensors",
            "sana_wm_refiner_checkpoint": f"hf://{_HF_ID}/refiner/refiner.safetensors",
            "sana_wm_refiner_gemma_root": f"hf://{_HF_ID}/refiner/text_encoder",
            "sana_wm_require_official_script": int(
                raw.get("sana_wm_require_official_script", 1)
            ),
            "sana_wm_action": str(raw.get("sana_wm_action", _DEFAULT_ACTION)),
            "sana_wm_translation_speed": float(
                raw.get("sana_wm_translation_speed", _DEFAULT_TRANSLATION_SPEED)
            ),
            "sana_wm_rotation_speed_deg": float(
                raw.get("sana_wm_rotation_speed_deg", _DEFAULT_ROTATION_SPEED_DEG)
            ),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "vae_latent_dim": int(vae.get("vae_latent_dim", raw.get("vae_latent_dim", 128))),
            "vae_downsample_rate": int(
                vae.get("vae_downsample_rate", raw.get("vae_downsample_rate", 32))
            ),
            "text_encoder_name": str(
                text_encoder.get("text_encoder_name")
                or text_encoder.get("model")
                or "gemma-2-2b-it"
            ),
            "text_encoder_max_length": int(text_encoder.get("model_max_length", 300)),
            "flow_shift": float(scheduler.get("inference_flow_shift", 9.8)),
        }


plugin = SanaWmPlugin()
