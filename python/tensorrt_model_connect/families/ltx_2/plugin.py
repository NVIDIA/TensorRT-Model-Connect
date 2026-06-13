"""LTX-2 family plugin (scaffold).

Onboarding scaffold for Lightricks' joint audio-video DiT model:
``Lightricks/LTX-2`` (HF). LTX-2 is an asymmetric dual-stream
transformer that generates synchronized video and audio. The HF
diffusers checkpoint exposes the following subfolders::

    audio_vae        AutoencoderKLLTX2Audio (audio latent VAE)
    connectors       LTX2TextConnectors    (text -> video/audio adapters)
    latent_upsampler LTX2LatentUpsamplerModel (spatial latent upsampler)
    scheduler        FlowMatchEulerDiscreteScheduler
    text_encoder     T5-v1_1-XXL           (caption_channels=3840 after proj)
    tokenizer        T5TokenizerFast
    transformer      LTX2VideoTransformer3DModel
                     (14B video stream + 5B audio stream, cross-modality
                      attention, qk_norm="rms_norm_across_heads")
    vae              AutoencoderKLLTX2Video
    vocoder          LTX2Vocoder

Status
------
This plugin is a *scaffold*: ``matches()``, ``load_weights()`` and
``get_diffusion_config()`` work end-to-end against an LTX-2 diffusers
checkpoint, but ``build_components()`` deliberately raises
``NotImplementedError`` because the C++ runtime
(``runtime_strategy="diffusion_ltx_2"``) and the dual-stream DiT /
audio_vae / vocoder / connectors / latent_upsampler TRT builders are
not yet implemented. The error message lists what is missing so a
follow-up MR can land each component.

The intent is *not* to inherit from :class:`LTXVideoPlugin` — LTX-2 is
architecturally a superset (audio branch, cross-modality attention,
text connectors, latent upsampler, vocoder) and almost every
component's weight layout differs from LTX-Video. A clean family with
its own builders matches the wan_t2v / flux pattern.

Architectural deltas vs LTX-Video
---------------------------------
LTX-Video (ltx_video family):
    T5 (d_model=4096) -> LTXVideoTransformer3DModel -> AutoencoderKLLTXVideo
    Single stream. patch_size=[1,1,1]. z=128. flow_match_euler.

LTX-2 (this family):
    T5 (d_model=4096, projected via connectors to 3840) ->
      LTX2TextConnectors (video_connector, audio_connector) ->
        LTX2VideoTransformer3DModel (14B video stream) <--> 5B audio stream
          (cross-modality AdaLN, audio-video cross-attention) ->
            AutoencoderKLLTX2Video    (video latents -> RGB)
            AutoencoderKLLTX2Audio    (audio latents -> mel-spec)
              LTX2Vocoder             (mel-spec -> waveform)
    Plus LTX2LatentUpsamplerModel for two-stage spatial upsampling.

Open questions for GPU validation
---------------------------------
1. Exact transformer config (num_attention_heads, attention_head_dim,
   num_layers per stream) — must be read from
   ``transformer/config.json``.
2. Whether ``LTX2VideoTransformer3DModel`` is a true monolithic 19B
   module or split into ``transformer/video`` / ``transformer/audio``
   subfolders on disk.
3. caption_channels=3840 implies a 4096 -> 3840 projection inside the
   text connector — confirm whether this lives in connectors/ or in
   the transformer's caption_projection.
4. ``qk_norm="rms_norm_across_heads"`` — needs a TRT graph_ops
   primitive that normalises across the head axis (not the head_dim
   axis as in standard rms_norm).
5. Audio VAE compression ratios (temporal/spectral) and vocoder
   architecture (HiFi-GAN-style? BigVGAN? Mel-spec dims).
6. Whether the latent_upsampler is always-on (two-stage pipeline) or
   only for high-resolution variants — affects whether it is an
   independent engine or fused into the DiT runtime path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


class LTX2Plugin:
    """LTX-2 joint audio-video DiT family.

    Scaffold only — ``build_components`` is not implemented.
    """

    name = "ltx_2"
    runtime_strategy = "diffusion_ltx_2"
    # diffusers pipeline classes that should auto-route here.
    pipeline_classes = [
        "LTX2Pipeline",
        "LTX2ImageToVideoPipeline",
        "LTX2LatentUpsamplePipeline",
    ]

    # ------------------------------------------------------------------
    # Architecture defaults (placeholders until validated on GPU)
    # ------------------------------------------------------------------

    # T5-v1_1-XXL text encoder defaults. ``caption_channels`` (3840)
    # is the *post-connector* dimension fed into the DiT; the raw T5
    # output is still d_model = 4096.
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 256

    # LTX2 video DiT (placeholders — read from transformer/config.json).
    _DIT_IN_CHANNELS = 128
    _DIT_OUT_CHANNELS = 128
    _DIT_CAPTION_CHANNELS = 3840
    _DIT_DIM = 4096
    _DIT_NUM_HEADS = 32
    _DIT_NUM_LAYERS = 48
    _DIT_QK_NORM = "rms_norm_across_heads"

    # LTX2 audio stream (placeholders — read from transformer/config.json).
    _AUDIO_DIM = 2048
    _AUDIO_NUM_HEADS = 16
    _AUDIO_NUM_LAYERS = 24

    # Text connectors (LTX2TextConnectors): map T5 4096 -> caption_channels.
    _VIDEO_CONNECTOR_DIM = 3840
    _VIDEO_CONNECTOR_NUM_LAYERS = 2
    _AUDIO_CONNECTOR_DIM = 2048
    _AUDIO_CONNECTOR_NUM_LAYERS = 2

    # Video VAE (AutoencoderKLLTX2Video).
    _VAE_Z_DIM = 128
    _VAE_SCALE_TEMPORAL = 8
    _VAE_SCALE_SPATIAL = 32

    # Audio VAE (AutoencoderKLLTX2Audio).
    _AUDIO_VAE_Z_DIM = 16
    _AUDIO_VAE_SCALE_TEMPORAL = 256  # mel-frame temporal compression
    _AUDIO_VAE_MEL_BINS = 80

    # Latent upsampler (LTX2LatentUpsamplerModel).
    _UPSAMPLER_INPUT_DIM = 128
    _UPSAMPLER_OUTPUT_DIM = 128

    # Default 5s @ 24fps clip at 704x480.
    _DEFAULT_HEIGHT = 480
    _DEFAULT_WIDTH = 704
    _DEFAULT_NUM_FRAMES = 121
    _DEFAULT_FRAME_RATE = 24
    _DEFAULT_NUM_STEPS = 30
    _DEFAULT_GUIDANCE_SCALE = 3.0
    _DEFAULT_NEGATIVE_PROMPT = (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    _DEFAULT_AUDIO_SAMPLE_RATE = 44100

    # Patch packing used by the LTX2 DiT.
    _PATCH_SIZE = [1, 1, 1]

    # ------------------------------------------------------------------
    # Plugin protocol
    # ------------------------------------------------------------------

    def matches(self, model_type: str) -> bool:
        """Recognize LTX-2 by model_type or family alias."""
        mt = (model_type or "").lower()
        return mt in (
            "ltx_2",
            "ltx-2",
            "ltx2",
            "ltx_v2",
            "ltx2pipeline",
        )

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Discover the LTX-2 diffusers layout and read component configs.

        Does not load tensor data — actual weight tensors are loaded
        inside ``build_components`` once that is implemented.
        """
        model_path = Path(model_dir)
        model_index_path = model_path / "model_index.json"
        if not model_index_path.exists():
            raise ValueError(
                f"Expected diffusers format with model_index.json in {model_dir}"
            )

        model_index = json.loads(model_index_path.read_text())
        pipeline_class = str(model_index.get("_class_name", ""))
        if pipeline_class and pipeline_class not in self.pipeline_classes:
            print(
                f"[ltx-2] Warning: unrecognised pipeline class {pipeline_class!r}; "
                f"continuing with LTX-2 scaffold.",
                file=sys.stderr,
            )

        weights = WeightDict()
        weights["_model_format"] = "diffusers"
        weights["_pipeline_class"] = pipeline_class

        # Required component directories — all must exist for LTX-2.
        required_subdirs = (
            ("_text_encoder_dir", "text_encoder"),
            ("_tokenizer_dir", "tokenizer"),
            ("_transformer_dir", "transformer"),
            ("_vae_dir", "vae"),
            ("_audio_vae_dir", "audio_vae"),
            ("_vocoder_dir", "vocoder"),
            ("_connectors_dir", "connectors"),
            ("_latent_upsampler_dir", "latent_upsampler"),
            ("_scheduler_dir", "scheduler"),
        )
        missing = []
        for key, rel in required_subdirs:
            subpath = model_path / rel
            if subpath.exists():
                weights[key] = str(subpath)
            else:
                missing.append(rel)
        if missing:
            # Don't hard-fail; some LTX-2 variants ship without the
            # latent_upsampler or audio branch. Surface what's missing so
            # build_components can decide.
            print(
                f"[ltx-2] Note: missing optional subdirs: {missing}",
                file=sys.stderr,
            )
            weights["_missing_subdirs"] = missing

        # Read per-component config.json files. These drive the actual
        # builder args and are also surfaced to get_diffusion_config().
        config_relpaths = (
            ("_text_encoder_config", "text_encoder/config.json"),
            ("_transformer_config", "transformer/config.json"),
            ("_vae_config", "vae/config.json"),
            ("_audio_vae_config", "audio_vae/config.json"),
            ("_vocoder_config", "vocoder/config.json"),
            ("_connectors_config", "connectors/config.json"),
            ("_latent_upsampler_config", "latent_upsampler/config.json"),
            ("_scheduler_config", "scheduler/scheduler_config.json"),
        )
        for key, rel in config_relpaths:
            path = model_path / rel
            if path.exists():
                try:
                    weights[key] = json.loads(path.read_text())
                    config.raw[key] = weights[key]
                except (OSError, ValueError) as exc:
                    print(
                        f"[ltx-2] Failed to parse {rel}: {exc}", file=sys.stderr,
                    )

        config.raw["_pipeline_class"] = pipeline_class

        # VAE latent normalisation stats live as tensors inside the VAE
        # safetensors blob and are needed at runtime. Mirror the LTX-Video
        # loader so the C++ runtime can consume them directly.
        vae_dir = weights.get("_vae_dir")
        if vae_dir:
            mean, std = _load_ltx2_vae_latent_stats(Path(vae_dir))
            if mean is not None and std is not None:
                weights["_vae_latents_mean"] = mean
                weights["_vae_latents_std"] = std
                config.raw["_vae_latents_mean"] = mean
                config.raw["_vae_latents_std"] = std
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
        raise NotImplementedError(
            "LTX-2 uses build_components(), not build_engine()"
        )

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        **_kwargs,
    ) -> dict:
        """Build all LTX-2 component engines.

        Not yet implemented. The following per-component builders need to
        land before this method can return a valid bundle dict:

        - ``t5_encoder_builder``  — share with ltx_video once the
          caption-projection path is confirmed.
        - ``ltx2_text_connectors_builder`` — projects T5 hidden states
          into video (3840-d) and audio (2048-d) caption streams.
        - ``ltx2_video_dit_builder`` — dual-stream DiT with
          rms_norm_across_heads, audio-video cross-attention, and
          cross-modality AdaLN.
        - ``ltx2_audio_dit_builder`` — the 5B audio branch (may live
          inside the same transformer/ checkpoint as the video stream).
        - ``ltx2_video_vae_builder`` — AutoencoderKLLTX2Video decoder.
        - ``ltx2_audio_vae_builder`` — AutoencoderKLLTX2Audio decoder.
        - ``ltx2_vocoder_builder`` — LTX2Vocoder (mel -> waveform).
        - ``ltx2_latent_upsampler_builder`` — LTX2LatentUpsamplerModel
          for two-stage spatial upsampling.

        See the module docstring for architectural notes and open
        questions for GPU validation.
        """
        del model_dir, config, weights, precision, verbose, _kwargs
        missing = [
            "ltx2_text_connectors (connectors/)",
            "ltx2_video_dit + ltx2_audio_dit (transformer/)",
            "ltx2_video_vae (vae/)",
            "ltx2_audio_vae (audio_vae/)",
            "ltx2_vocoder (vocoder/)",
            "ltx2_latent_upsampler (latent_upsampler/)",
            "diffusion_ltx_2 runtime strategy in C++",
        ]
        raise NotImplementedError(
            "LTX-2 build_components is not implemented yet. Missing builders/"
            "runtime pieces: " + "; ".join(missing) + ". See "
            "tensorrt_model_connect.families.ltx_2.plugin docstring for the "
            "scaffolding plan."
        )

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return the diffusion pipeline config for the bundle.

        Reads from per-component config.json blobs captured during
        :meth:`load_weights`. Falls back to scaffold defaults when a
        config field is absent.
        """
        transformer_cfg = config.raw.get("_transformer_config", {}) or {}
        scheduler_cfg = config.raw.get("_scheduler_config", {}) or {}
        vae_cfg = config.raw.get("_vae_config", {}) or {}
        audio_vae_cfg = config.raw.get("_audio_vae_config", {}) or {}
        vocoder_cfg = config.raw.get("_vocoder_config", {}) or {}
        connectors_cfg = config.raw.get("_connectors_config", {}) or {}
        upsampler_cfg = config.raw.get("_latent_upsampler_config", {}) or {}

        height = int(config.raw.get("video_height", self._DEFAULT_HEIGHT))
        width = int(config.raw.get("video_width", self._DEFAULT_WIDTH))
        num_frames = int(
            config.raw.get("video_num_frames", self._DEFAULT_NUM_FRAMES)
        )

        dit_dim = int(
            transformer_cfg.get("num_attention_heads", self._DIT_NUM_HEADS)
        ) * int(
            transformer_cfg.get("attention_head_dim", self._DIT_DIM // self._DIT_NUM_HEADS)
        )

        return {
            "diffusion_backend_type": "ltx_2",
            "scheduler": "flow_match_euler",
            "num_inference_steps": int(
                config.raw.get("num_inference_steps", self._DEFAULT_NUM_STEPS)
            ),
            "guidance_scale": float(
                config.raw.get("guidance_scale", self._DEFAULT_GUIDANCE_SCALE)
            ),
            # Video shape
            "video_height": height,
            "video_width": width,
            "video_num_frames": num_frames,
            "frame_rate": int(
                config.raw.get("frame_rate", self._DEFAULT_FRAME_RATE)
            ),
            "negative_prompt": str(
                config.raw.get("negative_prompt", self._DEFAULT_NEGATIVE_PROMPT)
            ),
            # Video DiT
            "z_dim": int(
                transformer_cfg.get("in_channels", self._DIT_IN_CHANNELS)
            ),
            "dit_dim": dit_dim,
            "dit_num_heads": int(
                transformer_cfg.get("num_attention_heads", self._DIT_NUM_HEADS)
            ),
            "dit_num_layers": int(
                transformer_cfg.get("num_layers", self._DIT_NUM_LAYERS)
            ),
            "dit_caption_channels": int(
                transformer_cfg.get(
                    "caption_channels", self._DIT_CAPTION_CHANNELS
                )
            ),
            "dit_qk_norm": str(
                transformer_cfg.get("qk_norm", self._DIT_QK_NORM)
            ),
            "patch_size": list(
                transformer_cfg.get("patch_size", self._PATCH_SIZE)
            ),
            # Video VAE
            "scale_factor_temporal": int(
                vae_cfg.get(
                    "temporal_compression_ratio", self._VAE_SCALE_TEMPORAL
                )
            ),
            "scale_factor_spatial": int(
                vae_cfg.get(
                    "spatial_compression_ratio", self._VAE_SCALE_SPATIAL
                )
            ),
            "vae_scaling_factor": float(vae_cfg.get("scaling_factor", 1.0)),
            "latents_mean": list(config.raw.get("_vae_latents_mean", [])),
            "latents_std": list(config.raw.get("_vae_latents_std", [])),
            # Text encoder
            "text_seq_len": int(
                config.raw.get("text_seq_len", self._T5_MAX_SEQ_LEN)
            ),
            "text_encoder_dim": self._T5_D_MODEL,
            # Connectors (T5 -> video/audio caption streams)
            "video_connector_dim": int(
                connectors_cfg.get(
                    "video_connector_hidden_size", self._VIDEO_CONNECTOR_DIM
                )
            ),
            "video_connector_num_layers": int(
                connectors_cfg.get(
                    "video_connector_num_layers",
                    self._VIDEO_CONNECTOR_NUM_LAYERS,
                )
            ),
            "audio_connector_dim": int(
                connectors_cfg.get(
                    "audio_connector_hidden_size", self._AUDIO_CONNECTOR_DIM
                )
            ),
            "audio_connector_num_layers": int(
                connectors_cfg.get(
                    "audio_connector_num_layers",
                    self._AUDIO_CONNECTOR_NUM_LAYERS,
                )
            ),
            # Audio branch
            "audio_dit_dim": int(
                transformer_cfg.get(
                    "audio_attention_head_dim", self._AUDIO_DIM // self._AUDIO_NUM_HEADS
                )
            ) * int(
                transformer_cfg.get(
                    "audio_num_attention_heads", self._AUDIO_NUM_HEADS
                )
            ),
            "audio_dit_num_heads": int(
                transformer_cfg.get(
                    "audio_num_attention_heads", self._AUDIO_NUM_HEADS
                )
            ),
            "audio_dit_num_layers": int(
                transformer_cfg.get(
                    "audio_num_layers", self._AUDIO_NUM_LAYERS
                )
            ),
            # Audio VAE
            "audio_z_dim": int(
                audio_vae_cfg.get("latent_channels", self._AUDIO_VAE_Z_DIM)
            ),
            "audio_scale_factor_temporal": int(
                audio_vae_cfg.get(
                    "temporal_compression_ratio",
                    self._AUDIO_VAE_SCALE_TEMPORAL,
                )
            ),
            "audio_mel_bins": int(
                audio_vae_cfg.get("mel_bins", self._AUDIO_VAE_MEL_BINS)
            ),
            # Vocoder
            "audio_sample_rate": int(
                vocoder_cfg.get(
                    "sample_rate", self._DEFAULT_AUDIO_SAMPLE_RATE
                )
            ),
            # Latent upsampler (optional; only populated when present)
            "latent_upsampler_input_dim": int(
                upsampler_cfg.get(
                    "input_channels", self._UPSAMPLER_INPUT_DIM
                )
            ),
            "latent_upsampler_output_dim": int(
                upsampler_cfg.get(
                    "output_channels", self._UPSAMPLER_OUTPUT_DIM
                )
            ),
            # Scheduler
            "flow_shift": float(scheduler_cfg.get("shift", 1.0)),
            "use_dynamic_shifting": int(
                bool(scheduler_cfg.get("use_dynamic_shifting", True))
            ),
            "base_shift": float(scheduler_cfg.get("base_shift", 0.95)),
            "max_shift": float(scheduler_cfg.get("max_shift", 2.05)),
            "base_image_seq_len": int(
                scheduler_cfg.get("base_image_seq_len", 1024)
            ),
            "max_image_seq_len": int(
                scheduler_cfg.get("max_image_seq_len", 4096)
            ),
            "shift_terminal": float(scheduler_cfg.get("shift_terminal", 0.1)),
        }


def _load_ltx2_vae_latent_stats(
    vae_dir: Path,
) -> tuple[list[float] | None, list[float] | None]:
    """Best-effort extract per-channel ``latents_mean`` / ``latents_std``.

    Mirrors :mod:`ltx_video` so the C++ runtime can denormalise latents
    before VAE decode. Returns ``(None, None)`` if the safetensors blob
    does not expose the stats (which means denorm is handled inside the
    VAE engine itself).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None, None

    try:
        candidates = sorted(vae_dir.glob("*.safetensors"))
    except OSError:
        return None, None

    for path in candidates:
        try:
            with safe_open(path, framework="np", device="cpu") as reader:
                keys = set(reader.keys())
                if "latents_mean" not in keys or "latents_std" not in keys:
                    continue
                mean = reader.get_tensor("latents_mean").astype(
                    "float32"
                ).reshape(-1)
                std = reader.get_tensor("latents_std").astype(
                    "float32"
                ).reshape(-1)
                return mean.tolist(), std.tolist()
        except Exception:  # pragma: no cover - defensive
            continue
    return None, None


plugin = LTX2Plugin()
