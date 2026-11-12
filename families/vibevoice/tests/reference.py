# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The official Transformers VibeVoice pipeline over the original checkpoint layout.

Transformers' ``VibeVoiceForConditionalGeneration`` reads a converted layout.
This module performs the conversion in memory with the same renaming and
config rewriting as Transformers' conversion script, so the reference runs the
exact tensors of the original checkpoint the family builds from. It also
creates small random checkpoints in the original layout for graph tests.

Provenance: ``_RENAMES`` and ``hf_config`` restate the key mapping and config
rewrite of ``src/transformers/models/vibevoice/convert_vibevoice_to_hf.py``
(huggingface/transformers commit 431e9e9d, Apache-2.0), which the Transformers
wheel does not ship. ``CHAT_TEMPLATE`` is the ``chat_template.jinja`` that this
script publishes with the converted checkpoint ``vibevoice/VibeVoice-1.5B-hf``
(revision d4cd82eb, MIT), byte for byte.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
GENERATION_DEFAULTS = {"guidance_scale": 1.3, "num_diffusion_steps": 10}

# Transformers' conversion script ships this template with the converted checkpoint
# (chat_template.jinja of vibevoice/VibeVoice-1.5B-hf); the reference processor uses it.
CHAT_TEMPLATE = (
    '{%- set system_prompt = system_prompt | default(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n'
    '") -%}\n'
    "{{ system_prompt -}}\n"
    '{%- set audio_bos_token = audio_bos_token | default("<|vision_start|>") %}\n'
    '{%- set audio_eos_token = audio_eos_token | default("<|vision_end|>") %}\n'
    '{%- set audio_token = audio_token | default("<|vision_pad|>") %}\n'
    '{%- set eos_token = eos_token | default("<|endoftext|>") %}\n'
    "{#- Training regime (`add_generation_prompt=False`): the last audio of the conversation is the target speech and is\n"
    '    placed after "Speech output:", followed by the audio EOS and text EOS tokens; any preceding audio is used as a\n'
    "    voice prompt. Generation regime (`add_generation_prompt=True`): all audios are voice prompts. #}\n"
    '{%- set ns = namespace(num_audio=0, num_seen=0, speakers_with_audio="") %}\n'
    "{%- for message in messages %}\n"
    "    {%- set ns.num_audio = ns.num_audio + (message['content'] | selectattr('type', 'equalto', 'audio') | list | length) %}\n"
    "{%- endfor %}\n"
    "{%- set has_target_audio = not add_generation_prompt and ns.num_audio > 0 %}\n"
    "{%- set num_voice_prompts = ns.num_audio - 1 if has_target_audio else ns.num_audio %}\n"
    "{%- for message in messages %}\n"
    "    {%- set role = message['role'] %}\n"
    "    {%- set audio_count = message['content'] | selectattr('type', 'equalto', 'audio') | list | length %}\n"
    "    {%- if audio_count > 0 and ns.num_seen < num_voice_prompts and role not in ns.speakers_with_audio %}\n"
    '        {%- set ns.speakers_with_audio = ns.speakers_with_audio + role + "," %}\n'
    "    {%- endif %}\n"
    "    {%- set ns.num_seen = ns.num_seen + audio_count %}\n"
    "{%- endfor %}\n"
    "\n"
    "{%- if ns.speakers_with_audio %}\n"
    '{{ " Voice input:\n'
    '" }}\n'
    "{%- for speaker in ns.speakers_with_audio.rstrip(',').split(',') %}\n"
    "{%- if speaker %}\n"
    ' Speaker {{ speaker }}:{{ audio_bos_token }}{{ audio_token }}{{ audio_eos_token }}{{ "\n'
    '" }}\n'
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- endif %}\n"
    ' Text input:{{ "\n'
    '" }}\n'
    "\n"
    "{%- for message in messages %}\n"
    "    {%- set role = message['role'] %}\n"
    "    {%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}\n"
    "    {%- for item in text_items %}\n"
    " Speaker {{ role }}: {{ item['text'] }}{{ \"\n"
    '" }}\n'
    "    {%- endfor %}\n"
    "{%- endfor %}\n"
    ' Speech output:{{ "\n'
    '" }}{{ audio_bos_token }}\n'
    "{%- if not add_generation_prompt %}\n"
    "{%- if has_target_audio %}{{ audio_token }}{{ audio_eos_token }}{% endif %}{{ eos_token }}\n"
    "{%- endif %}"
)

_RENAMES = (
    (
        r"semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.",
        "semantic_tokenizer_encoder.stem.conv.conv.",
    ),
    (r"semantic_tokenizer\.encoder\.stages\.0\.", "semantic_tokenizer_encoder.stem.stage."),
    (
        r"semantic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.",
        "semantic_tokenizer_encoder.conv_layers.{}.conv.conv.",
    ),
    (
        r"semantic_tokenizer\.encoder\.stages\.(\d+)\.",
        "semantic_tokenizer_encoder.conv_layers.{}.stage.",
    ),
    (r"semantic_tokenizer\.encoder\.head\.conv\.", "semantic_tokenizer_encoder.head."),
    (
        r"acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.",
        "audio_tower.encoder.stem.conv.conv.",
    ),
    (r"acoustic_tokenizer\.encoder\.stages\.0\.", "audio_tower.encoder.stem.stage."),
    (
        r"acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.",
        "audio_tower.encoder.conv_layers.{}.conv.conv.",
    ),
    (r"acoustic_tokenizer\.encoder\.stages\.(\d+)\.", "audio_tower.encoder.conv_layers.{}.stage."),
    (r"acoustic_tokenizer\.encoder\.head\.conv\.", "audio_tower.encoder.head."),
    (
        r"acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.",
        "audio_tower.decoder.stem.conv.conv.",
    ),
    (r"acoustic_tokenizer\.decoder\.stages\.0\.", "audio_tower.decoder.stem.stage."),
    (
        r"acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.",
        "audio_tower.decoder.conv_layers.{}.convtr.convtr.",
    ),
    (r"acoustic_tokenizer\.decoder\.stages\.(\d+)\.", "audio_tower.decoder.conv_layers.{}.stage."),
    (r"acoustic_tokenizer\.decoder\.head\.conv\.", "audio_tower.decoder.head."),
    (r"acoustic_tokenizer\.", "audio_tower."),
    (r"prediction_head\.t_embedder\.mlp\.0\.", "diffusion_head.timestep_proj.fc1."),
    (r"prediction_head\.t_embedder\.mlp\.2\.", "diffusion_head.timestep_proj.fc2."),
    (r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\.", "diffusion_head.layers.{0}.linear."),
    (
        r"prediction_head\.final_layer\.adaLN_modulation\.1\.",
        "diffusion_head.final_layer.linear_1.",
    ),
    (r"prediction_head\.final_layer\.linear\.", "diffusion_head.final_layer.linear_2."),
    (r"prediction_head\.", "diffusion_head."),
    (r"acoustic_connector\.fc1\.", "multi_modal_projector.linear_1."),
    (r"acoustic_connector\.norm\.", "multi_modal_projector.act."),
    (r"acoustic_connector\.fc2\.", "multi_modal_projector.linear_2."),
    (r"semantic_connector\.fc1\.", "semantic_connector.linear_1."),
    (r"semantic_connector\.norm\.", "semantic_connector.act."),
    (r"semantic_connector\.fc2\.", "semantic_connector.linear_2."),
    (r"^model\.speech_scaling_factor", "model.latent_scaling_factor"),
    (r"^model\.speech_bias_factor", "model.latent_bias_factor"),
    (r"mixer\.conv\.conv\.conv\.", "mixer.conv."),
    (r"\.conv\.conv\.conv\.", ".conv.conv."),
)


def converted_name(name: str) -> str:
    """Rename one original tensor the way Transformers' conversion script does."""
    for pattern, replacement in _RENAMES:
        match = re.search(pattern, name)
        if not match:
            continue
        if "{}" in replacement:
            # downsample/upsample layer N becomes conv layer N-1 (layer 0 is the stem).
            target = replacement.replace("{}", str(int(match.group(1)) - 1))
        elif "{0}" in replacement:
            target = replacement.replace("{0}", match.group(1))
        else:
            target = replacement
        name = name[: match.start()] + target + name[match.end() :]
    return name


def _tokenizer_config(raw: dict) -> dict:
    out = dict(raw)
    out["depths"] = [int(item) for item in out.pop("encoder_depths").split("-")]
    out["rms_norm_eps"] = out.pop("layernorm_eps")
    out["downsampling_ratios"] = list(reversed(out.pop("encoder_ratios")))
    out["num_filters"] = out.pop("encoder_n_filters")
    out["hidden_size"] = out.pop("vae_dim")
    for key in (
        "decoder_depths",
        "decoder_n_filters",
        "decoder_ratios",
        "std_dist_type",
        "fix_std",
        "pad_mode",
        "conv_bias",
        "causal",
        "mixer_layer",
        "layernorm",
        "disable_last_norm",
        "conv_norm",
        "corpus_normalize",
        "layernorm_elementwise_affine",
    ):
        out.pop(key, None)
    return out


def hf_config(original: dict):
    """The VibeVoiceConfig Transformers' conversion script builds from config.json."""
    from transformers import (
        VibeVoiceAcousticTokenizerConfig,
        VibeVoiceAcousticTokenizerEncoderConfig,
        VibeVoiceConfig,
    )

    raw = json.loads(json.dumps(original))
    semantic = _tokenizer_config(raw.pop("semantic_tokenizer_config"))
    acoustic = _tokenizer_config(raw.pop("acoustic_tokenizer_config"))
    head = raw.pop("diffusion_head_config")
    raw["diffusion_head_config"] = {
        "hidden_size": head["hidden_size"],
        "latent_size": acoustic["hidden_size"],
        "num_hidden_layers": head["head_layers"],
        "intermediate_size": int(head["head_ffn_ratio"] * head["hidden_size"]),
        "rms_norm_eps": head["rms_norm_eps"],
    }
    raw["text_config"] = raw.pop("decoder_config")
    raw["text_config"]["dtype"] = raw["text_config"].pop("torch_dtype")
    raw["dtype"] = raw.pop("torch_dtype", None)
    for key in (
        "acoustic_vae_dim",
        "semantic_vae_dim",
        "tie_word_embeddings",
        "transformers_version",
    ):
        raw.pop(key, None)
    raw["audio_config"] = VibeVoiceAcousticTokenizerConfig(**acoustic).to_dict()
    raw["semantic_model_config"] = VibeVoiceAcousticTokenizerEncoderConfig(**semantic).to_dict()
    return VibeVoiceConfig(**raw)


def load_reference(model_dir: str | Path, dtype, device: str):
    """``VibeVoiceForConditionalGeneration`` holding the original checkpoint's tensors."""
    import torch
    from safetensors.torch import load_file
    from transformers import VibeVoiceForConditionalGeneration

    root = Path(model_dir)
    config = hf_config(json.loads((root / "config.json").read_text(encoding="utf-8")))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        for name, tensor in load_file(str(root / shard)).items():
            state[converted_name(name)] = tensor
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    # Instantiate the way from_pretrained(dtype=...) does: parameters in `dtype`, while the
    # buffers the modules create in float32 (rotary and diffusion-timestep frequencies) stay
    # float32. A blanket .to(dtype) would round those buffers and perturb the reference.
    with torch.device(device):
        model = VibeVoiceForConditionalGeneration._from_config(config, dtype=dtype)
    model.load_state_dict(state, strict=True)
    frequencies = {
        name: buffer.dtype
        for name, buffer in model.named_buffers()
        if name.endswith(("inv_freq", ".freq"))
    }
    if not frequencies or any(kind != torch.float32 for kind in frequencies.values()):
        raise AssertionError(f"reference frequency buffers are not float32: {frequencies}")
    model.generation_config.do_sample = False
    model.generation_config.guidance_scale = GENERATION_DEFAULTS["guidance_scale"]
    model.generation_config.num_diffusion_steps = GENERATION_DEFAULTS["num_diffusion_steps"]
    model.generation_config.noise_scheduler_class = "DPMSolverMultistepScheduler"
    model.generation_config.noise_scheduler_config = {
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "v_prediction",
    }
    model.generation_config.eos_token_id = 151643
    model.generation_config.pad_token_id = 151643
    model.generation_config.max_new_tokens = 40500
    return model.eval()


def reference_processor(tokenizer_dir: str | Path, hop_length: int):
    from transformers import Qwen2TokenizerFast, VibeVoiceProcessor
    from transformers.models.vibevoice_acoustic_tokenizer import (
        VibeVoiceAcousticTokenizerFeatureExtractor,
    )

    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_dir))
    extractor = VibeVoiceAcousticTokenizerFeatureExtractor(
        sampling_rate=SAMPLE_RATE,
        normalize_audio=True,
        target_dB_FS=-25,
        eps=1e-6,
        pad_to_multiple_of=hop_length,
    )
    return VibeVoiceProcessor(
        feature_extractor=extractor, tokenizer=tokenizer, chat_template=CHAT_TEMPLATE
    )


# ---------------------------------------------------------------------------
# Small random checkpoints in the original layout.
# ---------------------------------------------------------------------------


def tiny_config() -> tuple[dict, dict]:
    """config.json and preprocessor_config.json of a small model with every structural feature."""
    tokenizer = {
        "causal": True,
        "channels": 1,
        "conv_bias": True,
        "conv_norm": "none",
        "corpus_normalize": 0.0,
        "disable_last_norm": True,
        "encoder_depths": "1-1-1-2",
        "encoder_n_filters": 4,
        "encoder_ratios": [4, 2, 2],
        "layer_scale_init_value": 1e-06,
        "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": True,
        "layernorm_eps": 1e-05,
        "mixer_layer": "depthwise_conv",
        "pad_mode": "constant",
        "weight_init_value": 0.01,
    }
    acoustic = dict(
        tokenizer,
        decoder_depths=None,
        decoder_n_filters=4,
        decoder_ratios=[4, 2, 2],
        fix_std=0.5,
        model_type="vibevoice_acoustic_tokenizer",
        std_dist_type="gaussian",
        vae_dim=8,
    )
    semantic = dict(
        tokenizer,
        fix_std=0,
        model_type="vibevoice_semantic_tokenizer",
        std_dist_type="none",
        vae_dim=6,
    )
    config = {
        "acoustic_vae_dim": 8,
        "acoustic_tokenizer_config": acoustic,
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "decoder_config": {
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 160,
            "max_position_embeddings": 4096,
            "max_window_layers": 2,
            "model_type": "qwen2",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        },
        "diffusion_head_config": {
            "ddpm_batch_mul": 4,
            "ddpm_beta_schedule": "cosine",
            "ddpm_num_inference_steps": 20,
            "ddpm_num_steps": 1000,
            "diffusion_type": "ddpm",
            "head_ffn_ratio": 3.0,
            "head_layers": 2,
            "hidden_size": 64,
            "latent_size": 8,
            "model_type": "vibevoice_diffusion_head",
            "prediction_type": "v_prediction",
            "rms_norm_eps": 1e-05,
            "speech_vae_dim": 8,
        },
        "model_type": "vibevoice",
        "semantic_tokenizer_config": semantic,
        "semantic_vae_dim": 6,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.3",
    }
    processor = {
        "processor_class": "VibeVoiceProcessor",
        "speech_tok_compress_ratio": 16,
        "db_normalize": True,
        "audio_processor": {
            "feature_extractor_type": "VibeVoiceTokenizerProcessor",
            "sampling_rate": 24000,
            "normalize_audio": True,
            "target_dB_FS": -25,
            "eps": 1e-06,
        },
        "language_model_pretrained_name": "Qwen/Qwen2.5-1.5B",
    }
    return config, processor


def original_tensor_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Names and shapes of every tensor of an original-layout checkpoint."""
    shapes: dict[str, tuple[int, ...]] = {}
    lm = config["decoder_config"]
    hidden, inter = lm["hidden_size"], lm["intermediate_size"]
    heads, kv_heads = lm["num_attention_heads"], lm["num_key_value_heads"]
    head_dim = hidden // heads
    base = "model.language_model."
    shapes[base + "embed_tokens.weight"] = (lm["vocab_size"], hidden)
    shapes[base + "norm.weight"] = (hidden,)
    for layer in range(lm["num_hidden_layers"]):
        prefix = f"{base}layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
        shapes[prefix + "self_attn.q_proj.weight"] = (heads * head_dim, hidden)
        shapes[prefix + "self_attn.q_proj.bias"] = (heads * head_dim,)
        shapes[prefix + "self_attn.k_proj.weight"] = (kv_heads * head_dim, hidden)
        shapes[prefix + "self_attn.k_proj.bias"] = (kv_heads * head_dim,)
        shapes[prefix + "self_attn.v_proj.weight"] = (kv_heads * head_dim, hidden)
        shapes[prefix + "self_attn.v_proj.bias"] = (kv_heads * head_dim,)
        shapes[prefix + "self_attn.o_proj.weight"] = (hidden, heads * head_dim)
        shapes[prefix + "mlp.gate_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.up_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, inter)

    def block(prefix: str, channels: int, expansion: int = 4) -> None:
        shapes[prefix + "norm.weight"] = (channels,)
        shapes[prefix + "ffn_norm.weight"] = (channels,)
        shapes[prefix + "gamma"] = (channels,)
        shapes[prefix + "ffn_gamma"] = (channels,)
        shapes[prefix + "mixer.conv.conv.conv.weight"] = (channels, 1, 7)
        shapes[prefix + "mixer.conv.conv.conv.bias"] = (channels,)
        shapes[prefix + "ffn.linear1.weight"] = (expansion * channels, channels)
        shapes[prefix + "ffn.linear1.bias"] = (expansion * channels,)
        shapes[prefix + "ffn.linear2.weight"] = (channels, expansion * channels)
        shapes[prefix + "ffn.linear2.bias"] = (channels,)

    def encoder(prefix: str, raw: dict) -> None:
        depths = [int(item) for item in raw["encoder_depths"].split("-")]
        ratios = list(reversed(raw["encoder_ratios"]))
        filters = raw["encoder_n_filters"]
        channels = [filters * 2**index for index in range(len(depths))]
        shapes[prefix + "downsample_layers.0.0.conv.conv.weight"] = (channels[0], 1, 7)
        shapes[prefix + "downsample_layers.0.0.conv.conv.bias"] = (channels[0],)
        for index, ratio in enumerate(ratios):
            shapes[f"{prefix}downsample_layers.{index + 1}.0.conv.conv.weight"] = (
                channels[index + 1],
                channels[index],
                2 * ratio,
            )
            shapes[f"{prefix}downsample_layers.{index + 1}.0.conv.conv.bias"] = (
                channels[index + 1],
            )
        for stage, depth in enumerate(depths):
            for item in range(depth):
                block(f"{prefix}stages.{stage}.{item}.", channels[stage])
        shapes[prefix + "head.conv.conv.weight"] = (raw["vae_dim"], channels[-1], 7)
        shapes[prefix + "head.conv.conv.bias"] = (raw["vae_dim"],)

    def decoder(prefix: str, raw: dict) -> None:
        depths = list(reversed([int(item) for item in raw["encoder_depths"].split("-")]))
        ratios = raw["encoder_ratios"]
        filters = raw["encoder_n_filters"]
        channels = list(reversed([filters * 2**index for index in range(len(depths))]))
        shapes[prefix + "upsample_layers.0.0.conv.conv.weight"] = (channels[0], raw["vae_dim"], 7)
        shapes[prefix + "upsample_layers.0.0.conv.conv.bias"] = (channels[0],)
        for index, ratio in enumerate(ratios):
            shapes[f"{prefix}upsample_layers.{index + 1}.0.convtr.convtr.weight"] = (
                channels[index],
                channels[index + 1],
                2 * ratio,
            )
            shapes[f"{prefix}upsample_layers.{index + 1}.0.convtr.convtr.bias"] = (
                channels[index + 1],
            )
        for stage, depth in enumerate(depths):
            for item in range(depth):
                block(f"{prefix}stages.{stage}.{item}.", channels[stage])
        shapes[prefix + "head.conv.conv.weight"] = (1, channels[-1], 7)
        shapes[prefix + "head.conv.conv.bias"] = (1,)

    encoder("model.acoustic_tokenizer.encoder.", config["acoustic_tokenizer_config"])
    decoder("model.acoustic_tokenizer.decoder.", config["acoustic_tokenizer_config"])
    encoder("model.semantic_tokenizer.encoder.", config["semantic_tokenizer_config"])

    head = config["diffusion_head_config"]
    width, latent = head["hidden_size"], head["latent_size"]
    head_inter = int(head["head_ffn_ratio"] * width)
    prefix = "model.prediction_head."
    shapes[prefix + "noisy_images_proj.weight"] = (width, latent)
    shapes[prefix + "cond_proj.weight"] = (width, width)
    shapes[prefix + "t_embedder.mlp.0.weight"] = (width, 256)
    shapes[prefix + "t_embedder.mlp.2.weight"] = (width, width)
    for layer in range(head["head_layers"]):
        shapes[f"{prefix}layers.{layer}.adaLN_modulation.1.weight"] = (3 * width, width)
        shapes[f"{prefix}layers.{layer}.norm.weight"] = (width,)
        shapes[f"{prefix}layers.{layer}.ffn.gate_proj.weight"] = (head_inter, width)
        shapes[f"{prefix}layers.{layer}.ffn.up_proj.weight"] = (head_inter, width)
        shapes[f"{prefix}layers.{layer}.ffn.down_proj.weight"] = (width, head_inter)
    shapes[prefix + "final_layer.adaLN_modulation.1.weight"] = (2 * width, width)
    shapes[prefix + "final_layer.linear.weight"] = (latent, width)
    for name, inputs in (
        ("acoustic_connector", config["acoustic_vae_dim"]),
        ("semantic_connector", config["semantic_vae_dim"]),
    ):
        shapes[f"model.{name}.fc1.weight"] = (hidden, inputs)
        shapes[f"model.{name}.fc1.bias"] = (hidden,)
        shapes[f"model.{name}.norm.weight"] = (hidden,)
        shapes[f"model.{name}.fc2.weight"] = (hidden, hidden)
        shapes[f"model.{name}.fc2.bias"] = (hidden,)
    shapes["model.speech_scaling_factor"] = ()
    shapes["model.speech_bias_factor"] = ()
    return shapes


def write_tiny_checkpoint(directory: str | Path, *, seed: int = 0) -> Path:
    """A random original-layout checkpoint whose activations stay O(1) through every tower."""
    import ml_dtypes
    from safetensors.numpy import save_file

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    config, processor = tiny_config()
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, shape in original_tensor_shapes(config).items():
        if name.endswith(("norm.weight", "_layernorm.weight")):
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith(("gamma", "ffn_gamma")):
            value = 0.5 + 0.1 * rng.standard_normal(shape)
        elif name == "model.speech_scaling_factor":
            value = np.asarray(0.75)
        elif name == "model.speech_bias_factor":
            value = np.asarray(-0.125)
        elif name.endswith("bias"):
            value = 0.05 * rng.standard_normal(shape)
        else:
            fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            value = rng.standard_normal(shape) / math.sqrt(max(fan_in, 1))
        tensors[name] = np.asarray(value, dtype=np.float32).astype(ml_dtypes.bfloat16)
    save_file(tensors, str(root / "model.safetensors"))
    index = {
        "metadata": {"total_size": int(sum(value.nbytes for value in tensors.values()))},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=1), encoding="utf-8"
    )
    (root / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    (root / "preprocessor_config.json").write_text(
        json.dumps(processor, indent=1), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# The family's noise stream, for feeding a reference run the same noise.
# ---------------------------------------------------------------------------

_MASK = (1 << 64) - 1


class NoiseStream:
    """SplitMix64 + Box-Muller in double precision, as runtime/noise.h draws them."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK
        self.spare = None

    def _next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
        return value ^ (value >> 31)

    def _uniform(self) -> float:
        return ((self._next() >> 11) + 1) * (1.0 / 9007199254740992.0)

    def normal(self) -> float:
        if self.spare is not None:
            value, self.spare = self.spare, None
            return float(np.float32(value))
        radius = math.sqrt(-2.0 * math.log(self._uniform()))
        angle = 2.0 * 3.141592653589793 * self._uniform()
        self.spare = radius * math.sin(angle)
        return float(np.float32(radius * math.cos(angle)))

    def normals(self, count: int) -> np.ndarray:
        return np.asarray([self.normal() for _ in range(count)], dtype=np.float32)
