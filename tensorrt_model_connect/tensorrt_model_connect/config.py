"""ModelConfig — parse HF config.json into a typed dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


_ELF_VARIANTS: dict[str, tuple[int, int, int]] = {
    "ELF-B": (12, 768, 12),
    "ELF-M": (24, 1056, 16),
    "ELF-L": (32, 1280, 16),
}


def _normalize_elf_variant(value: object) -> str:
    variant = str(value or "ELF-B").upper().replace("_", "-")
    return variant if variant in _ELF_VARIANTS else "ELF-B"


def _elf_yaml_to_config(raw: dict) -> dict:
    variant = _normalize_elf_variant(raw.get("model"))
    depth, hidden_size, num_heads = _ELF_VARIANTS[variant]
    converted = dict(raw)
    converted.update({
        "model_type": "elf",
        "model": variant,
        "elf_variant": variant,
        "hidden_size": int(raw.get("hidden_size") or raw.get("elf_hidden_size") or hidden_size),
        "num_hidden_layers": int(raw.get("depth") or raw.get("num_hidden_layers") or depth),
        "num_attention_heads": int(raw.get("num_heads") or raw.get("num_attention_heads") or num_heads),
        "max_position_embeddings": int(raw.get("max_length") or raw.get("max_position_embeddings") or 128),
        "text_encoder_dim": int(
            raw.get("text_encoder_dim")
            or raw.get("encoder_d_model")
            or raw.get("d_model")
            or 512
        ),
        "vocab_size": int(raw.get("vocab_size") or 0),
    })
    return converted


@dataclass
class ModelConfig:
    """Parsed model architecture from HF config.json."""

    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 8192
    hidden_act: str = ""

    # Explicit head_dim from config.json (0 = not set, fall back to computed).
    _head_dim: int = 0

    # Raw JSON dict for family-specific fields
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        if self._head_dim > 0:
            return self._head_dim
        if self.num_attention_heads <= 0:
            return 0
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @staticmethod
    def from_json(text: str) -> ModelConfig:
        d = json.loads(text)

        # VL models (e.g. Qwen3-VL) nest text model config under "text_config".
        # Merge text_config into top level so standard key lookup works.
        # Preserve top-level model_type and architectures (these identify the
        # VL model, not the text backbone).
        original_raw = d
        text_config = d.get("text_config")
        if text_config and isinstance(text_config, dict):
            top_model_type = d.get("model_type")
            top_architectures = d.get("architectures")
            merged = {**d, **text_config}
            if top_model_type:
                merged["model_type"] = top_model_type
            if top_architectures:
                merged["architectures"] = top_architectures
            d = merged

        # DeepSeek-VL-v2 models nest the language decoder config under
        # "language_config".  Merge into top level like text_config.
        if not d.get("hidden_size"):
            lang_config = d.get("language_config")
            if isinstance(lang_config, dict):
                top_model_type = d.get("model_type")
                top_architectures = d.get("architectures")
                top_vision_config = d.get("vision_config")
                merged = {**d, **lang_config}
                if top_model_type:
                    merged["model_type"] = top_model_type
                if top_architectures:
                    merged["architectures"] = top_architectures
                if top_vision_config:
                    merged["vision_config"] = top_vision_config
                d = merged

        # Eagle VLM / Nemotron VL models nest LLM config under "llm_config".
        # Merge into top level like text_config, preserving top-level
        # model_type, architectures, and vision_config.
        if not d.get("hidden_size"):
            llm_config = d.get("llm_config")
            if isinstance(llm_config, dict):
                top_model_type = d.get("model_type")
                top_architectures = d.get("architectures")
                top_vision_config = d.get("vision_config")
                merged = {**d, **llm_config}
                if top_model_type:
                    merged["model_type"] = top_model_type
                if top_architectures:
                    merged["architectures"] = top_architectures
                if top_vision_config:
                    merged["vision_config"] = top_vision_config
                d = merged

        # Omni models (e.g. Qwen3-Omni) nest the primary decoder config
        # under thinker_config.text_config. If top-level hidden_size is
        # still missing after the text_config merge above, look there.
        if not d.get("hidden_size"):
            thinker_cfg = d.get("thinker_config")
            if isinstance(thinker_cfg, dict):
                thinker_text = thinker_cfg.get("text_config")
                if isinstance(thinker_text, dict):
                    top_model_type = d.get("model_type")
                    top_architectures = d.get("architectures")
                    merged = {**d, **thinker_text}
                    if top_model_type:
                        merged["model_type"] = top_model_type
                    if top_architectures:
                        merged["architectures"] = top_architectures
                    # Also propagate vision_config from thinker_config
                    # so VL pipelines can find it.
                    if "vision_config" not in merged and "vision_config" in thinker_cfg:
                        merged["vision_config"] = thinker_cfg["vision_config"]
                    d = merged

        # Handle non-standard config key names:
        #   GPT-2: n_embd, n_head, n_layer, n_inner
        #   XGLM/Bloom: d_model, attention_heads, num_layers, ffn_dim
        #   DistilBERT: dim, n_heads, n_layers, hidden_dim
        hidden_size = (d.get("hidden_size", 0) or d.get("n_embd", 0)
                       or d.get("d_model", 0) or d.get("n_embed", 0)
                       or d.get("dim", 0))
        num_heads = (d.get("num_attention_heads", 0) or d.get("n_head", 0)
                     or d.get("attention_heads", 0) or d.get("num_heads", 0)
                     or d.get("n_heads", 0) or d.get("decoder_attention_heads", 0) or 1)
        num_layers = (d.get("num_hidden_layers", 0) or d.get("n_layer", 0)
                      or d.get("num_layers", 0) or d.get("n_layers", 0))
        intermediate = (d.get("intermediate_size", 0)
                        or d.get("n_inner", 0)
                        or d.get("ffn_dim", 0)
                        or d.get("hidden_dim", 0)
                        or hidden_size * 4)

        # Norm epsilon: try rms_norm_eps, then layer_norm_epsilon, then
        # layer_norm_eps, then norm_epsilon, then norm_eps (Nemotron).
        eps = (d.get("rms_norm_eps")
               or d.get("layer_norm_epsilon")
               or d.get("layer_norm_eps")
               or d.get("norm_epsilon")
               or d.get("norm_eps")
               or 1e-5)

        # rope_theta: check top-level first, then rope_parameters dict
        # (Llama-3.1 variants like Minitron/Nemotron-Nano store it there),
        # then rope_scaling dict (Qwen3 stores it there).
        rope_theta = d.get("rope_theta", None)
        if rope_theta is None:
            rope_params = d.get("rope_parameters")
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta", 10000.0)
            else:
                rope_scaling = d.get("rope_scaling")
                if isinstance(rope_scaling, dict):
                    rope_theta = rope_scaling.get("rope_theta", 10000.0)
                else:
                    rope_theta = 10000.0
        rope_theta = float(rope_theta)

        architecture = d.get("architecture", "")
        architectures = d.get("architectures", [])
        if not architectures and architecture:
            architectures = [architecture]

        return ModelConfig(
            model_type=d.get("model_type", "") or architecture,
            architectures=architectures,
            vocab_size=d.get("vocab_size", 0),
            hidden_size=hidden_size or d.get("num_features", 0),
            intermediate_size=intermediate,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=d.get("num_key_value_heads", num_heads),
            rms_norm_eps=eps,
            rope_theta=rope_theta,
            bos_token_id=d.get("bos_token_id", -1) or -1,
            eos_token_id=d.get("eos_token_id", -1) or -1,
            pad_token_id=d.get("pad_token_id", -1) or -1,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            max_position_embeddings=d.get("max_position_embeddings",
                                          d.get("n_positions", 8192)),
            hidden_act=d.get("hidden_act", "") or d.get("activation_function", ""),
            _head_dim=d.get("head_dim", 0),
            raw=original_raw,
        )

    @classmethod
    def create_tiny(cls, model_type: str, **overrides) -> "ModelConfig":
        """Create a minimal ModelConfig for testing (2 layers, hidden=16, vocab=32)."""
        defaults = {
            "model_type": model_type,
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 4, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "max_position_embeddings": 128,
        }
        defaults.update(overrides)
        return cls.from_json(json.dumps(defaults))

    @staticmethod
    def from_dir(model_dir: str | Path) -> ModelConfig:
        model_path = Path(model_dir)
        config_path = model_path / "config.json"
        if config_path.exists():
            return ModelConfig.from_json(config_path.read_text())

        yaml_candidates = [
            model_path / "config.yaml",
            model_path / "config.yml",
            *sorted(model_path.glob("*.yaml")),
            *sorted(model_path.glob("*.yml")),
        ]
        for yaml_path in yaml_candidates:
            if not yaml_path.exists():
                continue
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load ELF YAML configs") from exc
            data = yaml.safe_load(yaml_path.read_text()) or {}
            if not isinstance(data, dict):
                continue
            if str(data.get("model", "")).upper().replace("_", "-") in _ELF_VARIANTS:
                return ModelConfig.from_json(json.dumps(_elf_yaml_to_config(data)))

        return ModelConfig.from_json(config_path.read_text())
