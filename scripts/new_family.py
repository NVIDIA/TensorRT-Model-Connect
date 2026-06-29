#!/usr/bin/env python3
"""Scaffold a new model family plugin.

Downloads the model's config.json, detects architecture features, and generates
a flat family package in
python/tensorrt_model_connect/families/<family>/.

Usage:
    python3 scripts/new_family.py \
      --model-type phi3 \
      --hf-repo example-org/example-decoder \
      --family-name phi

    python3 scripts/new_family.py \
      --model-type yi \
      --hf-repo 01-ai/Yi-6B \
      --family-name yi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAMILIES_DIR = Path(__file__).resolve().parent.parent / "python" / "tensorrt_model_connect" / "families"


def fetch_config(hf_repo: str) -> dict:
    """Download config.json from a HuggingFace repo and return as dict."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    path = hf_hub_download(repo_id=hf_repo, filename="config.json")
    with open(path) as f:
        return json.load(f)


def detect_features(cfg: dict) -> dict[str, bool | int | float | str]:
    """Detect architecture features from config.json."""
    features: dict[str, bool | int | float | str] = {}

    features["model_type"] = cfg.get("model_type", "")
    features["architectures"] = cfg.get("architectures", [])
    features["hidden_size"] = cfg.get("hidden_size", 0)
    features["num_attention_heads"] = cfg.get("num_attention_heads", 0)
    features["num_key_value_heads"] = cfg.get("num_key_value_heads",
                                               cfg.get("num_attention_heads", 0))
    features["num_hidden_layers"] = cfg.get("num_hidden_layers", 0)
    features["vocab_size"] = cfg.get("vocab_size", 0)

    # Explicit head_dim (non-standard decoder variants)
    features["explicit_head_dim"] = "head_dim" in cfg

    # Tied embeddings
    features["tie_word_embeddings"] = cfg.get("tie_word_embeddings", False)

    # GQA (grouped query attention)
    nkv = features["num_key_value_heads"]
    nh = features["num_attention_heads"]
    features["is_gqa"] = nkv != nh if nkv and nh else False

    # RoPE theta
    features["rope_theta"] = cfg.get("rope_theta", 10000.0)

    # Norm type hints
    features["has_rms_norm_eps"] = "rms_norm_eps" in cfg
    features["has_layer_norm_eps"] = "layer_norm_eps" in cfg

    # Sliding window attention
    features["sliding_window"] = cfg.get("sliding_window", None) is not None

    # MoE hints
    features["is_moe"] = "num_local_experts" in cfg

    return features


def generate_plugin(family_name: str, model_type: str, features: dict,
                    hf_repo: str) -> str:
    """Generate plugin .py source code."""
    class_name = family_name.capitalize() + "Plugin"

    # Build comments about detected features
    notes = []
    if features.get("explicit_head_dim"):
        notes.append("# NOTE: This model has explicit head_dim in config.json.")
    if features.get("tie_word_embeddings"):
        notes.append("# NOTE: Tied word embeddings (lm_head reuses embedding).")
    if features.get("is_gqa"):
        notes.append(f"# NOTE: GQA — num_kv_heads={features['num_key_value_heads']} "
                     f"vs num_heads={features['num_attention_heads']}.")
    if features.get("sliding_window"):
        notes.append("# NOTE: Sliding window attention detected — may need custom graph builder.")
    if features.get("is_moe"):
        notes.append("# NOTE: MoE architecture detected — will need custom graph builder.")
    if features.get("has_layer_norm_eps") and not features.get("has_rms_norm_eps"):
        notes.append("# NOTE: Uses LayerNorm (not RMSNorm) — may need custom graph builder.")

    notes_block = "\n".join(notes) + "\n" if notes else ""

    # Determine if standard or needs customization
    needs_custom = features.get("is_moe") or (
        features.get("has_layer_norm_eps") and not features.get("has_rms_norm_eps")
    )

    if needs_custom:
        custom_warning = (
            "# WARNING: This model has non-standard architecture features.\n"
            "# The generated plugin uses the standard decoder builder as a starting point,\n"
            "# but you will likely need to customize load_weights() and/or build_engine().\n"
        )
    else:
        custom_warning = ""

    # Build matches() body
    match_expr = f'return model_type.lower().startswith("{model_type.lower()}")'

    # Build source directly — no textwrap.dedent to avoid indentation issues
    # with interpolated multi-line blocks.
    lines = [
        f'"""{family_name.capitalize()} family plugin — scaffolded from {hf_repo}."""',
        "",
        "from __future__ import annotations",
        "",
        "from .config import ModelConfig",
        "from .checkpoint_mapper import WeightDict, load_standard_weights",
        "from .standard_decoder_builder import build_standard_decoder_engine",
        "",
    ]
    if notes_block:
        lines.append(notes_block.rstrip())
    if custom_warning:
        lines.append(custom_warning.rstrip())
    lines.append("")
    lines += [
        f"class {class_name}:",
        f'    name = "{family_name}"',
        "",
        "    def matches(self, model_type: str) -> bool:",
        f"        {match_expr}",
        "",
        "    def load_weights(",
        "        self, model_dir: str, config: ModelConfig,",
        "    ) -> WeightDict:",
        "        return load_standard_weights(model_dir, config)",
        "",
        "    def build_engine(",
        "        self, config: ModelConfig, weights: WeightDict,",
        "        max_cache_length: int, *, verbose: bool = False,",
        "    ) -> bytes:",
        "        return build_standard_decoder_engine(",
        "            config, weights, max_cache_length, verbose=verbose)",
        "",
        "",
        f"plugin = {class_name}()",
        "",
    ]
    source = "\n".join(lines)

    return source


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new model family plugin for tensorrt_model_connect.")
    parser.add_argument("--model-type", required=True,
                        help="HF model_type string (e.g. phi3, yi, starcoder2)")
    parser.add_argument("--hf-repo", required=True,
                        help="HuggingFace repo ID (e.g. example-org/example-decoder)")
    parser.add_argument("--family-name", required=True,
                        help="Plugin family name (e.g. phi, yi)")
    args = parser.parse_args()

    # 1. Fetch config.json
    print(f"Fetching config.json from {args.hf_repo} ...", file=sys.stderr)
    cfg = fetch_config(args.hf_repo)

    # 2. Detect features
    features = detect_features(cfg)
    print("Detected features:", file=sys.stderr)
    for k, v in sorted(features.items()):
        if isinstance(v, list):
            print(f"  {k}: {v}", file=sys.stderr)
        else:
            print(f"  {k}: {v}", file=sys.stderr)

    # 3. Generate plugin
    source = generate_plugin(args.family_name, args.model_type, features,
                             args.hf_repo)

    # 4. Write file
    out_dir = FAMILIES_DIR / args.family_name
    out_path = out_dir / "plugin.py"
    init_path = out_dir / "__init__.py"
    if out_dir.exists():
        print(f"ERROR: {out_dir} already exists. Remove it first or choose a "
              "different --family-name.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir()
    out_path.write_text(source)
    init_path.write_text(
        f'"""{args.family_name} family package."""\n\n'
        "from __future__ import annotations\n\n"
        "import sys\n"
        "import types\n\n"
        "from . import plugin as _plugin\n\n"
        "globals().update({\n"
        "    _name: _value\n"
        "    for _name, _value in vars(_plugin).items()\n"
        "    if not _name.startswith(\"__\")\n"
        "})\n\n"
        "__all__ = [\n"
        "    _name for _name in globals()\n"
        "    if not _name.startswith(\"__\") and _name != \"_plugin\"\n"
        "]\n\n\n"
        "class _FamilyModule(types.ModuleType):\n"
        "    def __setattr__(self, name, value):\n"
        "        super().__setattr__(name, value)\n"
        "        if not name.startswith(\"__\") and name != \"_plugin\":\n"
        "            setattr(_plugin, name, value)\n\n\n"
        "sys.modules[__name__].__class__ = _FamilyModule\n"
    )
    print(f"\nGenerated: {out_path}", file=sys.stderr)

    # 5. Print next steps
    print(f"""
Next steps:
  1. Review the generated plugin: {out_path}
  2. Customize load_weights() if the model has non-standard weight naming
  3. Customize build_engine() if the model has a non-standard architecture
  4. Validate with:
     ./scripts/validate_family.sh {args.hf_repo}
""", file=sys.stderr)


if __name__ == "__main__":
    main()
