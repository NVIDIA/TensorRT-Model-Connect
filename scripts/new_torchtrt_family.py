#!/usr/bin/env python3
"""Scaffold a new Torch-TRT model family plugin.

Downloads the model's config.json, detects architecture features, and generates
a plugin file in python/tensorrt_model_connect/engine_defs/torch_trt/families/<family>.py.

For most standard decoder-only models, the generated plugin will work without
modification — it uses AutoModelForCausalLM and the generic make_export_args().

Usage:
    python3 scripts/new_torchtrt_family.py \\
      --model-type llama \\
      --hf-repo meta-llama/Llama-3.2-1B \\
      --family-name llama

    python3 scripts/new_torchtrt_family.py \\
      --model-type phi3 \\
      --hf-repo microsoft/Phi-3-mini-4k-instruct \\
      --family-name phi
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

FAMILIES_DIR = (Path(__file__).resolve().parent.parent
                / "python" / "tensorrt_model_connect" / "engine_defs" / "torch_trt" / "families")


def fetch_config(hf_repo: str) -> dict:
    """Download config.json from a HuggingFace repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub required. Install: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    path = hf_hub_download(repo_id=hf_repo, filename="config.json")
    with open(path) as f:
        return json.load(f)


def detect_features(cfg: dict) -> dict:
    """Detect architecture features from config.json."""
    features = {}
    features["model_type"] = cfg.get("model_type", "")
    features["architectures"] = cfg.get("architectures", [])
    features["hidden_size"] = cfg.get("hidden_size", 0)
    features["num_attention_heads"] = cfg.get("num_attention_heads", 0)
    features["num_key_value_heads"] = cfg.get(
        "num_key_value_heads", cfg.get("num_attention_heads", 0))
    features["num_hidden_layers"] = cfg.get("num_hidden_layers", 0)
    features["vocab_size"] = cfg.get("vocab_size", 0)
    features["explicit_head_dim"] = "head_dim" in cfg
    features["tie_word_embeddings"] = cfg.get("tie_word_embeddings", False)

    nkv = features["num_key_value_heads"]
    nh = features["num_attention_heads"]
    features["is_gqa"] = nkv != nh if nkv and nh else False
    features["is_moe"] = "num_local_experts" in cfg
    features["has_rms_norm_eps"] = "rms_norm_eps" in cfg
    features["has_layer_norm_eps"] = "layer_norm_eps" in cfg
    features["sliding_window"] = cfg.get("sliding_window") is not None

    return features


def generate_plugin(family_name: str, model_type: str, features: dict,
                    hf_repo: str) -> str:
    """Generate plugin .py source code."""
    class_name = family_name.capitalize() + "TorchTrtPlugin"

    # Build annotation comments
    notes = []
    if features.get("explicit_head_dim"):
        notes.append("# NOTE: Explicit head_dim in config.json.")
    if features.get("tie_word_embeddings"):
        notes.append("# NOTE: Tied word embeddings.")
    if features.get("is_gqa"):
        notes.append(f"# NOTE: GQA — kv_heads={features['num_key_value_heads']} "
                     f"vs heads={features['num_attention_heads']}.")
    if features.get("sliding_window"):
        notes.append("# NOTE: Sliding window attention — may need export adjustments.")
    if features.get("is_moe"):
        notes.append("# NOTE: MoE architecture — torch.export may need special handling.")

    notes_block = "\n".join(notes) + "\n" if notes else ""

    match_expr = f'return model_type.lower().startswith("{model_type.lower()}")'

    # Check if trust_remote_code might be needed
    needs_trust = "custom" in str(features.get("architectures", [])).lower()
    trust_kwarg = "\n            trust_remote_code=True," if needs_trust else ""

    source = textwrap.dedent(f'''\
        """{family_name.capitalize()} family plugin for Torch-TRT — scaffolded from {hf_repo}."""

        from __future__ import annotations

        import torch

        from ..config import ModelConfig
        from ..cache_config import ExportArgs, make_export_args

        {notes_block}
        class {class_name}:
            name = "{family_name}"

            def matches(self, model_type: str) -> bool:
                {match_expr}

            def load_model(
                self,
                model_dir: str,
                config: ModelConfig,
                max_cache_length: int,
            ) -> torch.nn.Module:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    device_map="cuda",{trust_kwarg}
                )
                model.eval()
                return model

            def get_export_args(
                self,
                model: torch.nn.Module,
                config: ModelConfig,
                max_cache_length: int,
                *,
                precision: str = "fp16",
            ) -> ExportArgs:
                return make_export_args(config, max_cache_length, precision=precision)


        plugin = {class_name}()
    ''')

    return source


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new Torch-TRT model family plugin.")
    parser.add_argument("--model-type", required=True,
                        help="HF model_type string (e.g. llama, phi3)")
    parser.add_argument("--hf-repo", required=True,
                        help="HF repo ID (e.g. meta-llama/Llama-3.2-1B)")
    parser.add_argument("--family-name", required=True,
                        help="Plugin family name (e.g. llama, phi)")
    args = parser.parse_args()

    # 1. Fetch config
    print(f"Fetching config.json from {args.hf_repo} ...", file=sys.stderr)
    cfg = fetch_config(args.hf_repo)

    # 2. Detect features
    features = detect_features(cfg)
    print("Detected features:", file=sys.stderr)
    for k, v in sorted(features.items()):
        print(f"  {k}: {v}", file=sys.stderr)

    # 3. Generate plugin
    source = generate_plugin(args.family_name, args.model_type, features,
                             args.hf_repo)

    # 4. Write file
    out_path = FAMILIES_DIR / f"{args.family_name}.py"
    if out_path.exists():
        print(f"ERROR: {out_path} already exists. Remove first or choose "
              "a different --family-name.", file=sys.stderr)
        sys.exit(1)

    out_path.write_text(source)
    print(f"\nGenerated: {out_path}", file=sys.stderr)

    # 5. Next steps
    print(f"""
Next steps:
  1. Review the generated plugin: {out_path}
  2. Customize load_model() if the model needs trust_remote_code or special loading
  3. Customize get_export_args() if the model has non-standard inputs
  4. Validate with:
     ./scripts/validate_torchtrt_family.sh {args.hf_repo}
  5. Add E2E manifest:
     tests/torchtrt_e2e/models/<model-name>.json
""", file=sys.stderr)


if __name__ == "__main__":
    main()
