#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Wan2.1-T2V-14B bundle with configurable frame count.

Usage (inside container):
    /opt/venv/bin/python -m tensorrt_model_connect.models.wan_t2v.build_wan14b --frames 33 \
        -o ./engines/wan21-14b-33fr.bundle
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build Wan2.1-T2V-14B bundle")
    parser.add_argument("--model-id",
                        default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1. Download model
    from tensorrt_model_connect.engine_builder import _resolve_model
    model_dir = _resolve_model(args.model_id)

    # 2. Read the actual DiT config from HF
    dit_cfg = json.loads(
        (Path(model_dir) / "transformer" / "config.json").read_text())
    dit_dim = dit_cfg["attention_head_dim"] * dit_cfg["num_attention_heads"]
    print(f"[14b] DiT: dim={dit_dim}, heads={dit_cfg['num_attention_heads']}, "
          f"layers={dit_cfg['num_layers']}, ffn={dit_cfg['ffn_dim']}",
          file=sys.stderr)

    # 3. Override plugin class variables for 14B
    from tensorrt_model_connect.models.wan_t2v import model as wan_model
    wan_model._DIT_DIM = dit_dim
    wan_model._DIT_NUM_HEADS = dit_cfg["num_attention_heads"]
    wan_model._DIT_NUM_LAYERS = dit_cfg["num_layers"]
    wan_model._DIT_FFN_DIM = dit_cfg["ffn_dim"]
    wan_model._DIT_CONTEXT_DIM = dit_cfg["text_dim"]

    # 4. Inject video dimensions into the model_index.json (config.raw)
    model_index_path = Path(model_dir) / "model_index.json"
    model_index = json.loads(model_index_path.read_text())
    model_index["video_height"] = args.height
    model_index["video_width"] = args.width
    model_index["video_num_frames"] = args.frames

    # Write modified model_index temporarily so the builder picks it up
    model_index_path.write_text(json.dumps(model_index, indent=2))

    # 5. Build using the standard pipeline
    from tensorrt_model_connect.engine_builder import build

    build(str(model_dir), args.output, verbose=args.verbose)

    print("\n[14b] Done! Run with:", file=sys.stderr)
    print(f"  ./build/trtmc generate-video {args.output} "
          f"--prompt 'A cat walking in the garden' "
          f"--num-steps 30 --hf-python /opt/venv/bin/python",
          file=sys.stderr)


if __name__ == "__main__":
    main()
