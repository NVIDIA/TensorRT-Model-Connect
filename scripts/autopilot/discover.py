#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover unsupported HuggingFace model families.

Queries HuggingFace Hub for popular transformer models, extracts unique
model_type values, and diffs against existing family plugin coverage.

Run inside a container (needs huggingface_hub + tensorrt_model_connect):
    docker exec trtmc-dev-gb300-agent-1 python3 scripts/autopilot/discover.py

Or from host if deps are available:
    python3 scripts/autopilot/discover.py --min-downloads 50000 --output gaps.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def get_supported_model_types() -> set[str]:
    """Import find_plugin and test a wide set of known model_type strings.

    Falls back to scanning source files if tensorrt_model_connect is not importable.
    """
    # Try the import path first (works inside containers)
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / "python"))
    try:
        from tensorrt_model_connect.families import (
            family_probe_model_types,
            find_plugin,
        )
        return _probe_with_find_plugin(find_plugin, family_probe_model_types())
    except ImportError:
        pass

    # Fallback: scan family metadata directly.
    return _scan_plugin_sources(project_root)


def _probe_with_find_plugin(find_plugin, seed_model_types: list[str]) -> set[str]:
    """Probe find_plugin with metadata-declared and previously discovered types."""
    supported = set()

    # Load the known types cache if it exists
    cache_path = Path(__file__).parent / ".known_model_types.json"
    probe_types: set[str] = set()

    if cache_path.exists():
        probe_types.update(json.loads(cache_path.read_text()))

    probe_types.update(seed_model_types)

    for t in probe_types:
        if find_plugin(t) is not None:
            supported.add(t)

    return supported


def _scan_plugin_sources(project_root: Path) -> set[str]:
    """Extract supported model_types by scanning family MODEL.toml metadata."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        tomllib = None

    families_dir = project_root / "python" / "tensorrt_model_connect" / "families"
    supported = set()

    for model_toml in families_dir.glob("*/MODEL.toml"):
        if tomllib is None:
            continue
        raw = tomllib.loads(model_toml.read_text(encoding="utf-8"))
        for value in (
            raw.get("aliases", [])
            + raw.get("prefixes", [])
            + [raw.get("id", ""), raw.get("plugin", ""), model_toml.parent.name]
        ):
            if isinstance(value, str) and value:
                supported.add(value.lower().replace("-", "_").replace(".", "_"))

    return supported


def query_hf_model_types(
    min_downloads: int = 10000,
    max_models: int = 2000,
    workers: int = 16,
) -> dict[str, dict]:
    """Query HF Hub for model_types with download counts.

    Returns {model_type: {"downloads": total, "count": num_models,
                          "representative": smallest_popular_model}}.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Querying HuggingFace for top {max_models} transformer models...",
          file=sys.stderr)
    models = list(api.list_models(
        filter="transformers",
        sort="downloads",
        limit=max_models,
        fetch_config=True,
    ))
    print(f"  Got {len(models)} models.", file=sys.stderr)

    # Extract model_type directly from config (fetch_config=True provides it)
    type_info: dict[str, dict] = defaultdict(
        lambda: {"downloads": 0, "count": 0, "models": []})

    for model_info in models:
        try:
            cfg = model_info.config or {}
            if isinstance(cfg, dict):
                model_type = cfg.get("model_type", "")
            else:
                continue
            if not model_type:
                continue

            downloads = getattr(model_info, "downloads", 0) or 0
            # Estimate param count from config for representative selection
            hidden = cfg.get("hidden_size", 0) or 0
            layers = cfg.get("num_hidden_layers", 0) or 0
            params_est = hidden * hidden * layers * 12  # rough estimate

            mt = model_type.lower()
            info = type_info[mt]
            info["downloads"] += downloads
            info["count"] += 1
            info["models"].append({
                "model_id": model_info.id,
                "model_type": mt,
                "downloads": downloads,
                "params_est": params_est,
                "trust_remote_code": "auto_map" in cfg,
            })
        except Exception:
            continue

    print(f"  Found {len(type_info)} unique model_types.", file=sys.stderr)

    # For each model_type, pick the best representative:
    # smallest model with >min_downloads (easiest to test)
    for mt, info in type_info.items():
        # Sort by param estimate (smallest first), filter by downloads
        candidates = sorted(info["models"], key=lambda m: m["params_est"])
        popular = [m for m in candidates if m["downloads"] >= min_downloads]
        representative = popular[0] if popular else candidates[0]
        info["representative"] = representative["model_id"]
        info["trust_remote_code"] = representative["trust_remote_code"]
        del info["models"]  # Don't need the full list in output

    return dict(type_info)


def check_plugin_coverage(model_type: str) -> bool:
    """Check if a model_type is supported by existing plugins.

    Uses find_plugin if available, otherwise uses startswith matching
    against known supported types.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / "python"))
    try:
        from tensorrt_model_connect.families import find_plugin
        return find_plugin(model_type) is not None
    except ImportError:
        # Fallback: check against scanned set
        supported = _scan_plugin_sources(project_root)
        # Check exact match and startswith for prefix-based plugins
        if model_type in supported:
            return True
        for s in supported:
            if model_type.startswith(s):
                return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Discover unsupported HuggingFace model families.")
    parser.add_argument("--min-downloads", type=int, default=10000,
                        help="Minimum total downloads to consider (default: 10000)")
    parser.add_argument("--max-models", type=int, default=2000,
                        help="Max models to scan from HF (default: 2000)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: stdout)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for HF API (default: 16)")
    args = parser.parse_args()

    # 1. Query HF
    all_types = query_hf_model_types(
        min_downloads=args.min_downloads,
        max_models=args.max_models,
        workers=args.workers,
    )

    # 2. Find gaps
    gaps = []
    for model_type, info in sorted(all_types.items(),
                                    key=lambda x: -x[1]["downloads"]):
        if info["downloads"] < args.min_downloads:
            continue
        if check_plugin_coverage(model_type):
            continue

        gaps.append({
            "model_type": model_type,
            "family_name": model_type.replace("-", "_").replace(".", "_"),
            "hf_id": info["representative"],
            "total_downloads": info["downloads"],
            "model_count": info["count"],
            "trust_remote_code": info.get("trust_remote_code", False),
        })

    # 3. Output
    print(f"\nFound {len(gaps)} unsupported model_types "
          f"(min {args.min_downloads:,} downloads):\n", file=sys.stderr)

    if gaps:
        # Pretty table to stderr
        print(f"{'#':>3}  {'model_type':<20} {'downloads':>12} "
              f"{'models':>7}  {'representative':<45} {'trust_rc'}", file=sys.stderr)
        print("-" * 110, file=sys.stderr)
        for i, g in enumerate(gaps, 1):
            print(f"{i:3}  {g['model_type']:<20} {g['total_downloads']:>12,} "
                  f"{g['model_count']:>7}  {g['hf_id']:<45} "
                  f"{'yes' if g['trust_remote_code'] else ''}", file=sys.stderr)

    # JSON output
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "min_downloads": args.min_downloads,
        "max_models_scanned": args.max_models,
        "total_gaps": len(gaps),
        "tasks": gaps,
    }

    json_str = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
