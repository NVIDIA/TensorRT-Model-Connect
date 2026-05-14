#!/usr/bin/env python3
"""Fetch a coverage_map.json artifact for CI use.

Tries an authenticated artifact URL first, then falls back to a local path.
Exits with code 1 if no map is available (signals full-tier fallback).

Usage:
    python tools/coverage_map/fetch_latest.py --output coverage_map.json
    python tools/coverage_map/fetch_latest.py --output coverage_map.json --artifact-url "$URL"
    python tools/coverage_map/fetch_latest.py --output coverage_map.json --local-fallback /shared/coverage_map.json
"""

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path


def _try_artifact_download(artifact_url: str, output_path: Path) -> bool:
    """Try to download coverage_map.json from an authenticated artifact URL."""
    import os
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not artifact_url:
        return False

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"

    try:
        req = urllib.request.Request(artifact_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())
            return True
    except (urllib.error.URLError, OSError) as e:
        print(f"WARNING: Artifact fetch failed: {e}", file=sys.stderr)
        return False


def resolve_coverage_map(
    output_path: Path,
    local_fallback: str = "",
    artifact_url: str | None = None,
) -> bool:
    """Try to resolve a coverage map, writing it to output_path.

    Tries in order:
    1. Artifact URL (if artifact_url is set)
    2. Local fallback path (if provided and exists)

    Returns True if a map was written to output_path, False otherwise.
    """
    if artifact_url:
        if _try_artifact_download(artifact_url, output_path):
            print("[fetch] Downloaded coverage map from artifact URL", file=sys.stderr)
            return True

    if local_fallback:
        local_path = Path(local_fallback)
        if local_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            print(f"[fetch] Using local fallback: {local_path}", file=sys.stderr)
            return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch latest coverage_map.json for CI.",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--local-fallback", default="",
                        help="Local path to fall back to if API fails")
    parser.add_argument("--artifact-url", default=None,
                        help="Authenticated coverage-map artifact URL")
    args = parser.parse_args()

    found = resolve_coverage_map(
        output_path=Path(args.output),
        local_fallback=args.local_fallback,
        artifact_url=args.artifact_url,
    )

    if not found:
        print("WARNING: No coverage map available. Run all tests.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
