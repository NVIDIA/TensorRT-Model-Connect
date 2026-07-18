# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record isolated candidate SAM3 timing caches for qualification only.

This tool is intentionally outside the installed family. A recorded cache is
not suitable for packaging until the resulting bundle passes the full SAM3
accuracy and performance gates on an otherwise idle target GPU.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import pprint
import re
import subprocess
import sys
import textwrap
from typing import Any, Mapping

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.engine_builder import _load_plugin_weights
from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.sam3 import timing_cache


_FILE_SNAPSHOT_ALGORITHM = "sha256-canonical-json-file-records-v1"
_HASH_CHUNK_BYTES = 8 << 20
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"could not hash SAM3 provenance file: {path}") from exc
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(root).as_posix()
        size_bytes = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid SAM3 provenance file: {path}") from exc
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": size_bytes,
    }


def _snapshot_from_records(records: list[dict[str, object]]) -> dict[str, object]:
    canonical = json.dumps(
        records,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "algorithm": _FILE_SNAPSHOT_ALGORITHM,
        "file_count": len(records),
        "records": records,
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
    }


def _directory_snapshot(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise RuntimeError(f"SAM3 model snapshot directory does not exist: {root}")
    paths: list[Path] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        for path in entries:
            if path.is_symlink():
                if not path.is_file():
                    raise RuntimeError(
                        f"SAM3 model snapshot contains a broken or directory symlink: {path}"
                    )
                paths.append(path)
            elif path.is_dir():
                continue
            elif path.is_file():
                paths.append(path)
            else:
                raise RuntimeError(f"SAM3 model snapshot contains a special file: {path}")
    except OSError as exc:
        raise RuntimeError(f"could not enumerate the SAM3 model snapshot: {root}") from exc
    if not paths:
        raise RuntimeError(f"SAM3 model snapshot has no files: {root}")
    return _snapshot_from_records([_file_record(path, root=root) for path in paths])


def _source_snapshot() -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[4]
    sam3_source = repo_root / "python" / "tensorrt_model_connect" / "families" / "sam3"
    paths = {Path(__file__).resolve(), *sam3_source.rglob("*.py")}
    records = [
        _file_record(path, root=repo_root)
        for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix())
    ]
    return repo_root, _snapshot_from_records(records)


def _git_commit_best_effort(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or _GIT_COMMIT.fullmatch(commit) is None:
        return None
    return commit


def _ensure_disjoint_output(model_dir: Path, output_directory: Path) -> None:
    try:
        output_directory.relative_to(model_dir)
    except ValueError:
        return
    raise ValueError("SAM3 recorder output directory must not be inside the model snapshot")


def _timing_cache_receipt(
    recorder: "_Recorder",
    *,
    output_directory: Path,
    python_data_path: Path,
) -> dict[str, object]:
    manifest_path = recorder.directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not verify SAM3 timing-cache manifest: {manifest_path}") from exc
    engines = manifest.get("engines") if isinstance(manifest, dict) else None
    if not isinstance(engines, dict) or set(engines) != recorder.engine_kinds:
        raise RuntimeError("SAM3 timing-cache receipt engine inventory mismatch")
    expected_cache_files: set[str] = set()
    for engine_kind in sorted(recorder.engine_kinds):
        entry = engines[engine_kind]
        if not isinstance(entry, dict) or entry != recorder.entries[engine_kind]:
            raise RuntimeError(f"SAM3 timing-cache receipt entry mismatch for {engine_kind}")
        expected_cache_files.add(str(entry["file"]))
    actual_cache_files = {path.name for path in recorder.directory.glob("*.cache")}
    if actual_cache_files != expected_cache_files:
        raise RuntimeError("SAM3 timing-cache receipt contains an unexpected cache inventory")
    caches: dict[str, dict[str, object]] = {}
    for engine_kind in sorted(recorder.engine_kinds):
        entry = engines[engine_kind]
        cache_path = recorder.directory / str(entry["file"])
        record = _file_record(cache_path, root=output_directory)
        if record["sha256"] != entry["sha256"]:
            raise RuntimeError(f"SAM3 timing-cache receipt SHA-256 mismatch for {engine_kind}")
        caches[engine_kind] = {
            **record,
            "tactic_count": entry["tactic_count"],
            "tactic_sha256": entry["tactic_sha256"],
        }
    return {
        "caches": caches,
        "manifest": _file_record(manifest_path, root=output_directory),
        "python_data": _file_record(python_data_path, root=output_directory),
    }


def _set_required_flag(config: Any, name: str) -> None:
    flag = getattr(getattr(trt_compat.get_trt(), "BuilderFlag", None), name, None)
    if flag is None or not hasattr(config, "set_flag"):
        raise RuntimeError(f"TensorRT does not support required builder flag {name}")
    config.set_flag(flag)


class _Recorder:
    def __init__(
        self,
        directory: Path,
        *,
        engine_kinds: frozenset[str],
        verbose: bool = False,
    ):
        if not engine_kinds or not engine_kinds <= timing_cache._ENGINE_KINDS:
            raise ValueError(f"invalid SAM3 recording inventory: {sorted(engine_kinds)}")
        self.directory = directory
        self.engine_kinds = engine_kinds
        self.verbose = verbose
        self.entries: dict[str, dict[str, object]] = {}
        self.graph_contracts: dict[str, str] = {}

    def build(
        self,
        builder: Any,
        network: Any,
        config: Any,
        *,
        engine_kind: str,
        graph_profile: Mapping[str, Any],
    ) -> Any:
        if engine_kind not in self.engine_kinds:
            raise ValueError(f"unexpected SAM3 recording engine kind: {engine_kind!r}")
        raw_builder = trt_compat.unwrap(builder)
        raw_network = trt_compat.unwrap(network)
        raw_config = trt_compat.unwrap(config)
        if self.verbose:
            print(
                f"[sam3-cache-recorder] Building isolated {engine_kind} candidate "
                f"from {raw_network.num_layers} layers ...",
                file=sys.stderr,
            )
        for flag_name in ("EDITABLE_TIMING_CACHE", "DISABLE_COMPILATION_CACHE"):
            _set_required_flag(raw_config, flag_name)
        cache = raw_config.create_timing_cache(b"")
        if not raw_config.set_timing_cache(cache, False):
            raise RuntimeError(f"TensorRT rejected SAM3 recording cache for {engine_kind}")
        plan = raw_builder.build_serialized_network(raw_network, raw_config)
        if plan is None:
            return None
        active_cache = raw_config.get_timing_cache() if hasattr(raw_config, "get_timing_cache") else cache
        payload = bytes(active_cache.serialize())
        tactics = timing_cache._query_tactics(active_cache)
        if not payload:
            raise RuntimeError(f"TensorRT generated an empty SAM3 cache for {engine_kind}")
        self.directory.mkdir(parents=True, exist_ok=True)
        file_name = f"{engine_kind}.cache"
        (self.directory / file_name).write_bytes(payload)
        self.entries[engine_kind] = {
            "file": file_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tactic_count": len(tactics),
            "tactic_sha256": timing_cache._tactic_sha256(tactics),
        }
        self.graph_contracts[engine_kind] = timing_cache._graph_contract_fingerprint(
            engine_kind, graph_profile
        )
        if self.verbose:
            print(
                f"[sam3-cache-recorder] Recorded {engine_kind}: "
                f"cache_bytes={len(payload)}, tactic_keys={len(tactics)}, "
                f"cache_sha256={hashlib.sha256(payload).hexdigest()}",
                file=sys.stderr,
            )
        return plan

    def _graph_contract(self) -> dict[str, object]:
        if set(self.graph_contracts) != self.engine_kinds:
            raise RuntimeError(
                "SAM3 recording did not produce the complete graph inventory: "
                f"{sorted(self.graph_contracts)}"
            )
        return {
            "algorithm": timing_cache._GRAPH_CONTRACT_ALGORITHM,
            "engines": self.graph_contracts,
        }

    def write_manifest(self) -> None:
        if set(self.entries) != self.engine_kinds:
            raise RuntimeError(
                "SAM3 recording did not produce the complete engine inventory: "
                f"{sorted(self.entries)}"
            )
        manifest = {
            "engines": self.entries,
            "graph_contract": self._graph_contract(),
            "schema_version": timing_cache._SCHEMA_VERSION,
            "target": timing_cache._runtime_metadata().as_dict(),
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_python_module(self, path: Path) -> None:
        runtime = timing_cache._runtime_metadata()
        contract_id = timing_cache._BUILTIN_TARGETS.get(runtime.signature)
        if contract_id is None:
            raise RuntimeError(f"no packaged SAM3 target ID for {runtime.signature!r}")
        manifest = {
            "engines": {
                engine_kind: {key: value for key, value in entry.items() if key != "file"}
                for engine_kind, entry in self.entries.items()
            },
            "graph_contract": self._graph_contract(),
            "schema_version": timing_cache._SCHEMA_VERSION,
            "target": runtime.as_dict(),
        }
        manifest_text = pprint.pformat(manifest, sort_dicts=True, width=96)
        lines = [
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "# SPDX-License-Identifier: Apache-2.0",
            "",
            '\"\"\"Generated SAM3 timing-cache candidates; qualification is still required.\"\"\"',
            "",
            "CONTRACTS = {",
            f"    {contract_id!r}: {{",
            "        'manifest': " + textwrap.indent(manifest_text, "        ").lstrip() + ",",
            "        'payloads': {",
        ]
        for engine_kind in sorted(self.entries):
            payload = (self.directory / str(self.entries[engine_kind]["file"])).read_bytes()
            encoded = base64.b64encode(payload).decode("ascii")
            lines.append(f"            {engine_kind!r}: (")
            lines.extend(
                f"                {encoded[index:index + 88]!r}"
                for index in range(0, len(encoded), 88)
            )
            lines.append("            ),")
        lines.extend(["        },", "    },", "}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output_directory")
    parser.add_argument(
        "--engine",
        choices=("text-core", "vision-encoder"),
        default="text-core",
        help="record either the existing text/core pair or one isolated vision candidate",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_directory = Path(args.output_directory).resolve()
    _ensure_disjoint_output(model_dir, output_directory)
    model_snapshot = _directory_snapshot(model_dir)
    repo_root, source_snapshot = _source_snapshot()
    source_snapshot["git_commit"] = _git_commit_best_effort(repo_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = output_directory / "timing_cache"
    engine_kinds = (
        frozenset({"vision-encoder"})
        if args.engine == "vision-encoder"
        else frozenset({"core", "text-encoder"})
    )
    recorder = _Recorder(
        cache_directory,
        engine_kinds=engine_kinds,
        verbose=args.verbose,
    )

    # Mirror the ordinary default build's model/config path. Each selection is
    # intentionally isolated so candidate plans can be overlaid and qualified
    # without paying for unrelated tracker or full-bundle builds.
    config = ModelConfig.from_dir(model_dir)
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_decoder_engine_layout"] = "split"
    config.raw["_decoder_engine_role"] = "decode"
    config.raw["_fp32_layers"] = []
    config.raw["_family_build_options"] = {}
    plugin = find_plugin(config)
    if plugin is None or getattr(plugin, "name", None) != "sam3":
        raise RuntimeError(f"expected the SAM3 family plugin, got {plugin!r}")
    # The engine builders import the wrapper by value, so replacing only the
    # timing_cache module attribute would not intercept real plugin builds.
    # Patch every bound owner and restore all of them even when a build fails.
    if args.engine == "vision-encoder":
        from tensorrt_model_connect.families.sam3 import vision_encoder_builder

        wrapper_owners = (timing_cache, vision_encoder_builder)
    else:
        from tensorrt_model_connect.families.sam3 import (
            core_builder,
            text_encoder_builder,
        )

        wrapper_owners = (timing_cache, core_builder, text_encoder_builder)
    originals = tuple(
        (owner, owner.build_sam3_serialized_network) for owner in wrapper_owners
    )
    for owner, _original in originals:
        owner.build_sam3_serialized_network = recorder.build
    try:
        if args.engine == "vision-encoder":
            # Avoid loading the text/core/tracker weights. Resolve the same
            # model and processor contract used by the ordinary plugin path,
            # then build only the shared vision plan.
            from tensorrt_model_connect.families.sam3.plugin import (
                _load_sam3_processor_config,
                _resolve_sam3_config,
            )

            resolved = _resolve_sam3_config(config.raw)
            resolved.update(_load_sam3_processor_config(str(model_dir)))
            config.raw["_sam3_config"] = resolved
            vision_plan = plugin.build_vision_engine(
                str(model_dir),
                config,
                {},
                precision="fp32",
                verbose=args.verbose,
            )
            if vision_plan is None:
                raise RuntimeError("SAM3 plugin did not produce a vision candidate plan")
            plans = {
                "vision_engine_plan": (
                    output_directory / "vision_engine_plan.bin",
                    vision_plan,
                )
            }
        else:
            weights = _load_plugin_weights(plugin, str(model_dir), config, precision="fp32")
            text_plan = plugin.build_engine(
                config,
                weights,
                256,
                precision="fp32",
                quant_ctx=None,
                verbose=args.verbose,
                parallel_config=None,
            )
            resolved = config.raw.get("_sam3_config")
            if not isinstance(resolved, dict) or not resolved.get("video_tracking_supported"):
                raise RuntimeError("SAM3 video configuration was not resolved during weight loading")
            resolved["video_tracking_supported"] = False
            try:
                extra_plans = plugin.build_extra_engines(
                    config,
                    weights,
                    256,
                    precision="fp32",
                    verbose=args.verbose,
                )
            finally:
                resolved["video_tracking_supported"] = True
            if not isinstance(extra_plans, dict) or set(extra_plans) != {
                "sam3_core_engine_plan"
            }:
                raise RuntimeError(f"unexpected SAM3 fast-build engine inventory: {extra_plans!r}")
            core_plan = extra_plans["sam3_core_engine_plan"]
            plans = {
                "engine_plan": (output_directory / "engine_plan.bin", text_plan),
                "sam3_core_engine_plan": (
                    output_directory / "sam3_core_engine_plan.bin",
                    core_plan,
                ),
            }
    finally:
        for owner, original in originals:
            owner.build_sam3_serialized_network = original
    for path, plan in plans.values():
        path.write_bytes(plan)
    recorder.write_manifest()
    python_data_path = output_directory / "timing_cache_data.py"
    recorder.write_python_module(python_data_path)
    receipt = {
        "artifact_type": "sam3_timing_cache_candidate_receipt",
        "engine_selection": args.engine,
        "model_dir": str(model_dir),
        "model_snapshot": model_snapshot,
        "plans": {
            name: _file_record(path, root=output_directory)
            for name, (path, _plan) in plans.items()
        },
        "schema_version": 1,
        "source": source_snapshot,
        "target": timing_cache._runtime_metadata().as_dict(),
        "timing_cache": _timing_cache_receipt(
            recorder,
            output_directory=output_directory,
            python_data_path=python_data_path,
        ),
    }
    (output_directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
