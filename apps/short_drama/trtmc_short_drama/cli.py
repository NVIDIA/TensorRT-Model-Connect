# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`python -m trtmc_short_drama up`: pick a tier, build missing bundles, serve the app."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

from . import catalog
from .engine import Engine, Served

REPOSITORY = Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="trtmc_short_drama", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    up = commands.add_parser("up", help="build what is missing and start the app")
    up.add_argument("--data", type=Path, default=Path("short-drama-data"),
                    help="projects, bundles and settings (default: ./short-drama-data)")
    up.add_argument("--tier", choices=["auto", *catalog.TIERS], default="auto")
    up.add_argument("--gpu", type=int, help="GPU index (default: the one with the most memory)")
    up.add_argument("--bundle", action="append", default=[], metavar="ROLE=PATH",
                    help="use an existing bundle for a role (llm, image, video, asr, tts)")
    up.add_argument("--runtime-root", type=Path, help="directory with trtmc-server and the runtime DSOs "
                    "(default: TRTMC_RUNTIME_ROOT or <checkout>/build)")
    up.add_argument("--host", default="127.0.0.1", help="address the app listens on")
    up.add_argument("--port", type=int, default=8700)
    up.add_argument("--server-port", type=int, default=8701)
    up.add_argument("--plugins", type=Path, default=Path(__file__).resolve().parents[1] / "plugins")
    up.add_argument("--dry-run", action="store_true", help="print the plan and the build commands only")
    return result


def loopback(host: str) -> bool:
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def runtime_root(argument: Path | None) -> Path:
    for candidate in (argument, os.environ.get("TRTMC_RUNTIME_ROOT"), REPOSITORY / "build"):
        if candidate and (Path(candidate) / "trtmc-server").is_file():
            return Path(candidate).resolve()
    raise SystemExit("trtmc-server not found: build TRTMC with -DTRTMC_BUILD_SERVER=ON or pass --runtime-root")


def checkpoint_view(model: catalog.Model, data: Path, *, create: bool) -> str:
    """The checkpoint to build from: a local view whose generation config forces the language.

    The Whisper family takes its decoder prompt (language and task tokens) from the checkpoint's
    forced decoder ids and falls back to English, which turns Chinese speech into English text.
    """

    if not model.language:
        return model.checkpoint
    view = data / "checkpoints" / f"{model.checkpoint.replace('/', '--')}-{model.language}"
    if not create or (view / "generation_config.json").is_file():
        return str(view)
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(model.checkpoint))
    view.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name != "generation_config.json" and not (view / path.name).exists():
            (view / path.name).symlink_to(path.resolve())
    config = json.loads((source / "generation_config.json").read_text(encoding="utf-8"))
    forced = [pair for pair in config.get("forced_decoder_ids") or [] if pair[0] != 1]
    config["forced_decoder_ids"] = [[1, config["lang_to_id"][f"<|{model.language}|>"]], *forced]
    (view / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(view)


def build_command(model: catalog.Model, output: Path, runtime: Path, checkpoint: str) -> list[str]:
    if model.builder == "minimax_h3":  # the family's explicit FastH3 OmniRef build command
        return [sys.executable, "-m", "tensorrt_model_connect", "minimax_h3", "build-omniref",
                "-o", str(output), "--scratch", str(output.parent / "scratch"),
                "--bridge", str(runtime / "libtrtmc_byok_tvm_ffi.so"),
                "--kernel-library", str(runtime / model.byok[0])]
    return [sys.executable, "-m", "tensorrt_model_connect", "build", checkpoint,
            "-o", str(output), *model.build_args]


def resolve_bundles(tier: catalog.Tier, gpu: catalog.Gpu, data: Path, overrides: dict[str, Path],
                    runtime: Path, *, dry_run: bool) -> dict[str, Path]:
    bundles = {}
    for role, model in tier.models.items():
        path = overrides.get(role) or data / "bundles" / model.bundle_name(gpu.compute_capability)
        if not path.is_file():
            checkpoint = checkpoint_view(model, data, create=not dry_run)
            command = build_command(model, path, runtime, checkpoint)
            print(f"build {role}: {' '.join(command)}", flush=True)
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                environment = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                               "CUDA_VISIBLE_DEVICES": str(gpu.index),
                               "PYTHONPATH": os.pathsep.join(filter(None, [
                                   str(REPOSITORY / "core" / "builder"), str(REPOSITORY),
                                   os.environ.get("PYTHONPATH")]))}
                subprocess.run(command, check=True, env=environment)
        bundles[role] = path
    return bundles


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required; install FFmpeg with libass (for subtitles)")
    found = catalog.gpus()
    if not found:
        raise SystemExit("no NVIDIA GPU found (nvidia-smi failed)")
    gpu = next((g for g in found if g.index == args.gpu), None) if args.gpu is not None else \
        max(found, key=lambda g: g.memory_gib)
    if gpu is None:
        raise SystemExit(f"GPU {args.gpu} not found")
    tier = catalog.pick_tier(gpu, args.tier)
    print(f"GPU {gpu.index}: {gpu.name}, {gpu.memory_gib:.0f} GiB, sm{gpu.compute_capability} -> "
          f"tier {tier.name.upper()} ({tier.summary})", flush=True)
    overrides = {}
    for value in args.bundle:
        role, _, path = value.partition("=")
        if role not in tier.models or not path:
            raise SystemExit(f"--bundle needs ROLE=PATH with ROLE in {', '.join(tier.models)}")
        overrides[role] = Path(path).expanduser().resolve()
    data = args.data.resolve()
    root = runtime_root(args.runtime_root)
    bundles = resolve_bundles(tier, gpu, data, overrides, root, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    import uvicorn

    from .web import Services, create_app

    source = REPOSITORY / "apps" / "perf_serving"
    engine = Engine(root / "trtmc-server", root, args.server_port, data / "logs" / "trtmc-server.log",
                    cuda_device=str(gpu.index), python_path=[source] if source.is_dir() else [],
                    scratch=data / "tmp")
    models = {model.name: (role, Served(model.name, bundles[role], model.byok))
              for role, model in tier.models.items()}
    info = {"gpu": f"{gpu.name} ({gpu.memory_gib:.0f} GiB)", "tier": tier.name, "tier_summary": tier.summary}
    hosts = ["127.0.0.1", "localhost", "::1"] + ([args.host] if args.host not in {"0.0.0.0", "::"} else [])
    token = None if loopback(args.host) else secrets.token_urlsafe(24)
    app = create_app(Services(engine, models, data), plugins=args.plugins, info=info,
                     allowed_hosts=None if token else hosts, token=token)
    if token:
        print(f"listening beyond loopback; open http://{args.host}:{args.port}/?token={token}", flush=True)
    else:
        print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
