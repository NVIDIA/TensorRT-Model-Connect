# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from trtmc_short_drama import catalog
from trtmc_short_drama.engine import Engine, Served
from trtmc_short_drama.flow import Flows, Step, Steps

PLUGIN = '''
from trtmc_short_drama.flow import Step

def run(context):
    context.project.log("polished")

STEP = Step("polish", run, role="llm", description="test plugin")
'''


def test_flow_versions_are_numbered_and_never_overwritten(tmp_path: Path):
    flows = Flows(tmp_path, {"llm": "q", "image": "i", "video": "v", "asr": "w"})
    reference, first = flows.load("default")
    assert reference == "default@1" and first["steps"][0] == {
        "step": "plan", "model": "q", "params": {"temperature": 0.7, "shot_seconds": 8, "max_tokens": 12000}}
    edited = {"steps": [{"step": "plan", "model": "other", "params": {}}, *first["steps"][1:]]}
    assert flows.save("default", edited, note="bigger model") == "default@2"
    assert flows.load("default")[1]["steps"][0]["model"] == "other"
    assert flows.load("default@1")[1]["steps"][0]["model"] == "q"
    assert flows.save("mine", edited) == "mine@1" and flows.names() == ["default", "mine"]


@pytest.mark.parametrize("flow", [{}, {"steps": []}, {"steps": [{"name": "plan"}]},
                                  {"steps": [{"step": "plan", "params": []}]}])
def test_invalid_flows_are_rejected(tmp_path: Path, flow):
    flows = Flows(tmp_path, {})
    with pytest.raises(ValueError):
        flows.save("default", flow)


def test_plugins_load_after_reload_and_disappear_with_their_file(tmp_path: Path):
    core = [Step("plan", lambda context: None)]
    steps = Steps(core, tmp_path)
    assert steps.get("polish") is None
    (tmp_path / "polish.py").write_text(PLUGIN, encoding="utf-8")
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    steps.reload()
    assert steps.get("polish").source == "polish.py" and "broken.py" in steps.errors
    (tmp_path / "polish.py").unlink()
    steps.reload()
    assert steps.get("polish") is None and steps.get("plan") is not None


def test_a_plugin_cannot_replace_a_core_step(tmp_path: Path):
    (tmp_path / "plan.py").write_text(PLUGIN.replace('"polish"', '"plan"'), encoding="utf-8")
    steps = Steps([Step("plan", lambda context: None)], tmp_path)
    assert steps.get("plan").source == "core" and "plan.py" in steps.errors


def test_server_command_registers_bundles_and_one_kernel(tmp_path: Path):
    engine = Engine(tmp_path / "trtmc-server", tmp_path, 8701, tmp_path / "log")
    command = engine.command(frozenset({Served("w", tmp_path / "w.bundle"),
                                        Served("v", tmp_path / "v.bundle", ("libvsa.so", "run", "vsa"))}))
    assert command[command.index("--port") + 1] == "8701"
    assert [command[i + 1] for i, part in enumerate(command) if part == "--model"] == [
        f"v={tmp_path / 'v.bundle'}", f"w={tmp_path / 'w.bundle'}"]
    assert command[command.index("--byok-library") + 1] == str(tmp_path / "libvsa.so")
    with pytest.raises(ValueError):
        engine.command(frozenset({Served("a", tmp_path / "a", ("x", "f", "k")),
                                  Served("b", tmp_path / "b", ("y", "f", "k"))}))


def test_tiers_follow_gpu_memory():
    big = catalog.parse_gpu("1, NVIDIA GB300, 262144, 10.3")
    assert big.compute_capability == "10.3" and catalog.pick_tier(big).name == "a"
    assert catalog.pick_tier(catalog.parse_gpu("0, RTX, 49152, 8.9")).name == "b"
    assert catalog.pick_tier(big, "b").name == "b"
    with pytest.raises(ValueError):
        catalog.pick_tier(catalog.parse_gpu("0, small, 16384, 8.6"))


def test_video_submission_waits_while_server_storage_is_full(tmp_path: Path, monkeypatch):
    import httpx

    from trtmc_short_drama import engine as engine_module

    request = httpx.Request("POST", "http://127.0.0.1:8701/v1/videos")
    answers = [httpx.Response(429, json={"error": {"message": "full"}}, request=request),
               httpx.Response(202, json={"id": "v1", "status": "completed"}, request=request)]
    posts = []
    monkeypatch.setattr(engine_module.httpx, "post", lambda *a, **k: posts.append(a) or answers[len(posts) - 1])
    monkeypatch.setattr(engine_module, "STORAGE_RETRY_SECONDS", 0)

    class Stream:
        def __init__(self, *args, **kwargs):
            self.response = httpx.Response(200, content=b"mp4", request=request)

        def __enter__(self):
            return self.response

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(engine_module.httpx, "stream", Stream)
    reference = tmp_path / "r.png"
    reference.write_bytes(b"png")
    engine = Engine(tmp_path / "trtmc-server", tmp_path, 8701, tmp_path / "log")
    engine.video("v", "p", [reference], tmp_path / "out.mp4", seed=1)
    assert len(posts) == 2 and (tmp_path / "out.mp4").read_bytes() == b"mp4"


def test_whisper_builds_from_a_view_that_forces_the_language(tmp_path: Path, monkeypatch):
    import json

    import huggingface_hub

    from trtmc_short_drama import cli

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "generation_config.json").write_text(json.dumps(
        {"lang_to_id": {"<|en|>": 50259, "<|zh|>": 50260}, "forced_decoder_ids": [[1, None], [2, 50360]]}))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: str(snapshot))
    model = catalog.TIERS["a"].models["asr"]
    assert model.bundle_name("10.3") == "whisper-zh-sm103.bundle"
    view = Path(cli.checkpoint_view(model, tmp_path / "data", create=True))
    assert (view / "model.safetensors").resolve() == (snapshot / "model.safetensors").resolve()
    forced = json.loads((view / "generation_config.json").read_text())["forced_decoder_ids"]
    assert forced == [[1, 50260], [2, 50360]]
    command = cli.build_command(model, tmp_path / "w.bundle", tmp_path, str(view))
    assert command[3:5] == ["build", str(view)]


def test_h3_is_built_through_the_family_command(tmp_path: Path):
    from trtmc_short_drama import cli

    command = cli.build_command(catalog.TIERS["a"].models["video"], tmp_path / "h3.bundle", tmp_path, "unused")
    assert command[1:5] == ["-m", "tensorrt_model_connect", "minimax_h3", "build-omniref"]
    assert command[command.index("--kernel-library") + 1] == str(tmp_path / "libtrtmc_minimax_h3_omniref_vsa.so")
    assert command[command.index("--bridge") + 1] == str(tmp_path / "libtrtmc_byok_tvm_ffi.so")
